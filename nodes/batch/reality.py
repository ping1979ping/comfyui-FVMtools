"""FVM_RealityCheck — ask a local vision model whether a rendered picture holds up.

Wraps :mod:`nodes.utils.reality_client`, which carries the measured design: the
model is asked perception questions with closed answers ("do you see breasts or
shoulder blades?", "how many heads?") and the impossibility is inferred in
Python from the contradictions between them. Asking a small model to judge
realism directly does not work — it clears every picture, including a torso
rotated through 180 degrees.

Defaults come from an acceptance run over a 17-image set with eight known-broken
pictures: all eight caught, all eight good ones kept, twice, with different
seeds. ``tests/live/reality/calibrate.py`` reproduces it.
"""

import numpy as np

from ..utils.reality_client import (
    DEFAULT_MAX_IMAGE_SIZE, DEFAULT_PASSES, DEFAULT_PROBES, DEFAULT_TEMPERATURE,
    DEFAULT_THRESHOLD, PROBES, SYSTEM_PROMPT, assess, verdict_to_json,
)
from ..utils.lmstudio_client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, probe

#: Probes offered as toggles, in run order. The confirmation stage is not a
#: toggle: it only ever runs when a suspicion it can overturn was raised, and
#: switching it off would just mean more false alarms.
TOGGLE_PROBES = ("parts", "people", "landmarks", "hands", "physics", "text")


def tensor_to_rgb(image):
    """First frame of a ComfyUI IMAGE batch as an (H, W, 3) uint8 array."""
    if image is None:
        return None
    array = image[0] if getattr(image, "ndim", 0) == 4 else image
    array = array.detach().cpu().numpy() if hasattr(array, "detach") else np.asarray(array)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


class FVM_RealityCheck:
    """Judge whether a generated picture is physically possible."""

    CATEGORY = "FVM Tools/Batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_url": ("STRING", {
                    "default": DEFAULT_BASE_URL,
                    "tooltip": "LM Studio's OpenAI-compatible endpoint.",
                }),
                "model": ("STRING", {
                    "default": "",
                    "tooltip": "Vision model id. Empty uses whatever LM Studio "
                               "has loaded. Needs a VLM — a text-only model "
                               "answers nonsense.",
                }),
                "expected_people": ("INT", {
                    "default": 1, "min": 1, "max": 10,
                    "tooltip": "How many people the picture should show. More "
                               "heads or limbs than this allows is a defect.",
                }),
                "threshold": ("FLOAT", {
                    "default": DEFAULT_THRESHOLD, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fail when the worst weighted finding reaches "
                               "this. Findings score ~1.0, so 0.5 is a wide "
                               "margin.",
                }),
                "passes": ("INT", {
                    "default": DEFAULT_PASSES, "min": 1, "max": 9,
                    "tooltip": "Runs per probe. At temperature 0.1 repeats give "
                               "identical answers — raise both together, or "
                               "leave at 1.",
                }),
                "temperature": ("FLOAT", {
                    "default": DEFAULT_TEMPERATURE, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "aggregation": (["median", "majority", "mean", "max"], {
                    "default": "median",
                    "tooltip": "How repeated passes are reduced. Only matters "
                               "when passes > 1.",
                }),
                "check_parts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Count heads, arms, hands, legs, feet — catches "
                               "duplicated bodies and extra limbs.",
                }),
                "check_people": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Count people and spot merged bodies.",
                }),
                "check_landmarks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Front-of-body and back-of-body landmarks seen at "
                               "once — catches impossible torso twists.",
                }),
                "check_hands": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Finger counts. Off by default: small models "
                               "cannot resolve fingers and raise false alarms.",
                }),
                "check_physics": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Floating bodies and limbs through furniture. Off "
                               "by default: reads a compressed mattress as a "
                               "body sinking through it.",
                }),
                "check_text": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Garbled lettering. Worth turning on for scenes "
                               "with signage.",
                }),
                "max_image_size": ("INT", {
                    "default": DEFAULT_MAX_IMAGE_SIZE, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Long edge sent to the model. Below ~900 the "
                               "giveaways blur away.",
                }),
                "timeout": ("INT", {"default": int(DEFAULT_TIMEOUT), "min": 5, "max": 900}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "on_error": (["pass", "fail"], {
                    "default": "pass",
                    "tooltip": "What to answer when the model cannot be reached. "
                               "pass: keep the picture and say so — an inspector "
                               "that cannot see should not reject a batch.",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Overrides the built-in instrument prompt. Empty "
                               "uses the calibrated one.",
                }),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "FLOAT", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("passed", "failed", "score", "report", "json", "image")
    FUNCTION = "execute"
    DESCRIPTION = ("Asks a local vision model perception questions about one "
                   "picture and infers whether it is physically possible.")

    def execute(self, image, base_url, model, expected_people, threshold, passes,
                temperature, aggregation, check_parts, check_people,
                check_landmarks, check_hands, check_physics, check_text,
                max_image_size, timeout, seed, on_error, system_prompt=""):
        enabled = [name for name, active in (
            ("parts", check_parts), ("people", check_people),
            ("landmarks", check_landmarks), ("hands", check_hands),
            ("physics", check_physics), ("text", check_text),
        ) if active]
        # The confirmation stage rides along whenever the probe that can raise a
        # suspicion is on, so a twist finding always gets its second opinion.
        if "landmarks" in enabled:
            enabled.append("pelvis_confirm")

        if not enabled:
            report = "No checks enabled — nothing to judge."
            print(f"[FVM Reality Check] {report}")
            return {"ui": {"text": [report]},
                    "result": (True, False, 0.0, report, "{}", image)}

        rgb = tensor_to_rgb(image)
        if rgb is None:
            report = "No image input."
            return {"ui": {"text": [report]},
                    "result": (True, False, 0.0, report, "{}", image)}

        result = assess(
            rgb, base_url=base_url, model_id=(model or "").strip(),
            probes=tuple(enabled), passes=int(passes), threshold=float(threshold),
            expected_people=int(expected_people), temperature=float(temperature),
            seed=int(seed) if seed else None, timeout=float(timeout),
            max_image_size=int(max_image_size), aggregation=aggregation,
            system_prompt=(system_prompt or "").strip() or None,
        )

        passed = result["passed"]
        if not result["ok"] and on_error == "fail":
            passed = False

        status = f"{'PASS' if passed else 'FAIL'}  {result['score']:.2f}"
        if result["issues"]:
            status += "  " + result["issues"][0][:60]
        elif not result["ok"]:
            status += "  (model unreachable)"

        print(f"[FVM Reality Check] {result['report'].splitlines()[0]}")
        for issue in result["issues"]:
            print(f"[FVM Reality Check]   • {issue}")

        return {"ui": {"text": [status]},
                "result": (passed, not passed, float(result["score"]),
                           result["report"], verdict_to_json(result), image)}


class FVM_RealityCheckProbe:
    """Check that LM Studio is up and report which models it offers."""

    CATEGORY = "FVM Tools/Batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
            "timeout": ("INT", {"default": 5, "min": 1, "max": 60}),
        }}

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("reachable", "models")
    FUNCTION = "execute"
    OUTPUT_NODE = True
    DESCRIPTION = "Reachability check for the reality check's LM Studio endpoint."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")     # a server's availability is not cacheable

    def execute(self, base_url, timeout):
        result = probe(base_url, timeout=float(timeout))
        if result["reachable"]:
            text = "\n".join(result["models"]) or "(no models loaded)"
            status = f"reachable — {len(result['models'])} models"
        else:
            text = result["error"] or "unreachable"
            status = "unreachable"
        print(f"[FVM Reality Check] {base_url}: {status}")
        return {"ui": {"text": [status]}, "result": (result["reachable"], text)}
