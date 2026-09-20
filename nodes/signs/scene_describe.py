"""Read the picture once, and let everyone downstream use the same answer.

Two steps need to know what the photograph shows, for different reasons. The
text proposer needs it or it guesses badly — asked to name a small sign with no
context it produced `LEFT TURN ONLY` in the middle of a shopping street, and,
worse, it fell back on reading the nonsense already printed there (`ANE PITEI`
is `SANE PITEI` with a letter shaved off). The caption needs it so the repainted
lettering sits in the same light and the same weather as everything around it.

Asking twice would cost two calls and, more importantly, could return two
different scenes for one picture — after which the words and the surface belong
to different photographs. So it is asked once, here, and wired to both.
"""

import numpy as np

SYSTEM = (
    "You describe a photograph for two readers: a writer who has to invent "
    "plausible signage for the shops in it, and a painter who has to repaint a "
    "few small surfaces without the repair looking pasted on. "
    "Answer with ONE JSON object, keys exactly "
    '{"summary", "background", "setting"}. '
    '"summary" is one sentence naming what the picture shows. '
    '"background" is two or three sentences on the light, the weather, the '
    "materials and the depth of field. "
    '"setting" is a short phrase a sign writer would need — the kind of place '
    "and the kind of businesses in it, for example 'a row of small old-town "
    "shops' or 'a supermarket aisle' or 'a railway platform'. "
    "Name no text and no lettering: the writing in this picture is nonsense and "
    "is about to be replaced."
)


class FVM_SceneDescribe:
    """One vision-model read of the whole picture, shared by the whole graph."""

    CATEGORY = "FVMtools/signs"
    FUNCTION = "read"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("summary", "background", "setting", "report")
    DESCRIPTION = (
        "Describe the photograph once and feed both the text proposer "
        "(`scene_hint`) and the caption builder. Without it the proposer invents "
        "signage for the wrong kind of place, or falls back on transcribing the "
        "nonsense already in the picture."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The picture to read."}),
                "base_url": (
                    "STRING",
                    {
                        "default": "http://localhost:1234/v1",
                        "tooltip": "LM Studio, OpenAI-compatible.",
                    },
                ),
                "model_id": (
                    "STRING",
                    {
                        "default": "qwen3-8b-vl-instruct-abliterated",
                        "tooltip": "Name it. 'Whatever is loaded' is not a default, "
                        "it is a silent failure: LM Studio lists models it "
                        "has not loaded, the call returns nothing, and the "
                        "graph carries on with an empty description. Must "
                        "be a VISION model.",
                    },
                ),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Off: pass the fallbacks through unchanged, so the "
                        "effect of the description can be measured.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "0.0. Measured over a hundred runs elsewhere in "
                        "this project: at 0.2 and above the model invents "
                        "one-off answers that never repeat.",
                    },
                ),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 900}),
            },
            "optional": {
                "fallback_summary": ("STRING", {"default": "", "multiline": True}),
                "fallback_background": ("STRING", {"default": "", "multiline": True}),
                "fallback_setting": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def read(
        self,
        image,
        base_url,
        model_id,
        enabled,
        temperature,
        timeout,
        fallback_summary="",
        fallback_background="",
        fallback_setting="",
    ):
        if not enabled:
            return (
                fallback_summary,
                fallback_background,
                fallback_setting,
                "[SceneDescribe] aus",
            )

        try:
            from ..utils.lmstudio_client import chat_vision, parse_json_response
        except Exception:
            import importlib.util
            import os

            pfad = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "utils",
                "lmstudio_client.py",
            )
            spec = importlib.util.spec_from_file_location("_lmc", pfad)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            chat_vision, parse_json_response = mod.chat_vision, mod.parse_json_response

        rgb = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        try:
            res = chat_vision(
                base_url=base_url,
                model_id=model_id,
                system_prompt=SYSTEM,
                user_prompt="Describe this photograph. Report the JSON.",
                images=[rgb],
                temperature=temperature,
                max_tokens=400,
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001
            grund = f"{type(e).__name__}: {e}"
            res = {"ok": False, "error": grund}

        if not res.get("ok"):
            # Said out loud. An empty description does not stop the run — it just
            # quietly removes the one thing that keeps the new lettering inside
            # the same photograph, and the proposer starts guessing.
            warnung = (
                f"[SceneDescribe] NICHT gelesen ({str(res.get('error'))[:120]}) "
                f"— es gelten die Rueckfallwerte"
            )
            print(warnung)
            return (fallback_summary, fallback_background, fallback_setting, warnung)

        d = parse_json_response(res.get("content", "")) or {}
        s = (d.get("summary") or fallback_summary or "").strip()
        b = (d.get("background") or fallback_background or "").strip()
        o = (d.get("setting") or fallback_setting or "").strip()
        kopf = f"[SceneDescribe] {s[:90]}"
        print(kopf)
        return (s, b, o, "\n".join([kopf, f"  Umgebung: {o}", f"  Hintergrund: {b}"]))


NODE_CLASS_MAPPINGS = {"FVM_SceneDescribe": FVM_SceneDescribe}
NODE_DISPLAY_NAME_MAPPINGS = {"FVM_SceneDescribe": "Scene Describe (LM Studio)"}
