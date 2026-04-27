"""SMP PromptSerialize — render a PROMPT_DICT to STRINGs.

Two formats in P5 (qwen_chatml + comma_tags deferred to P7):
- ``raw_json`` — pretty-printed dict, useful for power-users / Z-Image / debugging.
- ``natural_language`` — readable English flattening for Flux/SDXL/CLIP encoders.

Always returns three STRING outputs: positive prompt, negative prompt,
and raw JSON. The user picks which one to wire into the encoder.
"""

import json
from typing import Optional

try:
    from ...core.smp.assembler import (
        build_face_prompt,
        build_outfit_prompt,
        build_location_prompt,
    )
except ImportError:  # pragma: no cover
    from core.smp.assembler import (
        build_face_prompt,
        build_outfit_prompt,
        build_location_prompt,
    )


def _natural_language(pd: dict) -> str:
    pd = pd or {}
    parts: list[str] = []

    subjects = pd.get("subjects") or []
    outfits = pd.get("outfits") or {}
    location = pd.get("location") or {}

    if subjects:
        s = subjects[0]
        anchor = " ".join([s.get("age_desc") or "", s.get("gender") or ""]).strip()
        if anchor:
            parts.append(f"A photograph of a {anchor}")
        face = build_face_prompt(s, include_quality=False)
        if face and face != anchor:
            parts.append(face)

        outfit = outfits.get(s.get("id", "subject_1")) or {}
        outfit_str = build_outfit_prompt(outfit)
        if outfit_str:
            parts.append(f"wearing {outfit_str}")

        if pose := s.get("pose_hint"):
            parts.append(pose)

    location_str = build_location_prompt(location)
    if location_str:
        parts.append(f"set against {location_str}")

    quality = (pd.get("post_processing") or {}).get("quality_tags") or []
    if quality:
        parts.append(", ".join(quality))

    return ". ".join(p for p in parts if p) + ("." if parts else "")


def _negative_prompt(pd: dict) -> str:
    pp = (pd or {}).get("post_processing") or {}
    return pp.get("negative_prompt") or "blurry, low quality, distorted anatomy, extra limbs, watermark"


def _raw_json(pd: dict) -> str:
    try:
        return json.dumps(pd or {}, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps({"_serialize_error": "non-JSON value in PROMPT_DICT"})


class FVM_SMP_PromptSerialize:
    CATEGORY = "FVM Tools/SMP/Output"
    FUNCTION = "serialize"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "raw_json")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Serializes a PROMPT_DICT to three STRINGs:\n"
        "  • positive_prompt — natural-language or raw_json (configurable)\n"
        "  • negative_prompt — pulled from post_processing.negative_prompt\n"
        "  • raw_json — the full dict, pretty-printed (for sidecar logging)\n"
        "Wire the one you need into the encoder; the other two help debug."
    )

    _FORMATS = ["natural_language", "raw_json"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_dict": ("PROMPT_DICT",),
                "format":      (cls._FORMATS, {"default": "natural_language"}),
            },
        }

    def serialize(self, prompt_dict, format):
        pd = prompt_dict or {}
        if format == "raw_json":
            positive = _raw_json(pd)
        else:
            positive = _natural_language(pd)
        return (positive, _negative_prompt(pd), _raw_json(pd))
