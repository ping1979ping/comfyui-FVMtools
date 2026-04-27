"""SMP StructuredPromptAssembler — bridge to the existing PersonDetailer.

Splits a PROMPT_DICT (or direct OUTFIT_DICT + LOCATION_DICT + subject_json)
into four regional prompts (face / body / outfit / location), a region map
for sidecar logging, and a SAM3-class → fragment lookup that downstream
Detailer adapters can use to route per-mask prompts.
"""

try:
    from ...core.smp.assembler import assemble_structured
except ImportError:  # pragma: no cover
    from core.smp.assembler import assemble_structured


_SUBJECT_JSON_DEFAULT = (
    '{\n  "id": "subject_1",\n  "age_desc": "young",\n  "gender": "woman",\n'
    '  "skin_tags": ["smooth skin"],\n  "expression": "neutral expression",\n'
    '  "hair_color_length": "dark auburn hair"\n}'
)


class FVM_SMP_StructuredPromptAssembler:
    """Split a PROMPT_DICT into per-region prompts for the existing detailer.

    Accepts either a full PROMPT_DICT (typical pipeline output) or a
    combination of OUTFIT_DICT + LOCATION_DICT + subject_json. Direct dict
    inputs win when both are wired.
    """

    CATEGORY = "FVM Tools/SMP/Output"
    FUNCTION = "assemble"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",
                    "REGION_MAP", "STRUCTURED_PROMPTS")
    RETURN_NAMES = ("face_prompt", "body_prompt", "outfit_prompt", "location_prompt",
                    "region_map", "structured")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Splits a PROMPT_DICT (or direct OUTFIT_DICT + LOCATION_DICT +\n"
        "subject_json) into four tier-ordered regional prompt strings:\n"
        "  • face_prompt — character anchor, skin, eyes (boosted), expression\n"
        "  • body_prompt — anchor, build, pose, hands, full hair, anatomy tags\n"
        "  • outfit_prompt — head→toe garment fragments\n"
        "  • location_prompt — bg → fg → atmosphere\n"
        "Plus a REGION_MAP and a sam_class_lookup the Detailer can use\n"
        "to route per-class prompts."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "include_quality": ("BOOLEAN", {"default": True}),
                "face_eye_boost":  ("FLOAT", {"default": 1.1, "min": 1.0, "max": 1.4, "step": 0.05}),
                "subject_index":   ("INT", {"default": 0, "min": 0, "max": 9}),
            },
            "optional": {
                "prompt_dict":   ("PROMPT_DICT",),
                "outfit_dict":   ("OUTFIT_DICT",),
                "location_dict": ("LOCATION_DICT",),
                "subject_json":  ("STRING", {"multiline": True, "default": _SUBJECT_JSON_DEFAULT}),
            },
        }

    def assemble(self, include_quality, face_eye_boost, subject_index,
                 prompt_dict=None, outfit_dict=None, location_dict=None,
                 subject_json=None):
        # Empty multiline strings come through as "" — treat as None.
        subj = subject_json if subject_json and subject_json.strip() not in ("{}", "") else None
        structured = assemble_structured(
            prompt_dict=prompt_dict,
            outfit=outfit_dict,
            location=location_dict,
            subject=subj,
            subject_index=subject_index,
            eye_boost=face_eye_boost,
            include_quality=include_quality,
        )
        return (
            structured["face"],
            structured["body"],
            structured["outfit"],
            structured["location"],
            structured["region_map"],
            structured,
        )
