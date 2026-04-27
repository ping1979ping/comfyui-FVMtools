"""SMP SAMClassRouter — look up the prompt fragment for a given SAM3 mask class.

One-line bridge: hand a SAM3 mask class name (e.g. "upper_clothes") and a
STRUCTURED_PROMPTS bundle, and get back the regional prompt fragment for
that class. Lets you wire SMP regional prompts into the existing
PersonDetailer per-slot inputs without touching the Detailer code.
"""

try:
    from ...core.smp.assembler import build_sam_class_lookup
except ImportError:  # pragma: no cover
    from core.smp.assembler import build_sam_class_lookup


class FVM_SMP_SAMClassRouter:
    CATEGORY = "FVM Tools/SMP/Output"
    FUNCTION = "route"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_fragment",)
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Returns the regional prompt fragment matching a SAM3 mask class.\n\n"
        "Useful pattern: SAM3 emits class 'upper_clothes' for a detected\n"
        "person's torso → this node returns the upper-body fragment from\n"
        "the StructuredAssembler — which goes straight into a\n"
        "PersonDetailer per-slot prompt input."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "structured":  ("STRUCTURED_PROMPTS",),
                "class_name":  ("STRING", {"default": "upper_clothes"}),
            },
            "optional": {
                "fallback":    ("STRING", {"default": ""}),
            },
        }

    def route(self, structured, class_name, fallback=""):
        s = structured or {}
        lookup = s.get("sam_class_lookup") or {}
        if not lookup:
            # Re-derive on the fly if the assembler output is partial.
            lookup = build_sam_class_lookup(
                s.get("face", ""),
                {"garments": (s.get("raw_dict") or {}).get("outfits", {}).get("subject_1", {}).get("garments", {})},
                (s.get("raw_dict") or {}).get("location") or {},
            )
        result = lookup.get((class_name or "").strip()) or fallback
        return (result,)
