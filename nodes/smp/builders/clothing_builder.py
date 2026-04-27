"""SMP ClothingBuilder — attaches an OUTFIT_DICT to a PROMPT_DICT subject."""

try:
    from ....core.smp.merge import deep_merge
except ImportError:  # pragma: no cover
    from core.smp.merge import deep_merge


class FVM_SMP_ClothingBuilder:
    CATEGORY = "FVM Tools/SMP/Builders"
    FUNCTION = "build"
    RETURN_TYPES = ("PROMPT_DICT",)
    RETURN_NAMES = ("prompt_dict",)
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Attaches an OUTFIT_DICT to PROMPT_DICT.outfits[subject_id].\n"
        "The StructuredPromptAssembler reads outfits[subject.id] when\n"
        "splitting outfit into a regional prompt."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outfit_dict": ("OUTFIT_DICT",),
                "subject_id":  ("STRING", {"default": "subject_1"}),
            },
            "optional": {
                "prompt_dict_in": ("PROMPT_DICT",),
            },
        }

    def build(self, outfit_dict, subject_id, prompt_dict_in=None):
        partial = {"outfits": {subject_id or "subject_1": outfit_dict or {}}}
        merged = deep_merge(prompt_dict_in or {}, partial)
        return (merged,)
