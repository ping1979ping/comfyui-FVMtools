"""SMP Aggregator — deep-merge up to 4 PROMPT_DICT branches into one."""

try:
    from ...core.smp.merge import merge_many
except ImportError:  # pragma: no cover
    from core.smp.merge import merge_many


class FVM_SMP_Aggregator:
    CATEGORY = "FVM Tools/SMP/Plumbing"
    FUNCTION = "aggregate"
    RETURN_TYPES = ("PROMPT_DICT",)
    RETURN_NAMES = ("prompt_dict",)
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Deep-merges up to 4 PROMPT_DICT branches per spec §6.2:\n"
        "later wins, lists concatenate, None deletes a field."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_dict_a": ("PROMPT_DICT",),
            },
            "optional": {
                "prompt_dict_b": ("PROMPT_DICT",),
                "prompt_dict_c": ("PROMPT_DICT",),
                "prompt_dict_d": ("PROMPT_DICT",),
            },
        }

    def aggregate(self, prompt_dict_a, prompt_dict_b=None, prompt_dict_c=None,
                  prompt_dict_d=None):
        merged = merge_many(prompt_dict_a, prompt_dict_b, prompt_dict_c, prompt_dict_d)
        return (merged,)
