"""SMP EnvironmentBuilder — attaches a LOCATION_DICT to a PROMPT_DICT."""

try:
    from ....core.smp.merge import deep_merge
except ImportError:  # pragma: no cover
    from core.smp.merge import deep_merge


class FVM_SMP_EnvironmentBuilder:
    CATEGORY = "FVM Tools/SMP/Builders"
    FUNCTION = "build"
    RETURN_TYPES = ("PROMPT_DICT",)
    RETURN_NAMES = ("prompt_dict",)
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Attaches a LOCATION_DICT to PROMPT_DICT.location.\n"
        "Single-scene assumption — last EnvironmentBuilder wins."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "location_dict": ("LOCATION_DICT",),
            },
            "optional": {
                "prompt_dict_in": ("PROMPT_DICT",),
            },
        }

    def build(self, location_dict, prompt_dict_in=None):
        partial = {"location": location_dict or {}}
        merged = deep_merge(prompt_dict_in or {}, partial)
        return (merged,)
