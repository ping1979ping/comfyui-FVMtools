"""SMP LocationGenerator — emits a LOCATION_DICT_RAW.

Wraps ``core.location_engine.generate_location_records``. The output dict
contains per-element fragments with ``#ambient_light#`` / ``#shadow_tone#``
tokens still embedded — pair with the LocationCombiner to resolve them.
"""

try:
    from ...core.location_engine import (
        generate_location_records,
        get_available_location_sets,
    )
except ImportError:  # pragma: no cover
    from core.location_engine import (
        generate_location_records,
        get_available_location_sets,
    )


_FALLBACK_SETS = ["outdoor/urban/brutalist", "outdoor/beach/mediterranean", "indoor/studio/minimal"]


def _location_set_choices() -> list[str]:
    sets = get_available_location_sets()
    if sets:
        return sets
    # ComfyUI insists on a non-empty COMBO; fall back to the planned set names.
    return list(_FALLBACK_SETS)


class FVM_SMP_LocationGenerator:
    """Dict-emitting location generator for the StructPromptMaker pipeline."""

    CATEGORY = "FVM Tools/SMP/Generators"
    FUNCTION = "generate"
    RETURN_TYPES = ("LOCATION_DICT_RAW", "STRING")
    RETURN_NAMES = ("location_raw", "summary")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Dict-emitting location generator (StructPromptMaker).\n\n"
        "Emits a LOCATION_DICT_RAW with per-element records (background,\n"
        "midground, foreground, architecture_detail, props, time_of_day,\n"
        "weather), region hints, and #ambient_light# / #shadow_tone# tokens\n"
        "still embedded in prompt_fragment. Pair with the LocationCombiner."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "location_set": (_location_set_choices(),),
                "seed":         ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "enable_background":          ("BOOLEAN", {"default": True}),
                "enable_midground":           ("BOOLEAN", {"default": False}),
                "enable_architecture_detail": ("BOOLEAN", {"default": False}),
                "enable_props":               ("BOOLEAN", {"default": False}),
                "enable_foreground_element":  ("BOOLEAN", {"default": True}),
                "enable_time_of_day":         ("BOOLEAN", {"default": True}),
                "enable_weather":             ("BOOLEAN", {"default": True}),
                "color_tone": (["", "warm", "cool", "neutral"], {"default": ""}),
            },
        }

    def generate(self, location_set, seed,
                 enable_background, enable_midground, enable_architecture_detail,
                 enable_props, enable_foreground_element,
                 enable_time_of_day, enable_weather, color_tone):
        element_enables = {
            "background":          enable_background,
            "midground":           enable_midground,
            "architecture_detail": enable_architecture_detail,
            "props":               enable_props,
            "foreground_element":  enable_foreground_element,
            "time_of_day":         enable_time_of_day,
            "weather":             enable_weather,
        }
        rec = generate_location_records(
            seed=seed, location_set=location_set,
            element_enables=element_enables,
            color_tone=color_tone or None,
        )

        loc_raw = {
            "set_name":   rec["location_set"],
            "seed":       rec["seed"],
            "color_tone": rec["color_tone"],
            "elements":   rec["elements"],
        }

        lines = [f"Set: {rec['location_set']} | Seed: {seed} | Elements: {len(rec['elements'])}"]
        for elem_id, e in rec["elements"].items():
            lines.append(f"  {elem_id} [{e['layer']}, cov={e['coverage']:.2f}]: {e['prompt_fragment']}")
        return (loc_raw, "\n".join(lines))
