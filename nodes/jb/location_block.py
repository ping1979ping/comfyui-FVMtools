"""FVM_JB_LocationBlock — combo node merging Location Generator + Combiner.

One node, all the choices: pick a location set, twist the sliders, get
back a fully-resolved JSON location ready to feed into a JB Stitcher.

Edit List button uses a dedicated frontend modal pointed at
``location_lists/`` (same UX as the outfit Edit List).
"""

from __future__ import annotations

try:
    from ...core.location_engine import (
        generate_location_records,
        get_available_location_sets,
    )
    from ...core.style_presets import STYLE_PRESETS
    from ...core.jb.palette import build_palette, resolve_tokens
    from ...core.jb.serialize import ALL_FORMATS, emit, emit_strict_json
except ImportError:  # pragma: no cover
    from core.location_engine import (
        generate_location_records,
        get_available_location_sets,
    )
    from core.style_presets import STYLE_PRESETS
    from core.jb.palette import build_palette, resolve_tokens
    from core.jb.serialize import ALL_FORMATS, emit, emit_strict_json


_HARMONY_TYPES = ["auto", "analogous", "complementary", "split_complementary",
                  "triadic", "tetradic", "monochromatic"]


def _location_set_choices() -> list[str]:
    sets = get_available_location_sets()
    return sets or ["outdoor/urban/brutalist", "outdoor/beach/mediterranean", "indoor/studio/minimal"]


class FVM_JB_LocationBlock:
    """Location Generator + Location Combiner in a single node."""

    CATEGORY = "FVM Tools/JB"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("location_json", "location_string", "palette_summary")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Location Generator + Combiner merged into one node.\n\n"
        "Pick a location set, twist the sliders, get back a fully-resolved\n"
        "JSON location ready to feed into a JB Stitcher.\n\n"
        "Reuses the same location_lists/ data; Edit List modal lets you\n"
        "edit the source files directly from the node."
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
                # Color section — drives atmosphere tokens (#ambient_light#, #shadow_tone#).
                "num_colors":      ("INT", {"default": 5, "min": 2, "max": 8}),
                "harmony_type":    (_HARMONY_TYPES, {"default": "auto"}),
                "palette_style":   (sorted(STYLE_PRESETS.keys()), {"default": "general"}),
                "vibrancy":        ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "contrast":        ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "warmth":          ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "output_format":   (list(ALL_FORMATS), {"default": "loose_keys"}),
            },
            "optional": {
                "color_tone": (["", "warm", "cool", "neutral"], {"default": ""}),
            },
        }

    def build(self, location_set, seed,
              enable_background, enable_midground, enable_architecture_detail,
              enable_props, enable_foreground_element,
              enable_time_of_day, enable_weather,
              num_colors, harmony_type, palette_style, vibrancy, contrast, warmth,
              output_format, color_tone=""):
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

        palette = build_palette(
            seed=seed, num_colors=num_colors, harmony_type=harmony_type,
            style_preset=palette_style, vibrancy=vibrancy, contrast=contrast,
            warmth=warmth,
        )
        subs = palette["subs"]

        elements: dict = {}
        for elem_id, e in rec["elements"].items():
            elements[elem_id] = {
                "name":            e["name"],
                "coverage":        e["coverage"],
                "texture":         e.get("texture"),
                "layer":           e["layer"],
                "prompt_fragment": resolve_tokens(e["prompt_fragment"], subs),
            }

        location = {
            "location": {
                "set_name":   rec["location_set"],
                "seed":       rec["seed"],
                "color_tone": color_tone or palette["color_tone"],
                "elements":   elements,
            }
        }

        location_json = emit_strict_json(location, indent=2)
        location_string = emit(location, output_format)
        return (location_json, location_string, palette["palette_string"])
