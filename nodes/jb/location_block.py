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
    from ...core.jb.color_moods import MOOD_NAMES, mood_help
    from ...core.jb.serialize import (ALL_FORMATS, NATURAL, emit,
                                      emit_natural, emit_strict_json)
except ImportError:  # pragma: no cover
    from core.location_engine import (
        generate_location_records,
        get_available_location_sets,
    )
    from core.style_presets import STYLE_PRESETS
    from core.jb.palette import build_palette, resolve_tokens
    from core.jb.color_moods import MOOD_NAMES, mood_help
    from core.jb.serialize import (ALL_FORMATS, NATURAL, emit,
                                   emit_natural, emit_strict_json)


_HARMONY_TYPES = ["auto", "analogous", "complementary", "split_complementary",
                  "triadic", "tetradic", "monochromatic"]


def _location_set_choices() -> list[str]:
    sets = get_available_location_sets()
    return sets or ["indoor/everyday_us/family_living_room_tv", "outdoor/everyday_us/subdivision_sidewalk", "indoor/everyday_de/kitchen_cooking"]


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
                # ── Colour: drives the atmosphere tokens
                #    (#ambient_light#, #shadow_tone#) ──
                "color_mood":      (list(MOOD_NAMES), {"default": "everyday_muted",
                                    "tooltip": mood_help()}),
                "output_format":   (list(ALL_FORMATS), {"default": "loose_keys",
                                    "tooltip": "natural: plain prose, no keys and no "
                                    "metadata — use this for Krea 2 / Qwen text encoders.\n"
                                    "loose_keys / pretty_json / compact_json: structured, "
                                    "for Ideogram 4 style JSON prompting."}),
            },
            "optional": {
                "color_tone": (["", "warm", "cool", "neutral"], {"default": "",
                               "tooltip": "Overrides the tone derived from the palette."}),
                "num_colors":      ("INT", {"default": 5, "min": 2, "max": 8,
                                    "tooltip": "How many colours the scene draws from."}),
                "harmony_type":    (_HARMONY_TYPES, {"default": "auto",
                                    "tooltip": "Harmony engine only (color_mood = auto)."}),
                "palette_style":   (sorted(STYLE_PRESETS.keys()), {"default": "general",
                                    "tooltip": "Harmony engine only (color_mood = auto)."}),
                "vibrancy":        ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Harmony engine only (color_mood = auto)."}),
                "contrast":        ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Harmony engine only (color_mood = auto)."}),
                "warmth":          ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Warm/cool bias for ambient light and "
                                    "shadow phrases. Applies in every mood."}),
            },
        }

    def build(self, location_set, seed,
              enable_background, enable_midground, enable_architecture_detail,
              enable_props, enable_foreground_element,
              enable_time_of_day, enable_weather,
              color_mood="everyday_muted", output_format="loose_keys",
              color_tone="", num_colors=5, harmony_type="auto",
              palette_style="general", vibrancy=0.5, contrast=0.5, warmth=0.5):
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
            warmth=warmth, color_mood=color_mood,
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
        if output_format == NATURAL:
            location_string = emit_natural(location)
        else:
            location_string = emit(location, output_format)
        return (location_json, location_string, palette["palette_string"])
