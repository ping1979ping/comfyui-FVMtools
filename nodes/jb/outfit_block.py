"""FVM_JB_OutfitBlock — combo node merging Outfit Generator + Color Generator + Combiner.

One node, all the choices: pick an outfit set, twist the seed and sliders,
and get back a fully-resolved JSON describing the outfit ready for
stitching into a character prompt.

Reuses:
  - core.outfit_engine.generate_outfit_records (V2, dict-emitting)
  - core.jb.palette.build_palette (palette + token subs + atmosphere)
  - core.jb.palette.resolve_tokens (single-pass #token# substitution)
  - core.jb.serialize.emit (output formatting)

Edit List button (`web/js/fvm_outfit_generator.js`) wires automatically
because the dropdown widget is named ``outfit_set`` — same as the V1 node.
"""

from __future__ import annotations

try:
    from ...core.outfit_engine import generate_outfit_records
    from ...core.outfit_parser import parse_overrides
    from ...core.outfit_presets import OUTFIT_PRESETS
    from ...core.outfit_lists import get_available_sets
    from ...core.style_presets import STYLE_PRESETS
    from ...core.jb.palette import build_palette, resolve_tokens
    from ...core.jb.color_moods import MOOD_NAMES, mood_help
    from ...core.jb.serialize import (ALL_FORMATS, NATURAL, emit,
                                      emit_natural, emit_strict_json)
    from ...core.smp.defaults import (
        DEFAULT_COLOR_ROLE_BY_SLOT,
        DEFAULT_PERSON_REGIONS,
    )
except ImportError:  # pragma: no cover
    from core.outfit_engine import generate_outfit_records
    from core.outfit_parser import parse_overrides
    from core.outfit_presets import OUTFIT_PRESETS
    from core.outfit_lists import get_available_sets
    from core.style_presets import STYLE_PRESETS
    from core.jb.palette import build_palette, resolve_tokens
    from core.jb.color_moods import MOOD_NAMES, mood_help
    from core.jb.serialize import (ALL_FORMATS, NATURAL, emit,
                                   emit_natural, emit_strict_json)
    from core.smp.defaults import (
        DEFAULT_COLOR_ROLE_BY_SLOT,
        DEFAULT_PERSON_REGIONS,
    )


_HARMONY_TYPES = ["auto", "analogous", "complementary", "split_complementary",
                  "triadic", "tetradic", "monochromatic"]

# Engine slot names → canonical SMP region ids used in JSON output.
_SLOT_TO_REGION = {
    "headwear":    "headwear",
    "top":         "upper_body",
    "outerwear":   "upper_body",
    "bottom":      "lower_body",
    "footwear":    "footwear",
    "accessories": "accessories",
    "bag":         "bag",
}


def _formality_bucket(value: float) -> str:
    if value < 0.25:  return "casual"
    if value < 0.55:  return "smart_casual"
    if value < 0.85:  return "formal"
    return "evening"


def _slot_to_garment(rec: dict, region_id: str, subs: dict[str, str]) -> dict:
    slot = rec["slot"]
    color_role = DEFAULT_COLOR_ROLE_BY_SLOT.get(slot) or "primary"
    # Some outfit lists embed a literal '#color#' placement marker in the
    # garment name (see core/outfit_lists.py:72). Strip it from the cosmetic
    # ``name`` field so the JSON output never carries unresolved tokens.
    raw_name = (rec.get("name") or "").replace("#color#", "").strip()
    raw_name = " ".join(raw_name.split())  # collapse double spaces from the strip
    fragment = resolve_tokens(rec["prompt_fragment"], subs)
    return {
        "name":            raw_name,
        "fabric":          rec.get("fabric"),
        "color_role":      color_role,
        "color_resolved":  subs.get(f"#{color_role}#"),
        "prompt_fragment": fragment,
    }


class FVM_JB_OutfitBlock:
    """Outfit Generator + Color Generator + Outfit Combiner in a single node."""

    CATEGORY = "FVM Tools/JB"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("outfit_json", "outfit_string", "palette_summary")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Outfit Generator + Color Generator + Combiner merged into one node.\n\n"
        "Pick an outfit set, twist the sliders, get back a fully-resolved\n"
        "JSON outfit ready to feed into a JB Stitcher under your character title.\n\n"
        "Reuses the same outfit_lists/ data and Edit List modal as the V1\n"
        "FVM_OutfitGenerator — edit the source files directly from the node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outfit_set":    (get_available_sets(),),
                "seed":          ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "style_preset":  (sorted(OUTFIT_PRESETS.keys()), {"default": "general"}),
                "formality":     ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "coverage":      ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_headwear":    ("BOOLEAN", {"default": False}),
                "enable_top":         ("BOOLEAN", {"default": True}),
                "enable_bottom":      ("BOOLEAN", {"default": True}),
                "enable_footwear":    ("BOOLEAN", {"default": True}),
                "enable_outerwear":   ("BOOLEAN", {"default": False}),
                "enable_accessories": ("BOOLEAN", {"default": False}),
                "enable_bag":         ("BOOLEAN", {"default": False}),
                "print_probability": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Chance of a pattern per garment. "
                                      "'solid color' entries are never written into the "
                                      "prompt — they say nothing."}),
                "text_mode":         (["auto", "quoted", "descriptive", "off"], {"default": "off",
                                      "tooltip": "Slogans printed on the garment. Krea 2 "
                                      "renders quoted text literally onto the clothing, so "
                                      "'off' is the default; 'quoted' is for Ideogram-style "
                                      "text rendering."}),
                # ── Colour ──
                "color_mood":      (list(MOOD_NAMES), {"default": "everyday_muted",
                                    "tooltip": mood_help()}),
                "output_format":   (list(ALL_FORMATS), {"default": "loose_keys",
                                    "tooltip": "natural: plain prose, no keys and no "
                                    "metadata — use this for Krea 2 / Qwen text encoders.\n"
                                    "loose_keys / pretty_json / compact_json: structured, "
                                    "for Ideogram 4 style JSON prompting."}),
            },
            "optional": {
                "overrides": ("STRING", {"default": "", "multiline": True}),
                # Only consulted when color_mood is "auto" (except palette_style
                # and warmth, which also tint the atmosphere phrases).
                "num_colors":      ("INT", {"default": 5, "min": 2, "max": 8,
                                    "tooltip": "How many colours the outfit draws from."}),
                "harmony_type":    (_HARMONY_TYPES, {"default": "auto",
                                    "tooltip": "Harmony engine only (color_mood = auto)."}),
                "palette_style":   (sorted(STYLE_PRESETS.keys()), {"default": "general",
                                    "tooltip": "Harmony engine only (color_mood = auto)."}),
                "vibrancy":        ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Harmony engine only (color_mood = auto)."}),
                "contrast":        ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Harmony engine only (color_mood = auto)."}),
                "warmth":          ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Warm/cool bias. Also tints the ambient "
                                    "light and shadow phrases in every mood."}),
            },
        }

    def build(self, outfit_set, seed, style_preset, formality, coverage,
              enable_headwear, enable_top, enable_bottom, enable_footwear,
              enable_outerwear, enable_accessories, enable_bag,
              print_probability, text_mode,
              color_mood="everyday_muted", output_format="loose_keys",
              overrides="", num_colors=5, harmony_type="auto",
              palette_style="general", vibrancy=0.5, contrast=0.5, warmth=0.5):
        slot_enables = {
            "headwear":    enable_headwear,
            "top":         enable_top,
            "bottom":      enable_bottom,
            "footwear":    enable_footwear,
            "outerwear":   enable_outerwear,
            "accessories": enable_accessories,
            "bag":         enable_bag,
        }
        parsed_overrides = parse_overrides(overrides) if overrides else {}

        rec = generate_outfit_records(
            seed=seed, outfit_set=outfit_set, style_preset=style_preset,
            formality=formality, coverage=coverage,
            slot_enables=slot_enables, overrides=parsed_overrides,
            print_probability=print_probability, text_mode=text_mode,
        )

        palette = build_palette(
            seed=seed, num_colors=num_colors, harmony_type=harmony_type,
            style_preset=palette_style, vibrancy=vibrancy, contrast=contrast,
            warmth=warmth, color_mood=color_mood,
        )
        subs = palette["subs"]

        garments: dict = {}
        for slot, gr in rec["garments"].items():
            region_id = _SLOT_TO_REGION.get(slot, slot)
            # "top" and "outerwear" share the upper_body region, so writing both
            # under the same key silently dropped the shirt whenever a jacket was
            # enabled. Keep the region id for the first (the top) and give the
            # layer above its own key — existing consumers still read upper_body.
            key = region_id
            if key in garments:
                key = f"{region_id}_{slot}" if slot != region_id else f"{region_id}_2"
            garments[key] = _slot_to_garment(gr, region_id, subs)

        outfit = {
            "outfit": {
                "set_name":        rec["outfit_set"],
                "seed":            rec["seed"],
                "formality":       _formality_bucket(rec["effective_formality"]),
                "coverage_target": rec["coverage_target"],
                "color_tone":      palette["color_tone"],
                "garments":        garments,
            }
        }

        outfit_json = emit_strict_json(outfit, indent=2)
        if output_format == NATURAL:
            # "wearing …" makes the fragment drop straight into a sentence.
            outfit_string = emit_natural(outfit, prefix="wearing ")
        else:
            outfit_string = emit(outfit, output_format)
        return (outfit_json, outfit_string, palette["palette_string"])
