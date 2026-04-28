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
    from ...core.jb.serialize import ALL_FORMATS, emit, emit_strict_json
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
    from core.jb.serialize import ALL_FORMATS, emit, emit_strict_json
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
                "print_probability": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "text_mode":         (["auto", "quoted", "descriptive", "off"], {"default": "auto"}),
                # Color section
                "num_colors":      ("INT", {"default": 5, "min": 2, "max": 8}),
                "harmony_type":    (_HARMONY_TYPES, {"default": "auto"}),
                "palette_style":   (sorted(STYLE_PRESETS.keys()), {"default": "general"}),
                "vibrancy":        ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "contrast":        ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "warmth":          ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "output_format":   (list(ALL_FORMATS), {"default": "loose_keys"}),
            },
            "optional": {
                "overrides": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def build(self, outfit_set, seed, style_preset, formality, coverage,
              enable_headwear, enable_top, enable_bottom, enable_footwear,
              enable_outerwear, enable_accessories, enable_bag,
              print_probability, text_mode,
              num_colors, harmony_type, palette_style, vibrancy, contrast, warmth,
              output_format, overrides=""):
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
            warmth=warmth,
        )
        subs = palette["subs"]

        garments: dict = {}
        for slot, gr in rec["garments"].items():
            region_id = _SLOT_TO_REGION.get(slot, slot)
            garments[region_id] = _slot_to_garment(gr, region_id, subs)

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
        outfit_string = emit(outfit, output_format)
        return (outfit_json, outfit_string, palette["palette_string"])
