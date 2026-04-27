"""SMP OutfitGenerator — dict-emitting V2 of the outfit generator.

Wraps ``core.outfit_engine.generate_outfit_records`` and adds region hints
from ``core.smp.defaults.DEFAULT_PERSON_REGIONS``. The output is an
``OUTFIT_DICT_RAW`` whose ``prompt_fragment`` strings still contain
``#primary#``-style color tokens for the OutfitCombiner to resolve.
"""

try:
    from ..outfit_generator import FVM_OutfitGenerator as _V1  # for INPUT_TYPES reuse
    from ...core.outfit_engine import generate_outfit_records
    from ...core.outfit_parser import parse_overrides
    from ...core.outfit_presets import OUTFIT_PRESETS
    from ...core.outfit_lists import get_available_sets
    from ...core.smp.defaults import (
        DEFAULT_COLOR_ROLE_BY_SLOT,
        DEFAULT_PERSON_REGIONS,
    )
except ImportError:  # pragma: no cover — pytest path
    from nodes.outfit_generator import FVM_OutfitGenerator as _V1
    from core.outfit_engine import generate_outfit_records
    from core.outfit_parser import parse_overrides
    from core.outfit_presets import OUTFIT_PRESETS
    from core.outfit_lists import get_available_sets
    from core.smp.defaults import (
        DEFAULT_COLOR_ROLE_BY_SLOT,
        DEFAULT_PERSON_REGIONS,
    )


_DEFAULT_FORMALITY_BUCKETS = ["casual", "smart_casual", "formal", "evening", "sport"]


def _formality_bucket(value: float) -> str:
    if value < 0.25:
        return "casual"
    if value < 0.55:
        return "smart_casual"
    if value < 0.85:
        return "formal"
    return "evening"


def _slot_to_garment_entry(rec: dict, region_id: str) -> dict:
    """Convert one engine record to the GarmentEntry shape (still raw, with #tokens#)."""
    slot = rec["slot"]
    region = DEFAULT_PERSON_REGIONS.get(region_id) or DEFAULT_PERSON_REGIONS.get(slot) or {}
    region_hint = None
    if region:
        region_hint = {
            "region_id": region_id,
            "sam_class_hint": region.get("sam_class"),
            "bbox_relative": region.get("bbox"),
            "layer_depth": "subject",
        }
    color_role = DEFAULT_COLOR_ROLE_BY_SLOT.get(slot) or "primary"
    return {
        "name": rec["name"],
        "probability": 1.0,
        "coverage": 0.0,
        "fabric": rec.get("fabric"),
        "color_role": color_role,
        "color_resolved": None,
        "prompt_fragment": rec["prompt_fragment"],
        "region_hint": region_hint,
        # Keep engine-level extras around for debugging / sidecar.
        "_decoration": rec.get("decoration"),
        "_color_tag": rec.get("color_tag"),
        "_is_override": rec.get("is_override", False),
    }


# Map engine slot names → the canonical SMP region id used by Assembler.
_SLOT_TO_REGION = {
    "headwear":    "headwear",
    "top":         "upper_body",
    "outerwear":   "upper_body",
    "bottom":      "lower_body",
    "footwear":    "footwear",
    "accessories": "accessories",
    "bag":         "bag",
}


class FVM_SMP_OutfitGenerator:
    """Dict-emitting outfit generator for the StructPromptMaker pipeline.

    Output is an OUTFIT_DICT_RAW (plain dict on the wire) matching the
    schema in ``core.smp.schema.OutfitDict`` — but with ``#primary#`` /
    ``#secondary#`` / ``#accent#`` tokens still embedded in
    ``prompt_fragment``. Run ``FVM_SMP_OutfitCombiner`` afterwards with a
    ``COLOR_PALETTE_DICT`` to resolve those tokens.
    """

    CATEGORY = "FVM Tools/SMP/Generators"
    FUNCTION = "generate"
    RETURN_TYPES = ("OUTFIT_DICT_RAW", "STRING")
    RETURN_NAMES = ("outfit_raw", "summary")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Dict-emitting V2 outfit generator (StructPromptMaker).\n\n"
        "Emits an OUTFIT_DICT_RAW with per-slot garment records, region hints,\n"
        "and #color# tokens still embedded in prompt_fragment. Pair with the\n"
        "FVM_SMP_OutfitCombiner to resolve the color tokens.\n\n"
        "Reuses the same outfit_lists data and engine as FVM_OutfitGenerator."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outfit_set":   (get_available_sets(),),
                "seed":         ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "style_preset": (sorted(OUTFIT_PRESETS.keys()), {"default": "general"}),
                "formality":    ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "coverage":     ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_headwear":    ("BOOLEAN", {"default": False}),
                "enable_top":         ("BOOLEAN", {"default": True}),
                "enable_bottom":      ("BOOLEAN", {"default": True}),
                "enable_footwear":    ("BOOLEAN", {"default": True}),
                "enable_outerwear":   ("BOOLEAN", {"default": False}),
                "enable_accessories": ("BOOLEAN", {"default": False}),
                "enable_bag":         ("BOOLEAN", {"default": False}),
                "print_probability": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "text_mode":         (["auto", "quoted", "descriptive", "off"], {"default": "auto"}),
            },
            "optional": {
                "overrides": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def generate(self, outfit_set, seed, style_preset, formality, coverage,
                 enable_headwear, enable_top, enable_bottom, enable_footwear,
                 enable_outerwear, enable_accessories, enable_bag,
                 print_probability, text_mode, overrides=""):
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
            seed=seed,
            outfit_set=outfit_set,
            style_preset=style_preset,
            formality=formality,
            coverage=coverage,
            slot_enables=slot_enables,
            overrides=parsed_overrides,
            print_probability=print_probability,
            text_mode=text_mode,
        )

        garments: dict = {}
        for slot, gr in rec["garments"].items():
            region_id = _SLOT_TO_REGION.get(slot, slot)
            garments[region_id] = _slot_to_garment_entry(gr, region_id)

        outfit_raw = {
            "set_name":        rec["outfit_set"],
            "seed":            rec["seed"],
            "formality":       _formality_bucket(rec["effective_formality"]),
            "coverage_target": rec["coverage_target"],
            "color_tone":      None,
            "garments":        garments,
            "_active_slots":   rec["active_slots"],
            "_style_preset":   rec["style_preset"],
        }

        summary_lines = [f"Set: {rec['outfit_set']} | Seed: {seed} | Slots: {len(garments)}"]
        for rid, g in garments.items():
            summary_lines.append(f"  {rid}: {g['prompt_fragment']}")
        return (outfit_raw, "\n".join(summary_lines))
