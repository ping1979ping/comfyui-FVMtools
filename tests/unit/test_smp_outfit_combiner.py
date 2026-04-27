"""P2 — tests for FVM_SMP_OutfitCombiner."""

from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator
from nodes.smp.color_generator import FVM_SMP_ColorGenerator
from nodes.smp.outfit_combiner import FVM_SMP_OutfitCombiner


def _outfit_raw(seed=42):
    return FVM_SMP_OutfitGenerator().generate(
        outfit_set="general_female", seed=seed, style_preset="general",
        formality=0.5, coverage=0.7,
        enable_headwear=False, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=False,
        enable_accessories=False, enable_bag=False,
        print_probability=0.3, text_mode="auto",
    )[0]


def _palette(seed=1):
    return FVM_SMP_ColorGenerator().generate(
        seed=seed, num_colors=5, harmony_type="auto", style_preset="general",
        vibrancy=0.5, contrast=0.5, warmth=0.5,
    )[0]


def test_combiner_returns_two_tuple():
    raw, palette = _outfit_raw(), _palette()
    out = FVM_SMP_OutfitCombiner().combine(raw, palette)
    assert isinstance(out, tuple) and len(out) == 2
    resolved, summary = out
    assert isinstance(resolved, dict)
    assert isinstance(summary, str)


def test_combiner_resolves_all_color_tokens():
    raw, palette = _outfit_raw(), _palette()
    resolved, _ = FVM_SMP_OutfitCombiner().combine(raw, palette)
    for region_id, g in resolved["garments"].items():
        frag = g["prompt_fragment"]
        assert "#primary#" not in frag
        assert "#secondary#" not in frag
        assert "#accent#" not in frag
        assert "#neutral#" not in frag
        assert "#metallic#" not in frag
        assert "#tertiary#" not in frag
        # No leftover hash-tags at all
        assert "#" not in frag, f"unresolved token in {region_id}: {frag!r}"


def test_combiner_sets_color_resolved():
    raw, palette = _outfit_raw(), _palette()
    resolved, _ = FVM_SMP_OutfitCombiner().combine(raw, palette)
    for g in resolved["garments"].values():
        role = g.get("color_role")
        if role and role in palette["garment_colors"]:
            assert g["color_resolved"] == palette["garment_colors"][role]


def test_combiner_does_not_mutate_input():
    raw, palette = _outfit_raw(), _palette()
    raw_copy = {k: dict(v) if isinstance(v, dict) else v for k, v in raw.items()}
    raw_copy["garments"] = {k: dict(v) for k, v in raw["garments"].items()}
    FVM_SMP_OutfitCombiner().combine(raw, palette)
    # raw must remain with #tokens# intact
    for g in raw["garments"].values():
        assert "#" in g["prompt_fragment"], "input was mutated"


def test_combiner_inherits_palette_tone():
    raw, palette = _outfit_raw(), _palette()
    raw["color_tone"] = None
    resolved, _ = FVM_SMP_OutfitCombiner().combine(raw, palette)
    assert resolved["color_tone"] == palette["color_tone"]


def test_combiner_node_metadata():
    assert FVM_SMP_OutfitCombiner.CATEGORY.startswith("FVM Tools/SMP")
    assert FVM_SMP_OutfitCombiner.RETURN_TYPES == ("OUTFIT_DICT", "STRING")


def test_full_pipeline_dict_only_no_tokens():
    """End-to-end determinism on the OUTFIT_DICT (raw + combined)."""
    raw = _outfit_raw(seed=11)
    palette = _palette(seed=11)
    a, _ = FVM_SMP_OutfitCombiner().combine(raw, palette)
    b, _ = FVM_SMP_OutfitCombiner().combine(raw, palette)
    assert a == b


def test_combiner_resolves_color_marker_in_data_files():
    """Regression: outfit sets like aerobic_female embed `#color#` literally
    in garment names. The engine must substitute it with the role tag so the
    combiner sees a single role token to resolve, leaving no `#color#` behind.
    """
    # aerobic_female top.txt entries are of the form "#color# bodysuit ..."
    raw = FVM_SMP_OutfitGenerator().generate(
        outfit_set="aerobic_female", seed=0, style_preset="general",
        formality=0.0, coverage=0.5,
        enable_headwear=False, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=False,
        enable_accessories=False, enable_bag=False,
        print_probability=0.3, text_mode="auto",
    )[0]
    # Engine output must NOT contain the literal `#color#` marker any more
    for g in raw["garments"].values():
        assert "#color#" not in g["prompt_fragment"], (
            f"engine left literal #color# marker in: {g['prompt_fragment']!r}"
        )
    palette = _palette()
    resolved, _ = FVM_SMP_OutfitCombiner().combine(raw, palette)
    for g in resolved["garments"].values():
        assert "#" not in g["prompt_fragment"], (
            f"unresolved token in: {g['prompt_fragment']!r}"
        )
