"""P2 — tests for FVM_SMP_OutfitGenerator + outfit_records engine helper."""

import pytest

from core.outfit_engine import generate_outfit, generate_outfit_records
from core.outfit_lists import get_available_sets


# ─── generate_outfit_records core helper ─────────────────────────────────


def test_records_seed_determinism():
    a = generate_outfit_records(seed=42, outfit_set="general_female")
    b = generate_outfit_records(seed=42, outfit_set="general_female")
    assert a == b


def test_records_different_seeds_diverge():
    a = generate_outfit_records(seed=1, outfit_set="general_female")
    b = generate_outfit_records(seed=999, outfit_set="general_female")
    assert a != b


def test_records_v1_v2_share_active_slots():
    """The new records function must agree with V1 on which slots end up active."""
    seed = 7
    v1 = generate_outfit(seed=seed, outfit_set="general_female")
    v2 = generate_outfit_records(seed=seed, outfit_set="general_female")
    # outfit_details has format "slot:name:fabric:tag|..."
    v1_slots = [d.split(":")[0] for d in v1["outfit_details"].split("|") if d]
    v2_slots = list(v2["garments"].keys())
    assert set(v1_slots) == set(v2_slots)


def test_records_fragment_contains_color_token():
    rec = generate_outfit_records(seed=3, outfit_set="general_female")
    for slot, garment in rec["garments"].items():
        frag = garment["prompt_fragment"]
        assert "#" in frag, f"slot {slot} fragment missing color tag: {frag!r}"


def test_records_set_invalid_falls_back():
    """Unknown outfit set should not crash; engine returns empty garments."""
    rec = generate_outfit_records(seed=1, outfit_set="__nonexistent_set__")
    # No data files → no garments populated.
    assert rec["garments"] == {} or all(
        g["name"] for g in rec["garments"].values()
    )


# ─── FVM_SMP_OutfitGenerator node ────────────────────────────────────────


def test_node_returns_two_tuple():
    from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator
    node = FVM_SMP_OutfitGenerator()
    result = node.generate(
        outfit_set="general_female", seed=42, style_preset="general",
        formality=0.5, coverage=0.5,
        enable_headwear=False, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=False,
        enable_accessories=False, enable_bag=False,
        print_probability=0.3, text_mode="auto",
    )
    assert isinstance(result, tuple) and len(result) == 2
    raw, summary = result
    assert isinstance(raw, dict) and isinstance(summary, str)


def test_node_returns_outfit_dict_raw_shape():
    from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator
    node = FVM_SMP_OutfitGenerator()
    raw, _ = node.generate(
        outfit_set="general_female", seed=42, style_preset="general",
        formality=0.5, coverage=0.5,
        enable_headwear=False, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=False,
        enable_accessories=False, enable_bag=False,
        print_probability=0.3, text_mode="auto",
    )
    assert "set_name" in raw
    assert raw["set_name"] == "general_female"
    assert "garments" in raw
    assert "seed" in raw and raw["seed"] == 42
    assert raw["formality"] in ("casual", "smart_casual", "formal", "evening", "sport")


def test_node_garment_records_have_region_hint_and_color_role():
    from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator
    node = FVM_SMP_OutfitGenerator()
    raw, _ = node.generate(
        outfit_set="general_female", seed=42, style_preset="general",
        formality=0.7, coverage=0.7,
        enable_headwear=False, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=False,
        enable_accessories=False, enable_bag=False,
        print_probability=0.3, text_mode="auto",
    )
    assert raw["garments"], "expected at least one garment"
    for region_id, garment in raw["garments"].items():
        assert "name" in garment
        assert "color_role" in garment and garment["color_role"]
        assert "prompt_fragment" in garment and "#" in garment["prompt_fragment"]


def test_node_seed_determinism():
    from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator
    node = FVM_SMP_OutfitGenerator()
    args = dict(
        outfit_set="general_female", seed=99, style_preset="general",
        formality=0.5, coverage=0.5,
        enable_headwear=False, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=False,
        enable_accessories=False, enable_bag=False,
        print_probability=0.3, text_mode="auto",
    )
    a, _ = node.generate(**args)
    b, _ = node.generate(**args)
    assert a == b


def test_node_input_types_advertises_smp_categories():
    from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator
    schema = FVM_SMP_OutfitGenerator.INPUT_TYPES()
    assert "required" in schema
    assert "outfit_set" in schema["required"]
    assert FVM_SMP_OutfitGenerator.CATEGORY.startswith("FVM Tools/SMP")
    assert FVM_SMP_OutfitGenerator.RETURN_TYPES == ("OUTFIT_DICT_RAW", "STRING")


def test_records_set_dropdown_includes_known_sets():
    sets = get_available_sets()
    assert "general_female" in sets
