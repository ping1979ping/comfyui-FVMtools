"""P3 — tests for FVM_JB_OutfitBlock and FVM_JB_LocationBlock."""

import json

import pytest

from nodes.jb.outfit_block import FVM_JB_OutfitBlock
from nodes.jb.location_block import FVM_JB_LocationBlock


# ─── OutfitBlock ─────────────────────────────────────────────────────


def _outfit(seed=42, **overrides):
    args = dict(
        outfit_set="general_female", seed=seed, style_preset="general",
        formality=0.5, coverage=0.6,
        enable_headwear=False, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=False,
        enable_accessories=False, enable_bag=False,
        print_probability=0.3, text_mode="auto",
        num_colors=5, harmony_type="auto", palette_style="general",
        vibrancy=0.5, contrast=0.5, warmth=0.5,
        output_format="loose_keys",
    )
    args.update(overrides)
    return FVM_JB_OutfitBlock().build(**args)


def test_outfit_block_metadata():
    assert FVM_JB_OutfitBlock.CATEGORY.startswith("FVM Tools/JB")
    assert FVM_JB_OutfitBlock.RETURN_TYPES == ("STRING", "STRING", "STRING")
    assert FVM_JB_OutfitBlock.RETURN_NAMES == ("outfit_json", "outfit_string", "palette_summary")


def test_outfit_block_returns_three_strings():
    outfit_json, outfit_string, palette_summary = _outfit()
    assert isinstance(outfit_json, str)
    assert isinstance(outfit_string, str)
    assert isinstance(palette_summary, str)


def test_outfit_block_emits_valid_json():
    outfit_json, _, _ = _outfit()
    parsed = json.loads(outfit_json)
    assert "outfit" in parsed
    assert "garments" in parsed["outfit"]
    assert parsed["outfit"]["set_name"] == "general_female"


def test_outfit_block_resolves_all_color_tokens():
    """Per the user's locked semantics — no #color# / #primary# leftover."""
    outfit_json, outfit_string, _ = _outfit()
    for s in (outfit_json, outfit_string):
        assert "#" not in s, f"unresolved token in: {s[:200]}"


def test_outfit_block_hosiery_set_resolves_color_marker():
    """aerobic_female has #color# markers in garment names — must resolve."""
    outfit_json, outfit_string, _ = _outfit(
        seed=1, outfit_set="aerobic_female", formality=0.0, coverage=0.6,
    )
    assert "#color#" not in outfit_json
    assert "#color#" not in outfit_string


def test_outfit_block_seed_determinism():
    a = _outfit(seed=99)
    b = _outfit(seed=99)
    assert a == b


def test_outfit_block_loose_keys_has_unquoted_keys():
    _, outfit_string, _ = _outfit(seed=1)
    assert "outfit:" in outfit_string
    assert '"outfit"' not in outfit_string


def test_outfit_block_pretty_json_is_valid():
    outfit_json, outfit_string, _ = _outfit(seed=1, output_format="pretty_json")
    json.loads(outfit_json)
    json.loads(outfit_string)


def test_outfit_block_palette_summary_is_csv():
    _, _, palette_summary = _outfit()
    parts = [p.strip() for p in palette_summary.split(",")]
    assert len(parts) >= 2
    for p in parts:
        assert p


# ─── LocationBlock ───────────────────────────────────────────────────


def _location(seed=42, **overrides):
    args = dict(
        location_set="urban_brutalist", seed=seed,
        enable_background=True, enable_midground=False,
        enable_architecture_detail=False, enable_props=False,
        enable_foreground_element=True,
        enable_time_of_day=True, enable_weather=True,
        num_colors=5, harmony_type="auto", palette_style="general",
        vibrancy=0.5, contrast=0.5, warmth=0.5,
        output_format="loose_keys", color_tone="",
    )
    args.update(overrides)
    return FVM_JB_LocationBlock().build(**args)


def test_location_block_metadata():
    assert FVM_JB_LocationBlock.CATEGORY.startswith("FVM Tools/JB")
    assert FVM_JB_LocationBlock.RETURN_TYPES == ("STRING", "STRING", "STRING")


def test_location_block_returns_three_strings():
    location_json, location_string, palette_summary = _location()
    assert isinstance(location_json, str)
    assert isinstance(location_string, str)
    assert isinstance(palette_summary, str)


def test_location_block_emits_valid_json():
    location_json, _, _ = _location()
    parsed = json.loads(location_json)
    assert "location" in parsed
    assert "elements" in parsed["location"]
    assert parsed["location"]["set_name"] == "urban_brutalist"


def test_location_block_resolves_atmosphere_tokens():
    location_json, location_string, _ = _location()
    for s in (location_json, location_string):
        assert "#ambient_light#" not in s
        assert "#shadow_tone#" not in s
        assert "#" not in s


def test_location_block_seed_determinism():
    a = _location(seed=7)
    b = _location(seed=7)
    assert a == b


def test_location_block_three_sets_all_work():
    for set_name in ("urban_brutalist", "beach_mediterranean", "studio_minimal"):
        loc_json, _, _ = _location(seed=1, location_set=set_name)
        parsed = json.loads(loc_json)
        assert parsed["location"]["set_name"] == set_name


# ─── End-to-end JB pipeline through Stitcher ─────────────────────────


def test_outfit_plus_location_plus_stitcher_into_one_character():
    from nodes.jb.stitcher import FVM_JB_Stitcher

    outfit_json, _, _ = _outfit(seed=1)
    location_json, _, _ = _location(seed=1)

    raw, string_out = FVM_JB_Stitcher().stitch(
        title="character_1", output_format="loose_keys",
        input_1=outfit_json, input_2=location_json,
    )
    parsed = json.loads(raw)
    assert "character_1" in parsed
    assert "outfit" in parsed["character_1"]
    assert "location" in parsed["character_1"]

    # No leftover tokens in either form
    assert "#" not in raw
    assert "#" not in string_out


def test_outfit_block_then_extractor_pulls_subtree():
    from nodes.jb.extractor import FVM_JB_Extractor

    outfit_json, _, _ = _outfit(seed=1)
    raw, _, found = FVM_JB_Extractor().extract(
        outfit_json, "outfit.garments", "loose_keys"
    )
    assert found is True
    parsed = json.loads(raw)
    # Wrapped under the last path segment ("garments")
    assert "garments" in parsed
    garments = parsed["garments"]
    # At minimum top + bottom + footwear regions should land
    assert any(k in garments for k in ("upper_body", "lower_body", "footwear"))
