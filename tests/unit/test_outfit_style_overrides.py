"""Role-only outfit overrides: ``bottom: | primary`` keeps the auto garment
and only forces its colour role (mode "style")."""

import json

from core.outfit_engine import generate_outfit_records
from core.outfit_parser import parse_overrides
from nodes.jb.outfit_block import FVM_JB_OutfitBlock

SLOTS_ON = {
    s: True
    for s in (
        "headwear",
        "top",
        "outerwear",
        "bottom",
        "footwear",
        "accessories",
        "bag",
    )
}


def test_parser_role_without_garment_is_style_mode():
    ov = parse_overrides("bottom: | primary\nbag: | secondary | none\ntop: |")
    assert ov["bottom"] == {
        "garment": None,
        "fabric": None,
        "color_tag": "#primary#",
        "mode": "style",
        "decoration": None,
    }
    assert ov["bag"]["mode"] == "style" and ov["bag"]["decoration"] == "none"
    assert "top" not in ov  # empty spec after the pipe: ignored
    assert parse_overrides("top: silk blouse | accent")["top"]["mode"] == "override"


def test_style_override_keeps_garment_and_changes_role():
    for seed in range(20):
        plain = generate_outfit_records(seed, slot_enables=SLOTS_ON)["garments"]
        styled = generate_outfit_records(
            seed, slot_enables=SLOTS_ON, overrides=parse_overrides("bottom: | primary")
        )["garments"]
        assert plain.keys() == styled.keys()  # same slots rolled
        if "bottom" in styled:
            assert styled["bottom"]["name"] == plain["bottom"]["name"]
            assert styled["bottom"]["fabric"] == plain["bottom"]["fabric"]
            assert styled["bottom"]["color_tag"] == "#primary#"
            assert styled["bottom"]["is_override"] is False


def test_style_override_does_not_force_optional_slots_on():
    ov = parse_overrides("bag: | secondary")
    with_bag = sum(
        "bag"
        in generate_outfit_records(
            s, coverage=0.0, slot_enables=SLOTS_ON, overrides=ov
        )["garments"]
        for s in range(60)
    )
    without = sum(
        "bag"
        in generate_outfit_records(s, coverage=0.0, slot_enables=SLOTS_ON)["garments"]
        for s in range(60)
    )
    assert with_bag == without < 60


def _outfit(seed, overrides):
    return FVM_JB_OutfitBlock().build(
        outfit_set="general_female",
        seed=seed,
        style_preset="general",
        formality=0.5,
        coverage=0.6,
        enable_headwear=False,
        enable_top=True,
        enable_bottom=True,
        enable_footwear=True,
        enable_outerwear=False,
        enable_accessories=False,
        enable_bag=False,
        print_probability=0.0,
        text_mode="auto",
        overrides=overrides,
    )


def test_outfit_block_matching_top_and_bottom():
    matched = 0
    for seed in range(15):
        garments = json.loads(_outfit(seed, "top: | primary\nbottom: | primary")[0])[
            "outfit"
        ]["garments"]
        top = next((g for k, g in garments.items() if k.startswith("upper")), None)
        bottom = next((g for k, g in garments.items() if k.startswith("lower")), None)
        if not (top and bottom):
            continue  # dress sets leave the top empty
        matched += 1
        assert bottom["color_role"] == "primary"
        assert top["color_resolved"] == bottom["color_resolved"]
    assert matched >= 5
