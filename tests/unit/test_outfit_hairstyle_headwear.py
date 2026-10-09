"""Headwear entries without a material are hairstyles: no colour, no 'none'.

``polished low chignon | 0.30 | 0.8-1.0 | none`` used to come out as
"cornflower none polished low chignon" — a palette colour that Krea 2 paints
as hair colour, plus the literal placeholder word.
"""

import json

from core.outfit_engine import _build_description, is_hairstyle, is_no_fabric
from nodes.jb.outfit_block import FVM_JB_OutfitBlock


def test_placeholders_count_as_no_fabric():
    for f in (None, "", "-", "none", "None", " none "):
        assert is_no_fabric(f)
    assert not is_no_fabric("silk")
    assert not is_no_fabric("metal")


def test_hairstyle_only_in_headwear():
    assert is_hairstyle("headwear", "none")
    assert is_hairstyle("headwear", "-")
    assert not is_hairstyle("headwear", "silk")       # a headband is a garment
    assert not is_hairstyle("accessories", "none")    # earrings keep their metal tone


def test_hairstyle_gets_no_colour_and_no_placeholder():
    assert _build_description("#secondary#", "none", "polished low chignon",
                              slot="headwear") == "polished low chignon"
    assert _build_description("#secondary#", "-", "smooth side part",
                              slot="headwear") == "smooth side part"


def test_headband_still_coloured():
    assert _build_description("#secondary#", "silk", "slim headband",
                              slot="headwear") == "#secondary# silk slim headband"


def test_none_fabric_never_printed_outside_headwear():
    assert _build_description("#metallic#", "none", "small stud earrings",
                              slot="accessories") == "#metallic# small stud earrings"


def test_outfit_block_hairstyle_has_no_colour_in_text_or_json():
    """Any hairstyle draw: the sentence is the bare entry, the JSON has no colour."""
    block = FVM_JB_OutfitBlock()
    seen = 0
    for seed in range(40):
        j, txt, _ = block.build("female/business/power_meeting", seed, "business_executive", 0.9, 0.5,
                                True, True, True, True, False, False, False, 0.3, "off",
                                color_mood="auto", output_format="sentences")
        hw = json.loads(j)["outfit"]["garments"].get("headwear")
        if hw and hw["fabric"] is None:
            seen += 1
            assert f"The headwear is {hw['name']}." in txt, txt
            assert hw["color_resolved"] is None and hw["color_role"] is None
            assert " none " not in f" {txt} "
    assert seen >= 10, f"only {seen} hairstyle draws in 40 seeds"


def test_outerwear_over_a_dress_reads_as_outerwear():
    """Dress sets keep a "none" top stub; the blazer must not become "The top is"."""
    block = FVM_JB_OutfitBlock()
    seen = False
    for seed in range(80):
        j, txt, _ = block.build("female/business/dress", seed, "general", 0.7, 0.5,
                                False, True, True, True, True, False, False, 0.0, "off",
                                color_mood="auto", output_format="sentences")
        garments = json.loads(j)["outfit"]["garments"]
        if "upper_body_outerwear" in garments and "upper_body" not in garments:
            seen = True
            assert "The outerwear is" in txt
            assert "The top is" not in txt
    assert seen, "no dress + outerwear draw in 80 seeds"


BUSINESS_HEELS_SETS = [
    "female/business/board_meeting_suit_with_heels",
    "female/business/office_day_business_casual_with_heels",
    "female/business/senior_community_event_with_heels",
    "female/business/hospital_campus_visit_with_heels",
    "female/business/health_expo_booth_day_with_heels",
    "female/business/client_lunch_dress_with_heels",
    "female/business/awards_gala_evening_with_heels",
    "female/business/winter_workday_layers_with_heels",
]


def test_business_executive_preset_always_draws_the_hairstyle():
    """The face cue lives in headwear; 'general' drew it in only ~15 % of seeds."""
    block = FVM_JB_OutfitBlock()
    for outfit_set in BUSINESS_HEELS_SETS:
        for seed in range(12):
            _, txt, _ = block.build(outfit_set, seed, "business_executive", 0.7, 0.5,
                                    True, True, True, True, True, True, True, 0.15, "off",
                                    color_mood="everyday_muted", output_format="sentences")
            assert "The headwear is" in txt, (outfit_set, seed, txt)
            assert "toward the camera" in txt or "face" in txt, (outfit_set, seed, txt)
            assert "heel" in txt.split("The footwear is", 1)[1].split(".")[0], (outfit_set, seed, txt)
