"""Multi-select in the JB Outfit / Location Block (core/jb/multi_pick.py).

The checkbox modal writes the checked choices into a hidden ``*_multi``
widget; the block draws one per seed. The draw must not touch the engines'
RNG: the outfit for a drawn set has to equal the outfit you get by picking
that set alone with the same seed, or a good batch result can't be redone.
"""

import json

import pytest

from core.jb.color_moods import MOOD_NAMES
from core.jb.multi_pick import parse_choices, pick_choice
from core.location_engine import get_available_location_sets
from core.outfit_lists import get_available_sets
from nodes.jb.location_block import FVM_JB_LocationBlock
from nodes.jb.outfit_block import FVM_JB_OutfitBlock

OUTFIT_POOL = [
    "female/business/board_meeting_suit_with_heels",
    "female/business/client_lunch_dress_with_heels",
    "female/business/office_day_business_casual_with_heels",
]
LOCATION_POOL = [
    "indoor/luxury_hotel/hotel_bar",
    "outdoor/american_everyday_scene/sidewalk_cafe_patio",
    "outdoor/suburban_office_park/landscaped_front_entrance",
]
MOOD_POOL = ["warm_earth", "cool_muted", "pastel"]


def _outfit(seed, **extra):
    args = dict(
        outfit_set="female/general/general", seed=seed, style_preset="business_executive",
        formality=0.7, coverage=0.5,
        enable_headwear=True, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=True,
        enable_accessories=True, enable_bag=False,
        print_probability=0.3, text_mode="off", output_format="sentences",
    )
    args.update(extra)
    return FVM_JB_OutfitBlock().build(**args)


def _location(seed, **extra):
    args = dict(
        location_set="indoor/household_scene/laundry_room", seed=seed,
        enable_background=True, enable_midground=True,
        enable_architecture_detail=True, enable_props=True,
        enable_foreground_element=True, enable_time_of_day=True,
        enable_weather=True, output_format="sentences",
    )
    args.update(extra)
    return FVM_JB_LocationBlock().build(**args)


# ─── helper ──────────────────────────────────────────────────────────


def test_parse_choices_lines_blanks_comments_repeats():
    text = "b\r\n\n  a  \n# note\nb\n"
    assert parse_choices(text) == ["b", "a"]
    assert parse_choices("") == []
    assert parse_choices(None) == []


def test_empty_selection_falls_back_to_dropdown():
    assert pick_choice("", "x", ["x", "y"], 7, "s") == ("x", [], [])


def test_unknown_names_are_skipped_and_reported():
    choice, pool, skipped = pick_choice("y\ngone\nz", "x", ["x", "y", "z"], 7, "s")
    assert pool == ["y", "z"]
    assert skipped == ["gone"]
    assert choice in pool


def test_only_unknown_names_fall_back():
    assert pick_choice("gone", "x", ["x", "y"], 7, "s") == ("x", [], ["gone"])


def test_single_checked_choice_is_used():
    assert pick_choice("y", "x", ["x", "y"], 7, "s")[0] == "y"


def test_draw_is_deterministic_and_order_free():
    a = [pick_choice("a\nb\nc", "-", "abc", s, "salt")[0] for s in range(50)]
    b = [pick_choice("c\na\nb", "-", "abc", s, "salt")[0] for s in range(50)]
    assert a == b


def test_every_checked_choice_gets_drawn():
    drawn = {pick_choice("a\nb\nc", "-", "abc", s, "salt")[0] for s in range(100)}
    assert drawn == {"a", "b", "c"}


def test_salt_separates_set_and_mood_draws():
    sets = [pick_choice("a\nb\nc", "-", "abc", s, "outfit_set")[0] for s in range(40)]
    moods = [pick_choice("a\nb\nc", "-", "abc", s, "outfit_color_mood")[0] for s in range(40)]
    assert sets != moods


def test_pools_exist_on_disk():
    assert set(OUTFIT_POOL) <= set(get_available_sets())
    assert set(LOCATION_POOL) <= set(get_available_location_sets())
    assert set(MOOD_POOL) <= set(MOOD_NAMES)


# ─── Outfit Block ────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [1, 11, 22, 33])
def test_outfit_drawn_set_equals_single_selection(seed):
    multi = _outfit(seed, outfit_set_multi="\n".join(OUTFIT_POOL))
    drawn = json.loads(multi[0])["outfit"]["set_name"]
    assert drawn in OUTFIT_POOL
    assert multi == _outfit(seed, outfit_set=drawn)


def test_outfit_multi_varies_the_set_over_seeds():
    drawn = {json.loads(_outfit(s, outfit_set_multi="\n".join(OUTFIT_POOL))[0])
             ["outfit"]["set_name"] for s in range(30)}
    assert drawn == set(OUTFIT_POOL)


@pytest.mark.parametrize("seed", [1, 11, 22])
def test_outfit_drawn_mood_equals_single_selection(seed):
    multi = _outfit(seed, color_mood_multi="\n".join(MOOD_POOL))
    mood = pick_choice("\n".join(MOOD_POOL), "", MOOD_NAMES, seed, "outfit_color_mood")[0]
    single = _outfit(seed, color_mood=mood)
    assert multi[:2] == single[:2]
    assert multi[2] == f"{single[2]}  [mood: {mood}]"


def test_outfit_without_multi_is_unchanged():
    assert _outfit(5) == _outfit(5, outfit_set_multi="", color_mood_multi="")
    assert "[mood:" not in _outfit(5)[2]


# ─── Location Block ──────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [1, 11, 22, 33])
def test_location_drawn_set_equals_single_selection(seed):
    multi = _location(seed, location_set_multi="\n".join(LOCATION_POOL))
    drawn = json.loads(multi[0])["location"]["set_name"]
    assert drawn in LOCATION_POOL
    assert multi == _location(seed, location_set=drawn)


@pytest.mark.parametrize("seed", [1, 11, 22])
def test_location_drawn_mood_equals_single_selection(seed):
    multi = _location(seed, color_mood_multi="\n".join(MOOD_POOL))
    mood = pick_choice("\n".join(MOOD_POOL), "", MOOD_NAMES, seed, "location_color_mood")[0]
    single = _location(seed, color_mood=mood)
    assert multi[:2] == single[:2]
    assert multi[2] == f"{single[2]}  [mood: {mood}]"


# ─── widget order (saved workflows restore widgets_values by position) ───


@pytest.mark.parametrize("cls,tail", [
    (FVM_JB_OutfitBlock, ["text_probability", "outfit_set_multi", "color_mood_multi"]),
    (FVM_JB_LocationBlock, ["warmth", "location_set_multi", "color_mood_multi"]),
])
def test_multi_widgets_are_appended_last(cls, tail):
    spec = cls.INPUT_TYPES()
    order = list(spec["required"]) + list(spec["optional"])
    assert order[-len(tail):] == tail
    for name in tail[1:]:
        assert spec["optional"][name][0] == "STRING"
        assert spec["optional"][name][1]["default"] == ""
