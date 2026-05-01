"""Extended location_lists validation: prefix convention, min count, dedupe,
plausibility (probability range, name word-count, indoor/outdoor banlist).

Complements test_smp_location.py (which covers schema + determinism).
"""

import os

import pytest

from core.location_engine import (
    _location_lists_root,
    get_available_location_sets,
    load_location_elements,
)


REQUIRED_FILES = [
    "background.txt",
    "midground.txt",
    "foreground_element.txt",
    "architecture_detail.txt",
    "props.txt",
    "time_of_day.txt",
    "weather.txt",
]
ATMOSPHERE_FILES = {"time_of_day.txt", "weather.txt"}
NON_ATMOSPHERE_FILES = [f for f in REQUIRED_FILES if f not in ATMOSPHERE_FILES]

MIN_ENTRIES = 10

# Plausibility — element-name tokens that contradict the indoor/outdoor frame.
# Match against lowercase whole-word substrings of the name only (texture and
# atmosphere files exempt — atmosphere conventionally describes shared lighting
# moods like "golden hour" that read either way).
INDOOR_BANLIST = [
    "trail", "ridge", "mountain", "forest canopy", "ocean", "beach",
    "appalachian", "summit", "wilderness", "switchback",
]
OUTDOOR_BANLIST = [
    "hardwood floor", "sofa", "office desk", "kitchen counter", "duvet",
    "shower stall", "bath tub", "indoor pool",
]


ALL_SETS = get_available_location_sets()


# ─── Discovery / convention ──────────────────────────────────────────────


def test_at_least_one_set_discovered():
    assert ALL_SETS, "no location sets discovered on disk"


def _set_scope(set_name: str) -> str:
    """Return 'indoor' or 'outdoor' based on slug leading segment.

    Supports both legacy flat slugs ('indoor_office_corporate_open_plan') and
    hierarchical slugs ('indoor/office/corporate_open_plan'). Returns '' if
    neither prefix matches.
    """
    if set_name.startswith("indoor_") or set_name.startswith("indoor/"):
        return "indoor"
    if set_name.startswith("outdoor_") or set_name.startswith("outdoor/"):
        return "outdoor"
    return ""


@pytest.mark.parametrize("set_name", ALL_SETS)
def test_set_has_indoor_or_outdoor_prefix(set_name):
    assert _set_scope(set_name), (
        f"set '{set_name}' missing indoor/outdoor prefix (flat or hierarchical)"
    )


@pytest.mark.parametrize("set_name", ALL_SETS)
def test_set_has_all_required_files(set_name):
    set_dir = os.path.join(_location_lists_root(), set_name)
    for fname in REQUIRED_FILES:
        path = os.path.join(set_dir, fname)
        assert os.path.isfile(path), f"missing {set_name}/{fname}"


# ─── Min count ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("set_name", ALL_SETS)
@pytest.mark.parametrize("fname", REQUIRED_FILES)
def test_file_meets_min_entries(set_name, fname):
    element_id = fname.replace(".txt", "")
    entries = load_location_elements(element_id, set_name)
    assert len(entries) >= MIN_ENTRIES, (
        f"{set_name}/{fname}: only {len(entries)} entries; expected >= {MIN_ENTRIES}"
    )


# ─── Dedupe ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("set_name", ALL_SETS)
@pytest.mark.parametrize("fname", REQUIRED_FILES)
def test_no_duplicate_names_within_file(set_name, fname):
    element_id = fname.replace(".txt", "")
    entries = load_location_elements(element_id, set_name)
    names = [e["name"].strip().lower() for e in entries]
    duplicates = [n for n in set(names) if names.count(n) > 1]
    assert not duplicates, (
        f"{set_name}/{fname}: duplicate name(s) {duplicates}"
    )


# ─── Plausibility ────────────────────────────────────────────────────────


@pytest.mark.parametrize("set_name", ALL_SETS)
@pytest.mark.parametrize("fname", REQUIRED_FILES)
def test_probability_in_curated_range(set_name, fname):
    element_id = fname.replace(".txt", "")
    entries = load_location_elements(element_id, set_name)
    for e in entries:
        assert 0.3 <= e["probability"] <= 1.0, (
            f"{set_name}/{fname}: probability {e['probability']} out of "
            f"curated range [0.3, 1.0] for {e['name']!r}"
        )


@pytest.mark.parametrize("set_name", ALL_SETS)
@pytest.mark.parametrize("fname", NON_ATMOSPHERE_FILES)
def test_non_atmosphere_name_has_at_least_two_words(set_name, fname):
    element_id = fname.replace(".txt", "")
    entries = load_location_elements(element_id, set_name)
    for e in entries:
        word_count = len(e["name"].split())
        assert word_count >= 2, (
            f"{set_name}/{fname}: name {e['name']!r} has only {word_count} word(s); "
            f"non-atmosphere names should be descriptive (>=2 words)"
        )


@pytest.mark.parametrize("set_name", ALL_SETS)
@pytest.mark.parametrize("fname", NON_ATMOSPHERE_FILES)
def test_no_banlist_violations(set_name, fname):
    """Indoor sets must not reference outdoor-only landscape tokens, and vice versa."""
    element_id = fname.replace(".txt", "")
    entries = load_location_elements(element_id, set_name)
    scope = _set_scope(set_name)
    if scope == "indoor":
        banlist = INDOOR_BANLIST
    else:
        banlist = OUTDOOR_BANLIST
    for e in entries:
        lname = e["name"].lower()
        for token in banlist:
            assert token not in lname, (
                f"{set_name}/{fname}: {scope} set name {e['name']!r} contains "
                f"{scope}-incompatible token {token!r}"
            )


# ─── Cross-reference with test_smp_location.py REQUIRED_SETS ─────────────


def test_smp_location_required_sets_match_disk():
    """REQUIRED_SETS in test_smp_location.py must enumerate every disk set
    that ships in the repo. Catches forgotten renames / orphaned references."""
    from tests.unit.test_smp_location import REQUIRED_SETS
    for name in REQUIRED_SETS:
        assert name in ALL_SETS, (
            f"test_smp_location.py REQUIRED_SETS lists '{name}' but it is "
            f"not on disk (after rename/cleanup?)"
        )
