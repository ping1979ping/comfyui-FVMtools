"""Krea 2 composition rules for location sets built after 2026-10-09.

Render tests on Krea 2 Turbo (fixed seeds, A/B) showed what the older sets
get wrong: every noun the prompt mentions is painted, and painted large and
close. A coffee mug in ``props.txt`` lands next to the subject's hand, a
lamppost in ``architecture_detail.txt`` stands at her shoulder, anything in
``foreground_element.txt`` fills the lower third. What works instead:

* background and midground carry the scene — longer entries that describe
  the whole far wall / far side of the street *with a relative position*
  ("across the whole width of the image", "a few meters back on the left");
* props and details are few, explicitly small, explicitly far away;
* the foreground is the ground plane, nothing else;
* no people words, no implied writing, no parentheses (``CLIPTextEncode``
  parses them as weights), and the whole location stays short enough that
  outfit + person + location stay under Krea 2's ~512-token window.

The older sets predate these rules and are not checked. Categories listed
in ``KREA2_STRICT_CATEGORIES`` are — add a new category here when it is
built with ``/build-location-set``.
"""

from __future__ import annotations

import os
import re

import pytest

from core.list_braces import split_fields
from core.location_engine import _location_lists_root, get_available_location_sets

KREA2_STRICT_CATEGORIES = {
    "reading_pennsylvania_corporate_office",
    "reading_pennsylvania_downtown_spot",
    "senior_living_community",
    "hospital_campus",
    "business_event_venue",
    "business_dining_spot",
    "suburban_office_park",
}

# Older sets rewritten to the same rules, set by set (their categories also
# hold sets that were not rewritten).
KREA2_STRICT_EXTRA_SETS = {
    "indoor/american_office/break_room",
    "indoor/american_office/cubicle_pod",
    "indoor/american_office/plain_conference_room",
    "indoor/american_everyday_scene/independent_coffee_shop",
    "outdoor/american_everyday_scene/sidewalk_cafe_patio",
    "indoor/luxury_hotel/hotel_bar",
    "indoor/luxury_hotel/hotel_breakfast_restaurant",
    "indoor/luxury_hotel/hotel_cafe",
    "indoor/luxury_hotel/hotel_lobby",
}

STRICT_SETS = [s for s in get_available_location_sets()
               if (s.split("/")[1:2] and s.split("/")[1] in KREA2_STRICT_CATEGORIES)
               or s in KREA2_STRICT_EXTRA_SETS]

# name + texture, characters, after brace markup is stripped.
MAX_CHARS = {
    "background": 260,
    "midground": 210,
    "architecture_detail": 120,
    "props": 115,
    "foreground_element": 100,
    "time_of_day": 60,
    "weather": 60,
}
MIN_CHARS = {"background": 90, "midground": 60}
# Longest possible draw over all seven elements (name + texture). Lead-ins,
# intro sentence and atmosphere tokens add ~350 characters on top; outfit and
# person need the rest of the ~2400-character budget.
MAX_SET_CHARS = 950

POSITION_CUES = (
    "far", "behind", "back", "across", "distance", "distant", "beyond",
    "left", "right", "above", "overhead", "high up", "along", "end of",
    "rear", "edge", "corner", "middle", "center", "centre", "background",
    "meters", "down the", "up the", "side", "width", "horizon", "length",
)
SIZE_CUES = ("small", "tiny", "slim", "thin", "little", "compact", "narrow", "low")
FAR_CUES = ("far", "distant", "distance", "back", "background", "behind",
            "down the", "across", "end of", "rear", "beyond", "corner", "edge")

# A size cue that floats free of its object is filled with a small human
# figure: "a small trash can far down the sidewalk, tiny in the distance" put a
# child-sized figure on the sidewalk in 4 of 4 seeds; anchored to "at the curb
# far down the block" it was gone.
FREE_FLOATING_SIZE = ("tiny in the distance", "tiny in the frame", "small in the distance",
                      "small in the frame", "tiny far away", "far away, tiny")

# Seating without an "empty" cue gets occupied: "low cognac leather chairs
# around a plain wood table" put two seated guests into a coffee-roaster
# render; the same entry with "all empty" did not.
SEATING_WORDS = ("chair", "chairs", "armchair", "armchairs", "booth", "booths", "banquette",
                 "banquettes", "sofa", "sofas", "loveseat", "loveseats", "stool", "stools",
                 "bench", "benches", "seating", "seat", "seats", "table", "tables",
                 "high-top", "high-tops", "couch", "couches")
EMPTY_CUES = ("empty", "unoccupied", "vacant", "pushed in", "tucked under", "tucked in",
              "set for", "stacked", "folded")

GROUND_WORDS = (
    "floor", "carpet", "pavement", "sidewalk", "paver", "concrete", "tile",
    "plank", "wood", "lawn", "grass", "path", "asphalt", "gravel", "flagstone",
    "deck", "terrazzo", "stone", "rug", "runner", "boardwalk", "mulch", "brick",
    "slate", "marble", "granite", "parquet", "vinyl", "walkway", "ground",
    "curb", "pavers", "planking", "linoleum", "patio",
)
# Repetitive micro-patterns that the cielbleu Krea 2 finetune turns into
# smeared blotches on large floor areas (seen on carpet tile, 2026-10-09).
FLOOR_PATTERN_BAN = ("carpet tile", "mosaic", "checkered", "checkerboard")

PEOPLE_WORDS = {
    "people", "person", "persons", "man", "men", "woman", "women", "child",
    "children", "kid", "kids", "crowd", "staff", "employee", "employees",
    "receptionist", "nurse", "nurses", "doctor", "doctors", "patient",
    "patients", "resident", "residents", "visitor", "visitors", "guest",
    "guests", "diner", "diners", "waiter", "waiters", "waitress", "bartender",
    "barista", "attendee", "attendees", "audience", "speaker", "speakers",
    "vendor", "vendors", "pedestrian", "pedestrians", "passerby", "passersby",
    "couple", "family", "someone", "figure", "figures", "her", "his", "him",
    "she", "he", "exhibitor", "exhibitors", "worker", "workers", "senior",
    "seniors", "caregiver", "caregivers", "volunteer", "volunteers", "busy",
    "bustling", "lively", "crowded",
}
WRITING_WORDS = (
    "sign", "signs", "signage", "poster", "posters", "menu", "menus", "banner",
    "banners", "whiteboard", "chalkboard", "notice", "label", "labels",
    "plaque", "brochure", "brochures", "newspaper", "magazine", "logo",
    "lettering", "calendar", "directory", "marquee", "billboard", "placard",
    "nameplate", "name tag", "screen", "screens", "monitor", "monitors",
    "slide", "slides", "flyer", "flyers", "pamphlet", "book spines", "ticker",
)
WRITING_OK = ("blank", "unlit", "dark", "abstract", "pictogram", "plain",
              "empty", "switched off", "black", '"')


def _strip_braces(text: str) -> str:
    """Longest-case approximation: keep every brace option, drop markup."""
    return re.sub(r"%\d+(?:\.\d+)?%", "", text).replace("{", "").replace("}", "").replace("|", " ")


def _entries(set_name: str, element: str):
    path = os.path.join(_location_lists_root(), set_name, f"{element}.txt")
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = split_fields(line)
            name = parts[0]
            texture = parts[3] if len(parts) > 3 and parts[3] != "-" else ""
            yield line, _strip_braces(f"{name}, {texture}" if texture else name)


def _longest_option(text: str) -> int:
    """Length of the longest single draw of a ``{a|b}`` text."""
    def pick(m):
        return max(m.group(1).split("|"), key=len)
    prev = None
    while prev != text:
        prev, text = text, re.sub(r"\{([^{}]*)\}", pick, text)
    return len(re.sub(r"%\d+(?:\.\d+)?%", "", text))


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower()))


def _has(text: str, cues) -> bool:
    """Whole-word / whole-phrase match ("far" must not hit "farmhouse")."""
    return any(re.search(r"\b" + re.escape(c) + r"\b", text) for c in cues)


def test_strict_categories_exist():
    have = {s.split("/")[1] for s in STRICT_SETS}
    missing = KREA2_STRICT_CATEGORIES - have
    assert not missing, f"strict categories without sets on disk: {sorted(missing)}"


@pytest.mark.parametrize("set_name", STRICT_SETS)
def test_entry_lengths(set_name):
    bad = []
    for element, cap in MAX_CHARS.items():
        for line, text in _entries(set_name, element):
            longest = _longest_option(line.split("|", 1)[0]) if element in ("time_of_day", "weather") else None
            n = longest if longest is not None else len(text)
            if n > cap:
                bad.append(f"{element}: {n}>{cap}: {text[:70]}")
            if element in MIN_CHARS and len(text) < MIN_CHARS[element]:
                bad.append(f"{element}: {len(text)}<{MIN_CHARS[element]} (give the scope more words): {text}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("set_name", STRICT_SETS)
def test_set_fits_prompt_budget(set_name):
    total = 0
    for element in MAX_CHARS:
        lengths = [len(t) for _, t in _entries(set_name, element)]
        total += max(lengths) if lengths else 0
    assert total <= MAX_SET_CHARS, f"{set_name}: worst-case draw {total} chars > {MAX_SET_CHARS}"


@pytest.mark.parametrize("set_name", STRICT_SETS)
def test_scene_elements_carry_a_position(set_name):
    bad = []
    for element in ("background", "midground", "architecture_detail", "props"):
        for _, text in _entries(set_name, element):
            low = text.lower()
            if not _has(low, POSITION_CUES):
                bad.append(f"{element}: no relative position: {text}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("set_name", STRICT_SETS)
def test_props_are_small_and_far(set_name):
    bad = []
    for _, text in _entries(set_name, "props"):
        low = text.lower()
        if not _has(low, SIZE_CUES):
            bad.append(f"props: no size word ({'/'.join(SIZE_CUES)}): {text}")
        if not _has(low, FAR_CUES):
            bad.append(f"props: not placed far away: {text}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("set_name", STRICT_SETS)
def test_no_free_floating_size_phrase(set_name):
    bad = []
    for element in MAX_CHARS:
        for _, text in _entries(set_name, element):
            hit = [p for p in FREE_FLOATING_SIZE if p in text.lower()]
            if hit:
                bad.append(f"{element}: {hit} spawns a distant figure — size goes on the object: {text}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("set_name", STRICT_SETS)
def test_foreground_is_ground_plane(set_name):
    bad = []
    for _, text in _entries(set_name, "foreground_element"):
        low = text.lower()
        if not _has(low, GROUND_WORDS + tuple(w + "s" for w in GROUND_WORDS)):
            bad.append(f"foreground: not a ground surface: {text}")
        hit = [w for w in FLOOR_PATTERN_BAN if w in low]
        if hit:
            bad.append(f"foreground: busy floor pattern {hit} smears on Krea 2: {text}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("set_name", STRICT_SETS)
def test_no_people_words(set_name):
    bad = []
    for element in MAX_CHARS:
        for _, text in _entries(set_name, element):
            hit = PEOPLE_WORDS & _words(text)
            if hit:
                bad.append(f"{element}: {sorted(hit)}: {text}")
    assert not bad, "Krea 2 paints every person word:\n" + "\n".join(bad)


@pytest.mark.parametrize("set_name", STRICT_SETS)
def test_no_implied_writing(set_name):
    bad = []
    for element in MAX_CHARS:
        for _, text in _entries(set_name, element):
            low = text.lower()
            words = _words(low)
            hit = [w for w in WRITING_WORDS if (w in words if " " not in w else w in low)]
            if hit and not (_has(low, WRITING_OK[:-1]) or '"' in low):
                bad.append(f"{element}: {hit} without blank/quoted text: {text}")
    assert not bad, "implied writing renders as gibberish:\n" + "\n".join(bad)


@pytest.mark.parametrize("set_name", STRICT_SETS)
def test_no_parentheses_or_stray_tokens(set_name):
    bad = []
    for element in MAX_CHARS:
        for line, text in _entries(set_name, element):
            if "(" in text or ")" in text:
                bad.append(f"{element}: parentheses are prompt weights in CLIPTextEncode: {text}")
            if "#" in text:
                bad.append(f"{element}: '#' tokens are engine-only: {text}")
    assert not bad, "\n".join(bad)

@pytest.mark.parametrize("set_name", STRICT_SETS)
def test_seating_is_marked_empty(set_name):
    bad = []
    for element in ("background", "midground"):
        for _, text in _entries(set_name, element):
            low = text.lower()
            if _has(low, SEATING_WORDS) and not _has(low, EMPTY_CUES):
                bad.append(f"{element}: seating without an empty cue gets occupied: {text}")
    assert not bad, "\n".join(bad)
