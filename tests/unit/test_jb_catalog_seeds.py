"""P5 — verify the seeded prompt_catalog/ entries load and round-trip cleanly.

These tests exercise the actual on-disk catalog — they're essentially
contract tests for the seed content. If a contributor adds a new snippet
that doesn't parse, this catches it.
"""

import pytest

from core.jb.catalog import list_categories, list_entries, read_entry
from core.jb.serialize import dict_to_rows, rows_to_dict


def test_seed_categories_are_present():
    cats = list_categories()
    for required in ("faces", "garments", "locations", "scenes", "props"):
        assert required in cats, f"missing category: {required}"


@pytest.mark.parametrize(
    "category,name",
    [
        ("faces",     "east_asian_young_woman"),
        ("faces",     "editorial_man_30s"),
        ("faces",     "athletic_young_woman"),
        ("garments",  "hosiery_sheer_seamed"),
        ("garments",  "tailored_blazer_burgundy"),
        ("garments",  "denim_jacket_oversized"),
        ("locations", "brutalist_concrete_exterior"),
        ("locations", "beach_mediterranean_morning"),
        ("scenes",    "editorial_fashion_lookbook"),
        ("scenes",    "studio_high_key_portrait"),
        ("props",     "minimalist_studio_pedestal"),
    ],
)
def test_seed_entry_loads(category, name):
    """Each seeded snippet must read as a non-empty dict."""
    entry = read_entry(category, name)
    assert isinstance(entry, dict), f"{category}/{name} did not parse as a dict"
    assert entry, f"{category}/{name} parsed as empty dict"


def test_user_hosiery_example_verbatim():
    """The user-supplied hosiery example must be present byte-equivalent."""
    entry = read_entry("garments", "hosiery_sheer_seamed")
    assert "hosiery" in entry
    h = entry["hosiery"]
    assert "sheer black stockings" in h["type"]
    assert "20-30 denier" in h["opacity"]
    assert "smooth texture" in h["details"]


@pytest.mark.parametrize(
    "category,name,expected_root_keys",
    [
        ("faces",     "east_asian_young_woman", {"face"}),
        ("faces",     "editorial_man_30s",      {"face"}),
        ("garments",  "hosiery_sheer_seamed",   {"hosiery"}),
        ("garments",  "tailored_blazer_burgundy", {"blazer"}),
        ("locations", "brutalist_concrete_exterior", {"location"}),
        ("scenes",    "editorial_fashion_lookbook", {"scene"}),
        ("props",     "minimalist_studio_pedestal", {"props"}),
    ],
)
def test_seed_roots_use_expected_namespace(category, name, expected_root_keys):
    """Each snippet wraps its content under a sensible top-level key so
    multiple snippets can stitch together without colliding."""
    entry = read_entry(category, name)
    assert set(entry.keys()) == expected_root_keys


def test_user_hosiery_round_trips_through_rows():
    """The user's example: 4 rows, indent 0+1+1+1, then back to dict."""
    src = read_entry("garments", "hosiery_sheer_seamed")
    rows = dict_to_rows(src)
    assert len(rows) == 4
    assert [r["indent"] for r in rows] == [0, 1, 1, 1]
    assert rows[0]["key"] == "hosiery" and rows[0]["value"] == ""
    assert rows_to_dict(rows) == src


def test_all_seeds_round_trip_through_rows():
    """Every seed snippet must round-trip through dict_to_rows + rows_to_dict
    so the JB Builder widget can paste them as rows and emit the same JSON."""
    for cat in list_categories():
        for name in list_entries(cat):
            src = read_entry(cat, name)
            rows = dict_to_rows(src)
            back = rows_to_dict(rows)
            assert back == src, f"{cat}/{name} failed round-trip"


def test_minimum_seed_count_per_category():
    counts = {c: len(list_entries(c)) for c in list_categories()}
    # We deliberately seed at least one entry per category at this stage.
    for cat, n in counts.items():
        assert n >= 1, f"category {cat} is empty"
