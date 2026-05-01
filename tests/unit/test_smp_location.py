"""P3 — tests for the location engine + SMP location nodes."""

import os
import re

import pytest

from core.location_engine import (
    ELEMENT_LAYER,
    ELEMENT_ORDER,
    _location_lists_root,
    generate_location_records,
    get_available_location_sets,
    load_location_elements,
)


REQUIRED_SETS = ["outdoor_urban_brutalist", "outdoor_beach_mediterranean", "indoor_studio_minimal"]
REQUIRED_FILES = [
    "background.txt",
    "midground.txt",
    "foreground_element.txt",
    "architecture_detail.txt",
    "props.txt",
    "time_of_day.txt",
    "weather.txt",
]
LINE_RE = re.compile(r"^[^|]+\|[^|]+\|[^|]+\|.*$")


# ─── Data files ──────────────────────────────────────────────────────────


def test_all_three_sets_exist():
    sets = get_available_location_sets()
    for name in REQUIRED_SETS:
        assert name in sets, f"missing location set: {name}"


@pytest.mark.parametrize("set_name", REQUIRED_SETS)
def test_set_has_all_required_files(set_name):
    root = _location_lists_root()
    set_dir = os.path.join(root, set_name)
    for fname in REQUIRED_FILES:
        path = os.path.join(set_dir, fname)
        assert os.path.isfile(path), f"missing {set_name}/{fname}"


@pytest.mark.parametrize("set_name", REQUIRED_SETS)
def test_files_parse_clean(set_name):
    """Every non-comment line must be 4-pipe-separated and parse without warnings."""
    for element_id in [f.replace(".txt", "") for f in REQUIRED_FILES]:
        entries = load_location_elements(element_id, set_name)
        assert len(entries) >= 4, (
            f"{set_name}/{element_id}: only {len(entries)} entries; expected >= 4"
        )
        for e in entries:
            assert e["name"], f"{set_name}/{element_id}: empty name"
            assert 0.0 < e["probability"] <= 1.0, (
                f"{set_name}/{element_id}: probability out of range for {e['name']}"
            )
            cov_lo, cov_hi = e["coverage_range"]
            assert 0.0 <= cov_lo <= cov_hi <= 1.0, (
                f"{set_name}/{element_id}: bad coverage_range {e['coverage_range']}"
            )


# ─── Engine ──────────────────────────────────────────────────────────────


def test_seed_determinism():
    a = generate_location_records(seed=42, location_set="outdoor_urban_brutalist")
    b = generate_location_records(seed=42, location_set="outdoor_urban_brutalist")
    assert a == b


def test_different_seeds_diverge_within_one_set():
    seen = set()
    for s in range(0, 30):
        rec = generate_location_records(seed=s, location_set="outdoor_urban_brutalist")
        # Tuple of selected names per element → fingerprint of the draw
        fp = tuple(sorted((k, e["name"]) for k, e in rec["elements"].items()))
        seen.add(fp)
    # Across 30 seeds we expect at least a few distinct outcomes; otherwise the
    # weighted RNG is broken.
    assert len(seen) >= 3, f"only {len(seen)} unique outcomes across 30 seeds"


@pytest.mark.parametrize("set_name", REQUIRED_SETS)
def test_default_enables_produce_fragments(set_name):
    rec = generate_location_records(seed=11, location_set=set_name)
    assert rec["elements"], f"{set_name}: no elements emitted with defaults"
    for elem_id, e in rec["elements"].items():
        assert e["prompt_fragment"], f"{set_name}/{elem_id}: empty fragment"
        assert e["layer"] == ELEMENT_LAYER[elem_id]


def test_background_fragment_carries_ambient_token():
    rec = generate_location_records(
        seed=5, location_set="outdoor_urban_brutalist",
        element_enables={k: (k == "background") for k in ELEMENT_ORDER},
    )
    assert "background" in rec["elements"]
    assert "#ambient_light#" in rec["elements"]["background"]["prompt_fragment"]


def test_foreground_fragment_carries_shadow_token():
    rec = generate_location_records(
        seed=5, location_set="outdoor_urban_brutalist",
        element_enables={k: (k == "foreground_element") for k in ELEMENT_ORDER},
    )
    assert "foreground_element" in rec["elements"]
    assert "#shadow_tone#" in rec["elements"]["foreground_element"]["prompt_fragment"]


def test_atmosphere_fragments_have_no_tokens():
    rec = generate_location_records(
        seed=5, location_set="outdoor_urban_brutalist",
        element_enables={k: (k in {"time_of_day", "weather"}) for k in ELEMENT_ORDER},
    )
    for elem_id in ("time_of_day", "weather"):
        if elem_id in rec["elements"]:
            assert "#" not in rec["elements"][elem_id]["prompt_fragment"]


def test_engine_unknown_set_returns_empty():
    rec = generate_location_records(seed=1, location_set="__nonexistent__")
    assert rec["elements"] == {}


# ─── Location nodes ──────────────────────────────────────────────────────


def _gen_node():
    from nodes.smp.location_generator import FVM_SMP_LocationGenerator
    return FVM_SMP_LocationGenerator()


def _common_args():
    return dict(
        location_set="outdoor_urban_brutalist", seed=42,
        enable_background=True, enable_midground=False,
        enable_architecture_detail=False, enable_props=False,
        enable_foreground_element=True,
        enable_time_of_day=True, enable_weather=True,
        color_tone="",
    )


def test_location_generator_returns_two_tuple():
    raw, summary = _gen_node().generate(**_common_args())
    assert isinstance(raw, dict)
    assert isinstance(summary, str)


def test_location_generator_seed_byte_identical():
    a, _ = _gen_node().generate(**_common_args())
    b, _ = _gen_node().generate(**_common_args())
    assert a == b


def test_location_generator_metadata():
    from nodes.smp.location_generator import FVM_SMP_LocationGenerator
    assert FVM_SMP_LocationGenerator.CATEGORY.startswith("FVM Tools/SMP")
    assert FVM_SMP_LocationGenerator.RETURN_TYPES == ("LOCATION_DICT_RAW", "STRING")


# ─── Combiner ────────────────────────────────────────────────────────────


def _palette_with_atmosphere():
    from nodes.smp.color_generator import FVM_SMP_ColorGenerator
    return FVM_SMP_ColorGenerator().generate(
        seed=3, num_colors=5, harmony_type="auto", style_preset="general",
        vibrancy=0.5, contrast=0.5, warmth=0.5,
    )[0]


def test_combiner_resolves_atmosphere_tokens():
    from nodes.smp.location_combiner import FVM_SMP_LocationCombiner
    raw, _ = _gen_node().generate(**_common_args())
    palette = _palette_with_atmosphere()
    resolved, _ = FVM_SMP_LocationCombiner().combine(raw, palette)
    for elem_id, e in resolved["elements"].items():
        assert "#ambient_light#" not in e["prompt_fragment"], (
            f"{elem_id} still has ambient token: {e['prompt_fragment']!r}"
        )
        assert "#shadow_tone#" not in e["prompt_fragment"], (
            f"{elem_id} still has shadow token: {e['prompt_fragment']!r}"
        )
        assert "#" not in e["prompt_fragment"], (
            f"{elem_id} has unresolved token: {e['prompt_fragment']!r}"
        )


def test_combiner_does_not_mutate_input():
    from nodes.smp.location_combiner import FVM_SMP_LocationCombiner
    raw, _ = _gen_node().generate(**_common_args())
    palette = _palette_with_atmosphere()
    FVM_SMP_LocationCombiner().combine(raw, palette)
    for elem_id, e in raw["elements"].items():
        # Background and foreground have tokens; atmosphere doesn't.
        if elem_id == "background":
            assert "#ambient_light#" in e["prompt_fragment"]
        if elem_id == "foreground_element":
            assert "#shadow_tone#" in e["prompt_fragment"]


def test_combiner_inherits_palette_tone():
    from nodes.smp.location_combiner import FVM_SMP_LocationCombiner
    raw, _ = _gen_node().generate(**_common_args())
    raw["color_tone"] = None
    palette = _palette_with_atmosphere()
    resolved, _ = FVM_SMP_LocationCombiner().combine(raw, palette)
    assert resolved["color_tone"] == palette["color_tone"]


def test_combiner_metadata():
    from nodes.smp.location_combiner import FVM_SMP_LocationCombiner
    assert FVM_SMP_LocationCombiner.CATEGORY.startswith("FVM Tools/SMP")
    assert FVM_SMP_LocationCombiner.RETURN_TYPES == ("LOCATION_DICT", "STRING")


# ─── Cross-set determinism (10 runs same seed) ───────────────────────────


@pytest.mark.parametrize("set_name", REQUIRED_SETS)
def test_ten_runs_same_seed_byte_identical(set_name):
    seen = set()
    for _ in range(10):
        rec = generate_location_records(seed=2026, location_set=set_name)
        fp = tuple(sorted(
            (k, e["name"], round(e["coverage"], 4), e["prompt_fragment"])
            for k, e in rec["elements"].items()
        ))
        seen.add(fp)
    assert len(seen) == 1, f"{set_name}: non-deterministic across 10 runs"
