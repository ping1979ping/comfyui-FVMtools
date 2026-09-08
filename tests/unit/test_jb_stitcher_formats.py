"""Regression matrix for FVM_JB_Stitcher × all four output formats.

Guards the bug where a bare prose string handed to the stitcher vanished
completely in ``output_format="natural"``:

    stitch(title="c1", output_format="natural",
           input_1='ein ganz normaler prosa string')  ->  ''

Cause chain: the stitcher parked non-JSON input under the synthetic key
``__inputN``, and ``core.jb.serialize.natural_phrases`` drops every key
starting with ``_`` (private / generation metadata) *together with its whole
subtree*. Fix: the stitcher now uses the plain slot name (``input_1``), so
the ``_``-guard keeps doing its job for real metadata while user content
survives every format.

Matrix: {parseable JSON | plain prose | loose-keys string}
        × {pretty_json, compact_json, loose_keys, natural}
"""

import json

import pytest

from core.jb.serialize import (
    COMPACT_JSON,
    LOOSE_KEYS,
    NATURAL,
    PRETTY_JSON,
)
from nodes.jb.stitcher import ARRAY_KEY, FVM_JB_Stitcher

ALL_FOUR = (PRETTY_JSON, COMPACT_JSON, LOOSE_KEYS, NATURAL)

# The three input kinds a STRING socket can realistically carry.
JSON_INPUT = '{"top": "grey tee", "text": "VARSITY"}'
PROSE_INPUT = 'wearing grey tee with "VARSITY" text in varsity block'
LOOSE_INPUT = 'top: "grey tee", text: "VARSITY"'

ALL_KINDS = (JSON_INPUT, PROSE_INPUT, LOOSE_INPUT)


def _stitch(value, fmt, title="c1"):
    return FVM_JB_Stitcher().stitch(title, fmt, input_1=value)


# ─── The core regression: nothing is ever silently dropped ─────────────


@pytest.mark.parametrize("fmt", ALL_FOUR)
@pytest.mark.parametrize(
    "value", ALL_KINDS, ids=["json", "prose", "loose_keys"]
)
def test_no_output_format_silently_swallows_the_payload(value, fmt):
    """Every input kind must survive in every format — never an empty string."""
    _, string_out = _stitch(value, fmt)
    assert string_out.strip(), (
        f"format={fmt!r} produced empty output for input {value!r}"
    )
    assert "grey tee" in string_out
    assert "VARSITY" in string_out


@pytest.mark.parametrize(
    "value", ALL_KINDS, ids=["json", "prose", "loose_keys"]
)
def test_raw_json_output_always_carries_the_payload(value):
    """The strict raw_json output is format-independent and must never lose
    content either (it was already correct — this pins it)."""
    for fmt in ALL_FOUR:
        raw, _ = _stitch(value, fmt)
        parsed = json.loads(raw)
        assert list(parsed) == ["c1"]
        assert "grey tee" in raw and "VARSITY" in raw


# ─── Per-format shape ──────────────────────────────────────────────────


def test_prose_input_natural_is_the_prose_itself():
    """The regression's headline case: prose in → the same prose out."""
    _, out = _stitch(PROSE_INPUT, NATURAL)
    assert out == PROSE_INPUT


def test_prose_input_lands_under_its_slot_key_not_a_private_one():
    raw, _ = _stitch(PROSE_INPUT, PRETTY_JSON)
    child = json.loads(raw)["c1"]
    assert child == {"input_1": PROSE_INPUT}
    # No dunder / underscore key — that's what made it disappear in `natural`.
    assert not any(str(k).startswith("_") for k in child)


def test_prose_input_loose_keys_uses_the_readable_slot_key():
    _, out = _stitch(PROSE_INPUT, LOOSE_KEYS)
    assert "input_1:" in out
    assert "__input" not in out


@pytest.mark.parametrize(
    "value, fmt, expected",
    [
        (JSON_INPUT, COMPACT_JSON, '{"c1": {"top": "grey tee", "text": "VARSITY"}}'),
        (LOOSE_INPUT, COMPACT_JSON, '{"c1": {"top": "grey tee", "text": "VARSITY"}}'),
        (PROSE_INPUT, COMPACT_JSON, '{"c1": {"input_1": %s}}' % json.dumps(PROSE_INPUT)),
        (JSON_INPUT, NATURAL, "grey tee, VARSITY"),
        (LOOSE_INPUT, NATURAL, "grey tee, VARSITY"),
        (PROSE_INPUT, NATURAL, PROSE_INPUT),
    ],
)
def test_exact_output_per_kind_and_format(value, fmt, expected):
    _, out = _stitch(value, fmt)
    assert out == expected


def test_loose_keys_string_input_is_parsed_not_treated_as_prose():
    """A loose-keys fragment must deep-merge as a dict, i.e. NOT end up under
    a slot key — otherwise the extractor can no longer address its fields."""
    raw, _ = _stitch(LOOSE_INPUT, LOOSE_KEYS)
    assert json.loads(raw) == {"c1": {"top": "grey tee", "text": "VARSITY"}}


# ─── JSON output must not get uglier than before the fix ──────────────


@pytest.mark.parametrize("fmt", [PRETTY_JSON, COMPACT_JSON, LOOSE_KEYS])
def test_json_formats_stay_clean_for_dict_inputs(fmt):
    """Dict inputs never gain a synthetic wrapper key of any name."""
    _, out = _stitch(JSON_INPUT, fmt)
    assert "input_1" not in out
    assert "inputs" not in out


# ─── Array inputs have the same exposure ──────────────────────────────


def test_array_input_survives_natural_too():
    """Arrays were parked under '__inputs' and hit the exact same guard."""
    _, out = FVM_JB_Stitcher().stitch(
        "tags", NATURAL, input_1='["sunlit", "windswept"]'
    )
    assert out == "sunlit, windswept"


def test_array_bucket_key_is_public():
    assert not ARRAY_KEY.startswith("_")
    raw, _ = FVM_JB_Stitcher().stitch("tags", PRETTY_JSON, input_1='["a", "b"]')
    assert json.loads(raw) == {"tags": {ARRAY_KEY: ["a", "b"]}}


def test_array_bucket_collision_does_not_drop_data():
    """If a merged fragment already used 'inputs' for a non-list, the array
    goes to its own slot key instead of crashing or being swallowed."""
    raw, _ = FVM_JB_Stitcher().stitch(
        "scene",
        PRETTY_JSON,
        input_1='{"inputs": "a plain string, not a list"}',
        input_2='["kept", "anyway"]',
    )
    parsed = json.loads(raw)["scene"]
    assert parsed["inputs"] == "a plain string, not a list"
    assert parsed["input_2"] == ["kept", "anyway"]


# ─── Mixed dict + prose: both halves reach `natural` ──────────────────


def test_mixed_dict_and_prose_both_appear_in_natural():
    _, out = FVM_JB_Stitcher().stitch(
        "character_1",
        NATURAL,
        input_1='{"face": {"eyes": "warm amber"}}',
        input_2="standing in soft window light",
    )
    assert "warm amber" in out
    assert "standing in soft window light" in out


def test_real_metadata_keys_are_still_filtered_in_natural():
    """The '_' guard and NON_PROMPT_KEYS must keep working — the fix must not
    have opened the floodgates for generation metadata."""
    _, out = FVM_JB_Stitcher().stitch(
        "character_1",
        NATURAL,
        input_1='{"seed": 12345, "set_name": "lace_v2", "_debug": "internal note",'
                ' "top": "grey tee"}',
    )
    assert out == "grey tee"
    assert "lace_v2" not in out
    assert "internal note" not in out
