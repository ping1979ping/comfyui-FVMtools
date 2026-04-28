"""P1 — tests for core/jb/serialize.py."""

import json

import pytest

from core.jb.serialize import (
    COMPACT_JSON,
    LOOSE_KEYS,
    PRETTY_JSON,
    _coerce_leaf,
    dict_to_rows,
    emit,
    emit_loose_keys,
    emit_strict_json,
    parse_input,
    rows_to_dict,
)


# ─── Strict JSON emission ──────────────────────────────────────────────


def test_emit_strict_json_pretty():
    out = emit_strict_json({"a": 1, "b": "x"}, indent=2)
    assert '"a": 1' in out
    assert '"b": "x"' in out


def test_emit_strict_json_compact():
    out = emit_strict_json({"a": 1, "b": "x"}, indent=None)
    assert out == '{"a": 1, "b": "x"}'


# ─── Loose-keys emission ───────────────────────────────────────────────


def test_emit_loose_keys_simple_object():
    """Keys: bareword without quotes. Values: raw, no surrounding quotes."""
    out = emit_loose_keys({"hosiery": {"type": "stockings"}})
    assert '"hosiery"' not in out
    assert '"stockings"' not in out
    assert "hosiery:" in out
    assert "type:" in out
    assert "stockings" in out


def test_emit_loose_keys_handles_nesting():
    obj = {
        "hosiery": {
            "type": "sheer black stockings",
            "opacity": "semi-sheer",
        }
    }
    out = emit_loose_keys(obj)
    assert "hosiery:" in out
    assert "type:" in out
    assert "opacity:" in out
    assert "sheer black stockings" in out
    assert '"sheer black stockings"' not in out


def test_emit_loose_keys_array():
    out = emit_loose_keys({"tags": ["a", "b", "c"]})
    assert "tags:" in out
    # Values inside an array are bare strings now
    assert '"a"' not in out and '"b"' not in out and '"c"' not in out
    assert "a" in out and "b" in out and "c" in out
    assert "[" in out and "]" in out


def test_emit_loose_keys_quotes_keys_with_special_chars():
    """Keys with spaces / punctuation are bareword-unsafe → keep quotes."""
    out = emit_loose_keys({"weird key!": "value"})
    assert '"weird key!"' in out


def test_emit_loose_keys_preserves_explicit_quotes_in_values():
    """``\\"`` in the JSON source survives parse → emit as a literal "."""
    src = {"ethnicity": 'tanned european "SUPI"'}
    out = emit_loose_keys(src)
    # The value is bare (no surrounding quotes) BUT the inner quotes from
    # the source's \" escapes are preserved literally.
    assert "ethnicity:" in out
    assert 'tanned european "SUPI"' in out
    # No JSON-syntactic surrounding quotes around the value
    assert '"tanned european' not in out
    assert 'SUPI""' not in out


def test_emit_loose_keys_handles_scalars():
    assert emit_loose_keys(None) == "null"
    assert emit_loose_keys(True) == "true"
    assert emit_loose_keys(False) == "false"
    assert emit_loose_keys(42) == "42"
    # Strings: raw, no quotes
    assert emit_loose_keys("hello") == "hello"


def test_emit_loose_keys_empty_collections():
    assert emit_loose_keys({}) == "{}"
    assert emit_loose_keys([]) == "[]"


# ─── emit() dispatcher ─────────────────────────────────────────────────


def test_emit_dispatch():
    obj = {"a": 1}
    assert "  " in emit(obj, PRETTY_JSON)            # indent-2
    assert emit(obj, COMPACT_JSON) == '{"a": 1}'
    assert "a:" in emit(obj, LOOSE_KEYS)


# ─── parse_input ───────────────────────────────────────────────────────


def test_parse_input_strict_json():
    assert parse_input('{"a": 1}') == {"a": 1}
    assert parse_input('[1,2,3]') == [1, 2, 3]


def test_parse_input_empty():
    assert parse_input("") == {}
    assert parse_input("   ") == {}


def test_parse_input_loose_keys_with_quoted_values():
    """Strict JSON in / loose-keys-with-quoted-values still parses fine."""
    src = '{"hosiery": {"type": "stockings", "opacity": "20-30 denier"}}'
    parsed = parse_input(src)
    assert parsed == {"hosiery": {"type": "stockings", "opacity": "20-30 denier"}}
    # Same input but with bareword keys (still quoted values) parses too.
    near_loose = 'hosiery: {type: "stockings", opacity: "20-30 denier"}'
    assert parse_input(near_loose) == parsed


def test_parse_input_unparseable_returns_string():
    # Pure free-text shouldn't crash; returns the original string
    out = parse_input("just a plain prompt with no structure")
    assert isinstance(out, str)


# ─── _coerce_leaf ──────────────────────────────────────────────────────


def test_coerce_leaf_strings():
    assert _coerce_leaf("hello world") == "hello world"
    assert _coerce_leaf("") == ""
    assert _coerce_leaf(None) == ""


def test_coerce_leaf_literals():
    assert _coerce_leaf("true") is True
    assert _coerce_leaf("false") is False
    assert _coerce_leaf("null") is None


def test_coerce_leaf_numbers():
    assert _coerce_leaf("42") == 42
    assert _coerce_leaf("3.14") == 3.14
    assert _coerce_leaf("-1") == -1


def test_coerce_leaf_keeps_quoted_strings_as_text():
    """A bare quoted word in a value cell should stay as a string with quotes,
    not be unwrapped into the inner string. (Users who explicitly type quotes
    expect them in the output.)"""
    assert _coerce_leaf('"hello"') == '"hello"'


def test_coerce_leaf_array_value():
    assert _coerce_leaf("[1,2,3]") == [1, 2, 3]


# ─── rows_to_dict ──────────────────────────────────────────────────────


def test_rows_to_dict_user_hosiery_example():
    """The user's worked example from the brief."""
    rows = [
        {"key": "hosiery", "value": "", "indent": 0},
        {"key": "type",    "value": "sheer black stockings", "indent": 1},
        {"key": "opacity", "value": "semi-sheer (20-30 denier)", "indent": 1},
        {"key": "details", "value": "smooth texture", "indent": 1},
    ]
    out = rows_to_dict(rows)
    assert out == {
        "hosiery": {
            "type": "sheer black stockings",
            "opacity": "semi-sheer (20-30 denier)",
            "details": "smooth texture",
        }
    }


def test_rows_to_dict_flat():
    rows = [
        {"key": "a", "value": "1", "indent": 0},
        {"key": "b", "value": "two", "indent": 0},
    ]
    assert rows_to_dict(rows) == {"a": 1, "b": "two"}


def test_rows_to_dict_deep_nesting():
    rows = [
        {"key": "character", "value": "", "indent": 0},
        {"key": "face",      "value": "", "indent": 1},
        {"key": "eyes",      "value": "blue", "indent": 2},
        {"key": "lips",      "value": "full", "indent": 2},
        {"key": "outfit",    "value": "", "indent": 1},
        {"key": "top",       "value": "blazer", "indent": 2},
    ]
    out = rows_to_dict(rows)
    assert out == {
        "character": {
            "face": {"eyes": "blue", "lips": "full"},
            "outfit": {"top": "blazer"},
        }
    }


def test_rows_to_dict_skip_empty_keys():
    rows = [
        {"key": "", "value": "ignored", "indent": 0},
        {"key": "real", "value": "kept", "indent": 0},
    ]
    assert rows_to_dict(rows) == {"real": "kept"}


def test_rows_to_dict_empty_list():
    assert rows_to_dict([]) == {}


def test_rows_to_dict_branch_with_no_children_becomes_empty_string():
    """A row with empty value AND no nested children is a leaf, not a branch."""
    rows = [
        {"key": "a", "value": "", "indent": 0},
        {"key": "b", "value": "1", "indent": 0},
    ]
    assert rows_to_dict(rows) == {"a": "", "b": 1}


# ─── dict_to_rows ──────────────────────────────────────────────────────


def test_dict_to_rows_roundtrip():
    src = {
        "hosiery": {
            "type": "stockings",
            "opacity": "sheer",
        }
    }
    rows = dict_to_rows(src)
    # Round-trip back via rows_to_dict
    assert rows_to_dict(rows) == src


def test_dict_to_rows_indent_levels():
    rows = dict_to_rows({"a": {"b": {"c": "leaf"}}})
    indents = [r["indent"] for r in rows]
    assert indents == [0, 1, 2]
    assert rows[0]["key"] == "a" and rows[0]["value"] == ""
    assert rows[2]["key"] == "c" and rows[2]["value"] == "leaf"


def test_dict_to_rows_array_value_serialized():
    rows = dict_to_rows({"tags": ["a", "b"]})
    assert len(rows) == 1
    assert rows[0]["key"] == "tags"
    # Array values are JSON-encoded into the value cell
    assert json.loads(rows[0]["value"]) == ["a", "b"]


# ─── End-to-end roundtrip ──────────────────────────────────────────────


def test_full_roundtrip_strict_json():
    src = {"a": 1, "b": {"c": [1, 2, 3]}}
    s = emit(src, PRETTY_JSON)
    assert parse_input(s) == src


def test_loose_keys_is_one_way():
    """loose_keys is a one-way emit format for SD/CLIP encoders. It strips
    JSON-syntactic quotes around values so it is NOT round-trippable
    through parse_input — that's by design."""
    src = {"hosiery": {"type": "sheer black stockings", "opacity": "20-30 denier"}}
    out = emit(src, LOOSE_KEYS)
    assert "sheer black stockings" in out
    assert '"sheer black stockings"' not in out
