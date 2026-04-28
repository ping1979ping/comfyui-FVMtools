"""Tests for nodes/jb/builder.py.

The Builder's UI is the JS row widget; the Python side just consumes the
hidden ``rows`` JSON field and emits ``raw_json`` + ``string``.
"""

import json

from nodes.jb.builder import FVM_JB_Builder


def _build(rows="[]", output_format="loose_keys"):
    return FVM_JB_Builder().build(rows=rows, output_format=output_format)


def _rows(*items):
    return json.dumps([
        {"key": k, "value": v, "indent": ind}
        for k, v, ind in items
    ])


def test_node_metadata():
    assert FVM_JB_Builder.CATEGORY.startswith("FVM Tools/JB")
    assert FVM_JB_Builder.RETURN_TYPES == ("STRING", "STRING")
    assert FVM_JB_Builder.RETURN_NAMES == ("raw_json", "string")


def test_input_types_only_rows_and_format():
    schema = FVM_JB_Builder.INPUT_TYPES()
    assert set(schema["required"].keys()) == {"rows", "output_format"}
    # No optional fallbacks any more — the JS row widget is the only UI.
    assert "optional" not in schema or not schema["optional"]


def test_rows_with_user_hosiery_example():
    rows = _rows(
        ("hosiery", "", 0),
        ("type", "sheer black stockings", 1),
        ("opacity", "20-30 denier", 1),
        ("details", "smooth texture", 1),
    )
    raw_json, string_out = _build(rows=rows, output_format="loose_keys")
    assert json.loads(raw_json) == {
        "hosiery": {
            "type": "sheer black stockings",
            "opacity": "20-30 denier",
            "details": "smooth texture",
        }
    }
    # loose_keys: keys + values bare, no JSON-syntactic quotes anywhere.
    assert "hosiery:" in string_out
    assert "type:" in string_out
    assert "sheer black stockings" in string_out
    assert '"sheer black stockings"' not in string_out
    assert '"hosiery"' not in string_out


def test_rows_preserve_explicit_quotes_in_value():
    """A literal `"` in a row value (e.g. from a `\\"` escape in source JSON)
    survives into the loose-keys output."""
    rows = _rows(("ethnicity", 'tanned european "SUPI"', 0))
    raw_json, string_out = _build(rows=rows, output_format="loose_keys")
    assert json.loads(raw_json) == {"ethnicity": 'tanned european "SUPI"'}
    assert 'tanned european "SUPI"' in string_out
    assert "ethnicity:" in string_out


def test_loose_keys_output_has_no_extraneous_quotes():
    """The user's rule: loose_keys string output contains NO quotation
    marks except those that came from explicit ``\\"`` escapes."""
    rows = _rows(
        ("face", "", 0),
        ("eyes", "warm amber, focused", 1),
        ("expression", "intense focus", 1),
    )
    _, string_out = _build(rows=rows, output_format="loose_keys")
    # No `"` chars at all because none of the source values contained any.
    assert '"' not in string_out


def test_empty_rows_emits_empty_object():
    raw_json, string_out = _build()
    assert json.loads(raw_json) == {}
    assert string_out == "{}"


def test_invalid_rows_falls_back_to_empty():
    raw_json, _ = _build(rows="not valid json {{{")
    assert json.loads(raw_json) == {}


def test_compact_format():
    rows = _rows(("a", "1", 0), ("b", "two", 0))
    _, string_out = _build(rows=rows, output_format="compact_json")
    assert "\n" not in string_out
    assert string_out == '{"a": 1, "b": "two"}'


def test_pretty_json_round_trips():
    rows = _rows(("a", "", 0), ("b", "leaf", 1))
    raw_json, string_out = _build(rows=rows, output_format="pretty_json")
    assert json.loads(raw_json) == {"a": {"b": "leaf"}}
    assert json.loads(string_out) == {"a": {"b": "leaf"}}
