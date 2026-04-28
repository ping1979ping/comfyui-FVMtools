"""Tests for nodes/jb/builder.py.

The Builder's UI is the JS row widget; the Python side just consumes the
hidden `rows` JSON field and emits raw_json + string. The optional
``text`` parameter remains as a power-user / pytest fallback.
"""

import json

from nodes.jb.builder import FVM_JB_Builder


def _build(rows="[]", output_format="loose_keys", text=""):
    return FVM_JB_Builder().build(rows=rows, output_format=output_format, text=text)


def test_node_metadata():
    assert FVM_JB_Builder.CATEGORY.startswith("FVM Tools/JB")
    assert FVM_JB_Builder.RETURN_TYPES == ("STRING", "STRING")
    assert FVM_JB_Builder.RETURN_NAMES == ("raw_json", "string")


def test_text_fallback_strict_json_in_loose_keys_out():
    raw_json, string_out = _build(
        text='{"hosiery": {"type": "stockings"}}',
        output_format="loose_keys",
    )
    parsed = json.loads(raw_json)
    assert parsed == {"hosiery": {"type": "stockings"}}
    # loose_keys: keys bare, values bare
    assert "hosiery:" in string_out
    assert '"hosiery"' not in string_out
    assert "stockings" in string_out
    assert '"stockings"' not in string_out


def test_text_fallback_loose_in_strict_out():
    raw_json, _ = _build(
        text='hosiery: {type: "stockings"}',
        output_format="pretty_json",
    )
    parsed = json.loads(raw_json)
    assert parsed == {"hosiery": {"type": "stockings"}}


def test_rows_field_takes_precedence_over_text():
    rows = json.dumps([
        {"key": "a", "value": "from_rows", "indent": 0},
    ])
    raw_json, _ = _build(
        rows=rows,
        text='{"different": "from_text"}',
        output_format="pretty_json",
    )
    parsed = json.loads(raw_json)
    assert parsed == {"a": "from_rows"}


def test_rows_with_user_hosiery_example():
    rows = json.dumps([
        {"key": "hosiery", "value": "", "indent": 0},
        {"key": "type",    "value": "sheer black stockings", "indent": 1},
        {"key": "opacity", "value": "20-30 denier", "indent": 1},
        {"key": "details", "value": "smooth texture", "indent": 1},
    ])
    raw_json, string_out = _build(rows=rows, output_format="loose_keys")
    parsed = json.loads(raw_json)
    assert parsed == {
        "hosiery": {
            "type": "sheer black stockings",
            "opacity": "20-30 denier",
            "details": "smooth texture",
        }
    }
    # loose_keys output: keys + values both bare; the literal text "sheer
    # black stockings" appears without surrounding quotes.
    assert "hosiery:" in string_out
    assert "type:" in string_out
    assert "sheer black stockings" in string_out
    assert '"sheer black stockings"' not in string_out
    assert '"hosiery"' not in string_out


def test_rows_preserve_explicit_quotes_in_value():
    """A value that contained ``\\"`` in the source JSON keeps the literal
    quote character in the loose-keys output."""
    rows = json.dumps([
        {"key": "ethnicity", "value": 'tanned european "SUPI"', "indent": 0},
    ])
    raw_json, string_out = _build(rows=rows, output_format="loose_keys")
    # Strict JSON re-escapes the literal quotes when re-encoding.
    parsed = json.loads(raw_json)
    assert parsed == {"ethnicity": 'tanned european "SUPI"'}
    # Loose-keys keeps the literal quote chars but strips JSON-syntactic ones.
    assert 'tanned european "SUPI"' in string_out
    assert "ethnicity:" in string_out


def test_invalid_rows_falls_back_to_text():
    raw_json, _ = _build(
        rows="not valid json {{{",
        text='{"a": 1}',
        output_format="pretty_json",
    )
    assert json.loads(raw_json) == {"a": 1}


def test_empty_inputs_do_not_crash():
    raw_json, _ = _build()
    assert json.loads(raw_json) == {}


def test_unparseable_text_wrapped_safely():
    """Free-text that isn't JSON or loose-keys gets wrapped under a marker key."""
    raw_json, _ = _build(text="just a plain prompt with no structure at all")
    parsed = json.loads(raw_json)
    assert "_raw_text" in parsed


def test_compact_format():
    _, string_out = _build(text='{"a": 1, "b": 2}', output_format="compact_json")
    assert "\n" not in string_out
    assert string_out == '{"a": 1, "b": 2}'


def test_input_types_no_longer_expose_text():
    """The textarea is gone — only `rows` and `output_format` are public."""
    schema = FVM_JB_Builder.INPUT_TYPES()
    assert "rows" in schema["required"]
    assert "output_format" in schema["required"]
    assert "text" not in schema["required"]
    # text remains in optional as an internal fallback (not displayed by JS).
    assert "text" in schema["optional"]
