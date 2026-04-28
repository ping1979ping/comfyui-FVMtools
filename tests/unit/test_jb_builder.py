"""P1 — tests for nodes/jb/builder.py."""

import json

from nodes.jb.builder import FVM_JB_Builder


def test_node_metadata():
    assert FVM_JB_Builder.CATEGORY.startswith("FVM Tools/JB")
    assert FVM_JB_Builder.RETURN_TYPES == ("STRING", "STRING")
    assert FVM_JB_Builder.RETURN_NAMES == ("raw_json", "string")


def test_textarea_strict_json_in_loose_keys_out():
    raw_json, string_out = FVM_JB_Builder().build(
        text='{"hosiery": {"type": "stockings"}}',
        output_format="loose_keys",
    )
    parsed = json.loads(raw_json)
    assert parsed == {"hosiery": {"type": "stockings"}}
    assert "hosiery:" in string_out
    assert '"hosiery"' not in string_out
    assert '"stockings"' in string_out


def test_textarea_loose_keys_in_strict_out():
    raw_json, string_out = FVM_JB_Builder().build(
        text="hosiery: {type: \"stockings\"}",
        output_format="pretty_json",
    )
    parsed = json.loads(raw_json)
    assert parsed == {"hosiery": {"type": "stockings"}}


def test_rows_field_takes_precedence():
    """When rows is populated, it overrides whatever's in the textarea."""
    rows = json.dumps([
        {"key": "a", "value": "from_rows", "indent": 0},
    ])
    raw_json, _ = FVM_JB_Builder().build(
        text='{"different": "from_textarea"}',
        output_format="pretty_json",
        rows=rows,
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
    raw_json, string_out = FVM_JB_Builder().build(
        text="",
        output_format="loose_keys",
        rows=rows,
    )
    parsed = json.loads(raw_json)
    assert parsed == {
        "hosiery": {
            "type": "sheer black stockings",
            "opacity": "20-30 denier",
            "details": "smooth texture",
        }
    }
    # loose_keys output: keys unquoted, string values keep quotes
    assert "hosiery:" in string_out
    assert "type:" in string_out
    assert '"sheer black stockings"' in string_out
    assert '"hosiery"' not in string_out  # no key quotes


def test_invalid_rows_falls_back_to_text():
    raw_json, _ = FVM_JB_Builder().build(
        text='{"a": 1}',
        output_format="pretty_json",
        rows="not valid json {{{",
    )
    assert json.loads(raw_json) == {"a": 1}


def test_empty_text_does_not_crash():
    raw_json, string_out = FVM_JB_Builder().build(
        text="",
        output_format="pretty_json",
    )
    assert json.loads(raw_json) == {}


def test_unparseable_text_wrapped_safely():
    """Free-text that isn't JSON or loose-keys gets wrapped."""
    raw_json, _ = FVM_JB_Builder().build(
        text="just a plain prompt with no structure at all",
        output_format="pretty_json",
    )
    parsed = json.loads(raw_json)
    assert "_raw_text" in parsed


def test_compact_format():
    raw_json, string_out = FVM_JB_Builder().build(
        text='{"a": 1, "b": 2}',
        output_format="compact_json",
    )
    assert "\n" not in string_out  # compact = no newlines
    assert string_out == '{"a": 1, "b": 2}'


def test_input_types_advertise_jb_category():
    schema = FVM_JB_Builder.INPUT_TYPES()
    assert "required" in schema
    assert "text" in schema["required"]
    assert "output_format" in schema["required"]
    assert "rows" in schema["optional"]
