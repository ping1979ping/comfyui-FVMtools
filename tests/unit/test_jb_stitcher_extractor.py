"""P2 — tests for FVM_JB_Stitcher and FVM_JB_Extractor."""

import json

import pytest

from nodes.jb.stitcher import FVM_JB_Stitcher, MAX_INPUTS
from nodes.jb.extractor import FVM_JB_Extractor


# ─── Stitcher ─────────────────────────────────────────────────────────


def _stitch(title, output_format="loose_keys", **inputs):
    return FVM_JB_Stitcher().stitch(title, output_format, **inputs)


def test_stitcher_node_metadata():
    assert FVM_JB_Stitcher.CATEGORY.startswith("FVM Tools/JB")
    assert FVM_JB_Stitcher.RETURN_TYPES == ("STRING", "STRING")
    assert FVM_JB_Stitcher.RETURN_NAMES == ("raw_json", "string")


def test_stitcher_two_objects_merge_under_title():
    raw, _ = _stitch(
        "character_1",
        input_1='{"hosiery": {"type": "stockings"}}',
        input_2='{"face": {"eyes": "blue"}}',
    )
    parsed = json.loads(raw)
    assert parsed == {
        "character_1": {
            "hosiery": {"type": "stockings"},
            "face": {"eyes": "blue"},
        }
    }


def test_stitcher_deep_merge_adds_new_subfields():
    """User's locked semantics: same-level scalar leaves last-wins, but new
    sub-fields underneath get added recursively."""
    raw, _ = _stitch(
        "character_1",
        input_1='{"hosiery": {"type": "stockings", "opacity": "sheer"}}',
        input_2='{"hosiery": {"opacity": "matte", "details": "black"}}',
    )
    parsed = json.loads(raw)
    # opacity is a scalar collision → last input wins
    # details is a new sub-field → added
    # type was only in input_1 → preserved
    assert parsed == {
        "character_1": {
            "hosiery": {
                "type": "stockings",
                "opacity": "matte",
                "details": "black",
            }
        }
    }


def test_stitcher_skips_empty_slots():
    raw, _ = _stitch(
        "outfit",
        input_1='{"top": "blazer"}',
        input_2="",
        input_3=None,
        input_4='{"bottom": "skirt"}',
    )
    parsed = json.loads(raw)
    assert parsed == {"outfit": {"top": "blazer", "bottom": "skirt"}}


def test_stitcher_default_title():
    raw, _ = _stitch("", input_1='{"a": 1}')
    parsed = json.loads(raw)
    assert "untitled" in parsed


def test_stitcher_array_input_appends():
    raw, _ = _stitch(
        "tags",
        input_1='["a", "b"]',
        input_2='["c"]',
    )
    parsed = json.loads(raw)
    assert parsed == {"tags": {"__inputs": ["a", "b", "c"]}}


def test_stitcher_bare_string_input_synthetic_key():
    raw, _ = _stitch(
        "scene",
        input_1='{"location": "studio"}',
        input_2="just a free-text fragment",
    )
    parsed = json.loads(raw)
    assert parsed["scene"]["location"] == "studio"
    assert parsed["scene"]["__input2"] == "just a free-text fragment"


def test_stitcher_loose_keys_output():
    """loose_keys: keys bare, values bare, no JSON-syntactic quotes anywhere."""
    _, string_out = _stitch(
        "outfit",
        output_format="loose_keys",
        input_1='{"top": "blazer", "bottom": "skirt"}',
    )
    assert "outfit:" in string_out
    assert "top:" in string_out
    assert "blazer" in string_out
    assert "skirt" in string_out
    assert '"blazer"' not in string_out
    assert '"outfit"' not in string_out


def test_stitcher_preserves_explicit_quote_chars():
    """`\\"`-escapes in inputs survive as literal `"` in loose_keys output."""
    _, string_out = _stitch(
        "outfit",
        output_format="loose_keys",
        input_1='{"label": "tanned european \\"SUPI\\""}',
    )
    assert 'tanned european "SUPI"' in string_out


def test_stitcher_no_inputs_emits_empty_object():
    raw, _ = _stitch("empty")
    parsed = json.loads(raw)
    assert parsed == {"empty": {}}


def test_stitcher_max_inputs_constant():
    assert MAX_INPUTS >= 8


def test_stitcher_input_types_advertise_many_optional_slots():
    schema = FVM_JB_Stitcher.INPUT_TYPES()
    # Plenty of optional slots
    optional_slots = [k for k in schema["optional"] if k.startswith("input_")]
    assert len(optional_slots) == MAX_INPUTS


# ─── Extractor ────────────────────────────────────────────────────────


def _extract(json_input, category, output_format="loose_keys"):
    return FVM_JB_Extractor().extract(json_input, category, output_format)


def test_extractor_node_metadata():
    assert FVM_JB_Extractor.CATEGORY.startswith("FVM Tools/JB")
    assert FVM_JB_Extractor.RETURN_TYPES == ("STRING", "STRING", "BOOLEAN")
    assert FVM_JB_Extractor.RETURN_NAMES == ("raw_json", "string", "found")


def test_extractor_top_level_key_wraps():
    """Single-key category: result is wrapped under that key name."""
    src = '{"character_1": {"hosiery": {"type": "stockings"}}}'
    raw, _, found = _extract(src, "character_1")
    assert found is True
    assert json.loads(raw) == {"character_1": {"hosiery": {"type": "stockings"}}}


def test_extractor_recursive_search_user_face_example():
    """The user's specified behavior: search 'face' anywhere in the tree
    and return the wrapped subtree."""
    src = """
    {
      "character_1": {
        "face": {
          "age": "mid twenties",
          "ethnicity": "tanned european",
          "skin": "sun-kissed glow"
        }
      }
    }
    """
    raw, _, found = _extract(src, "face")
    assert found is True
    parsed = json.loads(raw)
    assert parsed == {
        "face": {
            "age": "mid twenties",
            "ethnicity": "tanned european",
            "skin": "sun-kissed glow",
        }
    }


def test_extractor_recursive_finds_first_match_at_any_depth():
    src = '{"a": {"b": {"target": {"x": 1}}}}'
    raw, _, found = _extract(src, "target")
    assert found is True
    assert json.loads(raw) == {"target": {"x": 1}}


def test_extractor_dot_path_nested_wraps_under_last_segment():
    src = '{"character_1": {"hosiery": {"type": "stockings", "opacity": "sheer"}}}'
    raw, _, found = _extract(src, "character_1.hosiery")
    assert found is True
    # Wrapped under the LAST path segment.
    assert json.loads(raw) == {"hosiery": {"type": "stockings", "opacity": "sheer"}}


def test_extractor_dot_path_to_leaf_wraps_too():
    src = '{"character_1": {"hosiery": {"type": "stockings"}}}'
    raw, _, found = _extract(src, "character_1.hosiery.type")
    assert found is True
    assert json.loads(raw) == {"type": "stockings"}


def test_extractor_missing_key_returns_empty():
    src = '{"a": 1}'
    raw, string_out, found = _extract(src, "nonexistent")
    assert found is False
    assert raw == ""
    assert string_out == ""


def test_extractor_empty_category_returns_whole_doc_unwrapped():
    src = '{"a": 1, "b": 2}'
    raw, _, found = _extract(src, "")
    assert found is True
    assert json.loads(raw) == {"a": 1, "b": 2}


def test_extractor_invalid_json_input():
    raw, _, found = _extract("not even close to json", "anything")
    assert found is False


def test_extractor_loose_keys_input_works():
    src = 'character_1: {hosiery: {type: "stockings"}}'
    raw, _, found = _extract(src, "character_1.hosiery")
    assert found is True
    assert json.loads(raw) == {"hosiery": {"type": "stockings"}}


def test_extractor_string_output_strips_value_quotes():
    """Loose-keys string output: keys bare, values bare, no JSON quotes."""
    src = '{"face": {"eyes": "warm amber", "expression": "intense focus"}}'
    _, string_out, _ = _extract(src, "face")
    assert "face:" in string_out
    assert "warm amber" in string_out
    assert '"warm amber"' not in string_out
    assert "intense focus" in string_out
    assert '"face"' not in string_out


def test_extractor_string_output_preserves_explicit_quotes():
    """Literal `"` chars from `\\"` survive into the loose_keys string output."""
    src = '{"face": {"ethnicity": "tanned european \\"SUPI\\""}}'
    _, string_out, _ = _extract(src, "face")
    assert 'tanned european "SUPI"' in string_out


# ─── Stitcher → Extractor end-to-end ──────────────────────────────────


def test_stitcher_then_extractor_roundtrip():
    stitched_raw, _ = FVM_JB_Stitcher().stitch(
        "character_1", "loose_keys",
        input_1='{"hosiery": {"type": "stockings"}}',
        input_2='{"face": {"eyes": "blue"}}',
    )
    # Recursive search by single key — extractor wraps the result.
    raw, _, found = FVM_JB_Extractor().extract(
        stitched_raw, "hosiery", "loose_keys"
    )
    assert found is True
    assert json.loads(raw) == {"hosiery": {"type": "stockings"}}

    # Same flow, dot-path — wraps under the last segment.
    raw2, _, found2 = FVM_JB_Extractor().extract(
        stitched_raw, "character_1.face", "loose_keys"
    )
    assert found2 is True
    assert json.loads(raw2) == {"face": {"eyes": "blue"}}
