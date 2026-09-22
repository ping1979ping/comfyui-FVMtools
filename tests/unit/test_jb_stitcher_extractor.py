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
    # Public key on purpose: '__inputs' was dropped by the natural / sentences
    # formats' underscore guard (see test_jb_stitcher_formats.py).
    assert parsed == {"tags": {"inputs": ["a", "b", "c"]}}


def test_stitcher_bare_string_input_synthetic_key():
    """Bare strings land under their slot name — an ordinary key, NOT a
    private '_'-prefixed one, so ``natural`` output keeps them."""
    raw, _ = _stitch(
        "scene",
        input_1='{"location": "studio"}',
        input_2="just a free-text fragment",
    )
    parsed = json.loads(raw)
    assert parsed["scene"]["location"] == "studio"
    assert parsed["scene"]["input_2"] == "just a free-text fragment"


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


def test_stitcher_preserves_quote_chars_in_loose_keys_output():
    """Quotes the user typed INSIDE a value are content and survive.

    Same deliberate rule as ``test_emit_loose_keys_preserves_quote_chars_inside_values``
    (commit d2dda5b, 2026-05-02, reversing cb4d9f0): only the structural
    JSON quotes around keys and values are dropped.
    """
    _, string_out = _stitch(
        "outfit",
        output_format="loose_keys",
        input_1='{"label": "tanned european \\"SUPI\\""}',
    )
    assert 'label: tanned european "SUPI"' in string_out
    assert '"label"' not in string_out
    assert '"outfit"' not in string_out


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


def test_extractor_preserves_quote_chars_in_loose_keys_output():
    """Same rule as the stitcher: value-internal quotes are content (d2dda5b)."""
    src = '{"face": {"ethnicity": "tanned european \\"SUPI\\""}}'
    _, string_out, _ = _extract(src, "face")
    assert 'ethnicity: tanned european "SUPI"' in string_out
    assert '"ethnicity"' not in string_out
    assert '"face"' not in string_out


# ─── Extractor negative list ([NOT] exclusions) ───────────────────────


def test_extractor_neg_bare_key_prunes_recursively():
    """[NOT]key drops that key and its subtree at every depth."""
    src = '{"c1": {"face": {"eyes": "blue"}, "hair": {"colour": "blonde"}}}'
    raw, _, found = _extract(src, "c1\n[NOT]hair")
    assert found is True
    assert json.loads(raw) == {"c1": {"face": {"eyes": "blue"}}}


def test_extractor_neg_dot_path_prunes_exact_only():
    """[NOT]a.b.c removes only that path, siblings survive."""
    src = '{"hair": {"colour": "blonde", "length": "long"}}'
    raw, _, found = _extract(src, "hair\n[NOT]hair.colour")
    assert found is True
    assert json.loads(raw) == {"hair": {"length": "long"}}


def test_extractor_neg_only_returns_whole_doc_minus_excluded():
    """Exclusions with no positive line → whole doc minus those keys."""
    src = '{"face": {"eyes": "blue"}, "nsfw": {"x": 1}, "hair": {"c": "red"}}'
    raw, _, found = _extract(src, "[NOT]nsfw")
    assert found is True
    assert json.loads(raw) == {"face": {"eyes": "blue"}, "hair": {"c": "red"}}


def test_extractor_neg_prefix_is_case_insensitive():
    src = '{"a": {"x": 1}, "b": {"y": 2}}'
    raw, _, found = _extract(src, "[not]b")
    assert found is True
    assert json.loads(raw) == {"a": {"x": 1}}


def test_extractor_neg_prefix_tolerates_space():
    src = '{"a": {"x": 1}, "b": {"y": 2}}'
    raw, _, found = _extract(src, "[NOT] b")
    assert found is True
    assert json.loads(raw) == {"a": {"x": 1}}


def test_extractor_neg_recursive_hits_all_depths():
    """A bare-key exclusion removes the key wherever it appears."""
    src = '{"c1": {"colour": "red"}, "c2": {"deep": {"colour": "blue"}}}'
    raw, _, found = _extract(src, "[NOT]colour")
    assert found is True
    assert json.loads(raw) == {"c1": {}, "c2": {"deep": {}}}


def test_extractor_neg_prunes_positive_to_empty_is_miss():
    """User's rule: if exclusions empty the result, found=False + empty."""
    src = '{"hair": {"colour": "blonde"}}'
    raw, string_out, found = _extract(src, "[NOT]hair")
    assert found is False
    assert raw == ""
    assert string_out == ""


def test_extractor_neg_combined_pull_and_drop():
    src = '{"face": {"eyes": "blue"}, "hair": {"colour": "red", "length": "long"}}'
    raw, _, found = _extract(src, "face\nhair\n[NOT]hair.colour")
    assert found is True
    assert json.loads(raw) == {
        "face": {"eyes": "blue"},
        "hair": {"length": "long"},
    }


def test_extractor_neg_dot_path_missing_is_silent_noop():
    src = '{"hair": {"length": "long"}}'
    raw, _, found = _extract(src, "hair\n[NOT]hair.colour")
    assert found is True
    assert json.loads(raw) == {"hair": {"length": "long"}}


def test_extractor_no_negatives_preserves_legacy_behavior():
    """Without any [NOT] line, behavior is unchanged from before."""
    src = '{"face": {"eyes": "blue"}}'
    raw, _, found = _extract(src, "face")
    assert found is True
    assert json.loads(raw) == {"face": {"eyes": "blue"}}


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


# ─── Extractor: collect_all ───────────────────────────────────────────

_LOCATION_DOC = json.dumps({
    "__input4": {"location": {
        "set_name": "indoor/family_activities/movie_night_living_room",
        "seed": 821490257792246,
        "color_tone": "neutral",
        "elements": {
            "background": {"name": "wallpapered feature wall", "coverage": 0.87,
                           "texture": "patterned printed paper", "layer": "background",
                           "prompt_fragment": "wallpapered feature wall, patterned printed paper"},
            "midground": {"name": "shag area rug underfoot", "coverage": 0.5996,
                          "texture": "plush deep-pile floor cover", "layer": "midground",
                          "prompt_fragment": "shag area rug underfoot, plush deep-pile floor cover"},
            "time_of_day": {"name": "post-sunset cozy hour", "coverage": 0.0,
                            "texture": None, "layer": "atmosphere",
                            "prompt_fragment": "post-sunset cozy hour"},
        }}},
    "__input5": {"outfit": {
        "top": {"prompt_fragment": "cropped knit sweater"},
    }},
})


def _extract_all(json_input, category, output_format="loose_keys"):
    return FVM_JB_Extractor().extract(json_input, category, output_format, True)


def test_extractor_collect_all_gathers_every_match():
    """collect_all returns EVERY prompt_fragment, not just the first."""
    raw, _, found = _extract_all(_LOCATION_DOC, "prompt_fragment")
    assert found is True
    assert json.loads(raw) == {"prompt_fragment": [
        "wallpapered feature wall, patterned printed paper",
        "shag area rug underfoot, plush deep-pile floor cover",
        "post-sunset cozy hour",
        "cropped knit sweater",
    ]}


def test_extractor_collect_all_loose_keys_is_flat_joined():
    """loose_keys + collect_all → one encoder-ready comma line, no brackets."""
    _, string_out, _ = _extract_all(_LOCATION_DOC, "prompt_fragment")
    assert string_out == (
        "wallpapered feature wall, patterned printed paper, "
        "shag area rug underfoot, plush deep-pile floor cover, "
        "post-sunset cozy hour, cropped knit sweater"
    )
    assert "[" not in string_out and "{" not in string_out


def test_extractor_collect_all_dot_path_scopes_the_sweep():
    """All but the last segment is a strict descent — scopes to one slot."""
    raw, _, found = _extract_all(_LOCATION_DOC, "__input5.prompt_fragment")
    assert found is True
    assert json.loads(raw) == {"prompt_fragment": ["cropped knit sweater"]}


def test_extractor_collect_all_multi_category_concats_same_wrap_key():
    """Two scoped paths sharing a last segment append instead of overwriting."""
    raw, _, found = _extract_all(
        _LOCATION_DOC, "__input5.prompt_fragment, __input4.location.elements.time_of_day.prompt_fragment"
    )
    assert found is True
    assert json.loads(raw) == {"prompt_fragment": ["cropped knit sweater", "post-sunset cozy hour"]}


def test_extractor_collect_all_skips_nulls_and_empties_in_join():
    """None / empty leaves drop out of the joined line (no ', ,')."""
    _, string_out, _ = _extract_all(_LOCATION_DOC, "texture")
    assert string_out == "patterned printed paper, plush deep-pile floor cover"


def test_extractor_collect_all_missing_key_reports_not_found():
    raw, string_out, found = _extract_all(_LOCATION_DOC, "does_not_exist")
    assert (raw, string_out, found) == ("", "", False)


def test_extractor_collect_all_off_keeps_first_match_behavior():
    """Default stays first-match — no regression for existing workflows."""
    raw, _, found = _extract(_LOCATION_DOC, "prompt_fragment")
    assert found is True
    assert json.loads(raw) == {
        "prompt_fragment": "wallpapered feature wall, patterned printed paper"}


def test_extractor_collect_all_json_format_keeps_list_form():
    """pretty_json is untouched by the flat-join special case."""
    _, string_out, _ = _extract_all(_LOCATION_DOC, "prompt_fragment", "pretty_json")
    assert json.loads(string_out)["prompt_fragment"][0].startswith("wallpapered")


def test_extractor_collect_all_multiple_distinct_keys():
    """Several keys at once, each collected in full under its own wrap key."""
    raw, _, found = _extract_all(_LOCATION_DOC, "prompt_fragment, layer")
    assert found is True
    payload = json.loads(raw)
    assert len(payload["prompt_fragment"]) == 4
    assert payload["layer"] == ["background", "midground", "atmosphere"]


def test_extractor_with_keys_labels_each_value_by_owner():
    """with_keys keeps the element name that owns each fragment."""
    raw, string_out, found = FVM_JB_Extractor().extract(
        _LOCATION_DOC, "prompt_fragment", "loose_keys", True, True)
    assert found is True
    assert json.loads(raw) == {"prompt_fragment": {
        "background": "wallpapered feature wall, patterned printed paper",
        "midground": "shag area rug underfoot, plush deep-pile floor cover",
        "time_of_day": "post-sunset cozy hour",
        "top": "cropped knit sweater",
    }}
    assert string_out == (
        "background: wallpapered feature wall, patterned printed paper, "
        "midground: shag area rug underfoot, plush deep-pile floor cover, "
        "time_of_day: post-sunset cozy hour, "
        "top: cropped knit sweater"
    )


def test_extractor_with_keys_repeated_owner_collects_into_list():
    """Two elements with the same name keep both values instead of overwriting."""
    doc = json.dumps({"a": {"background": {"prompt_fragment": "one"}},
                      "b": {"background": {"prompt_fragment": "two"}}})
    raw, string_out, _ = FVM_JB_Extractor().extract(
        doc, "prompt_fragment", "loose_keys", True, True)
    assert json.loads(raw) == {"prompt_fragment": {"background": ["one", "two"]}}
    assert string_out == "background: one, background: two"


def test_extractor_with_keys_ignored_without_collect_all():
    """with_keys is a collect_all modifier — on its own it changes nothing."""
    a = FVM_JB_Extractor().extract(_LOCATION_DOC, "prompt_fragment", "loose_keys", False, True)
    b = FVM_JB_Extractor().extract(_LOCATION_DOC, "prompt_fragment", "loose_keys", False, False)
    assert a == b


def test_extractor_collect_all_json_formats_stay_strict_json():
    """No loose_keys half-syntax in collect_all mode: text or real JSON."""
    for fmt in ("pretty_json", "compact_json"):
        _, string_out, _ = FVM_JB_Extractor().extract(
            _LOCATION_DOC, "prompt_fragment", fmt, True, True)
        assert json.loads(string_out)["prompt_fragment"]["background"].startswith("wallpapered")
    _, plain, _ = _extract_all(_LOCATION_DOC, "prompt_fragment")
    assert "{" not in plain and "[" not in plain and '"' not in plain
