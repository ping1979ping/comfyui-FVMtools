"""Stitcher merge rules: empty values never wipe filled ones; same key can keep both texts."""
import json

from nodes.jb.stitcher import FVM_JB_Stitcher, _merge_inputs

LOCATION = json.dumps({"location": {
    "set_name": "outdoor/pennsylvania_small_town/diner_parking_lot",
    "elements": {"background": {"name": "route highway", "prompt_fragment": "route highway beyond the lot"}},
}})


def _stitch(fmt="sentences", **inputs):
    return FVM_JB_Stitcher().stitch("scene", fmt, **inputs)


class TestEmptyNeverWipes:
    def test_empty_location_row_does_not_wipe_the_block(self):
        assert "route highway beyond the lot" in _stitch(input_6=LOCATION, input_7='{"location": ""}')[1]

    def test_empty_dict_does_not_wipe_it_either(self):
        assert "route highway beyond the lot" in _stitch(input_6=LOCATION, input_7='{"location": {}}')[1]

    def test_empty_value_without_a_filled_one_before_it_is_kept(self):
        raw = _stitch("pretty_json", input_1='{"note": ""}')[0]
        assert json.loads(raw)["scene"]["note"] == ""

    def test_empty_protection_recurses_into_nested_dicts(self):
        base = {"a": {"b": "x", "c": "y"}}
        assert _merge_inputs(base, {"a": {"b": "", "c": "z"}}, keep_both=False) == {"a": {"b": "x", "c": "z"}}


class TestSameKey:
    def test_default_is_last_wins(self):
        out = _stitch(input_1='{"mood": "calm"}', input_2='{"mood": "tense"}')[1]
        assert "tense" in out and "calm" not in out

    def test_keep_both_shows_the_texts_side_by_side(self):
        out = _stitch(on_conflict="keep_both",
                      input_1='{"mood": "calm", "light": "soft"}', input_2='{"mood": "tense", "light": "hard"}')[1]
        for word in ("calm", "tense", "soft", "hard"):
            assert word in out

    def test_json_shows_both_as_a_list(self):
        raw = _stitch("pretty_json", on_conflict="keep_both", input_1='{"mood": "calm"}', input_2='{"mood": "tense"}')[0]
        assert json.loads(raw)["scene"]["mood"] == ["calm", "tense"]

    def test_identical_text_is_not_doubled(self):
        raw = _stitch("pretty_json", on_conflict="keep_both", input_1='{"mood": "calm"}', input_2='{"mood": "calm"}')[0]
        assert json.loads(raw)["scene"]["mood"] == "calm"

    def test_third_input_extends_the_list(self):
        raw = _stitch("pretty_json", on_conflict="keep_both", input_1='{"m": "a"}', input_2='{"m": "b"}', input_3='{"m": "c"}')[0]
        assert json.loads(raw)["scene"]["m"] == ["a", "b", "c"]

    def test_numbers_and_nested_dicts_keep_their_old_rules(self):
        out = _merge_inputs({"seed": 1, "x": {"a": "p"}}, {"seed": 2, "x": {"b": "q"}}, keep_both=True)
        assert out == {"seed": 2, "x": {"a": "p", "b": "q"}}

    def test_lists_still_append(self):
        assert _merge_inputs({"p": ["lamp"]}, {"p": ["chair"]}, keep_both=True) == {"p": ["lamp", "chair"]}

    def test_none_still_deletes(self):
        assert _merge_inputs({"a": "x"}, {"a": None}, keep_both=True) == {}

    def test_inputs_are_not_mutated(self):
        base = {"a": "x"}
        _merge_inputs(base, {"a": "y"}, keep_both=True)
        assert base == {"a": "x"}
