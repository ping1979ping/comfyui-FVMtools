"""``sentences`` output format — deterministic prose with lead-ins for Krea 2.

Every sentence is mechanical: intro from the set path, one sentence per
element / garment, one sentence per branch for anything else. No person is
ever mentioned, no metadata leaks, no key survives as a bare word.
"""

import json
import re

import pytest

from core.jb.serialize import ALL_FORMATS, NON_PROMPT_KEYS, SENTENCES, emit
from core.jb.sentences import (
    GARMENT_LEAD_INS,
    LOCATION_LEAD_INS,
    ONE_PIECE_LEAD_IN,
    emit_sentences,
    is_one_piece,
    location_intro,
    outfit_intro,
    sentences,
)
from core.location_engine import get_available_location_sets
from core.outfit_lists import get_available_sets
from nodes.jb.builder import FVM_JB_Builder
from nodes.jb.extractor import FVM_JB_Extractor
from nodes.jb.location_block import FVM_JB_LocationBlock
from nodes.jb.outfit_block import FVM_JB_OutfitBlock
from nodes.jb.stitcher import FVM_JB_Stitcher

PERSON_WORDS = re.compile(r"\b(woman|man|she|he|her|his|female|male|person|wearing)\b", re.I)


def _location(seed=3, **extra):
    args = dict(
        location_set="indoor/family_events/graduation_party_home", seed=seed,
        enable_background=True, enable_midground=True,
        enable_architecture_detail=True, enable_props=True,
        enable_foreground_element=True, enable_time_of_day=True,
        enable_weather=True, output_format=SENTENCES,
    )
    args.update(extra)
    return FVM_JB_LocationBlock().build(**args)


def _outfit(seed=3, **extra):
    args = dict(
        outfit_set="female/business/dress", seed=seed, style_preset="general",
        formality=0.7, coverage=0.6,
        enable_headwear=False, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=True, enable_accessories=True,
        enable_bag=True, print_probability=0.3, text_mode="off",
        output_format=SENTENCES,
    )
    args.update(extra)
    return FVM_JB_OutfitBlock().build(**args)


# ─── Intros ──────────────────────────────────────────────────────────


class TestLocationIntro:
    def test_three_segment_path(self):
        assert location_intro("indoor/family_event/graduation_party_at_home") == (
            "The scene takes place indoors, it is a family event, "
            "namely the graduation party at home."
        )

    def test_outdoor_and_vowel_article(self):
        assert location_intro("outdoor/american_everyday_scene/cvs_parking_lot") == (
            "The scene takes place outdoors, it is an american everyday scene, "
            "namely the cvs parking lot."
        )

    def test_two_segments(self):
        assert location_intro("outdoor/beach") == \
            "The scene takes place outdoors, namely the beach."

    def test_scope_only(self):
        assert location_intro("indoor") == "The scene takes place indoors."

    def test_legacy_flat_slug(self):
        assert location_intro("indoor_business_skyscraper_lobby") == \
            "The scene takes place indoors, namely the business skyscraper lobby."

    def test_unknown_shape_still_reads(self):
        assert location_intro("studio_minimal") == "The scene is the studio minimal."

    def test_empty(self):
        assert location_intro("") == ""
        assert location_intro(None) == ""

    def test_deeper_paths_join_the_tail(self):
        assert location_intro("indoor/cat/b/c") == \
            "The scene takes place indoors, it is a cat, namely the b c."


class TestOutfitIntro:
    def test_category_and_leaf(self):
        assert outfit_intro("female/business/dress") == \
            "The outfit is a business look, in the dress style."

    def test_general_leaf_is_omitted(self):
        assert outfit_intro("male/casual/general") == "The outfit is a casual look."

    def test_vowel_article(self):
        assert outfit_intro("female/athleisure/hot_pilates") == \
            "The outfit is an athleisure look, in the hot pilates style."

    @pytest.mark.parametrize("set_name", [
        "female/lingerie/lace", "male/gothic/cyber_goth", "female/teen/casual_jeans",
    ])
    def test_gender_is_never_mentioned(self, set_name):
        assert not PERSON_WORDS.search(outfit_intro(set_name)), outfit_intro(set_name)

    def test_legacy_flat_slugs(self):
        assert outfit_intro("night_out_female") == "The outfit is a night out look."
        assert outfit_intro("business_female_dress") == \
            "The outfit is a business look, in the dress style."
        assert outfit_intro("general_female") == "The outfit is a general look."

    def test_gender_only_gives_nothing(self):
        assert outfit_intro("female") == ""
        assert outfit_intro("") == ""


# ─── Lead-ins ────────────────────────────────────────────────────────


def _record(fragment, name="n"):
    return {"name": name, "coverage": 0.5, "layer": "l", "texture": "t",
            "fabric": "wool", "color_role": "primary", "color_resolved": "x",
            "prompt_fragment": fragment}


class TestLeadIns:
    @pytest.mark.parametrize("key", sorted(LOCATION_LEAD_INS))
    def test_every_location_element(self, key):
        out = emit_sentences({"elements": {key: _record("red brick wall")}})
        assert out == LOCATION_LEAD_INS[key].format("red brick wall")

    @pytest.mark.parametrize("key", sorted(GARMENT_LEAD_INS))
    def test_every_garment_slot(self, key):
        out = emit_sentences({"garments": {key: _record("sand jersey crew tee", "crew tee")}})
        assert out == GARMENT_LEAD_INS[key].format("sand jersey crew tee")

    def test_record_metadata_never_leaks(self):
        out = emit_sentences({"elements": {"background": _record("red brick wall")}})
        for token in ("0.5", "coverage", "layer", "texture", "wool", "primary", "name"):
            assert token not in out, token

    def test_unknown_key_falls_back_to_key_is_value(self):
        assert emit_sentences({"hair": {"prompt_fragment": "long auburn hair"}}) == \
            "The hair is long auburn hair."

    def test_multi_word_key(self):
        assert emit_sentences({"eye_colour": {"prompt_fragment": "amber"}}) == \
            "The eye colour is amber."

    def test_other_phrase_keys_work_too(self):
        assert emit_sentences({"props": {"phrase": "paper cup"}}) == \
            "The props include paper cup."


class TestOnePiece:
    @pytest.mark.parametrize("name", [
        "sheath dress", "long-sleeve wrap dress", "silk slip dress", "sundress",
        "wide-leg jumpsuit", "denim overalls", "lace bodysuit", "competition leotard",
        "one-piece swimsuit", "satin chemise", "beach kaftan", "romper",
        "evening gown", "dirndl", "cotton nightgown", "playsuit", "catsuit",
    ])
    def test_positive(self, name):
        assert is_one_piece(name), name

    @pytest.mark.parametrize("name", [
        "dress shirt", "dress pants", "dress shoes", "bikini top", "crew tee",
        "pencil skirt", "tracksuit bottoms", "kimono jacket", "tankini top",
        "denim jacket", "jeans", "blazer", "swim shorts", "",
    ])
    def test_negative(self, name):
        assert not is_one_piece(name), name

    def test_decoration_clause_is_ignored(self):
        assert is_one_piece("sheath dress with pinstripe pattern")

    def test_colour_token_is_ignored(self):
        assert is_one_piece("long #color# dress")

    def test_only_body_slots_use_the_one_piece_lead_in(self):
        rec = _record("navy wrap dress", "wrap dress")
        assert emit_sentences({"garments": {"lower_body": rec}}) == \
            ONE_PIECE_LEAD_IN.format("navy wrap dress")
        assert emit_sentences({"garments": {"upper_body": rec}}) == \
            ONE_PIECE_LEAD_IN.format("navy wrap dress")
        assert emit_sentences({"garments": {"upper_body_outerwear": rec}}) == \
            "The outerwear is navy wrap dress."

    def test_fragment_is_used_when_no_name(self):
        assert emit_sentences({"garments": {"lower_body": {
            "prompt_fragment": "off-white wool sheath dress with pinstripe pattern"}}}) == \
            ONE_PIECE_LEAD_IN.format("off-white wool sheath dress with pinstripe pattern")

    def test_dress_set_never_claims_the_bottom(self):
        """female/business/dress carries the dress in the bottom slot."""
        for seed in range(8):
            _, out, _ = _outfit(seed=seed)
            assert "The bottom is" not in out, out
            assert "one-piece garment" in out, out


# ─── Metadata ────────────────────────────────────────────────────────


class TestMetadata:
    def test_every_non_prompt_key_is_filtered(self):
        data = {key: f"value-of-{key}" for key in NON_PROMPT_KEYS}
        data["top"] = "grey tee"
        assert emit_sentences(data) == "The top is grey tee."

    def test_numbers_bools_none_dropped(self):
        data = {"x": 0.5, "n": 12, "on": True, "off": False, "z": None, "text": "a red car"}
        assert emit_sentences(data) == "The text is a red car."

    def test_underscore_keys_dropped(self):
        assert emit_sentences({"_debug": "internal", "keep": "a blue door"}) == \
            "The keep is a blue door."

    def test_formality_and_colour_tone_dropped(self):
        outfit = {"outfit": {
            "set_name": "female/business/dress", "seed": 3, "formality": "evening",
            "coverage_target": 0.6, "color_tone": "warm",
            "garments": {"footwear": _record("khaki leather block heels", "block heels")},
        }}
        out = emit_sentences(outfit)
        assert out == ("The outfit is a business look, in the dress style. "
                       "The footwear is khaki leather block heels.")
        assert "evening" not in out and "warm" not in out and "0.6" not in out

    def test_block_output_has_no_structure_characters(self):
        for out in (_location()[1], _outfit()[1]):
            for ch in "{}[]#_/\"":
                assert ch not in out, f"{ch!r} in {out}"


# ─── Generic fallback (one sentence per branch) ──────────────────────


class TestGeneric:
    def test_one_sentence_per_branch(self):
        data = {"face": {"age": "twenties", "eyes": "amber"}, "hair": {"colour": "blonde"}}
        assert emit_sentences(data) == \
            "The face has age twenties and eyes amber. The hair has colour blonde."

    def test_three_leaves_use_commas_and_and(self):
        data = {"face": {"age": "twenties", "eyes": "amber", "skin": "freckled"}}
        assert emit_sentences(data) == \
            "The face has age twenties, eyes amber and skin freckled."

    def test_top_level_leaves(self):
        assert emit_sentences({"top": "grey tee", "text": "VARSITY"}) == \
            "The top is grey tee. The text is VARSITY."

    def test_string_lists(self):
        assert emit_sentences({"tags": ["sunlit", "windswept", "quiet"]}) == \
            "The tags are sunlit, windswept and quiet."

    def test_leaves_before_nested_branches(self):
        data = {"character": {"mood": "calm", "face": {"eyes": "amber"}}}
        assert emit_sentences(data) == \
            "The character has mood calm. The face has eyes amber."

    def test_lists_of_records_walk_with_the_parent_key(self):
        data = {"props": [{"prompt_fragment": "wooden bench"},
                          {"prompt_fragment": "paper cup"}]}
        assert emit_sentences(data) == \
            "The props include wooden bench. The props include paper cup."

    def test_duplicates_collapse(self):
        data = {"x": {"top": {"prompt_fragment": "navy coat"}},
                "y": {"top": {"prompt_fragment": "Navy Coat"}}}
        assert emit_sentences(data) == "The top is navy coat."

    def test_trailing_period_not_doubled(self):
        assert emit_sentences({"note": "a quiet street."}) == "The note is a quiet street."

    def test_whitespace_collapsed(self):
        assert emit_sentences({"note": "  a   quiet\nstreet "}) == "The note is a quiet street."

    def test_empty(self):
        assert emit_sentences({}) == ""
        assert emit_sentences({"outfit": {"seed": 1, "garments": {}}}) == ""
        assert emit({"seed": 1}, SENTENCES) == ""

    def test_sentences_list_api(self):
        assert sentences({"a": "x", "b": "y"}) == ["The a is x.", "The b is y."]


# ─── Nodes ───────────────────────────────────────────────────────────


class TestNodes:
    def test_format_registered_everywhere(self):
        assert SENTENCES in ALL_FORMATS
        for node in (FVM_JB_OutfitBlock, FVM_JB_LocationBlock, FVM_JB_Builder,
                     FVM_JB_Stitcher, FVM_JB_Extractor):
            assert SENTENCES in node.INPUT_TYPES()["required"]["output_format"][0], node

    def test_location_block(self):
        loc_json, out, _ = _location()
        assert out.startswith("The scene takes place indoors, it is a family event")
        for key in json.loads(loc_json)["location"]["elements"]:
            lead = LOCATION_LEAD_INS[key].split("{}")[0]
            assert lead in out, f"{key}: {lead!r} missing in {out}"
        assert "#" not in out

    def test_location_block_is_deterministic_and_json_unchanged(self):
        a = _location(seed=9)
        b = _location(seed=9)
        c = _location(seed=9, output_format="loose_keys")
        assert a == b
        assert a[0] == c[0]  # location_json is format-independent

    def test_outfit_block_describes_clothes_only(self):
        _, out, _ = _outfit()
        assert out.startswith("The outfit is a business look, in the dress style.")
        assert not PERSON_WORDS.search(out), out

    def test_outfit_block_other_formats_untouched(self):
        _, natural, _ = _outfit(output_format="natural")
        _, loose, _ = _outfit(output_format="loose_keys")
        assert natural.startswith("wearing ")
        assert "outfit:" in loose

    def test_stitcher_prefixes_title_and_keeps_both_intros(self):
        outfit_json, _, _ = _outfit(seed=1)
        location_json, _, _ = _location(seed=1)
        _, out = FVM_JB_Stitcher().stitch(
            "character_1", SENTENCES, input_1=outfit_json, input_2=location_json)
        assert out.startswith("character_1: The outfit is a business look")
        assert "The scene takes place indoors" in out
        assert "The footwear is" in out and "The background is" in out

    def test_stitcher_prose_input_survives(self):
        prose = 'wearing grey tee with "VARSITY" text in varsity block'
        _, out = FVM_JB_Stitcher().stitch("c1", SENTENCES, input_1=prose)
        assert out == f"c1: {prose}."

    def test_stitcher_array_and_loose_inputs(self):
        _, out = FVM_JB_Stitcher().stitch("tags", SENTENCES, input_1='["sunlit", "windswept"]')
        assert out == "tags: sunlit and windswept."
        _, out = FVM_JB_Stitcher().stitch("c1", SENTENCES, input_1='top: "grey tee", text: "VARSITY"')
        assert out == "c1: The top is grey tee. The text is VARSITY."

    def test_stitcher_empty_gives_empty(self):
        _, out = FVM_JB_Stitcher().stitch("c1", SENTENCES)
        assert out == ""

    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_stitcher_never_swallows_prose_in_any_format(self, fmt):
        _, out = FVM_JB_Stitcher().stitch("c1", fmt, input_1="a plain prose fragment")
        assert "a plain prose fragment" in out

    def test_extractor_garments_subtree_has_lead_ins_but_no_intro(self):
        outfit_json, _, _ = _outfit(seed=1)
        _, out, found = FVM_JB_Extractor().extract(outfit_json, "outfit.garments", SENTENCES)
        assert found
        assert "The outfit is" not in out
        assert "The footwear is" in out

    def test_extractor_whole_outfit_keeps_intro(self):
        outfit_json, _, _ = _outfit(seed=1)
        _, out, found = FVM_JB_Extractor().extract(outfit_json, "outfit", SENTENCES)
        assert found and out.startswith("The outfit is a business look")

    def test_builder_rows(self):
        rows = json.dumps([
            {"key": "face", "value": "", "indent": 0},
            {"key": "age", "value": "twenties", "indent": 1},
            {"key": "eyes", "value": "amber", "indent": 1},
        ])
        _, out = FVM_JB_Builder().build(rows=rows, seed=0, output_format=SENTENCES)
        assert out == "The face has age twenties and eyes amber."


# ─── Every set on disk reads cleanly ─────────────────────────────────


@pytest.mark.parametrize("set_name", get_available_location_sets())
def test_every_location_set_has_a_clean_intro(set_name):
    intro = location_intro(set_name)
    assert intro.startswith("The scene takes place ")
    assert intro.endswith(".")
    assert "_" not in intro and "/" not in intro


@pytest.mark.parametrize("set_name", [s for s in get_available_sets() if "/" in s])
def test_every_outfit_set_has_a_clean_intro(set_name):
    intro = outfit_intro(set_name)
    assert intro.startswith("The outfit is ")
    assert intro.endswith(".")
    assert "_" not in intro and "/" not in intro
    assert not PERSON_WORDS.search(intro), intro
