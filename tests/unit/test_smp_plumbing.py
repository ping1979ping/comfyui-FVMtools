"""P5 — tests for the PROMPT_DICT plumbing (merge, builders, aggregator, serialize)."""

import json

import pytest

from core.smp.merge import deep_merge, merge_many


# ─── deep_merge ─────────────────────────────────────────────────────────


def test_merge_empty_left():
    assert deep_merge({}, {"a": 1}) == {"a": 1}
    assert deep_merge(None, {"a": 1}) == {"a": 1}


def test_merge_empty_right():
    assert deep_merge({"a": 1}, {}) == {"a": 1}
    assert deep_merge({"a": 1}, None) == {"a": 1}


def test_merge_scalar_overwrite():
    assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_merge_dicts_recursive():
    a = {"k": {"x": 1, "y": 2}}
    b = {"k": {"y": 99, "z": 3}}
    assert deep_merge(a, b) == {"k": {"x": 1, "y": 99, "z": 3}}


def test_merge_lists_concatenate():
    a = {"subjects": [{"id": "s1"}]}
    b = {"subjects": [{"id": "s2"}]}
    assert deep_merge(a, b) == {"subjects": [{"id": "s1"}, {"id": "s2"}]}


def test_merge_none_deletes_field():
    a = {"a": 1, "b": 2}
    b = {"a": None}
    assert deep_merge(a, b) == {"b": 2}


def test_merge_does_not_mutate():
    a = {"k": {"x": 1}}
    b = {"k": {"y": 2}}
    out = deep_merge(a, b)
    out["k"]["x"] = 999
    assert a["k"]["x"] == 1


def test_merge_many_chains():
    a = {"meta": {"seed": 1}}
    b = {"meta": {"label": "demo"}}
    c = {"subjects": [{"id": "s1"}]}
    out = merge_many(a, b, c)
    assert out == {"meta": {"seed": 1, "label": "demo"},
                   "subjects": [{"id": "s1"}]}


# ─── SubjectBuilder ─────────────────────────────────────────────────────


def test_subject_builder_appends_to_list():
    from nodes.smp.builders.subject_builder import FVM_SMP_SubjectBuilder
    out_a, = FVM_SMP_SubjectBuilder().build(
        subject_id="s1", age_desc="young", gender="woman",
        expression="neutral", hair_color_length="dark hair",
        pose_hint="", extra_json="{}",
    )
    out_b, = FVM_SMP_SubjectBuilder().build(
        subject_id="s2", age_desc="middle-aged", gender="man",
        expression="confident", hair_color_length="grey hair",
        pose_hint="", extra_json="{}",
        prompt_dict_in=out_a,
    )
    assert len(out_b["subjects"]) == 2
    assert out_b["subjects"][0]["id"] == "s1"
    assert out_b["subjects"][1]["id"] == "s2"


def test_subject_builder_extras_json_merges():
    from nodes.smp.builders.subject_builder import FVM_SMP_SubjectBuilder
    out, = FVM_SMP_SubjectBuilder().build(
        subject_id="s1", age_desc="young", gender="woman",
        expression="neutral", hair_color_length="",
        pose_hint="",
        extra_json='{"eye_desc": "bright eyes", "skin_tags": ["smooth skin"]}',
    )
    s = out["subjects"][0]
    assert s["eye_desc"] == "bright eyes"
    assert s["skin_tags"] == ["smooth skin"]


def test_subject_builder_dict_input_wins():
    from nodes.smp.builders.subject_builder import FVM_SMP_SubjectBuilder
    out, = FVM_SMP_SubjectBuilder().build(
        subject_id="ignored", age_desc="ignored", gender="ignored",
        expression="ignored", hair_color_length="",
        pose_hint="",
        extra_json="{}",
        subject_dict={"id": "from_dict", "age_desc": "ancient",
                      "gender": "the_void"},
    )
    s = out["subjects"][0]
    assert s["id"] == "from_dict"
    assert s["age_desc"] == "ancient"


# ─── ClothingBuilder ────────────────────────────────────────────────────


def test_clothing_builder_attaches_outfit():
    from nodes.smp.builders.clothing_builder import FVM_SMP_ClothingBuilder
    outfit = {"set_name": "x", "garments": {"top": {"prompt_fragment": "shirt"}}}
    out, = FVM_SMP_ClothingBuilder().build(outfit, "subject_1")
    assert out["outfits"]["subject_1"] == outfit


def test_clothing_builder_two_subjects():
    from nodes.smp.builders.clothing_builder import FVM_SMP_ClothingBuilder
    a, = FVM_SMP_ClothingBuilder().build({"set_name": "a"}, "s1")
    b, = FVM_SMP_ClothingBuilder().build({"set_name": "b"}, "s2",
                                          prompt_dict_in=a)
    assert b["outfits"] == {"s1": {"set_name": "a"}, "s2": {"set_name": "b"}}


# ─── EnvironmentBuilder ─────────────────────────────────────────────────


def test_environment_builder_attaches_location():
    from nodes.smp.builders.environment_builder import FVM_SMP_EnvironmentBuilder
    loc = {"set_name": "indoor_studio_minimal", "elements": {}}
    out, = FVM_SMP_EnvironmentBuilder().build(loc)
    assert out["location"] == loc


def test_environment_builder_replaces_prior_location():
    from nodes.smp.builders.environment_builder import FVM_SMP_EnvironmentBuilder
    a, = FVM_SMP_EnvironmentBuilder().build({"set_name": "first"})
    b, = FVM_SMP_EnvironmentBuilder().build({"set_name": "second"},
                                             prompt_dict_in=a)
    # Last EnvironmentBuilder wins (single-scene assumption); merge replaces
    # the set_name scalar.
    assert b["location"]["set_name"] == "second"


# ─── Aggregator ─────────────────────────────────────────────────────────


def test_aggregator_two_branches():
    from nodes.smp.aggregator import FVM_SMP_Aggregator
    a = {"meta": {"seed": 1}}
    b = {"meta": {"label": "demo"}}
    out, = FVM_SMP_Aggregator().aggregate(a, b)
    assert out["meta"] == {"seed": 1, "label": "demo"}


def test_aggregator_four_branches():
    from nodes.smp.aggregator import FVM_SMP_Aggregator
    out, = FVM_SMP_Aggregator().aggregate(
        {"a": 1}, {"b": 2}, {"c": 3}, {"d": 4},
    )
    assert out == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_aggregator_lists_concatenate():
    from nodes.smp.aggregator import FVM_SMP_Aggregator
    a = {"subjects": [{"id": "s1"}]}
    b = {"subjects": [{"id": "s2"}]}
    out, = FVM_SMP_Aggregator().aggregate(a, b)
    assert [s["id"] for s in out["subjects"]] == ["s1", "s2"]


# ─── Serialize ──────────────────────────────────────────────────────────


def _full_prompt_dict():
    return {
        "meta": {"seed": 42, "label": "demo"},
        "subjects": [{
            "id": "subject_1",
            "age_desc": "young",
            "gender": "woman",
            "skin_tags": ["smooth skin"],
            "eye_desc": "bright eyes",
            "expression": "calm",
            "hair_color_length": "dark hair",
            "pose_hint": "seated on wide steps",
        }],
        "outfits": {
            "subject_1": {
                "garments": {
                    "upper_body": {"prompt_fragment": "fitted blazer in burgundy"},
                    "lower_body": {"prompt_fragment": "pencil skirt in charcoal"},
                    "footwear":   {"prompt_fragment": "patent black pumps"},
                },
                "formality": "formal",
            }
        },
        "location": {
            "elements": {
                "background": {"prompt_fragment": "concrete wall in warm afternoon"},
                "weather":    {"prompt_fragment": "clear sky"},
            }
        },
        "post_processing": {
            "negative_prompt": "blurry, low quality",
            "quality_tags": ["8k", "professional photography"],
        },
    }


def test_serialize_returns_three_strings():
    from nodes.smp.serialize import FVM_SMP_PromptSerialize
    pos, neg, raw = FVM_SMP_PromptSerialize().serialize(_full_prompt_dict(), "natural_language")
    assert isinstance(pos, str) and pos
    assert isinstance(neg, str) and neg
    assert isinstance(raw, str) and raw


def test_serialize_natural_language_includes_outfit():
    from nodes.smp.serialize import FVM_SMP_PromptSerialize
    pos, _, _ = FVM_SMP_PromptSerialize().serialize(_full_prompt_dict(), "natural_language")
    assert "fitted blazer" in pos
    assert "pencil skirt" in pos
    assert "patent black pumps" in pos
    assert "concrete wall" in pos


def test_serialize_negative_uses_post_processing():
    from nodes.smp.serialize import FVM_SMP_PromptSerialize
    _, neg, _ = FVM_SMP_PromptSerialize().serialize(_full_prompt_dict(), "natural_language")
    assert neg == "blurry, low quality"


def test_serialize_negative_default_when_absent():
    from nodes.smp.serialize import FVM_SMP_PromptSerialize
    pd = {"subjects": [{"id": "s1"}]}
    _, neg, _ = FVM_SMP_PromptSerialize().serialize(pd, "natural_language")
    assert "blurry" in neg


def test_serialize_raw_json_format():
    from nodes.smp.serialize import FVM_SMP_PromptSerialize
    pos, _, raw = FVM_SMP_PromptSerialize().serialize(_full_prompt_dict(), "raw_json")
    assert pos == raw
    parsed = json.loads(raw)
    assert parsed["meta"]["seed"] == 42


def test_serialize_raw_json_always_third_output():
    """Even in NL mode, raw_json output is the structured dict."""
    from nodes.smp.serialize import FVM_SMP_PromptSerialize
    _, _, raw = FVM_SMP_PromptSerialize().serialize(_full_prompt_dict(), "natural_language")
    parsed = json.loads(raw)
    assert "subjects" in parsed
    assert "outfits" in parsed


def test_serialize_handles_empty_dict():
    from nodes.smp.serialize import FVM_SMP_PromptSerialize
    pos, neg, raw = FVM_SMP_PromptSerialize().serialize({}, "natural_language")
    assert isinstance(pos, str)
    assert "blurry" in neg
    assert json.loads(raw) == {}


# ─── Full P2→P5 pipeline integration ──────────────────────────────────


def test_full_pipeline_generators_to_serialize():
    from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator
    from nodes.smp.color_generator import FVM_SMP_ColorGenerator
    from nodes.smp.outfit_combiner import FVM_SMP_OutfitCombiner
    from nodes.smp.location_generator import FVM_SMP_LocationGenerator
    from nodes.smp.location_combiner import FVM_SMP_LocationCombiner
    from nodes.smp.builders.subject_builder import FVM_SMP_SubjectBuilder
    from nodes.smp.builders.clothing_builder import FVM_SMP_ClothingBuilder
    from nodes.smp.builders.environment_builder import FVM_SMP_EnvironmentBuilder
    from nodes.smp.aggregator import FVM_SMP_Aggregator
    from nodes.smp.serialize import FVM_SMP_PromptSerialize

    palette, _ = FVM_SMP_ColorGenerator().generate(
        seed=42, num_colors=5, harmony_type="auto", style_preset="general",
        vibrancy=0.5, contrast=0.5, warmth=0.6,
    )
    outfit_raw, _ = FVM_SMP_OutfitGenerator().generate(
        outfit_set="general_female", seed=42, style_preset="general",
        formality=0.5, coverage=0.6,
        enable_headwear=False, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=False,
        enable_accessories=False, enable_bag=False,
        print_probability=0.3, text_mode="auto",
    )
    outfit_dict, _ = FVM_SMP_OutfitCombiner().combine(outfit_raw, palette)
    location_raw, _ = FVM_SMP_LocationGenerator().generate(
        location_set="outdoor_urban_brutalist", seed=42,
        enable_background=True, enable_midground=False,
        enable_architecture_detail=False, enable_props=False,
        enable_foreground_element=True,
        enable_time_of_day=True, enable_weather=True,
        color_tone="",
    )
    location_dict, _ = FVM_SMP_LocationCombiner().combine(location_raw, palette)

    pd_subj, = FVM_SMP_SubjectBuilder().build(
        subject_id="subject_1", age_desc="young", gender="woman",
        expression="confident", hair_color_length="dark auburn hair",
        pose_hint="seated on wide steps",
        extra_json='{"skin_tags": ["smooth skin"], "eye_desc": "bright green eyes"}',
    )
    pd_cloth, = FVM_SMP_ClothingBuilder().build(outfit_dict, "subject_1")
    pd_env, = FVM_SMP_EnvironmentBuilder().build(location_dict)

    aggregated, = FVM_SMP_Aggregator().aggregate(pd_subj, pd_cloth, pd_env)

    assert aggregated["subjects"][0]["age_desc"] == "young"
    assert aggregated["outfits"]["subject_1"] == outfit_dict
    assert aggregated["location"] == location_dict

    pos, neg, raw = FVM_SMP_PromptSerialize().serialize(aggregated, "natural_language")
    assert "young woman" in pos
    assert "wearing" in pos
    parsed = json.loads(raw)
    assert parsed["subjects"][0]["age_desc"] == "young"
