"""P4 — tests for the StructuredPromptAssembler + SAMClassRouter."""

import json

import pytest

from core.smp.assembler import (
    BODY_QUALITY_TAGS,
    DEFAULT_HANDS_FRAGMENT,
    FACE_QUALITY_TAGS,
    GARMENT_SPATIAL_ORDER,
    LOCATION_LAYER_ORDER,
    assemble_structured,
    build_body_prompt,
    build_face_prompt,
    build_location_prompt,
    build_outfit_prompt,
    build_region_map,
    build_sam_class_lookup,
)


# ─── Synthetic fixtures ────────────────────────────────────────────────


@pytest.fixture
def subject():
    return {
        "id": "subject_1",
        "age_desc": "young",
        "gender": "woman",
        "ethnicity_tag": "east asian",
        "skin_tags": ["smooth skin", "detailed skin texture"],
        "eye_desc": "bright green almond eyes",
        "brow_desc": "arched brows",
        "lip_desc": "soft full lips",
        "nose_desc": "delicate nose",
        "expression": "neutral confident gaze",
        "makeup": "natural office makeup",
        "hair_color_length": "dark auburn hair",
        "hair_full": {
            "color": "dark auburn",
            "style": "long straight, center-parted",
            "length": "waist-length",
            "details": "healthy shine",
        },
        "body_build": "slim build",
        "body_height": "average height",
        "pose_hint": "seated on wide steps, slight S-curve",
    }


@pytest.fixture
def outfit():
    return {
        "set_name": "business_female",
        "seed": 1,
        "formality": "formal",
        "garments": {
            "headwear": {
                "prompt_fragment": "minimalist felt beret in burgundy",
                "color_role": "accent",
                "region_hint": {
                    "region_id": "headwear",
                    "sam_class_hint": "hat",
                    "bbox_relative": (0.30, 0.0, 0.70, 0.18),
                    "layer_depth": "subject",
                },
            },
            "upper_body": {
                "prompt_fragment": "fitted wool blend blazer in burgundy, tailored lapels",
                "color_role": "primary",
                "region_hint": {
                    "region_id": "upper_body",
                    "sam_class_hint": "upper_clothes",
                    "bbox_relative": (0.20, 0.15, 0.80, 0.55),
                    "layer_depth": "subject",
                },
            },
            "lower_body": {
                "prompt_fragment": "slim pencil skirt in charcoal, knee length",
                "color_role": "secondary",
                "region_hint": {
                    "region_id": "lower_body",
                    "sam_class_hint": "skirt",
                    "bbox_relative": (0.25, 0.45, 0.75, 0.85),
                    "layer_depth": "subject",
                },
            },
            "footwear": {
                "prompt_fragment": "pointy closed toe high heel pumps in patent black",
                "color_role": "neutral",
                "region_hint": {
                    "region_id": "footwear",
                    "sam_class_hint": "shoes",
                    "bbox_relative": (0.30, 0.90, 0.70, 1.00),
                    "layer_depth": "subject",
                },
            },
        },
    }


@pytest.fixture
def location():
    return {
        "set_name": "outdoor_urban_brutalist",
        "elements": {
            "background": {
                "prompt_fragment": "monolithic concrete facade, raw poured surface, illuminated by warm amber afternoon",
                "layer": "background",
                "region_hint": {
                    "region_id": "background",
                    "sam_class_hint": "background",
                    "bbox_relative": (0.0, 0.0, 1.0, 0.7),
                    "layer_depth": "background",
                },
            },
            "foreground_element": {
                "prompt_fragment": "wide concrete steps with rough texture in deep cool blue",
                "layer": "foreground",
                "region_hint": {
                    "region_id": "foreground_element",
                    "sam_class_hint": None,
                    "bbox_relative": (0.0, 0.7, 1.0, 1.0),
                    "layer_depth": "foreground",
                },
            },
            "time_of_day": {
                "prompt_fragment": "bright overcast midday",
                "layer": "atmosphere",
                "region_hint": {"region_id": "time_of_day", "layer_depth": "atmosphere"},
            },
            "weather": {
                "prompt_fragment": "clear crisp atmosphere",
                "layer": "atmosphere",
                "region_hint": {"region_id": "weather", "layer_depth": "atmosphere"},
            },
        },
    }


# ─── face_prompt tier order ─────────────────────────────────────────────


def test_face_starts_with_anchor(subject):
    face = build_face_prompt(subject)
    assert face.startswith("young woman, ")


def test_face_has_eye_boost(subject):
    face = build_face_prompt(subject, eye_boost=1.1)
    assert "(bright green almond eyes:1.10)" in face


def test_face_skin_tags_before_features(subject):
    face = build_face_prompt(subject)
    # ethnicity comes after anchor, before eyes
    a = face.index("east asian")
    b = face.index("bright green almond eyes")
    assert a < b


def test_face_includes_quality_tag(subject):
    face = build_face_prompt(subject, include_quality=True)
    assert FACE_QUALITY_TAGS in face


def test_face_quality_tag_optional(subject):
    face = build_face_prompt(subject, include_quality=False)
    assert FACE_QUALITY_TAGS not in face


def test_face_has_no_outfit_garment_words(subject, outfit):
    face = build_face_prompt(subject)
    # face must not leak outfit fragments
    assert "blazer" not in face
    assert "skirt" not in face


# ─── body_prompt tier order ─────────────────────────────────────────────


def test_body_includes_hands_quality(subject):
    body = build_body_prompt(subject)
    assert DEFAULT_HANDS_FRAGMENT in body


def test_body_includes_full_hair(subject):
    body = build_body_prompt(subject)
    assert "long straight, center-parted" in body
    assert "waist-length" in body


def test_body_quality_tag_present(subject):
    body = build_body_prompt(subject)
    assert BODY_QUALITY_TAGS in body


def test_body_has_no_garment_words(subject, outfit):
    body = build_body_prompt(subject)
    assert "blazer" not in body
    assert "skirt" not in body


# ─── outfit_prompt tier order (head→toe) ────────────────────────────────


def test_outfit_spatial_order(outfit):
    s = build_outfit_prompt(outfit)
    head_pos = s.index("beret")
    upper_pos = s.index("blazer")
    lower_pos = s.index("pencil skirt")
    foot_pos = s.index("pumps")
    assert head_pos < upper_pos < lower_pos < foot_pos


def test_outfit_appends_formality(outfit):
    s = build_outfit_prompt(outfit)
    assert s.endswith("formal style")


def test_outfit_handles_empty():
    assert build_outfit_prompt({}) == ""
    assert build_outfit_prompt({"garments": {}}) == ""


# ─── location_prompt tier order ─────────────────────────────────────────


def test_location_layer_order(location):
    s = build_location_prompt(location)
    bg_pos = s.index("monolithic concrete facade")
    fg_pos = s.index("wide concrete steps")
    tod_pos = s.index("bright overcast midday")
    weather_pos = s.index("clear crisp atmosphere")
    assert bg_pos < fg_pos < tod_pos < weather_pos


def test_location_handles_empty():
    assert build_location_prompt({}) == ""
    assert build_location_prompt({"elements": {}}) == ""


# ─── region map ─────────────────────────────────────────────────────────


def test_region_map_combines_outfit_and_location(outfit, location):
    rm = build_region_map(outfit, location)
    region_ids = [r["region_id"] for r in rm]
    assert "headwear" in region_ids
    assert "upper_body" in region_ids
    assert "background" in region_ids
    assert "foreground_element" in region_ids


def test_region_map_carries_bbox_and_class(outfit, location):
    rm = build_region_map(outfit, location)
    by_id = {r["region_id"]: r for r in rm}
    assert by_id["upper_body"]["sam_class_hint"] == "upper_clothes"
    assert tuple(by_id["upper_body"]["bbox_relative"]) == (0.20, 0.15, 0.80, 0.55)
    assert by_id["background"]["layer_depth"] == "background"


# ─── SAM class lookup ───────────────────────────────────────────────────


def test_sam_class_lookup_routes_outfit(outfit, location):
    face = "young woman, sharp focus"
    lookup = build_sam_class_lookup(face, outfit, location)
    assert lookup["face"] == face
    assert lookup["upper_clothes"] == outfit["garments"]["upper_body"]["prompt_fragment"]
    assert lookup["skirt"] == outfit["garments"]["lower_body"]["prompt_fragment"]
    assert lookup["shoes"] == outfit["garments"]["footwear"]["prompt_fragment"]


# ─── full assemble ──────────────────────────────────────────────────────


def test_assemble_returns_all_keys(subject, outfit, location):
    result = assemble_structured(outfit=outfit, location=location, subject=subject)
    for key in ("face", "body", "outfit", "location", "region_map", "sam_class_lookup"):
        assert key in result


def test_assemble_subject_from_json_string(subject, outfit, location):
    js = json.dumps(subject)
    result = assemble_structured(outfit=outfit, location=location, subject=js)
    assert result["face"].startswith("young woman, ")


def test_assemble_with_prompt_dict(subject, outfit, location):
    pd = {
        "subjects": [subject],
        "outfits": {"subject_1": outfit},
        "location": location,
    }
    result = assemble_structured(prompt_dict=pd)
    assert result["face"].startswith("young woman, ")
    assert "blazer" in result["outfit"]
    assert "concrete" in result["location"]


def test_assemble_direct_args_override_prompt_dict(subject, outfit, location):
    pd = {"subjects": [{"id": "x", "age_desc": "old", "gender": "man"}],
          "outfits": {"x": {"garments": {}}}, "location": {}}
    result = assemble_structured(prompt_dict=pd, outfit=outfit,
                                  location=location, subject=subject)
    # Direct args win
    assert "young woman" in result["face"]
    assert "blazer" in result["outfit"]


def test_assemble_quality_toggle(subject, outfit, location):
    on = assemble_structured(outfit=outfit, location=location, subject=subject,
                              include_quality=True)
    off = assemble_structured(outfit=outfit, location=location, subject=subject,
                               include_quality=False)
    assert FACE_QUALITY_TAGS in on["face"]
    assert FACE_QUALITY_TAGS not in off["face"]


# ─── Node wrappers ──────────────────────────────────────────────────────


def test_assembler_node_returns_six_outputs(subject, outfit, location):
    from nodes.smp.structured_assembler import FVM_SMP_StructuredPromptAssembler
    out = FVM_SMP_StructuredPromptAssembler().assemble(
        include_quality=True, face_eye_boost=1.1, subject_index=0,
        outfit_dict=outfit, location_dict=location,
        subject_json=json.dumps(subject),
    )
    assert len(out) == 6
    face, body, outfit_p, loc_p, region_map, structured = out
    assert isinstance(face, str) and face
    assert isinstance(body, str) and body
    assert isinstance(outfit_p, str) and outfit_p
    assert isinstance(loc_p, str) and loc_p
    assert isinstance(region_map, list)
    assert isinstance(structured, dict)
    assert structured["face"] == face


def test_assembler_node_metadata():
    from nodes.smp.structured_assembler import FVM_SMP_StructuredPromptAssembler as N
    assert N.CATEGORY.startswith("FVM Tools/SMP")
    assert N.RETURN_TYPES == ("STRING", "STRING", "STRING", "STRING",
                              "REGION_MAP", "STRUCTURED_PROMPTS")


def test_router_returns_matching_fragment(subject, outfit, location):
    from nodes.smp.structured_assembler import FVM_SMP_StructuredPromptAssembler
    from nodes.smp.sam_class_router import FVM_SMP_SAMClassRouter
    _, _, _, _, _, structured = FVM_SMP_StructuredPromptAssembler().assemble(
        include_quality=True, face_eye_boost=1.1, subject_index=0,
        outfit_dict=outfit, location_dict=location,
        subject_json=json.dumps(subject),
    )
    out = FVM_SMP_SAMClassRouter().route(structured, "upper_clothes", fallback="")
    assert out[0] == outfit["garments"]["upper_body"]["prompt_fragment"]


def test_router_falls_back_when_unknown(subject, outfit, location):
    from nodes.smp.structured_assembler import FVM_SMP_StructuredPromptAssembler
    from nodes.smp.sam_class_router import FVM_SMP_SAMClassRouter
    _, _, _, _, _, structured = FVM_SMP_StructuredPromptAssembler().assemble(
        include_quality=True, face_eye_boost=1.1, subject_index=0,
        outfit_dict=outfit, location_dict=location,
        subject_json=json.dumps(subject),
    )
    out = FVM_SMP_SAMClassRouter().route(structured, "nonexistent_class",
                                          fallback="<fallback>")
    assert out[0] == "<fallback>"


def test_router_metadata():
    from nodes.smp.sam_class_router import FVM_SMP_SAMClassRouter as N
    assert N.CATEGORY.startswith("FVM Tools/SMP")
    assert N.RETURN_TYPES == ("STRING",)


# ─── End-to-end snapshot ────────────────────────────────────────────────


def test_e2e_smp_outfit_combiner_into_assembler():
    """SMP_OutfitGenerator → SMP_OutfitCombiner → Assembler must produce
    a fully-resolved outfit_prompt with no leftover #tokens# anywhere."""
    from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator
    from nodes.smp.color_generator import FVM_SMP_ColorGenerator
    from nodes.smp.outfit_combiner import FVM_SMP_OutfitCombiner
    from nodes.smp.location_generator import FVM_SMP_LocationGenerator
    from nodes.smp.location_combiner import FVM_SMP_LocationCombiner
    from nodes.smp.structured_assembler import FVM_SMP_StructuredPromptAssembler

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

    subject = {
        "id": "subject_1", "age_desc": "young", "gender": "woman",
        "skin_tags": ["smooth skin"],
        "eye_desc": "bright eyes",
        "expression": "calm",
        "hair_color_length": "dark hair",
    }
    face, body, outfit_p, loc_p, region_map, structured = (
        FVM_SMP_StructuredPromptAssembler().assemble(
            include_quality=True, face_eye_boost=1.1, subject_index=0,
            outfit_dict=outfit_dict, location_dict=location_dict,
            subject_json=json.dumps(subject),
        )
    )
    for s in (face, body, outfit_p, loc_p):
        assert "#" not in s, f"unresolved token in: {s!r}"
    # Region map carries entries from both outfit and location
    region_ids = {r["region_id"] for r in region_map}
    assert region_ids & {"upper_body", "lower_body", "footwear"}
    assert "background" in region_ids
