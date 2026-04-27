"""P1 — schema round-trip + defaults tests for SMP."""

import pytest

from core.smp.schema import (
    ColorPalette,
    GarmentEntry,
    LocationDict,
    LocationElement,
    Meta,
    OutfitDict,
    PromptDict,
    RegionEntry,
    StructuredPrompts,
    Subject,
    SCHEMA_VERSION,
    validate_prompt_dict,
)
from core.smp.defaults import (
    ATMOSPHERE_TOKEN_MAP,
    DEFAULT_COLOR_ROLE_BY_SLOT,
    DEFAULT_LOCATION_LAYERS,
    DEFAULT_PERSON_REGIONS,
    GARMENT_TOKEN_MAP,
)
from core.smp import types as smp_types


# ─── Models ─────────────────────────────────────────────────────────────


def test_meta_defaults():
    m = Meta()
    assert m.schema_version == "1.0"
    assert m.target_model == "generic"
    assert m.seed == 0


def test_subject_minimal():
    s = Subject(id="s1", age_desc="young", gender="woman")
    d = s.model_dump()
    assert d["id"] == "s1"
    assert d["visibility"] == "full"


def test_garment_entry_roundtrip():
    g = GarmentEntry(
        name="fitted blazer",
        probability=0.6,
        coverage=0.78,
        fabric="wool blend",
        color_role="primary",
        prompt_fragment="fitted wool blend blazer in #primary#",
    )
    payload = g.model_dump()
    g2 = GarmentEntry.model_validate(payload)
    assert g2 == g


def test_outfit_dict_roundtrip():
    o = OutfitDict(
        set_name="business_female",
        seed=42,
        formality="formal",
        coverage_target=0.75,
        garments={"top": GarmentEntry(name="blazer", color_role="primary")},
    )
    p = o.model_dump()
    assert p["set_name"] == "business_female"
    assert "top" in p["garments"]


def test_location_dict_roundtrip():
    e = LocationElement(name="concrete wall", coverage=0.7, layer="background")
    loc = LocationDict(set_name="urban_brutalist", seed=1, elements={"background": e})
    p = loc.model_dump()
    assert p["elements"]["background"]["layer"] == "background"


def test_color_palette_tokens():
    cp = ColorPalette(
        seed=7,
        style="warm earthy",
        garment_colors={"primary": "burgundy", "secondary": "charcoal"},
        atmosphere_colors={"ambient_light": "warm amber"},
        raw_tokens={
            "#primary#": "burgundy",
            "#secondary#": "charcoal",
            "#ambient_light#": "warm amber",
        },
    )
    assert cp.raw_tokens["#primary#"] == "burgundy"


def test_region_entry_bbox():
    r = RegionEntry(
        region_id="upper_body",
        sam_class_hint="upper_clothes",
        bbox_relative=(0.2, 0.15, 0.8, 0.55),
    )
    payload = r.model_dump()
    assert tuple(payload["bbox_relative"]) == (0.2, 0.15, 0.8, 0.55)


def test_structured_prompts_default():
    sp = StructuredPrompts()
    assert sp.face == "" and sp.outfit == ""
    assert sp.region_map == [] and sp.sam_class_lookup == {}


def test_prompt_dict_full():
    pd = PromptDict(
        meta=Meta(seed=42, target_model="z-image-turbo", label="demo"),
        subjects=[Subject(id="s1", age_desc="young", gender="woman")],
        outfits={
            "s1": OutfitDict(
                set_name="business_female",
                garments={"top": GarmentEntry(name="blazer", color_role="primary")},
            )
        },
        location=LocationDict(set_name="urban_brutalist"),
        palette=ColorPalette(garment_colors={"primary": "burgundy"}),
    )
    payload = pd.model_dump()
    assert payload["meta"]["target_model"] == "z-image-turbo"
    pd2 = validate_prompt_dict(payload)
    assert pd2.meta.label == "demo"


def test_validate_prompt_dict_idempotent():
    pd = PromptDict()
    assert validate_prompt_dict(pd) is pd


def test_validate_prompt_dict_rejects_invalid():
    with pytest.raises(TypeError):
        validate_prompt_dict("not-a-dict")


def test_extras_extra_allow():
    """Models allow extra fields so SMP can carry through unknown user keys."""
    o = OutfitDict.model_validate({
        "set_name": "x",
        "garments": {},
        "future_field": "preserved",
    })
    payload = o.model_dump()
    assert payload.get("future_field") == "preserved"


def test_schema_version_constant():
    assert SCHEMA_VERSION == "1.0"


# ─── Defaults ─────────────────────────────────────────────────────────────


def test_default_person_regions_keys():
    needed = {"face", "top", "bottom", "footwear", "headwear"}
    assert needed.issubset(DEFAULT_PERSON_REGIONS.keys())


def test_default_person_regions_bbox_in_unit_square():
    for region, info in DEFAULT_PERSON_REGIONS.items():
        bbox = info["bbox"]
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        assert 0.0 <= x0 <= 1.0 and 0.0 <= x1 <= 1.0
        assert 0.0 <= y0 <= 1.0 and 0.0 <= y1 <= 1.0
        assert x0 < x1 and y0 < y1, f"{region} bbox invalid"


def test_default_location_layers_present():
    assert "background" in DEFAULT_LOCATION_LAYERS
    assert "foreground_element" in DEFAULT_LOCATION_LAYERS
    assert DEFAULT_LOCATION_LAYERS["time_of_day"]["layer_depth"] == "atmosphere"


def test_token_maps_consistent():
    """Each token in the maps points at a string role/key, hash-quoted."""
    for token, role in GARMENT_TOKEN_MAP.items():
        assert token.startswith("#") and token.endswith("#")
        assert role.isalpha()
    for token, key in ATMOSPHERE_TOKEN_MAP.items():
        assert token.startswith("#") and token.endswith("#")
        assert "_" in key or key.isalpha()


def test_color_role_by_slot_subset_of_garment_tokens():
    roles_in_map = set(GARMENT_TOKEN_MAP.values())
    for slot, role in DEFAULT_COLOR_ROLE_BY_SLOT.items():
        assert role in roles_in_map, f"slot {slot} → role {role} missing in token map"


# ─── Type identifiers ────────────────────────────────────────────────────


def test_smp_types_unique():
    assert len(smp_types.ALL_SMP_TYPES) == len(set(smp_types.ALL_SMP_TYPES))


def test_smp_types_uppercase():
    for t in smp_types.ALL_SMP_TYPES:
        assert t.isupper(), f"{t} should be UPPERCASE"
