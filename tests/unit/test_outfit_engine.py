"""Unit tests for core/outfit_engine.py — main outfit generation logic."""

import pytest
from core.outfit_engine import generate_outfit, SLOT_ORDER, DEFAULT_COLOR_TAGS, INVISIBLE_FABRICS
from core.outfit_presets import OUTFIT_PRESETS


class TestDeterminism:
    """Seed determinism tests."""

    def test_same_seed_same_output(self):
        r1 = generate_outfit(seed=42, outfit_set="general_female")
        r2 = generate_outfit(seed=42, outfit_set="general_female")
        assert r1["outfit_prompt"] == r2["outfit_prompt"]
        assert r1["outfit_details"] == r2["outfit_details"]

    def test_different_seeds_different_output(self):
        r1 = generate_outfit(seed=42, outfit_set="general_female")
        r2 = generate_outfit(seed=9999, outfit_set="general_female")
        # Very unlikely to be identical with different seeds
        assert r1["outfit_prompt"] != r2["outfit_prompt"]


class TestOutputStructure:
    """Tests for output dict structure."""

    def test_output_has_required_keys(self):
        result = generate_outfit(seed=1, outfit_set="general_female")
        assert "outfit_prompt" in result
        assert "outfit_details" in result
        assert "outfit_info" in result

    def test_output_values_are_strings(self):
        result = generate_outfit(seed=1, outfit_set="general_female")
        assert isinstance(result["outfit_prompt"], str)
        assert isinstance(result["outfit_details"], str)
        assert isinstance(result["outfit_info"], str)

    def test_prompt_starts_with_prefix(self):
        result = generate_outfit(seed=1, outfit_set="general_female", prefix="wearing ")
        assert result["outfit_prompt"].startswith("wearing ")

    def test_custom_prefix(self):
        result = generate_outfit(seed=1, outfit_set="general_female", prefix="dressed in ")
        assert result["outfit_prompt"].startswith("dressed in ")

    def test_prompt_contains_color_tags(self):
        result = generate_outfit(seed=1, outfit_set="general_female")
        prompt = result["outfit_prompt"]
        # Should contain at least one color tag
        has_tag = any(tag in prompt for tag in DEFAULT_COLOR_TAGS.values())
        assert has_tag, f"No color tags found in prompt: {prompt}"

    def test_details_has_structured_format(self):
        result = generate_outfit(seed=1, outfit_set="general_female")
        details = result["outfit_details"]
        # Details are pipe-separated, each piece is colon-separated
        assert "|" in details or ":" in details


class TestSlotEnables:
    """Tests for slot enable/disable behavior."""

    def test_enabled_slots_appear(self):
        enables = {s: True for s in SLOT_ORDER}
        result = generate_outfit(seed=42, outfit_set="general_female", slot_enables=enables, coverage=1.0)
        # top, bottom, footwear are always active when enabled
        details = result["outfit_details"]
        assert "top:" in details
        assert "bottom:" in details
        assert "footwear:" in details

    def test_disabled_slots_dont_appear(self):
        enables = {s: False for s in SLOT_ORDER}
        enables["top"] = True  # only top enabled
        result = generate_outfit(seed=42, outfit_set="general_female", slot_enables=enables)
        details = result["outfit_details"]
        assert "top:" in details
        assert "bottom:" not in details
        assert "headwear:" not in details

    def test_disabled_slots_dont_affect_determinism(self):
        """Disabling a slot should not change generation for other slots
        (rng is consumed for all slots regardless)."""
        enables_all = {s: True for s in SLOT_ORDER}
        enables_some = {s: True for s in SLOT_ORDER}
        enables_some["headwear"] = False

        r1 = generate_outfit(seed=42, outfit_set="general_female", slot_enables=enables_all, coverage=1.0)
        r2 = generate_outfit(seed=42, outfit_set="general_female", slot_enables=enables_some, coverage=1.0)

        # top should be same in both since rng is consumed identically
        d1 = {p.split(":")[0]: p for p in r1["outfit_details"].split("|")}
        d2 = {p.split(":")[0]: p for p in r2["outfit_details"].split("|")}
        assert d1.get("top") == d2.get("top")


class TestFormality:
    """Tests for formality parameter."""

    def test_low_formality_produces_casual(self):
        result = generate_outfit(seed=42, outfit_set="general_female", formality=0.0)
        # Should work without error
        assert result["outfit_prompt"]

    def test_high_formality_produces_formal(self):
        result = generate_outfit(seed=42, outfit_set="general_female", formality=1.0)
        assert result["outfit_prompt"]

    def test_formality_clamped_by_preset(self):
        # evening_gala has formality_range (0.7, 1.0)
        result = generate_outfit(seed=42, outfit_set="general_female", style_preset="evening_gala", formality=0.0)
        # Info should show effective formality at 0.7 (clamped)
        assert "0.70" in result["outfit_info"]


class TestOverrides:
    """Tests for override functionality."""

    def test_override_string_overrides_slot(self):
        overrides = {"top": {"garment": "blouse", "fabric": "silk",
                             "color_tag": "#primary#", "mode": "override"}}
        result = generate_outfit(seed=42, outfit_set="general_female", overrides=overrides)
        assert "blouse" in result["outfit_prompt"]
        assert "silk" in result["outfit_prompt"]

    def test_exclude_override_skips_slot(self):
        enables = {s: True for s in SLOT_ORDER}
        overrides = {"top": {"garment": None, "fabric": None,
                             "color_tag": None, "mode": "exclude"}}
        result = generate_outfit(seed=42, outfit_set="general_female", slot_enables=enables, overrides=overrides)
        assert "top:" not in result["outfit_details"]

    def test_empty_overrides_is_fine(self):
        result = generate_outfit(seed=42, outfit_set="general_female", overrides={})
        assert result["outfit_prompt"]


class TestStylePresets:
    """Tests for all style presets."""

    @pytest.mark.parametrize("preset", sorted(OUTFIT_PRESETS.keys()))
    def test_preset_works_without_crash(self, preset):
        result = generate_outfit(seed=42, outfit_set="general_female", style_preset=preset)
        assert result["outfit_prompt"]
        assert result["outfit_details"]
        assert result["outfit_info"]


class TestInvisibleFabrics:
    """Tests for metal/plastic/rubber fabric omission."""

    def test_metal_fabric_omitted_from_prompt(self):
        """When a garment has 'metal' fabric, it should not appear in the prompt."""
        overrides = {"accessories": {"garment": "necklace", "fabric": "metal",
                                     "color_tag": "#metallic#", "mode": "override"}}
        enables = {s: False for s in SLOT_ORDER}
        enables["accessories"] = True
        result = generate_outfit(seed=42, outfit_set="general_female", slot_enables=enables, overrides=overrides)
        prompt = result["outfit_prompt"]
        assert "necklace" in prompt
        assert "metal" not in prompt.replace("#metallic#", "")

    def test_plastic_fabric_omitted_from_prompt(self):
        overrides = {"accessories": {"garment": "sunglasses", "fabric": "plastic",
                                     "color_tag": "#accent#", "mode": "override"}}
        enables = {s: False for s in SLOT_ORDER}
        enables["accessories"] = True
        result = generate_outfit(seed=42, outfit_set="general_female", slot_enables=enables, overrides=overrides)
        prompt = result["outfit_prompt"]
        assert "sunglasses" in prompt
        assert "plastic" not in prompt

    def test_rubber_fabric_omitted_from_prompt(self):
        overrides = {"footwear": {"garment": "boots", "fabric": "rubber",
                                  "color_tag": "#neutral#", "mode": "override"}}
        enables = {s: False for s in SLOT_ORDER}
        enables["footwear"] = True
        result = generate_outfit(seed=42, outfit_set="general_female", slot_enables=enables, overrides=overrides)
        prompt = result["outfit_prompt"]
        assert "boots" in prompt
        assert "rubber" not in prompt


class TestDefaultOutfitSet:
    """Tests that default outfit_set parameter works."""

    def test_default_outfit_set(self):
        """Calling without outfit_set should default to general_female."""
        r1 = generate_outfit(seed=42)
        r2 = generate_outfit(seed=42, outfit_set="general_female")
        assert r1["outfit_prompt"] == r2["outfit_prompt"]
        assert r1["outfit_details"] == r2["outfit_details"]
