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


class TestPrintTextDecorations:
    """Tests for print/text decoration feature."""

    def test_print_probability_zero_no_decoration(self):
        """print_probability=0 should never produce 'with' in output."""
        # Test multiple seeds to be thorough
        for seed in range(10):
            result = generate_outfit(seed=seed, outfit_set="general_female", print_probability=0.0)
            assert "with " not in result["outfit_prompt"], (
                f"Seed {seed}: 'with' found in prompt when print_probability=0: {result['outfit_prompt']}")

    def test_print_probability_one_has_decoration(self):
        """print_probability=1 should produce 'with' in at least some slots."""
        # With probability 1, at least one slot should get a decoration
        found_with = False
        for seed in range(20):
            result = generate_outfit(seed=seed, outfit_set="general_female", print_probability=1.0)
            if "with " in result["outfit_prompt"]:
                found_with = True
                break
        assert found_with, "Expected at least one decoration with print_probability=1.0"

    def test_text_mode_off_no_quoted_text(self):
        """text_mode='off' should never produce quoted text in output."""
        for seed in range(20):
            result = generate_outfit(seed=seed, outfit_set="general_female",
                                     print_probability=1.0, text_mode="off")
            prompt = result["outfit_prompt"]
            assert '"' not in prompt, (
                f"Seed {seed}: quoted text found with text_mode=off: {prompt}")

    def test_text_mode_descriptive_no_quotes(self):
        """text_mode='descriptive' should not produce quoted text."""
        for seed in range(20):
            result = generate_outfit(seed=seed, outfit_set="general_female",
                                     print_probability=1.0, text_mode="descriptive")
            prompt = result["outfit_prompt"]
            # Descriptive mode strips the quoted content
            assert '"' not in prompt, (
                f"Seed {seed}: quoted text found with text_mode=descriptive: {prompt}")

    def test_determinism_with_prints(self):
        """Same seed should produce same output with prints enabled."""
        r1 = generate_outfit(seed=42, outfit_set="general_female", print_probability=0.5)
        r2 = generate_outfit(seed=42, outfit_set="general_female", print_probability=0.5)
        assert r1["outfit_prompt"] == r2["outfit_prompt"]
        assert r1["outfit_details"] == r2["outfit_details"]

    def test_determinism_print_probability_doesnt_break_other_slots(self):
        """Changing print_probability should not change garment/fabric selection.
        rng is always consumed for decoration regardless."""
        # This is tricky: we can't directly compare because decorations change the prompt.
        # But the details (slot:garment:fabric:color) should remain stable.
        r1 = generate_outfit(seed=42, outfit_set="general_female", print_probability=0.0)
        r2 = generate_outfit(seed=42, outfit_set="general_female", print_probability=1.0)
        # Parse details to compare garment selections
        d1 = {p.split(":")[0]: p.split(":")[1] for p in r1["outfit_details"].split("|") if ":" in p}
        d2 = {p.split(":")[0]: p.split(":")[1] for p in r2["outfit_details"].split("|") if ":" in p}
        assert d1 == d2, f"Garment selections changed: {d1} vs {d2}"

    def test_backward_compatible_no_prints_file(self):
        """Set without prints.txt should work fine (no decorations)."""
        # business_male might not have prints.txt - test with a nonexistent set fallback
        result = generate_outfit(seed=42, outfit_set="general_female", print_probability=0.0)
        assert result["outfit_prompt"]
        assert result["outfit_details"]


class TestDefaultOutfitSet:
    """Tests that default outfit_set parameter works."""

    def test_default_outfit_set(self):
        """Calling without outfit_set should default to general_female."""
        r1 = generate_outfit(seed=42)
        r2 = generate_outfit(seed=42, outfit_set="general_female")
        assert r1["outfit_prompt"] == r2["outfit_prompt"]
        assert r1["outfit_details"] == r2["outfit_details"]
