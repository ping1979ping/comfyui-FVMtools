"""Unit tests for nodes/outfit_generator.py — ComfyUI node schema and execution."""

import pytest
from nodes.outfit_generator import FVM_OutfitGenerator


class TestSchema:
    """Tests for node class attributes and schema."""

    def test_category(self):
        assert FVM_OutfitGenerator.CATEGORY == "FVM Tools/Fashion"

    def test_function(self):
        assert FVM_OutfitGenerator.FUNCTION == "generate"

    def test_return_types(self):
        assert FVM_OutfitGenerator.RETURN_TYPES == ("STRING", "STRING", "STRING")

    def test_return_names(self):
        assert FVM_OutfitGenerator.RETURN_NAMES == ("outfit_prompt", "outfit_details", "outfit_info")

    def test_input_types_has_required(self):
        inputs = FVM_OutfitGenerator.INPUT_TYPES()
        assert "required" in inputs

    def test_input_types_has_all_expected_inputs(self):
        inputs = FVM_OutfitGenerator.INPUT_TYPES()
        required = inputs["required"]
        assert "outfit_set" in required
        assert "seed" in required
        assert "style_preset" in required
        assert "formality" in required
        assert "coverage" in required

    def test_outfit_set_is_first_input(self):
        inputs = FVM_OutfitGenerator.INPUT_TYPES()
        required_keys = list(inputs["required"].keys())
        assert required_keys[0] == "outfit_set"

    def test_outfit_set_contains_general_female(self):
        inputs = FVM_OutfitGenerator.INPUT_TYPES()
        outfit_sets = inputs["required"]["outfit_set"][0]
        assert "general_female" in outfit_sets

    def test_all_enable_checkboxes_exist(self):
        inputs = FVM_OutfitGenerator.INPUT_TYPES()
        required = inputs["required"]
        for slot in ["headwear", "top", "outerwear", "bottom",
                     "footwear", "accessories", "bag"]:
            key = f"enable_{slot}"
            assert key in required, f"Missing checkbox: {key}"
            assert required[key][0] == "BOOLEAN"

    def test_optional_inputs_exist(self):
        inputs = FVM_OutfitGenerator.INPUT_TYPES()
        assert "optional" in inputs
        optional = inputs["optional"]
        assert "override_string" in optional
        assert "prefix" in optional
        assert "separator" in optional


class TestExecution:
    """Tests for node execution."""

    def test_returns_tuple_of_3_strings(self):
        node = FVM_OutfitGenerator()
        result = node.generate(
            outfit_set="general_female",
            seed=42, style_preset="general", formality=0.5, coverage=0.5,
            enable_headwear=False, enable_top=True, enable_outerwear=False,
            enable_bottom=True, enable_footwear=True, enable_accessories=False,
            enable_bag=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, str)

    def test_default_execution(self):
        node = FVM_OutfitGenerator()
        result = node.generate(
            outfit_set="general_female",
            seed=0, style_preset="general", formality=0.5, coverage=0.5,
            enable_headwear=False, enable_top=True, enable_outerwear=False,
            enable_bottom=True, enable_footwear=True, enable_accessories=False,
            enable_bag=False,
        )
        prompt, details, info = result
        assert prompt.startswith("wearing ")
        assert len(details) > 0
        assert "Seed: 0" in info

    def test_with_override_string(self):
        node = FVM_OutfitGenerator()
        result = node.generate(
            outfit_set="general_female",
            seed=42, style_preset="general", formality=0.5, coverage=0.5,
            enable_headwear=False, enable_top=True, enable_outerwear=False,
            enable_bottom=True, enable_footwear=True, enable_accessories=False,
            enable_bag=False,
            override_string="top: silk blouse | #primary#",
        )
        prompt = result[0]
        assert "blouse" in prompt

    def test_all_slots_enabled(self):
        node = FVM_OutfitGenerator()
        result = node.generate(
            outfit_set="general_female",
            seed=42, style_preset="general", formality=0.5, coverage=1.0,
            enable_headwear=True, enable_top=True, enable_outerwear=True,
            enable_bottom=True, enable_footwear=True, enable_accessories=True,
            enable_bag=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
