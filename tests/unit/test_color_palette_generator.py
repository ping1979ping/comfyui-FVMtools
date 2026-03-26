import pytest
import torch
from nodes.color_palette_generator import FVM_ColorPaletteGenerator


class TestSchema:

    def test_category(self):
        assert FVM_ColorPaletteGenerator.CATEGORY == "FVM Tools/Color"

    def test_function(self):
        assert FVM_ColorPaletteGenerator.FUNCTION == "generate"

    def test_return_types_count(self):
        assert len(FVM_ColorPaletteGenerator.RETURN_TYPES) == 16

    def test_return_names_count(self):
        assert len(FVM_ColorPaletteGenerator.RETURN_NAMES) == 16

    def test_return_types_names_match(self):
        assert len(FVM_ColorPaletteGenerator.RETURN_TYPES) == len(FVM_ColorPaletteGenerator.RETURN_NAMES)

    def test_input_types_structure(self):
        inputs = FVM_ColorPaletteGenerator.INPUT_TYPES()
        assert "required" in inputs
        req = inputs["required"]
        for key in ("seed", "num_colors", "harmony_type", "style_preset",
                     "vibrancy", "contrast", "warmth", "neutral_ratio",
                     "include_metallics"):
            assert key in req, f"Missing required input: {key}"

    def test_optional_inputs(self):
        inputs = FVM_ColorPaletteGenerator.INPUT_TYPES()
        assert "optional" in inputs
        opt = inputs["optional"]
        for key in ("palette_source", "wildcard_file", "palette_index"):
            assert key in opt, f"Missing optional input: {key}"


class TestExecute:

    def test_returns_tuple(self):
        node = FVM_ColorPaletteGenerator()
        result = node.generate(seed=42, num_colors=5, harmony_type="auto",
                               style_preset="general", vibrancy=0.5, contrast=0.5,
                               warmth=0.5, neutral_ratio=0.4, include_metallics=True)
        assert isinstance(result, tuple)
        assert len(result) == 16

    def test_palette_preview_is_image_tensor(self):
        node = FVM_ColorPaletteGenerator()
        result = node.generate(seed=42, num_colors=5, harmony_type="auto",
                               style_preset="general", vibrancy=0.5, contrast=0.5,
                               warmth=0.5, neutral_ratio=0.4, include_metallics=True)
        preview = result[14]  # palette_preview
        assert isinstance(preview, torch.Tensor)
        assert preview.ndim == 4
        assert preview.shape[0] == 1
        assert preview.shape[3] == 3
        assert preview.dtype == torch.float32

    def test_all_string_outputs_are_strings(self):
        node = FVM_ColorPaletteGenerator()
        result = node.generate(seed=42, num_colors=5, harmony_type="auto",
                               style_preset="general", vibrancy=0.5, contrast=0.5,
                               warmth=0.5, neutral_ratio=0.4, include_metallics=True)
        # Indices 0-13 and 15 are strings, 14 is IMAGE
        for i in list(range(14)) + [15]:
            assert isinstance(result[i], str), f"Output {i} is {type(result[i])}, expected str"

    def test_empty_color_slots(self):
        node = FVM_ColorPaletteGenerator()
        result = node.generate(seed=42, num_colors=3, harmony_type="auto",
                               style_preset="general", vibrancy=0.5, contrast=0.5,
                               warmth=0.5, neutral_ratio=0.3, include_metallics=True)
        # color_4 through color_8 should be empty strings (indices 4-8)
        for i in range(4, 9):  # color_4=idx4, color_5=idx5, ..., color_8=idx8
            assert result[i] == "", f"Slot {i} should be empty, got '{result[i]}'"

    def test_palette_string_non_empty(self):
        node = FVM_ColorPaletteGenerator()
        result = node.generate(seed=42, num_colors=5, harmony_type="auto",
                               style_preset="general", vibrancy=0.5, contrast=0.5,
                               warmth=0.5, neutral_ratio=0.4, include_metallics=True)
        assert len(result[0]) > 0  # palette_string

    def test_from_file_mode(self):
        node = FVM_ColorPaletteGenerator()
        wildcard = "red, blue, green, white, gold\nnavy-blue, cream, rose"
        result = node.generate(seed=0, num_colors=5, harmony_type="auto",
                               style_preset="general", vibrancy=0.5, contrast=0.5,
                               warmth=0.5, neutral_ratio=0.4, include_metallics=True,
                               palette_source="from_file", wildcard_file=wildcard,
                               palette_index=0)
        assert isinstance(result, tuple)
        assert len(result) == 16
        # First line has 5 colors
        assert "red" in result[0]

    def test_generate_method_exists(self):
        node = FVM_ColorPaletteGenerator()
        assert hasattr(node, "generate")
        assert callable(node.generate)
