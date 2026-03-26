import pytest
import torch

from nodes.palette_from_image import FVM_PaletteFromImage


# ──── Fixtures ────

@pytest.fixture
def solid_green_image():
    img = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
    img[:, :, :, 1] = 1.0  # pure green
    return img


@pytest.fixture
def batch_image():
    """Batch of 2 images — node should use first."""
    return torch.rand(2, 64, 64, 3, dtype=torch.float32)


# ──── Schema Tests ────

class TestPaletteFromImageSchema:
    """Verify node class attributes and schema."""

    def test_category(self):
        assert FVM_PaletteFromImage.CATEGORY == "FVM Tools/Color"

    def test_function(self):
        assert FVM_PaletteFromImage.FUNCTION == "extract"

    def test_return_types_count(self):
        assert len(FVM_PaletteFromImage.RETURN_TYPES) == 17

    def test_return_names_count(self):
        assert len(FVM_PaletteFromImage.RETURN_NAMES) == 17

    def test_return_types_matches_names(self):
        assert len(FVM_PaletteFromImage.RETURN_TYPES) == len(FVM_PaletteFromImage.RETURN_NAMES)

    def test_input_types_structure(self):
        inputs = FVM_PaletteFromImage.INPUT_TYPES()
        assert "required" in inputs
        required = inputs["required"]
        assert "image" in required
        assert "num_colors" in required
        assert "extraction_mode" in required
        assert "ignore_background" in required
        assert "ignore_skin" in required
        assert "sample_region" in required
        assert "saturation_threshold" in required
        assert "include_neutrals" in required
        assert "include_metallics" in required
        assert "seed" in required

    def test_function_exists(self):
        node = FVM_PaletteFromImage()
        assert hasattr(node, "extract")
        assert callable(node.extract)


# ──── Execute Tests ────

class TestPaletteFromImageExecute:
    """Test node execution."""

    def test_returns_correct_length_tuple(self, solid_green_image):
        node = FVM_PaletteFromImage()
        result = node.extract(
            image=solid_green_image, num_colors=3, extraction_mode="dominant",
            ignore_background=False, ignore_skin=False, sample_region="full",
            saturation_threshold=0.1, include_neutrals=True,
            include_metallics=True, seed=42,
        )
        assert isinstance(result, tuple)
        assert len(result) == 17

    def test_palette_preview_is_image_tensor(self, solid_green_image):
        node = FVM_PaletteFromImage()
        result = node.extract(
            image=solid_green_image, num_colors=3, extraction_mode="dominant",
            ignore_background=False, ignore_skin=False, sample_region="full",
            saturation_threshold=0.1, include_neutrals=True,
            include_metallics=True, seed=42,
        )
        preview = result[14]  # palette_preview
        assert preview.ndim == 4
        assert preview.shape[0] == 1
        assert preview.shape[3] == 3
        assert preview.dtype == torch.float32

    def test_source_annotated_is_image_tensor(self, solid_green_image):
        node = FVM_PaletteFromImage()
        result = node.extract(
            image=solid_green_image, num_colors=3, extraction_mode="dominant",
            ignore_background=False, ignore_skin=False, sample_region="full",
            saturation_threshold=0.1, include_neutrals=True,
            include_metallics=True, seed=42,
        )
        annotated = result[15]  # source_annotated
        assert annotated.ndim == 4
        assert annotated.shape[0] == 1
        assert annotated.shape[3] == 3
        assert annotated.dtype == torch.float32

    def test_unused_color_slots_are_empty(self, solid_green_image):
        node = FVM_PaletteFromImage()
        result = node.extract(
            image=solid_green_image, num_colors=2, extraction_mode="dominant",
            ignore_background=False, ignore_skin=False, sample_region="full",
            saturation_threshold=0.1, include_neutrals=True,
            include_metallics=True, seed=42,
        )
        # color_3 through color_8 should be empty (indices 3-8)
        for i in range(3, 9):
            assert result[i] == "", f"color_{i-1+1} should be empty but is '{result[i]}'"

    def test_batch_input_uses_first(self, batch_image):
        node = FVM_PaletteFromImage()
        result = node.extract(
            image=batch_image, num_colors=3, extraction_mode="dominant",
            ignore_background=False, ignore_skin=False, sample_region="full",
            saturation_threshold=0.1, include_neutrals=True,
            include_metallics=True, seed=42,
        )
        assert isinstance(result, tuple)
        assert len(result) == 17

    def test_palette_string_is_first_output(self, solid_green_image):
        node = FVM_PaletteFromImage()
        result = node.extract(
            image=solid_green_image, num_colors=3, extraction_mode="dominant",
            ignore_background=False, ignore_skin=False, sample_region="full",
            saturation_threshold=0.1, include_neutrals=True,
            include_metallics=True, seed=42,
        )
        palette_string = result[0]
        assert isinstance(palette_string, str)
        assert len(palette_string) > 0

    def test_info_is_last_string_output(self, solid_green_image):
        node = FVM_PaletteFromImage()
        result = node.extract(
            image=solid_green_image, num_colors=3, extraction_mode="dominant",
            ignore_background=False, ignore_skin=False, sample_region="full",
            saturation_threshold=0.1, include_neutrals=True,
            include_metallics=True, seed=42,
        )
        info = result[16]  # palette_info
        assert isinstance(info, str)
        assert "Extracted" in info
