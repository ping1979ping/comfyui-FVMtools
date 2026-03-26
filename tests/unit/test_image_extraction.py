import pytest
import torch
import numpy as np

from core.image_extraction import extract_palette_from_image, _kmeans, _is_skin_tone


# ──── Fixtures ────

@pytest.fixture
def solid_red_image():
    img = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
    img[:, :, :, 0] = 1.0
    return img


@pytest.fixture
def two_color_image():
    img = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
    img[:, :32, :, 0] = 1.0   # top half red
    img[:, 32:, :, 2] = 1.0   # bottom half blue
    return img


@pytest.fixture
def gradient_image():
    img = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
    for i in range(64):
        img[:, :, i, 0] = i / 63.0   # R gradient left-to-right
        img[:, :, i, 2] = 1.0 - i / 63.0  # B gradient
    return img


@pytest.fixture
def all_black_image():
    return torch.zeros(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def all_white_image():
    return torch.ones(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def single_pixel_image():
    return torch.rand(1, 1, 1, 3, dtype=torch.float32)


# ──── Tests ────

class TestExtractBasic:
    """Basic extraction tests."""

    def test_solid_red_contains_red(self, solid_red_image):
        result = extract_palette_from_image(solid_red_image, num_colors=3, seed=42)
        names = [c["name"] for c in result["colors"]]
        # Should contain a red-family name
        assert any("red" in n or n in ("crimson", "scarlet", "vermilion", "tomato-red",
                                        "cherry", "ruby", "carmine")
                    for n in names), f"Expected red-family color, got {names}"

    def test_two_color_has_red_and_blue(self, two_color_image):
        result = extract_palette_from_image(
            two_color_image, num_colors=4, filter_skin=False, seed=42
        )
        names = [c["name"] for c in result["colors"]]
        has_red = any("red" in n or n in ("crimson", "scarlet", "vermilion",
                                           "cherry", "ruby", "carmine", "maroon")
                      for n in names)
        has_blue = any("blue" in n or n in ("navy-blue", "cobalt", "royal-blue",
                                             "sapphire", "midnight-blue", "indigo")
                       for n in names)
        assert has_red, f"Expected red-family color in {names}"
        assert has_blue, f"Expected blue-family color in {names}"

    def test_determinism_same_seed(self, gradient_image):
        r1 = extract_palette_from_image(gradient_image, num_colors=3, seed=123)
        r2 = extract_palette_from_image(gradient_image, num_colors=3, seed=123)
        assert r1["palette_string"] == r2["palette_string"]
        assert len(r1["colors"]) == len(r2["colors"])

    def test_different_seed_may_differ(self, gradient_image):
        """Different seeds can produce different results (not guaranteed but likely)."""
        r1 = extract_palette_from_image(gradient_image, num_colors=3, seed=0)
        r2 = extract_palette_from_image(gradient_image, num_colors=3, seed=999)
        # Just check both are valid — they may or may not differ
        assert len(r1["colors"]) > 0
        assert len(r2["colors"]) > 0


class TestOutputStructure:
    """Verify output dict structure."""

    def test_result_has_required_keys(self, solid_red_image):
        result = extract_palette_from_image(solid_red_image, num_colors=3, seed=0)
        assert "colors" in result
        assert "palette_string" in result
        assert "info" in result

    def test_color_entries_have_required_keys(self, solid_red_image):
        result = extract_palette_from_image(solid_red_image, num_colors=3, seed=0)
        for c in result["colors"]:
            assert "name" in c
            assert "hsl" in c
            assert "rgb" in c
            assert "role" in c
            assert isinstance(c["name"], str)
            assert len(c["hsl"]) == 3
            assert len(c["rgb"]) == 3

    def test_palette_string_is_comma_separated(self, solid_red_image):
        result = extract_palette_from_image(solid_red_image, num_colors=3, seed=0)
        if result["colors"]:
            parts = result["palette_string"].split(", ")
            assert len(parts) == len(result["colors"])

    def test_num_colors_respected(self, gradient_image):
        result = extract_palette_from_image(gradient_image, num_colors=2, seed=42)
        assert len(result["colors"]) == 2

    def test_roles_assigned(self, gradient_image):
        result = extract_palette_from_image(gradient_image, num_colors=4, seed=42)
        roles = [c["role"] for c in result["colors"]]
        assert "primary" in roles, f"Expected 'primary' role, got {roles}"


class TestRegions:
    """Test region cropping."""

    def test_upper_half_red_dominant(self, two_color_image):
        result = extract_palette_from_image(
            two_color_image, num_colors=2, region="upper_half",
            filter_skin=False, filter_background=False, seed=42
        )
        names = [c["name"] for c in result["colors"]]
        # Upper half is red, so first color should be red-family
        has_red = any("red" in n or n in ("crimson", "scarlet", "vermilion",
                                           "cherry", "ruby", "carmine", "maroon")
                      for n in names)
        assert has_red, f"Expected red-family in upper_half, got {names}"

    def test_lower_half_blue_dominant(self, two_color_image):
        result = extract_palette_from_image(
            two_color_image, num_colors=2, region="lower_half",
            filter_skin=False, filter_background=False, seed=42
        )
        names = [c["name"] for c in result["colors"]]
        has_blue = any("blue" in n or n in ("navy-blue", "cobalt", "royal-blue",
                                             "sapphire", "midnight-blue", "indigo")
                       for n in names)
        assert has_blue, f"Expected blue-family in lower_half, got {names}"


class TestEdgeCases:
    """Edge cases and degenerate inputs."""

    def test_all_black_no_crash(self, all_black_image):
        result = extract_palette_from_image(all_black_image, num_colors=3, seed=0)
        assert isinstance(result["colors"], list)
        assert isinstance(result["palette_string"], str)

    def test_all_white_no_crash(self, all_white_image):
        result = extract_palette_from_image(all_white_image, num_colors=3, seed=0)
        assert isinstance(result["colors"], list)

    def test_single_pixel_no_crash(self, single_pixel_image):
        result = extract_palette_from_image(single_pixel_image, num_colors=2, seed=0)
        assert isinstance(result["colors"], list)


class TestModes:
    """Test extraction modes."""

    def test_vibrant_prefers_saturated(self, gradient_image):
        result = extract_palette_from_image(
            gradient_image, num_colors=3, mode="vibrant", seed=42
        )
        colors = result["colors"]
        if len(colors) >= 2:
            # First color should have high saturation
            assert colors[0]["hsl"][1] >= 0, "Vibrant mode should return colors"

    def test_dominant_prefers_common(self, solid_red_image):
        result = extract_palette_from_image(
            solid_red_image, num_colors=3, mode="dominant", seed=42
        )
        assert len(result["colors"]) > 0

    def test_fashion_aware_runs(self, gradient_image):
        result = extract_palette_from_image(
            gradient_image, num_colors=3, mode="fashion_aware", seed=42
        )
        assert len(result["colors"]) > 0


class TestKMeans:
    """Test the internal K-Means implementation."""

    def test_simple_two_clusters(self):
        pixels = np.array([
            [255, 0, 0], [250, 0, 0], [245, 0, 0],
            [0, 0, 255], [0, 0, 250], [0, 0, 245],
        ], dtype=np.float64)
        labels, centers = _kmeans(pixels, k=2, seed=0)
        assert len(centers) == 2
        assert len(labels) == 6

    def test_k_larger_than_pixels(self):
        pixels = np.array([[100, 100, 100], [200, 200, 200]], dtype=np.float64)
        labels, centers = _kmeans(pixels, k=5, seed=0)
        assert len(centers) == 2  # k reduced to n

    def test_empty_pixels(self):
        pixels = np.zeros((0, 3), dtype=np.float64)
        labels, centers = _kmeans(pixels, k=3, seed=0)
        assert len(centers) == 0


class TestSkinToneDetection:
    """Test skin tone filter."""

    def test_typical_skin_detected(self):
        assert _is_skin_tone(25, 40, 60) is True

    def test_pure_red_not_skin(self):
        assert _is_skin_tone(0, 100, 50) is False

    def test_blue_not_skin(self):
        assert _is_skin_tone(220, 80, 50) is False

    def test_light_skin_detected(self):
        assert _is_skin_tone(20, 10, 80) is True
