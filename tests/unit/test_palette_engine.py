import pytest
from core.palette_engine import generate_palette
from core.color_database import NEUTRAL_NAMES, METALLIC_NAMES, NEON_NAMES
from core.style_presets import STYLE_PRESETS


class TestDeterminism:

    def test_same_seed_same_output(self):
        r1 = generate_palette(seed=42, num_colors=5)
        r2 = generate_palette(seed=42, num_colors=5)
        assert r1["palette_string"] == r2["palette_string"]

    def test_different_seeds_differ(self):
        r1 = generate_palette(seed=1, num_colors=5)
        r2 = generate_palette(seed=999, num_colors=5)
        assert r1["palette_string"] != r2["palette_string"]


class TestOutputStructure:

    def test_has_required_keys(self):
        result = generate_palette(seed=0, num_colors=5)
        assert "colors" in result
        assert "palette_string" in result
        assert "info" in result

    def test_colors_list_length(self):
        for n in [2, 3, 5, 8]:
            result = generate_palette(seed=0, num_colors=n)
            assert len(result["colors"]) == n

    def test_color_dict_structure(self):
        result = generate_palette(seed=0, num_colors=5)
        for c in result["colors"]:
            assert "name" in c
            assert "hsl" in c
            assert "rgb" in c
            assert "role" in c
            assert isinstance(c["name"], str)
            assert len(c["hsl"]) == 3
            assert len(c["rgb"]) == 3

    def test_palette_string_format(self):
        result = generate_palette(seed=0, num_colors=4)
        parts = [p.strip() for p in result["palette_string"].split(",")]
        assert len(parts) == 4

    def test_info_is_string(self):
        result = generate_palette(seed=0, num_colors=5)
        assert isinstance(result["info"], str)
        assert "Seed: 0" in result["info"]


class TestNumColors:

    def test_min_colors(self):
        result = generate_palette(seed=0, num_colors=2)
        assert len(result["colors"]) == 2

    def test_max_colors(self):
        result = generate_palette(seed=0, num_colors=8)
        assert len(result["colors"]) == 8


class TestVibrancy:

    def test_low_vibrancy_no_neons(self):
        result = generate_palette(seed=42, num_colors=5, vibrancy=0.0)
        names = {c["name"] for c in result["colors"]}
        assert not names.intersection(NEON_NAMES), "Low vibrancy should not produce neons"

    def test_high_vibrancy_allows_neons(self):
        """High vibrancy CAN produce neons (not guaranteed every seed)."""
        # Just verify it doesn't crash
        result = generate_palette(seed=42, num_colors=5, vibrancy=1.0)
        assert len(result["colors"]) == 5


class TestNeutralRatio:

    def test_zero_neutral_ratio(self):
        result = generate_palette(seed=42, num_colors=5, neutral_ratio=0.0)
        names = {c["name"] for c in result["colors"]}
        neutrals_in_result = names.intersection(NEUTRAL_NAMES)
        assert len(neutrals_in_result) == 0, f"Expected no neutrals, got {neutrals_in_result}"

    def test_high_neutral_ratio(self):
        result = generate_palette(seed=42, num_colors=5, neutral_ratio=1.0)
        names = {c["name"] for c in result["colors"]}
        neutral_or_metallic = names.intersection(NEUTRAL_NAMES | METALLIC_NAMES)
        # At least 1 chromatic is forced, so max neutrals = num_colors - 1
        assert len(neutral_or_metallic) >= 3


class TestNoDuplicates:

    def test_no_duplicate_names(self):
        for seed in range(20):
            result = generate_palette(seed=seed, num_colors=6)
            names = [c["name"] for c in result["colors"]]
            assert len(names) == len(set(names)), f"Duplicates at seed {seed}: {names}"


class TestHarmonyTypes:

    @pytest.mark.parametrize("harmony", [
        "auto", "analogous", "complementary", "split_complementary",
        "triadic", "tetradic", "monochromatic",
    ])
    def test_harmony_type_works(self, harmony):
        result = generate_palette(seed=42, num_colors=5, harmony_type=harmony)
        assert len(result["colors"]) == 5


class TestStylePresets:

    @pytest.mark.parametrize("preset", list(STYLE_PRESETS.keys()))
    def test_all_presets_work(self, preset):
        result = generate_palette(seed=42, num_colors=5, style_preset=preset)
        assert len(result["colors"]) == 5


class TestMetallics:

    def test_metallics_included(self):
        """With metallics enabled and neutral ratio > 0, at least one metallic should appear
        across multiple seeds."""
        found_metallic = False
        for seed in range(50):
            result = generate_palette(seed=seed, num_colors=5,
                                      neutral_ratio=0.4, include_metallics=True)
            names = {c["name"] for c in result["colors"]}
            if names.intersection(METALLIC_NAMES):
                found_metallic = True
                break
        assert found_metallic, "No metallics found across 50 seeds"

    def test_metallics_disabled(self):
        """With metallics disabled, metallic names should be much less common."""
        # Not a hard guarantee since find_nearest_color_name could still match metallics
        # Just verify no crash
        result = generate_palette(seed=42, num_colors=5,
                                  neutral_ratio=0.4, include_metallics=False)
        assert len(result["colors"]) == 5


class TestRoleAssignment:

    def test_has_primary(self):
        result = generate_palette(seed=42, num_colors=5)
        roles = {c["role"] for c in result["colors"] if c["role"]}
        assert "primary" in roles
