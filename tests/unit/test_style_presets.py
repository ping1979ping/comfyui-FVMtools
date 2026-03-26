import pytest
from core.style_presets import STYLE_PRESETS
from core.color_database import COLOR_DATABASE


EXPECTED_PRESETS = [
    "general", "beach", "urban_streetwear", "evening_gala", "casual_daywear",
    "vintage_retro", "cyberpunk_neon", "pastel_dream", "earthy_natural",
    "monochrome_chic", "tropical", "winter_cozy", "festival", "office_professional",
]

REQUIRED_KEYS = [
    "hue_ranges", "vibrancy_mod", "contrast_mod", "warmth_mod",
    "preferred_harmonies", "forbidden_names", "neutral_bias",
]


class TestPresetCompleteness:

    def test_all_14_presets_exist(self):
        assert len(STYLE_PRESETS) == 14
        for name in EXPECTED_PRESETS:
            assert name in STYLE_PRESETS, f"Missing preset: {name}"

    @pytest.mark.parametrize("preset_name", EXPECTED_PRESETS)
    def test_has_required_keys(self, preset_name):
        preset = STYLE_PRESETS[preset_name]
        for key in REQUIRED_KEYS:
            assert key in preset, f"Preset {preset_name} missing key: {key}"


class TestPresetValues:

    @pytest.mark.parametrize("preset_name", EXPECTED_PRESETS)
    def test_hue_ranges_valid(self, preset_name):
        ranges = STYLE_PRESETS[preset_name]["hue_ranges"]
        if ranges is None:
            return  # None means all hues allowed
        assert isinstance(ranges, list)
        for start, end in ranges:
            assert 0 <= start <= 360
            assert 0 <= end <= 360

    @pytest.mark.parametrize("preset_name", EXPECTED_PRESETS)
    def test_modifiers_in_range(self, preset_name):
        preset = STYLE_PRESETS[preset_name]
        for key in ("vibrancy_mod", "contrast_mod", "warmth_mod"):
            val = preset[key]
            assert -1.0 <= val <= 1.0, f"{preset_name}.{key} = {val} out of range"

    @pytest.mark.parametrize("preset_name", EXPECTED_PRESETS)
    def test_forbidden_names_exist_in_database(self, preset_name):
        forbidden = STYLE_PRESETS[preset_name]["forbidden_names"]
        for name in forbidden:
            assert name in COLOR_DATABASE, f"Forbidden name '{name}' not in COLOR_DATABASE"

    @pytest.mark.parametrize("preset_name", EXPECTED_PRESETS)
    def test_neutral_bias_is_list(self, preset_name):
        bias = STYLE_PRESETS[preset_name]["neutral_bias"]
        assert isinstance(bias, list)

    @pytest.mark.parametrize("preset_name", EXPECTED_PRESETS)
    def test_preferred_harmonies_valid(self, preset_name):
        harmonies = STYLE_PRESETS[preset_name]["preferred_harmonies"]
        if harmonies is None:
            return
        valid = {"analogous", "complementary", "split_complementary",
                 "triadic", "tetradic", "monochromatic"}
        for h in harmonies:
            assert h in valid, f"Invalid harmony '{h}' in {preset_name}"

    def test_general_preset_is_neutral(self):
        gen = STYLE_PRESETS["general"]
        assert gen["hue_ranges"] is None
        assert gen["vibrancy_mod"] == 0.0
        assert gen["contrast_mod"] == 0.0
        assert gen["warmth_mod"] == 0.0
