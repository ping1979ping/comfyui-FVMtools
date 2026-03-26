import pytest
from core.color_utils import hsl_to_rgb, rgb_to_hsl, hue_distance, find_nearest_color_name
from core.color_database import COLOR_DATABASE, NEUTRAL_NAMES, METALLIC_NAMES, NEON_NAMES


class TestHslToRgb:
    def test_red(self):
        r, g, b = hsl_to_rgb(0, 100, 50)
        assert r == 255 and g == 0 and b == 0

    def test_green(self):
        r, g, b = hsl_to_rgb(120, 100, 50)
        assert g == 255

    def test_blue(self):
        r, g, b = hsl_to_rgb(240, 100, 50)
        assert b == 255

    def test_white(self):
        r, g, b = hsl_to_rgb(0, 0, 100)
        assert r == 255 and g == 255 and b == 255

    def test_black(self):
        r, g, b = hsl_to_rgb(0, 0, 0)
        assert r == 0 and g == 0 and b == 0


class TestRgbToHsl:
    def test_red(self):
        h, s, l = rgb_to_hsl(255, 0, 0)
        assert h == 0 and s == 100 and l == 50

    def test_roundtrip(self):
        r, g, b = hsl_to_rgb(200, 60, 70)
        h, s, l = rgb_to_hsl(r, g, b)
        # Allow ±2 for rounding
        assert abs(h - 200) <= 2
        assert abs(s - 60) <= 2
        assert abs(l - 70) <= 2


class TestHueDistance:
    def test_same(self):
        assert hue_distance(100, 100) == 0

    def test_opposite(self):
        assert hue_distance(0, 180) == 180

    def test_wrap_around(self):
        assert hue_distance(10, 350) == 20

    def test_symmetric(self):
        assert hue_distance(30, 90) == hue_distance(90, 30)


class TestFindNearestColorName:
    def test_exact_match(self):
        # Black is (0, 0, 5) in database
        name = find_nearest_color_name(0, 0, 5)
        assert name == "black"

    def test_returns_string(self):
        name = find_nearest_color_name(180, 50, 50)
        assert isinstance(name, str)
        assert name in COLOR_DATABASE

    def test_exclude_names(self):
        name1 = find_nearest_color_name(0, 0, 5)
        name2 = find_nearest_color_name(0, 0, 5, exclude_names={name1})
        assert name2 != name1

    def test_neons_blocked_at_low_vibrancy(self):
        # Neon green is (120, 100, 50)
        name = find_nearest_color_name(120, 100, 50, vibrancy=0.5)
        assert name not in NEON_NAMES

    def test_neons_allowed_at_high_vibrancy(self):
        name = find_nearest_color_name(120, 100, 50, vibrancy=0.9)
        # Should match neon-green or similar
        assert name in COLOR_DATABASE

    def test_forbidden_set(self):
        name = find_nearest_color_name(0, 0, 5, forbidden_set={"black"})
        assert name != "black"


class TestColorDatabase:
    def test_has_entries(self):
        assert len(COLOR_DATABASE) >= 100

    def test_all_hsl_in_range(self):
        for name, (h, s, l) in COLOR_DATABASE.items():
            assert 0 <= h <= 360, f"{name}: hue {h} out of range"
            assert 0 <= s <= 100, f"{name}: saturation {s} out of range"
            assert 0 <= l <= 100, f"{name}: lightness {l} out of range"

    def test_neutral_names_in_db(self):
        for name in NEUTRAL_NAMES:
            assert name in COLOR_DATABASE, f"Neutral '{name}' not in COLOR_DATABASE"

    def test_metallic_names_in_db(self):
        for name in METALLIC_NAMES:
            assert name in COLOR_DATABASE, f"Metallic '{name}' not in COLOR_DATABASE"

    def test_neon_names_in_db(self):
        for name in NEON_NAMES:
            assert name in COLOR_DATABASE, f"Neon '{name}' not in COLOR_DATABASE"

    def test_no_overlap_neutral_neon(self):
        assert NEUTRAL_NAMES.isdisjoint(NEON_NAMES)

    def test_no_overlap_neutral_metallic(self):
        # Gold can be in both metallic and the main db, but not in neutrals
        overlap = NEUTRAL_NAMES & METALLIC_NAMES
        assert len(overlap) == 0, f"Overlap: {overlap}"
