import pytest
from core.harmony import generate_harmony_hues


class TestAnalogous:

    def test_produces_three_hues(self):
        hues = generate_harmony_hues(120, "analogous", 3)
        assert len(hues) == 3

    def test_hues_within_range(self):
        hues = generate_harmony_hues(120, "analogous", 3)
        for h in hues:
            assert 0 <= h < 360

    def test_hues_near_base(self):
        base = 120
        hues = generate_harmony_hues(base, "analogous", 3)
        for h in hues:
            dist = min(abs(h - base), 360 - abs(h - base))
            assert dist <= 40, f"Hue {h} too far from base {base}"

    def test_wrap_around(self):
        hues = generate_harmony_hues(350, "analogous", 3)
        assert len(hues) == 3
        for h in hues:
            assert 0 <= h < 360


class TestComplementary:

    def test_produces_two_hues(self):
        hues = generate_harmony_hues(60, "complementary", 2)
        assert len(hues) == 2

    def test_includes_opposite(self):
        hues = generate_harmony_hues(60, "complementary", 2)
        assert 60 in hues
        assert (60 + 180) % 360 in hues

    def test_wrap_around(self):
        hues = generate_harmony_hues(350, "complementary", 2)
        assert (350 + 180) % 360 in hues  # 170


class TestTriadic:

    def test_produces_three_hues(self):
        hues = generate_harmony_hues(0, "triadic", 3)
        assert len(hues) == 3

    def test_includes_120_240(self):
        hues = generate_harmony_hues(0, "triadic", 3)
        assert 0 in hues
        assert 120 in hues
        assert 240 in hues

    def test_wrap_around(self):
        hues = generate_harmony_hues(300, "triadic", 3)
        expected = {300, (300 + 120) % 360, (300 + 240) % 360}
        assert set(hues) == expected


class TestSplitComplementary:

    def test_produces_three_hues(self):
        hues = generate_harmony_hues(0, "split_complementary", 3)
        assert len(hues) == 3

    def test_correct_hues(self):
        hues = generate_harmony_hues(0, "split_complementary", 3)
        assert 0 in hues
        assert 150 in hues
        assert 210 in hues


class TestTetradic:

    def test_produces_four_hues(self):
        hues = generate_harmony_hues(0, "tetradic", 4)
        assert len(hues) == 4

    def test_correct_hues(self):
        hues = generate_harmony_hues(0, "tetradic", 4)
        assert set(hues) == {0, 90, 180, 270}


class TestMonochromatic:

    def test_all_same_hue(self):
        hues = generate_harmony_hues(45, "monochromatic", 5)
        assert all(h == 45 for h in hues)

    def test_count_respected(self):
        hues = generate_harmony_hues(45, "monochromatic", 3)
        assert len(hues) == 3


class TestEdgeCases:

    def test_count_one(self):
        hues = generate_harmony_hues(90, "complementary", 1)
        assert len(hues) == 1
        assert hues[0] == 90

    def test_count_zero(self):
        hues = generate_harmony_hues(90, "complementary", 0)
        assert hues == []

    def test_extra_hues_fills(self):
        hues = generate_harmony_hues(0, "complementary", 5)
        assert len(hues) == 5
        for h in hues:
            assert 0 <= h < 360

    def test_all_hues_in_range(self):
        for harmony in ["analogous", "complementary", "split_complementary",
                        "triadic", "tetradic", "monochromatic"]:
            hues = generate_harmony_hues(200, harmony, 6)
            for h in hues:
                assert 0 <= h < 360, f"{harmony}: hue {h} out of range"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            generate_harmony_hues(0, "unknown_type", 3)
