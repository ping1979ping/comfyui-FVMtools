"""Unit tests for core/outfit_lists.py — garment, fabric, and harmony loading."""

import pytest
from core.outfit_lists import (load_garments, load_fabrics, load_fabric_harmony,
                               get_available_sets, get_list_file_path,
                               load_prints, load_texts)


ALL_SLOTS = ["top", "bottom", "footwear", "headwear", "outerwear", "accessories", "bag"]


class TestGetAvailableSets:
    """Tests for get_available_sets()."""

    def test_returns_non_empty_list(self):
        sets = get_available_sets()
        assert len(sets) > 0

    def test_contains_general_female(self):
        sets = get_available_sets()
        assert "general_female" in sets

    def test_returns_sorted(self):
        sets = get_available_sets()
        assert sets == sorted(sets)

    def test_returns_list_of_strings(self):
        sets = get_available_sets()
        for s in sets:
            assert isinstance(s, str)


class TestGetListFilePath:
    """Tests for get_list_file_path()."""

    def test_returns_string(self):
        path = get_list_file_path("general_female", "top")
        assert isinstance(path, str)

    def test_contains_outfit_set_and_slot(self):
        path = get_list_file_path("general_female", "top")
        assert "general_female" in path
        assert "top.txt" in path


class TestLoadGarments:
    """Tests for load_garments()."""

    def test_top_returns_non_empty(self):
        garments = load_garments("top", "general_female")
        assert len(garments) > 0

    def test_garment_has_required_keys(self):
        garments = load_garments("top", "general_female")
        for g in garments:
            assert "name" in g
            assert "probability" in g
            assert "formality" in g
            assert "fabrics" in g

    def test_probability_range(self):
        garments = load_garments("top", "general_female")
        for g in garments:
            assert 0.0 <= g["probability"] <= 1.0, f"{g['name']} probability out of range"

    def test_formality_is_tuple_of_two_floats(self):
        garments = load_garments("top", "general_female")
        for g in garments:
            assert isinstance(g["formality"], tuple)
            assert len(g["formality"]) == 2
            assert isinstance(g["formality"][0], float)
            assert isinstance(g["formality"][1], float)
            assert g["formality"][0] <= g["formality"][1]

    def test_fabrics_non_empty_list(self):
        garments = load_garments("top", "general_female")
        for g in garments:
            assert isinstance(g["fabrics"], list)
            assert len(g["fabrics"]) > 0

    def test_invalid_slot_returns_empty(self):
        garments = load_garments("nonexistent_slot_xyz", "general_female")
        assert garments == []

    def test_invalid_set_returns_empty(self):
        garments = load_garments("top", "nonexistent_set_xyz")
        assert garments == []

    @pytest.mark.parametrize("slot", ALL_SLOTS)
    def test_all_slots_loadable(self, slot):
        garments = load_garments(slot, "general_female")
        assert len(garments) > 0, f"Slot '{slot}' returned no garments"

    def test_default_outfit_set_is_general_female(self):
        """Calling without outfit_set should default to general_female."""
        garments_default = load_garments("top")
        garments_explicit = load_garments("top", "general_female")
        assert garments_default == garments_explicit


class TestLoadFabrics:
    """Tests for load_fabrics()."""

    def test_returns_non_empty_dict(self):
        fabrics = load_fabrics("general_female")
        assert len(fabrics) > 0

    def test_fabric_has_required_keys(self):
        fabrics = load_fabrics("general_female")
        for name, data in fabrics.items():
            assert "formality" in data, f"{name} missing formality"
            assert "family" in data, f"{name} missing family"
            assert "weight" in data, f"{name} missing weight"

    def test_fabric_formality_is_float(self):
        fabrics = load_fabrics("general_female")
        for name, data in fabrics.items():
            assert isinstance(data["formality"], float), f"{name} formality not float"

    def test_known_fabrics_exist(self):
        fabrics = load_fabrics("general_female")
        for expected in ["cotton", "silk", "leather", "denim"]:
            assert expected in fabrics, f"Expected fabric '{expected}' not found"

    def test_default_outfit_set_is_general_female(self):
        """Calling without outfit_set should default to general_female."""
        fabrics_default = load_fabrics()
        fabrics_explicit = load_fabrics("general_female")
        assert fabrics_default == fabrics_explicit


class TestLoadFabricHarmony:
    """Tests for load_fabric_harmony()."""

    def test_returns_non_empty_dict(self):
        harmony = load_fabric_harmony()
        assert len(harmony) > 0

    def test_known_families_present(self):
        harmony = load_fabric_harmony()
        for family in ["luxury", "natural", "casual", "tough", "sporty"]:
            assert family in harmony, f"Family '{family}' not in harmony rules"

    def test_harmony_values_are_lists(self):
        harmony = load_fabric_harmony()
        for family, compatible in harmony.items():
            assert isinstance(compatible, list)
            assert len(compatible) > 0


class TestLoadPrints:
    """Tests for load_prints()."""

    def test_returns_non_empty_list(self):
        prints = load_prints("general_female")
        assert len(prints) > 0

    def test_print_has_required_keys(self):
        prints = load_prints("general_female")
        for p in prints:
            assert "name" in p
            assert "probability" in p
            assert "slots" in p
            assert "formality" in p

    def test_probability_range(self):
        prints = load_prints("general_female")
        for p in prints:
            assert 0.0 <= p["probability"] <= 1.0, f"{p['name']} probability out of range"

    def test_formality_is_tuple(self):
        prints = load_prints("general_female")
        for p in prints:
            assert isinstance(p["formality"], tuple)
            assert len(p["formality"]) == 2
            assert p["formality"][0] <= p["formality"][1]

    def test_slots_is_list(self):
        prints = load_prints("general_female")
        for p in prints:
            assert isinstance(p["slots"], list)
            assert len(p["slots"]) > 0

    def test_nonexistent_file_returns_empty(self):
        prints = load_prints("nonexistent_set_xyz")
        assert prints == []

    def test_default_outfit_set(self):
        prints_default = load_prints()
        prints_explicit = load_prints("general_female")
        assert prints_default == prints_explicit


class TestLoadTexts:
    """Tests for load_texts()."""

    def test_returns_non_empty_list(self):
        texts = load_texts("general_female")
        assert len(texts) > 0

    def test_text_has_required_keys(self):
        texts = load_texts("general_female")
        for t in texts:
            assert "text" in t
            assert "probability" in t
            assert "slots" in t
            assert "font" in t

    def test_probability_range(self):
        texts = load_texts("general_female")
        for t in texts:
            assert 0.0 <= t["probability"] <= 1.0, f"{t['text']} probability out of range"

    def test_slots_is_list(self):
        texts = load_texts("general_female")
        for t in texts:
            assert isinstance(t["slots"], list)
            assert len(t["slots"]) > 0

    def test_font_is_string(self):
        texts = load_texts("general_female")
        for t in texts:
            assert isinstance(t["font"], str)
            assert len(t["font"]) > 0

    def test_nonexistent_file_returns_empty(self):
        texts = load_texts("nonexistent_set_xyz")
        assert texts == []

    def test_default_outfit_set(self):
        texts_default = load_texts()
        texts_explicit = load_texts("general_female")
        assert texts_default == texts_explicit


class TestCrossValidation:
    """Cross-validate garment fabrics against fabric database."""

    @pytest.mark.parametrize("slot", ALL_SLOTS)
    def test_garment_fabrics_exist_in_database(self, slot):
        garments = load_garments(slot, "general_female")
        fabrics_db = load_fabrics("general_female")
        for g in garments:
            for fab in g["fabrics"]:
                assert fab in fabrics_db, (
                    f"Slot '{slot}', garment '{g['name']}' references "
                    f"unknown fabric '{fab}'"
                )
