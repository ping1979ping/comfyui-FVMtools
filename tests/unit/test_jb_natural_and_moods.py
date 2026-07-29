"""Natural-Prompt-Format, Farb-Moods und die Fragment-Bereinigung."""

import pytest

from core.jb.color_moods import (
    ACCENTS,
    COLOR_MOODS,
    DENIM,
    MONO_BASES,
    MOOD_NAMES,
    NEUTRALS,
    ROLE_ORDER,
    mood_colors,
    mood_engine_kwargs,
    mood_help,
)
from core.jb.serialize import (
    ALL_FORMATS,
    NATURAL,
    NON_PROMPT_KEYS,
    emit,
    emit_natural,
    natural_phrases,
)
from core.outfit_engine import (
    _build_description,
    _is_noop_decoration,
    garment_name_has_color,
)


class TestFragmentCleanup:
    def test_colour_not_prepended_when_name_has_one(self):
        """"white tennis shoes" darf nicht "charcoal-gray … white tennis shoes" werden."""
        out = _build_description("#neutral#", "canvas", "white tennis shoes")
        assert out == "canvas white tennis shoes"
        assert "#neutral#" not in out

    def test_colour_prepended_for_neutral_name(self):
        out = _build_description("#primary#", "jersey", "crew tee")
        assert out == "#primary# jersey crew tee"

    def test_fabric_not_repeated(self):
        out = _build_description("#secondary#", "denim", "basic denim jacket")
        assert out.count("denim") == 1

    def test_explicit_color_marker_still_wins(self):
        out = _build_description("#primary#", "silk", "long #color# dress")
        assert out == "long #primary# silk dress"

    @pytest.mark.parametrize("name", [
        "khaki shorts", "navy blazer", "black jeans", "off-white blouse",
        "olive parka", "cream cardigan", "denim skirt",
    ])
    def test_colour_words_detected(self, name):
        assert garment_name_has_color(name)

    @pytest.mark.parametrize("name", [
        "crew tee", "puffer vest", "straight-leg jeans", "ballet flats",
        "wool coat", "cotton blouse",
    ])
    def test_plain_names_not_flagged(self, name):
        assert not garment_name_has_color(name)

    @pytest.mark.parametrize("decoration", [
        "solid color", "solid colour", "plain", "none", "no print", "SOLID COLOR",
    ])
    def test_noop_decorations_recognised(self, decoration):
        assert _is_noop_decoration(decoration)

    def test_noop_decoration_not_emitted(self):
        out = _build_description("#primary#", "jersey", "crew tee", "solid color")
        assert "solid color" not in out
        assert "with" not in out

    def test_real_decoration_still_emitted(self):
        out = _build_description("#primary#", "jersey", "crew tee", "floral print")
        assert out.endswith("with floral print")


class TestNaturalFormat:
    @staticmethod
    def outfit():
        return {
            "outfit": {
                "set_name": "female/casual/everyday_basics",
                "seed": 77,
                "formality": "casual",
                "coverage_target": 0.5,
                "color_tone": "neutral",
                "garments": {
                    "upper_body": {
                        "name": "crew tee", "fabric": "jersey",
                        "color_role": "primary", "color_resolved": "sand",
                        "prompt_fragment": "sand jersey crew tee",
                    },
                    "lower_body": {
                        "name": "jeans", "fabric": "denim",
                        "color_role": "secondary", "color_resolved": "black",
                        "prompt_fragment": "black denim straight-leg jeans",
                    },
                },
            }
        }

    def test_natural_is_a_registered_format(self):
        assert NATURAL in ALL_FORMATS

    def test_only_phrases_survive(self):
        out = emit(self.outfit(), NATURAL)
        assert out == "sand jersey crew tee, black denim straight-leg jeans"

    def test_no_metadata_leaks(self):
        out = emit(self.outfit(), NATURAL)
        for token in ("seed", "77", "set_name", "female/casual", "coverage",
                      "0.5", "color_role", "primary", "formality", "{", "}", ":"):
            assert token not in out, f"metadata leaked: {token}"

    def test_name_and_fabric_are_not_duplicated(self):
        """Sie stecken schon im prompt_fragment."""
        out = emit(self.outfit(), NATURAL)
        assert out.count("crew tee") == 1
        assert out.count("jersey") == 1

    def test_prefix_is_applied(self):
        out = emit_natural(self.outfit(), prefix="wearing ")
        assert out.startswith("wearing sand jersey crew tee")

    def test_empty_structure_gives_empty_string(self):
        assert emit_natural({"outfit": {"seed": 1, "garments": {}}}) == ""

    def test_duplicates_are_collapsed(self):
        data = {"a": {"prompt_fragment": "navy coat"},
                "b": {"prompt_fragment": "navy coat"},
                "c": {"prompt_fragment": "Navy Coat"}}
        assert emit_natural(data) == "navy coat"

    def test_bare_numbers_and_bools_dropped(self):
        data = {"x": 0.5, "y": 12, "on": True, "off": False, "text": "a red car"}
        assert emit_natural(data) == "a red car"

    def test_nested_lists_are_walked(self):
        data = {"props": [{"prompt_fragment": "wooden bench"},
                          {"prompt_fragment": "paper cup"}]}
        assert emit_natural(data) == "wooden bench, paper cup"

    def test_underscore_keys_skipped(self):
        data = {"_debug": "internal note", "keep": "a blue door"}
        assert emit_natural(data) == "a blue door"

    def test_every_non_prompt_key_is_filtered(self):
        data = {key: f"value-of-{key}" for key in NON_PROMPT_KEYS}
        data["description"] = "a quiet street"
        assert emit_natural(data) == "a quiet street"

    def test_other_formats_unchanged(self):
        out = emit(self.outfit(), "loose_keys")
        assert "set_name" in out and "{" in out


class TestColorMoods:
    def test_all_moods_have_a_description(self):
        for name, spec in COLOR_MOODS.items():
            assert spec.get("description"), name

    def test_auto_defers_to_the_engine(self):
        assert mood_colors("auto", 1, 5) is None
        assert mood_engine_kwargs("auto") == {}

    def test_engine_moods_only_preset_sliders(self):
        assert mood_colors("bold", 1, 5) is None
        assert mood_engine_kwargs("bold")["vibrancy"] > 0.7

    def test_pool_moods_return_names(self):
        names = mood_colors("everyday_muted", 42, 5)
        assert names and len(names) == 5
        assert all(isinstance(n, str) and n for n in names)

    def test_deterministic_per_seed(self):
        assert mood_colors("everyday_muted", 7, 5) == mood_colors("everyday_muted", 7, 5)

    def test_different_seeds_differ(self):
        a = mood_colors("everyday_muted", 1, 5)
        b = mood_colors("everyday_muted", 2, 5)
        assert a != b

    def test_no_duplicates(self):
        for mood in ("everyday_muted", "neutral_basics", "warm_earth", "cool_muted"):
            names = mood_colors(mood, 3, 6)
            assert len(names) == len(set(names)), mood

    def test_neutral_basics_stays_neutral(self):
        names = mood_colors("neutral_basics", 11, 5)
        assert all(n in NEUTRALS for n in names)

    def test_one_accent_puts_the_accent_on_the_top(self):
        names = mood_colors("one_accent", 11, 5)
        primary = names[ROLE_ORDER.index("primary")]
        assert primary in ACCENTS
        # The rest stays calm.
        assert all(n in NEUTRALS for n in names[1:])

    def test_denim_lands_on_the_secondary_role(self):
        names = mood_colors("denim_casual", 11, 5)
        assert names[ROLE_ORDER.index("secondary")] in DENIM

    def test_denim_names_avoid_the_word_denim(self):
        """Sonst entsteht 'dark denim blue denim jeans'."""
        assert all("denim" not in name for name in DENIM)

    def test_monochrome_uses_one_base(self):
        names = mood_colors("monochrome", 5, 5)
        base = next(b for b in MONO_BASES if all(b in n for n in names))
        assert base
        assert len(names) == 5

    def test_monochrome_bases_take_a_shade_prefix(self):
        """'pale black' ist Unsinn — black/white gehören nicht in die Basen."""
        assert "black" not in MONO_BASES
        assert "white" not in MONO_BASES

    def test_count_is_clamped(self):
        assert len(mood_colors("everyday_muted", 1, 99)) <= 8
        assert len(mood_colors("everyday_muted", 1, 0)) >= 2

    def test_unknown_mood_falls_back_to_engine(self):
        assert mood_colors("does-not-exist", 1, 5) is None

    def test_help_lists_every_mood(self):
        text = mood_help()
        for name in MOOD_NAMES:
            assert name in text
