import pytest
from nodes.prompt_color_replace import FVM_PromptColorReplace, _build_replacement_map, _replace_tags


class TestPromptColorReplaceSchema:
    """Verify node schema / registration."""

    def test_has_required_class_attrs(self):
        assert FVM_PromptColorReplace.CATEGORY == "FVM Tools/Color"
        assert FVM_PromptColorReplace.FUNCTION == "replace"
        assert FVM_PromptColorReplace.RETURN_TYPES == ("STRING", "STRING")
        assert FVM_PromptColorReplace.RETURN_NAMES == ("prompt", "replacements_log")

    def test_input_types_structure(self):
        inputs = FVM_PromptColorReplace.INPUT_TYPES()
        assert "required" in inputs
        assert "prompt" in inputs["required"]
        assert "palette_string" in inputs["required"]
        assert "optional" in inputs
        for key in ("primary", "secondary", "accent", "neutral", "metallic",
                     "fallback_color", "strip_hyphens"):
            assert key in inputs["optional"], f"Missing optional input: {key}"

    def test_function_exists(self):
        node = FVM_PromptColorReplace()
        assert hasattr(node, "replace")
        assert callable(node.replace)


class TestBuildReplacementMap:
    """Test the internal replacement map builder."""

    def test_numbered_tags(self):
        m = _build_replacement_map("red, blue, green", "", "", "", "", "", "black")
        assert m["#color1#"] == "red"
        assert m["#color2#"] == "blue"
        assert m["#color3#"] == "green"
        assert m["#c1#"] == "red"
        assert m["#c2#"] == "blue"

    def test_missing_positions_use_fallback(self):
        m = _build_replacement_map("red", "", "", "", "", "", "fallback")
        assert m["#color1#"] == "red"
        assert m["#color2#"] == "fallback"
        assert m["#color8#"] == "fallback"

    def test_semantic_tags_map_to_positions(self):
        m = _build_replacement_map("a, b, c, d, e", "", "", "", "", "", "x")
        assert m["#primary#"] == "a"
        assert m["#secondary#"] == "b"
        assert m["#accent#"] == "c"
        assert m["#neutral#"] == "d"
        assert m["#metallic#"] == "e"

    def test_short_aliases(self):
        m = _build_replacement_map("a, b, c, d, e", "", "", "", "", "", "x")
        assert m["#pri#"] == "a"
        assert m["#sec#"] == "b"
        assert m["#acc#"] == "c"
        assert m["#neu#"] == "d"
        assert m["#met#"] == "e"

    def test_overrides_take_precedence(self):
        m = _build_replacement_map("a, b, c, d, e", "RED", "BLUE", "", "", "", "x")
        assert m["#primary#"] == "RED"
        assert m["#pri#"] == "RED"
        assert m["#secondary#"] == "BLUE"
        assert m["#sec#"] == "BLUE"
        # Non-overridden stay from palette
        assert m["#accent#"] == "c"

    def test_empty_palette(self):
        m = _build_replacement_map("", "", "", "", "", "", "fallback")
        assert m["#color1#"] == "fallback"
        assert m["#primary#"] == "fallback"

    def test_whitespace_in_palette(self):
        m = _build_replacement_map("  navy-blue , soft-pink  , gold  ", "", "", "", "", "", "x")
        assert m["#color1#"] == "navy-blue"
        assert m["#color2#"] == "soft-pink"
        assert m["#color3#"] == "gold"


class TestReplaceTags:
    """Test the tag replacement logic."""

    def test_numbered_replacement(self):
        replacements = {"#color1#": "red", "#color2#": "blue"}
        result, log = _replace_tags("a #color1# b #color2# c", replacements, False, "black")
        assert result == "a red b blue c"

    def test_semantic_replacement(self):
        replacements = {"#primary#": "navy", "#neutral#": "cream"}
        result, log = _replace_tags("#primary# dress with #neutral# shoes", replacements, False, "x")
        assert result == "navy dress with cream shoes"

    def test_short_alias_replacement(self):
        replacements = {"#primary#": "navy", "#pri#": "navy"}
        result, log = _replace_tags("#pri# dress", replacements, False, "x")
        assert result == "navy dress"

    def test_case_insensitive(self):
        replacements = {"#color1#": "red"}
        result, _ = _replace_tags("#COLOR1# and #Color1#", replacements, False, "black")
        assert result == "red and red"

    def test_strip_hyphens_on(self):
        replacements = {"#color1#": "navy-blue"}
        result, _ = _replace_tags("#color1#", replacements, True, "black")
        assert result == "navy blue"

    def test_strip_hyphens_off(self):
        replacements = {"#color1#": "navy-blue"}
        result, _ = _replace_tags("#color1#", replacements, False, "black")
        assert result == "navy-blue"

    def test_multiple_same_tag(self):
        replacements = {"#color1#": "red"}
        result, log = _replace_tags("#color1# top and #color1# skirt", replacements, False, "x")
        assert result == "red top and red skirt"
        assert log.count("red") == 2

    def test_no_tags_in_prompt(self):
        result, log = _replace_tags("plain prompt without tags", {}, False, "x")
        assert result == "plain prompt without tags"
        assert "No tags" in log

    def test_empty_prompt(self):
        result, log = _replace_tags("", {}, False, "x")
        assert result == ""
        assert "No tags" in log

    def test_mixed_numbered_and_semantic(self):
        replacements = {
            "#color1#": "red", "#color3#": "green",
            "#primary#": "red", "#accent#": "green",
        }
        result, _ = _replace_tags("#primary# with #color3#", replacements, False, "x")
        assert result == "red with green"

    def test_tags_inside_wildcards(self):
        replacements = {"#primary#": "red", "#neutral#": "white"}
        result, _ = _replace_tags(
            "#primary# {skirt|dress} with #neutral# top",
            replacements, False, "x"
        )
        assert result == "red {skirt|dress} with white top"

    def test_fallback_for_unknown_position(self):
        replacements = {"#color8#": "fallback"}
        result, _ = _replace_tags("#color8#", replacements, False, "fallback")
        assert result == "fallback"

    def test_log_format(self):
        replacements = {"#color1#": "red", "#color2#": "blue"}
        _, log = _replace_tags("#color1# and #color2#", replacements, False, "x")
        assert "#color1# -> red" in log
        assert "#color2# -> blue" in log


class TestNodeExecute:
    """Integration test of the full node execute path."""

    def test_basic_execute(self):
        node = FVM_PromptColorReplace()
        result = node.replace(
            prompt="wearing #primary# dress",
            palette_string="navy-blue, soft-pink, charcoal-gray, gold, cream",
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "wearing navy blue dress"
        assert "#primary# -> navy blue" in result[1]

    def test_execute_with_override(self):
        node = FVM_PromptColorReplace()
        result = node.replace(
            prompt="wearing #primary# dress",
            palette_string="navy-blue, soft-pink",
            primary="red",
        )
        assert result[0] == "wearing red dress"

    def test_execute_no_strip(self):
        node = FVM_PromptColorReplace()
        result = node.replace(
            prompt="#color1#",
            palette_string="navy-blue",
            strip_hyphens=False,
        )
        assert result[0] == "navy-blue"

    def test_execute_full_palette(self):
        node = FVM_PromptColorReplace()
        palette = "red, blue, green, white, gold, pink, orange, purple"
        prompt = "#c1# #c2# #c3# #c4# #c5# #c6# #c7# #c8#"
        result = node.replace(prompt=prompt, palette_string=palette)
        assert result[0] == "red blue green white gold pink orange purple"

    def test_execute_empty_inputs(self):
        node = FVM_PromptColorReplace()
        result = node.replace(prompt="", palette_string="")
        assert result[0] == ""
        assert "No tags" in result[1]

    def test_execute_complex_prompt(self):
        node = FVM_PromptColorReplace()
        prompt = (
            "wearing #primary# {miniskirt|skirt} with #neutral# bikini top, "
            "#accent# {open shirt|sarong}, #metallic# jewelry"
        )
        palette = "emerald-green, blush-pink, lavender, cream, gold"
        result = node.replace(prompt=prompt, palette_string=palette)
        out = result[0]
        assert "emerald green" in out
        assert "cream" in out
        assert "lavender" in out
        assert "gold" in out
