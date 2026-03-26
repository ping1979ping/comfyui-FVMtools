"""Unit tests for core/outfit_parser.py — override string parsing and wildcards."""

import random
import pytest
from core.outfit_parser import parse_overrides, resolve_wildcards


class TestParseOverrides:
    """Tests for parse_overrides()."""

    def test_empty_string_returns_empty_dict(self):
        assert parse_overrides("") == {}

    def test_none_returns_empty_dict(self):
        assert parse_overrides(None) == {}

    def test_whitespace_only_returns_empty_dict(self):
        assert parse_overrides("   \n  \n  ") == {}

    def test_exclude_mode(self):
        result = parse_overrides("top: exclude")
        assert "top" in result
        assert result["top"]["mode"] == "exclude"

    def test_auto_mode(self):
        result = parse_overrides("top: auto")
        assert "top" in result
        assert result["top"]["mode"] == "auto"

    def test_override_with_fabric_garment_and_color(self):
        result = parse_overrides("top: silk blouse | #primary#")
        assert "top" in result
        assert result["top"]["mode"] == "override"
        assert result["top"]["garment"] == "blouse"
        assert result["top"]["fabric"] == "silk"
        assert result["top"]["color_tag"] == "#primary#"

    def test_override_garment_only(self):
        result = parse_overrides("bottom: jeans")
        assert "bottom" in result
        assert result["bottom"]["mode"] == "override"
        assert result["bottom"]["garment"] == "jeans"

    def test_multiple_lines(self):
        text = "top: silk blouse | #primary#\nbottom: exclude\nfootwear: auto"
        result = parse_overrides(text)
        assert len(result) == 3
        assert result["top"]["mode"] == "override"
        assert result["bottom"]["mode"] == "exclude"
        assert result["footwear"]["mode"] == "auto"

    def test_comment_lines_ignored(self):
        text = "# this is a comment\ntop: auto"
        result = parse_overrides(text)
        assert len(result) == 1
        assert "top" in result

    def test_malformed_line_no_colon_ignored(self):
        text = "this has no colon\ntop: auto"
        result = parse_overrides(text)
        assert len(result) == 1
        assert "top" in result

    def test_slot_name_case_insensitive(self):
        result = parse_overrides("TOP: auto")
        assert "top" in result


class TestResolveWildcards:
    """Tests for resolve_wildcards()."""

    def test_returns_one_of_options(self):
        rng = random.Random(42)
        result = resolve_wildcards("{a|b|c}", rng)
        assert result in ("a", "b", "c")

    def test_no_wildcards_unchanged(self):
        rng = random.Random(42)
        text = "plain text without wildcards"
        assert resolve_wildcards(text, rng) == text

    def test_deterministic_same_seed(self):
        result1 = resolve_wildcards("{a|b|c|d|e}", random.Random(99))
        result2 = resolve_wildcards("{a|b|c|d|e}", random.Random(99))
        assert result1 == result2

    def test_empty_string(self):
        rng = random.Random(42)
        assert resolve_wildcards("", rng) == ""

    def test_none_returns_none(self):
        rng = random.Random(42)
        assert resolve_wildcards(None, rng) is None

    def test_multiple_wildcards(self):
        rng = random.Random(42)
        result = resolve_wildcards("{a|b} and {c|d}", rng)
        parts = result.split(" and ")
        assert parts[0] in ("a", "b")
        assert parts[1] in ("c", "d")
