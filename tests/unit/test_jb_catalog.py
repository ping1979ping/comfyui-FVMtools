"""P1 — tests for core/jb/catalog.py."""

import os

import pytest

from core.jb import catalog as cat
from core.jb.catalog import DEFAULT_CATEGORIES


@pytest.fixture
def isolated_catalog(tmp_path, monkeypatch):
    """Redirect catalog_root() to a tmp directory for each test."""
    test_root = tmp_path / "prompt_catalog"
    monkeypatch.setattr(cat, "catalog_root", lambda: str(test_root))
    test_root.mkdir(exist_ok=True)
    for c in DEFAULT_CATEGORIES:
        (test_root / c).mkdir(exist_ok=True)
    return test_root


def test_list_categories_includes_defaults(isolated_catalog):
    cats = cat.list_categories()
    for c in DEFAULT_CATEGORIES:
        assert c in cats


def test_list_categories_skips_hidden(isolated_catalog):
    (isolated_catalog / "_hidden").mkdir()
    (isolated_catalog / ".dotdir").mkdir()
    cats = cat.list_categories()
    assert "_hidden" not in cats
    assert ".dotdir" not in cats


def test_write_and_read_entry_roundtrip(isolated_catalog):
    payload = {"hosiery": {"type": "stockings", "opacity": "sheer"}}
    assert cat.write_entry("garments", "hosiery_test", payload) is True
    out = cat.read_entry("garments", "hosiery_test")
    assert out == payload


def test_list_entries_after_write(isolated_catalog):
    cat.write_entry("garments", "alpha", {"k": "v"})
    cat.write_entry("garments", "beta", {"k": "v"})
    entries = cat.list_entries("garments")
    assert "alpha" in entries
    assert "beta" in entries


def test_read_missing_returns_none(isolated_catalog):
    assert cat.read_entry("garments", "does_not_exist") is None


def test_delete_entry(isolated_catalog):
    cat.write_entry("garments", "tmp_entry", {"x": 1})
    assert cat.delete_entry("garments", "tmp_entry") is True
    assert "tmp_entry" not in cat.list_entries("garments")
    # Second delete returns False
    assert cat.delete_entry("garments", "tmp_entry") is False


def test_path_traversal_blocked(isolated_catalog):
    """Reject path-traversal attempts in category and entry names."""
    assert cat.write_entry("../etc", "evil", {}) is False
    assert cat.write_entry("garments", "../../../etc/passwd", {}) is False
    assert cat.read_entry("../etc", "evil") is None
    assert cat.list_entries("../etc") == []
    assert cat.delete_entry("../etc", "evil") is False


def test_hidden_names_rejected(isolated_catalog):
    assert cat.write_entry("garments", "_secret", {}) is False
    assert cat.write_entry("_hidden_cat", "x", {}) is False


def test_empty_category_returns_empty_list(isolated_catalog):
    """A clean category subdir with no .json files yields empty list."""
    assert cat.list_entries("scenes") == []


def test_list_entries_skips_non_json(isolated_catalog):
    sub = isolated_catalog / "garments"
    (sub / "alpha.json").write_text('{"a": 1}', encoding="utf-8")
    (sub / "beta.txt").write_text("not json", encoding="utf-8")
    (sub / "_hidden.json").write_text('{"x": 1}', encoding="utf-8")
    entries = cat.list_entries("garments")
    assert entries == ["alpha"]


def test_corrupt_json_returns_none_not_raise(isolated_catalog):
    sub = isolated_catalog / "garments"
    (sub / "broken.json").write_text("{ this is not json", encoding="utf-8")
    assert cat.read_entry("garments", "broken") is None
