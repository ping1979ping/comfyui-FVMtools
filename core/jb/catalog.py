"""JB catalog — directory walker for ``prompt_catalog/<category>/<name>.json``.

Catalog layout mirrors ``outfit_lists/`` and ``location_lists/``:

  prompt_catalog/
  ├── faces/
  │   └── east_asian_young_woman.json
  ├── garments/
  │   └── hosiery_sheer_seamed.json
  ├── locations/
  ├── scenes/
  └── props/

Each ``.json`` file holds one snippet of structured prompt data the
JB Builder pastes as rows when picked from the Insert From Catalog
dropdown. The Edit Catalog modal calls these helpers via HTTP routes
registered in ``__init__.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_CATEGORIES = ("faces", "garments", "locations", "scenes", "props")


def catalog_root() -> str:
    """Return the absolute path to ``prompt_catalog/``.

    Auto-creates the directory + default-category subdirs on first call so
    the modal never sees a missing tree.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", "prompt_catalog"))
    if not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)
    for cat in DEFAULT_CATEGORIES:
        sub = os.path.join(root, cat)
        if not os.path.isdir(sub):
            os.makedirs(sub, exist_ok=True)
    return root


def list_categories() -> list[str]:
    """List visible category subdirectories under ``prompt_catalog/``."""
    root = catalog_root()
    cats = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and not entry.startswith(("_", ".")):
            cats.append(entry)
    return cats


def list_entries(category: str) -> list[str]:
    """List snippet names (filenames without ``.json``) in a category."""
    if not _is_safe_name(category):
        return []
    root = catalog_root()
    sub = os.path.join(root, category)
    if not os.path.isdir(sub):
        return []
    out = []
    for f in sorted(os.listdir(sub)):
        if f.endswith(".json") and not f.startswith(("_", ".")):
            out.append(f[:-5])
    return out


def read_entry(category: str, name: str) -> Any:
    """Read a snippet's JSON content. Returns ``None`` if missing."""
    path = _entry_path(category, name)
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return None


def write_entry(category: str, name: str, data: Any) -> bool:
    """Write a snippet's JSON content. Auto-creates the category dir."""
    if not _is_safe_name(category) or not _is_safe_name(name):
        return False
    root = catalog_root()
    sub = os.path.join(root, category)
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return True


def delete_entry(category: str, name: str) -> bool:
    """Delete a snippet file. Returns True if the file existed and was removed."""
    path = _entry_path(category, name)
    if path is None or not os.path.isfile(path):
        return False
    os.remove(path)
    return True


# ─── helpers ───────────────────────────────────────────────────────────


def _is_safe_name(name: str) -> bool:
    """Reject path-traversal / weird characters in category and entry names."""
    if not isinstance(name, str) or not name:
        return False
    if "/" in name or "\\" in name or ".." in name:
        return False
    if name.startswith(("_", ".")):
        return False
    return True


def _entry_path(category: str, name: str) -> str | None:
    if not _is_safe_name(category) or not _is_safe_name(name):
        return None
    root = catalog_root()
    return os.path.join(root, category, f"{name}.json")
