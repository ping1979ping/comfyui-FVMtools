"""Loads garment, fabric, and harmony data from .txt list files."""

import os
from typing import Optional

from .config import get_config, _get_project_root


def _get_lists_path():
    """Get the lists directory path. Checks outfit_config.ini first, falls back to outfit_lists/."""
    config = get_config()
    custom = config.get("paths", "custom_lists_path", fallback="").strip()
    if custom and os.path.isdir(custom):
        return custom

    return os.path.join(_get_project_root(), "outfit_lists")


def get_available_sets():
    """Scan outfit_lists/ recursively for outfit sets.

    A directory is a "set" if it contains fabrics.txt (the structural anchor
    file every outfit set has). Returned names are POSIX-form paths relative
    to outfit_lists/ root, supporting both flat legacy slugs ("casual_female")
    and hierarchical paths ("female/casual/general"). Hidden (.) and private
    (_) dirs are skipped.
    """
    lists_path = _get_lists_path()
    if not os.path.isdir(lists_path):
        return []

    sets = []
    for dirpath, dirnames, filenames in os.walk(lists_path):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith("_") and not d.startswith(".")]
        if "fabrics.txt" in filenames:
            rel = os.path.relpath(dirpath, lists_path)
            if rel != ".":
                sets.append(rel.replace(os.sep, "/"))
            dirnames.clear()
    return sorted(sets)


def _resolve_legacy_outfit_slug(outfit_set: str, lists_path: str) -> Optional[str]:
    """Map an old flat slug like 'business_female_dress' to its new hierarchical
    path 'female/business/dress' if disk has it. Used as fallback so saved
    workflow JSONs from before the hierarchy migration keep working.

    Pattern: legacy slugs always contain 'female' or 'male' as a token. Tokens
    before gender = style; tokens after gender = sub-variant (defaults to
    'general' if absent). Multiple plausible style/sub splits are tried; the
    first one that resolves to an existing directory is returned.
    """
    if "/" in outfit_set:
        return None
    tokens = outfit_set.split("_")
    if "female" in tokens:
        gender = "female"
    elif "male" in tokens:
        gender = "male"
    else:
        return None
    gender_idx = tokens.index(gender)
    combined = tokens[:gender_idx] + tokens[gender_idx + 1:]
    if not combined:
        return None
    candidates = []
    if len(combined) == 1:
        candidates.append(f"{gender}/{combined[0]}/general")
        candidates.append(f"{gender}/{combined[0]}")
    else:
        for split_at in range(1, len(combined)):
            style = "_".join(combined[:split_at])
            sub = "_".join(combined[split_at:])
            candidates.append(f"{gender}/{style}/{sub}")
        candidates.append(f"{gender}/{'_'.join(combined)}/general")
    for cand in candidates:
        if os.path.isdir(os.path.join(lists_path, cand)):
            return cand
    return None


def _resolve_outfit_path(outfit_set: str, filename: str) -> Optional[str]:
    """Resolve a (outfit_set, filename) pair to an absolute file path.

    Tries the direct path first; if missing, applies the legacy slug fallback.
    Returns None if neither resolves to an existing file.
    """
    lists_path = _get_lists_path()
    direct = os.path.join(lists_path, outfit_set, filename)
    if os.path.isfile(direct):
        return direct
    legacy = _resolve_legacy_outfit_slug(outfit_set, lists_path)
    if legacy is None:
        return None
    fallback = os.path.join(lists_path, legacy, filename)
    if os.path.isfile(fallback):
        return fallback
    return None


def get_list_file_path(outfit_set, slot_name):
    """Returns absolute path to the .txt file for a given outfit set and slot.

    Args:
        outfit_set: e.g. "general_female" or "female/general/default"
        slot_name: e.g. "top" or "fabrics"

    Returns: str — absolute path to the .txt file (existing or planned).
    Resolves legacy flat slugs to their hierarchical equivalent when possible.
    """
    resolved = _resolve_outfit_path(outfit_set, f"{slot_name}.txt")
    if resolved is not None:
        return resolved
    lists_path = _get_lists_path()
    return os.path.join(lists_path, outfit_set, f"{slot_name}.txt")


def load_garments(slot_name, outfit_set="general_female"):
    """Load garment list for a slot from .txt file.

    Args:
        slot_name: e.g. "top", "bottom", "footwear"
        outfit_set: e.g. "general_female", "business_male"

    Returns: list of dicts:
    [{"name": "t-shirt", "probability": 0.85, "formality": (0.0, 0.3), "fabrics": ["cotton", "jersey"]}, ...]
    """
    file_path = _resolve_outfit_path(outfit_set, f"{slot_name}.txt")
    if file_path is None:
        return []

    garments = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Lines starting with # are comments, UNLESS they contain pipe-separated data
            # (garment lines start with #color# tag followed by garment_name | prob | ...)
            if line.startswith("#") and "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue

            try:
                name = parts[0]
                probability = float(parts[1])
                formality_parts = parts[2].split("-")
                formality = (float(formality_parts[0]), float(formality_parts[1]))
                fabrics = [f.strip() for f in parts[3].split(",") if f.strip()]

                garments.append({
                    "name": name,
                    "probability": probability,
                    "formality": formality,
                    "fabrics": fabrics,
                })
            except (ValueError, IndexError):
                continue

    return garments


def load_fabrics(outfit_set="general_female"):
    """Load fabric database from fabrics.txt within an outfit set.

    Args:
        outfit_set: e.g. "general_female", "business_male"

    Returns: dict {"cotton": {"formality": 0.3, "family": "natural", "weight": "light"}, ...}
    """
    file_path = _resolve_outfit_path(outfit_set, "fabrics.txt")
    if file_path is None:
        return {}

    fabrics = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue

            try:
                name = parts[0]
                formality = float(parts[1])
                family = parts[2]
                weight = parts[3]

                fabrics[name] = {
                    "formality": formality,
                    "family": family,
                    "weight": weight,
                }
            except (ValueError, IndexError):
                continue

    return fabrics


def load_prints(outfit_set="general_female"):
    """Load prints/patterns/logos from prints.txt.

    Returns: list of dicts:
    [{"name": "floral print", "probability": 0.5, "slots": ["top","bottom"], "formality": (0.1, 0.6)}, ...]
    If prints.txt doesn't exist, returns empty list.
    """
    file_path = _resolve_outfit_path(outfit_set, "prints.txt")
    if file_path is None:
        return []

    prints = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue

            try:
                name = parts[0]
                probability = float(parts[1])
                slots = [s.strip() for s in parts[2].split(",") if s.strip()]
                formality_parts = parts[3].split("-")
                formality = (float(formality_parts[0]), float(formality_parts[1]))

                prints.append({
                    "name": name,
                    "probability": probability,
                    "slots": slots,
                    "formality": formality,
                })
            except (ValueError, IndexError):
                continue

    return prints


def load_texts(outfit_set="general_female"):
    """Load text/slogan data from texts.txt.

    Returns: list of dicts:
    [{"text": '"REBEL"', "probability": 0.3, "slots": ["top"], "font": "gothic font"}, ...]
    If texts.txt doesn't exist, returns empty list.
    """
    file_path = _resolve_outfit_path(outfit_set, "texts.txt")
    if file_path is None:
        return []

    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue

            try:
                text = parts[0]
                probability = float(parts[1])
                slots = [s.strip() for s in parts[2].split(",") if s.strip()]
                font = parts[3]

                texts.append({
                    "text": text,
                    "probability": probability,
                    "slots": slots,
                    "font": font,
                })
            except (ValueError, IndexError):
                continue

    return texts


def load_fabric_harmony():
    """Load fabric harmony rules from fabric_harmony.txt (global, at outfit_lists/ root).
    Returns: dict {"luxury": ["luxury", "natural"], ...}
    """
    lists_path = _get_lists_path()
    file_path = os.path.join(lists_path, "fabric_harmony.txt")

    if not os.path.isfile(file_path):
        return {}

    harmony = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue

            family = parts[0]
            compatible = [f.strip() for f in parts[1].split(",") if f.strip()]
            harmony[family] = compatible

    return harmony
