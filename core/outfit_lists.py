"""Loads garment, fabric, and harmony data from .txt list files."""

import configparser
import os


def _get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_lists_path():
    """Get the lists directory path. Checks outfit_config.ini first, falls back to outfit_lists/."""
    root = _get_project_root()
    config_path = os.path.join(root, "outfit_config.ini")

    if os.path.isfile(config_path):
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        custom = config.get("paths", "custom_lists_path", fallback="").strip()
        if custom and os.path.isdir(custom):
            return custom

    return os.path.join(root, "outfit_lists")


def get_available_sets():
    """Scan outfit_lists/ for subdirectories and return sorted names.

    Returns: list of str, e.g. ["business_female", "business_male", "general_female", "general_male"]
    """
    lists_path = _get_lists_path()
    if not os.path.isdir(lists_path):
        return []

    sets = []
    for entry in os.listdir(lists_path):
        full_path = os.path.join(lists_path, entry)
        if os.path.isdir(full_path):
            sets.append(entry)

    return sorted(sets)


def get_list_file_path(outfit_set, slot_name):
    """Returns absolute path to the .txt file for a given outfit set and slot.

    Args:
        outfit_set: e.g. "general_female"
        slot_name: e.g. "top" or "fabrics"

    Returns: str — absolute path to the .txt file
    """
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
    lists_path = _get_lists_path()
    file_path = os.path.join(lists_path, outfit_set, f"{slot_name}.txt")

    if not os.path.isfile(file_path):
        return []

    garments = []
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
    lists_path = _get_lists_path()
    file_path = os.path.join(lists_path, outfit_set, "fabrics.txt")

    if not os.path.isfile(file_path):
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
