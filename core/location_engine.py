"""Location generation engine for the StructPromptMaker (SMP) pipeline.

Mirrors the data-format conventions of ``core/outfit_engine.py`` and
``core/outfit_lists.py``:

  * Each location set is a directory under ``location_lists/<set_name>/``
    containing one ``.txt`` file per element id.
  * File format (one entry per non-comment line):

        name | probability | coverage_range | texture

  * ``coverage_range`` is ``"a-b"`` (e.g. ``"0.4-1.0"``) or ``"-"`` to
    indicate the value is irrelevant for that element kind (atmosphere
    elements like time_of_day / weather).
  * ``texture`` is a free-form phrase or ``"-"``.
  * Lines starting with ``#`` and blank lines are ignored.

The ``prompt_fragment`` string is built with ``#ambient_light#`` or
``#shadow_tone#`` tokens still embedded — the LocationCombiner resolves
them with values from a ``COLOR_PALETTE_DICT``.
"""

from __future__ import annotations

import os
import random
from typing import Optional

from .smp.defaults import DEFAULT_LOCATION_LAYERS


ELEMENT_ORDER = [
    "background",
    "midground",
    "architecture_detail",
    "props",
    "foreground_element",
    "time_of_day",
    "weather",
]

ATMOSPHERE_ELEMENTS = {"time_of_day", "weather"}

ELEMENT_LAYER = {
    "background":          "background",
    "midground":           "midground",
    "architecture_detail": "midground",
    "props":               "midground",
    "foreground_element":  "foreground",
    "time_of_day":         "atmosphere",
    "weather":             "atmosphere",
}


# ─── data location ──────────────────────────────────────────────────────


def _location_lists_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "location_lists"))


def get_available_location_sets() -> list[str]:
    """Return location set directory names available on disk (sorted)."""
    root = _location_lists_root()
    if not os.path.isdir(root):
        return []
    sets = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and not entry.startswith("_") and not entry.startswith("."):
            sets.append(entry)
    return sets


# ─── parsing ──────────────────────────────────────────────────────────


def _parse_coverage(token: str) -> tuple[float, float]:
    token = token.strip()
    if not token or token == "-":
        return (0.0, 0.0)
    if "-" in token:
        a, b = token.split("-", 1)
        try:
            return (max(0.0, float(a.strip())), min(1.0, float(b.strip())))
        except ValueError:
            return (0.0, 0.0)
    try:
        v = float(token)
        return (v, v)
    except ValueError:
        return (0.0, 0.0)


def load_location_elements(element_id: str, location_set: str) -> list[dict]:
    """Load entries for a single element file in a location set.

    Returns: list of {name, probability, coverage_range, texture}.
    """
    root = _location_lists_root()
    path = os.path.join(root, location_set, f"{element_id}.txt")
    if not os.path.isfile(path):
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            name = parts[0]
            try:
                probability = float(parts[1]) if parts[1] not in ("", "-") else 1.0
            except ValueError:
                probability = 1.0
            coverage_range = _parse_coverage(parts[2]) if len(parts) > 2 else (0.0, 0.0)
            texture = parts[3] if len(parts) > 3 and parts[3] not in ("", "-") else None
            entries.append({
                "name":            name,
                "probability":     probability,
                "coverage_range":  coverage_range,
                "texture":         texture,
            })
    return entries


# ─── fragment templates ─────────────────────────────────────────────────


def _build_fragment(element_id: str, name: str, texture: Optional[str]) -> str:
    """Compose the per-element prompt fragment with #atmosphere# tokens."""
    if element_id in ATMOSPHERE_ELEMENTS:
        return name

    if element_id == "background":
        if texture:
            return f"{name}, {texture}, illuminated by #ambient_light#"
        return f"{name}, illuminated by #ambient_light#"

    if element_id == "foreground_element":
        if texture:
            return f"{name}, {texture}, in #shadow_tone#"
        return f"{name} in #shadow_tone#"

    # Midground / architecture / props
    if texture:
        return f"{name}, {texture}"
    return name


# ─── generator ──────────────────────────────────────────────────────────


def generate_location_records(seed: int, location_set: str = "urban_brutalist",
                               element_enables: Optional[dict[str, bool]] = None,
                               color_tone: Optional[str] = None) -> dict:
    """Pick one entry per enabled element, build prompt fragments with tokens.

    Returns: dict {
        "seed", "location_set", "color_tone",
        "elements": dict[element_id, record],
    }

    Each record is {name, probability, coverage, texture, layer,
                   prompt_fragment, region_hint}.

    Determinism: identical seed + identical inputs → identical dict.
    """
    rng = random.Random(seed)

    if element_enables is None:
        element_enables = {
            "background":          True,
            "midground":            False,
            "architecture_detail":  False,
            "props":                False,
            "foreground_element":   True,
            "time_of_day":          True,
            "weather":              True,
        }

    elements_out: dict[str, dict] = {}

    # Iterate in canonical order so RNG consumption is stable.
    for element_id in ELEMENT_ORDER:
        if not element_enables.get(element_id, False):
            # No RNG draw for disabled elements — keeps determinism predictable
            # at the slot level.
            continue
        candidates = load_location_elements(element_id, location_set)
        if not candidates:
            continue

        weights = [c["probability"] for c in candidates]
        chosen = rng.choices(candidates, weights=weights, k=1)[0]

        cov_lo, cov_hi = chosen["coverage_range"]
        coverage = cov_lo if cov_lo == cov_hi else rng.uniform(cov_lo, cov_hi)

        layer = ELEMENT_LAYER.get(element_id, "midground")
        region_hint_template = DEFAULT_LOCATION_LAYERS.get(element_id, {})
        region_hint = {
            "region_id":      element_id,
            "sam_class_hint": region_hint_template.get("sam_class") or element_id,
            "bbox_relative":  region_hint_template.get("bbox"),
            "layer_depth":    region_hint_template.get("layer_depth", layer),
        }

        prompt_fragment = _build_fragment(element_id, chosen["name"], chosen["texture"])

        elements_out[element_id] = {
            "name":            chosen["name"],
            "probability":     chosen["probability"],
            "coverage":        round(coverage, 4),
            "texture":         chosen["texture"],
            "layer":           layer,
            "prompt_fragment": prompt_fragment,
            "region_hint":     region_hint,
        }

    return {
        "seed":          seed,
        "location_set":  location_set,
        "color_tone":    color_tone,
        "elements":      elements_out,
    }
