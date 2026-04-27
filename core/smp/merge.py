"""Deep-merge for PROMPT_DICT segments — spec §6.2.

Rules:
- dicts merge recursively (later wins per leaf).
- ``None`` (later) deletes the field on the earlier dict.
- Lists concatenate (later appended; duplicates in dict-of-dicts left intact).
- Scalars: later overwrites earlier.
"""

from __future__ import annotations

import copy
from typing import Any


def deep_merge(base: dict | None, overlay: dict | None) -> dict:
    """Return a fresh dict that is base ⊕ overlay per the rules above.

    Neither input is mutated. Caller-friendly when chaining builders.
    """
    if base is None:
        return copy.deepcopy(overlay or {})
    if overlay is None:
        return copy.deepcopy(base)

    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            out.pop(key, None)
            continue
        if key not in out:
            out[key] = copy.deepcopy(value)
            continue
        existing = out[key]
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = deep_merge(existing, value)
        elif isinstance(existing, list) and isinstance(value, list):
            out[key] = existing + copy.deepcopy(value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def merge_many(*partials: dict | None) -> dict:
    """Fold deep_merge over an arbitrary number of partials in order."""
    result: dict = {}
    for p in partials:
        result = deep_merge(result, p)
    return result
