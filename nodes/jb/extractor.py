"""FVM_JB_Extractor — recursive key search + dot-path lookup.

Two lookup modes auto-selected by the ``category`` value:

  Single key  (``"face"``)        → recursive depth-first search through
                                     every nested dict; returns the FIRST
                                     match wrapped as ``{key: value}``.

  Dot-path    (``"a.b.c"``)        → strict descent along the explicit path.
                                     Returns ``{c: value}`` (wrapped under
                                     the LAST segment).

  Empty       (``""``)             → returns the whole input unwrapped.

  Examples
  --------
  Input::

      {"character_1": {"face": {"age": "twenties", "eyes": "amber"}}}

  category="face" → ``{"face": {"age": "twenties", "eyes": "amber"}}``
  category="character_1.face" → same as above.
  category="character_1" → ``{"character_1": {...}}``.

Missing key → empty outputs + ``found=False``.
"""

from __future__ import annotations

try:
    from ...core.jb.serialize import (
        ALL_FORMATS,
        emit,
        emit_strict_json,
        parse_input,
    )
except ImportError:  # pragma: no cover
    from core.jb.serialize import (
        ALL_FORMATS,
        emit,
        emit_strict_json,
        parse_input,
    )


def _walk_dot_path(payload, parts: list[str]):
    """Strict descent along the explicit dot-path."""
    cur = payload
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return False, None
    return True, cur


def _search_recursive(obj, key: str):
    """DFS through every nested dict / list and return the first match."""
    if isinstance(obj, dict):
        if key in obj:
            return True, obj[key]
        for v in obj.values():
            found, val = _search_recursive(v, key)
            if found:
                return True, val
    elif isinstance(obj, list):
        for v in obj:
            found, val = _search_recursive(v, key)
            if found:
                return True, val
    return False, None


def _resolve_category(payload, category: str):
    """Return (found, value, wrap_key)."""
    cat = (category or "").strip()
    if not cat:
        return True, payload, None
    if "." in cat:
        parts = [p for p in cat.split(".") if p]
        found, val = _walk_dot_path(payload, parts)
        return found, val, (parts[-1] if found else None)
    found, val = _search_recursive(payload, cat)
    return found, val, (cat if found else None)


class FVM_JB_Extractor:
    """Pull a named subtree out of a JSON string by recursive key search."""

    CATEGORY = "FVM Tools/JB"
    FUNCTION = "extract"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("raw_json", "string", "found")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Pulls a named subtree out of a JSON string.\n\n"
        "Modes — auto-selected by the category value:\n"
        "  • single key   → recursive search through every nesting level,\n"
        "                   returns the first match wrapped as {key: value}.\n"
        "  • dot-path     → strict descent (e.g. 'character_1.hosiery'),\n"
        "                   returns {last_segment: value}.\n"
        "  • empty        → returns the whole input unwrapped.\n\n"
        "Missing key → empty outputs and found=False."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input":    ("STRING", {"forceInput": True}),
                "category":      ("STRING", {"default": ""}),
                "output_format": (list(ALL_FORMATS), {"default": "loose_keys"}),
            },
        }

    def extract(self, json_input, category, output_format):
        parsed = parse_input(json_input) if isinstance(json_input, str) else json_input
        if not isinstance(parsed, (dict, list)):
            return ("", "", False)

        found, value, wrap_key = _resolve_category(parsed, category)
        if not found:
            return ("", "", False)

        out_obj = {wrap_key: value} if wrap_key is not None else value

        if isinstance(out_obj, (dict, list)):
            raw = emit_strict_json(out_obj, indent=2)
        else:
            raw = emit_strict_json(out_obj, indent=None)
        string_out = emit(out_obj, output_format)
        return (raw, string_out, True)
