"""FVM_JB_Extractor — recursive key search + dot-path lookup.

Three lookup modes auto-selected by the ``category`` value:

  Single key  (``"face"``)        → recursive depth-first search through
                                     every nested dict; returns the FIRST
                                     match wrapped as ``{key: value}``.

  Dot-path    (``"a.b.c"``)        → strict descent along the explicit path.
                                     Returns ``{c: value}`` (wrapped under
                                     the LAST segment).

  Multi       (``"face, hair"``    → run each category separately and
   newline   /  ``"face\\nhair"``)   combine the results into a single
   /comma   /                       dict keyed by each match's last
   /semicolon)                      segment. Any category not found is
                                     silently skipped; ``found=True`` as
                                     long as at least one matched.

  Empty       (``""``)             → returns the whole input unwrapped.

  Examples
  --------
  Input::

      {"character_1": {"face": {"age": "twenties", "eyes": "amber"},
                        "hair": {"colour": "blonde"}}}

  category="face" → ``{"face": {"age": "twenties", "eyes": "amber"}}``
  category="face, hair" →
      ``{"face": {"age": "twenties", "eyes": "amber"},
        "hair": {"colour": "blonde"}}``
  category="character_1.face" → same as ``face``.

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


def _split_categories(category: str) -> list[str]:
    """Split user-entered categories on common separators.

    Newlines, commas, and semicolons all act as separators so the user
    can type ``face, hair`` or write each category on its own line in a
    multiline widget without thinking about it. Dots inside a category
    are preserved — they're dot-path segments, not separators.
    """
    if not isinstance(category, str) or not category.strip():
        return []
    cleaned = category.replace("\r", "\n").replace(";", "\n").replace(",", "\n")
    return [c.strip() for c in cleaned.split("\n") if c.strip()]


def _merge_into(combined: dict, key: str, value):
    """Slot ``value`` under ``key`` in ``combined``.

    If both the existing and incoming values are dicts the keys are
    merged shallowly (incoming wins on direct collisions). Otherwise the
    incoming value replaces the existing one.
    """
    existing = combined.get(key)
    if isinstance(existing, dict) and isinstance(value, dict):
        merged = dict(existing)
        merged.update(value)
        combined[key] = merged
    else:
        combined[key] = value


class FVM_JB_Extractor:
    """Pull a named subtree out of a JSON string by recursive key search."""

    CATEGORY = "FVM Tools/JB"
    FUNCTION = "extract"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("raw_json", "string", "found")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Pulls one or more named subtrees out of a JSON string.\n\n"
        "Modes — auto-selected by the category value:\n"
        "  • single key   → recursive search through every nesting level,\n"
        "                   returns the first match wrapped as {key: value}.\n"
        "  • dot-path     → strict descent (e.g. 'character_1.hosiery'),\n"
        "                   returns {last_segment: value}.\n"
        "  • multi        → comma / newline / semicolon separated list;\n"
        "                   each category is resolved independently and\n"
        "                   the results are merged into one {key: value}\n"
        "                   dict. Categories that aren't found are\n"
        "                   skipped silently.\n"
        "  • empty        → returns the whole input unwrapped.\n\n"
        "found=True if at least one category matched."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input":    ("STRING", {"forceInput": True}),
                # Multiline so the user can list categories on separate
                # lines as well as comma-separated on one line.
                "category":      ("STRING", {"default": "", "multiline": True}),
                "output_format": (list(ALL_FORMATS), {"default": "loose_keys"}),
            },
        }

    def extract(self, json_input, category, output_format):
        parsed = parse_input(json_input) if isinstance(json_input, str) else json_input
        if not isinstance(parsed, (dict, list)):
            return ("", "", False)

        cats = _split_categories(category)

        if not cats:
            # Empty category → the whole input unwrapped (legacy behavior).
            out_obj = parsed
        elif len(cats) == 1:
            found, value, wrap_key = _resolve_category(parsed, cats[0])
            if not found:
                return ("", "", False)
            out_obj = {wrap_key: value} if wrap_key is not None else value
        else:
            # Multi-category — resolve each and merge into one dict.
            combined: dict = {}
            any_found = False
            for cat in cats:
                found, value, wrap_key = _resolve_category(parsed, cat)
                if not found or wrap_key is None:
                    continue
                any_found = True
                _merge_into(combined, wrap_key, value)
            if not any_found:
                return ("", "", False)
            out_obj = combined

        if isinstance(out_obj, (dict, list)):
            raw = emit_strict_json(out_obj, indent=2)
        else:
            raw = emit_strict_json(out_obj, indent=None)
        string_out = emit(out_obj, output_format)
        return (raw, string_out, True)
