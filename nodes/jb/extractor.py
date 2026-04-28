"""FVM_JB_Extractor — pull a sub-tree out of a JSON string by dot-path key.

Examples (input = the Stitcher's output for two characters):

  category="character_1"            → the whole character_1 object.
  category="character_1.hosiery"    → just the hosiery sub-object.
  category="character_1.hosiery.type"  → the leaf string value.

Missing path → empty raw_json string and ``found=False``.
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


def _walk_dot_path(payload, path: str):
    """Return (found, value) for a dot-separated path inside payload."""
    if not path:
        return True, payload
    parts = [p for p in path.split(".") if p]
    cur = payload
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return False, None
    return True, cur


class FVM_JB_Extractor:
    """Pull a named subtree out of a JSON string."""

    CATEGORY = "FVM Tools/JB"
    FUNCTION = "extract"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("raw_json", "string", "found")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Pulls a named subtree out of a JSON string.\n\n"
        "category supports dot-paths for nested lookup, e.g.\n"
        "  'character_1' → top-level character object\n"
        "  'character_1.hosiery' → hosiery sub-object\n"
        "  'character_1.hosiery.type' → the leaf value\n\n"
        "Missing path → empty outputs and found=False."
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

        found, value = _walk_dot_path(parsed, (category or "").strip())
        if not found:
            return ("", "", False)

        # Wrap leaf scalars so the JSON output is always valid syntax.
        if isinstance(value, (dict, list)):
            raw = emit_strict_json(value, indent=2)
            string_out = emit(value, output_format)
        else:
            raw = emit_strict_json(value, indent=None)
            string_out = emit(value, output_format)
        return (raw, string_out, True)
