"""FVM_JB_Builder — universal JSON prompt builder.

P1: textarea fallback. The user types or pastes either strict JSON or a
loose-keys form into the multiline widget; the node parses it and
re-emits in the chosen output format. The custom row-based widget UI
lands in P4 — it will write a row-list JSON into the same hidden field
that this node already consumes.

P4 row-list contract (forward-compat):

    [
      {"key": "hosiery", "value": "", "indent": 0},
      {"key": "type",    "value": "sheer black stockings", "indent": 1},
      ...
    ]

If the textarea content parses as a row-list (a JSON array of objects
with at least the ``key`` field), it is converted via ``rows_to_dict``.
Otherwise it is parsed as a regular JSON object / loose-keys blob.
"""

from __future__ import annotations

import json

try:
    from ...core.jb.serialize import (
        ALL_FORMATS,
        PRETTY_JSON,
        emit,
        emit_loose_keys,
        emit_strict_json,
        parse_input,
        rows_to_dict,
    )
except ImportError:  # pragma: no cover
    from core.jb.serialize import (
        ALL_FORMATS,
        PRETTY_JSON,
        emit,
        emit_loose_keys,
        emit_strict_json,
        parse_input,
        rows_to_dict,
    )


_DEFAULT_TEXT = """{
  "hosiery": {
    "type": "sheer black stockings, dark nylon with bold seams",
    "opacity": "semi-sheer (20-30 denier), glossy finish",
    "details": "smooth texture, visible skin tone underneath"
  }
}"""


def _is_row_list(payload) -> bool:
    """A row-list is a JSON array where every element has a 'key' field."""
    if not isinstance(payload, list):
        return False
    if not payload:
        return False
    return all(isinstance(r, dict) and "key" in r for r in payload)


class FVM_JB_Builder:
    """Universal hand-authored JSON prompt builder.

    Outputs strict JSON (``raw_json``) and a chosen format (``string``,
    typically ``loose_keys`` for SD/CLIP encoders that handle bare keys).
    """

    CATEGORY = "FVM Tools/JB"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("raw_json", "string")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Universal JSON prompt builder.\n\n"
        "P1 fallback: type strict JSON or a loose-keys blob in the textarea;\n"
        "the node parses it and emits raw_json (strict) and string (chosen format).\n\n"
        "P4 will replace the textarea with a custom row-based widget that\n"
        "writes the same row-list JSON into the hidden 'rows' field."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": _DEFAULT_TEXT, "multiline": True}),
                "output_format": (list(ALL_FORMATS), {"default": "loose_keys"}),
            },
            "optional": {
                # Hidden row-list field for the P4 JS widget. Treated as a
                # JSON-encoded list of {key,value,indent}. When non-empty it
                # takes precedence over the textarea.
                "rows": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def build(self, text, output_format, rows=""):
        # Prefer the rows widget if populated.
        payload = None
        if isinstance(rows, str) and rows.strip():
            try:
                row_list = json.loads(rows)
                if _is_row_list(row_list):
                    payload = rows_to_dict(row_list)
            except (json.JSONDecodeError, TypeError):
                payload = None

        if payload is None:
            payload = parse_input(text)
            # parse_input returns the raw string when nothing parses; in that
            # case wrap it so we still emit something sensible.
            if isinstance(payload, str):
                payload = {"_raw_text": payload}

        raw_json = emit_strict_json(payload, indent=2)
        string_out = emit(payload, output_format)
        return (raw_json, string_out)
