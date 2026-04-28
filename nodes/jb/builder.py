"""FVM_JB_Builder — universal JSON prompt builder.

The visible UI is the JS row widget (``web/js/fvm_jb_builder.js``):
  - A stack of ``[key] [value] [⋮]`` rows with indent-based nesting.
  - ``+ Add Row`` button, drag-handle reordering, Insert From Catalog,
    Edit Catalog modal.
  - The widget serialises its state into the hidden ``rows`` STRING
    widget that this node consumes.

Row-list contract (what the JS writes into ``rows``)::

    [
      {"key": "hosiery", "value": "", "indent": 0},
      {"key": "type",    "value": "sheer black stockings", "indent": 1},
      ...
    ]

A row whose value is empty AND has children at indent+1 is a branch
(object container). A leaf row sets the parent[key] to its value.
"""

from __future__ import annotations

import json

try:
    from ...core.jb.serialize import (
        ALL_FORMATS,
        emit,
        emit_strict_json,
        rows_to_dict,
    )
except ImportError:  # pragma: no cover
    from core.jb.serialize import (
        ALL_FORMATS,
        emit,
        emit_strict_json,
        rows_to_dict,
    )


def _is_row_list(payload) -> bool:
    """A row-list is a JSON array where every element has a 'key' field."""
    if not isinstance(payload, list) or not payload:
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
        "Hand-author a tree of key:value rows with indent-based nesting,\n"
        "or insert a snippet from the catalog. Emits raw_json (strict)\n"
        "and string (chosen format)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Hidden row-list field — the JS widget owns the visible UI
                # (rows + Insert From Catalog + Edit Catalog) and writes a
                # JSON-encoded list of {key,value,indent} into this string.
                "rows": ("STRING", {"default": "[]", "multiline": True}),
                "output_format": (list(ALL_FORMATS), {"default": "loose_keys"}),
            },
        }

    def build(self, rows, output_format):
        payload: dict = {}
        if isinstance(rows, str) and rows.strip():
            try:
                row_list = json.loads(rows)
                if _is_row_list(row_list):
                    payload = rows_to_dict(row_list)
            except (json.JSONDecodeError, TypeError):
                payload = {}

        raw_json = emit_strict_json(payload, indent=2)
        string_out = emit(payload, output_format)
        return (raw_json, string_out)
