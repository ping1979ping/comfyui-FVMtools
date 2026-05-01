"""FVM_JB_Builder — universal JSON prompt builder.

The visible UI is the JS row widget (``web/js/fvm_jb_builder.js``):
  - A stack of ``[key] [value] [⋮]`` rows with indent-based nesting.
  - ``+ Add Row`` button, drag-handle reordering, Insert From Catalog,
    Edit Catalog modal, Edit Wildcards modal, ``__wildcard__`` autocomplete.
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

Wildcard expansion — every string leaf is run through
``core.jb.wildcards.resolve_text`` so ``__name__`` patterns expand to a
random line from ``<wildcards-root>/<name>.txt``. The seed is salted
with the leaf's path so two distinct row values draw independently
under the same base seed.
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
    from ...core.jb.wildcards import resolve_text
except ImportError:  # pragma: no cover
    from core.jb.serialize import (
        ALL_FORMATS,
        emit,
        emit_strict_json,
        rows_to_dict,
    )
    from core.jb.wildcards import resolve_text


def _is_row_list(payload) -> bool:
    """A row-list is a JSON array where every element has a 'key' field."""
    if not isinstance(payload, list) or not payload:
        return False
    return all(isinstance(r, dict) and "key" in r for r in payload)


def _resolve_leaves(node, seed, context, path=""):
    """Walk a dict/list tree and resolve wildcards in every string leaf.

    The leaf's path is mixed into the resolver salt so distinct call
    sites under the same base seed pick different lines from the same
    wildcard file. Mutates ``node`` in place when it's a dict/list.
    """
    if isinstance(node, dict):
        for k, v in list(node.items()):
            sub = f"{path}.{k}" if path else k
            node[k] = _resolve_leaves(v, seed, context, sub)
        return node
    if isinstance(node, list):
        for i, v in enumerate(node):
            node[i] = _resolve_leaves(v, seed, context, f"{path}[{i}]")
        return node
    if isinstance(node, str) and any(tok in node for tok in ("__", "{", "##")):
        resolved, _ = resolve_text(node, seed, context, salt=path)
        return resolved
    return node


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
        "or insert a snippet from the catalog. Row values may contain\n"
        "``__wildcard__`` tokens that expand from text files in the\n"
        "wildcards directory. Emits raw_json (strict) and string\n"
        "(chosen format)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Hidden row-list field — the JS widget owns the visible UI
                # (rows + Insert From Catalog + Edit Catalog) and writes a
                # JSON-encoded list of {key,value,indent} into this string.
                "rows": ("STRING", {"default": "[]", "multiline": True}),
                # Seed renders above output_format because dict order
                # determines the widget order in the node body.
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed for wildcards. Same seed + same rows + "
                               "same wildcards → identical output.",
                }),
                "output_format": (list(ALL_FORMATS), {"default": "loose_keys"}),
            },
            "optional": {
                # Adaptiveprompts-compatible variable bag, shape:
                # {var_name: {origin_key: value_str, ...}, ...}
                # Wire this to the ``context`` output of any
                # adaptiveprompts PromptGenerator / PromptGeneratorAdvanced
                # node. Plain STRING / INT / etc. won't connect — that's
                # by design (it's a typed dict, not free text).
                "context_from_prompt_generator": ("DICT", {
                    "tooltip": "Optional variable bag from an "
                               "adaptiveprompts PromptGenerator's "
                               "`context` output. Enables `__^var__` "
                               "recall of values previously bound with "
                               "`__name^var__`.",
                }),
            },
        }

    def build(self, rows, seed, output_format,
              context_from_prompt_generator=None):
        payload: dict = {}
        if isinstance(rows, str) and rows.strip():
            try:
                row_list = json.loads(rows)
                if _is_row_list(row_list):
                    payload = rows_to_dict(row_list)
            except (json.JSONDecodeError, TypeError):
                payload = {}

        # Expand wildcards in every string leaf. Done before emitting so
        # the JSON output already contains the resolved values.
        _resolve_leaves(payload, seed, context_from_prompt_generator)

        raw_json = emit_strict_json(payload, indent=2)
        string_out = emit(payload, output_format)
        return (raw_json, string_out)
