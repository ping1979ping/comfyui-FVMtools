"""JB (JSON Builder) — universal JSON-prompt assembly utilities.

The JB suite produces text-form JSON (strict or loose-keys) that you wire
directly into a CLIPTextEncode (or any other encoder). Five nodes:

  - FVM_JB_OutfitBlock    — Outfit + Color + Combiner in one node.
  - FVM_JB_LocationBlock  — Location + Combiner in one node.
  - FVM_JB_Builder        — universal hand-authored JSON tree (rows + indent).
  - FVM_JB_Stitcher       — wrap N JSON fragments under a common title (deep-merge).
  - FVM_JB_Extractor      — pull a sub-tree out by dot-path key.

This package contains the shared logic; the node classes live under nodes/jb/.
"""

from .serialize import (
    OutputFormat,
    emit,
    emit_strict_json,
    emit_loose_keys,
    parse_input,
    rows_to_dict,
    dict_to_rows,
)
from .catalog import (
    catalog_root,
    list_categories,
    list_entries,
    read_entry,
    write_entry,
    delete_entry,
)

__all__ = [
    "OutputFormat",
    "emit",
    "emit_strict_json",
    "emit_loose_keys",
    "parse_input",
    "rows_to_dict",
    "dict_to_rows",
    "catalog_root",
    "list_categories",
    "list_entries",
    "read_entry",
    "write_entry",
    "delete_entry",
]
