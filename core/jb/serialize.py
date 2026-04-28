"""JB serialization — JSON ↔ row-list and strict/loose-keys emitters.

Internal data flow:

    user input (textarea / row widget) ── parse_input ──► dict (Python)
    dict ── emit(format) ──► string (output)

Row-list shape (used by the P4 JS widget):

    [
      {"key": "hosiery", "value": "", "indent": 0},
      {"key": "type",    "value": "sheer black stockings ...", "indent": 1},
      {"key": "opacity", "value": "semi-sheer (20-30 denier)", "indent": 1},
    ]

A row whose value is empty AND has children at indent+1 is a branch
(object container). A row with key but no children is a leaf with that
value (string-coerced unless it parses as JSON literal).
"""

from __future__ import annotations

import json
from typing import Any, Iterable

OutputFormat = str  # "pretty_json" | "compact_json" | "loose_keys"

PRETTY_JSON   = "pretty_json"
COMPACT_JSON  = "compact_json"
LOOSE_KEYS    = "loose_keys"
ALL_FORMATS   = (PRETTY_JSON, COMPACT_JSON, LOOSE_KEYS)


# ─── Parsing user input ────────────────────────────────────────────────


def parse_input(text: str) -> Any:
    """Coerce a textarea / widget string into a Python value.

    Tries strict JSON first; if that fails, tries the loose-keys form
    (auto-quoting unquoted keys); if that fails, returns the raw string.
    """
    if not isinstance(text, str):
        return text
    s = text.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try loose-keys: quote bareword keys before colons / commas.
    strict = _loose_to_strict(s)
    try:
        return json.loads(strict)
    except (json.JSONDecodeError, TypeError):
        pass
    # Loose-keys form may be missing the outer braces (`a: 1, b: 2`)
    # — wrap and retry.
    if not strict.lstrip().startswith(("{", "[")):
        try:
            return json.loads("{" + strict + "}")
        except (json.JSONDecodeError, TypeError):
            pass
    return s


def _loose_to_strict(text: str) -> str:
    """Convert a loose-keys string back into strict JSON.

    Handles `key: value` → `"key": value`. Walks the string char-by-char
    so we don't quote things inside string literals.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    string_quote = ""
    while i < n:
        ch = text[i]
        # Toggle string-literal mode on unescaped quote.
        if ch in ('"', "'"):
            if not in_string:
                in_string = True
                string_quote = ch
            elif ch == string_quote and (i == 0 or text[i - 1] != "\\"):
                in_string = False
            out.append(ch)
            i += 1
            continue
        if in_string:
            out.append(ch)
            i += 1
            continue
        # Look for a bareword followed by ":" (a key).
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] in "_-"):
                j += 1
            # Skip whitespace, then check if next non-space is ":"
            k = j
            while k < n and text[k] in " \t":
                k += 1
            if k < n and text[k] == ":":
                bareword = text[i:j]
                # Don't quote JSON literals
                if bareword in ("true", "false", "null"):
                    out.append(bareword)
                else:
                    out.append(f'"{bareword}"')
                i = j
                continue
        out.append(ch)
        i += 1
    return "".join(out)


# ─── Rows ↔ dict ───────────────────────────────────────────────────────


def rows_to_dict(rows: Iterable[dict]) -> dict:
    """Convert a row-list with indent levels into a nested dict.

    Algorithm: maintain a stack of dicts indexed by indent level. Each
    new row attaches as a child of the most recent row at indent-1.

    A row with empty value AND a following row at higher indent → branch.
    A row with empty value and no children → empty string leaf.
    A row with non-empty value → leaf (JSON-parsed if it parses, else string).
    """
    rows = list(rows)
    if not rows:
        return {}

    root: dict = {}
    # Stack of (indent_level, dict_at_that_level) entries.
    stack: list[tuple[int, dict]] = [(-1, root)]

    for idx, row in enumerate(rows):
        key = (row.get("key") or "").strip()
        value_raw = row.get("value", "")
        indent = int(row.get("indent", 0) or 0)
        if not key:
            continue

        # Pop stack until we're at the right parent indent.
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            stack = [(-1, root)]
        parent = stack[-1][1]

        # Peek at the next row to decide branch vs leaf.
        next_row = rows[idx + 1] if idx + 1 < len(rows) else None
        next_indent = int((next_row or {}).get("indent", -1))
        is_branch = (
            (value_raw is None or value_raw == "")
            and next_row is not None
            and next_indent > indent
        )

        if is_branch:
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_leaf(value_raw)

    return root


def _coerce_leaf(value_raw: Any) -> Any:
    """Convert a raw row value into a Python leaf.

    - None / empty string → empty string.
    - Pure JSON literal (number / bool / null / array / object) → parsed.
    - Otherwise → string.
    """
    if value_raw is None:
        return ""
    if not isinstance(value_raw, str):
        return value_raw
    s = value_raw.strip()
    if not s:
        return ""
    if s in ("true", "false", "null"):
        return json.loads(s)
    # Try numeric / structural literals only — DON'T parse a bare quoted word
    # like '"hello"' as JSON, since the user clearly typed the quotes literally.
    if s[0] in "-0123456789[{":
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            pass
    return value_raw


def dict_to_rows(obj: Any, indent: int = 0, key: str | None = None) -> list[dict]:
    """Inverse of rows_to_dict — flatten a nested dict for the editor.

    Used when the user picks a catalog snippet: we paste its rows into
    the Builder's row state.
    """
    out: list[dict] = []
    if key is not None:
        if isinstance(obj, dict):
            out.append({"key": key, "value": "", "indent": indent})
            for k, v in obj.items():
                out.extend(dict_to_rows(v, indent + 1, k))
            return out
        # Leaf: emit a single row with the value.
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            out.append({"key": key, "value": _leaf_repr(obj), "indent": indent})
        else:
            # Array or other — store as JSON-encoded string in the value cell.
            out.append({"key": key, "value": json.dumps(obj), "indent": indent})
        return out

    # Top-level: walk dict items
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(dict_to_rows(v, indent, k))
    return out


def _leaf_repr(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# ─── Emission ──────────────────────────────────────────────────────────


def emit(obj: Any, fmt: OutputFormat = PRETTY_JSON) -> str:
    """Emit a Python value as a string in the requested format."""
    if fmt == COMPACT_JSON:
        return emit_strict_json(obj, indent=None)
    if fmt == LOOSE_KEYS:
        return emit_loose_keys(obj)
    return emit_strict_json(obj, indent=2)


def emit_strict_json(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)


def emit_loose_keys(obj: Any, level: int = 0, _indent: int = 2) -> str:
    """Loose-keys output for SD/CLIP encoders.

    Strips **all** ``"`` characters from string values — both the JSON-
    syntactic surrounding quotes and any literal quote characters that
    were embedded in the content (e.g. text overlays in data files like
    ``"NO LIMITS"`` from texts.txt). Object/array structure is still
    emitted with ``{}`` / ``[]`` / commas; only key + value text is bare.

    Note: this format is NOT designed to round-trip back through
    ``parse_input`` — it's a one-way emit for encoder consumption.
    """
    pad = " " * (level * _indent)
    inner_pad = " " * ((level + 1) * _indent)

    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float)):
        return json.dumps(obj)
    if isinstance(obj, str):
        # Raw — no surrounding quotes. Strip all `"` chars from the value
        # itself too: the user's rule is "no quotation marks anywhere in
        # the loose_keys string output".
        return obj.replace('"', "")

    if isinstance(obj, list):
        if not obj:
            return "[]"
        items = [emit_loose_keys(v, level + 1, _indent) for v in obj]
        return "[\n" + inner_pad + (",\n" + inner_pad).join(items) + "\n" + pad + "]"

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            key_part = _bare_key(str(k))
            val_part = emit_loose_keys(v, level + 1, _indent)
            lines.append(f"{key_part}: {val_part}")
        return "{\n" + inner_pad + (",\n" + inner_pad).join(lines) + "\n" + pad + "}"

    # Fallback for other types
    return json.dumps(obj, default=str, ensure_ascii=False)


_BAREWORD_RE = None


def _bare_key(key: str) -> str:
    """Emit a key without quotes if it's a safe bareword, else fall back to JSON."""
    global _BAREWORD_RE
    if _BAREWORD_RE is None:
        import re
        _BAREWORD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
    if _BAREWORD_RE.match(key):
        return key
    return json.dumps(key, ensure_ascii=False)
