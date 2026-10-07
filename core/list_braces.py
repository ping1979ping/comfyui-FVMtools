"""Shared ``{a|b|c}`` bracket support for pipe-delimited list files.

List files (location_lists/, outfit_lists/) use ``|`` as the field delimiter.
``split_fields`` ignores pipes inside ``{...}``, and ``resolve_braces`` expands
the brackets with the JB wildcard resolver (weights ``{%3%a|b}``, empty options
``{|a}``, nesting) from a caller-supplied seed, so the shared RNG stream of the
engines is never consumed and results stay deterministic.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    from .jb.wildcards import resolve_text as _resolve_text
except ImportError:  # pragma: no cover - direct import outside the package
    from core.jb.wildcards import resolve_text as _resolve_text

_WS_RE = re.compile(r" {2,}")


def split_fields(line: str) -> list[str]:
    """Split a list line on ``|`` — except inside ``{...}`` brackets."""
    parts, buf, depth = [], [], 0
    for ch in line:
        if ch == "{":
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts


def resolve_braces(text: Optional[str], seed: int, salt: str) -> Optional[str]:
    """Resolve ``{a|b}`` brackets in ``text`` (no-op when there are none)."""
    if not text or "{" not in text:
        return text
    resolved, _ = _resolve_text(text, seed, None, salt=salt)
    resolved = _WS_RE.sub(" ", resolved)
    return re.sub(r"\s+([,.])", lambda m: m.group(1), resolved).strip()
