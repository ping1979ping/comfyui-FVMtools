"""Shared wildcard-resolution helper for JB nodes.

Walks a parsed JSON tree (dict / list / scalars) and expands every wildcard
token in each *string* leaf via :func:`core.jb.wildcards.resolve_text`. The
leaf's JSON path is mixed into the resolver salt so distinct call-sites under
one base seed draw independently. dict / list nodes are mutated in place.

Used by the JB Builder and the Ideogram nodes so they all share one identical
resolution pass (wildcards, ``{a|b}`` brackets, ``__^var__`` recall, ``##``
comments).
"""

from __future__ import annotations

from typing import Any

try:
    from .wildcards import resolve_text
except ImportError:  # pragma: no cover - direct import outside the package
    from core.jb.wildcards import resolve_text


def resolve_leaves(node: Any, seed: int, context: Any | None = None,
                   path: str = "") -> Any:
    """Resolve wildcard tokens in every string leaf of ``node``.

    Numbers, booleans and ``None`` pass through untouched — so bbox integers
    and other structural values are never mangled. Strings without a token
    (no ``__`` / ``{`` / ``##``) are returned verbatim by ``resolve_text``.
    """
    if isinstance(node, dict):
        for k, v in list(node.items()):
            sub = f"{path}.{k}" if path else k
            node[k] = resolve_leaves(v, seed, context, sub)
        return node
    if isinstance(node, list):
        for i, v in enumerate(node):
            node[i] = resolve_leaves(v, seed, context, f"{path}[{i}]")
        return node
    if isinstance(node, str) and any(tok in node for tok in ("__", "{", "##")):
        resolved, _ = resolve_text(node, seed, context, salt=path)
        return resolved
    return node
