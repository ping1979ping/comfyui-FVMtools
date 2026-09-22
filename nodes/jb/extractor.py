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

``collect_all`` flips the search from first-match to every-match: the
category resolves to a *list* of all values found under that key at any
depth, and a dot-path descends strictly along all but its last segment
(scoping the search) before collecting. Multi-category works as usual —
each key keeps its own wrap key::

    category="prompt_fragment", collect_all=True
    → raw_json {"prompt_fragment": ["wallpapered feature wall, ...",
                                     "shag area rug underfoot, ...", ...]}
    → string   "wallpapered feature wall, ..., shag area rug underfoot, ..."

``with_keys`` (collect_all only) labels every value with the key of the
dict that owns it, so identical search keys stay distinguishable::

    category="prompt_fragment", collect_all=True, with_keys=True
    → raw_json {"prompt_fragment": {"background": "wallpapered ...",
                                     "midground":  "shag area rug ..."}}
    → string   "background: wallpapered ..., midground: shag area rug ..."

The ``string`` output in collect_all mode is never the half-quoted
loose_keys form: ``loose_keys`` emits plain text, the two JSON formats
emit strict JSON.

Negative list — any category line prefixed with ``[NOT]`` (case-insensitive)
is an *exclusion* instead of a pull. Exclusions mirror the positive match
modes and are pruned from the SOURCE tree before extraction runs:

  ``[NOT]colour``        (bare key)  → recursively removes every ``colour``
                                       key — and its subtree — at any depth.
  ``[NOT]hair.colour``   (dot-path)  → removes only that exact path.

Combining them::

    face                 → pull face …
    hair                 … and hair …
    [NOT]hair.colour     … but drop hair.colour from the result.

With only exclusions (no positive lines) the whole document is returned
minus the excluded keys ("everything except X"). If the exclusions prune
the result down to nothing, the outputs are empty and ``found=False``.

Missing key → empty outputs + ``found=False``.
"""

from __future__ import annotations

try:
    from ...core.jb.serialize import (
        ALL_FORMATS,
        LOOSE_KEYS,
        emit,
        emit_strict_json,
        parse_input,
    )
except ImportError:  # pragma: no cover
    from core.jb.serialize import (
        ALL_FORMATS,
        LOOSE_KEYS,
        emit,
        emit_strict_json,
        parse_input,
    )


# Shown greyed-out in the empty category box — the syntax cheat-sheet.
_CATEGORY_PLACEHOLDER = (
    "face                     one key, searched at any depth\n"
    "face, hair               several keys (comma / semicolon / new line)\n"
    "character_1.face         dot-path, strict from the root\n"
    "character_1.face.eyes    down to a single leaf\n"
    "(empty)                  pass the whole document through\n"
    "\n"
    "with collect_all ON:\n"
    "prompt_fragment          every match, anywhere\n"
    "input_4.prompt_fragment  scoped: search only inside input_4\n"
    "\n"
    "[NOT]hair.colour         exclude: drop that path from the result"
)

_CATEGORY_TOOLTIP = (
    "What to pull out. Four syntaxes, auto-detected:\n\n"
    "  face                    → single key, recursive search through every\n"
    "                            nesting level. Wraps as {face: ...}.\n"
    "  face, hair              → several keys at once. Separators: comma,\n"
    "                            semicolon, or one per line. Keys that aren't\n"
    "                            found are skipped silently.\n"
    "  character_1.face        → dot-path, strict descent from the root.\n"
    "                            Wraps under the LAST segment: {face: ...}.\n"
    "  (empty)                 → the whole document, unwrapped.\n\n"
    "Dots are path separators, never list separators — 'a.b, c.d' is two paths.\n\n"
    "With collect_all ON the meaning of a dot-path changes: all but the last\n"
    "segment is a strict descent that SCOPES the search, and the last segment "
    "is\nthen collected recursively below it. So 'input_4.prompt_fragment' "
    "sweeps\nevery prompt_fragment inside input_4 only.\n\n"
    "Careful: two dot-paths ending in the same segment share one wrap key. "
    "Their\nvalues merge (collect_all: appended; otherwise: last one wins).\n\n"
    "Prefix a line with [NOT] to EXCLUDE instead of pull:\n"
    "  [NOT]key                → drop that key + subtree everywhere\n"
    "  [NOT]a.b.c              → drop just that path\n"
    "Exclusions are pruned from the source first. [NOT] lines alone → the\n"
    "whole document minus those keys. If they empty the result → found=False."
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


def _collect_recursive(obj, key: str, out: list, parent: str | None = None) -> None:
    """DFS collecting ``(owner, value)`` for EVERY match of ``key``, any depth.

    ``owner`` is the key of the dict the match was found in — for
    ``elements.background.prompt_fragment`` that's ``background``, which is
    what identifies a match when the searched key is the same everywhere.
    Falls back to ``key`` itself for a match at the very top level.

    A matched key is not descended into — a nested ``{"a": {"a": 1}}``
    yields the outer value once, not twice.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append((parent or key, v))
            else:
                _collect_recursive(v, key, out, k)
    elif isinstance(obj, list):
        for v in obj:
            _collect_recursive(v, key, out, parent)


def _pairs_to_dict(pairs: list) -> dict:
    """``[(owner, value), ...]`` → ``{owner: value}``.

    Repeated owners collect into a list rather than overwriting, so a
    document with two ``background`` elements keeps both values.
    """
    out: dict = {}
    for k, v in pairs:
        if k in out:
            existing = out[k]
            out[k] = existing + [v] if isinstance(existing, list) else [existing, v]
        else:
            out[k] = v
    return out


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


def _resolve_category_all(payload, category: str, with_keys: bool = False):
    """Every-match variant of :func:`_resolve_category`.

    Returns (found, values, wrap_key). ``values`` is a plain list of the
    collected values, or — with ``with_keys`` — an ``{owner: value}`` dict
    so each fragment stays labelled with the element it came from.

    A dot-path descends strictly along all segments *but the last*, then
    collects the last segment recursively inside that subtree — so
    ``input_4.prompt_fragment`` scopes the sweep to one stitcher slot.
    """
    cat = (category or "").strip()
    if not cat:
        return True, payload, None
    parts = [p for p in cat.split(".") if p]
    scope = payload
    if len(parts) > 1:
        ok, scope = _walk_dot_path(payload, parts[:-1])
        if not ok:
            return False, None, None
    pairs: list = []
    _collect_recursive(scope, parts[-1], pairs)
    if not pairs:
        return False, None, None
    values = _pairs_to_dict(pairs) if with_keys else [v for _, v in pairs]
    return True, values, parts[-1]


def _flatten_scalars(obj, out: list, key: str | None = None,
                     with_keys: bool = False) -> None:
    """Collect every non-empty scalar leaf, in document order.

    With ``with_keys`` each leaf is prefixed by the key that owns it
    (``background: wallpapered feature wall``).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_scalars(v, out, k, with_keys)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_scalars(v, out, key, with_keys)
    elif obj is not None:
        s = str(obj).strip()
        if s:
            out.append(f"{key}: {s}" if with_keys and key else s)


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
    if isinstance(existing, list) and isinstance(value, list):
        combined[key] = existing + value
    elif isinstance(existing, dict) and isinstance(value, dict):
        merged = dict(existing)
        merged.update(value)
        combined[key] = merged
    else:
        combined[key] = value


# ─── Negative list (exclusions) ───────────────────────────────────────

_NEG_PREFIX = "[not]"


def _partition_categories(category: str):
    """Split entries into (positives, negatives).

    A negative is any entry whose (case-insensitive) prefix is ``[NOT]``;
    the remainder after the prefix is the key / dot-path to exclude.
    """
    positives, negatives = [], []
    for entry in _split_categories(category):
        if entry.lower().startswith(_NEG_PREFIX):
            rest = entry[len(_NEG_PREFIX):].strip()
            if rest:
                negatives.append(rest)
        else:
            positives.append(entry)
    return positives, negatives


def _prune_key_recursive(obj, key: str) -> None:
    """Delete every ``key`` (and its subtree) at any depth, in place."""
    if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
            _prune_key_recursive(v, key)
    elif isinstance(obj, list):
        for v in obj:
            _prune_key_recursive(v, key)


def _prune_dot_path(obj, parts: list[str]) -> None:
    """Delete the exact ``a.b.c`` path in place; silent no-op if absent."""
    *head, last = parts
    cur = obj
    for p in head:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return
    if isinstance(cur, dict):
        cur.pop(last, None)


def _apply_negatives(payload, negatives: list[str]) -> None:
    """Prune every negative entry from ``payload`` in place."""
    for neg in negatives:
        if "." in neg:
            parts = [p for p in neg.split(".") if p]
            if parts:
                _prune_dot_path(payload, parts)
        else:
            _prune_key_recursive(payload, neg)


def _is_empty(obj) -> bool:
    """A container that pruning may have emptied → treat as a miss."""
    return obj is None or (isinstance(obj, (dict, list)) and len(obj) == 0)


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
        "collect_all flips first-match to EVERY-match: the category becomes a\n"
        "list of all values found under that key at any depth, and a dot-path\n"
        "scopes the sweep (all but the last segment is a strict descent).\n"
        "Several keys can be collected at once — each keeps its own wrap key.\n"
        "with_keys labels every value with the element it came from\n"
        "('background: wallpapered feature wall').\n\n"
        "In collect_all mode the `string` output is never half-quoted:\n"
        "loose_keys gives PLAIN TEXT (one flat comma-joined line, ready for a\n"
        "text encoder), the JSON formats give strict JSON.\n\n"
        "Negative list: prefix a line with [NOT] to EXCLUDE instead of\n"
        "pull — [NOT]key drops that key (and its subtree) everywhere,\n"
        "[NOT]a.b.c drops just that path. Exclusions are pruned from the\n"
        "source before extraction. [NOT] lines alone → whole doc minus\n"
        "those keys.\n\n"
        "found=True if at least one category matched (and the result\n"
        "wasn't fully pruned away)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input":    ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON to search. Wire a raw_json output here — the "
                               "loose_keys `string` output is one-way and cannot be "
                               "parsed back."}),
                # Multiline so the user can list categories on separate
                # lines as well as comma-separated on one line.
                "category":      ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": _CATEGORY_PLACEHOLDER,
                    "tooltip": _CATEGORY_TOOLTIP}),
                "output_format": (list(ALL_FORMATS), {
                    "default": "loose_keys",
                    "tooltip": "pretty_json / compact_json → strict JSON. loose_keys → "
                               "unquoted keys for SD/CLIP encoders; with collect_all it "
                               "collapses to one flat comma-joined line."}),
                "collect_all":   ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF: return the FIRST match of a key.\n"
                               "ON: collect EVERY match at any depth into a list — the "
                               "classic 'give me all prompt_fragments' case. A dot-path "
                               "then SCOPES the sweep instead of pinpointing one value: "
                               "all but the last segment is a strict descent, the last "
                               "segment is searched recursively below it.\n"
                               "With output_format=loose_keys the string output becomes "
                               "one flat comma-joined line."}),
                "with_keys":     ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only applies when collect_all is ON.\n"
                               "OFF: bare values — 'wallpapered feature wall, shag area rug'.\n"
                               "ON: each value keeps the key of the element it came from — "
                               "'background: wallpapered feature wall, midground: shag area "
                               "rug'. raw_json becomes {key: {background: ..., midground: ...}} "
                               "instead of a flat list. Repeated element names collect into a "
                               "list rather than overwriting."}),
            },
        }

    def extract(self, json_input, category, output_format,
                collect_all=False, with_keys=False):
        parsed = parse_input(json_input) if isinstance(json_input, str) else json_input
        if not isinstance(parsed, (dict, list)):
            return ("", "", False)

        positives, negatives = _partition_categories(category)

        # Prune exclusions from the source first, so dot-path negatives and
        # the "whole doc minus X" mode line up with the original structure.
        if negatives:
            _apply_negatives(parsed, negatives)

        if collect_all:
            def resolve(payload, cat):
                return _resolve_category_all(payload, cat, with_keys)
        else:
            resolve = _resolve_category

        if not positives:
            # No positive category → the whole (pruned) input unwrapped.
            out_obj = parsed
        elif len(positives) == 1:
            found, value, wrap_key = resolve(parsed, positives[0])
            if not found:
                return ("", "", False)
            out_obj = {wrap_key: value} if wrap_key is not None else value
        else:
            # Multi-category — resolve each and merge into one dict.
            combined: dict = {}
            any_found = False
            for cat in positives:
                found, value, wrap_key = resolve(parsed, cat)
                if not found or wrap_key is None:
                    continue
                any_found = True
                _merge_into(combined, wrap_key, value)
            if not any_found:
                return ("", "", False)
            out_obj = combined

        # Exclusions may have emptied the result → treat as a miss.
        if negatives and _is_empty(out_obj):
            return ("", "", False)

        if isinstance(out_obj, (dict, list)):
            raw = emit_strict_json(out_obj, indent=2)
        else:
            raw = emit_strict_json(out_obj, indent=None)

        if collect_all and positives and output_format == LOOSE_KEYS:
            # Plain text, no JSON syntax at all — the half-quoted loose_keys
            # form would only get in a text encoder's way here.
            scalars: list = []
            _flatten_scalars(out_obj, scalars, with_keys=with_keys)
            string_out = ", ".join(scalars)
        else:
            string_out = emit(out_obj, output_format)
        return (raw, string_out, True)
