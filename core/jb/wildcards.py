"""JB wildcards — directory walker + resolver for ``__name__`` patterns.

Mirrors the layout of ``core/jb/catalog.py`` but stores plain ``.txt``
files (one option per line, ``#``-comments allowed) under a configurable
root.

The root is read from ``wildcards.ini`` at the FVMtools project root
(``wildcards.ini.local`` overrides if present — gitignored, personal):

    [wildcards]
    path = wildcards    ; relative to FVMtools, or an absolute path

If the INI is missing or unreadable the default ``<FVMtools>/wildcards/``
is used and auto-created.

Supported syntax (compatible subset of ``comfyui-adaptiveprompts``):

  Wildcards
    ``__name__``                 random line from ``<root>/<name>.txt``
    ``__cat/sub/name__``         nested category (subdir)
    ``__cat/*__``                pick a random ``.txt`` in ``cat/``
    ``__cat/pre*__``             pick a ``.txt`` whose name starts with ``pre``
    ``__name^var__``             pick a line, bind it to ``var``
    ``__name^v1^v2__``           same, bind to multiple vars (chained)
    ``__^var__``                 recall any value bound to ``var``
    ``__^pre*__``                recall from any var whose name starts with ``pre``

  Brackets
    ``{a|b|c}``                  pick one
    ``{%2%a|%1%b|c}``            weighted choices (per-choice ``%w%`` prefix)
    ``{N$$a|b|c}``               pick N distinct (deck mode)
    ``{N-M$$a|b|c}``             pick a random count between N and M
    ``{*$$a|b|c}``               include all choices (joined)
    ``{N??a|b|c}``               pick N with replacement (roulette mode)
    ``{N$$<sep>$$a|b|c}``        custom join separator
    ``{a|b}^var``                bind the picked text to a variable

  Per-line weights in ``.txt``
    ``%2.5%silk``                weight 2.5 for ``silk`` when this file is drawn

  Comments
    ``# trailing``               line comment in ``.txt`` files (skip line)
    ``##block##``                block comment in row values (resolved for
                                 side-effects, then stripped from output)

  Escaping
    ``\\__name__``               literal ``__name__`` — never resolved
    ``\\{...\\}``                literal braces — never resolved
    ``\\%`` / ``\\#``            literal ``%`` / ``#`` in line text

Recursion is bounded (depth + total passes) so a self-referencing
wildcard can't lock the resolver up.
"""

from __future__ import annotations

import configparser
import os
import random
import re
from typing import Any

DEFAULT_DIR_NAME = "wildcards"
INI_FILENAME = "wildcards.ini"
INI_LOCAL_FILENAME = "wildcards.ini.local"  # gitignored personal override
MAX_RECURSION = 16
MAX_PASSES = 24             # outer bracket+wildcard sweep iterations
MAX_BRACKET_PASSES = 64     # innermost-first {...} replacements per pass

# Wildcard pattern. ``name`` may include ``/`` and ``*`` (for globbing); the
# variable suffix is one or more ``^name`` segments (chained assignment).
_WC_PATTERN = re.compile(
    r"__([a-zA-Z0-9_\-/*]+)?((?:\^[a-zA-Z0-9_\-*]+)+)?__"
)
# Innermost-first bracket: matches a ``{...}`` whose interior has no
# nested braces. Iterating the substitution handles arbitrary nesting.
_BRACKET_RE = re.compile(r"\{([^{}]+)\}")
# ``##...##`` block comments (across newlines).
_COMMENT_RE = re.compile(r"##(.*?)##", flags=re.DOTALL)
# Backslash-escapes for wildcards and brackets.
_ESC_WC_RE = re.compile(r"\\(__[a-zA-Z0-9_\-/^*]+__)")
_ESC_BR_RE = re.compile(r"\\\{|\\\}")
# ``%2.5%text`` per-line / per-choice weight prefix. Negative lookbehind
# means ``\%`` is never treated as a weight marker.
_WEIGHT_PREFIX_RE = re.compile(r"^(?<!\\)%([0-9]*\.?[0-9]+)%(.*)$", flags=re.DOTALL)
# Bracket header: count + mode token. ``count`` is ``*``, ``N``, or ``N-M``;
# ``mode`` is ``$$`` (deck — without replacement) or ``??`` (roulette).
_BR_HEADER_RE = re.compile(r"^(\*|\d+(?:-\d+)?)(\$\$|\?\?)")

# Cache of parsed wildcard files: name -> (lines, weights). Cleared by
# ``invalidate_cache()`` whenever the wildcards directory is mutated.
_FILE_CACHE: dict[str, tuple[list[str], list[float]]] = {}


# ─── path resolution ────────────────────────────────────────────────────


def _project_root() -> str:
    """Absolute path to the FVMtools package root."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", ".."))


def wildcards_root() -> str:
    """Resolve and ensure the wildcards directory exists.

    Lookup order for the path setting:
      1. ``wildcards.ini.local`` (gitignored — user's personal override).
      2. ``wildcards.ini`` (tracked default).
      3. ``<FVMtools>/wildcards/`` fallback if neither exists.
    """
    project = _project_root()
    configured: str | None = None
    for fname in (INI_LOCAL_FILENAME, INI_FILENAME):
        ini_path = os.path.join(project, fname)
        if not os.path.isfile(ini_path):
            continue
        cp = configparser.ConfigParser()
        try:
            cp.read(ini_path, encoding="utf-8")
            configured = cp.get("wildcards", "path", fallback=None)
        except (configparser.Error, OSError):
            configured = None
        if configured:
            break
    if configured:
        path = configured.strip()
        if not os.path.isabs(path):
            path = os.path.join(project, path)
    else:
        path = os.path.join(project, DEFAULT_DIR_NAME)
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
    return path


# ─── safe-name guards (file CRUD) ───────────────────────────────────────


def _is_safe_name(name: str) -> bool:
    """Allow ``cat/sub/foo`` style nesting; reject path traversal."""
    if not isinstance(name, str) or not name:
        return False
    if ".." in name.split("/"):
        return False
    if "\\" in name or name.startswith(("/", "_", ".")):
        return False
    for part in name.split("/"):
        if not part or part.startswith(("_", ".")):
            return False
        if not re.fullmatch(r"[a-zA-Z0-9_\-]+", part):
            return False
    return True


def _entry_path(name: str) -> str | None:
    if not _is_safe_name(name):
        return None
    return os.path.join(wildcards_root(), f"{name}.txt")


# ─── CRUD ───────────────────────────────────────────────────────────────


def list_all() -> list[str]:
    """Return every wildcard's slash-path (e.g. ``outfits/colors``)."""
    root = wildcards_root()
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(("_", "."))]
        for fname in filenames:
            if not fname.endswith(".txt") or fname.startswith(("_", ".")):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root)
            slash = rel.replace(os.sep, "/")
            if slash.endswith(".txt"):
                slash = slash[:-4]
            out.append(slash)
    out.sort()
    return out


def read_wildcard(name: str) -> str | None:
    """Return the raw text of a wildcard file, or ``None`` if missing."""
    path = _entry_path(name)
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_wildcard(name: str, text: str) -> bool:
    """Create or replace a wildcard file's text contents."""
    path = _entry_path(name)
    if path is None or not isinstance(text, str):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    invalidate_cache()
    return True


def delete_wildcard(name: str) -> bool:
    """Remove a wildcard file. Returns True if it existed."""
    path = _entry_path(name)
    if path is None or not os.path.isfile(path):
        return False
    os.remove(path)
    invalidate_cache()
    return True


def invalidate_cache() -> None:
    """Drop the in-memory file cache so the next resolve re-reads disk."""
    _FILE_CACHE.clear()


# ─── file loader (with per-line weights) ────────────────────────────────


def _load_lines_weighted(name: str) -> tuple[list[str], list[float]] | None:
    """Read a wildcard's non-comment lines, extracting any ``%w%`` weights.

    Cached. Returns ``None`` if the file is missing.
    """
    if name in _FILE_CACHE:
        return _FILE_CACHE[name]
    raw = read_wildcard(name)
    if raw is None:
        return None
    lines: list[str] = []
    weights: list[float] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        # Strip a trailing inline comment (unescaped ``#``).
        s = re.sub(r"(?<!\\)#.*$", "", s).rstrip()
        if not s:
            continue
        # Per-line weight prefix.
        m = _WEIGHT_PREFIX_RE.match(s)
        if m:
            try:
                w = float(m.group(1))
            except ValueError:
                w = 1.0
            text = m.group(2).strip()
        else:
            w = 1.0
            text = s
        if not text:
            continue
        # Restore escaped meta-chars after parsing.
        text = text.replace("\\%", "%").replace("\\#", "#")
        lines.append(text)
        weights.append(max(0.0, w))
    _FILE_CACHE[name] = (lines, weights)
    return lines, weights


# ─── context normalization ──────────────────────────────────────────────


def _normalize_context(ctx: Any) -> dict[str, dict[str, str]]:
    """Coerce an incoming context into the dict-of-dicts shape.

    Accepts the shapes adaptiveprompts uses: dict-of-dicts (preferred),
    dict-of-lists, dict-of-singles. Anything else becomes ``{}``.
    """
    if not isinstance(ctx, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for var, bucket in ctx.items():
        key = str(var)
        if isinstance(bucket, dict):
            out[key] = {str(k): str(v) for k, v in bucket.items()}
        elif isinstance(bucket, (list, tuple, set)):
            out[key] = {f"slot_{i}": str(v) for i, v in enumerate(bucket)}
        else:
            out[key] = {"slot_0": str(bucket)}
    return out


# ─── escape protection ──────────────────────────────────────────────────


def _protect_escaped(text: str) -> tuple[str, dict[str, str]]:
    """Replace ``\\__...__`` and ``\\{`` / ``\\}`` with sentinels.

    Sentinels survive the resolution pipeline untouched and are restored
    to their literal form by ``_restore_escaped`` at the end.
    """
    placeholders: dict[str, str] = {}
    counter = [0]

    def sub_wc(m: re.Match) -> str:
        token = m.group(1)
        key = f"\x00ESC{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = token
        return key

    text = _ESC_WC_RE.sub(sub_wc, text)

    def sub_br(m: re.Match) -> str:
        literal = m.group(0)[1]  # the `{` or `}`
        key = f"\x00ESC{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = literal
        return key

    text = _ESC_BR_RE.sub(sub_br, text)
    return text, placeholders


def _restore_escaped(text: str, placeholders: dict[str, str]) -> str:
    for k, v in placeholders.items():
        text = text.replace(k, v)
    return text


# ─── bracket helpers ────────────────────────────────────────────────────


def _split_top_level(text: str, delim: str) -> list[str]:
    """Split ``text`` by ``delim`` only at brace depth 0."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
            cur.append(c); i += 1; continue
        if c == "}":
            depth -= 1
            cur.append(c); i += 1; continue
        if depth == 0 and text.startswith(delim, i):
            out.append("".join(cur))
            cur = []
            i += len(delim)
            continue
        cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


def _find_top_level(text: str, delim: str) -> int:
    """First index of ``delim`` at brace depth 0, or ``-1``."""
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0 and text.startswith(delim, i):
            return i
        i += 1
    return -1


def _weighted_sample_no_replace(
    choices: list[str], weights: list[float], n: int, rng: random.Random,
) -> list[str]:
    """Pick ``n`` distinct items, weighted (no replacement)."""
    pool = list(zip(choices, weights))
    out: list[str] = []
    for _ in range(n):
        if not pool:
            break
        total = sum(w for _, w in pool)
        if total <= 0:
            idx = rng.randrange(len(pool))
        else:
            r = rng.uniform(0, total)
            acc = 0.0
            idx = 0
            for i, (_, w) in enumerate(pool):
                acc += w
                if r <= acc:
                    idx = i
                    break
        out.append(pool.pop(idx)[0])
    return out


def _parse_bracket(content: str, rng: random.Random) -> str:
    """Parse a single ``{...}`` body and return the joined picked text."""
    sep = ", "
    count_spec: Any = None  # None | int | (lo, hi) | "*"
    mode = "single"

    head = _BR_HEADER_RE.match(content)
    if head:
        count_str = head.group(1)
        mode_tok = head.group(2)
        content = content[head.end():]
        mode = "deck" if mode_tok == "$$" else "roulette"
        if count_str == "*":
            count_spec = "*"
        elif "-" in count_str:
            lo, hi = count_str.split("-", 1)
            try:
                count_spec = (int(lo), int(hi))
            except ValueError:
                count_spec = 1
        else:
            try:
                count_spec = int(count_str)
            except ValueError:
                count_spec = 1

        # Optional second header: ``<sep>$$rest``. Detected as a $$ that
        # appears before the first | at the top level.
        sep_idx = _find_top_level(content, "$$")
        pipe_idx = _find_top_level(content, "|")
        if sep_idx >= 0 and (pipe_idx < 0 or sep_idx < pipe_idx):
            sep = content[:sep_idx]
            content = content[sep_idx + 2:]

    raw_choices = _split_top_level(content, "|")
    choices: list[str] = []
    weights: list[float] = []
    for raw in raw_choices:
        m = _WEIGHT_PREFIX_RE.match(raw)
        if m:
            try:
                weights.append(max(0.0, float(m.group(1))))
            except ValueError:
                weights.append(1.0)
            choices.append(m.group(2))
        else:
            weights.append(1.0)
            choices.append(raw)

    if not choices:
        return ""

    if count_spec is None:
        picked = [rng.choices(choices, weights=weights, k=1)[0]]
    elif count_spec == "*":
        picked = list(choices)
    else:
        if isinstance(count_spec, tuple):
            lo, hi = count_spec
            n = rng.randint(min(lo, hi), max(lo, hi))
        else:
            n = int(count_spec)
        n = max(0, n)
        if mode == "deck":
            # Without replacement — can't pick more than the pool size.
            picked = _weighted_sample_no_replace(
                choices, weights, min(n, len(choices)), rng,
            )
        else:  # roulette — with replacement, n may exceed pool size.
            picked = list(rng.choices(choices, weights=weights, k=n))

    return sep.join(picked)


def _resolve_brackets(text: str, rng: random.Random,
                      ctx: dict[str, dict[str, str]]) -> str:
    """Replace innermost ``{...}`` repeatedly until none remain or limit hit.

    Honours an optional ``^var`` (or chained ``^v1^v2``) suffix after the
    closing ``}``: the bracket's resolved text is bound to those vars.
    """
    var_suffix_re = re.compile(r"\^[a-zA-Z0-9_\-]+(?:\^[a-zA-Z0-9_\-]+)*")
    for _ in range(MAX_BRACKET_PASSES):
        m = _BRACKET_RE.search(text)
        if not m:
            break
        replacement = _parse_bracket(m.group(1), rng)
        end = m.end()
        var_m = var_suffix_re.match(text, end)
        if var_m:
            for vn in var_m.group(0).lstrip("^").split("^"):
                bucket = ctx.setdefault(vn, {})
                bucket[f"slot_{len(bucket)}"] = replacement
            end = var_m.end()
        text = text[:m.start()] + replacement + text[end:]
    return text


# ─── wildcard helpers ───────────────────────────────────────────────────


def _glob_pick(name: str, rng: random.Random) -> str | None:
    """Resolve a globbed wildcard name like ``cat/*`` or ``cat/pre*``.

    Returns the chosen concrete name (e.g. ``cat/red``) or ``None`` if
    nothing matches.
    """
    if "*" not in name:
        return name
    root = wildcards_root()
    if "/" in name:
        dir_part, file_pat = name.rsplit("/", 1)
        scan_dir = os.path.join(root, *dir_part.split("/"))
    else:
        dir_part = ""
        file_pat = name
        scan_dir = root
    if not os.path.isdir(scan_dir):
        return None
    prefix = file_pat.rstrip("*")
    candidates: list[str] = []
    for fn in os.listdir(scan_dir):
        if not fn.endswith(".txt") or fn.startswith(("_", ".")):
            continue
        stem = fn[:-4]
        if prefix and not stem.startswith(prefix):
            continue
        candidates.append(stem)
    if not candidates:
        return None
    chosen = rng.choice(sorted(candidates))
    return f"{dir_part}/{chosen}" if dir_part else chosen


def _recall_var(varspec: str, ctx: dict[str, dict[str, str]],
                rng: random.Random) -> str:
    """Recall a value bound to ``varspec`` (or any var matching a ``*``-suffix)."""
    candidates: list[str] = []
    if varspec.endswith("*"):
        prefix = varspec[:-1]
        for name, bucket in ctx.items():
            if name.startswith(prefix):
                candidates.extend(bucket.values())
    else:
        bucket = ctx.get(varspec) or {}
        candidates.extend(bucket.values())
    if not candidates:
        return ""
    return rng.choice(candidates)


def _resolve_wildcards(text: str, rng: random.Random,
                       ctx: dict[str, dict[str, str]],
                       _depth: int = 0) -> str:
    """Replace ``__name__`` tokens. Recursion-bounded by ``MAX_RECURSION``."""
    if _depth >= MAX_RECURSION:
        return text

    def replace(m: re.Match) -> str:
        name = (m.group(1) or "").strip("/")
        var_chain = m.group(2) or ""

        # Pure recall: ``__^var__`` or ``__^v1^v2__``.
        if not name and var_chain:
            specs = var_chain.lstrip("^").split("^")
            return _recall_var(rng.choice(specs), ctx, rng)

        if not name:
            return m.group(0)

        if "*" in name:
            picked = _glob_pick(name, rng)
            if picked is None:
                return m.group(0)
            name = picked

        loaded = _load_lines_weighted(name)
        if loaded is None:
            return m.group(0)
        lines, weights = loaded
        if not lines:
            return ""
        chosen = rng.choices(lines, weights=weights, k=1)[0]

        # Recurse into the picked line so it can itself contain wildcards
        # and brackets.
        chosen = _resolve_brackets(chosen, rng, ctx)
        chosen = _resolve_wildcards(chosen, rng, ctx, _depth + 1)

        if var_chain:
            for vn in var_chain.lstrip("^").split("^"):
                bucket = ctx.setdefault(vn, {})
                bucket[f"slot_{len(bucket)}"] = chosen

        return chosen

    return _WC_PATTERN.sub(replace, text)


# ─── public resolver ────────────────────────────────────────────────────


def resolve_text(
    text: str,
    seed: int,
    context: Any | None = None,
    *,
    salt: str = "",
) -> tuple[str, dict[str, dict[str, str]]]:
    """Expand every ``__...__`` and ``{...}`` token in ``text``.

    ``salt`` is mixed into the seed so distinct call-sites (e.g. each
    leaf in the row tree) produce independent draws even with the same
    base seed.
    """
    if not isinstance(text, str):
        return ("" if text is None else str(text)), _normalize_context(context)
    if "__" not in text and "{" not in text and "##" not in text:
        return text, _normalize_context(context)

    rng = random.Random(f"{seed}|{salt}")
    ctx = _normalize_context(context)

    text, placeholders = _protect_escaped(text)

    # Outer pipeline: alternate brackets and wildcards until stable.
    for _ in range(MAX_PASSES):
        before = text
        text = _resolve_brackets(text, rng, ctx)
        text = _resolve_wildcards(text, rng, ctx)
        if text == before:
            break

    # Strip ``##...##`` blocks AFTER resolution so any side-effects
    # (variable bindings inside the comment) have already taken effect.
    text = _COMMENT_RE.sub("", text)

    text = _restore_escaped(text, placeholders)
    return text, ctx
