"""Dataset prompt lists — one prompt per line, wildcards expanded per image.

Built for character-LoRA datasets: a short list of hand-written shot
descriptions (angle, gaze, expression) is multiplied by ``variations``;
every copy resolves its ``__wildcards__`` and ``{a|b}`` brackets with its
own draw, so clothing, background and light change while the shot stays.

Line syntax (see ``documentation/WILDCARDS_GUIDE.md``)::

    # comment line — ignored
    [close] Straight-on frontal portrait ..., wearing __dataset/upper__
    [half, profile] Waist-up profile ...
    Untagged lines work too; they only pass the ``all`` filter.

The leading ``[tag, tag]`` block is optional. ``close`` / ``half`` /
``full`` are the shot sizes the node filters on; any other tag is kept
for your own bookkeeping.

Shipped data lives in ``<FVMtools>/dataset_presets/``:
  ``presets/*.txt``           prompt lists, loadable in the node editor
  ``wildcards/dataset/*.txt`` default wildcard files, copied into the
                              wildcards root on first use (never overwritten)
"""

from __future__ import annotations

import os
import random
import re
import shutil
from dataclasses import dataclass

try:
    from .jb.wildcards import resolve_text, wildcards_root, invalidate_cache
except ImportError:  # pragma: no cover - direct import outside the package
    from core.jb.wildcards import resolve_text, wildcards_root, invalidate_cache

SHOT_SIZES = ("close", "half", "full")
SHOT_FILTERS = ("all",) + SHOT_SIZES
ORDERS = ("rounds", "per_prompt", "shuffle")

_TAG_RE = re.compile(r"^\[([a-zA-Z0-9_ ,\-]+)\]\s*")
# Separates prefix / body / suffix during one shared resolve pass, so
# variables bound in the prefix can be recalled in the body. Not a token
# the resolver knows, so it passes through untouched.
_SEP = "\x01"
_PRESET_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _package_root() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )


def presets_dir() -> str:
    return os.path.join(_package_root(), "dataset_presets", "presets")


def shipped_wildcards_dir() -> str:
    return os.path.join(_package_root(), "dataset_presets", "wildcards")


# ─── parsing ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PromptLine:
    index: int  # position among all prompt lines of the text (0-based)
    text: str
    tags: tuple[str, ...]

    @property
    def shot(self) -> str:
        for t in self.tags:
            if t in SHOT_SIZES:
                return t
        return ""


def parse_lines(text: str) -> list[PromptLine]:
    """Split ``text`` into prompt lines; blank and ``#`` lines are skipped."""
    out: list[PromptLine] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        tags: tuple[str, ...] = ()
        m = _TAG_RE.match(s)
        if m:
            tags = tuple(t.strip().lower() for t in m.group(1).split(",") if t.strip())
            s = s[m.end() :].strip()
        if s:
            out.append(PromptLine(len(out), s, tags))
    return out


def select_lines(
    lines: list[PromptLine],
    shot_filter: str = "all",
    start_index: int = 0,
    max_rows: int = 1000,
) -> list[PromptLine]:
    """Filter by shot size first, then take ``max_rows`` from ``start_index``."""
    if shot_filter and shot_filter != "all":
        lines = [ln for ln in lines if shot_filter in ln.tags]
    start = max(0, int(start_index))
    return lines[start : start + max(0, int(max_rows))]


# ─── resolving ──────────────────────────────────────────────────────────


def _tidy(text: str) -> str:
    """Collapse whitespace and punctuation left behind by empty wildcards."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r",(\s*,)+", ",", text)
    text = re.sub(r"([,.])\1+", r"\1", text)
    # Wildcard lines start with a colour, so "a" may land before a vowel.
    text = re.sub(r"\b([Aa]) (?=[aeiouAEIOU])", r"\1n ", text)
    return text.strip(" ,")


def _join(*parts: str) -> str:
    """Join prefix / body / suffix as comma-separated segments."""
    return ", ".join(p.strip(" ,") for p in parts if p.strip(" ,"))


def expand(
    text: str,
    *,
    shot_filter: str = "all",
    start_index: int = 0,
    max_rows: int = 1000,
    variations: int = 1,
    order: str = "rounds",
    seed: int = 0,
    prefix: str = "",
    suffix: str = "",
    context=None,
) -> list[dict]:
    """Resolve the selected lines ``variations`` times each.

    Every (line, variation) pair draws with its own salt derived from the
    line's index in the *full* text, so a given line and variation always
    produce the same result for a seed — regardless of filter or range.
    """
    picked = select_lines(parse_lines(text), shot_filter, start_index, max_rows)
    variations = max(1, int(variations))
    if order == "per_prompt":
        pairs = [(ln, v) for ln in picked for v in range(variations)]
    else:
        pairs = [(ln, v) for v in range(variations) for ln in picked]

    items: list[dict] = []
    for ln, v in pairs:
        joined = _SEP.join((prefix or "", ln.text, suffix or ""))
        resolved, _ = resolve_text(joined, seed, context, salt=f"line{ln.index}|var{v}")
        pre, body, suf = (resolved.split(_SEP) + ["", "", ""])[:3]
        caption = _tidy(body)
        items.append(
            {
                "prompt": _tidy(_join(pre.strip(), caption, suf.strip())),
                "caption": caption,
                "shot": ln.shot,
                "index": ln.index,
                "variation": v,
            }
        )

    if order == "shuffle":
        random.Random(f"{seed}|shuffle").shuffle(items)
    return items


def listing(items: list[dict]) -> str:
    """Human-readable overview: counts per shot size, then every prompt."""
    counts = {s: 0 for s in SHOT_SIZES + ("",)}
    for it in items:
        counts[it["shot"]] = counts.get(it["shot"], 0) + 1
    head = (
        f"{len(items)} prompts — close {counts['close']}, half {counts['half']}, "
        f"full {counts['full']}, untagged {counts['']}"
    )
    rows = [
        f"#{it['index']}.{it['variation']} [{it['shot'] or '-'}] {it['prompt']}"
        for it in items
    ]
    return "\n".join([head, ""] + rows)


# ─── presets ────────────────────────────────────────────────────────────


def list_presets() -> list[str]:
    d = presets_dir()
    if not os.path.isdir(d):
        return []
    return sorted(
        f[:-4]
        for f in os.listdir(d)
        if f.endswith(".txt") and not f.startswith(("_", "."))
    )


def read_preset(name: str) -> str | None:
    if not _PRESET_NAME_RE.match(name or ""):
        return None
    path = os.path.join(presets_dir(), f"{name}.txt")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_preset(name: str, text: str) -> bool:
    if not _PRESET_NAME_RE.match(name or "") or not isinstance(text, str):
        return False
    os.makedirs(presets_dir(), exist_ok=True)
    with open(
        os.path.join(presets_dir(), f"{name}.txt"), "w", encoding="utf-8", newline="\n"
    ) as f:
        f.write(text)
    return True


def install_default_wildcards(target_root: str | None = None) -> list[str]:
    """Copy shipped wildcard files into the wildcards root if missing.

    Existing files are never touched, so edits made in the wildcard editor
    survive updates. Returns the slash-names that were copied.
    """
    src_root = shipped_wildcards_dir()
    dst_root = target_root or wildcards_root()
    copied: list[str] = []
    if not os.path.isdir(src_root):
        return copied
    for dirpath, _, filenames in os.walk(src_root):
        for fname in filenames:
            if not fname.endswith(".txt"):
                continue
            src = os.path.join(dirpath, fname)
            rel = os.path.relpath(src, src_root)
            dst = os.path.join(dst_root, rel)
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append(rel.replace(os.sep, "/")[:-4])
    if copied:
        invalidate_cache()
    return copied
