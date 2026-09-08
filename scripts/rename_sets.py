"""Rename location / outfit set directories per ``scripts/rename_sets_map.txt``.

Why: the JB ``sentences`` output format reads set names back mechanically
(``_`` -> space), so the directory names *are* the wording of the intro
sentence. This script applies a reviewed map of renames and rewrites every
reference in code, tests, docs, examples and list files.

Map syntax (one rule per line, ``#`` comments)::

    old_path => new_path        POSIX, relative to location_lists/ or outfit_lists/

  * root: first segment ``indoor``/``outdoor`` -> location_lists, else outfit_lists
  * 2-segment rule under a scope / gender = category rule (every set below moves)
  * 3-segment rule = set rule, wins over the category rule for that set
  * 1-segment rule = legacy flat outfit dir (may be moved to ``_archive/<name>``)

What it does:
  1. expands category rules over the directories on disk into per-set moves,
     refuses to run when a target already exists or two rules collide;
  2. ``git mv`` per set (falls back to a plain move for untracked dirs) and
     removes emptied category directories;
  3. rewrites references — set paths first (longest first), then category
     prefixes — in *.py *.md *.txt *.json *.js *.html across the repo.

``--dry-run`` prints the moves and the per-file reference counts only.
Saved ComfyUI workflows are NOT touched: a stale value in a COMBO widget
fails ComfyUI's own validation ("value not in list"), so it has to be
re-picked once in the UI anyway.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MAP_FILE = os.path.join(REPO_ROOT, "scripts", "rename_sets_map.txt")
LOCATION_ROOT = os.path.join(REPO_ROOT, "location_lists")
OUTFIT_ROOT = os.path.join(REPO_ROOT, "outfit_lists")

LOCATION_SCOPES = ("indoor", "outdoor")
OUTFIT_GENDERS = ("female", "male", "unisex")

TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".js", ".html"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache", "venv"}
SKIP_FILES = {os.path.normpath(MAP_FILE),
              os.path.normpath(os.path.abspath(__file__))}


# ─── Map parsing ───────────────────────────────────────────────────────


def parse_map(path: str) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" not in line:
                sys.exit(f"{path}:{lineno}: expected 'old => new', got: {line}")
            old, new = (s.strip().strip("/") for s in line.split("=>", 1))
            if not old or not new:
                sys.exit(f"{path}:{lineno}: empty side in rule: {line}")
            if old == new:
                continue
            rules.append((old, new))
    return rules


def root_for(path: str) -> str:
    return LOCATION_ROOT if path.split("/")[0] in LOCATION_SCOPES else OUTFIT_ROOT


def is_category_rule(old: str) -> bool:
    parts = old.split("/")
    return len(parts) == 2 and parts[0] in LOCATION_SCOPES + OUTFIT_GENDERS


def list_sets_under(root: str, category: str) -> list[str]:
    base = os.path.join(root, *category.split("/"))
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d)) and not d.startswith((".", "_")))


def expand(rules: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (set_moves, category_rules) with set rules overriding category rules."""
    category_rules = [(o, n) for o, n in rules if is_category_rule(o)]
    set_rules = [(o, n) for o, n in rules if not is_category_rule(o)]

    moves: dict[str, str] = {}
    for old_cat, new_cat in category_rules:
        root = root_for(old_cat)
        for leaf in list_sets_under(root, old_cat):
            moves[f"{old_cat}/{leaf}"] = f"{new_cat}/{leaf}"
    for old, new in set_rules:
        moves[old] = new

    # Sanity: sources exist, targets don't, no duplicate targets.
    problems = []
    targets: dict[str, str] = {}
    for old, new in moves.items():
        src = os.path.join(root_for(old), *old.split("/"))
        dst = os.path.join(root_for(old), *new.split("/"))
        if not os.path.isdir(src):
            problems.append(f"missing source: {old}")
        if os.path.exists(dst):
            problems.append(f"target exists: {new}")
        if new in targets:
            problems.append(f"two rules point at {new}: {targets[new]} and {old}")
        targets[new] = old
    if problems:
        sys.exit("Map problems — nothing done:\n  " + "\n  ".join(problems))
    return sorted(moves.items()), category_rules


# ─── Moving ───────────────────────────────────────────────────────────


def move_dir(src_abs: str, dst_abs: str) -> str:
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    result = subprocess.run(["git", "mv", src_abs, dst_abs], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    if result.returncode == 0:
        return "git mv"
    # Untracked directory (nothing for git to move) — move it on disk.
    shutil.move(src_abs, dst_abs)
    return "moved (untracked)"


def prune_empty_dirs(root: str) -> list[str]:
    removed = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        if not dirnames and not filenames:
            os.rmdir(dirpath)
            removed.append(os.path.relpath(dirpath, root).replace(os.sep, "/"))
    return removed


# ─── Reference rewriting ───────────────────────────────────────────────


def _pattern(old: str) -> re.Pattern:
    # Not preceded or followed by a word char — so "a/b/c" matches in
    # "location_lists/a/b/c/props.txt" and "'a/b/c'" but not in "a/b/cd" or
    # "xa/b/c". A preceding "/" is fine: that is the lists root.
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(old) + r"(?![A-Za-z0-9_])")


def build_replacements(moves, category_rules) -> list[tuple[re.Pattern, str]]:
    reps = []
    for old, new in sorted(moves, key=lambda m: -len(m[0])):
        reps.append((_pattern(old), new))
    for old, new in sorted(category_rules, key=lambda m: -len(m[0])):
        reps.append((_pattern(old), new))
    return reps


def iter_text_files(repo: str):
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in TEXT_EXTENSIONS:
                continue
            full = os.path.normpath(os.path.join(dirpath, name))
            if full in SKIP_FILES:
                continue
            yield full


def rewrite_references(reps, dry_run: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in iter_text_files(REPO_ROOT):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        new_text = text
        n_total = 0
        for pattern, new in reps:
            new_text, n = pattern.subn(new, new_text)
            n_total += n
        if n_total:
            counts[os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")] = n_total
            if not dry_run:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(new_text)
    return counts


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", default=MAP_FILE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rewrite-only", action="store_true",
                    help="directories are already moved — only rewrite references "
                         "(set rules + category prefixes from the map)")
    args = ap.parse_args()

    rules = parse_map(args.map)
    if args.rewrite_only:
        category_rules = [(o, n) for o, n in rules if is_category_rule(o)]
        moves = [(o, n) for o, n in rules if not is_category_rule(o)]
    else:
        moves, category_rules = expand(rules)
        print(f"{len(rules)} rules -> {len(moves)} set moves "
              f"({len(category_rules)} category rules)")
        for old, new in moves:
            print(f"  {old}  ->  {new}")

    reps = build_replacements(moves, category_rules)
    counts = rewrite_references(reps, dry_run=True)
    print(f"\nreferences in {len(counts)} files:")
    for path, n in sorted(counts.items()):
        print(f"  {n:4d}  {path}")

    if args.dry_run:
        print("\n--dry-run: nothing changed.")
        return 0

    if args.rewrite_only:
        counts = rewrite_references(reps, dry_run=False)
        print(f"\nrewrote references in {len(counts)} files.")
        return 0

    print("\nmoving …")
    for old, new in moves:
        root = root_for(old)
        how = move_dir(os.path.join(root, *old.split("/")), os.path.join(root, *new.split("/")))
        print(f"  {how:18s} {old} -> {new}")
    for root in (LOCATION_ROOT, OUTFIT_ROOT):
        for d in prune_empty_dirs(root):
            print(f"  removed empty     {os.path.basename(root)}/{d}")

    print("\nrewriting references …")
    counts = rewrite_references(reps, dry_run=False)
    for path, n in sorted(counts.items()):
        print(f"  {n:4d}  {path}")
    print("\ndone. Next: run the unit tests, then git add the rewritten files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
