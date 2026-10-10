"""Multi-select for the JB blocks: draw one of several checked choices per seed.

The JB Outfit / Location Block keep their single-choice dropdowns
(``outfit_set``, ``location_set``, ``color_mood``). A hidden STRING widget
next to each (``*_multi``, filled by the checkbox modal in
``web/js/fvm_jb_multi_select.js``) holds the checked choices, one per line.
When it names at least one valid choice, the block draws one of them per
seed and the dropdown is ignored.

The draw hashes ``salt:seed`` instead of using the engines' RNG, so the
outfit or location generated for the drawn choice is identical to the one
you get by selecting that choice alone with the same seed.
"""

from __future__ import annotations

import hashlib
from typing import Iterable


def parse_choices(text) -> list[str]:
    """One choice per line; blank lines, ``#`` comments and repeats dropped."""
    if not text or not isinstance(text, str):
        return []
    seen: dict[str, None] = {}
    for line in text.splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            seen.setdefault(value, None)
    return list(seen)


def pick_choice(multi, fallback: str, valid: Iterable[str], seed: int,
                salt: str) -> tuple[str, list[str], list[str]]:
    """Draw one choice from ``multi`` for ``seed``.

    Returns ``(choice, pool, skipped)``: ``pool`` are the valid checked
    choices in dropdown order, ``skipped`` the checked names that no longer
    exist (renamed or deleted sets). An empty pool falls back to the
    dropdown value.
    """
    valid = list(valid)
    checked = set(parse_choices(multi))
    pool = [v for v in valid if v in checked]
    skipped = sorted(checked - set(pool))
    if not pool:
        return fallback, pool, skipped
    digest = hashlib.sha256(f"{salt}:{int(seed)}".encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(pool)
    return pool[index], pool, skipped


def log_pick(node: str, field: str, choice: str, pool: list[str],
             skipped: list[str]) -> None:
    """Console line so a batch run shows which choice each seed drew."""
    if skipped:
        print(f"[FVM {node}] {field}: skipped unknown {', '.join(skipped)}")
    if len(pool) > 1:
        print(f"[FVM {node}] {field}: drew {choice} ({len(pool)} checked)")
