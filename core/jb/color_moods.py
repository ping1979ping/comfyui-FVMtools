"""Colour moods — one dropdown instead of six sliders.

The harmony engine is good at *designed* palettes: it spreads hues evenly and
hands out things like boysenberry, pistachio-green and champagne-gold. On an
everyday outfit that reads as costume. Real clothes come from a much narrower,
mostly desaturated set, and a whole outfit usually shares two or three of them
plus at most one accent.

A mood therefore either
  * draws from a curated pool of colours people actually wear (``pool``), or
  * configures the harmony engine (``engine``) for the deliberately designed looks.

``auto`` keeps the previous behaviour exactly, so existing workflows are
unaffected.
"""

from __future__ import annotations

import random

# ── Curated pools ────────────────────────────────────────────────────────

NEUTRALS = [
    "white", "off-white", "cream", "light grey", "grey", "charcoal grey",
    "black", "navy", "beige", "sand", "taupe", "camel", "khaki", "brown",
]

MUTED = [
    "dusty rose", "sage green", "olive green", "muted teal", "burgundy",
    "mustard yellow", "rust orange", "dusty blue", "faded denim blue",
    "plum", "forest green", "terracotta",
]

# Without the word "denim" in them: the fabric is usually denim too, and
# "dark denim blue denim jeans" reads badly.
DENIM = ["faded blue", "indigo", "washed light blue", "dark blue", "mid blue"]

ACCENTS = [
    "red", "cobalt blue", "emerald green", "coral", "bright yellow",
    "fuchsia", "orange",
]

WARM_EARTH = [
    "cream", "camel", "rust orange", "terracotta", "olive green", "brown",
    "mustard yellow", "sand", "burgundy",
]

COOL_MUTED = [
    "light grey", "charcoal grey", "navy", "slate blue", "sage green",
    "muted teal", "off-white", "dusty blue",
]

# Only bases that take a shade prefix sensibly — "pale black" is nonsense,
# so black and white are deliberately absent here.
MONO_BASES = ["grey", "navy", "blue", "beige", "brown", "olive green", "green"]


# ── Mood table ───────────────────────────────────────────────────────────
#   pool         colours to draw from
#   accents      optional extra pool, ``accent_count`` entries are added
#   engine       harmony-engine kwargs (used instead of a pool)
#   description  shown in the node tooltip

COLOR_MOODS: dict[str, dict] = {
    "auto": {
        "description": "Keep the classic harmony engine and the six sliders. "
                       "Unchanged behaviour for existing workflows.",
    },
    "everyday_muted": {
        "pool": NEUTRALS + MUTED,
        "neutral_bias": 0.6,
        "description": "What people actually wear: mostly neutrals with a few "
                       "desaturated colours. Best default for casual sets.",
    },
    "neutral_basics": {
        "pool": NEUTRALS,
        "description": "White, grey, navy, beige, black only — plain basics, "
                       "nothing that draws attention.",
    },
    "one_accent": {
        "pool": NEUTRALS,
        # On the "primary" role, i.e. the top — the piece that is nearly always
        # present. Putting the accent last would land it on metallic/tertiary,
        # which most outfits never use, and the accent would be invisible.
        "role_pools": {"primary": ACCENTS},
        "description": "Neutral outfit plus exactly one saturated piece (the top).",
    },
    "denim_casual": {
        "pool": NEUTRALS,
        "role_pools": {"secondary": DENIM},
        "description": "Denim bottoms and jacket with neutral tops and shoes.",
    },
    "warm_earth": {
        "pool": WARM_EARTH,
        "description": "Cream, camel, rust, olive — warm and earthy.",
    },
    "cool_muted": {
        "pool": COOL_MUTED,
        "description": "Grey, navy, slate, sage — cool and restrained.",
    },
    "monochrome": {
        "mono": True,
        "description": "One base colour in several shades.",
    },
    "bold": {
        "engine": {"vibrancy": 0.85, "contrast": 0.75, "harmony_type": "complementary"},
        "description": "Saturated and high contrast — designed, not everyday.",
    },
    "pastel": {
        "engine": {"vibrancy": 0.3, "contrast": 0.3, "harmony_type": "analogous"},
        "description": "Soft, light, low contrast.",
    },
}

MOOD_NAMES = tuple(COLOR_MOODS)

SHADE_PREFIXES = ("light", "", "dark", "deep", "pale")

# Must match core.jb.palette.CANONICAL_ROLES — a mood addresses roles by
# position, because that is how the colour list is mapped onto them.
ROLE_ORDER = ("primary", "secondary", "accent", "neutral", "metallic", "tertiary")


def mood_help() -> str:
    """Tooltip text listing every mood and what it does."""
    lines = [
        "Picks the colour scheme with one setting instead of six sliders.",
        "",
    ]
    for name, spec in COLOR_MOODS.items():
        lines.append(f"{name}: {spec['description']}")
    lines.append("")
    lines.append(
        "The six classic sliders (num_colors, harmony_type, palette_style, "
        "vibrancy, contrast, warmth) stay available as optional inputs and are "
        "used when the mood is 'auto' — or to override single values."
    )
    return "\n".join(lines)


def _shaded(rng: random.Random, base: str) -> str:
    prefix = rng.choice(SHADE_PREFIXES)
    return f"{prefix} {base}".strip()


def mood_colors(mood: str, seed: int, count: int) -> list[str] | None:
    """Deterministic colour names for a pool/mono mood, or None for engine moods.

    Returning None means "let the harmony engine handle it".
    """
    spec = COLOR_MOODS.get(mood)
    if not spec or ("pool" not in spec and not spec.get("mono")):
        return None

    rng = random.Random((seed * 2_654_435_761) ^ 0x5EED)
    count = max(2, min(int(count), 8))

    if spec.get("mono"):
        base = rng.choice(MONO_BASES)
        shades = list(SHADE_PREFIXES)
        rng.shuffle(shades)
        out = []
        for index in range(count):
            prefix = shades[index % len(shades)]
            out.append(f"{prefix} {base}".strip())
        # Keep it readable: no duplicates.
        seen, unique = set(), []
        for name in out:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        while len(unique) < count:
            unique.append(_shaded(rng, base))
        return unique[:count]

    pool = list(spec["pool"])
    rng.shuffle(pool)
    chosen = pool[:count]

    # Bias towards neutrals so an outfit does not end up in five colours at once.
    bias = spec.get("neutral_bias")
    if bias:
        neutral_slots = max(1, int(round(count * bias)))
        neutral_pool = [c for c in NEUTRALS]
        rng.shuffle(neutral_pool)
        for index in range(min(neutral_slots, count)):
            chosen[index] = neutral_pool[index % len(neutral_pool)]

    # Deduplicate while keeping the requested length.
    seen, unique = set(), []
    for name in chosen:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    fill = [c for c in pool if c not in seen]
    while len(unique) < count and fill:
        unique.append(fill.pop(0))
    unique = unique[:count]

    # Role-targeted pools last, so neutral_bias cannot overwrite them again.
    for role, role_pool in (spec.get("role_pools") or {}).items():
        if role not in ROLE_ORDER:
            continue
        index = ROLE_ORDER.index(role)
        if index < len(unique):
            unique[index] = rng.choice(list(role_pool))
    return unique


def mood_engine_kwargs(mood: str) -> dict:
    """Harmony-engine overrides for an engine mood ({} when it has none)."""
    return dict(COLOR_MOODS.get(mood, {}).get("engine") or {})


__all__ = [
    "COLOR_MOODS",
    "MOOD_NAMES",
    "mood_colors",
    "mood_engine_kwargs",
    "mood_help",
]
