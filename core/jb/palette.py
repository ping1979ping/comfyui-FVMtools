"""Shared palette + token-substitution helpers for the JB combo nodes.

Wraps ``core.palette_engine.generate_palette()`` and produces:

  - ``garment_colors``: role → name dict (always 6 canonical roles —
    primary / secondary / accent / neutral / metallic / tertiary).
  - ``atmosphere_colors``: ambient_light / shadow_tone phrases derived
    deterministically from seed + warmth.
  - ``subs``: token → value map ready for fragment substitution.

Same canonical-role-backfill logic the SMP ColorGenerator uses, lifted
here so the JB blocks don't depend on SMP modules.
"""

from __future__ import annotations

import random

try:
    from ..palette_engine import generate_palette
    from ..smp.defaults import ATMOSPHERE_TOKEN_MAP, GARMENT_TOKEN_MAP
except ImportError:  # pragma: no cover
    from core.palette_engine import generate_palette
    from core.smp.defaults import ATMOSPHERE_TOKEN_MAP, GARMENT_TOKEN_MAP


CANONICAL_ROLES = ("primary", "secondary", "accent",
                   "neutral", "metallic", "tertiary")

_AMBIENT_WARM = [
    "warm amber afternoon",
    "soft golden hour glow",
    "honey-toned sunlight",
    "rich late-afternoon warmth",
]
_AMBIENT_NEUTRAL = [
    "balanced natural daylight",
    "soft diffuse midday light",
    "even overcast daylight",
]
_AMBIENT_COOL = [
    "cool overcast morning",
    "blue-hour soft daylight",
    "muted cool diffused light",
    "crisp morning daylight",
]
_SHADOW_WARM = [
    "deep cool blue shadows",
    "muted indigo shadows",
    "soft slate shadows",
]
_SHADOW_NEUTRAL = [
    "soft neutral grey shadows",
    "even diffused soft shadows",
]
_SHADOW_COOL = [
    "warm sienna shadows",
    "deep umber shadows",
    "subtle warm taupe shadows",
]


def _atmosphere_pair(rng: random.Random, warmth: float) -> tuple[str, str]:
    if warmth >= 0.66:
        return rng.choice(_AMBIENT_WARM), rng.choice(_SHADOW_WARM)
    if warmth <= 0.33:
        return rng.choice(_AMBIENT_COOL), rng.choice(_SHADOW_COOL)
    return rng.choice(_AMBIENT_NEUTRAL), rng.choice(_SHADOW_NEUTRAL)


def build_palette(*, seed: int, num_colors: int = 5, harmony_type: str = "auto",
                  style_preset: str = "general", vibrancy: float = 0.5,
                  contrast: float = 0.5, warmth: float = 0.5) -> dict:
    """Return a dict with garment_colors, atmosphere_colors, raw_tokens, subs.

    ``subs`` is the merged token map ready for fragment substitution
    (covers both garment and atmosphere tokens in a single pass).
    """
    result = generate_palette(
        seed=seed, num_colors=num_colors, harmony_type=harmony_type,
        style_preset=style_preset, vibrancy=vibrancy, contrast=contrast,
        warmth=warmth,
    )

    # Canonical-role backfill: ensure every role in CANONICAL_ROLES has a
    # color even if the harmony emitted fewer than 6.
    garment_colors: dict[str, str] = {}
    used_names: set[str] = set()
    for c in result["colors"]:
        role = c.get("role")
        if role and role not in garment_colors:
            garment_colors[role] = c["name"]
            used_names.add(c["name"])

    leftover = [c["name"] for c in result["colors"] if c["name"] not in used_names]
    leftover_iter = iter(leftover)
    for role in CANONICAL_ROLES:
        if role in garment_colors:
            continue
        try:
            garment_colors[role] = next(leftover_iter)
        except StopIteration:
            if result["colors"]:
                garment_colors[role] = result["colors"][0]["name"]

    # Atmosphere — deterministic from seed + warmth.
    atm_rng = random.Random((seed * 1_000_003) ^ 0xA73B)
    ambient_light, shadow_tone = _atmosphere_pair(atm_rng, warmth)
    atmosphere_colors = {
        "ambient_light": ambient_light,
        "shadow_tone":   shadow_tone,
    }

    # Build the substitution map — both garment and atmosphere tokens.
    subs: dict[str, str] = {}
    for token, role in GARMENT_TOKEN_MAP.items():
        if role in garment_colors:
            subs[token] = garment_colors[role]
    for token, key in ATMOSPHERE_TOKEN_MAP.items():
        if key in atmosphere_colors:
            subs[token] = atmosphere_colors[key]

    if warmth >= 0.66:
        tone = "warm"
    elif warmth <= 0.33:
        tone = "cool"
    else:
        tone = "neutral"

    return {
        "seed":              seed,
        "style":             style_preset,
        "color_tone":        tone,
        "num_colors":        num_colors,
        "garment_colors":    garment_colors,
        "atmosphere_colors": atmosphere_colors,
        "subs":              subs,
        "palette_string":    result["palette_string"],
        "raw_tokens":        dict(subs),
    }


def resolve_tokens(text: str, subs: dict[str, str]) -> str:
    """Replace every #token# in ``text`` with its mapped value."""
    if not text or not subs:
        return text or ""
    out = text
    for token, value in subs.items():
        if token in out:
            out = out.replace(token, value)
    return out
