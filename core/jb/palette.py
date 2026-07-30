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
    from .color_moods import mood_colors, mood_engine_kwargs
except ImportError:  # pragma: no cover
    from core.palette_engine import generate_palette
    from core.smp.defaults import ATMOSPHERE_TOKEN_MAP, GARMENT_TOKEN_MAP
    from core.jb.color_moods import mood_colors, mood_engine_kwargs


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
                  contrast: float = 0.5, warmth: float = 0.5,
                  color_mood: str = "auto") -> dict:
    """Return a dict with garment_colors, atmosphere_colors, raw_tokens, subs.

    ``subs`` is the merged token map ready for fragment substitution
    (covers both garment and atmosphere tokens in a single pass).

    ``color_mood`` replaces the six sliders with one choice. Pool-based moods
    ("everyday_muted", "neutral_basics", …) bypass the harmony engine entirely
    and draw from colours people actually wear; engine moods ("bold", "pastel")
    just preset the sliders. ``auto`` keeps the previous behaviour untouched.
    """
    engine_kwargs = {
        "num_colors": num_colors, "harmony_type": harmony_type,
        "style_preset": style_preset, "vibrancy": vibrancy,
        "contrast": contrast, "warmth": warmth,
    }
    engine_kwargs.update(mood_engine_kwargs(color_mood))

    pool = mood_colors(color_mood, seed, num_colors)
    if pool is not None:
        return _palette_from_names(
            names=pool, seed=seed, color_mood=color_mood,
            style_preset=style_preset, warmth=warmth, num_colors=num_colors,
        )

    result = generate_palette(seed=seed, **engine_kwargs)
    num_colors = engine_kwargs["num_colors"]
    warmth = engine_kwargs["warmth"]
    style_preset = engine_kwargs["style_preset"]

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
        "color_mood":        color_mood,
        "color_tone":        tone,
        "num_colors":        num_colors,
        "garment_colors":    garment_colors,
        "atmosphere_colors": atmosphere_colors,
        "subs":              subs,
        "palette_string":    result["palette_string"],
        "raw_tokens":        dict(subs),
    }


def _palette_from_names(*, names: list[str], seed: int, color_mood: str,
                        style_preset: str, warmth: float, num_colors: int) -> dict:
    """Build the same payload as build_palette from a fixed list of colour names.

    Used by the pool-based moods, which pick real clothing colours instead of
    computing a hue harmony.
    """
    garment_colors: dict[str, str] = {}
    for index, role in enumerate(CANONICAL_ROLES):
        garment_colors[role] = names[index % len(names)]
    # Metallic is the one role a curated pool should not fill with fabric colour.
    if len(names) >= 2:
        garment_colors["metallic"] = "brushed silver" if seed % 2 else "warm gold"

    atm_rng = random.Random((seed * 1_000_003) ^ 0xA73B)
    ambient_light, shadow_tone = _atmosphere_pair(atm_rng, warmth)
    atmosphere_colors = {"ambient_light": ambient_light, "shadow_tone": shadow_tone}

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
        "color_mood":        color_mood,
        "color_tone":        tone,
        "num_colors":        len(names),
        "garment_colors":    garment_colors,
        "atmosphere_colors": atmosphere_colors,
        "subs":              subs,
        "palette_string":    ", ".join(names),
        "raw_tokens":        dict(subs),
    }


def apply_color_overrides(palette: dict, overrides: dict[str, str]) -> dict:
    """Force specific colour roles over a generated palette, in place.

    ``overrides`` maps role → colour name, e.g. ``{"primary": "navy blue"}``.
    Garment roles (primary/secondary/accent/neutral/metallic/tertiary) and the
    two atmosphere keys (ambient_light/shadow_tone) are accepted; unknown
    roles are ignored so a typo cannot break the build. The ``subs`` token map
    and ``palette_string`` are updated to match.
    """
    if not overrides:
        return palette

    applied: dict[str, str] = {}
    for role, value in overrides.items():
        role = str(role).strip().lower().strip("#")
        value = str(value).strip()
        if not role or not value:
            continue
        if role in CANONICAL_ROLES:
            palette["garment_colors"][role] = value
            applied[role] = value
        elif role in ("ambient_light", "shadow_tone"):
            palette["atmosphere_colors"][role] = value
            applied[role] = value

    if not applied:
        return palette

    subs = palette["subs"]
    for token, role in GARMENT_TOKEN_MAP.items():
        if role in palette["garment_colors"]:
            subs[token] = palette["garment_colors"][role]
    for token, key in ATMOSPHERE_TOKEN_MAP.items():
        if key in palette["atmosphere_colors"]:
            subs[token] = palette["atmosphere_colors"][key]
    palette["raw_tokens"] = dict(subs)
    note = ", ".join(f"{r}={v}" for r, v in applied.items())
    palette["palette_string"] = f"{palette['palette_string']}  [overridden: {note}]"
    return palette


def resolve_tokens(text: str, subs: dict[str, str]) -> str:
    """Replace every #token# in ``text`` with its mapped value."""
    if not text or not subs:
        return text or ""
    out = text
    for token, value in subs.items():
        if token in out:
            out = out.replace(token, value)
    return out
