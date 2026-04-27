"""SMP ColorGenerator — emits a structured COLOR_PALETTE_DICT.

Wraps ``core.palette_engine.generate_palette`` (same algorithm as the
existing V1 ``FVM_ColorPaletteGenerator``) and adds:

  * ``garment_colors`` — primary/secondary/accent/neutral/metallic role map.
  * ``atmosphere_colors`` — ``ambient_light`` / ``shadow_tone``, derived
    deterministically from ``warmth`` + seed for use in the LocationCombiner.
  * ``raw_tokens`` — ready-to-substitute mapping for both Outfit and Location
    combiners.
"""

import random

try:
    from ...core.palette_engine import generate_palette
    from ...core.style_presets import STYLE_PRESETS
    from ...core.smp.defaults import ATMOSPHERE_TOKEN_MAP, GARMENT_TOKEN_MAP
except ImportError:  # pragma: no cover
    from core.palette_engine import generate_palette
    from core.style_presets import STYLE_PRESETS
    from core.smp.defaults import ATMOSPHERE_TOKEN_MAP, GARMENT_TOKEN_MAP


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


_HARMONY_TYPES = ["auto", "analogous", "complementary", "split_complementary",
                  "triadic", "tetradic", "monochromatic"]


class FVM_SMP_ColorGenerator:
    """Dict-emitting palette generator for the StructPromptMaker pipeline."""

    CATEGORY = "FVM Tools/SMP/Generators"
    FUNCTION = "generate"
    RETURN_TYPES = ("COLOR_PALETTE_DICT", "STRING")
    RETURN_NAMES = ("color_palette", "summary")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Dict-emitting V2 color generator (StructPromptMaker).\n\n"
        "Same algorithm as FVM_ColorPaletteGenerator, but outputs a typed\n"
        "COLOR_PALETTE_DICT with garment role colors, atmosphere tokens\n"
        "(ambient_light / shadow_tone), and a raw_tokens substitution map\n"
        "consumed by FVM_SMP_OutfitCombiner and FVM_SMP_LocationCombiner."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed":         ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "num_colors":   ("INT", {"default": 5, "min": 2, "max": 8}),
                "harmony_type": (_HARMONY_TYPES, {"default": "auto"}),
                "style_preset": (sorted(STYLE_PRESETS.keys()), {"default": "general"}),
                "vibrancy":     ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "contrast":     ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "warmth":       ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def generate(self, seed, num_colors, harmony_type, style_preset,
                 vibrancy, contrast, warmth):
        result = generate_palette(
            seed=seed, num_colors=num_colors, harmony_type=harmony_type,
            style_preset=style_preset, vibrancy=vibrancy, contrast=contrast,
            warmth=warmth,
        )
        # Roles are baked into colors[i]['role'] by palette_engine, but the
        # engine may emit fewer than the canonical 5 roles depending on the
        # chosen harmony. Backfill any missing canonical role from the
        # remaining unused colors so the combiner contract stays clean
        # ("every #role# token resolves to a name").
        canonical_roles = ["primary", "secondary", "accent", "neutral",
                           "metallic", "tertiary"]
        garment_colors: dict[str, str] = {}
        used_names: set[str] = set()
        for c in result["colors"]:
            role = c.get("role")
            if role and role not in garment_colors:
                garment_colors[role] = c["name"]
                used_names.add(c["name"])

        leftover = [c["name"] for c in result["colors"] if c["name"] not in used_names]
        leftover_iter = iter(leftover)
        for role in canonical_roles:
            if role in garment_colors:
                continue
            try:
                garment_colors[role] = next(leftover_iter)
            except StopIteration:
                # Palette exhausted — fall back to the first emitted color so
                # tokens still resolve to a real name rather than vanishing.
                if result["colors"]:
                    garment_colors[role] = result["colors"][0]["name"]

        # Deterministic atmosphere phrasing per seed + warmth.
        atm_rng = random.Random((seed * 1_000_003) ^ 0xA73B)
        ambient_light, shadow_tone = _atmosphere_pair(atm_rng, warmth)
        atmosphere_colors = {
            "ambient_light": ambient_light,
            "shadow_tone":   shadow_tone,
        }

        # Build raw_tokens for the combiners. Prefer garment_colors for garment
        # tokens, atmosphere_colors for atmosphere tokens.
        raw_tokens: dict[str, str] = {}
        for token, role in GARMENT_TOKEN_MAP.items():
            if role in garment_colors:
                raw_tokens[token] = garment_colors[role]
        for token, key in ATMOSPHERE_TOKEN_MAP.items():
            if key in atmosphere_colors:
                raw_tokens[token] = atmosphere_colors[key]

        # Heuristic color_tone label based on warmth.
        if warmth >= 0.66:
            tone = "warm"
        elif warmth <= 0.33:
            tone = "cool"
        else:
            tone = "neutral"

        palette: dict = {
            "seed": seed,
            "style": style_preset,
            "color_tone": tone,
            "num_colors": num_colors,
            "garment_colors": garment_colors,
            "atmosphere_colors": atmosphere_colors,
            "raw_tokens": raw_tokens,
            "palette_string": result["palette_string"],
        }

        summary = (
            f"Palette: {result['palette_string']} | tone={tone}\n"
            f"Atmosphere: ambient='{ambient_light}', shadow='{shadow_tone}'"
        )
        return (palette, summary)
