"""Main palette generation engine. Combines harmony, style presets, and color matching."""

import random

from .color_database import COLOR_DATABASE, NEUTRAL_NAMES, METALLIC_NAMES
from .color_utils import hsl_to_rgb, find_nearest_color_name
from .harmony import generate_harmony_hues
from .style_presets import STYLE_PRESETS
from .role_assignment import assign_roles


def generate_palette(seed, num_colors, harmony_type="auto", style_preset="general",
                     vibrancy=0.5, contrast=0.5, warmth=0.5, neutral_ratio=0.4,
                     include_metallics=True):
    """
    Generate a named color palette.

    Args:
        seed: Integer seed for deterministic generation
        num_colors: Number of colors in palette (2-8)
        harmony_type: Harmony algorithm or "auto"
        style_preset: Key from STYLE_PRESETS
        vibrancy: 0-1, controls color saturation
        contrast: 0-1, controls lightness spread
        warmth: 0-1, biases hue toward warm (>0.5) or cool (<0.5)
        neutral_ratio: 0-1, fraction of colors that are neutrals
        include_metallics: Whether to include metallic colors in neutral slots

    Returns:
        dict with keys: "colors" (list of dicts), "palette_string" (str), "info" (str)
    """
    rng = random.Random(seed)

    # Clamp num_colors
    num_colors = max(2, min(8, num_colors))

    # Load style preset
    preset = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["general"])

    # Apply style modifiers (clamp to 0-1)
    eff_vibrancy = _clamp(vibrancy + preset["vibrancy_mod"])
    eff_contrast = _clamp(contrast + preset["contrast_mod"])
    eff_warmth = _clamp(warmth + preset["warmth_mod"])

    # Pick base hue
    base_hue = _pick_base_hue(rng, preset["hue_ranges"], eff_warmth)

    # Pick harmony type
    if harmony_type == "auto":
        harmonies = preset.get("preferred_harmonies")
        if harmonies:
            harmony_type = rng.choice(harmonies)
        else:
            harmony_type = rng.choice(["analogous", "complementary", "triadic",
                                       "split_complementary"])

    # Determine neutral/chromatic split
    num_neutrals = round(num_colors * neutral_ratio)
    num_neutrals = min(num_neutrals, num_colors - 1)  # at least 1 chromatic
    if num_colors <= 2 and num_neutrals > 0:
        num_neutrals = min(num_neutrals, 1)
    num_chromatic = num_colors - num_neutrals

    # Generate harmony hues for chromatic colors
    hues = generate_harmony_hues(base_hue, harmony_type, num_chromatic)

    # Build chromatic colors
    exclude_names = set()
    forbidden = preset.get("forbidden_names", set())
    colors = []

    for i, hue in enumerate(hues):
        # Add jitter to saturation and lightness
        jitter_s = rng.uniform(-8, 8)
        jitter_l = rng.uniform(-5, 5)

        s = _lerp(20, 95, eff_vibrancy) + jitter_s
        s = _clamp_range(s, 5, 100)

        # Spread lightness based on contrast
        center_l = 50
        spread = eff_contrast * 35  # max ±35 from center
        if num_chromatic > 1:
            position = i / (num_chromatic - 1)  # 0 to 1
            l = center_l + (position - 0.5) * 2 * spread + jitter_l
        else:
            l = center_l + jitter_l
        l = _clamp_range(l, 10, 90)

        name = find_nearest_color_name(
            hue, s, l,
            vibrancy=eff_vibrancy,
            exclude_names=exclude_names,
            forbidden_set=forbidden,
        )
        exclude_names.add(name)

        db_hsl = COLOR_DATABASE[name]
        rgb = hsl_to_rgb(*db_hsl)
        colors.append({
            "name": name,
            "hsl": db_hsl,
            "rgb": rgb,
            "role": None,
        })

    # Build neutral/metallic colors
    neutral_bias = preset.get("neutral_bias", [])
    metallic_added = False

    for i in range(num_neutrals):
        # First neutral slot: try metallic if enabled
        if include_metallics and not metallic_added and i == 0 and num_neutrals >= 1:
            metallic_name = _pick_neutral_or_metallic(
                rng, exclude_names, forbidden, neutral_bias,
                prefer_metallic=True,
            )
            if metallic_name and metallic_name in METALLIC_NAMES:
                metallic_added = True
            name = metallic_name
        else:
            name = _pick_neutral_or_metallic(
                rng, exclude_names, forbidden, neutral_bias,
                prefer_metallic=False,
            )

        if name is None:
            name = "black"  # ultimate fallback

        exclude_names.add(name)
        db_hsl = COLOR_DATABASE[name]
        rgb = hsl_to_rgb(*db_hsl)
        colors.append({
            "name": name,
            "hsl": db_hsl,
            "rgb": rgb,
            "role": None,
        })

    # Shuffle to mix neutrals among chromatic (deterministic)
    rng.shuffle(colors)

    # Assign roles
    assign_roles(colors)

    # Build output
    palette_string = ", ".join(c["name"] for c in colors)
    info = _build_info(colors, harmony_type, style_preset, seed)

    return {
        "colors": colors,
        "palette_string": palette_string,
        "info": info,
    }


# ── Internal helpers ──

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _clamp_range(v, lo, hi):
    return max(lo, min(hi, v))


def _lerp(a, b, t):
    return a + (b - a) * t


def _pick_base_hue(rng, hue_ranges, warmth):
    """Pick a base hue within allowed ranges, biased by warmth."""
    if hue_ranges is None:
        # Warmth bias: warm hues 0-60, 300-360; cool hues 150-270
        if warmth > 0.6:
            base = rng.choice([rng.uniform(0, 60), rng.uniform(300, 360)])
        elif warmth < 0.4:
            base = rng.uniform(150, 270)
        else:
            base = rng.uniform(0, 360)
        return base % 360

    # Pick from allowed ranges
    # Flatten ranges and pick randomly weighted by range size
    total = sum(((end - start) % 360) or 360 for start, end in hue_ranges)
    pick = rng.uniform(0, total)
    cumulative = 0
    for start, end in hue_ranges:
        size = ((end - start) % 360) or 360
        cumulative += size
        if pick <= cumulative:
            offset = pick - (cumulative - size)
            return (start + offset) % 360
    # Fallback
    start, end = hue_ranges[0]
    return rng.uniform(start, end) % 360


def _pick_neutral_or_metallic(rng, exclude_names, forbidden, neutral_bias,
                               prefer_metallic=False):
    """Pick a neutral or metallic color name."""
    if prefer_metallic:
        # Try metallics first
        available = [n for n in METALLIC_NAMES
                     if n not in exclude_names and n not in forbidden]
        if available:
            # Prefer from neutral_bias if any metallics are listed there
            biased = [n for n in neutral_bias if n in available]
            if biased:
                return rng.choice(biased)
            return rng.choice(available)

    # Try neutrals
    available = [n for n in NEUTRAL_NAMES
                 if n not in exclude_names and n not in forbidden]
    if not available:
        # Fall back to metallics
        available = [n for n in METALLIC_NAMES
                     if n not in exclude_names and n not in forbidden]
    if not available:
        return None

    # Prefer from neutral_bias
    biased = [n for n in neutral_bias if n in available]
    if biased:
        return rng.choice(biased)
    return rng.choice(available)


def _build_info(colors, harmony_type, style_preset, seed):
    """Build a human-readable info string."""
    role_map = {}
    for c in colors:
        if c["role"]:
            role_map[c["role"]] = c["name"]

    lines = [
        f"Seed: {seed}",
        f"Harmony: {harmony_type}",
        f"Style: {style_preset}",
        f"Colors: {len(colors)}",
    ]
    for role in ("primary", "secondary", "accent", "neutral", "metallic"):
        if role in role_map:
            lines.append(f"  {role}: {role_map[role]}")

    return "\n".join(lines)
