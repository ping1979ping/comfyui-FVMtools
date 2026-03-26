"""Core color utility functions shared across all FVM Color nodes."""
import math
import colorsys
from .color_database import COLOR_DATABASE, NEUTRAL_NAMES, METALLIC_NAMES, NEON_NAMES


def hsl_to_rgb(h, s, l):
    """Convert HSL (h: 0-360, s: 0-100, l: 0-100) to RGB (0-255) tuple."""
    h_norm = h / 360.0
    s_norm = s / 100.0
    l_norm = l / 100.0
    r, g, b = colorsys.hls_to_rgb(h_norm, l_norm, s_norm)
    return int(r * 255), int(g * 255), int(b * 255)


def rgb_to_hsl(r, g, b):
    """Convert RGB (0-255) to HSL (h: 0-360, s: 0-100, l: 0-100) tuple."""
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
    return int(h * 360), int(s * 100), int(l * 100)


def hue_distance(h1, h2):
    """Circular hue distance (0-180)."""
    d = abs(h1 - h2)
    return min(d, 360 - d)


def find_nearest_color_name(h, s, l, vibrancy=0.5, exclude_names=None,
                             allowed_set=None, forbidden_set=None):
    """
    Find the closest color name from COLOR_DATABASE.

    Args:
        h, s, l: Target color in HSL
        vibrancy: Controls whether neon names are allowed (>= 0.8)
        exclude_names: Set of names already used (for deduplication)
        allowed_set: If provided, only consider these names
        forbidden_set: If provided, exclude these names

    Returns:
        Nearest color name string
    """
    best_name = "black"
    best_dist = float('inf')

    exclude = exclude_names or set()
    forbidden = forbidden_set or set()

    for name, (ch, cs, cl) in COLOR_DATABASE.items():
        if name in exclude:
            continue
        if name in forbidden:
            continue
        if allowed_set and name not in allowed_set:
            continue
        if name in NEON_NAMES and vibrancy < 0.8:
            continue

        # Weighted Euclidean distance in HSL space
        dh = hue_distance(h, ch) / 180.0  # 0-1
        ds = abs(s - cs) / 100.0           # 0-1
        dl = abs(l - cl) / 100.0           # 0-1

        # Hue weighted 2x, saturation 1x, lightness 1x
        dist = (dh * 2.0) ** 2 + ds ** 2 + dl ** 2

        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name
