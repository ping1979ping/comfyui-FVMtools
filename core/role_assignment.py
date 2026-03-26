"""Assign semantic roles (primary, secondary, accent, neutral, metallic) to palette colors."""

from .color_database import NEUTRAL_NAMES, METALLIC_NAMES
from .color_utils import hue_distance


def assign_roles(colors):
    """
    Assign semantic roles to a list of palette color dicts.

    Each color dict must have at minimum: {"name": str, "hsl": (h, s, l)}
    Adds a "role" key to each dict: "primary", "secondary", "accent", "neutral", "metallic", or None.

    Rules:
    - primary: highest saturation chromatic color
    - secondary: second highest saturation chromatic color
    - accent: chromatic color with greatest hue distance from primary
    - neutral: first color whose name is in NEUTRAL_NAMES (or lowest saturation)
    - metallic: first color whose name is in METALLIC_NAMES
    """
    if not colors:
        return colors

    # Reset roles
    for c in colors:
        c["role"] = None

    # Separate chromatic vs neutral/metallic by name
    chromatic = []
    neutrals = []
    metallics = []

    for c in colors:
        name = c["name"]
        if name in METALLIC_NAMES:
            metallics.append(c)
        elif name in NEUTRAL_NAMES:
            neutrals.append(c)
        else:
            chromatic.append(c)

    # Sort chromatic by saturation (descending)
    chromatic.sort(key=lambda c: c["hsl"][1], reverse=True)

    # Assign primary
    if chromatic:
        chromatic[0]["role"] = "primary"

    # Assign accent (most hue-distinct from primary)
    if len(chromatic) >= 2:
        primary_hue = chromatic[0]["hsl"][0]
        best_accent = None
        best_dist = -1
        for c in chromatic[1:]:
            dist = hue_distance(primary_hue, c["hsl"][0])
            if dist > best_dist:
                best_dist = dist
                best_accent = c
        if best_accent is not None:
            best_accent["role"] = "accent"

    # Assign secondary (highest saturation unassigned chromatic)
    if len(chromatic) >= 2:
        for c in chromatic:
            if c["role"] is None:
                c["role"] = "secondary"
                break

    # Assign neutral
    if neutrals:
        neutrals[0]["role"] = "neutral"
    elif not chromatic:
        # All colors are metallics or empty — assign neutral to lowest saturation overall
        all_unassigned = [c for c in colors if c["role"] is None]
        if all_unassigned:
            all_unassigned.sort(key=lambda c: c["hsl"][1])
            all_unassigned[0]["role"] = "neutral"

    # Assign metallic
    if metallics:
        metallics[0]["role"] = "metallic"
    elif len(neutrals) >= 2:
        # Second neutral as metallic fallback
        neutrals[1]["role"] = "metallic"

    return colors
