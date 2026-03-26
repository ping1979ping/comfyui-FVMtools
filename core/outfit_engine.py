"""Main outfit generation engine. Combines garment lists, fabrics, presets, and overrides."""

import random

from .outfit_lists import load_garments, load_fabrics, load_fabric_harmony
from .outfit_presets import OUTFIT_PRESETS
from .outfit_parser import parse_overrides, resolve_wildcards


# Slot processing order (head-to-toe)
SLOT_ORDER = ["headwear", "top", "outerwear", "bottom", "footwear", "accessories", "bag"]

# Slots that are always active (not subject to probability roll)
ALWAYS_ACTIVE_SLOTS = {"top", "bottom", "footwear"}

# Default color tags per slot
DEFAULT_COLOR_TAGS = {
    "headwear": "#accent#",
    "top": "#primary#",
    "outerwear": "#secondary#",
    "bottom": "#secondary#",
    "footwear": "#neutral#",
    "accessories": "#metallic#",
    "bag": "#accent#",
}

# Fabrics that should NOT appear in the prompt text
INVISIBLE_FABRICS = {"metal", "plastic", "rubber"}


def generate_outfit(seed, outfit_set="general_female", style_preset="general", formality=0.5,
                    coverage=0.5, slot_enables=None, overrides=None,
                    prefix="wearing ", separator=", "):
    """Generate a complete outfit description with color tags.

    Args:
        seed: INT for deterministic generation
        outfit_set: name of the outfit set directory (e.g. "general_female")
        style_preset: key from OUTFIT_PRESETS
        formality: 0-1 casual to formal
        coverage: 0-1 minimal to maximal (affects garment selection)
        slot_enables: dict {slot_name: bool} -- which slots are enabled
        overrides: dict from parse_overrides() or None
        prefix: text before outfit
        separator: between garment descriptions

    Returns: dict {
        "outfit_prompt": "wearing #primary# silk blouse, ...",
        "outfit_details": "top:blouse:silk:#primary#|...",
        "outfit_info": "Seed: 42, Style: general, ..."
    }
    """
    rng = random.Random(seed)

    # Load preset
    preset = OUTFIT_PRESETS.get(style_preset, OUTFIT_PRESETS["general"])

    # Effective formality: clamp within preset range
    f_min, f_max = preset["formality_range"]
    eff_formality = max(f_min, min(f_max, formality))

    # Load data
    all_garments = {}
    for slot in SLOT_ORDER:
        all_garments[slot] = load_garments(slot, outfit_set)

    fabrics_db = load_fabrics(outfit_set)
    fabric_harmony = load_fabric_harmony()

    # Default slot enables
    if slot_enables is None:
        slot_enables = {s: True for s in SLOT_ORDER}

    # Parse overrides
    if overrides is None:
        overrides = {}

    # Resolve wildcards in override garment/fabric specs
    resolved_overrides = {}
    for slot_name, ov in overrides.items():
        resolved = dict(ov)
        if resolved.get("garment"):
            resolved["garment"] = resolve_wildcards(resolved["garment"], rng)
        if resolved.get("fabric"):
            resolved["fabric"] = resolve_wildcards(resolved["fabric"], rng)
        resolved_overrides[slot_name] = resolved
    overrides = resolved_overrides

    # Determine active slots
    # IMPORTANT: Always consume rng for ALL slots (even skipped) for determinism
    active_slots = []
    for slot in SLOT_ORDER:
        roll = rng.random()  # always consume
        if not slot_enables.get(slot, False):
            continue
        ov_mode = overrides.get(slot, {}).get("mode")
        if ov_mode == "exclude":
            continue
        # Force-activate slot if override specifies a garment
        if ov_mode == "override":
            active_slots.append(slot)
        elif slot in ALWAYS_ACTIVE_SLOTS:
            active_slots.append(slot)
        else:
            # Coverage modifies the slot probability
            base_prob = preset["slot_probabilities"].get(slot, 0.5)
            adjusted_prob = base_prob * (0.5 + coverage)  # coverage 0 -> 50% of base, 1 -> 150% of base
            adjusted_prob = min(1.0, adjusted_prob)
            if roll < adjusted_prob:
                active_slots.append(slot)

    # Determine base fabric family from formality
    preferred_families = preset.get("preferred_fabric_families")

    # Generate outfit pieces
    descriptions = []
    details = []

    for slot in SLOT_ORDER:
        if slot not in active_slots:
            continue

        ov = overrides.get(slot, {})

        if ov.get("mode") == "override" and ov.get("garment"):
            # Use override values
            garment_name = ov["garment"]
            fabric_name = ov.get("fabric")
            color_tag = ov.get("color_tag") or DEFAULT_COLOR_TAGS.get(slot, "#primary#")

            desc = _build_description(color_tag, fabric_name, garment_name)
            descriptions.append(desc)
            details.append(f"{slot}:{garment_name}:{fabric_name or 'none'}:{color_tag}")
            continue

        # Auto-generate
        garments = all_garments.get(slot, [])
        if not garments:
            continue

        # Filter by formality
        compatible = [g for g in garments
                      if g["formality"][0] <= eff_formality <= g["formality"][1]]
        if not compatible:
            # Fallback: find closest formality match
            compatible = garments

        # Weight by probability
        weights = [g["probability"] for g in compatible]
        chosen_garment = rng.choices(compatible, weights=weights, k=1)[0]

        # Pick fabric
        garment_fabrics = chosen_garment["fabrics"]
        fabric_name = _pick_fabric(rng, garment_fabrics, fabrics_db, fabric_harmony,
                                   eff_formality, preferred_families)

        color_tag = DEFAULT_COLOR_TAGS.get(slot, "#primary#")
        desc = _build_description(color_tag, fabric_name, chosen_garment["name"])
        descriptions.append(desc)
        details.append(f"{slot}:{chosen_garment['name']}:{fabric_name or 'none'}:{color_tag}")

    # Build outputs
    outfit_prompt = prefix + separator.join(descriptions) if descriptions else prefix.rstrip()
    outfit_details = "|".join(details)
    outfit_info = _build_info(seed, style_preset, eff_formality, coverage, active_slots, details)

    return {
        "outfit_prompt": outfit_prompt,
        "outfit_details": outfit_details,
        "outfit_info": outfit_info,
    }


def _build_description(color_tag, fabric_name, garment_name):
    """Build a single garment description like '#primary# silk blouse'."""
    if fabric_name and fabric_name not in INVISIBLE_FABRICS:
        return f"{color_tag} {fabric_name} {garment_name}"
    else:
        return f"{color_tag} {garment_name}"


def _pick_fabric(rng, garment_fabrics, fabrics_db, fabric_harmony,
                 formality, preferred_families):
    """Pick a compatible fabric for a garment."""
    if not garment_fabrics:
        return None

    # Filter by preferred families if set
    candidates = garment_fabrics
    if preferred_families:
        family_filtered = [f for f in candidates
                          if f in fabrics_db and fabrics_db[f]["family"] in preferred_families]
        if family_filtered:
            candidates = family_filtered

    # Among candidates, prefer fabrics closer to target formality
    if len(candidates) > 1 and fabrics_db:
        # Weight by inverse distance to target formality
        weights = []
        for fab in candidates:
            if fab in fabrics_db:
                dist = abs(fabrics_db[fab]["formality"] - formality)
                weights.append(1.0 / (dist + 0.1))
            else:
                weights.append(0.5)
        return rng.choices(candidates, weights=weights, k=1)[0]

    return rng.choice(candidates)


def _build_info(seed, style_preset, formality, coverage, active_slots, details):
    """Build a human-readable info string."""
    lines = [
        f"Seed: {seed}",
        f"Style: {style_preset}",
        f"Formality: {formality:.2f}",
        f"Coverage: {coverage:.2f}",
        f"Active slots: {', '.join(active_slots)}",
        f"Pieces: {len(details)}",
    ]
    for detail in details:
        parts = detail.split(":")
        if len(parts) >= 3:
            lines.append(f"  {parts[0]}: {parts[1]} ({parts[2]})")
    return "\n".join(lines)
