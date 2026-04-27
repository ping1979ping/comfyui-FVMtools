"""Default region hints and layer maps shared across SMP generators / assembler.

Bbox values are relative to a canonical full-body portrait frame and are used
as fallback hints when SAM3 detection misses a class. Source: spec §8.2.
"""

from __future__ import annotations


# Person regions — keyed by outfit slot or face-region id.
# bbox is (x_min, y_min, x_max, y_max) in [0, 1].
DEFAULT_PERSON_REGIONS: dict[str, dict] = {
    "face":       {"sam_class": "face",          "bbox": (0.35, 0.05, 0.65, 0.25)},
    "hair":       {"sam_class": "hair",          "bbox": (0.25, 0.00, 0.75, 0.20)},
    "headwear":   {"sam_class": "hat",           "bbox": (0.30, 0.00, 0.70, 0.18)},
    "top":        {"sam_class": "upper_clothes", "bbox": (0.20, 0.15, 0.80, 0.55)},
    "upper_body": {"sam_class": "upper_clothes", "bbox": (0.20, 0.15, 0.80, 0.55)},
    "outerwear":  {"sam_class": "upper_clothes", "bbox": (0.15, 0.12, 0.85, 0.60)},
    "bottom":     {"sam_class": "skirt",         "bbox": (0.25, 0.45, 0.75, 0.85)},
    "lower_body": {"sam_class": "skirt",         "bbox": (0.25, 0.45, 0.75, 0.85)},
    "legwear":    {"sam_class": "legs",          "bbox": (0.30, 0.65, 0.70, 0.95)},
    "footwear":   {"sam_class": "shoes",         "bbox": (0.30, 0.90, 0.70, 1.00)},
    "bag":        {"sam_class": "bag",           "bbox": (0.05, 0.40, 0.35, 0.75)},
    "accessories": {"sam_class": "accessory",    "bbox": None},
    "hands":      {"sam_class": "hands",         "bbox": None},
}


DEFAULT_LOCATION_LAYERS: dict[str, dict] = {
    "background":          {"layer_depth": "background", "bbox": (0.0, 0.0, 1.0, 0.7)},
    "midground":           {"layer_depth": "midground",  "bbox": (0.0, 0.3, 1.0, 0.85)},
    "foreground_element":  {"layer_depth": "foreground", "bbox": (0.0, 0.7, 1.0, 1.0)},
    "architecture_detail": {"layer_depth": "midground",  "bbox": None},
    "props":               {"layer_depth": "midground",  "bbox": None},
    "time_of_day":         {"layer_depth": "atmosphere", "bbox": None},
    "weather":             {"layer_depth": "atmosphere", "bbox": None},
}


# Default color-role assignment per outfit slot.
# Mirrors core/outfit_engine.DEFAULT_COLOR_TAGS but resolves the hash-tag form.
DEFAULT_COLOR_ROLE_BY_SLOT: dict[str, str] = {
    "headwear":    "accent",
    "top":         "primary",
    "outerwear":   "secondary",
    "bottom":      "secondary",
    "legwear":     "tertiary",
    "footwear":    "neutral",
    "accessories": "metallic",
    "bag":         "accent",
}


# Atmosphere token mapping for the LocationCombiner. Order matters: the first
# matching key in the palette's atmosphere_colors wins.
ATMOSPHERE_TOKEN_MAP: dict[str, str] = {
    "#ambient_light#": "ambient_light",
    "#shadow_tone#":   "shadow_tone",
}


# Garment token mapping for the OutfitCombiner.
GARMENT_TOKEN_MAP: dict[str, str] = {
    "#primary#":   "primary",
    "#secondary#": "secondary",
    "#accent#":    "accent",
    "#neutral#":   "neutral",
    "#metallic#":  "metallic",
    "#tertiary#":  "tertiary",
}
