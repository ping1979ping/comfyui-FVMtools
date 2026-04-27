"""StructPromptMaker (SMP) — typed dict-based structural prompt pipeline.

Provides the schema, custom-type registrations, and shared defaults that the
SMP node suite consumes. See PROJECT.md / ROADMAP.md for the v1-smp milestone.
"""

from .schema import (
    Meta,
    Subject,
    GarmentEntry,
    OutfitDict,
    LocationElement,
    LocationDict,
    ColorPalette,
    RegionEntry,
    StructuredPrompts,
    PromptDict,
)
from .defaults import DEFAULT_PERSON_REGIONS, DEFAULT_LOCATION_LAYERS

__all__ = [
    "Meta", "Subject", "GarmentEntry", "OutfitDict",
    "LocationElement", "LocationDict", "ColorPalette",
    "RegionEntry", "StructuredPrompts", "PromptDict",
    "DEFAULT_PERSON_REGIONS", "DEFAULT_LOCATION_LAYERS",
]
