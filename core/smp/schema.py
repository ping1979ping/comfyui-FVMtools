"""Pydantic v2 schema for the StructPromptMaker pipeline.

All nodes pass plain dicts on the wire; pydantic models are used for validation
and round-trip serialization in tests and the SidecarSaver. Models are kept
lean — fields parked for P7 (camera, lighting, presets, batch lock/vary) are
intentionally absent.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

LayerDepth = Literal["background", "midground", "foreground", "subject", "atmosphere"]
ColorRole = Literal["primary", "secondary", "accent", "neutral", "metallic", "tertiary"]
TargetModel = Literal[
    "z-image-turbo", "nano-banana", "flux", "flux2", "sdxl", "generic"
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=False)


# ─── Meta & Subject ────────────────────────────────────────────────────────


class Meta(_Base):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    target_model: TargetModel = "generic"
    seed: int = 0
    label: Optional[str] = None


class Subject(_Base):
    """Subject description used by SubjectBuilder and StructuredPromptAssembler."""

    id: str = "subject_1"
    age_desc: Optional[str] = None
    gender: Optional[str] = None
    ethnicity_tag: Optional[str] = None
    skin_tags: list[str] = Field(default_factory=list)
    eye_desc: Optional[str] = None
    brow_desc: Optional[str] = None
    lip_desc: Optional[str] = None
    nose_desc: Optional[str] = None
    expression: Optional[str] = None
    hair_color_length: Optional[str] = None
    hair_full: Optional[dict[str, Any]] = None
    body_build: Optional[str] = None
    body_height: Optional[str] = None
    pose_hint: Optional[str] = None
    visibility: Literal["full", "partial", "background"] = "full"


# ─── Region hints ──────────────────────────────────────────────────────────


class RegionHint(_Base):
    region_id: str
    sam_class_hint: Optional[str] = None
    bbox_relative: Optional[tuple[float, float, float, float]] = None
    layer_depth: LayerDepth = "subject"


# ─── Outfit ────────────────────────────────────────────────────────────────


class GarmentEntry(_Base):
    name: str
    probability: float = 1.0
    coverage: float = 0.0
    fabric: Optional[str] = None
    color_role: Optional[ColorRole] = None
    color_resolved: Optional[str] = None
    prompt_fragment: str = ""
    region_hint: Optional[RegionHint] = None


class OutfitDict(_Base):
    set_name: str
    seed: int = 0
    formality: Optional[str] = None
    coverage_target: float = 0.5
    color_tone: Optional[str] = None
    garments: dict[str, GarmentEntry] = Field(default_factory=dict)


# ─── Location ──────────────────────────────────────────────────────────────


class LocationElement(_Base):
    name: str
    probability: float = 1.0
    coverage: float = 0.0
    texture: Optional[str] = None
    layer: LayerDepth = "background"
    prompt_fragment: str = ""
    region_hint: Optional[RegionHint] = None


class LocationDict(_Base):
    set_name: str
    seed: int = 0
    color_tone: Optional[str] = None
    elements: dict[str, LocationElement] = Field(default_factory=dict)


# ─── Color palette ─────────────────────────────────────────────────────────


class ColorPalette(_Base):
    seed: int = 0
    style: str = "neutral"
    garment_colors: dict[str, str] = Field(default_factory=dict)
    atmosphere_colors: dict[str, str] = Field(default_factory=dict)
    raw_tokens: dict[str, str] = Field(default_factory=dict)


# ─── Structured prompts ────────────────────────────────────────────────────


class RegionEntry(_Base):
    region_id: str
    sam_class_hint: Optional[str] = None
    bbox_relative: Optional[tuple[float, float, float, float]] = None
    layer_depth: LayerDepth = "subject"
    prompt_fragment: str = ""


class StructuredPrompts(_Base):
    face: str = ""
    body: str = ""
    outfit: str = ""
    location: str = ""
    region_map: list[RegionEntry] = Field(default_factory=list)
    sam_class_lookup: dict[str, str] = Field(default_factory=dict)
    raw_dict: dict[str, Any] = Field(default_factory=dict)


# ─── Master container ─────────────────────────────────────────────────────


class PromptDict(_Base):
    """Top-level container flowing through the SMP pipeline."""

    meta: Meta = Field(default_factory=Meta)
    subjects: list[Subject] = Field(default_factory=list)
    outfits: dict[str, OutfitDict] = Field(default_factory=dict)
    location: Optional[LocationDict] = None
    palette: Optional[ColorPalette] = None
    extras: dict[str, Any] = Field(default_factory=dict)


def validate_prompt_dict(payload: Any) -> PromptDict:
    """Best-effort coercion of an arbitrary dict / model instance into PromptDict."""
    if isinstance(payload, PromptDict):
        return payload
    if isinstance(payload, dict):
        return PromptDict.model_validate(payload)
    raise TypeError(f"Cannot coerce {type(payload).__name__} into PromptDict")
