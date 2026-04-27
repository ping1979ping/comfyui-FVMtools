"""Tier-based regional prompt assembler — pure-python core for unit testing.

Implements spec §7: tier ordering for face / body / outfit / location strings.
Kept dependency-free so it can be imported under pytest mocks.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional


# Spatial garment order: head → toe → side → accessories
GARMENT_SPATIAL_ORDER = [
    "headwear",
    "upper_body",
    "outerwear",   # in case caller used the slot name directly
    "top",
    "lower_body",
    "bottom",
    "legwear",
    "footwear",
    "bag",
    "accessories",
    "accessory",
]

# Layer order back-to-front, atmosphere last
LOCATION_LAYER_ORDER = [
    "background",
    "midground",
    "architecture_detail",
    "props",
    "foreground_element",
    "time_of_day",
    "weather",
]


FACE_QUALITY_TAGS = "sharp focus, detailed face, high quality"
BODY_QUALITY_TAGS = "correct anatomy, natural proportions"
DEFAULT_HANDS_FRAGMENT = "(detailed hands:1.05), natural finger position"


def _join(parts: Iterable[Optional[str]], sep: str = ", ") -> str:
    return sep.join(p.strip() for p in parts if p and p.strip())


def _safe_subject_dict(subject: Any) -> dict:
    if not subject:
        return {}
    if isinstance(subject, dict):
        return subject
    if isinstance(subject, str):
        try:
            return json.loads(subject)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def build_face_prompt(subject: dict, *, eye_boost: float = 1.1,
                      include_quality: bool = True) -> str:
    s = subject or {}
    age = s.get("age_desc") or ""
    gender = s.get("gender") or ""
    anchor = _join([age, gender], sep=" ")

    skin_parts: list[str] = []
    if s.get("ethnicity_tag"):
        skin_parts.append(s["ethnicity_tag"])
    skin_parts.extend(s.get("skin_tags") or [])

    feature_parts: list[str] = []
    if eye := s.get("eye_desc"):
        if eye_boost and eye_boost != 1.0:
            feature_parts.append(f"({eye}:{eye_boost:.2f})")
        else:
            feature_parts.append(eye)
    for key in ("brow_desc", "lip_desc", "nose_desc"):
        if v := s.get(key):
            feature_parts.append(v)

    expression = s.get("expression")
    makeup = s.get("makeup")
    hair_short = s.get("hair_color_length")

    parts = [anchor]
    parts.extend(skin_parts)
    parts.extend(feature_parts)
    if expression:
        parts.append(expression)
    if makeup:
        parts.append(makeup)
    if hair_short:
        parts.append(hair_short)
    if include_quality:
        parts.append(FACE_QUALITY_TAGS)
    return _join(parts)


def build_body_prompt(subject: dict, *, include_quality: bool = True) -> str:
    s = subject or {}
    anchor = _join([s.get("age_desc"), s.get("gender")], sep=" ")

    build_height: list[str] = []
    if v := s.get("body_build"):
        build_height.append(v)
    if v := s.get("body_height"):
        build_height.append(v)

    pose = s.get("pose_hint")

    hair_full = s.get("hair_full")
    hair_str = None
    if isinstance(hair_full, dict):
        # Compose from common keys
        bits = [hair_full.get("color"), hair_full.get("style"),
                hair_full.get("length"), hair_full.get("details")]
        hair_str = _join(bits, sep=" ")
    elif isinstance(hair_full, str):
        hair_str = hair_full

    skin_body = ", ".join(s.get("skin_tags") or [])

    parts = [anchor]
    parts.extend(build_height)
    if pose:
        parts.append(pose)
    parts.append(DEFAULT_HANDS_FRAGMENT)
    if hair_str:
        parts.append(hair_str)
    if skin_body:
        parts.append(skin_body)
    if include_quality:
        parts.append(BODY_QUALITY_TAGS)
    return _join(parts)


def _ordered_garments(garments: dict, custom_order: Optional[list[str]] = None) -> list[tuple[str, dict]]:
    order = custom_order or GARMENT_SPATIAL_ORDER
    seen = set()
    out: list[tuple[str, dict]] = []
    for key in order:
        if key in garments and key not in seen:
            out.append((key, garments[key]))
            seen.add(key)
    # Append any garments the caller used a non-canonical key for, in their
    # original insertion order, so we never lose data.
    for key, g in garments.items():
        if key not in seen:
            out.append((key, g))
            seen.add(key)
    return out


def build_outfit_prompt(outfit: dict, *, custom_order: Optional[list[str]] = None) -> str:
    o = outfit or {}
    garments = o.get("garments") or {}
    fragments = []
    for _, g in _ordered_garments(garments, custom_order):
        if frag := (g or {}).get("prompt_fragment"):
            fragments.append(frag)
    if formality := o.get("formality"):
        fragments.append(f"{formality} style")
    return _join(fragments)


def build_location_prompt(location: dict) -> str:
    l = location or {}
    elements = l.get("elements") or {}
    fragments = []
    for key in LOCATION_LAYER_ORDER:
        if key in elements and elements[key]:
            if frag := elements[key].get("prompt_fragment"):
                fragments.append(frag)
    return _join(fragments)


def build_region_map(outfit: dict, location: dict) -> list[dict]:
    """Flatten region hints from outfit + location into a single REGION_MAP."""
    regions: list[dict] = []
    for region_id, g in (outfit or {}).get("garments", {}).items():
        hint = (g or {}).get("region_hint") or {}
        regions.append({
            "region_id":      hint.get("region_id") or region_id,
            "sam_class_hint": hint.get("sam_class_hint"),
            "bbox_relative":  hint.get("bbox_relative"),
            "layer_depth":    hint.get("layer_depth", "subject"),
            "prompt_fragment": (g or {}).get("prompt_fragment", ""),
        })
    for elem_id, e in (location or {}).get("elements", {}).items():
        hint = (e or {}).get("region_hint") or {}
        regions.append({
            "region_id":      hint.get("region_id") or elem_id,
            "sam_class_hint": hint.get("sam_class_hint"),
            "bbox_relative":  hint.get("bbox_relative"),
            "layer_depth":    hint.get("layer_depth", e.get("layer", "background")),
            "prompt_fragment": (e or {}).get("prompt_fragment", ""),
        })
    return regions


def build_sam_class_lookup(face: str, outfit: dict, location: dict) -> dict[str, str]:
    """Map SAM3 class hints to the corresponding regional prompt fragment."""
    lookup: dict[str, str] = {}
    if face:
        lookup["face"] = face
    for _, g in (outfit or {}).get("garments", {}).items():
        hint = (g or {}).get("region_hint") or {}
        sc = hint.get("sam_class_hint")
        if sc and sc not in lookup:
            lookup[sc] = (g or {}).get("prompt_fragment", "")
    for _, e in (location or {}).get("elements", {}).items():
        hint = (e or {}).get("region_hint") or {}
        sc = hint.get("sam_class_hint")
        if sc and sc not in lookup:
            lookup[sc] = (e or {}).get("prompt_fragment", "")
    return lookup


def assemble_structured(prompt_dict: Optional[dict] = None,
                        *,
                        outfit: Optional[dict] = None,
                        location: Optional[dict] = None,
                        subject: Optional[Any] = None,
                        subject_index: int = 0,
                        eye_boost: float = 1.1,
                        include_quality: bool = True,
                        custom_outfit_order: Optional[list[str]] = None) -> dict:
    """Assemble the full STRUCTURED_PROMPTS dict.

    Either pass a complete ``prompt_dict`` (post-Aggregator) or pass
    ``outfit`` / ``location`` / ``subject`` directly to bypass the builder
    pipeline. Direct args take precedence over those derived from
    ``prompt_dict``.
    """
    # Resolve subject/outfit/location from prompt_dict if not given directly.
    if prompt_dict:
        if subject is None:
            subjects = prompt_dict.get("subjects") or []
            if subjects and 0 <= subject_index < len(subjects):
                subject = subjects[subject_index]
        if outfit is None:
            outfits_map = prompt_dict.get("outfits") or {}
            sid = (_safe_subject_dict(subject) or {}).get("id", "subject_1")
            outfit = outfits_map.get(sid)
        if location is None:
            location = prompt_dict.get("location")

    subj = _safe_subject_dict(subject)
    o = outfit or {}
    l = location or {}

    face = build_face_prompt(subj, eye_boost=eye_boost, include_quality=include_quality)
    body = build_body_prompt(subj, include_quality=include_quality)
    outfit_p = build_outfit_prompt(o, custom_order=custom_outfit_order)
    loc_p = build_location_prompt(l)

    region_map = build_region_map(o, l)
    sam_lookup = build_sam_class_lookup(face, o, l)

    return {
        "face":              face,
        "body":              body,
        "outfit":            outfit_p,
        "location":          loc_p,
        "region_map":        region_map,
        "sam_class_lookup":  sam_lookup,
        "raw_dict":          prompt_dict or {},
    }
