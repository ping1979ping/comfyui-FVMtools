"""K2 Lab — Prompt-Kompilierung für regionales Krea-2-Prompting.

Krea 2 hat keinen getrennten regionalen Konditionierungszweig: alle Regionen
landen in *einem* Prompt. Räumliche Zuordnung entsteht durch zwei Dinge:

1. eine sprachliche Ortsangabe pro Region ("In the upper left side, render …"),
2. den Attention-Router, der die Tokenspanne dieser Klausel an die Bildtoken
   ihrer Box bindet (siehe ``attention.py``).

Dieses Modul erzeugt (1) und merkt sich, welcher Zeichenbereich zu welcher
Region gehört. ``binding.py`` übersetzt das später in Tokenindizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .geometry import (
    CanvasGeometry,
    PixelBox,
    apply_subject_competition,
)

BACKEND = "fvm-k2-unified-spatial-attention-v1"
GLOBAL_SCOPE = "__global__"

ROLE_AUTO = "auto"
ROLE_SUBJECT = "subject"
ROLE_BACKGROUND = "background"
ROLES = (ROLE_AUTO, ROLE_SUBJECT, ROLE_BACKGROUND)


@dataclass
class RegionDefinition:
    """Eine benutzerdefinierte Region vor der Kompilierung."""

    region_id: str
    name: str
    box: PixelBox
    prompt: str = ""
    identity_prompt: str = ""
    negative_prompt: str = ""
    enabled: bool = True
    priority: int = 0
    role: str = ROLE_AUTO

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"Region {self.name!r}: role muss {ROLES} sein, war {self.role!r}"
            )

    @property
    def description(self) -> str:
        """Szene und Identität zu *einem* Satzteil verbinden.

        Wichtig: die Identität wird angehängt ("… , with <identity>") und nicht
        als eigener Satz vorangestellt. Ein vorangestellter Satz wie
        "a freckled face with green eyes." wird vom Modell als eigenständiges
        Bildobjekt gelesen — es malt dann buchstäblich ein zweites Gesicht in
        die Region.
        """
        identity = _clean(self.identity_prompt)
        scene = _clean(self.prompt)
        if identity and scene:
            return f"{scene}, with {identity}"
        return identity or scene


@dataclass
class EmphasisRequest:
    """Exakte Phrase, deren Bindung an ihre Region verstärkt werden soll."""

    scope_id: str
    phrase: str
    strength: float = 0.5
    occurrence: int = 0

    def __post_init__(self) -> None:
        if not self.phrase.strip():
            raise ValueError("Emphasis-Phrase darf nicht leer sein")
        if not 0.0 <= self.strength <= 2.0:
            raise ValueError("Emphasis-Stärke muss zwischen 0 und 2 liegen")
        if self.occurrence < 0:
            raise ValueError("Emphasis-Occurrence darf nicht negativ sein")


@dataclass
class CompiledRegion:
    region_id: str
    name: str
    box: PixelBox
    role: str
    clause: str
    char_span: tuple[int, int]
    field: np.ndarray  # weiches Attention-Feld pro Bildtoken
    mask: np.ndarray  # harte Boxmaske pro Bildtoken
    prompt: str
    identity_prompt: str
    negative_prompt: str
    identity_char_span: tuple[int, int] | None = None
    trigger_char_spans: tuple[tuple[int, int], ...] = ()


@dataclass
class CompiledEmphasis:
    scope_id: str
    phrase: str
    strength: float
    char_span: tuple[int, int]
    field: np.ndarray


@dataclass
class CompiledPlan:
    """Ergebnis der Prompt-Kompilierung — noch ohne Tokenbindung."""

    geometry: CanvasGeometry
    prompt: str
    regions: tuple[CompiledRegion, ...]
    emphases: tuple[CompiledEmphasis, ...] = ()
    strength: float = 1.0
    outside_penalty: float = 1.0
    falloff_pixels: float = 128.0
    subject_competition: bool = True
    subject_fill: bool = True
    late_step_scale: float = 0.35
    backend: str = BACKEND
    identity_triggers: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return self.geometry.token_count

    def region_by_id(self, region_id: str) -> CompiledRegion | None:
        return next((r for r in self.regions if r.region_id == region_id), None)

    def union_field(self) -> np.ndarray:
        values = np.zeros(self.token_count, dtype=np.float32)
        for region in self.regions:
            values = np.maximum(values, region.field)
        return values

    def summary(self) -> dict:
        return {
            "backend": self.backend,
            "compiled_prompt": self.prompt,
            "strength": self.strength,
            "outside_penalty": self.outside_penalty,
            "falloff_pixels": self.falloff_pixels,
            "subject_competition": self.subject_competition,
            "subject_fill": self.subject_fill,
            "late_step_scale": self.late_step_scale,
            "image_token_grid": [
                self.geometry.token_width,
                self.geometry.token_height,
            ],
            "regions": [
                {
                    "id": r.region_id,
                    "name": r.name,
                    "box": [round(v, 1) for v in r.box.as_tuple()],
                    "role": r.role,
                    "char_span": list(r.char_span),
                    "peak_field": float(r.field.max()) if r.field.size else 0.0,
                    "mask_tokens": int((r.mask > 0.0).sum()),
                }
                for r in self.regions
            ],
            "emphases": [
                {
                    "scope_id": e.scope_id,
                    "phrase": e.phrase,
                    "strength": e.strength,
                    "char_span": list(e.char_span),
                }
                for e in self.emphases
            ],
        }


# ── Textbausteine ────────────────────────────────────────────────────────


def _clean(text: str) -> str:
    return text.strip().rstrip(".!? ")


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else text + "."


def _horizontal(percent: float) -> str:
    if percent < 20.0:
        return "far-left side"
    if percent < 40.0:
        return "left side"
    if percent < 60.0:
        return "center"
    if percent < 80.0:
        return "right side"
    return "far-right side"


def _vertical(percent: float) -> str:
    if percent < 20.0:
        return "top"
    if percent < 40.0:
        return "upper portion"
    if percent < 60.0:
        return "middle portion"
    if percent < 80.0:
        return "lower portion"
    return "bottom"


def _framing(height_percent: float) -> str:
    if height_percent >= 70.0:
        return "a large prominent near-frame-height foreground subject"
    if height_percent >= 45.0:
        return "a prominent medium-to-large subject"
    if height_percent >= 25.0:
        return "a medium-size subject"
    return "a small distant subject"


def identity_instruction(trigger: str) -> str:
    return (
        f"{trigger} identifies the person in this region. Generate this person's "
        f"face and facial identity from {trigger}, preserving one coherent person."
    )


def effective_role(region: RegionDefinition, box: PixelBox, canvas_width: int) -> str:
    if region.role != ROLE_AUTO:
        return region.role
    return ROLE_BACKGROUND if box.width >= 0.70 * canvas_width else ROLE_SUBJECT


def region_clause(
    description: str,
    box: PixelBox,
    width: int,
    height: int,
    *,
    role: str,
    subject_fill: bool,
) -> str:
    center_x = 100.0 * (box.x0 + box.x1) / (2.0 * width)
    center_y = 100.0 * (box.y0 + box.y1) / (2.0 * height)
    height_percent = 100.0 * box.height / height

    if role == ROLE_BACKGROUND:
        return (
            f"Across the {_vertical(center_y)} of the image, occupying about "
            f"{height_percent:.0f}% of its height, there is {description}."
        )

    location = f"In the {_vertical(center_y)} {_horizontal(center_x)}"
    base = f"{location}, render {description} as {_framing(height_percent)}."
    if not subject_fill:
        return base
    return (
        f"{base} The visible subject itself should fill most of its assigned image "
        "area with minimal empty margin. Keep the complete visible subject inside "
        "that area without drawing guides, borders, coordinates, labels, text, or "
        "annotations."
    )


def relationship_clause(regions: Sequence[CompiledRegion], height: int) -> str:
    """Explizite Links/Rechts/Vorne-Beziehungen zwischen Subjekten."""
    subjects = [r for r in regions if r.role == ROLE_SUBJECT]
    if len(subjects) < 2:
        return ""

    left_to_right = sorted(subjects, key=lambda r: r.box.center[0])
    names = [r.name for r in left_to_right]
    if len(names) == 2:
        ordering = f"{names[0]} is to the left of {names[1]}"
    else:
        ordering = (
            "From left to right, the subjects are "
            + ", ".join(names[:-1])
            + f", and {names[-1]}"
        )

    lowest = max(subjects, key=lambda r: r.box.center[1])
    others = [r.box.center[1] for r in subjects if r.region_id != lowest.region_id]
    if others and lowest.box.center[1] - sum(others) / len(others) > 0.08 * height:
        ordering += f"; {lowest.name} is positioned below the other subjects"

    # Gleich große Subjekte zusammenfassen. Bei vier Boxen wären das sonst sechs
    # fast gleichlautende Paar-Sätze, die einander im Prompt verwässern — das
    # Modell staffelt die Personen dann doch in die Tiefe.
    equal_pairs = []
    for i, first in enumerate(left_to_right):
        for second in left_to_right[i + 1 :]:
            ratio = first.box.height / second.box.height
            center_delta = abs(first.box.center[1] - second.box.center[1]) / height
            if 0.85 <= ratio <= 1.15 and center_delta <= 0.10:
                equal_pairs.append((first.name, second.name))

    total_pairs = len(subjects) * (len(subjects) - 1) // 2
    if equal_pairs and len(equal_pairs) == total_pairs:
        ordering += (
            "; all subjects are equally large, stand at the same camera distance "
            "and share the same top and bottom levels"
        )
    else:
        for first_name, second_name in equal_pairs[:2]:
            ordering += (
                f"; {first_name} and {second_name} are equally large, at the "
                "same camera distance, with matching top and bottom levels"
            )

    depth = []
    for i, front in enumerate(subjects):
        for behind in subjects[i + 1 :]:
            ow = min(front.box.x1, behind.box.x1) - max(front.box.x0, behind.box.x0)
            oh = min(front.box.y1, behind.box.y1) - max(front.box.y0, behind.box.y0)
            if ow > 0.0 and oh > 0.0:
                depth.append(
                    f"{front.name} appears in front of {behind.name} where their "
                    "target boxes overlap; both occupy the shared image area as "
                    f"distinct subjects, with {behind.name} naturally and partially "
                    f"occluded behind {front.name}"
                )
    return ". ".join([ordering, *depth]) + "."


def _nth_occurrence(text: str, phrase: str, occurrence: int) -> int:
    start = 0
    for index in range(occurrence + 1):
        start = text.find(phrase, start)
        if start < 0:
            raise ValueError(
                f"Phrase {phrase!r} kommt im Prompt-Bereich nicht (oft genug) vor"
            )
        if index < occurrence:
            start += len(phrase)
    return start


def _all_occurrences(text: str, phrase: str) -> tuple[int, ...]:
    offsets: list[int] = []
    start = 0
    while (offset := text.find(phrase, start)) >= 0:
        offsets.append(offset)
        start = offset + len(phrase)
    return tuple(offsets)


# ── Compiler ─────────────────────────────────────────────────────────────


def compile_plan(
    width: int,
    height: int,
    global_prompt: str,
    regions: Sequence[RegionDefinition],
    *,
    strength: float = 1.0,
    outside_penalty: float = 1.0,
    falloff_pixels: float = 128.0,
    subject_competition: bool = True,
    subject_fill: bool = True,
    late_step_scale: float = 0.35,
    emphases: Sequence[EmphasisRequest] = (),
    identity_triggers: dict[str, tuple[str, ...]] | None = None,
    spatial_instructions: bool = True,
) -> CompiledPlan:
    """Baut den vereinheitlichten Prompt und alle räumlichen Felder.

    `spatial_instructions=False` unterdrückt die generierten Ortsangaben — dann
    trägt allein der Attention-Router die räumliche Information.
    """
    if not 0.0 < strength <= 10.0:
        raise ValueError("Spatial strength muss in (0, 10] liegen")
    if not 0.0 <= outside_penalty <= 10.0:
        raise ValueError("Outside penalty muss zwischen 0 und 10 liegen")
    if not 0.0 <= falloff_pixels <= 2048.0:
        raise ValueError("Falloff muss zwischen 0 und 2048 Pixeln liegen")
    if not 0.0 <= late_step_scale <= 1.0:
        raise ValueError("Late-step scale muss zwischen 0 und 1 liegen")

    geometry = CanvasGeometry.resolve(width, height)
    triggers = dict(identity_triggers or {})

    active: list[tuple[RegionDefinition, PixelBox]] = []
    for region in regions:
        if not region.enabled or not region.description:
            continue
        active.append((region, region.box.clipped(width, height)))
    # Höhere Priorität kompiliert zuerst; gleiche Priorität behält Listenreihenfolge.
    active.sort(key=lambda item: -item[0].priority)

    roles = [effective_role(region, box, width) for region, box in active]
    raw_fields = [
        geometry.subject_target_field(
            box, float(falloff_pixels), edge_weight=0.85 if subject_fill else 0.5
        )
        if role == ROLE_SUBJECT
        else geometry.soft_box_field(box, float(falloff_pixels))
        for (_, box), role in zip(active, roles)
    ]
    fields = (
        apply_subject_competition(raw_fields, roles)
        if subject_competition
        else raw_fields
    )

    prompt = _sentence(global_prompt.strip())
    compiled: list[CompiledRegion] = []

    for (region, box), role, region_field in zip(active, roles, fields):
        description = region.description
        if spatial_instructions:
            clause = region_clause(
                description,
                box,
                geometry.aligned_width,
                geometry.aligned_height,
                role=role,
                subject_fill=subject_fill,
            )
        else:
            clause = _sentence(description)

        identity_span_local: tuple[int, int] | None = None
        identity_text = _clean(region.identity_prompt)
        if identity_text:
            offset = clause.find(identity_text)
            if offset < 0:
                raise ValueError(
                    f"Identity-Prompt der Region {region.name!r} nicht in der Klausel gefunden"
                )
            identity_span_local = (offset, offset + len(identity_text))

        trigger_spans_local: list[tuple[int, int]] = []
        for trigger in dict.fromkeys(triggers.get(region.region_id, ())):
            trigger = trigger.strip()
            if not trigger:
                raise ValueError("Character-Identity-Trigger darf nicht leer sein")
            instruction = identity_instruction(trigger)
            instruction_start = len(clause) + 1
            clause = f"{clause} {instruction}"
            for offset in _all_occurrences(instruction, trigger):
                trigger_spans_local.append(
                    (
                        instruction_start + offset,
                        instruction_start + offset + len(trigger),
                    )
                )

        if prompt:
            prompt += "\n"
        start = len(prompt)
        prompt += clause
        end = len(prompt)

        compiled.append(
            CompiledRegion(
                region_id=region.region_id,
                name=region.name,
                box=box,
                role=role,
                clause=clause,
                char_span=(start, end),
                field=region_field,
                mask=geometry.rasterize_box(box),
                prompt=region.prompt.strip(),
                identity_prompt=region.identity_prompt.strip(),
                negative_prompt=region.negative_prompt.strip(),
                identity_char_span=(
                    (start + identity_span_local[0], start + identity_span_local[1])
                    if identity_span_local
                    else None
                ),
                trigger_char_spans=tuple(
                    (start + a, start + b) for a, b in trigger_spans_local
                ),
            )
        )

    if spatial_instructions:
        relationship = relationship_clause(compiled, geometry.aligned_height)
        if relationship:
            prompt += f"\n{relationship}"

    resolved_emphases = _resolve_emphases(
        prompt, compiled, tuple(emphases), geometry.token_count
    )

    return CompiledPlan(
        geometry=geometry,
        prompt=prompt,
        regions=tuple(compiled),
        emphases=resolved_emphases,
        strength=float(strength),
        outside_penalty=float(outside_penalty),
        falloff_pixels=float(falloff_pixels),
        subject_competition=bool(subject_competition),
        subject_fill=bool(subject_fill),
        late_step_scale=float(late_step_scale),
        identity_triggers=triggers,
    )


def _resolve_emphases(
    prompt: str,
    regions: list[CompiledRegion],
    emphases: tuple[EmphasisRequest, ...],
    token_count: int,
) -> tuple[CompiledEmphasis, ...]:
    if not emphases:
        return ()
    by_id = {r.region_id: r for r in regions}
    global_end = regions[0].char_span[0] if regions else len(prompt)
    resolved: list[CompiledEmphasis] = []
    for emphasis in emphases:
        if emphasis.scope_id == GLOBAL_SCOPE:
            start = _nth_occurrence(
                prompt[:global_end], emphasis.phrase, emphasis.occurrence
            )
            emphasis_field = np.ones(token_count, dtype=np.float32)
        else:
            region = by_id.get(emphasis.scope_id)
            if region is None:
                raise ValueError(
                    f"Emphasis verweist auf Region ohne aktiven Prompt: {emphasis.scope_id}"
                )
            # Die Phrase wird im Original-Regionsprompt gesucht und ihr Offset in
            # die kompilierte Klausel übertragen.
            source_offset = _nth_occurrence(
                region.prompt, emphasis.phrase, emphasis.occurrence
            )
            description = _clean(region.prompt)
            description_offset = region.clause.find(description)
            if description_offset < 0:
                raise ValueError(
                    f"Emphasis-Phrase {emphasis.phrase!r} nicht in der kompilierten "
                    "Regionsklausel auffindbar"
                )
            leading = len(region.prompt) - len(region.prompt.lstrip())
            start = (
                region.char_span[0] + description_offset + source_offset - leading
            )
            emphasis_field = region.field
        if prompt[start : start + len(emphasis.phrase)] != emphasis.phrase:
            raise ValueError(
                f"Emphasis-Phrase {emphasis.phrase!r} konnte im kompilierten Prompt "
                "nicht exakt lokalisiert werden"
            )
        resolved.append(
            CompiledEmphasis(
                scope_id=emphasis.scope_id,
                phrase=emphasis.phrase,
                strength=float(emphasis.strength),
                char_span=(start, start + len(emphasis.phrase)),
                field=emphasis_field,
            )
        )
    return tuple(resolved)


__all__ = [
    "BACKEND",
    "GLOBAL_SCOPE",
    "ROLES",
    "ROLE_AUTO",
    "ROLE_BACKGROUND",
    "ROLE_SUBJECT",
    "CompiledEmphasis",
    "CompiledPlan",
    "CompiledRegion",
    "EmphasisRequest",
    "RegionDefinition",
    "compile_plan",
    "effective_role",
    "identity_instruction",
    "region_clause",
    "relationship_clause",
]
