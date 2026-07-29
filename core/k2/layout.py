"""K2 Lab — Layout-Format des Region Builders.

Der visuelle Editor lebt im Frontend (``web/js/fvm_k2_builder.js``); hier wird
sein Layout-JSON in die üblichen K2-Objekte übersetzt.

Boxen werden **normalisiert** (0..1) gespeichert, nicht in Pixeln. Zusätzlich
merkt sich das Layout die Leinwand, auf der es entstanden ist — nur damit lässt
sich beim Wechsel des Seitenverhältnisses die *Form* einer Box erhalten statt
sie nicht-affin mitzuquetschen (siehe ``rescale_layout``).
"""

from __future__ import annotations

import json

from .geometry import PixelBox
from .lora import ROUTING_MODES, STANDARD_ROUTING, LoraSpec
from .prompt import ROLES, RegionDefinition

LAYOUT_VERSION = 1

DEFAULT_LAYOUT = {
    "version": LAYOUT_VERSION,
    "canvas": {"width": 1024, "height": 1024},
    "boxes": [],
}

# Feste Farbreihenfolge, damit Node-Vorschau, Editorfenster und Region-Preview
# dieselbe Region gleich einfärben.
BOX_COLORS = (
    "#e0564f", "#4f8fe0", "#5fc27e", "#e0b64f",
    "#a765e0", "#4fd0d0", "#e07fb4", "#8ea34f",
)


def default_layout_json() -> str:
    return json.dumps(DEFAULT_LAYOUT, indent=2)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _read_rect(raw: dict) -> tuple[float, float, float, float]:
    x = float(raw.get("x", 0.0))
    y = float(raw.get("y", 0.0))
    w = float(raw.get("w", raw.get("width", 0.0)))
    h = float(raw.get("h", raw.get("height", 0.0)))
    return x, y, w, h


def rescale_rect(
    rect: tuple[float, float, float, float],
    old_size: tuple[float, float],
    new_size: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Rechnet eine normalisierte Box auf ein neues Seitenverhältnis um.

    Erhalten bleibt die *Pixelform* der Box (ihr Breite-zu-Höhe-Verhältnis) und
    die relative Lage ihres Mittelpunkts. Passt sie nicht mehr in die neue
    Leinwand, wird sie gleichmäßig verkleinert — nie einseitig gestaucht.

    Ohne diesen Schritt bleibt eine 0.3×0.3-Box beim Wechsel von 1:1 auf 16:9
    zwar rechnerisch „gleich", wird real aber zu einem breiten Rechteck. Genau
    daran scheitern pixel- bzw. fraction-basierte Box-Editoren.
    """
    x, y, w, h = rect
    old_w, old_h = float(old_size[0]), float(old_size[1])
    new_w, new_h = float(new_size[0]), float(new_size[1])
    if min(old_w, old_h, new_w, new_h) <= 0 or w <= 0 or h <= 0:
        return rect

    # Nur die *Seitenverhältnis*-Änderung kompensieren, nicht die Größenänderung:
    # 512→1024 ist reine Skalierung und muss die Box unverändert lassen.
    ratio = (new_w / new_h) / (old_w / old_h)
    if ratio == 1.0:
        return rect

    # Symmetrisch auf beide Achsen aufgeteilt (sqrt). Damit ist die Umrechnung
    # umkehrbar — hin und zurück landet exakt wieder beim Ausgangsrechteck,
    # statt die Box bei jedem Formatwechsel weiter schrumpfen zu lassen.
    root = ratio**0.5
    new_frac_w = w / root
    new_frac_h = h * root

    # Passt sie danach nicht mehr auf die Leinwand, gleichmäßig verkleinern —
    # niemals einseitig stauchen, sonst wäre die Form doch wieder dahin.
    contain = min(
        1.0,
        1.0 / new_frac_w if new_frac_w > 1.0 else 1.0,
        1.0 / new_frac_h if new_frac_h > 1.0 else 1.0,
    )
    new_frac_w *= contain
    new_frac_h *= contain

    centre_x = x + w / 2.0
    centre_y = y + h / 2.0
    new_x = centre_x - new_frac_w / 2.0
    new_y = centre_y - new_frac_h / 2.0
    # In die Leinwand schieben statt beschneiden — die Form bleibt so erhalten.
    new_x = min(max(new_x, 0.0), max(0.0, 1.0 - new_frac_w))
    new_y = min(max(new_y, 0.0), max(0.0, 1.0 - new_frac_h))
    return (new_x, new_y, new_frac_w, new_frac_h)


def rescale_layout(layout_json: str, new_width: int, new_height: int) -> str:
    """Ganzes Layout auf eine neue Leinwand umrechnen (Frontend-Hilfe)."""
    data = json.loads(layout_json or "{}")
    canvas = data.get("canvas") or {}
    old = (float(canvas.get("width", new_width)), float(canvas.get("height", new_height)))
    for box in data.get("boxes", []):
        rect = box.get("rect") or box
        x, y, w, h = rescale_rect(_read_rect(rect), old, (new_width, new_height))
        box["rect"] = {"x": x, "y": y, "w": w, "h": h}
    data["canvas"] = {"width": int(new_width), "height": int(new_height)}
    return json.dumps(data, indent=2)


def parse_layout(layout_json: str, width: int, height: int):
    """Layout-JSON → (RegionDefinition-Liste, LoraSpec-Liste, Diagnose)."""
    import folder_paths

    try:
        data = json.loads(layout_json or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(
            f"K2 Region Builder: layout is not valid JSON — {error}"
        ) from error
    if not isinstance(data, dict):
        raise ValueError("K2 Region Builder: layout must be a JSON object")

    version = int(data.get("version", LAYOUT_VERSION))
    if version > LAYOUT_VERSION:
        raise ValueError(
            f"K2 Region Builder: layout version {version} is newer than supported "
            f"version {LAYOUT_VERSION}"
        )

    canvas = data.get("canvas") or {}
    stored_width = float(canvas.get("width", width) or width)
    stored_height = float(canvas.get("height", height) or height)

    boxes = data.get("boxes")
    if boxes is None:
        boxes = data.get("regions", [])
    if not isinstance(boxes, list):
        raise ValueError("K2 Region Builder: 'boxes' must be a JSON array")

    regions: list[RegionDefinition] = []
    loras: list[LoraSpec] = []
    notes: list[str] = []
    available = set(folder_paths.get_filename_list("loras"))
    used_names: set[str] = set()

    for index, item in enumerate(boxes):
        if not isinstance(item, dict):
            raise ValueError(f"K2 Region Builder: box #{index + 1} is not an object")
        if not bool(item.get("enabled", True)):
            continue

        box_id = str(item.get("id") or f"box-{index + 1}")
        name = str(item.get("name") or f"Box {index + 1}").strip()
        if name.casefold() in used_names:
            raise ValueError(
                f"K2 Region Builder: region name {name!r} is used twice — names appear "
                "in the generated spatial instructions and must be unique."
            )
        used_names.add(name.casefold())

        x, y, w, h = _read_rect(item.get("rect") or item)
        if max(x, y, w, h) > 1.5:
            # Pixelangaben eines älteren Layouts über die gespeicherte Leinwand
            # normalisieren, sonst sitzt die Box im neuen Format falsch.
            x /= stored_width
            y /= stored_height
            w /= stored_width
            h /= stored_height
            notes.append(f"box {name!r}: converted pixel rect via stored canvas")

        x = _clamp01(x)
        y = _clamp01(y)
        w = _clamp01(w)
        h = _clamp01(h)
        if w <= 0.0 or h <= 0.0:
            raise ValueError(f"K2 Region Builder: box {name!r} has zero width or height")
        x1 = _clamp01(x + w)
        y1 = _clamp01(y + h)

        pixel_box = PixelBox(x * width, y * height, x1 * width, y1 * height).clipped(
            int(width), int(height)
        )

        role = str(item.get("role", "auto"))
        if role not in ROLES:
            notes.append(f"box {name!r}: unknown role {role!r} → auto")
            role = "auto"

        regions.append(
            RegionDefinition(
                region_id=box_id,
                name=name,
                box=pixel_box,
                prompt=str(item.get("prompt", "")),
                identity_prompt=str(item.get("identity_prompt", "")),
                negative_prompt=str(item.get("negative_prompt", "")),
                enabled=True,
                priority=int(item.get("priority", 100 - index)),
                role=role,
            )
        )

        for slot, entry in enumerate(item.get("loras") or []):
            if not isinstance(entry, dict) or not bool(entry.get("enabled", True)):
                continue
            lora_name = str(entry.get("name") or "").strip()
            if not lora_name or lora_name == "None":
                continue
            if lora_name not in available:
                raise ValueError(
                    f"K2 Region Builder: box {name!r} references a LoRA that is not "
                    f"installed: {lora_name}"
                )
            routing = str(entry.get("routing", STANDARD_ROUTING))
            if routing not in ROUTING_MODES:
                notes.append(
                    f"box {name!r}: unknown routing {routing!r} → {STANDARD_ROUTING}"
                )
                routing = STANDARD_ROUTING
            loras.append(
                LoraSpec(
                    lora_id=f"{box_id}-lora-{slot + 1}",
                    lora_name=lora_name,
                    strength=float(entry.get("strength", 1.0)),
                    global_scope=False,
                    region_ids=(box_id,),
                    routing_mode=routing,
                    trigger_phrase=str(entry.get("trigger", "")),
                    display_name=str(entry.get("display_name", "")),
                )
            )

    for slot, entry in enumerate(data.get("global_loras") or []):
        if not isinstance(entry, dict) or not bool(entry.get("enabled", True)):
            continue
        lora_name = str(entry.get("name") or "").strip()
        if not lora_name or lora_name == "None":
            continue
        if lora_name not in available:
            raise ValueError(
                f"K2 Region Builder: global LoRA is not installed: {lora_name}"
            )
        loras.append(
            LoraSpec(
                lora_id=f"global-lora-{slot + 1}",
                lora_name=lora_name,
                strength=float(entry.get("strength", 1.0)),
                global_scope=True,
                region_ids=(),
                routing_mode=STANDARD_ROUTING,
                trigger_phrase=str(entry.get("trigger", "")),
                display_name=str(entry.get("display_name", "")),
            )
        )

    return regions, loras, notes


__all__ = [
    "BOX_COLORS",
    "DEFAULT_LAYOUT",
    "LAYOUT_VERSION",
    "default_layout_json",
    "parse_layout",
    "rescale_layout",
    "rescale_rect",
]
