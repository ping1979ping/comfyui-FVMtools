"""MOSS-Image layout prompts from detected boxes and a JB prompt.

MOSS-Image reads a JSON layout: a ``summary`` (scene caption, meta tags,
canvas kind) plus ``items``, each with a ``label`` (character / object /
text), a ``caption`` and a ``bbox``. Boxes are in pixels of the render
canvas, measured from the canvas centre, ordered ``[ymin, xmin, ymax, xmax]``
(``bbox_coord_space: abs_centralized``, ``bbox_format: yxyx``).

This module is pure: it takes boxes that someone already detected (SAM3 in
:mod:`nodes.jb.moss_layout`) and turns them into that JSON.

Box rules, one per line::

    sam3 prompt | label | slot
    person      | character | character_{n}
    table       | object    | table

``{n}`` numbers the hits of one rule left to right, so two people become
``character_1`` and ``character_2``. The slot names a key in the JB prompt;
its sub-tree becomes the box caption. A box whose slot is missing from the
prompt is captioned with its SAM3 prompt.

The JB prompt is the ``raw_json`` of a Stitcher. A single top-level key
that is no slot (the Stitcher's title, e.g. ``scene``) is unwrapped first.
Everything a box does not claim — usually the location — goes into the
summary caption.
"""

from __future__ import annotations

import json
import math

try:
    from .sentences import emit_sentences
    from .serialize import parse_input
except ImportError:  # pragma: no cover
    from core.jb.sentences import emit_sentences
    from core.jb.serialize import parse_input


LABELS = ("character", "object", "text")
CANVAS_KINDS = ("single", "multi_panel", "collage")


def parse_box_rules(text: str) -> list[dict]:
    """``prompt | label | slot`` lines → rule dicts. ``#`` starts a comment."""
    rules = []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        prompt = parts[0]
        if not prompt:
            continue
        label = parts[1].lower() if len(parts) > 1 and parts[1] else "object"
        if label not in LABELS:
            label = "object"
        slot = parts[2] if len(parts) > 2 else ""
        rules.append({"prompt": prompt, "label": label, "slot": slot})
    return rules


def box_iou(a, b) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def select_hits(hits, max_hits: int, min_area: float, src_w: int, src_h: int,
                iou: float = 0.6) -> list[dict]:
    """Best-first NMS of one rule's hits, then size floor and count cap.

    ``hits`` are dicts with ``bbox`` ``[x1, y1, x2, y2]`` (source pixels) and
    ``score``. ``min_area`` is a share of the source image.
    """
    floor = min_area * src_w * src_h
    kept = []
    for h in sorted(hits, key=lambda h: h["score"], reverse=True):
        x1, y1, x2, y2 = h["bbox"]
        if (x2 - x1) * (y2 - y1) < floor:
            continue
        if any(box_iou(h["bbox"], k["bbox"]) >= iou for k in kept):
            continue
        kept.append(h)
        if len(kept) >= max_hits:
            break
    return kept


def number_slots(rule: dict, hits: list[dict]) -> list[dict]:
    """Order a rule's hits left to right and give each its slot name."""
    ordered = sorted(hits, key=lambda h: (h["bbox"][0] + h["bbox"][2]) / 2)
    out = []
    for n, h in enumerate(ordered, 1):
        slot = rule["slot"].replace("{n}", str(n)) if rule["slot"] else ""
        out.append({**h, "prompt": rule["prompt"], "label": rule["label"], "slot": slot})
    return out


def fit_canvas(src_w: int, src_h: int, megapixels: float, multiple: int = 16):
    """Source aspect at ``megapixels`` (ComfyUI's megapixel: 1024², 0 keeps
    the source size), snapped down to ``multiple``."""
    if megapixels > 0:
        scale = math.sqrt(megapixels * 1024 * 1024 / (src_w * src_h))
    else:
        scale = 1.0
    return scale_canvas(src_w, src_h, scale, multiple)


def scale_canvas(width: int, height: int, factor: float, multiple: int = 16):
    """``width``/``height`` times ``factor``, snapped down to ``multiple``."""
    w = max(multiple, int(width * factor) // multiple * multiple)
    h = max(multiple, int(height * factor) // multiple * multiple)
    return w, h


def centered_yxyx(bbox, src_w: int, src_h: int, width: int, height: int) -> list[int]:
    """Source ``[x1, y1, x2, y2]`` → MOSS ``abs_centralized`` yxyx on the canvas.

    Same rounding as MOSS's ``compile_prompt.py``: floor top/left, ceil
    bottom/right, then subtract the canvas centre.
    """
    sx, sy = width / src_w, height / src_h
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1, x2 = sorted((max(0.0, x1 * sx), min(width, x2 * sx)))
    y1, y2 = sorted((max(0.0, y1 * sy), min(height, y2 * sy)))
    return [
        math.floor(y1) - height // 2,
        math.floor(x1) - width // 2,
        math.ceil(y2) - height // 2,
        math.ceil(x2) - width // 2,
    ]


def split_prompt(prompt: str, slots) -> tuple[dict, dict | str]:
    """JB prompt → (slot sub-trees, everything else).

    "Everything else" is a dict for a JSON prompt and the prompt itself for
    prose: a non-JSON prompt has no slot content and is all summary.
    """
    text = (prompt or "").strip()
    if not text:
        return {}, ""
    try:
        doc = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        doc = parse_input(text) if text.startswith("{") else None
    if not isinstance(doc, dict):
        return {}, text

    slots = set(slots)
    if len(doc) == 1:
        (only_key, only_val), = doc.items()
        if only_key not in slots and isinstance(only_val, dict):
            doc = only_val

    claimed = {k: v for k, v in doc.items() if k in slots}
    rest = {k: v for k, v in doc.items() if k not in slots}
    return claimed, rest


def _caption(value) -> str:
    if isinstance(value, str):
        return value.strip()
    return emit_sentences(value)


def build_layout(boxes: list[dict], src_w: int, src_h: int, width: int, height: int,
                 prompt: str = "", high_level_description: str = "",
                 meta: str = "", canvas: str = "single") -> tuple[dict, dict]:
    """Assemble the MOSS layout dict. Returns ``(layout, report)``.

    ``boxes`` come from :func:`number_slots`: ``bbox`` (source pixels),
    ``label``, ``slot``, ``prompt``.
    """
    slots = {b["slot"] for b in boxes if b.get("slot")}
    claimed, rest = split_prompt(prompt, slots)
    rest_prose = rest if isinstance(rest, str) else emit_sentences(rest)

    parts = [p.strip() for p in (high_level_description, rest_prose) if p and p.strip()]
    summary = {"caption": " ".join(parts)}
    tags = [t.strip() for t in (meta or "").split(",") if t.strip()]
    if tags:
        summary["meta"] = tags
    summary["canvas"] = canvas if canvas in CANVAS_KINDS else "single"

    order = {label: i for i, label in enumerate(LABELS)}
    items, filled, fallback = [], [], []
    for b in sorted(boxes, key=lambda b: (order[b["label"]], (b["bbox"][0] + b["bbox"][2]) / 2)):
        slot = b.get("slot", "")
        if slot in claimed:
            caption = _caption(claimed[slot])
            filled.append(slot)
        else:
            caption = ""
        if not caption:
            caption = b["prompt"]
            fallback.append(slot or b["prompt"])
        item = {"label": b["label"], "caption": caption}
        if b["label"] == "text":
            item["text"] = b.get("text", "")
        item["bbox"] = centered_yxyx(b["bbox"], src_w, src_h, width, height)
        items.append(item)

    layout = {"summary": summary, "items": items,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    report = {"filled": filled, "fallback": fallback,
              "summary_keys": sorted(rest) if isinstance(rest, dict) else []}
    return layout, report


def dump_layout(layout: dict) -> str:
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))
