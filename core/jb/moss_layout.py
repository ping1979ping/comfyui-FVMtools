"""MOSS-Image layout prompts from detected boxes and a JB prompt.

MOSS-Image reads a JSON layout: a ``summary`` (scene caption, meta tags,
canvas kind) plus ``items``, each with a ``label`` (character / object /
text), a ``caption`` and a ``bbox``. Boxes are in pixels of the render
canvas, measured from the canvas centre, ordered ``[ymin, xmin, ymax, xmax]``
(``bbox_coord_space: abs_centralized``, ``bbox_format: yxyx``).

This module is pure: it takes boxes that someone already detected (SAM3 in
:mod:`nodes.jb.moss_layout`) and turns them into that JSON.

Box rules, one per line::

    sam3 prompt | label | slot          | caption (optional)
    person      | character | character_{n}
    table       | object    | table
    plant       | object    | plant_{n}   | a tall potted fern

``{n}`` numbers the hits of one rule left to right, so two people become
``character_1`` and ``character_2``. The slot names a key in the JB prompt;
its sub-tree becomes the box caption. A caption in the rule wins over the
slot. A box with neither is captioned with its SAM3 prompt.

Rules can also be suggested from a vocabulary of SAM3 nouns
(:func:`auto_rule`, :func:`merge_rules`, :func:`rules_text`), and a boxes
document converts to an Ideogram-4 caption (:func:`to_ideogram`).

The JB prompt is the ``raw_json`` of a Stitcher. A single top-level key
that is no slot (the Stitcher's title, e.g. ``scene``) is unwrapped first.
Everything a box does not claim — usually the location — goes into the
summary caption.
"""

from __future__ import annotations

import json
import math
import re

try:
    from .sentences import emit_sentences
    from .serialize import parse_input
except ImportError:  # pragma: no cover
    from core.jb.sentences import emit_sentences
    from core.jb.serialize import parse_input


LABELS = ("character", "object", "text")
CANVAS_KINDS = ("single", "multi_panel", "collage")

# Common photo nouns SAM3 grounds well — the default auto-rules vocabulary.
DEFAULT_VOCABULARY = (
    "person", "table", "chair", "stool", "bench", "sofa", "bed", "desk", "shelf",
    "counter", "cabinet", "plant", "flower vase", "window", "door", "mirror",
    "painting", "poster", "sign", "lamp", "clock", "tv", "laptop", "phone", "book",
    "cup", "bottle", "glass", "plate", "bowl", "bag", "umbrella", "car", "bicycle",
    "motorcycle", "dog", "cat", "horse", "bird", "tree",
)


def parse_box_rules(text: str) -> list[dict]:
    """``prompt | label | slot | caption`` lines → rule dicts.

    ``#`` starts a comment. Missing columns default to label ``object``, no
    slot and no caption; a caption may itself contain ``|``.
    """
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
        caption = " | ".join(p for p in parts[3:] if p)
        rules.append({"prompt": prompt, "label": label, "slot": slot, "caption": caption})
    return rules


def parse_vocabulary(text: str) -> list[str]:
    """Vocabulary text (one word per line or comma separated, ``#`` comments)
    → unique words in order, compared case-insensitively."""
    words, seen = [], set()
    for raw in (text or "").splitlines():
        for word in raw.split("#", 1)[0].split(","):
            word = " ".join(word.split())
            if word and word.lower() not in seen:
                seen.add(word.lower())
                words.append(word)
    return words


def slot_name(word: str) -> str:
    """A SAM3 word → slot stem: lowercase, runs of non-word chars → ``_``."""
    return re.sub(r"[^0-9a-z]+", "_", word.lower()).strip("_") or "object"


def auto_rule(word: str) -> dict:
    """The rule a vocabulary hit becomes: ``person`` is a character, every
    other word an object numbered by its own slot stem."""
    if word.lower() == "person":
        return {"prompt": word, "label": "character", "slot": "character_{n}", "caption": ""}
    return {"prompt": word, "label": "object", "slot": slot_name(word) + "_{n}", "caption": ""}


def merge_rules(manual: list[dict], words) -> tuple[list[dict], list[str]]:
    """Manual rules first, then an :func:`auto_rule` for every word no
    manual rule already grounds. Returns ``(rules, auto_words)``."""
    covered = {r["prompt"].lower() for r in manual}
    extra = [w for w in words if w.lower() not in covered]
    return list(manual) + [auto_rule(w) for w in extra], extra


def rules_text(rules: list[dict]) -> str:
    """Rule dicts → ``box_rules`` text, one line each (inverse of
    :func:`parse_box_rules`)."""
    lines = []
    for r in rules:
        cols = [r["prompt"], r.get("label") or "object", r.get("slot", "")]
        if r.get("caption"):
            cols.append(r["caption"])
        lines.append(" | ".join(cols))
    return "\n".join(lines) + ("\n" if lines else "")


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


def pick_auto_hits(found: list[tuple[dict, list[dict]]], taken: list[dict],
                   max_boxes: int, iou: float = 0.5) -> list[tuple[dict, list[dict]]]:
    """Vocabulary hits across all words, best score first: a hit whose box
    overlaps an already kept box (manual or an earlier word) by ``iou`` is the
    same thing under another name and dropped; at most ``max_boxes`` survive.

    ``found`` is ``[(rule, hits)]`` per word (hits already through
    :func:`select_hits`), ``taken`` the manual rules' hits. Returns
    ``[(rule, kept hits)]`` for the words that keep at least one hit, in
    vocabulary order.
    """
    pool = sorted(((h["score"], i, h) for i, (_r, hits) in enumerate(found) for h in hits),
                  key=lambda t: -t[0])
    kept_boxes = [t["bbox"] for t in taken]
    kept: dict[int, list[dict]] = {}
    for _score, i, h in pool:
        if sum(len(v) for v in kept.values()) >= max_boxes:
            break
        if any(box_iou(h["bbox"], b) >= iou for b in kept_boxes):
            continue
        kept_boxes.append(h["bbox"])
        kept.setdefault(i, []).append(h)
    return [(found[i][0], kept[i]) for i in sorted(kept)]


def number_slots(rule: dict, hits: list[dict]) -> list[dict]:
    """Order a rule's hits left to right and give each its slot name.

    A rule ``caption`` (``{n}`` numbered as in the slot) is copied onto each
    box, where :func:`make_boxes_doc` lets it win over the JB slot.
    """
    ordered = sorted(hits, key=lambda h: (h["bbox"][0] + h["bbox"][2]) / 2)
    out = []
    for n, h in enumerate(ordered, 1):
        slot = rule["slot"].replace("{n}", str(n)) if rule["slot"] else ""
        box = {**h, "prompt": rule["prompt"], "label": rule["label"], "slot": slot}
        if rule.get("caption"):
            box["caption"] = rule["caption"].replace("{n}", str(n))
        out.append(box)
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


def make_boxes_doc(boxes: list[dict], src_w: int, src_h: int, width: int, height: int,
                   up_width: int = 0, up_height: int = 0, prompt: str = "",
                   high_level_description: str = "", meta: str = "",
                   canvas: str = "single") -> tuple[dict, dict]:
    """Detected boxes + JB prompt → the editable ``MOSS_BOXES`` document.

    ``boxes`` come from :func:`number_slots`: ``bbox`` (source pixels),
    ``label``, ``slot``, ``prompt`` and ``score``; a ``caption`` already on
    a box wins over its JB slot. Returns ``(doc, report)``.

    The document is what the layout nodes pass along (socket ``MOSS_BOXES``)::

        {"version": 1,
         "src_w", "src_h",            # reference image, for previews
         "width", "height",           # first-pass canvas
         "up_width", "up_height",     # refine canvas
         "summary": {"caption", "meta", "canvas"},
         "boxes": [{"slot", "label", "caption", "text", "prompt",
                    "rect": [x1, y1, x2, y2],    # 0..1 of the canvas
                    "score", "source"}]}         # source: sam3 / user

    ``rect`` is relative, so one document renders on any canvas of the same
    aspect (:func:`render_moss`). Boxes are ordered character → object →
    text, then left to right — MOSS's item order.
    """
    slots = {b["slot"] for b in boxes if b.get("slot")}
    claimed, rest = split_prompt(prompt, slots)
    rest_prose = rest if isinstance(rest, str) else emit_sentences(rest)

    parts = [p.strip() for p in (high_level_description, rest_prose) if p and p.strip()]
    summary = {"caption": " ".join(parts)}
    summary["meta"] = [t.strip() for t in (meta or "").split(",") if t.strip()]
    summary["canvas"] = canvas if canvas in CANVAS_KINDS else "single"

    order = {label: i for i, label in enumerate(LABELS)}
    out, filled, fallback = [], [], []
    for b in sorted(boxes, key=lambda b: (order[b["label"]], (b["bbox"][0] + b["bbox"][2]) / 2)):
        slot = b.get("slot", "")
        caption = (b.get("caption") or "").strip()
        if caption:
            filled.append(slot or b["prompt"])
        elif slot in claimed:
            caption = _caption(claimed[slot])
            if caption:
                filled.append(slot)
        if not caption:
            caption = b["prompt"]
            fallback.append(slot or b["prompt"])
        x1, y1, x2, y2 = (float(v) for v in b["bbox"])
        rect = [min(x1, x2) / src_w, min(y1, y2) / src_h, max(x1, x2) / src_w, max(y1, y2) / src_h]
        out.append({
            "slot": slot, "label": b["label"], "caption": caption, "text": b.get("text", ""),
            "prompt": b.get("prompt", ""),
            "rect": [round(min(1.0, max(0.0, v)), 5) for v in rect],
            "score": round(float(b.get("score", 1.0)), 4), "source": b.get("source", "sam3"),
        })

    doc = {"version": 1, "src_w": int(src_w), "src_h": int(src_h),
           "width": int(width), "height": int(height),
           "up_width": int(up_width or width), "up_height": int(up_height or height),
           "summary": summary, "boxes": out}
    report = {"filled": filled, "fallback": fallback,
              "summary_keys": sorted(rest) if isinstance(rest, dict) else []}
    return doc, report


def render_moss(doc: dict, width: int | None = None, height: int | None = None) -> dict:
    """``MOSS_BOXES`` document → MOSS layout dict on a ``width``×``height``
    canvas (default: the document's first-pass canvas)."""
    width = int(width or doc["width"])
    height = int(height or doc["height"])
    src = doc.get("summary") or {}
    summary = {"caption": src.get("caption", "")}
    if src.get("meta"):
        summary["meta"] = list(src["meta"])
    summary["canvas"] = src.get("canvas") if src.get("canvas") in CANVAS_KINDS else "single"

    items = []
    for b in doc.get("boxes", []):
        label = b.get("label") if b.get("label") in LABELS else "object"
        item = {"label": label, "caption": b.get("caption") or b.get("prompt") or label}
        if label == "text":
            item["text"] = b.get("text", "")
        item["bbox"] = centered_yxyx(b["rect"], 1, 1, width, height)
        items.append(item)
    return {"summary": summary, "items": items,
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}


def to_ideogram(doc: dict, high_level_description: str = "") -> dict:
    """``MOSS_BOXES`` document → Ideogram-4 caption dict, the format KJNodes'
    *Ideogram 4 Prompt Builder* imports and :class:`FVM_Ideogram_Assembler`
    fills.

    ``bbox`` is ``[ymin, xmin, ymax, xmax]`` on the 0–1000 grid. A box with a
    slot gets the slot key as ``desc`` (the Assembler refills it from the JB
    prompt); a box without one gets its caption, prefixed ``character:`` for
    a character so :mod:`core.jb.moss_ideogram` reads it back as one. Text
    boxes are ``type:
    "text"`` with a ``text`` field. The summary caption is the background;
    ``high_level_description`` (default: the summary caption) opens the
    caption and is cut from the front of the background when it repeats it.
    """
    caption = ((doc.get("summary") or {}).get("caption") or "").strip()
    high = (high_level_description or "").strip()
    background = caption
    if high and caption.startswith(high) and caption[len(high):].strip():
        background = caption[len(high):].strip()

    elements = []
    for b in doc.get("boxes", []):
        x1, y1, x2, y2 = b["rect"]
        bbox = [max(0, min(1000, round(v * 1000))) for v in (y1, x1, y2, x2)]
        is_text = b.get("label") == "text"
        el = {"type": "text" if is_text else "obj", "bbox": bbox}  # KJ's key order
        if is_text:
            el["text"] = b.get("text", "")
        desc = b.get("slot") or b.get("caption") or b.get("prompt") or ""
        if not b.get("slot") and b.get("label") == "character":
            desc = "character: " + desc      # keeps the label for core.jb.moss_ideogram
        el["desc"] = desc
        elements.append(el)
    return {"high_level_description": high or caption,
            "compositional_deconstruction": {"background": background, "elements": elements}}


def rect_pixels(rect, width: int, height: int) -> tuple[int, int, int, int]:
    """Relative ``rect`` → integer ``(x1, y1, x2, y2)`` pixels, clamped."""
    x1, y1, x2, y2 = rect
    return (max(0, math.floor(x1 * width)), max(0, math.floor(y1 * height)),
            min(width, math.ceil(x2 * width)), min(height, math.ceil(y2 * height)))


def build_layout(boxes: list[dict], src_w: int, src_h: int, width: int, height: int,
                 prompt: str = "", high_level_description: str = "",
                 meta: str = "", canvas: str = "single") -> tuple[dict, dict]:
    """Assemble the MOSS layout dict. Returns ``(layout, report)``.

    Shortcut for :func:`make_boxes_doc` + :func:`render_moss`.
    """
    doc, report = make_boxes_doc(boxes, src_w, src_h, width, height, prompt=prompt,
                                 high_level_description=high_level_description,
                                 meta=meta, canvas=canvas)
    return render_moss(doc), report


def dump_layout(layout: dict) -> str:
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))
