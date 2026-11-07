"""MOSS-Image layout prompts from an Ideogram-4 caption.

KJNodes' *Ideogram 4 Prompt Builder* is a canvas: draw boxes, type a text
into each box's ``desc``. Its output is an Ideogram caption::

    {"high_level_description"?, "style_description"?,
     "compositional_deconstruction": {
        "background": "...",
        "elements": [{"type": "obj" | "text", "bbox": [ymin, xmin, ymax, xmax],
                      "text"?: "...", "desc": "...", "color_palette"?: [...]}]}}

``bbox`` sits on a 0-1000 grid in ``[ymin, xmin, ymax, xmax]`` order unless
the builder was switched to absolute pixels and/or xy order; the caption
does not say which, so the caller names the space (:data:`BBOX_SPACES`).

Each placed element becomes one box of the ``MOSS_BOXES`` document of
:func:`core.jb.moss_layout.make_boxes_doc`. Its caption is resolved in
this order:

1. ``desc`` is exactly a key of the JB prompt (``@`` tolerated, after the
   Stitcher-title unwrap) → slot = desc, caption from the JB sub-tree.
2. ``desc`` starts with ``character:`` → label character, caption = rest.
3. otherwise the desc itself is the caption. A desc that looks like a key
   (one word, e.g. ``character_3``) still becomes the box's slot, so later
   nodes see the name.

Label: type ``text`` → text; a slot starting with ``character`` or
``person`` → character; anything else → object. Elements without a usable
bbox are skipped and reported.

The scene caption is: high level description, then the caption's
background, then every JB key no box claims — each text once
(:func:`join_unique`), because a caption from the SAM3 layout node already
carries description and JB rest in its background.
"""

from __future__ import annotations

import json
import re

try:
    from .moss_layout import make_boxes_doc, split_prompt
    from .sentences import emit_sentences
    from .serialize import parse_input
except ImportError:  # pragma: no cover
    from core.jb.moss_layout import make_boxes_doc, split_prompt
    from core.jb.sentences import emit_sentences
    from core.jb.serialize import parse_input


BBOX_SPACES = ("normalized yx", "normalized xy", "pixels yx", "pixels xy")
_GRID = 1000.0
_KEY_LIKE = re.compile(r"^[A-Za-z_][\w.\-]*$")
_CHAR_PREFIX = "character:"


def parse_caption(text) -> tuple[dict | None, str]:
    """Caption JSON string (or dict) → ``(caption, error)``.

    Strict JSON first, then the loose-keys repair of :func:`parse_input`.
    """
    if isinstance(text, dict):
        return text, ""
    s = (text or "").strip() if isinstance(text, str) else ""
    if not s:
        return None, "caption_json is empty"
    try:
        doc = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        doc = parse_input(s)
    if not isinstance(doc, dict):
        return None, "caption_json is not a JSON object"
    return doc, ""


def desc_key(desc) -> str:
    """A desc as slot key: trimmed, ``@`` prefix dropped (as the Assembler)."""
    return (desc if isinstance(desc, str) else "").strip().lstrip("@").strip()


def element_rect(bbox, space: str = "normalized yx", grid_w: int = 1000,
                 grid_h: int = 1000) -> list[float] | None:
    """Element ``bbox`` → relative ``[x1, y1, x2, y2]`` (0..1, clamped).

    ``space`` is one of :data:`BBOX_SPACES`; ``pixels`` measures against
    ``grid_w``×``grid_h`` (the builder's width/height). ``None`` for a
    missing, malformed or empty box.
    """
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        a, b, c, d = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if space.endswith("xy"):
        x1, y1, x2, y2 = a, b, c, d
    else:
        y1, x1, y2, x2 = a, b, c, d
    sx, sy = (float(grid_w), float(grid_h)) if space.startswith("pixels") else (_GRID, _GRID)
    if sx <= 0 or sy <= 0:
        return None
    x1, x2 = sorted((x1 / sx, x2 / sx))
    y1, y2 = sorted((y1 / sy, y2 / sy))
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _label_for(el: dict, slot: str) -> str:
    if el.get("type") == "text":
        return "text"
    if slot.lower().startswith(("character", "person")):
        return "character"
    return "object"


def _short(s: str, n: int = 40) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1] + "…"


def ideogram_boxes(caption: dict, prompt_keys, src_w: int, src_h: int,
                   space: str = "normalized yx", grid_w: int = 1000,
                   grid_h: int = 1000, prompt_given: bool | None = None) -> tuple[list[dict], dict]:
    """Caption elements → box dicts for :func:`make_boxes_doc`.

    ``prompt_keys`` are the JB keys a desc may name (see
    :func:`ideogram_doc`); ``prompt_given`` (default: any keys) decides
    whether an unknown key-like desc is reported as missing. Boxes carry ``bbox`` in ``src_w``×``src_h``
    pixels. Returns ``(boxes, notes)``; ``notes`` lists per element
    ``from_prompt`` / ``own_text`` / ``key_not_in_prompt`` / ``skipped`` and
    ``warnings``.
    """
    cd = caption.get("compositional_deconstruction")
    elements = cd.get("elements") if isinstance(cd, dict) else None
    keys = set(prompt_keys or ())
    if prompt_given is None:
        prompt_given = bool(keys)
    boxes = []
    notes = {"from_prompt": [], "own_text": [], "key_not_in_prompt": [], "skipped": [],
             "warnings": []}
    big = 0
    for i, el in enumerate(elements if isinstance(elements, list) else [], 1):
        if not isinstance(el, dict):
            notes["skipped"].append(f"#{i} not an element")
            continue
        raw_desc = el.get("desc") if isinstance(el.get("desc"), str) else ""
        key = desc_key(raw_desc)
        name = f"#{i} {_short(key) or '(empty desc)'}"
        bbox = el.get("bbox")
        if bbox is None:
            notes["skipped"].append(f"{name}: no bbox (unplaced)")
            continue
        rect = element_rect(bbox, space, grid_w, grid_h)
        if rect is None:
            notes["skipped"].append(f"{name}: bad bbox {bbox!r}")
            continue
        if space.startswith("normalized"):
            try:
                big += any(float(v) > _GRID for v in bbox)
            except (TypeError, ValueError):
                pass

        slot, caption, label = "", "", None
        if key and key in keys:
            slot = key
            notes["from_prompt"].append(f"#{i} {slot}")
        elif raw_desc.strip().lower().startswith(_CHAR_PREFIX):
            caption = raw_desc.strip()[len(_CHAR_PREFIX):].strip()
            label = "character"
            notes["own_text"].append(f"#{i} character: {_short(caption)}")
        else:
            caption = raw_desc.strip()
            if _KEY_LIKE.match(key):
                slot = caption = key
                if prompt_given:
                    notes["key_not_in_prompt"].append(f"#{i} {key}")
                else:
                    notes["own_text"].append(f"#{i} {key}")
            else:
                notes["own_text"].append(f"{name}")
        if label is None or el.get("type") == "text":
            label = _label_for(el, slot)

        x1, y1, x2, y2 = rect
        box = {"bbox": [x1 * src_w, y1 * src_h, x2 * src_w, y2 * src_h],
               "label": label, "slot": slot, "caption": caption,
               "prompt": caption or key or label, "score": 1.0, "source": "ideogram"}
        if label == "text":
            box["text"] = str(el.get("text") or "")
        boxes.append(box)
    if big:
        notes["warnings"].append(
            f"{big} bbox(es) exceed the 0-1000 grid — the builder may be in absolute "
            f"(pixel) mode; set bbox_space to 'pixels …'.")
    return boxes, notes


def _norm(s: str) -> str:
    return " ".join(s.split()).casefold()


def join_unique(*parts) -> str:
    """Join caption parts with spaces, each text once.

    A part already contained in an earlier one is dropped; a part that
    contains an earlier one replaces it in place (whitespace and case are
    ignored). The SAM3 layout node's caption carries the high level
    description and the JB rest inside its ``background`` — joined again
    with the same JB prompt, they must not appear twice.
    """
    out = []
    for p in parts:
        p = p.strip() if isinstance(p, str) else ""
        if not p or any(_norm(p) in _norm(q) for q in out):
            continue
        swallowed = [i for i, q in enumerate(out) if _norm(q) in _norm(p)]
        if swallowed:
            out[swallowed[0]] = p
            out = [q for i, q in enumerate(out) if i not in swallowed[1:]]
        else:
            out.append(p)
    # A background typed as a fragment ("a beach") still ends its sentence.
    return " ".join(q if q[-1] in ".!?…:;\"')" or q is out[-1] else q + "." for q in out)


def ideogram_doc(caption: dict, width: int, height: int, up_width: int = 0,
                 up_height: int = 0, prompt: str = "", high_level_description: str = "",
                 meta: str = "", canvas: str = "single", space: str = "normalized yx",
                 grid_w: int = 0, grid_h: int = 0, src_w: int = 0,
                 src_h: int = 0) -> tuple[dict, dict]:
    """Ideogram caption (+ JB prompt) → ``(MOSS_BOXES doc, report)``.

    ``width``/``height`` is the first-pass canvas; ``grid_w``/``grid_h``
    the builder's pixel grid for ``pixels`` spaces (default: the canvas);
    ``src_w``/``src_h`` the reference image for previews (default: the
    canvas). An empty ``high_level_description`` takes the caption's own.
    """
    src_w, src_h = int(src_w or width), int(src_h or height)
    grid_w, grid_h = int(grid_w or width), int(grid_h or height)

    cd = caption.get("compositional_deconstruction")
    elements = cd.get("elements") if isinstance(cd, dict) else None
    candidates = {desc_key(el.get("desc")) for el in (elements or [])
                  if isinstance(el, dict)} - {""}
    claimed, _rest = split_prompt(prompt, candidates)

    boxes, notes = ideogram_boxes(caption, claimed.keys(), src_w, src_h, space, grid_w, grid_h,
                                  prompt_given=bool((prompt or "").strip()))

    hld = high_level_description if (high_level_description or "").strip() \
        else caption.get("high_level_description", "")
    background = cd.get("background", "") if isinstance(cd, dict) else ""

    doc, base = make_boxes_doc(boxes, src_w, src_h, width, height, up_width, up_height,
                               prompt=prompt or "", meta=meta, canvas=canvas)
    # Re-compose the scene caption so nothing lands twice: a caption from the
    # SAM3 layout node already holds description + JB rest in its background.
    slots = {b["slot"] for b in boxes if b.get("slot")}
    _claimed, rest = split_prompt(prompt, slots)
    rest_prose = rest if isinstance(rest, str) else emit_sentences(rest)
    doc["summary"]["caption"] = join_unique(hld if isinstance(hld, str) else "",
                                            background if isinstance(background, str) else "",
                                            rest_prose)
    doc["source"] = "ideogram"
    for b in doc["boxes"]:
        b["source"] = "ideogram"
    # A JB slot whose sub-tree emitted nothing fell back to its key.
    empty = [s for s in base["fallback"] if s in claimed]
    report = {**notes, "empty_in_prompt": empty, "summary_keys": base["summary_keys"]}
    return doc, report
