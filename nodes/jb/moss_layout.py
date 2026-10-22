"""FVM_MOSS_SAM3Layout — a MOSS-Image layout prompt from a reference image.

SAM3 finds the boxes (one text prompt per rule), the JB prompt describes
them, and :mod:`core.jb.moss_layout` writes the MOSS JSON. The reference
image only lends its composition: the render is a fresh text-to-image on a
canvas of the same aspect.

Two prompts come out: one for the first pass and one with every box scaled
to the upscaled canvas for the refine pass, because MOSS boxes are pixels of
the canvas they are rendered on.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..utils.masker import NativeSAM3, sam3_prepare, sam3_ground, sam3_last_error
from ..utils.tensor_utils import tensor2np, np2tensor

try:
    from ...core.jb.moss_layout import (
        CANVAS_KINDS, build_layout, dump_layout, fit_canvas, number_slots, scale_canvas,
        parse_box_rules, select_hits,
    )
except ImportError:  # pragma: no cover
    from core.jb.moss_layout import (
        CANVAS_KINDS, build_layout, dump_layout, fit_canvas, number_slots, scale_canvas,
        parse_box_rules, select_hits,
    )


DEFAULT_RULES = "person | character | character_{n}\n"

_COLORS = {"character": (255, 80, 80), "object": (80, 160, 255), "text": (255, 210, 60)}


def draw_preview(image_rgb: np.ndarray, boxes: list[dict], width: int, height: int) -> np.ndarray:
    """The reference at canvas size with every box and its slot drawn on it."""
    src_h, src_w = image_rgb.shape[:2]
    img = Image.fromarray(image_rgb).resize((width, height), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, width // 50))
    except OSError:
        font = ImageFont.load_default()
    sx, sy = width / src_w, height / src_h
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        rect = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        color = _COLORS.get(b["label"], (255, 255, 255))
        draw.rectangle(rect, outline=color, width=max(2, width // 300))
        tag = f"{b['slot'] or b['prompt']} ({b['score']:.2f})"
        draw.text((rect[0] + 4, rect[1] + 2), tag, fill=color, font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)


class FVM_MOSS_SAM3Layout:
    CATEGORY = "FVM Tools/JB"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT", "IMAGE", "STRING")
    RETURN_NAMES = ("moss_prompt", "moss_prompt_upscaled", "width", "height",
                    "up_width", "up_height", "preview", "report")
    DESCRIPTION = (
        "Turns the composition of a reference image into a MOSS-Image layout\n"
        "prompt. SAM3 finds a box per rule; the JB prompt fills each box from\n"
        "its slot (character_1, ...) and the rest — usually the location —\n"
        "becomes the scene caption.\n\n"
        "box_rules, one per line:  sam3 prompt | label | slot\n"
        "  person | character | character_{n}\n"
        "  table  | object    | table\n"
        "label: character / object / text. {n} numbers hits left to right.\n\n"
        "width/height feed the first pass; up_width/up_height and\n"
        "moss_prompt_upscaled feed the refine pass at upscale_by."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Reference image. Only its layout and aspect are used."}),
                "box_rules": ("STRING", {
                    "default": DEFAULT_RULES, "multiline": True,
                    "tooltip": "One rule per line: sam3 prompt | label | slot.\n"
                               "label: character, object or text (default object).\n"
                               "slot: key in the JB prompt that describes the box; "
                               "{n} numbers the hits left to right. Without a "
                               "matching key the box is captioned with its SAM3 prompt."}),
                "threshold": ("FLOAT", {
                    "default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "SAM3 confidence for every rule."}),
                "max_per_rule": ("INT", {
                    "default": 3, "min": 1, "max": 16,
                    "tooltip": "At most this many boxes per rule, best score first."}),
                "min_area": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "Smaller boxes (share of the image) are dropped."}),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Size of the first pass at the reference's aspect, "
                               "snapped to 16 px. 0 = reference size."}),
                "upscale_by": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Factor of the refine pass. Sets up_width/up_height "
                               "and the boxes of moss_prompt_upscaled."}),
                "high_level_description": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Opening of the scene caption, before the JB rest."}),
                "meta": ("STRING", {
                    "default": "photo",
                    "tooltip": "Comma-separated medium/genre tags, e.g. photo, portrait."}),
                "canvas": (list(CANVAS_KINDS), {"default": "single"}),
            },
            "optional": {
                "prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JB prompt: a Stitcher raw_json (keyed by slot), or "
                               "plain prose for the scene caption."}),
                "sam3_native_model": ("MODEL", {
                    "tooltip": "Native SAM3: MODEL from CheckpointLoaderSimple on "
                               "sam3.1_multiplex_fp16. Wire its CLIP below as well."}),
                "sam3_native_clip": ("CLIP", {
                    "tooltip": "Native SAM3: CLIP from the SAME CheckpointLoaderSimple."}),
                "sam3_model": ("SAM3_MODEL_CONFIG", {
                    "tooltip": "Alternative: Load SAM3 Model of the comfyui-sam3 extension. "
                               "Used only when no native model is wired."}),
            },
        }

    def build(self, image, box_rules, threshold, max_per_rule, min_area,
              megapixels, upscale_by, high_level_description, meta, canvas, prompt="",
              sam3_native_model=None, sam3_native_clip=None, sam3_model=None):
        if sam3_native_model is not None and sam3_native_clip is not None:
            sam3_model = NativeSAM3(sam3_native_model, sam3_native_clip)
        image_rgb = tensor2np(image)
        src_h, src_w = image_rgb.shape[:2]
        width, height = fit_canvas(src_w, src_h, megapixels)
        up_width, up_height = scale_canvas(width, height, upscale_by)

        rules = parse_box_rules(box_rules)
        lines = []
        boxes = []
        if rules and sam3_model is None:
            lines.append("ERROR: no SAM3 wired — connect sam3_native_model + sam3_native_clip "
                         "or sam3_model.")
        elif rules:
            processor, base_state = sam3_prepare(sam3_model, image_rgb)
            if processor is None:
                lines.append(f"ERROR: SAM3 did not load — {sam3_last_error() or 'see console'}")
            else:
                for rule in rules:
                    raw = sam3_ground(processor, base_state, image_rgb.shape,
                                      rule["prompt"], threshold=threshold)
                    hits = [{"bbox": [float(v) for v in bbox], "score": float(score)}
                            for _mask, score, bbox in raw]
                    kept = select_hits(hits, max_per_rule, min_area, src_w, src_h)
                    boxes += number_slots(rule, kept)
                    lines.append(f"'{rule['prompt']}': {len(raw)} found, {len(kept)} kept")

        args = dict(prompt=prompt or "", high_level_description=high_level_description,
                    meta=meta, canvas=canvas)
        layout, report = build_layout(boxes, src_w, src_h, width, height, **args)
        layout_up, _ = build_layout(boxes, src_w, src_h, up_width, up_height, **args)

        lines.append(f"canvas {width}x{height} -> {up_width}x{up_height}")
        lines.append(f"filled from prompt : {report['filled']}")
        lines.append(f"captioned by SAM3  : {report['fallback']}")
        lines.append(f"scene caption keys : {report['summary_keys']}")
        if not boxes:
            lines.append("WARN: no boxes — the layout carries the scene caption only.")

        preview = np2tensor(draw_preview(image_rgb, boxes, width, height))
        return (dump_layout(layout), dump_layout(layout_up), width, height,
                up_width, up_height, preview, "\n".join(lines))
