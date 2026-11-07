"""FVM_MOSS_FromIdeogram — a MOSS-Image layout prompt from an Ideogram-4 caption.

The boxes come from KJNodes' *Ideogram 4 Prompt Builder* (drawn, moved and
typed by hand, optionally varied by :class:`FVM_Ideogram_BoxJitter`); the
JB prompt fills every box whose desc names one of its keys, and
:mod:`core.jb.moss_ideogram` writes the MOSS JSON.

Like the SAM3 layout node, two prompts come out: one for the first pass and
one with every box scaled to the upscaled canvas for the refine pass.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..utils.tensor_utils import tensor2np, np2tensor

try:
    from ...core.jb.moss_layout import (
        CANVAS_KINDS, dump_layout, rect_pixels, render_moss, scale_canvas,
    )
    from ...core.jb.moss_ideogram import BBOX_SPACES, ideogram_doc, parse_caption
except ImportError:  # pragma: no cover
    from core.jb.moss_layout import (
        CANVAS_KINDS, dump_layout, rect_pixels, render_moss, scale_canvas,
    )
    from core.jb.moss_ideogram import BBOX_SPACES, ideogram_doc, parse_caption


_COLORS = {"character": (255, 80, 80), "object": (80, 160, 255), "text": (255, 210, 60)}


def draw_doc_preview(image_rgb: np.ndarray | None, doc: dict, width: int,
                     height: int) -> np.ndarray:
    """The reference (or neutral grey) at canvas size with every box, its
    slot and the start of its caption drawn on it."""
    if image_rgb is None:
        img = Image.new("RGB", (width, height), (128, 128, 128))
    else:
        img = Image.fromarray(image_rgb).convert("RGB").resize((width, height), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, width // 50))
    except OSError:
        font = ImageFont.load_default()
    for b in doc.get("boxes", []):
        x1, y1, x2, y2 = rect_pixels(b["rect"], width, height)
        color = _COLORS.get(b.get("label"), (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, width // 300))
        caption = " ".join((b.get("caption") or "").split())
        if len(caption) > 40:
            caption = caption[:39] + "…"
        tag = f"{b['slot']}: {caption}" if b.get("slot") and b["slot"] != caption else caption
        if b.get("label") == "text" and b.get("text"):
            tag += f' "{b["text"]}"'
        draw.text((x1 + 4, y1 + 2), tag or b.get("label", ""), fill=color, font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)


class FVM_MOSS_FromIdeogram:
    CATEGORY = "FVM Tools/JB"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT", "IMAGE", "STRING", "MOSS_BOXES")
    RETURN_NAMES = ("moss_prompt", "moss_prompt_upscaled", "width", "height",
                    "up_width", "up_height", "preview", "report", "moss_boxes")
    DESCRIPTION = (
        "Turns an Ideogram-4 caption — e.g. from KJNodes' Ideogram 4 Prompt\n"
        "Builder, where you draw and move the boxes by hand — into a\n"
        "MOSS-Image layout prompt.\n\n"
        "Each box's desc decides its caption:\n"
        "  character_1          a key of the JB prompt → filled from the JB\n"
        "  character: a nurse   a character box with that caption\n"
        "  a red armchair       used as it is\n"
        "Label: text boxes → text; slots starting with character/person →\n"
        "character; else object. Boxes without bbox are skipped.\n"
        "Scene caption = high_level_description + the caption's background +\n"
        "every JB key no box claims.\n\n"
        "width/height feed the first pass; up_width/up_height and\n"
        "moss_prompt_upscaled feed the refine pass at upscale_by."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption_json": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Ideogram-4 caption JSON: the `prompt` output of KJNodes' "
                               "Ideogram 4 Prompt Builder, or of FVM Ideogram BoxJitter."}),
                "width": ("INT", {
                    "default": 1024, "min": 64, "max": 16384, "step": 16,
                    "tooltip": "First-pass width, snapped to 16 px. Wire the builder's "
                               "width output here."}),
                "height": ("INT", {
                    "default": 1024, "min": 64, "max": 16384, "step": 16,
                    "tooltip": "First-pass height, snapped to 16 px. Wire the builder's "
                               "height output here."}),
                "upscale_by": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Factor of the refine pass. Sets up_width/up_height "
                               "and the boxes of moss_prompt_upscaled."}),
                "high_level_description": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Opening of the scene caption. Empty = the caption's own "
                               "high_level_description."}),
                "meta": ("STRING", {
                    "default": "photo",
                    "tooltip": "Comma-separated medium/genre tags, e.g. photo, portrait."}),
                "canvas": (list(CANVAS_KINDS), {"default": "single"}),
                "bbox_space": (list(BBOX_SPACES), {
                    "default": BBOX_SPACES[0],
                    "tooltip": "How the caption's bbox is measured — match the builder's "
                               "toolbar. normalized = 0-1000 grid (Ideogram default), "
                               "pixels = width/height grid; yx = [ymin, xmin, ymax, xmax] "
                               "(Ideogram default), xy = [xmin, ymin, xmax, ymax]."}),
            },
            "optional": {
                "prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JB prompt: a Stitcher raw_json keyed by slot. A box whose "
                               "desc is one of its keys is filled from it; the unclaimed "
                               "keys join the scene caption."}),
                "image": ("IMAGE", {
                    "tooltip": "Background of the preview only (e.g. the SAM3 reference). "
                               "Without it the preview is grey."}),
            },
        }

    def build(self, caption_json, width, height, upscale_by, high_level_description,
              meta, canvas, bbox_space=BBOX_SPACES[0], prompt="", image=None):
        grid_w, grid_h = int(width), int(height)
        width, height = scale_canvas(grid_w, grid_h, 1.0)
        up_width, up_height = scale_canvas(width, height, upscale_by)
        image_rgb = tensor2np(image) if image is not None else None
        src_h, src_w = image_rgb.shape[:2] if image_rgb is not None else (height, width)
        space = bbox_space if bbox_space in BBOX_SPACES else BBOX_SPACES[0]

        lines = []
        caption, error = parse_caption(caption_json)
        if caption is None:
            lines.append(f"ERROR: {error} — the layout carries the scene caption only.")
            caption = {}

        doc, report = ideogram_doc(
            caption, width, height, up_width, up_height, prompt=prompt or "",
            high_level_description=high_level_description, meta=meta, canvas=canvas,
            space=space, grid_w=grid_w, grid_h=grid_h, src_w=src_w, src_h=src_h)
        layout = render_moss(doc)
        layout_up = render_moss(doc, up_width, up_height)

        lines += [f"WARN: {w}" for w in report["warnings"]]
        lines.append(f"canvas {width}x{height} -> {up_width}x{up_height} ({space})")
        lines.append(f"filled from prompt : {report['from_prompt']}")
        lines.append(f"own text           : {report['own_text']}")
        if report["key_not_in_prompt"]:
            lines.append(f"key not in prompt  : {report['key_not_in_prompt']} (desc used)")
        if report["empty_in_prompt"]:
            lines.append(f"empty in prompt    : {report['empty_in_prompt']} (key used)")
        lines.append(f"skipped            : {report['skipped']}")
        lines.append(f"scene caption keys : {report['summary_keys']}")
        if not doc["boxes"]:
            lines.append("WARN: no boxes — the layout carries the scene caption only.")

        preview = np2tensor(draw_doc_preview(image_rgb, doc, width, height))
        return (dump_layout(layout), dump_layout(layout_up), width, height,
                up_width, up_height, preview, "\n".join(lines), doc)
