"""FVM_MOSS_RegionalLoRA — a LoRA that acts only inside one layout box.

A person LoRA on the whole model pulls every face towards that person. Here
each LoRA becomes a ComfyUI hook on a copy of the conditioning that is masked
to its slot's box; the original conditioning keeps the rest of the canvas,
masked to everything the boxes leave free. The sampler runs one model pass per
copy and blends the predictions by the masks, so the LoRA's weights only shape
its box (tested on MOSS-Image: the other person stays identical to a render
without the LoRA, no seam).

Masks are relative to the ``MOSS_BOXES`` canvas and ComfyUI scales them to the
latent, so the same node serves the first pass and the upscaled refine pass.
"""

from __future__ import annotations

import torch

import comfy.hooks
import folder_paths

from ..utils.lora_cache import _LoraFileCache

try:
    from ...core.jb.moss_layout import rect_pixels
except ImportError:  # pragma: no cover
    from core.jb.moss_layout import rect_pixels


ROWS = 4


def box_mask(rect, width: int, height: int, feather: float = 0.0, expand: float = 0.0) -> torch.Tensor:
    """``[H, W]`` mask of a relative ``rect``: 1 inside, a linear ramp of
    ``feather`` px outside the edge, grown by ``expand`` px first."""
    x1, y1, x2, y2 = rect_pixels(rect, width, height)
    x1, y1, x2, y2 = x1 - expand, y1 - expand, x2 + expand, y2 + expand
    xs = torch.arange(width, dtype=torch.float32) + 0.5
    ys = torch.arange(height, dtype=torch.float32) + 0.5
    dx = torch.clamp(torch.maximum(x1 - xs, xs - x2), min=0)
    dy = torch.clamp(torch.maximum(y1 - ys, ys - y2), min=0)
    if feather > 0:
        mx = torch.clamp(1 - dx / feather, min=0)
        my = torch.clamp(1 - dy / feather, min=0)
    else:
        mx, my = (dx <= 0).float(), (dy <= 0).float()
    return my[:, None] * mx[None, :]


def parse_rows(rows: list[tuple[str, str, float]]) -> dict[str, list[tuple[str, float]]]:
    """``(slot, lora, strength)`` rows → ``{slot: [(lora, strength), ...]}``,
    skipping empty slots, ``None`` LoRAs and zero strength."""
    out: dict[str, list[tuple[str, float]]] = {}
    for slot, lora, strength in rows:
        slot = (slot or "").strip()
        if not slot or not lora or lora == "None" or not strength:
            continue
        out.setdefault(slot, []).append((lora, float(strength)))
    return out


def slot_masks(doc: dict, slots, feather: float, expand: float):
    """Mask per slot (union of its boxes) on the document's canvas, plus the
    rest mask ``1 - max(slot masks)``. Slots without a box are left out."""
    w, h = int(doc["width"]), int(doc["height"])
    masks = {}
    for slot in slots:
        boxes = [b for b in doc.get("boxes", []) if b.get("slot") == slot]
        if not boxes:
            continue
        m = torch.zeros(h, w)
        for b in boxes:
            m = torch.maximum(m, box_mask(b["rect"], w, h, feather, expand))
        masks[slot] = m
    rest = torch.ones(h, w)
    for m in masks.values():
        rest = torch.minimum(rest, 1 - m)
    return masks, rest


class FVM_MOSS_RegionalLoRA:
    CATEGORY = "FVM Tools/JB"
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE", "STRING")
    RETURN_NAMES = ("positive", "negative", "mask_preview", "report")
    DESCRIPTION = (
        "Applies LoRAs only inside a slot's box of the MOSS layout, e.g. a\n"
        "person LoRA on character_1 while character_2 stays untouched.\n\n"
        "Each row: slot | LoRA | strength. Several rows with the same slot\n"
        "stack their LoRAs on that box. Load these LoRAs here and NOT in the\n"
        "model's LoRA loader — there they act on the whole image.\n\n"
        "Wire the encoded MOSS prompt as positive (and the negative when\n"
        "CFG > 1, so both sides carry the LoRA). The KSampler then runs one\n"
        "extra model pass per slot. Works for the first pass and the refine\n"
        "pass: the masks follow the latent size."
    )

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        rows = {}
        for i in range(1, ROWS + 1):
            rows[f"slot_{i}"] = ("STRING", {
                "default": f"character_{i}" if i <= 2 else "",
                "tooltip": "Slot of the box this LoRA acts in (character_1, plant_1, ...). "
                           "Empty = row off."})
            rows[f"lora_{i}"] = (loras, {"tooltip": "LoRA for that box."})
            rows[f"strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                "tooltip": "Model strength of the LoRA."})
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "Encoded MOSS prompt (CLIPTextEncode)."}),
                "moss_boxes": ("MOSS_BOXES", {
                    "tooltip": "Boxes from JB · MOSS Layout from SAM3 / from Ideogram."}),
                "feather": ("FLOAT", {
                    "default": 48.0, "min": 0.0, "max": 512.0, "step": 1.0,
                    "tooltip": "Soft edge outside each box, in pixels of the first-pass "
                               "canvas. 0 = hard edge."}),
                "expand": ("FLOAT", {
                    "default": 0.0, "min": -256.0, "max": 256.0, "step": 1.0,
                    "tooltip": "Grow (or shrink) every box before feathering, in "
                               "first-pass pixels."}),
                **rows,
            },
            "optional": {
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning. Split the same way when wired; "
                               "needed for CFG > 1."}),
            },
        }

    @staticmethod
    def _hooks(entries):
        group = comfy.hooks.HookGroup()
        for lora, strength in entries:
            path = folder_paths.get_full_path("loras", lora)
            if path is None:
                raise FileNotFoundError(f"LoRA not found: {lora}")
            sd = _LoraFileCache.get(path)
            group = group.clone_and_combine(
                comfy.hooks.create_hook_lora(lora=sd, strength_model=strength, strength_clip=0.0))
        return group

    @staticmethod
    def _split(cond, masks, rest, hooks):
        out = comfy.hooks.set_conds_props(
            conds=[cond], strength=1.0, set_cond_area="default", mask=rest)[0]
        for slot, mask in masks.items():
            out = out + comfy.hooks.set_conds_props(
                conds=[cond], strength=1.0, set_cond_area="default", mask=mask,
                hooks=hooks[slot])[0]
        return out

    def apply(self, positive, moss_boxes, feather, expand, negative=None, **rows):
        wanted = parse_rows([(rows.get(f"slot_{i}"), rows.get(f"lora_{i}"), rows.get(f"strength_{i}"))
                             for i in range(1, ROWS + 1)])
        masks, rest = slot_masks(moss_boxes, wanted, feather, expand)
        lines = []
        for slot, entries in wanted.items():
            names = ", ".join(f"{l.replace(chr(92), '/').rsplit('/', 1)[-1]} @{s:g}" for l, s in entries)
            lines.append(f"{slot}: {names}" + ("" if slot in masks else "  — NO BOX, skipped"))
        preview = self._tint(rest, masks)
        if not masks:
            lines.append("No active row matches a box — conditioning passed through.")
            return (positive, negative, preview, "\n".join(lines))

        hooks = {slot: self._hooks(wanted[slot]) for slot in masks}
        pos = self._split(positive, masks, rest, hooks)
        neg = self._split(negative, masks, rest, hooks) if negative is not None else None
        lines.append(f"canvas {moss_boxes['width']}x{moss_boxes['height']}, feather {feather:g}, "
                     f"{len(masks) + 1} model passes per step")
        return (pos, neg, preview, "\n".join(lines))

    @staticmethod
    def _tint(rest, masks):
        """Rest grey, each slot in its own colour — what the sampler blends."""
        colors = [(1.0, 0.3, 0.3), (0.3, 0.6, 1.0), (1.0, 0.8, 0.2), (0.4, 1.0, 0.5)]
        img = rest[..., None] * torch.tensor([0.35, 0.35, 0.35])
        for i, m in enumerate(masks.values()):
            img = img + m[..., None] * torch.tensor(colors[i % len(colors)])
        return img.clamp(0, 1)[None]
