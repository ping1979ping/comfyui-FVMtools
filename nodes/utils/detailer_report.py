"""Shared helpers for the PersonDetailer family.

- split_aux_parts / detail_parts_individually: an aux mask (two hands, two feet)
  is detailed part by part, each part in its own crop, instead of one crop
  spanning every part.
- tile_crops: stitch the per-part crops into one preview image.
- build_status_text: node status text — one line per batch image plus a batch
  total line.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def split_aux_parts(mask_2d, min_area_fraction=0.0005):
    """Split a mask into its parts (connected components), reading order.

    Crumbs below min_area_fraction are merged into the nearest real part, so no
    masked pixel is lost. Returns a list of [H, W] float32 tensors; [] for an
    empty mask, [mask] when there is only one part.
    """
    m = mask_2d.detach().float().cpu()
    if m.dim() == 3:
        m = m[0]
    soft = m.numpy()
    binm = (soft > 0.5).astype(np.uint8)
    if not binm.any():
        return []
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binm, connectivity=8)
    h, w = binm.shape
    min_area = max(1, int(h * w * min_area_fraction))
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if len(big) <= 1:
        return [m]

    owner = {}
    for i in range(1, n):
        if i in big:
            owner[i] = i
        else:
            owner[i] = min(big, key=lambda j: float(np.hypot(*(centroids[i] - centroids[j]))))

    # Reading order: top-to-bottom, then left-to-right (row bands of ~1/8 height)
    band = max(1, h // 8)
    big.sort(key=lambda j: (int(centroids[j][1]) // band, centroids[j][0]))
    parts = []
    for j in big:
        ids = [i for i, o in owner.items() if o == j]
        member = np.isin(labels, ids)
        parts.append(torch.from_numpy(np.where(member, soft, 0.0).astype(np.float32)))
    return parts


def tile_crops(crops, gap=4, background=0.08):
    """Stitch [1, h, w, C] crops into one [1, H, W, C] grid image.

    Canvas size = the first crop's size, so the tile concatenates with the other
    refined crops of the same run. Each crop keeps its aspect ratio.
    """
    if not crops:
        return None
    if len(crops) == 1:
        return crops[0]
    first = crops[0]
    _, H, W, C = first.shape
    n = len(crops)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_h = max(1, (H - gap * (rows - 1)) // rows)
    cell_w = max(1, (W - gap * (cols - 1)) // cols)
    canvas = torch.full((1, H, W, C), background, dtype=first.dtype, device=first.device)
    for k, crop in enumerate(crops):
        r, c = divmod(k, cols)
        ch, cw = crop.shape[1], crop.shape[2]
        scale = min(cell_h / ch, cell_w / cw)
        nh, nw = max(1, int(ch * scale)), max(1, int(cw * scale))
        resized = F.interpolate(crop.to(first.device, first.dtype).permute(0, 3, 1, 2),
                                size=(nh, nw), mode="bilinear", align_corners=False)
        resized = resized.permute(0, 2, 3, 1)
        y0 = r * (cell_h + gap) + (cell_h - nh) // 2
        x0 = c * (cell_w + gap) + (cell_w - nw) // 2
        canvas[:, y0:y0 + nh, x0:x0 + nw, :] = resized
    return canvas


def detail_parts_individually(inpaint_fn, image, mask_2d, slot, cached=None, **inpaint_kwargs):
    """Run inpaint_fn once per part of mask_2d.

    Returns (image, tile_or_None, part_count). inpaint_fn has the
    PersonDetailer._inpaint_mask signature and returns (stitched, refined).
    """
    parts = split_aux_parts(mask_2d)
    crops = []
    for part in parts:
        image, refined = inpaint_fn(
            image, part, slot, **inpaint_kwargs,
            cached_model=cached["model"] if cached else None,
            cached_cond=cached["cond"] if cached else None)
        if refined is not None:
            crops.append(refined)
    return image, tile_crops(crops), len(parts)


_STATUS_TOKENS = {
    "no match": "skip",
    "no parts": "aux(0)",
    "no ref": "no ref",
    "empty": "0 faces",
    "no aux data": "no aux",
    "no SEGS": "aux(no input)",
}


def _slot_token(label, info):
    status = info.get("status", "?")
    mask_type = info.get("mask_type", "?")
    if status == "ok":
        if mask_type == "aux":
            return f"{label} aux({info.get('parts', 0)})"
        if "faces" in info:
            return f"{label} {info['faces']}x {mask_type}"
        return f"{label} {mask_type}"
    return f"{label} {_STATUS_TOKENS.get(status, status)}"


def build_status_text(summaries, batch_size, num_refinements, elapsed_s=0, cn_info="",
                      max_image_lines=8):
    """Status text: one 'Img i/B: ...' line per image, then a 'Batch: ...' total line.

    summaries: list of per-image dicts {slot_label: info}; the optional key
    '_refined' holds that image's refinement count (absent = still running).
    Lines are separated by '\\n'.
    """
    lines = []
    shown = summaries if len(summaries) <= max_image_lines else summaries[:max_image_lines - 1]
    for i, s in enumerate(shown):
        tokens = [_slot_token(k, v) for k, v in s.items() if not k.startswith("_")]
        body = " · ".join(tokens) if tokens else "nothing to do"
        suffix = f" → {s['_refined']} refined" if "_refined" in s else " …"
        lines.append(f"Img {i + 1}/{batch_size}: {body}{suffix}")
    if len(summaries) > len(shown):
        lines.append(f"… +{len(summaries) - len(shown)} more images")

    agg = {}
    for s in summaries:
        for k, v in s.items():
            if k.startswith("_"):
                continue
            a = agg.setdefault(k, {"ok": 0, "n": 0, "units": 0, "aux": False})
            a["n"] += 1
            if v.get("status") == "ok":
                a["ok"] += 1
                a["units"] += v.get("parts", v.get("faces", 1))
                a["aux"] = a["aux"] or v.get("mask_type") == "aux"
    done = sum(1 for s in summaries if "_refined" in s)
    total = [f"Batch: {done}/{batch_size} img", f"{num_refinements} refined"]
    for k, a in agg.items():
        t = f"{k} {a['ok']}/{a['n']}"
        if a["aux"]:
            t += f" ({a['units']} parts)"
        elif a["units"] > a["ok"]:
            t += f" ({a['units']} faces)"
        total.append(t)
    if cn_info:
        total.append(f"[{cn_info}]")
    if elapsed_s > 0:
        total.append(f"{elapsed_s}s")
    lines.append(" · ".join(total))
    return "\n".join(lines)
