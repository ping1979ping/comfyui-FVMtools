"""SignSelectorSAM3 — finds text-bearing regions (signs, labels, prints) via SAM3.

Grounds every enabled text class against a single SAM3 vision encode, gates the
hits by minimum size, scores how implausible the existing lettering looks, and
groups near-identical regions so a shelf of identical bottles is treated as one
decision instead of twelve.

Output is SIGN_DATA, consumed by SignTextProposer and SignDetailer.
"""

import numpy as np
import cv2
import torch

from ..utils.masker import sam3_prepare, sam3_ground
from ..utils.tensor_utils import tensor2np, empty_mask
from ..utils.ocr_backend import ocr_region, get_available_backends
try:  # relative inside ComfyUI's loader, absolute under pytest
    from ...core.signs.classes import (
        SIGN_CLASSES, all_class_names, get_class, parse_custom_prompts,
    )
    from ...core.signs.slop import score_slop
    from ...core.signs.cluster import cluster_crops, pick_cluster_representative
except ImportError:
    from core.signs.classes import (
        SIGN_CLASSES, all_class_names, get_class, parse_custom_prompts,
    )
    from core.signs.slop import score_slop
    from core.signs.cluster import cluster_crops, pick_cluster_representative


# Preview overlay colours per class (RGB)
_CLASS_COLORS = {
    "sign":          (255, 70, 70),
    "label":         (70, 200, 255),
    "garment_print": (120, 255, 90),
    "poster":        (255, 190, 40),
    "screen":        (180, 120, 255),
    "book":          (255, 120, 200),
    "plate":         (250, 250, 250),
    "paper":         (150, 220, 180),
    "graffiti":      (255, 140, 60),
    "custom":        (200, 200, 200),
}

# Crops are padded to this canvas so they can travel as one IMAGE batch.
CROP_CANVAS = 512


def _bbox_from_mask(mask_np):
    """Tight integer bbox [x1, y1, x2, y2] of a binary mask, or None if empty."""
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _mask_iou(a, b):
    """Intersection-over-union of two float masks."""
    ab = (a > 0.5)
    bb = (b > 0.5)
    union = np.logical_or(ab, bb).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(ab, bb).sum()) / float(union)


def _short_side_px(mask_np):
    """Short side of the mask's min-area rect — the effective text height."""
    m = (mask_np > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return 0
    (_, _), (w, h), _ = cv2.minAreaRect(largest)
    return int(min(w, h))


def _crop_to_canvas(image_rgb, bbox, canvas=CROP_CANVAS, pad_ratio=0.10):
    """Crop the bbox with a little context and letterbox it onto a square canvas."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    pw = int((x2 - x1) * pad_ratio) + 2
    ph = int((y2 - y1) * pad_ratio) + 2
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w - 1, x2 + pw)
    y2 = min(h - 1, y2 + ph)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((canvas, canvas, 3), dtype=np.uint8)

    patch = image_rgb[y1:y2 + 1, x1:x2 + 1]
    ph_, pw_ = patch.shape[:2]
    scale = min(canvas / pw_, canvas / ph_)
    nw, nh = max(1, int(pw_ * scale)), max(1, int(ph_ * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(patch, (nw, nh), interpolation=interp)

    out = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    ox, oy = (canvas - nw) // 2, (canvas - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = resized
    return out


class SignSelectorSAM3:
    """Locates text-bearing regions and judges whether their lettering is believable."""

    CATEGORY = "FVM Tools/Text"
    FUNCTION = "execute"
    RETURN_TYPES = ("SIGN_DATA", "MASK", "IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("sign_data", "masks", "crops", "preview", "region_count", "report")
    OUTPUT_NODE = True

    DESCRIPTION = (
        "Finds signs, labels, prints and other text-bearing regions with SAM3 text grounding.\n\n"
        "Nine built-in classes (sign, label, garment_print, poster, screen, book, plate,\n"
        "paper, graffiti), each with its own prompts, threshold and minimum size. All classes\n"
        "share ONE SAM3 vision encode per image, so enabling all nine is cheap.\n\n"
        "Regions below the size gate are flagged too_small rather than dropped — the Detailer\n"
        "decides whether to skip them or soften them into believable out-of-focus text.\n\n"
        "When an OCR backend is installed, each region also gets a slop score: SAM3 saying\n"
        "'there is text here' while OCR reads nothing is the classic pseudo-glyph signature.\n\n"
        "Connect LoadSAM3Model -> sam3_model. Feed sign_data into Sign Text Proposer."
    )

    @classmethod
    def INPUT_TYPES(cls):
        class_toggles = {}
        for name in all_class_names():
            cfg = SIGN_CLASSES[name]
            class_toggles[f"class_{name}"] = ("BOOLEAN", {
                "default": True,
                "tooltip": f"Ground '{name}' — prompts: {', '.join(cfg['sam3_prompts'])} "
                           f"(default threshold {cfg['threshold']}, min height {cfg['min_height_px']}px)",
            })

        return {
            "required": {
                "sam3_model": ("SAM3_MODEL_CONFIG", {"tooltip": "SAM3 model from the LoadSAM3Model node"}),
                "image": ("IMAGE", {"tooltip": "Image(s) to scan for text regions. Batch supported."}),
                **class_toggles,
                "custom_prompts": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Extra SAM3 prompts beyond the built-in classes.\n"
                               "Format: 'neon sign:0.25, bottle label:0.3' — the threshold is optional."}),
                "threshold_scale": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 2.0, "step": 0.05,
                    "tooltip": "Multiplies every class's default threshold.\n"
                               "Below 1.0 finds more (and more false positives), above 1.0 is stricter."}),
                "min_height_px": ("INT", {"default": 24, "min": 4, "max": 512, "step": 1,
                    "tooltip": "Global floor for text height in the ORIGINAL image (min-area-rect short side).\n"
                               "Per-class minimums still apply on top of this."}),
                "min_area_ratio": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 0.5, "step": 0.0001,
                    "tooltip": "Region must cover at least this fraction of the image area."}),
                "max_regions": ("INT", {"default": 12, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Cost brake — keeps only the top N regions after sorting."}),
                "merge_iou": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Two classes hitting the same object are merged above this IoU.\n"
                               "The higher-scoring detection keeps its class."}),
                "slop_detection": (["off", "ocr", "vlm", "ocr+vlm"], {"default": "ocr",
                    "tooltip": "How to judge whether existing lettering is believable.\n"
                               "- ocr: OCR confidence + dictionary + bigram plausibility\n"
                               "- vlm: leave the judgement to the Proposer's vision model\n"
                               "- ocr+vlm: both, combined in the Proposer\n"
                               "Falls back gracefully when no OCR backend is installed."}),
                "slop_threshold": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Regions scoring at or above this are marked as needing a re-render."}),
                "only_slop": ("BOOLEAN", {"default": False,
                    "tooltip": "ON: drop regions whose text already looks fine. OFF: keep everything and let the Detailer decide."}),
                "cluster_similar": ("BOOLEAN", {"default": True,
                    "tooltip": "Group near-identical regions (a shelf of identical bottles) so they share one text decision."}),
                "cluster_distance": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.9, "step": 0.01,
                    "tooltip": "Lower = stricter grouping. Combines perceptual hash and colour signature."}),
                "sort_order": (["area_desc", "score_desc", "left_right", "top_down"], {"default": "area_desc",
                    "tooltip": "Order of regions — also the order in which the Detailer renders them."}),
            },
            "optional": {
                "restrict_mask": ("MASK", {"tooltip": "Only search inside this mask. Regions must overlap it by 30%+."}),
                "ocr_backend": (["auto", "onnx", "easyocr", "none"], {"default": "auto",
                    "tooltip": "OCR engine for slop detection. 'auto' picks the first installed one.\n"
                               "Missing models degrade to VLM-only judgement, never an error."}),
            },
        }

    # ── Grounding ──

    def _collect_prompts(self, kwargs, custom_prompts, threshold_scale):
        """Build the (class_name, prompt, threshold) work list from the toggles."""
        jobs = []
        for name in all_class_names():
            if not kwargs.get(f"class_{name}", False):
                continue
            cfg = SIGN_CLASSES[name]
            thr = float(np.clip(cfg["threshold"] * threshold_scale, 0.05, 0.99))
            for prompt in cfg["sam3_prompts"]:
                jobs.append((name, prompt, thr))

        for prompt, thr in parse_custom_prompts(custom_prompts):
            jobs.append(("custom", prompt, float(np.clip(thr * threshold_scale, 0.05, 0.99))))
        return jobs

    def _ground_all(self, sam3_model, image_rgb, jobs):
        """One vision encode, then every prompt grounded against it."""
        processor, base_state = sam3_prepare(sam3_model, image_rgb)
        if processor is None:
            print("[SignSelector] SAM3 prepare failed — no regions found")
            return []

        raw = []
        for class_name, prompt, thr in jobs:
            try:
                results = sam3_ground(processor, base_state, image_rgb.shape, prompt, threshold=thr)
            except Exception as exc:  # a single bad prompt must not kill the run
                print(f"[SignSelector] grounding '{prompt}' failed: {exc}")
                continue
            for mask_np, score, bbox in results:
                raw.append({"class": class_name, "prompt": prompt, "mask": mask_np,
                            "score": float(score), "bbox": bbox})
        return raw

    def _merge_overlaps(self, raw, merge_iou):
        """Collapse detections of the same object found by several classes."""
        if merge_iou >= 1.0 or not raw:
            return raw
        ordered = sorted(raw, key=lambda r: r["score"], reverse=True)
        kept = []
        for cand in ordered:
            duplicate = False
            for k in kept:
                if _mask_iou(cand["mask"], k["mask"]) >= merge_iou:
                    k.setdefault("also_matched", []).append(f"{cand['class']}:{cand['prompt']}")
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        return kept

    # ── Region assembly ──

    def _build_regions(self, raw, image_rgb, batch_index, min_height_px, min_area_ratio,
                       restrict_np):
        """Turn raw detections into region dicts with geometry and size verdicts."""
        h, w = image_rgb.shape[:2]
        image_area = float(h * w)
        regions = []

        for det in raw:
            mask_np = det["mask"].astype(np.float32)
            bbox = _bbox_from_mask(mask_np)
            if bbox is None:
                continue

            if restrict_np is not None:
                inside = float(((mask_np > 0.5) & (restrict_np > 0.5)).sum())
                total = float((mask_np > 0.5).sum())
                if total <= 0 or inside / total < 0.30:
                    continue

            area_px = int((mask_np > 0.5).sum())
            if area_px / image_area < min_area_ratio:
                continue

            height_px = _short_side_px(mask_np)
            class_min = get_class(det["class"])["min_height_px"]
            too_small = height_px < max(min_height_px, class_min)

            regions.append({
                "class": det["class"],
                "prompt": det["prompt"],
                "score": det["score"],
                "mask": mask_np,
                "bbox": bbox,
                "batch_index": batch_index,
                "area_px": area_px,
                "height_px": height_px,
                "too_small": too_small,
                "also_matched": det.get("also_matched", []),
                "cluster_id": -1,
                "slop": {"score": 0.0, "verdict": "unknown", "ocr_text": "",
                         "ocr_conf": 0.0, "signals": {}},
                "proposal": None,
            })
        return regions

    def _sort_regions(self, regions, sort_order):
        if sort_order == "score_desc":
            return sorted(regions, key=lambda r: r["score"], reverse=True)
        if sort_order == "left_right":
            return sorted(regions, key=lambda r: r["bbox"][0])
        if sort_order == "top_down":
            return sorted(regions, key=lambda r: r["bbox"][1])
        return sorted(regions, key=lambda r: r["area_px"], reverse=True)

    # ── Slop scoring ──

    def _score_regions(self, regions, image_rgb, mode, backend, threshold, has_backend=True):
        """Attach OCR readings and a slop score to every region.

        With no OCR backend installed the scoring is skipped entirely. Running it
        anyway would read every region as an empty OCR result and therefore as a
        pseudo-glyph hit — 'OCR is absent' is not the same finding as 'OCR read
        nothing', and conflating them marks every region as slop.
        """
        if mode in ("off", "vlm") or not has_backend:
            for r in regions:
                r["slop"]["verdict"] = "clean" if mode == "off" else "unknown"
                r["slop"]["needs_fix"] = mode != "off"
            return

        for r in regions:
            reading = ocr_region(image_rgb, r["mask"], backend=backend)
            scored = score_slop(
                ocr_text=reading.get("text", ""),
                ocr_conf=reading.get("conf", 0.0),
                char_confs=reading.get("char_confs", []),
                text_region_detected=True,
            )
            scored["ocr_text"] = reading.get("text", "")
            scored["ocr_conf"] = reading.get("conf", 0.0)
            scored["needs_fix"] = scored["score"] >= threshold
            r["slop"] = scored

    # ── Preview ──

    def _draw_preview(self, image_rgb, regions):
        """Numbered outlines with class, size verdict and slop score."""
        canvas = image_rgb.copy()
        overlay = canvas.copy()

        for i, r in enumerate(regions):
            color = _CLASS_COLORS.get(r["class"], (200, 200, 200))
            mask_u8 = (r["mask"] > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, cv2.FILLED)
            cv2.drawContours(canvas, contours, -1, color, 2)

            x1, y1 = r["bbox"][0], r["bbox"][1]
            tags = [f"#{i + 1}", r["class"]]
            if r["cluster_id"] >= 0:
                tags.append(f"c{r['cluster_id']}")
            if r["too_small"]:
                tags.append(f"small {r['height_px']}px")
            if r["slop"].get("verdict") != "unknown":
                tags.append(f"slop {r['slop']['score']:.2f}")
            label = " ".join(tags)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 4, y1 - 4)
            cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (0, 0, 0), cv2.FILLED)
            cv2.putText(canvas, label, (x1 + 3, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

    # ── Main ──

    def execute(self, sam3_model, image, custom_prompts="", threshold_scale=1.0,
                min_height_px=24, min_area_ratio=0.0005, max_regions=12, merge_iou=0.50,
                slop_detection="ocr", slop_threshold=0.50, only_slop=False,
                cluster_similar=True, cluster_distance=0.15, sort_order="area_desc",
                restrict_mask=None, ocr_backend="auto", **kwargs):

        batch_size = image.shape[0]
        h, w = int(image.shape[1]), int(image.shape[2])
        jobs = self._collect_prompts(kwargs, custom_prompts, threshold_scale)

        available = get_available_backends()
        effective_backend = "none" if ocr_backend == "none" else ocr_backend
        if slop_detection in ("ocr", "ocr+vlm") and not available and ocr_backend != "none":
            print("[SignSelector] no OCR backend installed — slop detection falls back to the vision model")

        all_regions = []
        preview_frames = []
        report_lines = [
            f"Sign Selector — {batch_size} image(s), {len(jobs)} grounding prompt(s)",
            f"OCR backends available: {', '.join(available) if available else 'none'}",
        ]

        for b in range(batch_size):
            single = image[b:b + 1]
            rgb = tensor2np(single)

            restrict_np = None
            if restrict_mask is not None:
                idx = min(b, restrict_mask.shape[0] - 1)
                restrict_np = restrict_mask[idx].cpu().numpy().astype(np.float32)
                if restrict_np.shape != (h, w):
                    restrict_np = cv2.resize(restrict_np, (w, h), interpolation=cv2.INTER_NEAREST)

            raw = self._ground_all(sam3_model, rgb, jobs)
            raw = self._merge_overlaps(raw, merge_iou)
            regions = self._build_regions(raw, rgb, b, min_height_px, min_area_ratio, restrict_np)
            regions = self._sort_regions(regions, sort_order)[:max_regions]

            self._score_regions(regions, rgb, slop_detection, effective_backend, slop_threshold,
                                has_backend=bool(available) and ocr_backend != "none")

            if only_slop and slop_detection in ("ocr", "ocr+vlm"):
                before = len(regions)
                regions = [r for r in regions if r["slop"].get("needs_fix", True)]
                report_lines.append(f"  image {b + 1}: dropped {before - len(regions)} already-legible region(s)")

            # Attach crops, then cluster on them
            for r in regions:
                r["crop"] = _crop_to_canvas(rgb, r["bbox"])

            if cluster_similar and len(regions) > 1:
                labels = cluster_crops([r["crop"] for r in regions], distance=cluster_distance)
                for r, cid in zip(regions, labels):
                    r["cluster_id"] = int(cid)
                for cid in sorted(set(labels)):
                    members = [i for i, lab in enumerate(labels) if lab == cid]
                    if len(members) > 1:
                        rep_local = pick_cluster_representative(
                            [regions[i]["crop"] for i in members], [0] * len(members), 0)
                        rep = members[rep_local]
                        for i in members:
                            regions[i]["cluster_rep"] = (i == rep)
                        report_lines.append(
                            f"  image {b + 1}: cluster {cid} groups {len(members)} regions "
                            f"(representative #{rep + 1})")
                    else:
                        regions[members[0]]["cluster_rep"] = True
            else:
                for r in regions:
                    r["cluster_rep"] = True

            for i, r in enumerate(regions):
                r["index"] = len(all_regions) + i
            all_regions.extend(regions)

            preview_frames.append(self._draw_preview(rgb, regions))
            small = sum(1 for r in regions if r["too_small"])
            report_lines.append(
                f"  image {b + 1}: {len(regions)} region(s), {small} below the size gate")
            for r in regions:
                report_lines.append(
                    f"    #{r['index'] + 1} {r['class']:<14} {r['height_px']:>4}px "
                    f"score={r['score']:.2f} slop={r['slop']['score']:.2f} "
                    f"({r['slop'].get('verdict', '?')}) text={r['slop'].get('ocr_text', '')!r}")

        # ── Outputs ──
        if all_regions:
            masks = torch.stack([torch.from_numpy(r["mask"]) for r in all_regions])
            crops = torch.stack([torch.from_numpy(r["crop"].astype(np.float32) / 255.0)
                                 for r in all_regions])
        else:
            masks = empty_mask(h, w)
            crops = torch.zeros(1, CROP_CANVAS, CROP_CANVAS, 3, dtype=torch.float32)
            report_lines.append("  no regions found — try lowering threshold_scale or min_height_px")

        preview = torch.stack([torch.from_numpy(p.astype(np.float32) / 255.0) for p in preview_frames])

        sign_data = {
            "regions": all_regions,
            "image_shape": (h, w),
            "batch_size": batch_size,
            "slop_mode": slop_detection,
            "slop_threshold": slop_threshold,
            "ocr_backends": available,
        }

        report = "\n".join(report_lines)
        print(f"[SignSelector] {len(all_regions)} region(s) across {batch_size} image(s)")
        return (sign_data, masks, crops, preview, len(all_regions), report)
