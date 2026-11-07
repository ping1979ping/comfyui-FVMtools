"""FVM_MOSS_BoxPersonData — PERSON_DATA from the boxes of a MOSS layout.

The Person Selectors decide who is who by matching faces against reference
images. A MOSS render already says who should be where: every character box
of the layout (``MOSS_BOXES``) has a slot (``character_1``, ...) and a
caption. This node grounds the persons in the rendered image with SAM3 and
gives each slot the person whose box overlaps the slot box most. The result
is a PERSON_DATA in the Person Selector SAM3 layout, so a PersonDetailer
details ``character_1`` with reference_1's LoRA and prompt, ``character_2``
with reference_2's, and so on — no reference images needed.

Masks come from the same SAM3 prompts and thresholds as Person Selector SAM3
(``person`` body, ``face``, ``head``, ``hair``, BiSeNet facial subtypes per
face); face/head/hair are tied to a person by overlap with its body mask.
Persons that land in no slot box are left unassigned — PersonDetailer's
Generic slot picks them up.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from ..person_selector_sam3 import (
    BISENET_SUBTYPES, PersonSelectorSAM3, SAM3_MASK_CONFIG, _PREVIEW_COLORS,
)
from ..utils.masker import (
    MASK_TYPE_LABELS, MaskGenerator, NativeSAM3, _clip_labels_to_body,
    assign_masks_by_body_overlap, sam3_ground, sam3_last_error, sam3_prepare,
)
from ..utils.tensor_utils import empty_mask, mask2tensor, np2tensor, tensor2np

try:
    from ...core.jb.moss_layout import box_iou, rect_pixels
except ImportError:  # pragma: no cover
    from core.jb.moss_layout import box_iou, rect_pixels


MAX_REFERENCES = 5
DEFAULT_SLOTS = ", ".join(f"character_{i}" for i in range(1, MAX_REFERENCES + 1))

# Every mask type a PersonDetailer slot can ask for (person_data[f"{mt}_masks"]).
MASK_TYPES = ["face", "head", "body", "hair"] + list(BISENET_SUBTYPES)

# Same box NMS as the layout node: SAM3 sometimes returns one person twice.
PERSON_NMS_IOU = 0.6

_UNMATCHED_COLOR = (160, 160, 160)


# ── Pure helpers (no SAM3) ──

def _natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def resolve_slots(slots_text: str, moss_boxes: dict) -> tuple[list[str], list[str]]:
    """Slot names for reference_1..5, in order. Returns ``(slots, warnings)``.

    ``slots_text`` is a comma/newline separated list. Empty takes every
    ``character`` box of the layout, ordered by slot name.
    """
    warnings = []
    names = []
    for part in re.split(r"[,\n]", slots_text or ""):
        name = part.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        boxes = (moss_boxes or {}).get("boxes") or []
        names = sorted({b.get("slot") for b in boxes
                        if b.get("label") == "character" and b.get("slot")}, key=_natural_key)
    if len(names) > MAX_REFERENCES:
        warnings.append(f"WARN: {len(names)} slots, only the first {MAX_REFERENCES} "
                        f"map to reference_1..{MAX_REFERENCES}: dropped {names[MAX_REFERENCES:]}")
        names = names[:MAX_REFERENCES]
    return names, warnings


def slot_boxes(moss_boxes: dict, slots: list[str]) -> list[dict | None]:
    """The layout box of each slot (first box carrying that slot name), or None."""
    by_slot = {}
    for b in (moss_boxes or {}).get("boxes") or []:
        name = b.get("slot")
        if name and name not in by_slot and b.get("rect"):
            by_slot[name] = b
    return [by_slot.get(s) for s in slots]


def box_inside(person, slot) -> float:
    """Share of the ``person`` box ``[x1, y1, x2, y2]`` that lies inside ``slot``."""
    ix = max(0.0, min(person[2], slot[2]) - max(person[0], slot[0]))
    iy = max(0.0, min(person[3], slot[3]) - max(person[1], slot[1]))
    area = max((person[2] - person[0]) * (person[3] - person[1]), 1e-9)
    return ix * iy / area


def dedupe_persons(results, iou: float = PERSON_NMS_IOU):
    """Best-first box NMS over SAM3 ``(mask, score, bbox)`` results."""
    kept = []
    for r in sorted(results, key=lambda r: r[1], reverse=True):
        if any(box_iou(r[2], k[2]) >= iou for k in kept):
            continue
        kept.append(r)
    return kept


def assign_persons_to_slots(person_boxes, slot_rects, min_overlap: float):
    """Greedy best-first: each slot gets the person whose box overlaps its box most.

    ``person_boxes`` are pixel ``[x1, y1, x2, y2]``; ``slot_rects`` the same or
    None for a slot without a layout box. Score = IoU of the two boxes; a pair
    below ``min_overlap`` is never assigned. One person per slot and vice
    versa. Returns ``({slot_idx: (person_idx, iou)}, iou_matrix [P, S])``.
    """
    n_p, n_s = len(person_boxes), len(slot_rects)
    scores = np.zeros((n_p, n_s), dtype=np.float64)
    for pi, pb in enumerate(person_boxes):
        for si, sr in enumerate(slot_rects):
            if sr is not None:
                scores[pi, si] = box_iou(pb, sr)
    pairs = sorted(((scores[pi, si], pi, si) for pi in range(n_p) for si in range(n_s)),
                   key=lambda t: (-t[0], t[2], t[1]))
    taken_p, out = set(), {}
    for score, pi, si in pairs:
        if score < min_overlap or score <= 0:
            break
        if si in out or pi in taken_p:
            continue
        out[si] = (pi, float(score))
        taken_p.add(pi)
    return out, scores


def _mask_box(mask_np):
    ys, xs = np.nonzero(mask_np > 0.5)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _bisenet_subtypes(rgb, face_box, body_np, device):
    """BiSeNet facial subtype masks for one face box, clipped to its body.

    Same derivation as Person Selector SAM3, with the SAM3 face box standing
    in for the InsightFace bbox (``_run_bisenet`` only reads ``.bbox``).
    """
    label_map = MaskGenerator._run_bisenet(rgb, SimpleNamespace(bbox=face_box), device)
    if body_np is not None and body_np.max() > 0.5:
        label_map = _clip_labels_to_body(label_map, body_np)
    return {mt: np.isin(label_map, list(MASK_TYPE_LABELS[mt])).astype(np.float32)
            for mt in BISENET_SUBTYPES}


class FVM_MOSS_BoxPersonData:
    CATEGORY = "FVM Tools/JB"
    FUNCTION = "build"
    RETURN_TYPES = ("PERSON_DATA",) + ("STRING",) * MAX_REFERENCES + ("IMAGE", "STRING")
    RETURN_NAMES = (("person_data",)
                    + tuple(f"reference_{i}_prompt" for i in range(1, MAX_REFERENCES + 1))
                    + ("preview", "report"))
    DESCRIPTION = (
        "PERSON_DATA from the boxes of a MOSS layout instead of face references.\n\n"
        "SAM3 finds the persons in the rendered image (body, face, head, hair as\n"
        "in Person Selector SAM3). Each slot listed in `slots` becomes\n"
        "reference_1..5 and gets the person whose box overlaps the slot's layout\n"
        "box most (IoU, one person per slot). Persons in no box stay unassigned\n"
        "for the PersonDetailer's Generic slot.\n\n"
        "reference_N_prompt carries the caption of slot N — convert the\n"
        "PersonDetailer's reference_N_prompt widget to an input and wire it."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The rendered image(s). Any resolution of the layout's aspect: "
                               "the boxes are relative."}),
                "moss_boxes": ("MOSS_BOXES", {
                    "tooltip": "Layout boxes from MOSS SAM3 Layout (or the Layout Editor)."}),
                "slots": ("STRING", {
                    "default": DEFAULT_SLOTS, "multiline": False,
                    "tooltip": "Slot names for reference_1..5, in order, comma separated.\n"
                               "Empty = every character box of the layout, sorted by slot name."}),
                "threshold": ("FLOAT", {
                    "default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "SAM3 confidence for 'person'. face/head/hair use the "
                               "Person Selector SAM3 thresholds."}),
                "min_overlap": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum IoU between a person's box and a slot box for the "
                               "person to take that slot."}),
                "depth_sort_order": (["front_last", "front_first", "off"], {
                    "default": "front_last",
                    "tooltip": "Rendering order handed to the PersonDetailer. Depth proxy: "
                               "the height of the face in the image (lower = nearer)."}),
            },
            "optional": {
                "sam3_native_model": ("MODEL", {
                    "tooltip": "Native SAM3: MODEL from CheckpointLoaderSimple on "
                               "sam3.1_multiplex_fp16. Wire its CLIP below as well."}),
                "sam3_native_clip": ("CLIP", {
                    "tooltip": "Native SAM3: CLIP from the SAME CheckpointLoaderSimple."}),
                "sam3_model": ("SAM3_MODEL_CONFIG", {
                    "tooltip": "Alternative: Load SAM3 Model of the comfyui-sam3 extension. "
                               "Used only when no native model is wired."}),
                "aux_prompt": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional SAM3 prompt for aux_masks (mask_type 'aux' in the "
                               "PersonDetailer), e.g. 'hand'. Each hit goes to the person it "
                               "overlaps most. Empty = no aux data."}),
                "aux_threshold": ("FLOAT", {
                    "default": 0.30, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "SAM3 confidence for aux_prompt."}),
            },
        }

    def build(self, images, moss_boxes, slots=DEFAULT_SLOTS, threshold=0.4, min_overlap=0.1,
              depth_sort_order="front_last", sam3_native_model=None, sam3_native_clip=None,
              sam3_model=None, aux_prompt="", aux_threshold=0.30):
        if sam3_native_model is not None and sam3_native_clip is not None:
            sam3_model = NativeSAM3(sam3_native_model, sam3_native_clip)
        if images.shape[-1] > 3:
            images = images[..., :3]
        batch_size, h, w = images.shape[0], images.shape[1], images.shape[2]
        aux_prompt = (aux_prompt or "").strip()
        use_aux = bool(aux_prompt)

        slot_names, lines = resolve_slots(slots, moss_boxes)
        num_refs = len(slot_names)
        boxes = slot_boxes(moss_boxes, slot_names)
        slot_rects = [list(rect_pixels(b["rect"], w, h)) if b else None for b in boxes]
        captions = [(b.get("caption") or "").strip() if b else "" for b in boxes]

        cw, ch = (moss_boxes or {}).get("width"), (moss_boxes or {}).get("height")
        if cw and ch and abs((w / h) - (cw / ch)) > 0.02 * (cw / ch):
            lines.append(f"WARN: image {w}x{h} has another aspect than the layout canvas "
                         f"{cw}x{ch} — the boxes are stretched onto the image.")
        lines.append(f"slots: " + (", ".join(f"reference_{i + 1}={s}" for i, s in enumerate(slot_names))
                                   or "none"))
        if sam3_model is None:
            lines.append("ERROR: no SAM3 wired — connect sam3_native_model + sam3_native_clip "
                         "or sam3_model. PERSON_DATA carries no persons.")

        ref_masks = {mt: [[] for _ in range(num_refs)] for mt in MASK_TYPES + ["aux"]}
        all_faces, matched_faces = [], []
        per_face_all, face_to_ref_all, matches_all, depths_all = [], [], [], []
        aux_counts_all, aux_unassigned_all = [], []
        previews = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for b in range(batch_size):
            rgb = tensor2np(images[b:b + 1])
            persons, aux_loose, err = self._detect(sam3_model, rgb, threshold, device,
                                                   aux_prompt if use_aux else "", aux_threshold)
            assignment, scores = assign_persons_to_slots(
                [p["bbox"] for p in persons], slot_rects, min_overlap)
            p_to_ref = {pi: si for si, (pi, _s) in assignment.items()}

            for ri in range(num_refs):
                pi = assignment[ri][0] if ri in assignment else None
                for mt in MASK_TYPES + ["aux"]:
                    ref_masks[mt][ri].append(persons[pi]["masks"][mt] if pi is not None
                                             else empty_mask(h, w))

            per_face_all.append([p["masks"] for p in persons])
            face_to_ref_all.append([p_to_ref.get(pi) for pi in range(len(persons))])
            matches_all.append([ri in assignment for ri in range(num_refs)])
            all_faces.append(self._union([p["masks"]["face"] for p in persons], h, w))
            matched_faces.append(self._union([persons[pi]["masks"]["face"] for pi in p_to_ref], h, w))
            depths_all.append({ri: persons[pi]["depth"] for ri, (pi, _s) in assignment.items()})

            if use_aux:
                counts = {ri: 0 for ri in range(num_refs)}
                loose = aux_loose
                for pi, p in enumerate(persons):
                    if pi in p_to_ref:
                        counts[p_to_ref[pi]] += p["aux_count"]
                    elif p["aux_count"]:
                        # aux of a person without a slot is nobody's — as in the selector
                        loose = np.maximum(loose, p["masks"]["aux"][0].cpu().numpy())
                aux_counts_all.append(counts)
                aux_unassigned_all.append(torch.from_numpy(loose).unsqueeze(0))

            lines.extend(self._report_image(b, batch_size, persons, slot_names, boxes,
                                            slot_rects, assignment, scores, err))
            previews.append(self._draw_preview(rgb, persons, slot_names, slot_rects,
                                               assignment, p_to_ref))

        person_data = {
            "batch_size": batch_size,
            "num_references": num_refs,
            "image_height": h,
            "image_width": w,
            "matches": matches_all,
            "all_faces_mask": torch.cat(all_faces, dim=0),
            "matched_faces_mask": torch.cat(matched_faces, dim=0),
            "per_face_masks": per_face_all,
            "face_to_ref": face_to_ref_all,
            "ref_depths": depths_all,
            "depth_sort_order": depth_sort_order,
        }
        for mt in MASK_TYPES:
            person_data[f"{mt}_masks"] = [torch.cat(ref_masks[mt][ri], dim=0)
                                          for ri in range(num_refs)]
        if use_aux:
            person_data["aux_masks"] = [torch.cat(ref_masks["aux"][ri], dim=0)
                                        for ri in range(num_refs)]
            person_data["aux_part_counts"] = aux_counts_all
            person_data["aux_unassigned_masks"] = torch.cat(aux_unassigned_all, dim=0)
            lines.append(f"aux '{aux_prompt}': " + " | ".join(
                ", ".join(f"R{ri + 1}={c}" for ri, c in counts.items()) or "-"
                for counts in aux_counts_all))

        prompts = tuple(captions[i] if i < num_refs else "" for i in range(MAX_REFERENCES))
        preview = torch.cat(previews, dim=0)
        return (person_data,) + prompts + (preview, "\n".join(lines))

    # ── SAM3 ──

    def _detect(self, sam3_model, rgb, threshold, device, aux_prompt, aux_threshold):
        """Persons of one image: ``[{bbox, score, masks{mt: [1,H,W]}, depth, aux_count}]``.

        Returns ``(persons, aux_loose, error_text_or_None)``; ``aux_loose`` is
        the [H, W] union of aux hits that overlap no person.
        """
        h, w = rgb.shape[:2]
        aux_loose = np.zeros((h, w), dtype=np.float32)
        if sam3_model is None:
            return [], aux_loose, None
        processor, state = sam3_prepare(sam3_model, rgb)
        if processor is None:
            return [], aux_loose, sam3_last_error() or "SAM3 did not load — see console"

        bodies = dedupe_persons(sam3_ground(processor, state, rgb.shape, "person",
                                            threshold=threshold))
        body_map = {pi: r[0] for pi, r in enumerate(bodies)}

        found = {}
        for mt in ("face", "head", "hair"):
            prompt, thr = SAM3_MASK_CONFIG[mt]
            res = sam3_ground(processor, state, rgb.shape, prompt, threshold=thr) if bodies else []
            found[mt] = (res, assign_masks_by_body_overlap(res, body_map, []) if res else {})

        aux_res, aux_assign = [], {}
        if aux_prompt and bodies:
            aux_res = sam3_ground(processor, state, rgb.shape, aux_prompt, threshold=aux_threshold)
            aux_assign = PersonSelectorSAM3._assign_aux_multi(aux_res, body_map) if aux_res else {}
        elif aux_prompt:
            aux_res = sam3_ground(processor, state, rgb.shape, aux_prompt, threshold=aux_threshold)
        used = {mi for mis in aux_assign.values() for mi in mis}
        for mi, (m, _s, _b) in enumerate(aux_res):
            if mi not in used:
                aux_loose = np.maximum(aux_loose, (m > 0.5).astype(np.float32))

        persons = []
        for pi, (body_np, score, bbox) in enumerate(bodies):
            np_masks = {"body": body_np.astype(np.float32)}
            for mt in ("face", "head", "hair"):
                res, assign = found[mt]
                np_masks[mt] = (res[assign[pi]][0].astype(np.float32) if pi in assign
                                else np.zeros((h, w), dtype=np.float32))
            # Fallbacks as in Person Selector SAM3, minus InsightFace:
            # head = face ∪ hair, face = head.
            if np_masks["head"].max() <= 0.5:
                np_masks["head"] = np.maximum(np_masks["face"], np_masks["hair"])
            if np_masks["face"].max() <= 0.5:
                np_masks["face"] = np_masks["head"].copy()

            face_box = _mask_box(np_masks["face"])
            subtypes = None
            if face_box is not None:
                try:
                    subtypes = _bisenet_subtypes(rgb, face_box, np_masks["body"], device)
                except Exception as e:
                    if not FVM_MOSS_BoxPersonData._bisenet_warned:
                        print(f"[FVM_MOSS_BoxPersonData] BiSeNet subtype masks unavailable ({e}) — "
                              f"{'/'.join(BISENET_SUBTYPES)} stay empty")
                        FVM_MOSS_BoxPersonData._bisenet_warned = True
            for mt in BISENET_SUBTYPES:
                np_masks[mt] = (subtypes[mt] if subtypes is not None
                                else np.zeros((h, w), dtype=np.float32))

            aux_np = np.zeros((h, w), dtype=np.float32)
            for mi in aux_assign.get(pi, []):
                aux_np = np.maximum(aux_np, aux_res[mi][0].astype(np.float32))

            masks = {mt: mask2tensor(np_masks[mt]) for mt in MASK_TYPES}
            masks["aux"] = mask2tensor(aux_np)
            # Depth proxy like the selector without a depth map: face centre y / h.
            anchor = face_box or [float(v) for v in bbox]
            persons.append({
                "bbox": [float(v) for v in bbox], "score": float(score), "masks": masks,
                "depth": ((anchor[1] + anchor[3]) / 2) / h,
                "aux_count": len(aux_assign.get(pi, [])),
            })
        return persons, aux_loose, sam3_last_error()

    _bisenet_warned = False

    @staticmethod
    def _union(masks, h, w):
        if not masks:
            return empty_mask(h, w)
        return torch.max(torch.cat(masks, dim=0), dim=0, keepdim=True)[0]

    # ── Report / preview ──

    @staticmethod
    def _report_image(b, batch_size, persons, slot_names, boxes, slot_rects, assignment,
                      scores, err):
        out = [f"[Image {b + 1}/{batch_size}] {len(persons)} person(s), "
               f"{len(assignment)}/{len(slot_names)} slot(s) found"]
        if err:
            out.append(f"  ERROR SAM3: {err}")
        for ri, name in enumerate(slot_names):
            ref = f"  reference_{ri + 1} ({name})"
            if boxes[ri] is None:
                out.append(f"{ref}: slot missing in moss_boxes")
            elif ri in assignment:
                pi, iou = assignment[ri]
                inside = box_inside(persons[pi]["bbox"], slot_rects[ri])
                out.append(f"{ref}: found P{pi + 1} (IoU {iou:.2f}, {inside:.0%} of the person "
                           f"inside the box)")
            else:
                best = float(scores[:, ri].max()) if len(persons) else 0.0
                out.append(f"{ref}: not found (best IoU {best:.2f})")
        taken = {pi for pi, _ in assignment.values()}
        for pi in range(len(persons)):
            if pi not in taken:
                best = float(scores[pi].max()) if scores.shape[1] else 0.0
                out.append(f"  P{pi + 1}: unassigned -> generic (best IoU {best:.2f})")
        return out

    @staticmethod
    def _draw_preview(rgb, persons, slot_names, slot_rects, assignment, p_to_ref):
        """Slot boxes, persons tinted in their reference colour (grey = unassigned)."""
        img = rgb.astype(np.float32).copy()
        h, w = rgb.shape[:2]
        for pi, p in enumerate(persons):
            ri = p_to_ref.get(pi)
            color = _PREVIEW_COLORS[ri % len(_PREVIEW_COLORS)] if ri is not None else _UNMATCHED_COLOR
            m = p["masks"]["body"][0].cpu().numpy() > 0.5
            img[m] = img[m] * 0.6 + np.array(color, dtype=np.float32) * 0.4
        img = img.clip(0, 255).astype(np.uint8)

        thick = max(2, int(2 * max(h, w) / 1000))
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, max(h, w) / 1000.0 * 0.7)
        for pi, p in enumerate(persons):
            ri = p_to_ref.get(pi)
            color = _PREVIEW_COLORS[ri % len(_PREVIEW_COLORS)] if ri is not None else _UNMATCHED_COLOR
            m = (p["masks"]["body"][0].cpu().numpy() > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, thick)
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            tag = f"P{pi + 1}" if ri is None else f"P{pi + 1}->R{ri + 1} {assignment[ri][1]:.2f}"
            _label(img, tag, (max(0, x1) + 4, min(h - 4, max(0, y2) - 6)), font, scale * 0.8, color)

        for ri, rect in enumerate(slot_rects):
            if rect is None:
                continue
            color = _PREVIEW_COLORS[ri % len(_PREVIEW_COLORS)]
            x1, y1, x2, y2 = rect
            _dashed_rect(img, (x1, y1), (min(w - 1, x2), min(h - 1, y2)), color, thick)
            tag = f"R{ri + 1} {slot_names[ri]}" + ("" if ri in assignment else " (empty)")
            _label(img, tag, (x1 + 4, y1 + int(26 * scale) + 4), font, scale, color)
        return np2tensor(img)


def _label(img, text, org, font, scale, color):
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 4), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def _dashed_rect(img, p1, p2, color, thick, dash=12):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thick)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thick)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thick)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thick)
