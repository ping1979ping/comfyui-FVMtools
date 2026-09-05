"""PersonDataFilter — split PERSON_DATA into a filtered and a remaining set.

Takes the PERSON_DATA of a Person Selector (SAM3 / Multi) and splits it into two
independent PERSON_DATA outputs: the persons that match the criteria, and all the
others. Both outputs are re-indexed as fresh reference slots, so either one can be
fed straight into a PersonDetailer.

Selection happens in three stages:

1. **Gate** — a content filter. Either ``gender`` (female/male) or, when
   ``sam3_prompt`` is non-empty, a free SAM3 text prompt. The prompt always wins
   over the gender widget. Gender is resolved by SAM3 grounding ("woman"/"man")
   against each person's body mask, with InsightFace's genderage classifier as
   the fallback for every person SAM3 could not label.

2. **Rank** — sort the survivors by ``sort_by`` in ``order`` direction.

3. **Cut** — keep either the first ``count`` (mode ``top_n``) or exactly the
   ``count``-th one (mode ``nth``). Everything else goes to ``remaining``.
"""

import numpy as np
import torch
import cv2

from .utils.face_analyzer import FaceAnalyzer
from .utils.masker import sam3_prepare, sam3_ground
from .utils.tensor_utils import tensor2np, empty_mask, np2tensor

# Sort criteria. "ascending" is always the natural reading direction:
#   area       -> smallest first
#   horizontal -> left to right
#   vertical   -> top to bottom
#   depth      -> far to near (needs a depth_map on the selector)
#   index      -> original detection / slot order
SORT_CRITERIA = ["area", "horizontal", "vertical", "depth", "index"]

GENDER_PROMPTS = {"female": "woman", "male": "man"}

_KEEP_COLOR = (60, 220, 90)
_DROP_COLOR = (230, 70, 70)


def _to_np2d(mask):
    """Tensor [1,H,W] or [H,W] -> float32 numpy [H,W]."""
    if mask is None:
        return None
    m = mask
    if m.dim() == 3:
        m = m[0]
    return m.detach().cpu().numpy().astype(np.float32)


def _primary_mask(cand):
    """Body mask if present, otherwise head, otherwise face — as numpy [H,W]."""
    for mt in ("body", "head", "face"):
        m = cand["masks"].get(mt)
        if m is None:
            continue
        npm = _to_np2d(m)
        if npm is not None and npm.max() > 0.5:
            return npm
    return None


def _geometry(cand, h, w):
    """Area in pixels and centroid (cx, cy) of the candidate's primary mask."""
    m = _primary_mask(cand)
    if m is None:
        return 0, w / 2.0, h / 2.0
    ys, xs = np.where(m > 0.5)
    if len(ys) == 0:
        return 0, w / 2.0, h / 2.0
    return int(len(ys)), float(xs.mean()), float(ys.mean())


class PersonDataFilter:
    """Split PERSON_DATA into filtered + remaining by gender / free text / rank."""

    _face_analyzer = None
    _last_det_size = None

    MAX_SLOTS = 32

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("PERSON_DATA", "PERSON_DATA", "MASK", "MASK", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("filtered", "remaining", "filtered_masks", "remaining_masks",
                    "preview", "filtered_count", "remaining_count", "report")
    OUTPUT_NODE = True

    DESCRIPTION = (
        "Split PERSON_DATA into a filtered and a remaining set.\n\n"
        "source:  all_detected = every person the selector found (works without\n"
        "         reference images); matched_refs = only the reference-matched slots.\n"
        "gate:    gender (female/male) via SAM3 'woman'/'man' grounding with\n"
        "         InsightFace fallback — or any free SAM3 text prompt, which\n"
        "         overrides the gender widget when non-empty.\n"
        "rank:    sort_by (area / horizontal / vertical / depth / index) where\n"
        "         ascending = smallest, leftmost, topmost, farthest, first.\n"
        "cut:     top_n keeps the first N, nth keeps exactly the N-th.\n\n"
        "Both outputs are re-indexed as fresh reference slots — feed either one\n"
        "directly into a PersonDetailer."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_data": ("PERSON_DATA", {"tooltip": "PERSON_DATA from Person Selector SAM3 / Multi / Data Refiner"}),
                "source": (["all_detected", "matched_refs"], {"default": "all_detected",
                    "tooltip": "all_detected: every person the selector found (per_face_masks).\n"
                               "  Works even with no reference image connected.\n"
                               "matched_refs: only the slots that matched a reference image."}),
                "gender": (["any", "female", "male"], {"default": "any",
                    "tooltip": "Content gate. 'any' disables it.\n"
                               "Ignored when sam3_prompt is filled."}),
                "sort_by": (SORT_CRITERIA, {"default": "area",
                    "tooltip": "Ranking criterion:\n"
                               "  area        - body mask pixel count\n"
                               "  horizontal  - body centroid X\n"
                               "  vertical    - body centroid Y\n"
                               "  depth       - needs depth_map on the selector\n"
                               "  index       - original detection / slot order"}),
                "order": (["descending", "ascending"], {"default": "descending",
                    "tooltip": "ascending = smallest / leftmost / topmost / farthest first.\n"
                               "descending = largest / rightmost / bottommost / nearest first."}),
                "mode": (["top_n", "nth"], {"default": "top_n",
                    "tooltip": "top_n: keep the first `count` after sorting (0 = keep all).\n"
                               "nth:   keep exactly the `count`-th one (1-based)."}),
                "count": ("INT", {"default": 1, "min": 0, "max": cls.MAX_SLOTS, "step": 1,
                    "tooltip": "top_n: how many persons to keep (0 = all that pass the gate).\n"
                               "nth:   which one to keep, 1-based."}),
                "sam3_prompt": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Free SAM3 text prompt, e.g. 'woman in red dress', 'child',\n"
                               "'person with backpack'. When non-empty it REPLACES the\n"
                               "gender criterion. Empty = the gender widget applies."}),
                "sam3_threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "SAM3 grounding confidence threshold for the gate."}),
                "min_overlap": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum fraction of a grounded mask that must fall inside a\n"
                               "person's body mask before that person counts as a hit."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "The image(s) the person_data was built from.\n"
                                                "Needed for the gender / free-text gate and the preview."}),
                "sam3_model": ("SAM3_MODEL_CONFIG", {"tooltip": "SAM3 model from LoadSAM3Model.\n"
                                                                "Needed for the free-text gate and SAM3-based gender."}),
                "det_size": (["320", "480", "640", "768"], {"default": "640",
                    "tooltip": "Face detection resolution for the InsightFace gender fallback."}),
                "invert": ("BOOLEAN", {"default": False,
                    "tooltip": "Swap the two outputs — filtered becomes remaining and vice versa."}),
            },
        }

    # ── Structure helpers ──

    @staticmethod
    def _mask_types(pd):
        """Discover the mask types a PERSON_DATA dict carries ('face', 'body', ... 'aux').

        Only per-slot lists of [B,H,W] tensors qualify — ``per_face_masks``
        (a list of per-face dicts) and ``aux_unassigned_masks`` (a bare tensor)
        also end in ``_masks`` but are not slot lists.
        """
        types = []
        for key, value in pd.items():
            if not key.endswith("_masks") or not isinstance(value, list):
                continue
            mt = key[: -len("_masks")]
            if mt in ("aux_unassigned", "per_face"):
                continue
            if value and not all(isinstance(v, torch.Tensor) for v in value):
                continue
            types.append(mt)
        # Fallback for a PERSON_DATA that only carries per-face dicts.
        if not types:
            seen = set()
            for faces in (pd.get("per_face_masks") or []):
                for md in faces:
                    seen.update(md.keys())
            types = list(seen)

        preferred = ["face", "head", "body", "hair", "facial_skin", "eyes",
                     "mouth", "neck", "accessories", "aux"]
        ordered = [mt for mt in preferred if mt in types]
        ordered += [mt for mt in types if mt not in preferred]
        return ordered

    @staticmethod
    def _collect(pd, b, source, mask_types, h, w):
        """Build the candidate list for batch image `b`."""
        cands = []
        if source == "matched_refs":
            num_refs = int(pd.get("num_references", 0))
            all_matches = pd.get("matches") or []
            matches_b = all_matches[b] if b < len(all_matches) else []
            depths_all = pd.get("ref_depths") or []
            depths_b = depths_all[b] if b < len(depths_all) else {}
            counts_all = pd.get("aux_part_counts") or []
            counts_b = counts_all[b] if b < len(counts_all) else {}
            for ri in range(num_refs):
                if ri < len(matches_b) and not matches_b[ri]:
                    continue
                masks = {}
                for mt in mask_types:
                    lst = pd.get(f"{mt}_masks")
                    if lst and ri < len(lst) and b < lst[ri].shape[0]:
                        masks[mt] = lst[ri][b:b + 1]
                cands.append({
                    "masks": masks,
                    "index": ri,
                    "label": f"Ref{ri + 1}",
                    "depth": float(depths_b.get(ri, 0.5)) if isinstance(depths_b, dict) else 0.5,
                    "aux_count": int(counts_b.get(ri, 0)) if isinstance(counts_b, dict) else 0,
                })
        else:
            pfm = pd.get("per_face_masks") or []
            faces = pfm[b] if b < len(pfm) else []
            f2r_all = pd.get("face_to_ref") or []
            f2r_b = f2r_all[b] if b < len(f2r_all) else []
            depths_all = pd.get("ref_depths") or []
            depths_b = depths_all[b] if b < len(depths_all) else {}
            for fi, md in enumerate(faces):
                ri = f2r_b[fi] if fi < len(f2r_b) else None
                masks = {mt: md[mt] for mt in mask_types if md.get(mt) is not None}
                depth = 0.5
                if ri is not None and isinstance(depths_b, dict):
                    depth = float(depths_b.get(ri, 0.5))
                aux = masks.get("aux")
                cands.append({
                    "masks": masks,
                    "index": fi,
                    "label": f"R{ri + 1}" if ri is not None else f"P{fi + 1}",
                    "depth": depth,
                    "aux_count": 1 if (aux is not None and float(aux.max()) > 0.5) else 0,
                })
        return cands

    @staticmethod
    def _build_person_data(src, per_batch, mask_types, h, w):
        """Assemble a fresh PERSON_DATA from per-batch candidate lists."""
        batch_size = len(per_batch)
        num_refs = max((len(c) for c in per_batch), default=0)

        out = {
            "batch_size": batch_size,
            "num_references": num_refs,
            "image_height": h,
            "image_width": w,
            "depth_sort_order": src.get("depth_sort_order", "front_last"),
        }

        matches, per_face, face_to_ref, ref_depths, all_faces, aux_counts = [], [], [], [], [], []
        for b in range(batch_size):
            cands = per_batch[b]
            matches.append([ri < len(cands) for ri in range(num_refs)])
            per_face.append([c["masks"] for c in cands])
            face_to_ref.append(list(range(len(cands))))
            ref_depths.append({ri: cands[ri]["depth"] for ri in range(len(cands))})
            aux_counts.append({ri: cands[ri]["aux_count"] for ri in range(len(cands))})
            face_parts = [c["masks"]["face"] for c in cands if c["masks"].get("face") is not None]
            if face_parts:
                all_faces.append(torch.max(torch.cat(face_parts, dim=0), dim=0, keepdim=True)[0])
            else:
                all_faces.append(empty_mask(h, w))

        out["matches"] = matches
        out["per_face_masks"] = per_face
        out["face_to_ref"] = face_to_ref
        out["ref_depths"] = ref_depths
        out["all_faces_mask"] = torch.cat(all_faces, dim=0)
        out["matched_faces_mask"] = out["all_faces_mask"].clone()

        for mt in mask_types:
            per_ref = []
            for ri in range(num_refs):
                parts = []
                for b in range(batch_size):
                    cands = per_batch[b]
                    m = cands[ri]["masks"].get(mt) if ri < len(cands) else None
                    parts.append(m if m is not None else empty_mask(h, w))
                per_ref.append(torch.cat(parts, dim=0))
            out[f"{mt}_masks"] = per_ref

        if "aux" in mask_types:
            out["aux_part_counts"] = aux_counts
            unassigned = src.get("aux_unassigned_masks")
            out["aux_unassigned_masks"] = (unassigned if unassigned is not None
                                           else torch.zeros(batch_size, h, w, dtype=torch.float32))
        return out

    # ── Gate helpers ──

    def _sam3_assign(self, processor, base_state, image_rgb, cands, prompts,
                     threshold, min_overlap):
        """Ground each prompt and assign its masks to candidates by body overlap.

        Returns {cand_idx: (label, score)} — the best-scoring label per candidate.
        """
        if processor is None or base_state is None or not cands:
            return {}

        bodies = []
        for c in cands:
            m = _primary_mask(c)
            bodies.append(None if m is None else (m > 0.5))

        scores = {}  # cand_idx -> {label: accumulated score}
        for label, prompt in prompts.items():
            results = sam3_ground(processor, base_state, image_rgb.shape, prompt, threshold)
            for mask_np, score, _bbox in results:
                mb = mask_np > 0.5
                mask_area = int(mb.sum())
                if mask_area == 0:
                    continue
                best_ci, best_ratio = None, 0.0
                for ci, body in enumerate(bodies):
                    if body is None:
                        continue
                    inter = int(np.logical_and(mb, body).sum())
                    if inter == 0:
                        continue
                    ratio = inter / float(mask_area)
                    if ratio > best_ratio:
                        best_ci, best_ratio = ci, ratio
                if best_ci is None or best_ratio < min_overlap:
                    continue
                bucket = scores.setdefault(best_ci, {})
                bucket[label] = bucket.get(label, 0.0) + float(score) * best_ratio

        return {ci: max(bucket.items(), key=lambda kv: kv[1])
                for ci, bucket in scores.items() if bucket}

    def _insightface_gender(self, image_rgb, cands, det_size_int):
        """Fallback gender per candidate via InsightFace genderage.

        Returns {cand_idx: 'female'|'male'}.
        """
        if (PersonDataFilter._face_analyzer is None
                or PersonDataFilter._last_det_size != det_size_int):
            PersonDataFilter._face_analyzer = FaceAnalyzer(det_size_int)
            PersonDataFilter._last_det_size = det_size_int
        analyzer = PersonDataFilter._face_analyzer

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = analyzer.detect_faces(bgr)
        if not faces:
            return {}

        h, w = image_rgb.shape[:2]
        out = {}
        for face in faces:
            sex = getattr(face, "sex", None)
            if sex is None:
                g = getattr(face, "gender", None)
                sex = None if g is None else ("M" if int(g) == 1 else "F")
            if sex is None:
                continue
            label = "male" if str(sex).upper().startswith("M") else "female"

            x1, y1, x2, y2 = face.bbox
            cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
            cy = int(np.clip((y1 + y2) / 2, 0, h - 1))

            # Prefer the person whose face mask covers the face center; fall back
            # to head/body, then to the nearest centroid.
            hit = None
            for key in ("face", "head", "body"):
                for ci, c in enumerate(cands):
                    m = c["masks"].get(key)
                    if m is None:
                        continue
                    npm = _to_np2d(m)
                    if npm is not None and npm[cy, cx] > 0.5:
                        hit = ci
                        break
                if hit is not None:
                    break
            if hit is None:
                best_d = None
                for ci, c in enumerate(cands):
                    _a, ccx, ccy = _geometry(c, h, w)
                    d = (ccx - cx) ** 2 + (ccy - cy) ** 2
                    if best_d is None or d < best_d:
                        best_d, hit = d, ci
            if hit is not None and hit not in out:
                out[hit] = label
        return out

    # ── Preview ──

    @staticmethod
    def _render_preview(image_tensor, cands, kept_idx, notes):
        """Overlay kept persons in green, dropped ones in red, with labels."""
        rgb = tensor2np(image_tensor).copy()
        h, w = rgb.shape[:2]
        overlay = rgb.copy()

        for ci, c in enumerate(cands):
            m = _primary_mask(c)
            if m is None:
                continue
            color = _KEEP_COLOR if ci in kept_idx else _DROP_COLOR
            binm = (m > 0.5).astype(np.uint8)
            overlay[binm > 0] = color
            contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, color, max(2, h // 400))

        rgb = cv2.addWeighted(overlay, 0.25, rgb, 0.75, 0)

        scale = max(0.5, min(w, h) / 900.0)
        for ci, c in enumerate(cands):
            _area, cx, cy = _geometry(c, h, w)
            color = _KEEP_COLOR if ci in kept_idx else _DROP_COLOR
            text = notes.get(ci, c["label"])
            org = (int(np.clip(cx - 60, 4, max(5, w - 10))), int(np.clip(cy, 20, max(21, h - 8))))
            cv2.putText(rgb, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                        max(3, int(4 * scale)), cv2.LINE_AA)
            cv2.putText(rgb, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        max(1, int(2 * scale)), cv2.LINE_AA)
        return np2tensor(rgb)

    # ── Main ──

    def execute(self, person_data, source, gender, sort_by, order, mode, count,
                sam3_prompt="", sam3_threshold=0.30, min_overlap=0.15,
                images=None, sam3_model=None, det_size="640", invert=False):
        pd = person_data or {}
        h = int(pd.get("image_height", 0)) or (int(images.shape[1]) if images is not None else 64)
        w = int(pd.get("image_width", 0)) or (int(images.shape[2]) if images is not None else 64)
        batch_size = int(pd.get("batch_size", 1)) or 1
        det_size_int = int(det_size)

        mask_types = self._mask_types(pd)
        prompt = (sam3_prompt or "").strip()
        use_prompt = bool(prompt)
        want_gate = use_prompt or gender != "any"

        warnings = []
        if want_gate and images is None:
            warnings.append("no `images` connected - gate skipped (rank/cut still applied)")
        if use_prompt and sam3_model is None:
            warnings.append("free-text prompt needs `sam3_model` - gate skipped")
        if (not use_prompt) and gender != "any" and sam3_model is None:
            warnings.append("no `sam3_model` - gender resolved by InsightFace only")

        kept_batches, dropped_batches, report_lines = [], [], []
        preview = torch.zeros(1, 64, 64, 3, dtype=torch.float32)

        for b in range(batch_size):
            cands = self._collect(pd, b, source, mask_types, h, w)
            image_rgb = None
            if images is not None and b < images.shape[0]:
                image_rgb = tensor2np(images[b:b + 1])

            # ── Stage 1: gate ──
            gate_pass = list(range(len(cands)))
            labels = {}
            gate_desc = "off"

            if want_gate and cands and image_rgb is not None:
                sam_labels = {}
                if sam3_model is not None:
                    processor, base_state = sam3_prepare(sam3_model, image_rgb)
                    prompts = {"match": prompt} if use_prompt else dict(GENDER_PROMPTS)
                    sam_labels = self._sam3_assign(processor, base_state, image_rgb, cands,
                                                   prompts, sam3_threshold, min_overlap)

                if use_prompt:
                    if sam3_model is not None:
                        gate_pass = sorted(sam_labels.keys())
                        labels = {ci: f"'{prompt}'" for ci in gate_pass}
                    gate_desc = f"prompt '{prompt}'"
                else:
                    for ci, (lbl, _score) in sam_labels.items():
                        labels[ci] = lbl
                    missing = [ci for ci in range(len(cands)) if ci not in labels]
                    if missing:
                        fallback = self._insightface_gender(image_rgb, cands, det_size_int)
                        for ci in missing:
                            if ci in fallback:
                                labels[ci] = fallback[ci] + "*"
                    gate_pass = [ci for ci in range(len(cands))
                                 if labels.get(ci, "").rstrip("*") == gender]
                    gate_desc = f"gender {gender}"

            # ── Stage 2: rank ──
            def metric(ci, _cands=cands):
                c = _cands[ci]
                area, cx, cy = _geometry(c, h, w)
                if sort_by == "area":
                    return area
                if sort_by == "horizontal":
                    return cx
                if sort_by == "vertical":
                    return cy
                if sort_by == "depth":
                    return c["depth"]
                return c["index"]

            ranked = sorted(gate_pass, key=metric, reverse=(order == "descending"))

            # ── Stage 3: cut ──
            if mode == "nth":
                n = max(1, int(count))
                kept = [ranked[n - 1]] if n <= len(ranked) else []
            else:
                kept = list(ranked) if int(count) <= 0 else ranked[:int(count)]

            kept_set = set(kept)
            dropped = [ci for ci in range(len(cands)) if ci not in kept_set]

            if invert:
                kept, dropped = dropped, kept
                kept_set = set(kept)

            kept_batches.append([cands[ci] for ci in kept])
            dropped_batches.append([cands[ci] for ci in dropped])

            # ── Report + preview ──
            notes = {}
            rank_pos = {ci: i + 1 for i, ci in enumerate(ranked)}
            lines = [f"[Image {b + 1}/{batch_size}] {len(cands)} persons | gate: {gate_desc} "
                     f"| keep {len(kept)} / drop {len(dropped)}"]
            if cands:
                lines.append("Person | Label | Area | X | Y | Rank | Result")
                lines.append("--- | --- | --- | --- | --- | --- | ---")
            for ci, c in enumerate(cands):
                area, cx, cy = _geometry(c, h, w)
                lbl = labels.get(ci, "-")
                rk = str(rank_pos[ci]) if ci in rank_pos else "-"
                res = "**KEEP**" if ci in kept_set else "drop"
                lines.append(f"{c['label']} | {lbl} | {area} | {int(cx)} | {int(cy)} | {rk} | {res}")
                notes[ci] = " ".join(x for x in (c["label"], lbl if lbl != "-" else "", f"#{rk}") if x)
            report_lines.append("\n".join(lines))

            if b == 0 and image_rgb is not None:
                preview = self._render_preview(images[b:b + 1], cands, kept_set, notes)

        filtered = self._build_person_data(pd, kept_batches, mask_types, h, w)
        remaining = self._build_person_data(pd, dropped_batches, mask_types, h, w)

        def _stack_body(out_pd):
            lst = out_pd.get("body_masks") or out_pd.get("face_masks")
            if not lst:
                return empty_mask(h, w)
            return torch.cat(lst, dim=0)

        filtered_count = filtered["num_references"]
        remaining_count = remaining["num_references"]

        gate_str = "prompt" if use_prompt else (f"gender {gender}" if gender != "any" else "off")
        header = [
            "## PersonDataFilter",
            f"**{filtered_count}** filtered | **{remaining_count}** remaining | source: {source}",
            "",
            f"Gate: {gate_str} | Sort: {sort_by} {order} | Cut: {mode}={count}"
            + (" | inverted" if invert else ""),
        ]
        if warnings:
            header += ["", "> " + " | ".join(warnings)]
        header += ["", "---", ""]
        report = "\n".join(header + report_lines)
        print(f"[PersonDataFilter] {filtered_count} filtered / {remaining_count} remaining "
              f"({source}, {sort_by} {order}, {mode}={count})")

        return (filtered, remaining, _stack_body(filtered), _stack_body(remaining),
                preview, filtered_count, remaining_count, report)
