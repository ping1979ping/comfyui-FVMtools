import torch
import numpy as np
import cv2

from .utils.face_analyzer import FaceAnalyzer
from .utils.matcher import compute_similarity, aggregate_similarities
from .utils.masker import MaskGenerator, FACE_LABELS, HEAD_LABELS, MASK_TYPE_LABELS, BISENET_MASK_TYPES
from .utils.tensor_utils import tensor2np, tensor2cv2, mask2tensor, np2tensor, empty_mask, apply_gaussian_blur, fill_mask_holes
from .utils.mask_utils import clean_mask_crumbs
from .utils.segs_utils import assign_segs_to_references, run_detector

# Colors for preview overlay (RGB), one per reference
_PREVIEW_COLORS = [
    (255, 50, 50), (50, 255, 50), (50, 100, 255), (255, 255, 50), (255, 50, 255),
    (50, 255, 255), (255, 160, 30), (160, 50, 255), (50, 255, 160), (255, 160, 160),
]


class PersonSelectorMulti:
    """ComfyUI node that matches multiple reference persons and generates per-person masks.
    Supports batch input for the PersonDetailer pipeline."""

    # Shared class-level cache with PersonSelector
    _face_analyzer = None
    _last_det_size = None

    MAX_REFERENCES = 10
    AUTO_FLOOR = 0.10

    DESCRIPTION = (
        "Matches multiple reference persons in the current image using InsightFace (ArcFace) embeddings.\n\n"
        "Each reference input accepts a batch of images for one person.\n"
        "Connect a reference to auto-create the next input slot.\n\n"
        "Supports batch input: pass multiple images and get PERSON_DATA for PersonDetailer.\n\n"
        "Faces are assigned exclusively: each face matches at most one reference.\n\n"
        "Optional: connect SEGS or a detector for body-part detection.\n"
        "Detected parts are assigned to references via body mask overlap\n"
        "and stored as 'aux' masks in PERSON_DATA for PersonDetailer."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ("SAM_MODEL", {"tooltip": "SAM model from Impact Pack SAMLoader — required for body masks"}),
                "current_image": ("IMAGE",),
                "auto_threshold": ("BOOLEAN", {"default": True,
                                               "tooltip": "Auto: finds optimal 1:1 face-reference assignment (ignores threshold). Off: uses manual threshold."}),
                "threshold": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Minimum cosine similarity to count as match. Ignored when auto_threshold is on."}),
                "aggregation": (["max", "mean", "min"],
                                {"tooltip": "How to combine similarity scores across multiple reference images of the same person"}),
                "mask_fill_holes": ("BOOLEAN", {"default": True,
                                               "tooltip": "Fill holes inside the mask (closes gaps in segmentation)"}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100,
                                      "tooltip": "Gaussian blur radius for mask edges"}),
                "det_size": (["320", "480", "640", "768"],
                             {"tooltip": "Face detection resolution — higher finds smaller faces but uses more VRAM"}),
                "aux_mask_type": (["none", "hair", "facial_skin", "eyes", "mouth", "neck", "accessories"],
                                  {"default": "none", "tooltip": "Additional mask type for aux_masks output (derived from BiSeNet, no extra cost)"}),
                "detect_threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                                                "tooltip": "Detection confidence threshold for body-part detector (only used when detector/SEGS connected)"}),
                "detect_dilation": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1,
                                             "tooltip": "Mask dilation in pixels for detected body parts"}),
                "detect_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1,
                                                   "tooltip": "Crop expansion factor passed to detector"}),
                "reference_1": ("IMAGE",),
            },
            "optional": {
                "segs": ("SEGS", {"tooltip": "Pre-computed body-part SEGS (e.g. from Impact Pack). Takes priority over detector inputs."}),
                "bbox_detector": ("BBOX_DETECTOR", {"tooltip": "Bounding box detector for body-part detection. Ignored if SEGS connected."}),
                "segm_detector": ("SEGM_DETECTOR", {"tooltip": "Segmentation detector (preferred over bbox). Ignored if SEGS connected."}),
                **{f"reference_{i}": ("IMAGE",) for i in range(2, cls.MAX_REFERENCES + 1)},
            },
        }

    RETURN_TYPES = ("PERSON_DATA", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "IMAGE", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("person_data", "face_masks", "head_masks", "body_masks",
                    "combined_face", "combined_head", "combined_body", "aux_masks",
                    "preview", "similarities", "matches", "matched_count", "face_count", "report")
    FUNCTION = "execute"
    CATEGORY = "FVM Tools/Face"
    OUTPUT_NODE = True

    def _collect_references(self, reference_1, **kwargs):
        refs = [reference_1]
        for i in range(2, self.MAX_REFERENCES + 1):
            key = f"reference_{i}"
            if key in kwargs and kwargs[key] is not None:
                refs.append(kwargs[key])
            else:
                break
        return refs

    def _extract_ref_embeddings(self, analyzer, ref_batch):
        embs = []
        for i in range(ref_batch.shape[0]):
            frame = ref_batch[i:i+1]
            bgr = tensor2cv2(frame)
            faces = analyzer.detect_faces(bgr)
            if faces:
                embs.append(analyzer.get_embedding(faces[0]))
        return embs

    def _build_similarity_matrix(self, ref_emb_sets, face_embs, aggregation):
        num_refs = len(ref_emb_sets)
        num_faces = len(face_embs)
        sim_matrix = np.zeros((num_refs, num_faces))
        for ri, ref_embs in enumerate(ref_emb_sets):
            if not ref_embs:
                continue
            for fi, face_emb in enumerate(face_embs):
                sims = [compute_similarity(face_emb, ref_emb) for ref_emb in ref_embs]
                sim_matrix[ri, fi] = aggregate_similarities(sims, aggregation)
        return sim_matrix

    def _assign_greedy(self, sim_matrix, threshold):
        num_refs, num_faces = sim_matrix.shape
        candidates = []
        for ri in range(num_refs):
            for fi in range(num_faces):
                if sim_matrix[ri, fi] >= threshold:
                    candidates.append((ri, fi, sim_matrix[ri, fi]))
        candidates.sort(key=lambda x: x[2], reverse=True)

        assigned_faces = set()
        assignments = {}
        for ri, fi, sim in candidates:
            if ri in assignments or fi in assigned_faces:
                continue
            assignments[ri] = (fi, sim)
            assigned_faces.add(fi)
        return assignments

    def _generate_all_masks(self, cur_rgb, face, device, sam_model, mask_fill_holes, mask_blur):
        """Generate all mask types from a single BiSeNet run + SAM body mask."""
        label_map = MaskGenerator._run_bisenet(cur_rgb, face, device)

        masks = {}

        # All BiSeNet-derived masks from the same label_map (nearly free)
        for mask_type, labels in MASK_TYPE_LABELS.items():
            mask_np = np.isin(label_map, list(labels)).astype(np.float32)
            mask = mask2tensor(mask_np)
            if mask_fill_holes:
                mask = fill_mask_holes(mask)
            if mask_blur > 0:
                mask = apply_gaussian_blur(mask, mask_blur)
            masks[mask_type] = mask

        # Body mask via SAM (separate from BiSeNet)
        body_mask_np = MaskGenerator.generate_body_mask(cur_rgb, face, sam_model)
        # Clean up crumbs/artifacts from SAM
        body_mask_np = clean_mask_crumbs(body_mask_np, min_area_fraction=0.005)
        mask = mask2tensor(body_mask_np)
        if mask_fill_holes:
            mask = fill_mask_holes(mask)
        if mask_blur > 0:
            mask = apply_gaussian_blur(mask, mask_blur)
        masks["body"] = mask

        return masks

    def _render_preview(self, current_image, assignments, cur_faces, body_masks_list, h, w):
        preview = tensor2np(current_image).copy()

        for ri, (fi, sim) in assignments.items():
            color = _PREVIEW_COLORS[ri % len(_PREVIEW_COLORS)]
            mask_np = (body_masks_list[ri][0].cpu().numpy() * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(preview, contours, -1, color, 2)

            face = cur_faces[fi]
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cx = (x1 + x2) // 2
            label = str(ri + 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.7, min(2.0, (x2 - x1) / 80.0))
            thickness = max(1, int(font_scale * 2))
            (tw, th_text), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            tx = cx - tw // 2
            ty = max(th_text + 4, y1 - 8)

            cv2.rectangle(preview, (tx - 4, ty - th_text - 4), (tx + tw + 4, ty + 4), (0, 0, 0), cv2.FILLED)
            cv2.putText(preview, label, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

        return np2tensor(preview)

    def _process_single_image(self, single_image, analyzer, ref_emb_sets, num_refs,
                               sam_model, aggregation, effective_threshold,
                               mask_fill_holes, mask_blur, device):
        """Process a single image and return per-ref masks, assignments, faces, etc."""
        h, w = single_image.shape[1], single_image.shape[2]

        cur_bgr = tensor2cv2(single_image)
        cur_rgb = tensor2np(single_image)
        cur_faces = analyzer.detect_faces(cur_bgr)
        face_count = len(cur_faces)
        face_embs = [analyzer.get_embedding(f) for f in cur_faces]

        print(f"[PersonSelectorMulti] Image {single_image.shape} → {face_count} faces detected")
        if face_count > 0:
            for fi, face in enumerate(cur_faces):
                bbox = [int(v) for v in face.bbox]
                area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                print(f"  face #{fi}: bbox={bbox} area={area}px")

        sim_matrix = self._build_similarity_matrix(ref_emb_sets, face_embs, aggregation)
        if sim_matrix.size > 0:
            import numpy as _np
            print(f"[PersonSelectorMulti] sim_matrix:\n{_np.array2string(sim_matrix, precision=4)}")
        assignments = self._assign_greedy(sim_matrix, effective_threshold)

        # Collect all mask types per reference
        from .utils.masker import ALL_MASK_TYPES
        masks_per_type = {mt: [] for mt in ALL_MASK_TYPES}

        for ri in range(num_refs):
            if ri in assignments:
                fi, sim = assignments[ri]
                masks = self._generate_all_masks(cur_rgb, cur_faces[fi], device, sam_model, mask_fill_holes, mask_blur)
                for mt in ALL_MASK_TYPES:
                    masks_per_type[mt].append(masks.get(mt, empty_mask(h, w)))
                print(f"[PersonSelectorMulti] ref {ri+1} → face #{fi} (sim={sim:.4f})")
            else:
                for mt in ALL_MASK_TYPES:
                    masks_per_type[mt].append(empty_mask(h, w))

        # all_faces_mask: OR of all detected faces (matched or not)
        if face_count > 0:
            all_face_mask_parts = []
            for face in cur_faces:
                fm = MaskGenerator.generate_face_mask(cur_rgb, face, device)
                all_face_mask_parts.append(mask2tensor(fm))
            all_faces_mask = torch.max(torch.cat(all_face_mask_parts, dim=0), dim=0, keepdim=True)[0]
        else:
            all_faces_mask = empty_mask(h, w)

        # matched_faces_mask: OR of matched face masks only
        matched_indices = set(fi for fi, sim in assignments.values())
        if matched_indices:
            matched_parts = []
            for fi in matched_indices:
                fm = MaskGenerator.generate_face_mask(cur_rgb, cur_faces[fi], device)
                matched_parts.append(mask2tensor(fm))
            matched_faces_mask = torch.max(torch.cat(matched_parts, dim=0), dim=0, keepdim=True)[0]
        else:
            matched_faces_mask = empty_mask(h, w)

        return {
            "masks_per_type": masks_per_type,
            "assignments": assignments,
            "cur_faces": cur_faces,
            "face_count": face_count,
            "sim_matrix": sim_matrix,
            "all_faces_mask": all_faces_mask,
            "matched_faces_mask": matched_faces_mask,
        }

    def execute(self, sam_model, current_image, reference_1, auto_threshold, threshold, aggregation,
                mask_fill_holes, mask_blur, det_size, aux_mask_type="none",
                detect_threshold=0.3, detect_dilation=10, detect_crop_factor=3.0,
                segs=None, bbox_detector=None, segm_detector=None, **kwargs):
        det_size_int = int(det_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if PersonSelectorMulti._face_analyzer is None or PersonSelectorMulti._last_det_size != det_size_int:
            PersonSelectorMulti._face_analyzer = FaceAnalyzer(det_size_int)
            PersonSelectorMulti._last_det_size = det_size_int
        analyzer = PersonSelectorMulti._face_analyzer

        batch_size = current_image.shape[0]
        h, w = current_image.shape[1], current_image.shape[2]
        refs = self._collect_references(reference_1, **kwargs)
        num_refs = len(refs)

        ref_emb_sets = [self._extract_ref_embeddings(analyzer, rb) for rb in refs]
        effective_threshold = self.AUTO_FLOOR if auto_threshold else threshold

        print(f"[PersonSelectorMulti] batch_size={batch_size}, refs={num_refs}, "
              f"auto_threshold={auto_threshold}, effective={effective_threshold}")

        # Process each image in the batch
        batch_results = []
        for b in range(batch_size):
            single = current_image[b:b+1]
            result = self._process_single_image(
                single, analyzer, ref_emb_sets, num_refs,
                sam_model, aggregation, effective_threshold,
                mask_fill_holes, mask_blur, device,
            )
            batch_results.append(result)

        # Build PERSON_DATA with all mask types
        from .utils.masker import ALL_MASK_TYPES
        person_data_masks = {mt: [] for mt in ALL_MASK_TYPES}  # per type: list of [B,H,W] per ref
        person_data_matches = []

        for ri in range(num_refs):
            ref_masks_per_type = {mt: [] for mt in ALL_MASK_TYPES}
            for b in range(batch_size):
                mpt = batch_results[b]["masks_per_type"]
                for mt in ALL_MASK_TYPES:
                    ref_masks_per_type[mt].append(mpt[mt][ri])
            for mt in ALL_MASK_TYPES:
                person_data_masks[mt].append(torch.cat(ref_masks_per_type[mt], dim=0))

        for b in range(batch_size):
            matches_for_image = [ri in batch_results[b]["assignments"] for ri in range(num_refs)]
            person_data_matches.append(matches_for_image)

        all_faces_masks = torch.cat([br["all_faces_mask"] for br in batch_results], dim=0)
        matched_faces_masks = torch.cat([br["matched_faces_mask"] for br in batch_results], dim=0)

        person_data = {
            "batch_size": batch_size,
            "num_references": num_refs,
            "image_height": h,
            "image_width": w,
            "matches": person_data_matches,
            "all_faces_mask": all_faces_masks,
            "matched_faces_mask": matched_faces_masks,
        }
        # Add all mask types: face_masks, head_masks, body_masks, hair_masks, etc.
        for mt in ALL_MASK_TYPES:
            person_data[f"{mt}_masks"] = person_data_masks[mt]

        # Body-part detection via SEGS/detector (optional)
        has_detector = segs is not None or bbox_detector is not None or segm_detector is not None
        if has_detector:
            aux_per_ref = [[] for _ in range(num_refs)]  # list of [1,H,W] per ref per batch
            aux_unassigned = []  # [1,H,W] per batch
            aux_part_counts = []  # [{ri: count}, ...] per batch

            for b in range(batch_size):
                single_image = current_image[b]  # [H, W, C]

                # Get or compute SEGS
                if segs is not None:
                    current_segs = segs
                else:
                    detector = segm_detector if segm_detector is not None else bbox_detector
                    current_segs = run_detector(detector, single_image.unsqueeze(0),
                                                 detect_threshold, detect_dilation, detect_crop_factor)

                # Assign SEGS to references via body mask overlap
                body_masks = person_data.get("body_masks", [])
                seg_assignments = assign_segs_to_references(current_segs, body_masks, num_refs, b)

                counts = {}
                for ri in range(num_refs):
                    parts = seg_assignments.get(ri, [])
                    counts[ri] = len(parts)
                    if parts:
                        merged = torch.max(torch.stack(parts), dim=0)[0]  # [H, W]
                        aux_per_ref[ri].append(merged.unsqueeze(0))  # [1, H, W]
                    else:
                        aux_per_ref[ri].append(torch.zeros(1, h, w, dtype=torch.float32))

                # Unassigned parts
                unassigned_parts = seg_assignments.get(-1, [])
                if unassigned_parts:
                    merged_unassigned = torch.max(torch.stack(unassigned_parts), dim=0)[0]
                    aux_unassigned.append(merged_unassigned.unsqueeze(0))
                else:
                    aux_unassigned.append(torch.zeros(1, h, w, dtype=torch.float32))

                aux_part_counts.append(counts)

                total_parts = sum(counts.values()) + len(unassigned_parts)
                print(f"[PersonSelectorMulti] Image {b+1}: {total_parts} body parts detected, "
                      f"{sum(1 for c in counts.values() if c > 0)} refs matched")

            # Store in PERSON_DATA
            person_data["aux_masks"] = [torch.cat(aux_per_ref[ri], dim=0) for ri in range(num_refs)]  # [B, H, W] per ref
            person_data["aux_unassigned_masks"] = torch.cat(aux_unassigned, dim=0)  # [B, H, W]
            person_data["aux_part_counts"] = aux_part_counts

        # Legacy mask outputs: face, head, body stacked across batch
        all_face_out = []
        all_head_out = []
        all_body_out = []
        for b in range(batch_size):
            mpt = batch_results[b]["masks_per_type"]
            for ri in range(num_refs):
                all_face_out.append(mpt["face"][ri])
                all_head_out.append(mpt["head"][ri])
                all_body_out.append(mpt["body"][ri])

        face_masks_batch = torch.cat(all_face_out, dim=0)
        head_masks_batch = torch.cat(all_head_out, dim=0)
        body_masks_batch = torch.cat(all_body_out, dim=0)

        total_matched = sum(len(br["assignments"]) for br in batch_results)
        total_faces = sum(br["face_count"] for br in batch_results)

        if total_matched > 0:
            combined_face = torch.max(face_masks_batch, dim=0, keepdim=True)[0]
            combined_head = torch.max(head_masks_batch, dim=0, keepdim=True)[0]
            combined_body = torch.max(body_masks_batch, dim=0, keepdim=True)[0]
        else:
            combined_face = empty_mask(h, w)
            combined_head = empty_mask(h, w)
            combined_body = empty_mask(h, w)

        # Auxiliary masks output
        if aux_mask_type != "none" and aux_mask_type in MASK_TYPE_LABELS:
            aux_list = []
            for b in range(batch_size):
                mpt = batch_results[b]["masks_per_type"]
                for ri in range(num_refs):
                    aux_list.append(mpt[aux_mask_type][ri])
            aux_masks_batch = torch.cat(aux_list, dim=0)
        else:
            aux_masks_batch = empty_mask(h, w)

        # Preview
        preview_parts = []
        for b in range(batch_size):
            body_masks_for_preview = batch_results[b]["masks_per_type"]["body"]
            p = self._render_preview(
                current_image[b:b+1], batch_results[b]["assignments"],
                batch_results[b]["cur_faces"], body_masks_for_preview, h, w
            )
            preview_parts.append(p)
        preview = torch.cat(preview_parts, dim=0)

        # Report: per batch item
        sim_values = []
        match_values = []
        report_lines = [
            f"Batch size: {batch_size}",
            f"Faces (total): {total_faces}",
            f"Reference persons: {num_refs}",
            f"Threshold: {'Auto' if auto_threshold else threshold} | Aggregation: {aggregation}",
            f"Matched (total): {total_matched}",
            "",
        ]

        for b in range(batch_size):
            br = batch_results[b]
            report_lines.append(f"  [Image {b+1}/{batch_size}] {br['face_count']} faces, {len(br['assignments'])} matched")
            for ri in range(num_refs):
                if ri in br["assignments"]:
                    fi, sim = br["assignments"][ri]
                    if b == 0:
                        sim_values.append(f"{sim:.4f}")
                        match_values.append("true")
                    report_lines.append(f"    ref {ri+1}: MATCH face #{fi} (sim {sim:.4f})")
                else:
                    if b == 0:
                        has_embs = len(ref_emb_sets[ri]) > 0
                        if has_embs and br["face_count"] > 0:
                            best_sim = float(np.max(br["sim_matrix"][ri]))
                            sim_values.append(f"{best_sim:.4f}")
                        else:
                            sim_values.append("0.0000")
                        match_values.append("false")
                    report_lines.append(f"    ref {ri+1}: no match")

        similarities_str = ", ".join(sim_values)
        matches_str = ", ".join(match_values)
        report = "\n".join(report_lines)

        ui_text = f"{total_matched}|{total_faces}|{similarities_str}"

        return {
            "ui": {"text": [ui_text]},
            "result": (person_data, face_masks_batch, head_masks_batch, body_masks_batch,
                       combined_face, combined_head, combined_body, aux_masks_batch,
                       preview, similarities_str, matches_str,
                       total_matched, total_faces, report),
        }
