"""PersonDataRefiner: regenerate PERSON_DATA masks at a new resolution,
preserving face-to-reference assignments from the original.
Optionally runs YOLO body-part detection and injects aux_masks into person_data.
Chainable: when resolution hasn't changed, skips the full face-regen and just
runs YOLO, allowing multiple refiners in sequence (each with a different aux model/label)."""

import torch
import numpy as np

from .utils.face_analyzer import FaceAnalyzer
from .utils.masker import MaskGenerator, ALL_MASK_TYPES, generate_all_masks_for_face
from .utils.tensor_utils import tensor2np, tensor2cv2, mask2tensor, empty_mask
from .utils.yolo_detector import get_available_yolo_models, detect_objects, assign_detections_to_references
from .utils.mask_utils import fill_mask_holes_2d, expand_mask, feather_mask


class PersonDataRefiner:
    """Regenerate PERSON_DATA masks at a new image resolution.

    Takes the original PERSON_DATA (from a low-res pass) and new hi-res images,
    re-detects faces, matches them to the original references by position,
    and regenerates all masks at the new resolution.

    Optional depth map input enables depth-guided mask reconstruction
    (gap filling + overlap removal).
    """

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("PERSON_DATA", "MASK", "STRING")
    RETURN_NAMES = ("person_data", "aux_masks", "report")
    DESCRIPTION = (
        "Regenerate PERSON_DATA masks at a new image resolution.\n\n"
        "Use after upscaling: takes original person_data + hi-res images,\n"
        "re-detects faces and regenerates all masks at the new resolution\n"
        "while preserving face-to-reference assignments.\n\n"
        "Accepts either sam_model (SAM2) or sam3_model (SAM3) for body masks.\n"
        "SAM3 takes priority when both are connected.\n\n"
        "Chainable: when image resolution matches person_data, skips\n"
        "face re-detection entirely and just runs YOLO aux detection.\n"
        "Chain multiple refiners with different aux_model/aux_label combos\n"
        "to build separate aux mask passes (hands, feet, etc.).\n\n"
        "Optional depth map input improves masks by filling gaps\n"
        "and removing overlapping objects using depth coherence."
    )

    @classmethod
    def INPUT_TYPES(cls):
        yolo_models = ["none"] + get_available_yolo_models()
        return {
            "required": {
                "person_data": ("PERSON_DATA", {"tooltip": "Original PERSON_DATA from Person Selector Multi / SAM3"}),
                "images": ("IMAGE", {"tooltip": "New images (batch size must match original person_data)"}),
                "mask_fill_holes": ("BOOLEAN", {"default": True}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100}),
                "det_size": (["320", "480", "640", "768"], {"default": "640",
                             "tooltip": "Face detection resolution"}),
            },
            "optional": {
                "sam_model": ("SAM_MODEL", {"tooltip": "SAM2 model for body mask generation (fallback when sam3_model not connected)"}),
                "sam3_model": ("SAM3_MODEL_CONFIG", {"tooltip": "SAM3 model from LoadSAM3Model. Takes priority over sam_model when both connected."}),
                "depth_map": ("IMAGE", {"tooltip": "Depth map batch for depth-guided mask refinement"}),
                "depth_edge_threshold": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.30, "step": 0.01,
                                                    "tooltip": "Depth gradient threshold for edge detection"}),
                "depth_carve_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "How strongly depth edges cut masks"}),
                "depth_grow_pixels": ("INT", {"default": 30, "min": 0, "max": 200, "step": 5,
                                               "tooltip": "Gap filling between depth edges"}),
                "aux_model": (yolo_models, {
                    "tooltip": "YOLO segm model for body-part detection (hands, feet, etc.).\n"
                               "Runs on the current images and injects results into person_data[\"aux_masks\"].\n"
                               "Chainable: each refiner replaces aux_masks, so chain multiple refiners\n"
                               "with different models/labels for separate aux passes."}),
                "aux_confidence": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05,
                                              "tooltip": "YOLO detection confidence threshold"}),
                "aux_label": ("STRING", {"default": "",
                                          "tooltip": "Filter YOLO detections by class label (substring match).\n"
                                                     "Empty = all classes. Comma-separated: 'hand,foot'"}),
                "aux_fill_holes": ("BOOLEAN", {"default": False,
                                                "tooltip": "Fill holes inside YOLO aux masks"}),
                "aux_expand_pixels": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                                                "tooltip": "Dilate YOLO aux masks by N pixels"}),
                "aux_blend_pixels": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                                               "tooltip": "Gaussian blur radius for YOLO aux mask edges"}),
            },
        }

    def _match_faces_by_position(self, old_per_face_masks, old_face_to_ref, new_faces,
                                  scale_x, scale_y, max_distance_factor=2.0):
        """Match newly detected faces to old references by centroid proximity.

        Returns: list[int|None] — new_face_to_ref[new_fi] = ref_index or None
        """
        # Compute old face centroids, scaled to new resolution
        old_centroids = []
        old_face_widths = []
        for fi, face_masks in enumerate(old_per_face_masks):
            face_mask = face_masks.get("face", face_masks.get("head"))
            if face_mask is None:
                old_centroids.append(None)
                old_face_widths.append(50)
                continue
            m = face_mask[0].cpu().numpy() if face_mask.dim() == 3 else face_mask.cpu().numpy()
            ys, xs = np.where(m > 0.5)
            if len(ys) > 0:
                cy = float(ys.mean()) * scale_y
                cx = float(xs.mean()) * scale_x
                fw = float(xs.max() - xs.min()) * scale_x
                old_centroids.append((cx, cy))
                old_face_widths.append(max(fw, 20))
            else:
                old_centroids.append(None)
                old_face_widths.append(50)

        # Compute new face centroids
        new_centroids = []
        for face in new_faces:
            x1, y1, x2, y2 = face.bbox
            new_centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))

        # Build distance matrix and greedy match
        used_old = set()
        used_new = set()
        pairs = []

        for old_fi in range(len(old_centroids)):
            if old_centroids[old_fi] is None:
                continue
            ox, oy = old_centroids[old_fi]
            max_dist = old_face_widths[old_fi] * max_distance_factor

            for new_fi in range(len(new_centroids)):
                nx, ny = new_centroids[new_fi]
                dist = np.sqrt((ox - nx) ** 2 + (oy - ny) ** 2)
                if dist < max_dist:
                    pairs.append((dist, old_fi, new_fi))

        # Sort by distance, assign greedily
        pairs.sort(key=lambda x: x[0])
        old_to_new = {}
        for dist, old_fi, new_fi in pairs:
            if old_fi in used_old or new_fi in used_new:
                continue
            old_to_new[old_fi] = new_fi
            used_old.add(old_fi)
            used_new.add(new_fi)

        # Build new face_to_ref: transfer old assignments
        new_face_to_ref = [None] * len(new_faces)
        for old_fi, new_fi in old_to_new.items():
            ref_idx = old_face_to_ref[old_fi]
            if ref_idx is not None:
                new_face_to_ref[new_fi] = ref_idx

        return new_face_to_ref

    def execute(self, person_data, images, mask_fill_holes, mask_blur, det_size,
                sam_model=None, sam3_model=None,
                depth_map=None, depth_edge_threshold=0.05, depth_carve_strength=0.8, depth_grow_pixels=30,
                aux_model="none", aux_confidence=0.35, aux_label="",
                aux_fill_holes=False, aux_expand_pixels=0, aux_blend_pixels=0):

        import time as _time
        import cv2
        _t0 = _time.monotonic()

        batch_size = images.shape[0]
        new_h, new_w = images.shape[1], images.shape[2]
        old_h = person_data["image_height"]
        old_w = person_data["image_width"]
        scale_y = new_h / old_h
        scale_x = new_w / old_w
        num_refs = person_data["num_references"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolution_changed = (new_h != old_h or new_w != old_w)

        # Smart skip: if resolution hasn't changed AND aux_model is connected,
        # skip the full face-regen and just run YOLO on the existing person_data.
        # This makes chaining multiple refiners lightweight (YOLO-only pass).
        if not resolution_changed and aux_model != "none":
            print(f"\n{'='*50}")
            print(f"  PersonDataRefiner (YOLO-only pass, same resolution)")
            print(f"  {new_w}x{new_h} | Refs: {num_refs} | aux_model: {aux_model}")
            print(f"{'='*50}")
            import copy
            new_person_data = copy.copy(person_data)
            new_person_data = self._run_yolo_aux(
                new_person_data, images, num_refs, new_h, new_w, batch_size,
                aux_model, aux_confidence, aux_label,
                aux_fill_holes, aux_expand_pixels, aux_blend_pixels)
            _elapsed = int(_time.monotonic() - _t0)
            report = f"YOLO-only pass ({_elapsed}s): resolution unchanged, skipped face regen"
            print(f"  Done in {_elapsed}s\n{'='*50}\n")
            aux_masks_batch = self._build_aux_output(new_person_data, num_refs, batch_size, new_h, new_w)
            return (new_person_data, aux_masks_batch, report)

        det_size_int = int(det_size)
        analyzer = FaceAnalyzer(det_size_int)

        from .utils.depth_refine import compute_depth_edges, deconflict_masks
        use_depth = depth_map is not None
        depth_nps = []
        depth_edges_list = []
        if use_depth:
            for b in range(batch_size):
                dm = depth_map[b].cpu().numpy()
                dnp = dm[:, :, 0] if dm.ndim == 3 else dm
                if dnp.shape[0] != new_h or dnp.shape[1] != new_w:
                    dnp = cv2.resize(dnp, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                depth_nps.append(dnp)
                depth_edges_list.append(compute_depth_edges(dnp, depth_edge_threshold))

        print(f"\n{'='*50}")
        print(f"  PersonDataRefiner")
        print(f"  {old_w}x{old_h} → {new_w}x{new_h} (scale {scale_x:.2f}x{scale_y:.2f})")
        print(f"  Batch: {batch_size} | Refs: {num_refs}" +
              (f" | Depth: yes" if use_depth else ""))
        print(f"{'='*50}")

        report_lines = [
            f"Resolution: {old_w}x{old_h} → {new_w}x{new_h}",
            f"Scale: {scale_x:.2f}x{scale_y:.2f}",
            f"References: {num_refs}",
            f"Depth: {'enabled' if use_depth else 'disabled'}",
            "",
        ]

        # Per-batch processing
        all_masks_per_type = {mt: [] for mt in ALL_MASK_TYPES}  # [ri] → list of [1,H,W] per batch
        all_per_face_masks = []
        all_face_to_ref = []
        all_faces_masks_list = []
        matched_faces_masks_list = []
        matches_list = []

        for b in range(batch_size):
            single = images[b:b + 1]
            cur_bgr = tensor2cv2(single)
            cur_rgb = tensor2np(single)
            depth_np = depth_nps[b] if use_depth else None

            # Re-detect faces at new resolution
            new_faces = analyzer.detect_faces(cur_bgr)
            new_face_count = len(new_faces)

            # Match new faces to old references by position
            old_pfm = person_data.get("per_face_masks", [[]])[b] if b < len(person_data.get("per_face_masks", [])) else []
            old_f2r = person_data.get("face_to_ref", [[]])[b] if b < len(person_data.get("face_to_ref", [])) else []

            new_face_to_ref = self._match_faces_by_position(
                old_pfm, old_f2r, new_faces, scale_x, scale_y)

            # Build ref→new_face mapping
            ref_to_new_fi = {}
            for nfi, ref_idx in enumerate(new_face_to_ref):
                if ref_idx is not None:
                    ref_to_new_fi[ref_idx] = nfi

            matched_count = sum(1 for r in new_face_to_ref if r is not None)
            old_face_count = len(old_f2r) if old_f2r else 0
            print(f"  [Img {b+1}] InsightFace detected {new_face_count} faces "
                  f"(old had {old_face_count}), {matched_count}/{num_refs} matched to refs")
            if new_face_count > 0 and matched_count == 0:
                for ofi in range(len(old_pfm)):
                    fm = old_pfm[ofi].get("face", old_pfm[ofi].get("head"))
                    if fm is None: continue
                    m = fm[0].cpu().numpy() if fm.dim() == 3 else fm.cpu().numpy()
                    import numpy as _np
                    ys, xs = _np.where(m > 0.5)
                    if len(ys) > 0:
                        ocx = float(xs.mean()) * scale_x
                        ocy = float(ys.mean()) * scale_y
                        ofw = float(xs.max() - xs.min()) * scale_x
                        print(f"      old face {ofi+1} ref={old_f2r[ofi] if ofi < len(old_f2r) else '?'} "
                              f"center=({ocx:.0f},{ocy:.0f}) width={ofw:.0f} max_match_dist={ofw*2:.0f}")
                for nfi_, f in enumerate(new_faces):
                    x1,y1,x2,y2 = f.bbox
                    print(f"      new face {nfi_+1} bbox=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) "
                          f"center=({(x1+x2)/2:.0f},{(y1+y2)/2:.0f})")

            depth_kwargs = dict(
                depth_edges_data=depth_edges_list[b] if use_depth else None,
                depth_np=depth_nps[b] if use_depth else None,
                depth_carve_strength=depth_carve_strength,
                depth_grow=depth_grow_pixels,
            )

            # Generate masks per reference
            masks_for_refs = {mt: [] for mt in ALL_MASK_TYPES}
            matches_for_image = []
            for ri in range(num_refs):
                if ri in ref_to_new_fi:
                    nfi = ref_to_new_fi[ri]
                    others = [f for j, f in enumerate(new_faces) if j != nfi]
                    masks = generate_all_masks_for_face(
                        cur_rgb, new_faces[nfi], device, sam_model,
                        mask_fill_holes, mask_blur, other_faces=others,
                        sam3_config=sam3_model, **depth_kwargs)
                    for mt in ALL_MASK_TYPES:
                        masks_for_refs[mt].append(masks.get(mt, empty_mask(new_h, new_w)))
                    matches_for_image.append(True)
                    print(f"  [Img {b+1}] ref {ri+1} → face #{ref_to_new_fi[ri]+1}")
                else:
                    for mt in ALL_MASK_TYPES:
                        masks_for_refs[mt].append(empty_mask(new_h, new_w))
                    matches_for_image.append(False)
                    print(f"  [Img {b+1}] ref {ri+1} → no face found")

            matches_list.append(matches_for_image)

            # Per-face masks for ALL detected faces
            per_face_masks = []
            fi_to_ri = {nfi: ri for ri, nfi in ref_to_new_fi.items()}
            for nfi in range(new_face_count):
                if nfi in fi_to_ri:
                    ri = fi_to_ri[nfi]
                    pf = {mt: masks_for_refs[mt][ri] for mt in ALL_MASK_TYPES}
                else:
                    others = [f for j, f in enumerate(new_faces) if j != nfi]
                    pf = generate_all_masks_for_face(
                        cur_rgb, new_faces[nfi], device, sam_model,
                        mask_fill_holes, mask_blur, other_faces=others,
                        sam3_config=sam3_model, **depth_kwargs)
                per_face_masks.append(pf)

            all_per_face_masks.append(per_face_masks)
            all_face_to_ref.append(new_face_to_ref)

            # all_faces_mask and matched_faces_mask
            if new_face_count > 0:
                face_parts = []
                for face in new_faces:
                    fm = MaskGenerator.generate_face_mask(cur_rgb, face, device)
                    face_parts.append(mask2tensor(fm))
                all_faces_mask = torch.max(torch.cat(face_parts, dim=0), dim=0, keepdim=True)[0]
            else:
                all_faces_mask = empty_mask(new_h, new_w)

            matched_fi = set(ref_to_new_fi.values())
            if matched_fi and new_face_count > 0:
                mp = []
                for fi in matched_fi:
                    fm = MaskGenerator.generate_face_mask(cur_rgb, new_faces[fi], device)
                    mp.append(mask2tensor(fm))
                matched_faces_mask = torch.max(torch.cat(mp, dim=0), dim=0, keepdim=True)[0]
            else:
                matched_faces_mask = empty_mask(new_h, new_w)

            all_faces_masks_list.append(all_faces_mask)
            matched_faces_masks_list.append(matched_faces_mask)

            # Accumulate per-ref masks
            for mt in ALL_MASK_TYPES:
                all_masks_per_type[mt].append(masks_for_refs[mt])

            matched_count = sum(1 for m in matches_for_image if m)
            report_lines.append(f"  [Image {b+1}/{batch_size}] {new_face_count} faces, {matched_count} matched to refs")

        # Assemble PERSON_DATA
        # Stack per-ref masks: person_data["{type}_masks"][ri] = [B, H, W]
        pd_masks = {mt: [] for mt in ALL_MASK_TYPES}
        for ri in range(num_refs):
            for mt in ALL_MASK_TYPES:
                ref_batch = []
                for b in range(batch_size):
                    ref_batch.append(all_masks_per_type[mt][b][ri])
                pd_masks[mt].append(torch.cat(ref_batch, dim=0))

        # Cross-reference deconfliction: resolve overlapping masks per batch image
        if use_depth and num_refs >= 2:
            from .utils.depth_refine import deconflict_masks
            for mt in ("body", "head", "face"):
                for b in range(batch_size):
                    overlap_dict = {}
                    for ri in range(num_refs):
                        m = pd_masks[mt][ri][b].cpu().numpy()
                        if m.sum() > 0:
                            overlap_dict[ri] = m
                    if len(overlap_dict) >= 2:
                        eb = depth_edges_list[b][1] if b < len(depth_edges_list) else None
                        resolved = deconflict_masks(overlap_dict, depth_nps[b], edges_binary=eb)
                        for ri, m in resolved.items():
                            pd_masks[mt][ri][b] = torch.from_numpy(m)

        new_person_data = {
            "batch_size": batch_size,
            "num_references": num_refs,
            "image_height": new_h,
            "image_width": new_w,
            "matches": matches_list,
            "all_faces_mask": torch.cat(all_faces_masks_list, dim=0),
            "matched_faces_mask": torch.cat(matched_faces_masks_list, dim=0),
            "per_face_masks": all_per_face_masks,
            "face_to_ref": all_face_to_ref,
        }
        for mt in ALL_MASK_TYPES:
            new_person_data[f"{mt}_masks"] = pd_masks[mt]

        # Run YOLO aux detection if aux_model connected (full-regen path)
        if aux_model != "none":
            new_person_data = self._run_yolo_aux(
                new_person_data, images, num_refs, new_h, new_w, batch_size,
                aux_model, aux_confidence, aux_label,
                aux_fill_holes, aux_expand_pixels, aux_blend_pixels)

        _elapsed = int(_time.monotonic() - _t0)
        report_lines.insert(0, f"Runtime: {_elapsed}s")
        report = "\n".join(report_lines)

        print(f"\n  Done in {_elapsed}s")
        print(f"{'='*50}\n")

        aux_masks_batch = self._build_aux_output(new_person_data, num_refs, batch_size, new_h, new_w)
        return (new_person_data, aux_masks_batch, report)

    def _build_aux_output(self, person_data, num_refs, batch_size, h, w):
        """Build a stacked MASK tensor from person_data['aux_masks'] for the output port."""
        if "aux_masks" not in person_data or num_refs == 0:
            return empty_mask(h, w)
        aux_list = []
        for b in range(batch_size):
            for ri in range(num_refs):
                if ri < len(person_data["aux_masks"]):
                    aux_list.append(person_data["aux_masks"][ri][b:b+1])
        return torch.cat(aux_list, dim=0) if aux_list else empty_mask(h, w)

    def _run_yolo_aux(self, person_data, images, num_refs, h, w, batch_size,
                      aux_model, aux_confidence, aux_label,
                      aux_fill_holes, aux_expand_pixels, aux_blend_pixels):
        """Run YOLO detection on images and inject aux_masks into person_data.

        Reuses the same logic as PersonSelectorMulti's YOLO block:
        detect → assign to refs via body mask overlap → merge → post-process.
        Replaces any existing aux_masks in person_data.
        """
        def _aux_post(mask_2d):
            if aux_expand_pixels > 0:
                mask_2d = expand_mask(mask_2d, aux_expand_pixels)
            if aux_fill_holes:
                mask_2d = fill_mask_holes_2d(mask_2d)
            if aux_blend_pixels > 0:
                mask_2d = feather_mask(mask_2d, aux_blend_pixels)
            return mask_2d

        aux_per_ref = [[] for _ in range(num_refs)]
        aux_unassigned = []
        aux_part_counts = []

        for b in range(batch_size):
            single_image = images[b]  # [H, W, C]
            detections = detect_objects(single_image, aux_model,
                                        confidence=aux_confidence, label_filter=aux_label)

            body_masks_list = person_data.get("body_masks", [])
            det_assignments = assign_detections_to_references(
                detections, body_masks_list, num_refs, b)

            counts = {}
            for ri in range(num_refs):
                parts = det_assignments.get(ri, [])
                counts[ri] = len(parts)
                if parts:
                    merged = torch.max(torch.stack(parts), dim=0)[0]
                    merged = _aux_post(merged)
                    aux_per_ref[ri].append(merged.unsqueeze(0))
                else:
                    aux_per_ref[ri].append(torch.zeros(1, h, w, dtype=torch.float32))

            unassigned_parts = det_assignments.get(-1, [])
            if unassigned_parts:
                merged_unassigned = torch.max(torch.stack(unassigned_parts), dim=0)[0]
                merged_unassigned = _aux_post(merged_unassigned)
                aux_unassigned.append(merged_unassigned.unsqueeze(0))
            else:
                aux_unassigned.append(torch.zeros(1, h, w, dtype=torch.float32))

            aux_part_counts.append(counts)
            total_parts = sum(counts.values()) + len(unassigned_parts)
            labels_found = set(d["label"] for d in detections)
            print(f"  [PersonDataRefiner YOLO] Image {b+1}: {total_parts} detections "
                  f"({', '.join(labels_found) or 'none'}), "
                  f"{sum(1 for c in counts.values() if c > 0)} refs matched")

        # Inject into person_data (replaces any existing aux_masks)
        person_data["aux_masks"] = [torch.cat(aux_per_ref[ri], dim=0) for ri in range(num_refs)]
        person_data["aux_unassigned_masks"] = torch.cat(aux_unassigned, dim=0)
        person_data["aux_part_counts"] = aux_part_counts

        return person_data
