"""PersonDataRefiner: regenerate PERSON_DATA masks at a new resolution,
preserving face-to-reference assignments from the original."""

import torch
import numpy as np

from .utils.face_analyzer import FaceAnalyzer
from .utils.masker import MaskGenerator, ALL_MASK_TYPES, generate_all_masks_for_face
from .utils.tensor_utils import tensor2np, tensor2cv2, mask2tensor, empty_mask


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
    RETURN_TYPES = ("PERSON_DATA", "STRING")
    RETURN_NAMES = ("person_data", "report")
    DESCRIPTION = (
        "Regenerate PERSON_DATA masks at a new image resolution.\n\n"
        "Use after upscaling: takes original person_data + hi-res images,\n"
        "re-detects faces and regenerates all masks at the new resolution\n"
        "while preserving face-to-reference assignments.\n\n"
        "Optional depth map input improves masks by filling gaps\n"
        "and removing overlapping objects using depth coherence."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_data": ("PERSON_DATA", {"tooltip": "Original PERSON_DATA from Person Selector Multi"}),
                "images": ("IMAGE", {"tooltip": "New images (batch size must match original person_data)"}),
                "sam_model": ("SAM_MODEL", {"tooltip": "SAM model for body mask generation"}),
                "mask_fill_holes": ("BOOLEAN", {"default": True}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100}),
                "det_size": (["320", "480", "640", "768"], {"default": "640",
                             "tooltip": "Face detection resolution"}),
            },
            "optional": {
                "depth_map": ("IMAGE", {"tooltip": "Depth map batch for depth-guided mask refinement"}),
                "depth_edge_threshold": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.30, "step": 0.01,
                                                    "tooltip": "Depth gradient threshold for edge detection"}),
                "depth_carve_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "How strongly depth edges cut masks"}),
                "depth_grow_pixels": ("INT", {"default": 30, "min": 0, "max": 200, "step": 5,
                                               "tooltip": "Gap filling between depth edges"}),
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

    def execute(self, person_data, images, sam_model, mask_fill_holes, mask_blur, det_size,
                depth_map=None, depth_edge_threshold=0.05, depth_carve_strength=0.8, depth_grow_pixels=30):

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
                    masks = generate_all_masks_for_face(
                        cur_rgb, new_faces[nfi], device, sam_model,
                        mask_fill_holes, mask_blur, **depth_kwargs)
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
                    pf = generate_all_masks_for_face(
                        cur_rgb, new_faces[nfi], device, sam_model,
                        mask_fill_holes, mask_blur, **depth_kwargs)
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

        # Cross-reference deconfliction
        if use_depth and num_refs >= 2:
            for mt in ("body", "head", "face"):
                for b in range(batch_size):
                    overlap_dict = {}
                    for ri in range(num_refs):
                        m = pd_masks[mt][ri][b].cpu().numpy()
                        if m.sum() > 0:
                            overlap_dict[ri] = m
                    if len(overlap_dict) >= 2:
                        resolved = deconflict_masks(overlap_dict, depth_nps[b])
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

        _elapsed = int(_time.monotonic() - _t0)
        report_lines.insert(0, f"Runtime: {_elapsed}s")
        report = "\n".join(report_lines)

        print(f"\n  Done in {_elapsed}s")
        print(f"{'='*50}\n")

        return (new_person_data, report)
