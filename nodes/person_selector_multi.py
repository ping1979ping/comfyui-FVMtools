import torch
import numpy as np
import cv2

from .utils.face_analyzer import FaceAnalyzer
from .utils.matcher import compute_similarity, aggregate_similarities, build_appearance_matrix
from .utils.masker import MaskGenerator, FACE_LABELS, HEAD_LABELS, MASK_TYPE_LABELS, BISENET_MASK_TYPES, generate_all_masks_for_face
from .utils.tensor_utils import tensor2np, tensor2cv2, mask2tensor, np2tensor, empty_mask, apply_gaussian_blur, fill_mask_holes
from .utils.mask_utils import clean_mask_crumbs
from .utils.segs_utils import assign_segs_to_references, run_detector
from .utils.appearance import (
    extract_hair_color,
    extract_head_histogram,
    extract_palette_histogram,
    extract_clothing_histogram,
    parse_match_weights,
)

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
        "Matches multiple reference persons in the current image using InsightFace (ArcFace) embeddings,\n"
        "optionally combined with hair color, head appearance, and outfit color matching.\n\n"
        "Each reference input accepts a batch of images for one person.\n"
        "Connect a reference to auto-create the next input slot.\n\n"
        "Supports batch input: pass multiple images and get PERSON_DATA for PersonDetailer.\n\n"
        "Faces are assigned exclusively: each face matches at most one reference.\n\n"
        "match_weights controls the scoring blend: 'face/hair/head' (e.g. '60/20/20')\n"
        "or 'face/hair/head/outfit' (e.g. '50/15/15/20') when outfit_palettes is connected.\n"
        "Set to '100/0/0' for pure face matching (previous behavior).\n\n"
        "Outfit matching: connect palette_preview images as a batch to outfit_palettes.\n"
        "Compares palette color distribution with detected clothing colors (BiSeNet).\n\n"
        "Optional: connect SEGS or a detector for body-part detection.\n"
        "Detected parts are assigned to references via body mask overlap\n"
        "and stored as 'aux' masks in PERSON_DATA for PersonDetailer."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ("SAM_MODEL", {"tooltip": "SAM model from Impact Pack SAMLoader — required for body masks"}),
                "current_image": ("IMAGE", {"tooltip": "Image(s) to search for faces. Supports batch input — each image is processed independently."}),
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
                "match_weights": ("STRING", {"default": "60/20/20",
                                             "tooltip": "Matching weight blend: face/hair/head or face/hair/head/outfit.\n"
                                                        "Controls how much each signal contributes to the final similarity score.\n\n"
                                                        "3 values (no outfit):\n"
                                                        "  60/20/20 — balanced: face + hair + head (default)\n"
                                                        "  100/0/0 — pure face matching\n"
                                                        "  50/30/20 — strong hair weight\n\n"
                                                        "4 values (with outfit_palettes connected):\n"
                                                        "  50/15/15/20 — face + hair + head + outfit colors\n"
                                                        "  40/15/15/30 — heavy outfit weight\n"
                                                        "  60/20/20/0 — ignore outfit even when connected\n\n"
                                                        "Hair = BiSeNet hair color (HSV). Head = head crop histogram.\n"
                                                        "Outfit = clothing region vs. palette color distribution.\n"
                                                        "Values are auto-normalized, so 3/1/1 equals 60/20/20."}),
                "reference_1": ("IMAGE", {"tooltip": "Reference image(s) for person 1. Pass a batch of images of the same person for better matching accuracy."}),
            },
            "optional": {
                "outfit_palettes": ("IMAGE", {"tooltip": "Palette preview image batch for outfit color matching.\n"
                                                          "Image[0] = palette for reference 1, image[1] = for reference 2, etc.\n"
                                                          "If batch is smaller than number of references, remaining refs skip outfit matching.\n"
                                                          "Connect palette_preview outputs from Color Palette Generator.\n"
                                                          "Use match_weights with 4 values to control outfit weight (e.g. 50/15/15/20)."}),
                "segs": ("SEGS", {"tooltip": "Pre-computed body-part SEGS (e.g. from Impact Pack). Takes priority over detector inputs."}),
                "bbox_detector": ("BBOX_DETECTOR", {"tooltip": "Bounding box detector for body-part detection. Ignored if SEGS connected."}),
                "segm_detector": ("SEGM_DETECTOR", {"tooltip": "Segmentation detector (preferred over bbox). Ignored if SEGS connected."}),
                "depth_map": ("IMAGE", {"tooltip": "Depth map batch from Depth Anything V2 or similar. Improves masks via edge carving and cross-reference deconfliction."}),
                "depth_edge_threshold": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.30, "step": 0.01,
                                                    "tooltip": "Depth gradient threshold for edge detection. Lower = more edges detected."}),
                "depth_carve_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "How strongly depth edges cut masks. 0=off, 1=full cut."}),
                "depth_grow_pixels": ("INT", {"default": 30, "min": 0, "max": 200, "step": 5,
                                               "tooltip": "Gap filling between depth edges. 0 = no growing."}),
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

    def _extract_ref_appearance(self, analyzer, ref_batch, device):
        """Extract appearance features (hair color, head histogram) from reference images.

        Returns: (hair_color_hsv_or_None, head_hist_or_None) aggregated across batch.
        """
        hair_colors = []
        head_hists = []

        for i in range(ref_batch.shape[0]):
            frame = ref_batch[i:i+1]
            rgb = tensor2np(frame)
            bgr = tensor2cv2(frame)
            faces = analyzer.detect_faces(bgr)
            if not faces:
                continue

            face = faces[0]
            # Run BiSeNet for hair color
            label_map = MaskGenerator._run_bisenet(rgb, face, device)
            hc = extract_hair_color(rgb, label_map)
            if hc is not None:
                hair_colors.append(hc)

            # Head histogram
            hh = extract_head_histogram(rgb, face.bbox)
            if hh is not None:
                head_hists.append(hh)

        # Aggregate: median hair color, average histogram
        agg_hair = np.median(np.stack(hair_colors), axis=0).astype(np.float32) if hair_colors else None
        agg_hist = None
        if head_hists:
            agg_hist = head_hists[0].copy()
            for h in head_hists[1:]:
                agg_hist += h
            cv2.normalize(agg_hist, agg_hist)

        return agg_hair, agg_hist

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

    def _generate_all_masks(self, cur_rgb, face, device, sam_model, mask_fill_holes, mask_blur,
                            depth_edges_data=None, depth_np=None,
                            depth_carve_strength=0.8, depth_grow=30,
                            other_faces=None):
        """Generate all mask types. Delegates to shared generate_all_masks_for_face()."""
        return generate_all_masks_for_face(
            cur_rgb, face, device, sam_model, mask_fill_holes, mask_blur,
            depth_edges_data=depth_edges_data, depth_np=depth_np,
            depth_carve_strength=depth_carve_strength, depth_grow=depth_grow,
            other_faces=other_faces,
        )

    def _render_preview(self, current_image, assignments, cur_faces, body_masks_list, h, w,
                         depth_np=None, depth_edges_binary=None):
        preview = tensor2np(current_image).copy()

        # Depth map overlay (subtle blue tint)
        if depth_np is not None:
            depth_vis = (np.clip(depth_np, 0, 1) * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            preview = cv2.addWeighted(preview, 0.7, depth_rgb, 0.3, 0)

        # Depth edges overlay (cyan thin lines)
        if depth_edges_binary is not None:
            edge_overlay = np.zeros_like(preview)
            edge_overlay[depth_edges_binary] = (0, 255, 255)  # cyan
            preview = cv2.addWeighted(preview, 1.0, edge_overlay, 0.4, 0)

        for ri, (fi, sim) in assignments.items():
            color = _PREVIEW_COLORS[ri % len(_PREVIEW_COLORS)]
            mask_np = (body_masks_list[ri][0].cpu().numpy() * 255).astype(np.uint8)

            # Transparent fill
            fill_overlay = np.zeros_like(preview)
            fill_overlay[mask_np > 128] = color
            preview = cv2.addWeighted(preview, 1.0, fill_overlay, 0.15, 0)

            # Thick contour lines
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(preview, contours, -1, color, 3)

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
                               mask_fill_holes, mask_blur, device,
                               ref_appearances=None, match_weights=(1.0, 0.0, 0.0, 0.0),
                               ref_outfit_hists=None,
                               depth_edges_data=None, depth_np=None,
                               depth_carve_strength=0.8, depth_grow=30):
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

        # Build face embedding similarity matrix
        face_sim_matrix = self._build_similarity_matrix(ref_emb_sets, face_embs, aggregation)

        # Appearance-enhanced matching
        w_face, w_hair, w_head = match_weights[0], match_weights[1], match_weights[2]
        w_outfit = match_weights[3] if len(match_weights) > 3 else 0.0
        use_appearance = (w_hair > 0 or w_head > 0 or w_outfit > 0) and ref_appearances is not None and face_count > 0

        if use_appearance:
            # Extract appearance features for detected faces
            # Run BiSeNet once per face and reuse for hair, head, and clothing
            face_hair_colors = []
            face_head_hists = []
            face_label_maps = []
            for face in cur_faces:
                label_map = MaskGenerator._run_bisenet(cur_rgb, face, device)
                face_label_maps.append(label_map)
                face_hair_colors.append(extract_hair_color(cur_rgb, label_map))
                face_head_hists.append(extract_head_histogram(cur_rgb, face.bbox))

            ref_hair_colors = [ra[0] for ra in ref_appearances]
            ref_head_hists = [ra[1] for ra in ref_appearances]

            # Extract clothing histograms for outfit matching
            # BiSeNet label 16 = cloth — reuses label map from above, no extra cost.
            face_clothing_hists = None
            if w_outfit > 0 and ref_outfit_hists is not None:
                face_clothing_hists = []
                for fi, face in enumerate(cur_faces):
                    cloth_mask = (face_label_maps[fi] == 16).astype(np.float32)
                    # Cloth label already excludes head — pass empty head mask
                    head_mask_np = np.zeros_like(cloth_mask)
                    face_clothing_hists.append(
                        extract_clothing_histogram(cur_rgb, cloth_mask, head_mask_np)
                    )

            sim_matrix = build_appearance_matrix(
                face_sim_matrix, ref_hair_colors, face_hair_colors,
                ref_head_hists, face_head_hists, match_weights,
                ref_outfit_hists=ref_outfit_hists,
                face_clothing_hists=face_clothing_hists,
            )

            if sim_matrix.size > 0:
                weight_str = f"{w_face:.0%}/{w_hair:.0%}/{w_head:.0%}"
                if w_outfit > 0:
                    weight_str += f"/{w_outfit:.0%}"
                print(f"[PersonSelectorMulti] face_sim:\n{np.array2string(face_sim_matrix, precision=4)}")
                print(f"[PersonSelectorMulti] combined_sim (weights {weight_str}):\n"
                      f"{np.array2string(sim_matrix, precision=4)}")
        else:
            sim_matrix = face_sim_matrix
            if sim_matrix.size > 0:
                print(f"[PersonSelectorMulti] sim_matrix:\n{np.array2string(sim_matrix, precision=4)}")

        assignments = self._assign_greedy(sim_matrix, effective_threshold)

        # Collect all mask types per reference
        from .utils.masker import ALL_MASK_TYPES
        masks_per_type = {mt: [] for mt in ALL_MASK_TYPES}

        for ri in range(num_refs):
            if ri in assignments:
                fi, sim = assignments[ri]
                others = [f for j, f in enumerate(cur_faces) if j != fi]
                masks = self._generate_all_masks(cur_rgb, cur_faces[fi], device, sam_model, mask_fill_holes, mask_blur,
                                                  depth_edges_data=depth_edges_data, depth_np=depth_np,
                                                  depth_carve_strength=depth_carve_strength, depth_grow=depth_grow,
                                                  other_faces=others)
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

        # Per-face masks for ALL detected faces (for generic slot in PersonDetailer)
        # Matched faces reuse already-computed masks; unmatched faces get new masks generated
        fi_to_ri = {fi: ri for ri, (fi, sim) in assignments.items()}
        per_face_masks = []
        for fi in range(face_count):
            if fi in fi_to_ri:
                ri = fi_to_ri[fi]
                per_face = {mt: masks_per_type[mt][ri] for mt in ALL_MASK_TYPES}
            else:
                others = [f for j, f in enumerate(cur_faces) if j != fi]
                per_face = self._generate_all_masks(cur_rgb, cur_faces[fi], device, sam_model, mask_fill_holes, mask_blur,
                                                      depth_edges_data=depth_edges_data, depth_np=depth_np,
                                                      depth_carve_strength=depth_carve_strength, depth_grow=depth_grow,
                                                      other_faces=others)
            per_face_masks.append(per_face)
        face_to_ref = [fi_to_ri.get(fi) for fi in range(face_count)]

        return {
            "masks_per_type": masks_per_type,
            "assignments": assignments,
            "cur_faces": cur_faces,
            "face_count": face_count,
            "sim_matrix": sim_matrix,
            "all_faces_mask": all_faces_mask,
            "matched_faces_mask": matched_faces_mask,
            "per_face_masks": per_face_masks,
            "face_to_ref": face_to_ref,
        }

    def execute(self, sam_model, current_image, reference_1, auto_threshold, threshold, aggregation,
                mask_fill_holes, mask_blur, det_size, aux_mask_type="none",
                detect_threshold=0.3, detect_dilation=10, detect_crop_factor=3.0,
                match_weights="60/20/20",
                outfit_palettes=None,
                segs=None, bbox_detector=None, segm_detector=None,
                depth_map=None, depth_edge_threshold=0.05, depth_carve_strength=0.8, depth_grow_pixels=30,
                **kwargs):
        import time as _time
        _t0 = _time.monotonic()
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

        # Parse appearance weights (3 or 4 values)
        weights = parse_match_weights(match_weights)
        w_face, w_hair, w_head, w_outfit = weights
        use_appearance = w_hair > 0 or w_head > 0 or w_outfit > 0

        # Extract palette histograms from outfit_palettes batch (one per reference)
        ref_outfit_hists = None
        if outfit_palettes is not None and w_outfit > 0:
            ref_outfit_hists = []
            palette_batch_size = outfit_palettes.shape[0]
            for ri in range(num_refs):
                if ri < palette_batch_size:
                    palette_rgb = (outfit_palettes[ri].cpu().numpy() * 255).astype(np.uint8)
                    ref_outfit_hists.append(extract_palette_histogram(palette_rgb))
                else:
                    ref_outfit_hists.append(None)
            print(f"[PersonSelectorMulti] Outfit palettes: {palette_batch_size} palette(s) for {num_refs} refs")

        # Extract reference appearance features (hair color, head histogram)
        ref_appearances = None
        if use_appearance:
            ref_appearances = [self._extract_ref_appearance(analyzer, rb, device) for rb in refs]
            weight_parts = [f"face {w_face:.0%}", f"hair {w_hair:.0%}", f"head {w_head:.0%}"]
            if w_outfit > 0:
                weight_parts.append(f"outfit {w_outfit:.0%}")
            print(f"[PersonSelectorMulti] Appearance matching: {' / '.join(weight_parts)}")
            for ri, (hc, hh) in enumerate(ref_appearances):
                hair_info = f"HSV({hc[0]:.0f},{hc[1]:.0f},{hc[2]:.0f})" if hc is not None else "no hair"
                outfit_info = ""
                if ref_outfit_hists is not None:
                    outfit_info = f", palette={'yes' if ref_outfit_hists[ri] is not None else 'no'}"
                print(f"  ref {ri+1}: hair={hair_info}, histogram={'yes' if hh is not None else 'no'}{outfit_info}")

        # Prepare depth data per batch image
        from .utils.depth_refine import compute_depth_edges, deconflict_masks
        use_depth = depth_map is not None
        depth_nps = []
        depth_edges_list = []
        if use_depth:
            for b in range(batch_size):
                dm = depth_map[b].cpu().numpy()
                dnp = dm[:, :, 0] if dm.ndim == 3 else dm
                if dnp.shape[0] != h or dnp.shape[1] != w:
                    dnp = cv2.resize(dnp, (w, h), interpolation=cv2.INTER_LINEAR)
                depth_nps.append(dnp)
                depth_edges_list.append(compute_depth_edges(dnp, depth_edge_threshold))
            print(f"[PersonSelectorMulti] Depth: edge_thr={depth_edge_threshold}, carve={depth_carve_strength}, grow={depth_grow_pixels}")

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
                ref_appearances=ref_appearances, match_weights=weights,
                ref_outfit_hists=ref_outfit_hists,
                depth_edges_data=depth_edges_list[b] if use_depth else None,
                depth_np=depth_nps[b] if use_depth else None,
                depth_carve_strength=depth_carve_strength,
                depth_grow=depth_grow_pixels,
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

        # Cross-reference deconfliction: resolve overlapping masks per batch image
        if use_depth and num_refs >= 2:
            for mt in ("body", "head", "face"):
                for b in range(batch_size):
                    overlap_dict = {}
                    for ri in range(num_refs):
                        m = person_data_masks[mt][ri][b].cpu().numpy()
                        if m.sum() > 0:
                            overlap_dict[ri] = m
                    if len(overlap_dict) >= 2:
                        eb = depth_edges_list[b][1] if b < len(depth_edges_list) else None
                        resolved = deconflict_masks(overlap_dict, depth_nps[b], edges_binary=eb)
                        for ri, m in resolved.items():
                            person_data_masks[mt][ri][b] = torch.from_numpy(m)

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
            "per_face_masks": [batch_results[b]["per_face_masks"] for b in range(batch_size)],
            "face_to_ref": [batch_results[b]["face_to_ref"] for b in range(batch_size)],
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
                batch_results[b]["cur_faces"], body_masks_for_preview, h, w,
                depth_np=depth_nps[b] if use_depth else None,
                depth_edges_binary=depth_edges_list[b][1] if use_depth else None,
            )
            preview_parts.append(p)
        preview = torch.cat(preview_parts, dim=0)

        # Report: per batch item
        sim_values = []
        match_values = []
        _elapsed = int(_time.monotonic() - _t0)
        if use_appearance:
            weight_str = f"face {w_face:.0%} / hair {w_hair:.0%} / head {w_head:.0%}"
            if w_outfit > 0:
                weight_str += f" / outfit {w_outfit:.0%}"
        else:
            weight_str = "face only"
        report_lines = [
            f"Batch size: {batch_size}",
            f"Faces (total): {total_faces}",
            f"Reference persons: {num_refs}",
            f"Threshold: {'Auto' if auto_threshold else threshold} | Aggregation: {aggregation}",
            f"Match weights: {weight_str}",
            f"Matched (total): {total_matched}",
            f"Runtime: {_elapsed}s",
            "",
        ]

        matched_per_image = []
        faces_per_image = []
        for b in range(batch_size):
            br = batch_results[b]
            matched_per_image.append(str(len(br["assignments"])))
            faces_per_image.append(str(br["face_count"]))
            report_lines.append(f"  [Image {b+1}/{batch_size}] {br['face_count']} faces, {len(br['assignments'])} matched")
            for ri in range(num_refs):
                if ri in br["assignments"]:
                    fi, sim = br["assignments"][ri]
                    if b == 0:
                        sim_values.append(f"{round(sim * 100)}%")
                        match_values.append("true")
                    report_lines.append(f"    ref {ri+1}: MATCH face #{fi} (sim {round(sim * 100)}%)")
                else:
                    if b == 0:
                        has_embs = len(ref_emb_sets[ri]) > 0
                        if has_embs and br["face_count"] > 0:
                            best_sim = float(np.max(br["sim_matrix"][ri]))
                            sim_values.append(f"{round(best_sim * 100)}%")
                        else:
                            sim_values.append("0%")
                        match_values.append("false")
                    report_lines.append(f"    ref {ri+1}: no match")

        similarities_str = ", ".join(sim_values)
        matches_str = ", ".join(match_values)
        matched_faces_str = "|".join(matched_per_image)
        faces_count_str = "|".join(faces_per_image)
        report = "\n".join(report_lines)

        ui_text = f"{matched_faces_str}|{faces_count_str}|{similarities_str}"

        return {
            "ui": {"text": [ui_text]},
            "result": (person_data, face_masks_batch, head_masks_batch, body_masks_batch,
                       combined_face, combined_head, combined_body, aux_masks_batch,
                       preview, similarities_str, matches_str,
                       total_matched, total_faces, report),
        }
