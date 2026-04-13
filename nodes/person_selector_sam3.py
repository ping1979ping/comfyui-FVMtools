"""PersonSelectorSAM3 — SAM3-only person selection with text-prompt masks.

Uses SAM3 text grounding exclusively for ALL mask types (body, face, head, hair, aux).
No BiSeNet, no SAM2, no peeling, no depth-overlap-resolution needed —
SAM3 grounding produces non-overlapping masks by design.
"""
import torch
import numpy as np
import cv2

from .utils.face_analyzer import FaceAnalyzer
from .utils.matcher import compute_similarity, aggregate_similarities, build_appearance_matrix
from .utils.masker import MaskGenerator, run_sam3_grounding, assign_masks_to_faces
from .utils.tensor_utils import tensor2np, tensor2cv2, mask2tensor, np2tensor, empty_mask
from .utils.appearance import (
    extract_hair_color,
    extract_head_histogram,
    extract_palette_histogram,
    extract_palette_colors,
    extract_clothing_histogram,
    extract_clothing_colors,
    parse_match_weights,
)

# Colors for preview overlay (RGB), one per reference
_PREVIEW_COLORS = [
    (255, 50, 50), (50, 255, 50), (50, 100, 255), (255, 255, 50), (255, 50, 255),
    (50, 255, 255), (255, 160, 30), (160, 50, 255), (50, 255, 160), (255, 160, 160),
]

# SAM3 mask types with their text prompts and thresholds
SAM3_MASK_CONFIG = {
    "body":  ("person", 0.15),
    "face":  ("face", 0.40),
    "head":  ("head", 0.30),
    "hair":  ("hair", 0.40),
}

# Preset aux prompts (name → (prompt, threshold))
AUX_PRESETS = {
    "none":           None,
    "upper_body":     ("upper body", 0.30),
    "lower_body":     ("lower body", 0.30),
    "clothing":       ("person clothing", 0.25),
    "hands":          ("hand", 0.30),
    "feet":           ("feet pair", 0.40),
    "arms":           ("arm", 0.30),
    "legs":           ("leg", 0.30),
    "headless_body":  None,  # computed: body minus head
    "custom":         None,  # uses aux_custom_prompt
}


class PersonSelectorSAM3:
    """SAM3-only person selector. Uses text grounding for all mask types.

    Replaces BiSeNet + SAM2 with pure SAM3 text-prompt segmentation.
    Non-overlapping masks by design — no peeling or deconfliction needed.
    """

    _face_analyzer = None
    _last_det_size = None

    MAX_REFERENCES = 10
    AUTO_FLOOR = 0.10

    DESCRIPTION = (
        "SAM3-powered person selector — uses text grounding for all masks.\n\n"
        "Produces body, face, head (face+hair), and aux masks per person using\n"
        "SAM3's natural language segmentation. Non-overlapping masks by design.\n\n"
        "Aux mask presets: upper/lower body, clothing, hands, feet, arms, legs,\n"
        "headless body, or any custom text prompt.\n\n"
        "Connect LoadSAM3Model → sam3_model. No SAM2 or BiRefNet needed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        aux_choices = list(AUX_PRESETS.keys())
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL_CONFIG", {"tooltip": "SAM3 model from LoadSAM3Model node"}),
                "current_image": ("IMAGE", {"tooltip": "Image(s) to process. Supports batch input."}),
                "auto_threshold": ("BOOLEAN", {"default": True,
                                               "tooltip": "Auto: optimal 1:1 face-reference assignment. Off: manual threshold."}),
                "threshold": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Minimum similarity for matching. Ignored when auto is on."}),
                "guaranteed_refs": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1,
                                            "tooltip": "Force-assign first N references to best face."}),
                "aggregation": (["max", "mean", "min"],
                                {"tooltip": "How to combine similarity scores across reference images."}),
                "det_size": (["320", "480", "640", "768"],
                             {"tooltip": "Face detection resolution."}),
                "aux_preset": (aux_choices, {"default": "none",
                    "tooltip": "Preset aux mask type:\n"
                               "- upper_body / lower_body: body halves\n"
                               "- clothing: all clothing (no shoes/socks)\n"
                               "- hands / feet / arms / legs: body parts\n"
                               "- headless_body: body minus head (computed)\n"
                               "- custom: uses aux_custom_prompt below"}),
                "aux_custom_prompt": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Custom SAM3 text prompt for aux mask (only used when aux_preset='custom').\n"
                               "Any noun phrase works: 'shoes', 'necklace', 'backpack', etc."}),
                "aux_threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Confidence threshold for aux detection."}),
                "match_weights": ("STRING", {"default": "50/15/15/20",
                    "tooltip": "Face/hair/head/outfit blend: '50/15/15/20'"}),
            },
            "optional": {
                "reference_1": ("IMAGE", {"tooltip": "Reference image(s) for person 1."}),
                "outfit_palettes": ("IMAGE", {"tooltip": "Palette preview images for outfit matching."}),
                "depth_map": ("IMAGE", {"tooltip": "Depth map for render order sorting."}),
                "depth_sort_order": (["front_last", "front_first", "off"], {"default": "front_last",
                    "tooltip": "Rendering order for PersonDetailer."}),
                **{f"reference_{i}": ("IMAGE",) for i in range(2, cls.MAX_REFERENCES + 1)},
            },
        }

    RETURN_TYPES = ("PERSON_DATA", "MASK", "MASK", "MASK", "MASK", "IMAGE", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("person_data", "face_masks", "head_masks", "body_masks", "aux_masks",
                    "preview", "similarities", "matches", "matched_count", "face_count", "report")
    FUNCTION = "execute"
    CATEGORY = "FVM Tools/Face"
    OUTPUT_NODE = True

    # ── Helpers (shared with PersonSelectorMulti) ──

    def _collect_references(self, reference_1=None, **kwargs):
        refs = []
        if reference_1 is not None:
            refs.append(reference_1)
        for i in range(2, self.MAX_REFERENCES + 1):
            key = f"reference_{i}"
            if key in kwargs and kwargs[key] is not None:
                refs.append(kwargs[key])
            else:
                if not refs:
                    continue
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
            label_map = MaskGenerator._run_bisenet(rgb, face, device)
            hc = extract_hair_color(rgb, label_map)
            if hc is not None:
                hair_colors.append(hc)
            hh = extract_head_histogram(rgb, face.bbox)
            if hh is not None:
                head_hists.append(hh)
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

    def _assign_greedy(self, sim_matrix, threshold, guaranteed_refs=0):
        num_refs, num_faces = sim_matrix.shape
        assigned_faces = set()
        assignments = {}
        if guaranteed_refs > 0 and num_faces > 0:
            for ri in range(min(guaranteed_refs, num_refs)):
                best_fi = -1
                best_sim = -1.0
                for fi in range(num_faces):
                    if fi not in assigned_faces and sim_matrix[ri, fi] > best_sim:
                        best_sim = sim_matrix[ri, fi]
                        best_fi = fi
                if best_fi >= 0:
                    assignments[ri] = (best_fi, float(best_sim))
                    assigned_faces.add(best_fi)
        candidates = []
        for ri in range(num_refs):
            if ri in assignments:
                continue
            for fi in range(num_faces):
                if sim_matrix[ri, fi] >= threshold:
                    candidates.append((ri, fi, sim_matrix[ri, fi]))
        candidates.sort(key=lambda x: x[2], reverse=True)
        for ri, fi, sim in candidates:
            if ri in assignments or fi in assigned_faces:
                continue
            assignments[ri] = (fi, sim)
            assigned_faces.add(fi)
        return assignments

    # ── SAM3 Grounding Pipeline ──

    def _run_all_sam3_masks(self, sam3_config, cur_rgb, cur_faces, aux_preset, aux_custom_prompt, aux_threshold):
        """Run SAM3 grounding for all mask types + aux. Returns per-face mask dicts."""
        h, w = cur_rgb.shape[:2]
        face_count = len(cur_faces)

        # Run grounding for each mask type
        mask_results = {}
        for mask_type, (prompt, threshold) in SAM3_MASK_CONFIG.items():
            results = run_sam3_grounding(sam3_config, cur_rgb, prompt, threshold=threshold)
            assignment = assign_masks_to_faces(results, cur_faces) if results else {}
            mask_results[mask_type] = (results, assignment)

        # Run aux grounding
        aux_results = []
        aux_assignment = {}
        if aux_preset != "none":
            if aux_preset == "custom" and aux_custom_prompt.strip():
                aux_results = run_sam3_grounding(sam3_config, cur_rgb, aux_custom_prompt.strip(), threshold=aux_threshold)
                aux_assignment = assign_masks_to_faces(aux_results, cur_faces) if aux_results else {}
            elif aux_preset == "headless_body":
                pass  # computed below from body - head
            elif aux_preset in AUX_PRESETS and AUX_PRESETS[aux_preset] is not None:
                prompt, default_thresh = AUX_PRESETS[aux_preset]
                aux_results = run_sam3_grounding(sam3_config, cur_rgb, prompt, threshold=aux_threshold)
                aux_assignment = assign_masks_to_faces(aux_results, cur_faces) if aux_results else {}

        # Build per-face mask dict
        per_face = []
        for fi in range(face_count):
            masks = {}

            # Body mask
            body_data, body_assign = mask_results["body"]
            if fi in body_assign:
                masks["body"] = mask2tensor(body_data[body_assign[fi]][0])
            else:
                masks["body"] = empty_mask(h, w)

            # Face mask
            face_data, face_assign = mask_results["face"]
            if fi in face_assign:
                masks["face"] = mask2tensor(face_data[face_assign[fi]][0])
            else:
                # Fallback: generate face mask from face bbox
                masks["face"] = mask2tensor(MaskGenerator.generate_face_mask(cur_rgb, cur_faces[fi],
                                            torch.device("cuda" if torch.cuda.is_available() else "cpu")))

            # Head mask (SAM3 "head" or fallback: face + hair union)
            head_data, head_assign = mask_results["head"]
            if fi in head_assign:
                masks["head"] = mask2tensor(head_data[head_assign[fi]][0])
            else:
                # Fallback: union of face + hair
                face_np = masks["face"][0].cpu().numpy()
                hair_data, hair_assign = mask_results["hair"]
                if fi in hair_assign:
                    hair_np = hair_data[hair_assign[fi]][0]
                    head_np = np.maximum(face_np, hair_np)
                else:
                    head_np = face_np
                masks["head"] = mask2tensor(head_np)

            # Hair mask
            hair_data, hair_assign = mask_results["hair"]
            if fi in hair_assign:
                masks["hair"] = mask2tensor(hair_data[hair_assign[fi]][0])
            else:
                masks["hair"] = empty_mask(h, w)

            # Aux mask
            if aux_preset == "headless_body":
                # Computed: body minus head minus hair (dilated to catch fringe)
                body_np = masks["body"][0].cpu().numpy()
                head_np = masks["head"][0].cpu().numpy()
                hair_np = masks["hair"][0].cpu().numpy()
                # Union head+hair, dilate to cover edges SAM3 missed (long hair etc.)
                head_hair = np.maximum(head_np, hair_np)
                dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                head_hair = cv2.dilate((head_hair * 255).astype(np.uint8), dilate_k).astype(np.float32) / 255.0
                headless = np.clip(body_np - head_hair, 0, 1).astype(np.float32)
                # Clean small fragments
                headless = (headless > 0.5).astype(np.float32)
                masks["aux"] = mask2tensor(headless)
            elif fi in aux_assignment:
                masks["aux"] = mask2tensor(aux_results[aux_assignment[fi]][0])
            else:
                masks["aux"] = empty_mask(h, w)

            # Fill remaining mask types with empty (PersonDetailer expects all 9)
            for mt in ["facial_skin", "eyes", "mouth", "neck", "accessories"]:
                masks[mt] = empty_mask(h, w)

            per_face.append(masks)

        return per_face

    # ── Preview Rendering ──

    def _render_preview(self, current_image, assignments, cur_faces, per_face_masks,
                        h, w, num_refs, ref_depths=None, depth_sort_order="front_last",
                        depth_np=None):
        preview = tensor2np(current_image).copy()

        # Determine render order — includes ALL faces (matched + unmatched)
        render_items = []  # (label_idx, fi, depth) — label_idx = ri for matched, -1 for unmatched
        matched_fis = set()
        if num_refs > 0:
            for ri in range(num_refs):
                if ri in assignments:
                    fi, sim = assignments[ri]
                    depth = ref_depths.get(ri, 0.5) if ref_depths else 0.5
                    render_items.append((ri, fi, depth))
                    matched_fis.add(fi)
        # Add unmatched faces with depth from body mask
        for fi in range(len(cur_faces)):
            if fi not in matched_fis:
                depth = 0.5
                if depth_np is not None and fi < len(per_face_masks):
                    body_np = per_face_masks[fi]["body"][0].cpu().numpy()
                    body_pixels = body_np > 0.5
                    if body_pixels.any():
                        depth = float(np.percentile(depth_np[body_pixels], 85))
                render_items.append((-1, fi, depth))
        # Sort by depth
        if depth_sort_order == "front_last":
            render_items.sort(key=lambda x: x[2])
        elif depth_sort_order == "front_first":
            render_items.sort(key=lambda x: -x[2])

        # Build face→ref mapping
        fi_to_ri = {fi: ri for ri, (fi, sim) in assignments.items()} if assignments else {}

        def _paint_mask(preview, fi, color_idx):
            if fi >= len(per_face_masks):
                return preview
            color = _PREVIEW_COLORS[color_idx % len(_PREVIEW_COLORS)]
            body_mask = per_face_masks[fi]["body"]
            mask_np = (body_mask[0].cpu().numpy() * 255).astype(np.uint8)
            if mask_np.max() == 0:
                return preview
            mask_bool = mask_np > 128
            fill_color = np.array(color, dtype=np.float32)
            pf = preview.astype(np.float32)
            pf[mask_bool] = pf[mask_bool] * 0.6 + fill_color * 0.4
            preview = pf.astype(np.uint8)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            thick = max(2, int(2 * max(h, w) / 1000))
            cv2.drawContours(preview, contours, -1, (255, 255, 255), thick + 1)
            cv2.drawContours(preview, contours, -1, color, thick)
            return preview

        # Paint in render order (back→front)
        for ri, fi, depth in render_items:
            color_idx = fi_to_ri.get(fi, fi) if ri < 0 else ri
            preview = _paint_mask(preview, fi, color_idx)

        # Labels: every person gets P1..Pn, matched ones show ref number instead
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(h, w) / 1000.0 * 1.2
        for fi, face in enumerate(cur_faces):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cx = (x1 + x2) // 2

            if fi in fi_to_ri:
                ri = fi_to_ri[fi]
                sim = assignments[ri][1]
                color = _PREVIEW_COLORS[ri % len(_PREVIEW_COLORS)]
                label = f"{ri+1}"
                sim_label = f"{sim:.0%}"
            else:
                color = (160, 160, 160)
                label = f"P{fi+1}"
                sim_label = None

            # Main label
            (tw, th_text), _ = cv2.getTextSize(label, font, scale, 2)
            tx = cx - tw // 2
            ty = y1 - 5
            cv2.rectangle(preview, (tx - 3, ty - th_text - 3), (tx + tw + 3, ty + 3), (0, 0, 0), cv2.FILLED)
            cv2.putText(preview, label, (tx, ty), font, scale, color, 2, cv2.LINE_AA)

            # Sim label (only for matched)
            if sim_label:
                (sw, sh), _ = cv2.getTextSize(sim_label, font, scale * 0.6, 1)
                sx = cx - sw // 2
                sy = ty + sh + 8
                cv2.rectangle(preview, (sx - 2, sy - sh - 2), (sx + sw + 2, sy + 2), (0, 0, 0), cv2.FILLED)
                cv2.putText(preview, sim_label, (sx, sy), font, scale * 0.6, color, 1, cv2.LINE_AA)

        # Render order text — all faces with labels
        if len(render_items) > 1:
            labels = []
            for ri, fi, depth in render_items:
                if fi in fi_to_ri:
                    labels.append(str(fi_to_ri[fi] + 1))
                else:
                    labels.append(f"P{fi+1}")
            order_text = f"Render: {' > '.join(labels)} (back>front)"
            (ow, oh), _ = cv2.getTextSize(order_text, font, scale * 0.5, 1)
            ox = (w - ow) // 2
            oy = h - 10
            cv2.rectangle(preview, (ox - 3, oy - oh - 3), (ox + ow + 3, oy + 3), (0, 0, 0), cv2.FILLED)
            cv2.putText(preview, order_text, (ox, oy), font, scale * 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        return np2tensor(preview)

    # ── Main Execute ──

    def execute(self, sam3_model, current_image, auto_threshold, threshold, guaranteed_refs,
                aggregation, det_size, aux_preset="none", aux_custom_prompt="",
                aux_threshold=0.30, match_weights="50/15/15/20",
                reference_1=None, outfit_palettes=None,
                depth_map=None, depth_sort_order="front_last",
                **kwargs):
        import time as _time
        _t0 = _time.monotonic()
        det_size_int = int(det_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if PersonSelectorSAM3._face_analyzer is None or PersonSelectorSAM3._last_det_size != det_size_int:
            PersonSelectorSAM3._face_analyzer = FaceAnalyzer(det_size_int)
            PersonSelectorSAM3._last_det_size = det_size_int
        analyzer = PersonSelectorSAM3._face_analyzer

        batch_size = current_image.shape[0]
        h, w = current_image.shape[1], current_image.shape[2]
        refs = self._collect_references(reference_1, **kwargs)
        num_refs = len(refs)

        ref_emb_sets = [self._extract_ref_embeddings(analyzer, rb) for rb in refs] if refs else []
        effective_threshold = self.AUTO_FLOOR if auto_threshold else threshold

        weights = parse_match_weights(match_weights)
        w_face, w_hair, w_head, w_outfit = weights
        use_appearance = w_hair > 0 or w_head > 0 or w_outfit > 0

        # Extract palettes
        ref_outfit_hists = None
        ref_palette_colors = None
        if outfit_palettes is not None and w_outfit > 0:
            ref_outfit_hists = []
            ref_palette_colors = []
            for ri in range(num_refs):
                if ri < outfit_palettes.shape[0]:
                    palette_rgb = (outfit_palettes[ri].cpu().numpy() * 255).astype(np.uint8)
                    ref_outfit_hists.append(extract_palette_histogram(palette_rgb))
                    ref_palette_colors.append(extract_palette_colors(palette_rgb))
                else:
                    ref_outfit_hists.append(None)
                    ref_palette_colors.append([])

        ref_appearances = None
        if use_appearance and refs:
            ref_appearances = [self._extract_ref_appearance(analyzer, rb, device) for rb in refs]

        # Depth maps
        use_depth = depth_map is not None
        depth_nps = []
        if use_depth:
            for b in range(batch_size):
                d = depth_map[b].cpu().numpy()
                if d.ndim == 3:
                    d = d.mean(axis=2)
                depth_nps.append(d.astype(np.float32))

        # ── Process each batch image ──
        MASK_TYPES = ["face", "head", "body", "hair", "facial_skin", "eyes", "mouth", "neck", "accessories"]
        person_data_masks = {mt: [] for mt in MASK_TYPES}
        person_data_masks["aux"] = []
        person_data_matches = []
        all_faces_parts = []
        matched_faces_parts = []
        preview_parts = []
        ref_depths_per_batch = []
        all_per_face_masks = []
        all_face_to_ref = []
        sim_values = []
        match_values = []
        per_image_reports = []

        for b in range(batch_size):
            single = current_image[b:b+1]
            cur_bgr = tensor2cv2(single)
            cur_rgb = tensor2np(single)
            cur_faces = analyzer.detect_faces(cur_bgr)
            face_count = len(cur_faces)
            depth_np = depth_nps[b] if use_depth else None

            print(f"[PersonSelectorSAM3] Image {b+1}/{batch_size}: {face_count} faces detected")

            # Face embeddings + matching
            face_embs = [analyzer.get_embedding(f) for f in cur_faces]
            print(f"[PersonSelectorSAM3] {num_refs} refs, {len(face_embs)} face embeddings")
            if num_refs > 0 and face_embs:
                sim_matrix = self._build_similarity_matrix(ref_emb_sets, face_embs, aggregation)
                for ri in range(num_refs):
                    best_fi = int(np.argmax(sim_matrix[ri]))
                    print(f"    ref {ri+1}: best face #{best_fi} sim={sim_matrix[ri, best_fi]:.4f}")

                # Appearance matching: extract per-face features via BiSeNet
                if use_appearance and ref_appearances:
                    ref_hair_colors = [app[0] for app in ref_appearances]
                    ref_head_hists = [app[1] for app in ref_appearances]
                    face_hair_colors = []
                    face_head_hists = []
                    face_clothing_hists = []
                    face_clothing_colors = []
                    for fi, face in enumerate(cur_faces):
                        label_map = MaskGenerator._run_bisenet(cur_rgb, face, device)
                        face_hair_colors.append(extract_hair_color(cur_rgb, label_map))
                        face_head_hists.append(extract_head_histogram(cur_rgb, face.bbox))
                        if w_outfit > 0:
                            from .utils.masker import MASK_TYPE_LABELS
                            cloth_labels = MASK_TYPE_LABELS.get("neck", set()) | {14, 15, 16}
                            cloth_mask = np.isin(label_map, list(cloth_labels)).astype(np.float32)
                            head_mask_np = np.isin(label_map, list(MASK_TYPE_LABELS.get("head", set()))).astype(np.float32)
                            face_clothing_hists.append(extract_clothing_histogram(cur_rgb, cloth_mask, head_mask_np))
                            upper, lower = extract_clothing_colors(cur_rgb, cloth_mask, face.bbox)
                            face_clothing_colors.append((upper, lower))
                        else:
                            face_clothing_hists.append(None)
                            face_clothing_colors.append((None, None))

                    sim_matrix = build_appearance_matrix(
                        sim_matrix, ref_hair_colors, face_hair_colors,
                        ref_head_hists, face_head_hists, weights,
                        ref_outfit_hists=ref_outfit_hists,
                        face_clothing_hists=face_clothing_hists,
                        ref_palette_colors=ref_palette_colors,
                        face_clothing_colors=face_clothing_colors,
                    )

                if auto_threshold:
                    assignments = self._assign_greedy(sim_matrix, self.AUTO_FLOOR, guaranteed_refs)
                else:
                    assignments = self._assign_greedy(sim_matrix, threshold, guaranteed_refs)
            else:
                sim_matrix = np.zeros((num_refs, face_count)) if num_refs > 0 else np.zeros((0, 0))
                assignments = {}

            # ── SAM3 grounding: all masks at once ──
            per_face_masks = self._run_all_sam3_masks(
                sam3_model, cur_rgb, cur_faces, aux_preset, aux_custom_prompt, aux_threshold)

            # Build per-ref masks from per-face masks via assignments
            fi_to_ri = {fi: ri for ri, (fi, sim) in assignments.items()}
            ref_masks = {mt: [] for mt in MASK_TYPES}
            ref_masks["aux"] = []
            for ri in range(num_refs):
                if ri in assignments:
                    fi, sim = assignments[ri]
                    for mt in MASK_TYPES:
                        ref_masks[mt].append(per_face_masks[fi][mt] if fi < len(per_face_masks) else empty_mask(h, w))
                    ref_masks["aux"].append(per_face_masks[fi]["aux"] if fi < len(per_face_masks) else empty_mask(h, w))
                else:
                    for mt in MASK_TYPES:
                        ref_masks[mt].append(empty_mask(h, w))
                    ref_masks["aux"].append(empty_mask(h, w))

            for mt in MASK_TYPES:
                person_data_masks[mt].append(ref_masks[mt])
            person_data_masks["aux"].append(ref_masks["aux"])

            # Matches
            matches_for_image = [ri in assignments for ri in range(num_refs)]
            person_data_matches.append(matches_for_image)

            # All/matched faces masks
            if face_count > 0:
                all_fm = [per_face_masks[fi]["face"] for fi in range(face_count)]
                all_faces_mask = torch.max(torch.cat(all_fm, dim=0), dim=0, keepdim=True)[0]
            else:
                all_faces_mask = empty_mask(h, w)
            all_faces_parts.append(all_faces_mask)

            matched_fis = set(fi for fi, sim in assignments.values())
            if matched_fis:
                mfm = [per_face_masks[fi]["face"] for fi in matched_fis if fi < len(per_face_masks)]
                matched_faces_mask = torch.max(torch.cat(mfm, dim=0), dim=0, keepdim=True)[0] if mfm else empty_mask(h, w)
            else:
                matched_faces_mask = empty_mask(h, w)
            matched_faces_parts.append(matched_faces_mask)

            # Depth for render order
            depths = {}
            for ri, (fi, sim) in assignments.items():
                if use_depth:
                    body_np = per_face_masks[fi]["body"][0].cpu().numpy() if fi < len(per_face_masks) else np.zeros((h, w))
                    masked = depth_np[body_np > 0.5] if body_np.sum() > 0 else np.array([])
                    depths[ri] = float(np.percentile(masked, 85)) if len(masked) > 0 else 0.5
                else:
                    face = cur_faces[fi]
                    fy = (face.bbox[1] + face.bbox[3]) / 2
                    depths[ri] = fy / h
            ref_depths_per_batch.append(depths)

            # Per-face + face_to_ref
            all_per_face_masks.append(per_face_masks)
            all_face_to_ref.append([fi_to_ri.get(fi) for fi in range(face_count)])

            # Preview
            if num_refs > 0:
                body_list = [ref_masks["body"][ri] for ri in range(num_refs)]
            else:
                body_list = [pf["body"] for pf in per_face_masks]
            preview = self._render_preview(single, assignments, cur_faces, per_face_masks,
                                            h, w, num_refs, ref_depths=depths, depth_sort_order=depth_sort_order,
                                            depth_np=depth_np)
            preview_parts.append(preview)

            # Sim/match strings
            if num_refs > 0:
                sims = [f"{assignments[ri][1]:.0%}" if ri in assignments else "—" for ri in range(num_refs)]
                sim_values.append(", ".join(sims))
                match_values.append(str(len(assignments)))
            else:
                sim_values.append("")
                match_values.append("0")

            # Per-image report: markdown table with P1..Pn rows × ref columns
            matched_fis_report = set(fi for fi, sim in assignments.values())
            fi_to_ri_report = {fi: ri for ri, (fi, sim) in assignments.items()}

            batch_report_lines = [f"[Image {b+1}/{batch_size}] {face_count} faces, {len(assignments)} matched"]

            if num_refs > 0 and face_count > 0:
                # Markdown table header
                ref_headers = " | ".join(f"Ref{ri+1}" for ri in range(num_refs))
                batch_report_lines.append(f"Person | {ref_headers} | Match")
                batch_report_lines.append(f"--- | {'--- | ' * num_refs}---")

                for fi in range(face_count):
                    label = f"R{fi_to_ri_report[fi]+1}" if fi in fi_to_ri_report else f"P{fi+1}"
                    sims = []
                    for ri in range(num_refs):
                        s = sim_matrix[ri, fi] if sim_matrix.shape[1] > fi else 0
                        marker = " **←**" if fi_to_ri_report.get(fi) == ri else ""
                        sims.append(f"{s:.0%}{marker}")
                    sim_str = " | ".join(sims)
                    match_str = f"**→ Ref{fi_to_ri_report[fi]+1}**" if fi in fi_to_ri_report else "—"
                    batch_report_lines.append(f"{label} | {sim_str} | {match_str}")
            elif face_count > 0:
                for fi in range(face_count):
                    batch_report_lines.append(f"P{fi+1}: detected (no refs connected)")

            batch_report_lines.append("---")

            # Render order
            all_render = []
            for fi in range(face_count):
                if use_depth and fi < len(per_face_masks):
                    body_np = per_face_masks[fi]["body"][0].cpu().numpy()
                    body_px = body_np > 0.5
                    fd = float(np.percentile(depth_np[body_px], 85)) if body_px.any() else 0.5
                else:
                    fd = 0.5
                lbl = f"R{fi_to_ri_report[fi]+1}" if fi in fi_to_ri_report else f"P{fi+1}"
                all_render.append((lbl, fd))
            if depth_sort_order == "front_last":
                all_render.sort(key=lambda x: x[1])
            elif depth_sort_order == "front_first":
                all_render.sort(key=lambda x: -x[1])
            if all_render:
                order_str = " > ".join(f"{lbl}(d={d:.2f})" for lbl, d in all_render)
                batch_report_lines.append(f"Render: {order_str} (back>front)")
                batch_report_lines.append("---")

            per_image_reports.append("\n".join(batch_report_lines))

        # ── Build outputs ──
        _elapsed = int(_time.monotonic() - _t0)

        # Stack masks across batches
        final_masks = {}
        for mt in MASK_TYPES + ["aux"]:
            per_ref = []
            for ri in range(num_refs):
                batch_tensors = [person_data_masks[mt][b][ri] for b in range(batch_size)]
                per_ref.append(torch.cat(batch_tensors, dim=0))
            final_masks[mt] = per_ref

        # PERSON_DATA
        person_data = {
            "batch_size": batch_size,
            "num_references": num_refs,
            "image_height": h,
            "image_width": w,
            "matches": person_data_matches,
            "all_faces_mask": torch.cat(all_faces_parts, dim=0),
            "matched_faces_mask": torch.cat(matched_faces_parts, dim=0),
            "per_face_masks": all_per_face_masks,
            "face_to_ref": all_face_to_ref,
            "ref_depths": ref_depths_per_batch,
            "depth_sort_order": depth_sort_order,
        }
        for mt in MASK_TYPES:
            person_data[mt] = final_masks[mt]
        if aux_preset != "none":
            person_data["aux_masks"] = final_masks["aux"]

        # Combined masks (OR across all refs)
        def _combine_masks(mask_list):
            if not mask_list:
                return empty_mask(h, w).expand(batch_size, -1, -1)
            return torch.max(torch.stack(mask_list, dim=0), dim=0)[0]

        combined_face = _combine_masks(final_masks["face"])
        combined_head = _combine_masks(final_masks["head"])
        combined_body = _combine_masks(final_masks["body"])
        combined_aux = _combine_masks(final_masks["aux"]) if aux_preset != "none" else empty_mask(h, w).expand(batch_size, -1, -1)

        preview_out = torch.cat(preview_parts, dim=0)
        sims_str = " | ".join(sim_values)
        matches_str = " | ".join(match_values)
        matched_count = sum(len(person_data_matches[b]) and any(person_data_matches[b]) for b in range(batch_size))
        total_faces = sum(len(all_per_face_masks[b]) for b in range(batch_size))

        aux_info = f" | Aux: {aux_preset}" + (f" ('{aux_custom_prompt}')" if aux_preset == "custom" else "") if aux_preset != "none" else ""
        report_lines = [
            f"## PersonSelectorSAM3",
            f"**{total_faces}** faces | **{num_refs}** refs | **{matched_count}** matched | {_elapsed}s",
            f"",
            f"Threshold: {'Auto' if auto_threshold else threshold} | Aggregation: {aggregation}{aux_info}",
            f"Weights: face {w_face:.0%} / hair {w_hair:.0%} / head {w_head:.0%} / outfit {w_outfit:.0%}",
            f"",
            f"---",
            f"",
        ] + per_image_reports
        report = "\n".join(report_lines)
        print(f"[PersonSelectorSAM3] Done in {_elapsed}s")

        return (person_data, combined_face, combined_head, combined_body, combined_aux,
                preview_out, sims_str, matches_str, matched_count, total_faces, report)
