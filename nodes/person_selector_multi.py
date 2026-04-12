import torch
import numpy as np
import cv2

from .utils.face_analyzer import FaceAnalyzer
from .utils.matcher import compute_similarity, aggregate_similarities, build_appearance_matrix
from .utils.masker import MaskGenerator, FACE_LABELS, HEAD_LABELS, MASK_TYPE_LABELS, BISENET_MASK_TYPES, generate_all_masks_for_face, split_person_mask_by_anchors, split_person_mask_by_seeds
from .utils.tensor_utils import tensor2np, tensor2cv2, mask2tensor, np2tensor, empty_mask, apply_gaussian_blur, fill_mask_holes
from .utils.mask_utils import clean_mask_crumbs, fill_mask_holes_2d, expand_mask, feather_mask
from .utils.yolo_detector import get_available_yolo_models, detect_objects, assign_detections_to_references
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
        "All references are optional — with none connected, all faces go to the generic slot.\n\n"
        "Supports batch input: pass multiple images and get PERSON_DATA for PersonDetailer.\n\n"
        "Faces are assigned exclusively: each face matches at most one reference.\n\n"
        "match_weights controls the scoring blend: 'face/hair/head/outfit' (default 50/15/15/20).\n"
        "Set to '100/0/0/0' for pure face matching, or '60/20/20/0' to disable outfit.\n\n"
        "Outfit matching: connect palette_preview images as a batch to outfit_palettes.\n"
        "Compares palette color distribution with detected clothing colors (BiSeNet).\n\n"
        "Built-in YOLO detection: select a model from aux_model to detect body parts.\n"
        "Detected parts are assigned to references via body mask overlap\n"
        "and stored as 'aux' masks in PERSON_DATA for PersonDetailer.\n"
        "Drop YOLO .pt models into models/ultralytics/segm/ to add more detectors."
    )

    @classmethod
    def INPUT_TYPES(cls):
        yolo_models = ["none"] + get_available_yolo_models()
        return {
            "required": {
                "sam_model": ("SAM_MODEL", {"tooltip": "SAM model from Impact Pack SAMLoader — required for body masks"}),
                "current_image": ("IMAGE", {"tooltip": "Image(s) to search for faces. Supports batch input — each image is processed independently."}),
                "auto_threshold": ("BOOLEAN", {"default": True,
                                               "tooltip": "Auto: finds optimal 1:1 face-reference assignment (ignores threshold). Off: uses manual threshold."}),
                "threshold": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Minimum cosine similarity to count as match. Ignored when auto_threshold is on."}),
                "guaranteed_refs": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1,
                                            "tooltip": "Force-assign the first N references to their best matching face, ignoring threshold.\n"
                                                        "0 = off (default). 2 = Ref1 and Ref2 always get assigned.\n"
                                                        "Useful when key characters must always be detailed."}),
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
                "aux_model": (yolo_models, {
                    "tooltip": "YOLO model for body-part detection. Runs per batch image internally.\n\n"
                               "Select a model from models/ultralytics/ to detect body parts\n"
                               "(hands, persons, etc.). Detections are assigned to references via\n"
                               "body mask overlap and stored as aux_masks in PERSON_DATA.\n\n"
                               "none = no body-part detection (default)\n\n"
                               "Drop .pt files into models/ultralytics/segm/ to add more models."}),
                "aux_confidence": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05,
                                              "tooltip": "YOLO detection confidence threshold.\n\n"
                                                         "Lower = more detections (may include false positives)\n"
                                                         "Higher = fewer, more confident detections\n\n"
                                                         "0.25-0.35 recommended for most models."}),
                "aux_label": ("STRING", {"default": "",
                                          "tooltip": "Filter YOLO detections by class label (substring match).\n\n"
                                                     "Empty = keep all detected classes (default)\n"
                                                     "Comma-separated: 'person' or 'leg,foot'\n\n"
                                                     "Substring match: 'leg' hits 'Left-leg', 'right_leg', etc.\n"
                                                     "The class list for the selected model is displayed\n"
                                                     "above the Matching section after model selection."}),
                "aux_fill_holes": ("BOOLEAN", {"default": False,
                                                "tooltip": "Fill holes inside YOLO aux masks (closes interior gaps).\n"
                                                           "Off by default — segm models usually produce solid masks."}),
                "aux_expand_pixels": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                                                "tooltip": "Dilate YOLO aux masks by N pixels (elliptical kernel).\n"
                                                           "Useful when segm masks hug the silhouette too tightly\n"
                                                           "for inpainting. 0 = no growth."}),
                "aux_blend_pixels": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                                               "tooltip": "Gaussian blur radius for YOLO aux mask edges (symmetric).\n"
                                                          "Blurs ~N pixels inward AND outward from the current edge.\n"
                                                          "If you want the grown shape to stay fully opaque, set\n"
                                                          "aux_expand_pixels at least as large as aux_blend_pixels."}),
                "match_weights": ("STRING", {"default": "50/15/15/20",
                                             "tooltip": "Matching weight blend: face/hair/head/outfit.\n"
                                                        "Controls how much each signal contributes to the final similarity score.\n\n"
                                                        "  50/15/15/20 — balanced with outfit (default)\n"
                                                        "  60/20/20/0 — no outfit matching\n"
                                                        "  100/0/0/0 — pure face matching\n"
                                                        "  40/15/15/30 — heavy outfit weight\n\n"
                                                        "3 values also work (outfit=0): 60/20/20 equals 60/20/20/0.\n\n"
                                                        "Hair = BiSeNet hair color (HSV). Head = head crop histogram.\n"
                                                        "Outfit = clothing region vs. palette color distribution.\n"
                                                        "Values are auto-normalized, so 3/1/1/1 equals 50/17/17/17."}),
            },
            "optional": {
                "reference_1": ("IMAGE", {"tooltip": "Reference image(s) for person 1. Pass a batch of images of the same person for better matching accuracy.\n\n"
                                                      "Optional — if no references connected, all detected faces go to the generic slot in PersonDetailer."}),
                "outfit_palettes": ("IMAGE", {"tooltip": "Palette preview image batch for outfit color matching.\n"
                                                          "Image[0] = palette for reference 1, image[1] = for reference 2, etc.\n"
                                                          "If batch is smaller than number of references, remaining refs skip outfit matching.\n"
                                                          "Connect palette_preview outputs from Color Palette Generator.\n"
                                                          "Use match_weights with 4 values to control outfit weight (e.g. 50/15/15/20)."}),
                "person_mask": ("MASK", {"tooltip":
                    "Foreground person mask (BiRefNet, RMBG-2.0, or similar).\n\n"
                    "When connected, acts as a hard clip for ALL mask types — face, head, "
                    "body, aux, and BiSeNet label seeds are zeroed outside this silhouette. "
                    "Body masks are replaced entirely by the BiRefNet envelope (SAM/seed_grow skipped).\n\n"
                    "Single mask [1,H,W] or per-image batch [B,H,W]. If batch size is 1 but "
                    "current_image has a larger batch, the mask is broadcast to all images.\n\n"
                    "Multi-person split: the foreground is divided per reference using depth "
                    "(closest depth wins) when depth_map is also connected, or by face-center "
                    "distance otherwise. When no references are connected, the split falls back "
                    "to detected face count — each detected face becomes a pseudo-reference."}),
                "depth_map": ("IMAGE", {"tooltip": "Depth map batch from Depth Anything V2 or similar. Improves masks via edge carving and cross-reference deconfliction."}),
                "depth_edge_threshold": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.30, "step": 0.01,
                                                    "tooltip": "Depth gradient threshold for edge detection. Lower = more edges detected."}),
                "depth_carve_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "How strongly depth edges cut masks. 0=off, 1=full cut."}),
                "depth_grow_pixels": ("INT", {"default": 30, "min": 0, "max": 200, "step": 5,
                                               "tooltip": "Gap filling between depth edges. 0 = no growing."}),
                "body_mask_mode": (["auto", "seed_grow", "sam"], {"default": "auto",
                                    "tooltip": "Body mask strategy:\n"
                                               "- auto: seed_grow (recommended)\n"
                                               "- seed_grow: BiSeNet + SAM seed, carved by image/depth edges\n"
                                               "- sam: legacy SAM-only body segmentation"}),
                "depth_sort_order": (["front_last", "front_first", "off"], {"default": "front_last",
                                      "tooltip": "Rendering order for PersonDetailer:\n"
                                                 "- front_last: closest person rendered last (correct for depth_map with bright=near)\n"
                                                 "- front_first: closest person rendered first (for inverted depth maps)\n"
                                                 "- off: no sorting, uses slot order"}),
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
                    continue  # keep scanning — ref_1 might be empty but ref_2 connected
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

    def _assign_greedy(self, sim_matrix, threshold, guaranteed_refs=0):
        num_refs, num_faces = sim_matrix.shape
        assigned_faces = set()
        assignments = {}

        # Phase 1: Force-assign guaranteed refs (best face, no threshold)
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
                    print(f"    Ref{ri+1} force-assigned → face {best_fi+1} (sim={best_sim:.3f})")

        # Phase 2: Greedy assignment for remaining refs (with threshold)
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

    def _generate_all_masks(self, cur_rgb, face, device, sam_model, mask_fill_holes, mask_blur,
                            depth_edges_data=None, depth_np=None,
                            depth_carve_strength=0.8, depth_grow=30,
                            other_faces=None, body_mask_mode="auto",
                            person_mask_envelope=None):
        """Generate all mask types. Delegates to shared generate_all_masks_for_face()."""
        return generate_all_masks_for_face(
            cur_rgb, face, device, sam_model, mask_fill_holes, mask_blur,
            depth_edges_data=depth_edges_data, depth_np=depth_np,
            depth_carve_strength=depth_carve_strength, depth_grow=depth_grow,
            other_faces=other_faces, body_mask_mode=body_mask_mode,
            person_mask_envelope=person_mask_envelope,
        )

    def _render_mask_layers(self, current_image, assignments, cur_faces,
                             body_masks_list, h, w, num_refs, ref_depths=None,
                             outfit_palettes=None, per_face_masks=None):
        """Render mask overlay: original image with semi-transparent colored body silhouettes.

        Shows all reference masks layered on top of the original image with clear
        color coding, contour outlines, and reference number labels.
        Includes render order info when depth data is available.

        In no-refs mode (num_refs=0), each detected face is treated as its own
        pseudo-reference labeled F1/F2/...  — body masks come from body_masks_list
        indexed by face idx, and labels use the same _PREVIEW_COLORS palette.

        When per_face_masks is provided and num_refs>0, unmatched faces (detected
        but not assigned to any reference) are drawn with a dimmed gray overlay
        and labeled '?' so users can see which faces are going to the generic slot.
        """
        preview = tensor2np(current_image).copy()

        # In no-refs mode, synthesize pseudo-assignments (fi → (fi, 1.0)) so the
        # label-drawing code below can iterate uniformly. Labels get an "F" prefix
        # and no similarity score.
        face_mode = num_refs == 0 and cur_faces
        if face_mode:
            effective_assignments = {fi: (fi, 1.0) for fi in range(len(cur_faces))}
            effective_count = len(cur_faces)
        else:
            effective_assignments = assignments
            effective_count = num_refs

        # Paint all masks as colored overlays (back-to-front by slot index).
        # Slot index = ref index (refs mode) OR face index (no-refs mode).
        for ri in range(effective_count):
            color = _PREVIEW_COLORS[ri % len(_PREVIEW_COLORS)]
            if ri >= len(body_masks_list):
                continue
            mask_np = (body_masks_list[ri][0].cpu().numpy() * 255).astype(np.uint8)
            if mask_np.max() == 0:
                continue

            mask_bool = mask_np > 128
            fill_color = np.array(color, dtype=np.float32)

            # 40% opacity fill — strong enough to see, transparent enough to see image
            preview_float = preview.astype(np.float32)
            preview_float[mask_bool] = preview_float[mask_bool] * 0.6 + fill_color * 0.4
            preview = preview_float.astype(np.uint8)

            # Contour outlines — thickness scales with image size
            contour_thick = max(2, int(2 * max(h, w) / 1000))
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(preview, contours, -1, (255, 255, 255), contour_thick + 1)
            cv2.drawContours(preview, contours, -1, color, contour_thick)

        # Scale factor for all text — based on image size, 2x larger than before
        img_scale = max(h, w) / 1000.0  # 1.0 at 1000px, 2.0 at 2000px, etc.
        base_font = img_scale * 1.2  # doubled from ~0.6

        # Reference number labels on matched faces (or F-labels in no-refs face mode)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for ri, (fi, sim) in effective_assignments.items():
            color = _PREVIEW_COLORS[ri % len(_PREVIEW_COLORS)]
            face = cur_faces[fi]
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cx = (x1 + x2) // 2
            label = f"F{ri + 1}" if face_mode else str(ri + 1)

            font_scale = max(base_font, min(base_font * 2.5, (x2 - x1) / 40.0))
            thickness = max(2, int(font_scale * 2))
            (tw, th_text), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            tx = cx - tw // 2
            ty = max(th_text + 4, y1 - int(8 * img_scale))

            pad = int(6 * img_scale)
            cv2.rectangle(preview, (tx - pad, ty - th_text - pad), (tx + tw + pad, ty + pad), (0, 0, 0), cv2.FILLED)
            cv2.putText(preview, label, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

            # Similarity percentage below label (ref mode only — face mode has no sim)
            if not face_mode:
                sim_label = f"{round(sim * 100)}%"
                sim_scale = font_scale * 0.6
                sim_thick = max(1, int(sim_scale * 2))
                (stw, sth), _ = cv2.getTextSize(sim_label, font, sim_scale, sim_thick)
                stx = cx - stw // 2
                sty = ty + sth + int(10 * img_scale)
                sp = int(4 * img_scale)
                cv2.rectangle(preview, (stx - sp, sty - sth - sp), (stx + stw + sp, sty + sp), (0, 0, 0), cv2.FILLED)
                cv2.putText(preview, sim_label, (stx, sty), font, sim_scale, color, sim_thick, cv2.LINE_AA)

            # Palette swatch overlay — centered above face label
            if outfit_palettes is not None and ri < outfit_palettes.shape[0]:
                pal_rgb = (outfit_palettes[ri].cpu().numpy() * 255).astype(np.uint8)
                pw = max(40, x2 - x1)
                ph = max(10, int(pw * pal_rgb.shape[0] / max(1, pal_rgb.shape[1])))
                pal_small = cv2.resize(pal_rgb, (pw, ph), interpolation=cv2.INTER_AREA)
                px = max(0, min(w - pw, cx - pw // 2))
                py = max(0, ty - th_text - pad - ph - int(4 * img_scale))
                border = max(1, int(2 * img_scale))
                cv2.rectangle(preview, (px - border, py - border),
                              (px + pw + border, py + ph + border), (0, 0, 0), cv2.FILLED)
                # Clamp paste region to image bounds
                paste_h = min(ph, h - py)
                paste_w = min(pw, w - px)
                if paste_h > 0 and paste_w > 0:
                    preview[py:py+paste_h, px:px+paste_w] = pal_small[:paste_h, :paste_w]

        # Unmatched faces: detected but not assigned to any reference.
        # Draw them with a dimmed gray overlay and "?" label so users know
        # which faces go to PersonDetailer's generic slot.
        if not face_mode and per_face_masks is not None:
            matched_fis = {fi for fi, sim in assignments.values()}
            unmatched_color = (160, 160, 160)  # neutral gray
            for fi, face in enumerate(cur_faces):
                if fi in matched_fis:
                    continue
                # Body mask from per_face_masks
                if fi < len(per_face_masks) and "body" in per_face_masks[fi]:
                    umask_np = (per_face_masks[fi]["body"][0].cpu().numpy() * 255).astype(np.uint8)
                    if umask_np.max() == 0:
                        continue

                    umask_bool = umask_np > 128
                    # 25% opacity — dimmer than matched refs (40%) to visually distinguish
                    preview_float = preview.astype(np.float32)
                    fill = np.array(unmatched_color, dtype=np.float32)
                    preview_float[umask_bool] = preview_float[umask_bool] * 0.75 + fill * 0.25
                    preview = preview_float.astype(np.uint8)

                    # Dashed-style contour (thin, gray)
                    contour_thick = max(1, int(1.5 * max(h, w) / 1000))
                    contours, _ = cv2.findContours(umask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(preview, contours, -1, unmatched_color, contour_thick)

                    # "?" label above face
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    cx = (x1 + x2) // 2
                    label = "?"
                    font_scale = max(base_font, min(base_font * 2.5, (x2 - x1) / 40.0))
                    thickness = max(2, int(font_scale * 2))
                    (tw, th_text), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    tx = cx - tw // 2
                    ty = max(th_text + 4, y1 - int(8 * img_scale))
                    pad = int(6 * img_scale)
                    cv2.rectangle(preview, (tx - pad, ty - th_text - pad),
                                  (tx + tw + pad, ty + pad), (40, 40, 40), cv2.FILLED)
                    cv2.putText(preview, label, (tx, ty), font, font_scale,
                                unmatched_color, thickness, cv2.LINE_AA)

        # Render order text at bottom-left — 4x larger than before
        if ref_depths and assignments:
            sorted_refs = sorted(assignments.keys(), key=lambda ri: ref_depths.get(ri, 0.5))
            order_parts = []
            for ri in sorted_refs:
                order_parts.append(str(ri + 1))
            order_text = "Render: " + " > ".join(order_parts) + " (back>front)"

            order_scale = h / 800.0  # text height ~2.5% of image height, always readable
            order_thick = max(2, int(order_scale * 2))
            (otw, oth), _ = cv2.getTextSize(order_text, font, order_scale, order_thick)
            ox = int(15 * img_scale)
            oy = h - int(15 * img_scale)
            op = int(8 * img_scale)
            cv2.rectangle(preview, (ox - op, oy - oth - op), (ox + otw + op, oy + op), (0, 0, 0), cv2.FILLED)
            cv2.putText(preview, order_text, (ox, oy), font, order_scale, (220, 220, 220), order_thick, cv2.LINE_AA)

        return np2tensor(preview)

    def _process_single_image(self, single_image, analyzer, ref_emb_sets, num_refs,
                               sam_model, aggregation, effective_threshold,
                               mask_fill_holes, mask_blur, device,
                               ref_appearances=None, match_weights=(1.0, 0.0, 0.0, 0.0),
                               ref_outfit_hists=None, ref_palette_colors=None,
                               depth_edges_data=None, depth_np=None,
                               depth_carve_strength=0.8, depth_grow=30,
                               body_mask_mode="auto", guaranteed_refs=0,
                               person_mask_np=None):
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

            # Extract clothing data for outfit matching
            # BiSeNet label 16 = cloth — reuses label map from above, no extra cost.
            face_clothing_hists = None
            face_clothing_colors = None
            if w_outfit > 0 and ref_outfit_hists is not None:
                face_clothing_hists = []
                face_clothing_colors = []
                for fi, face in enumerate(cur_faces):
                    cloth_mask = (face_label_maps[fi] == 16).astype(np.float32)
                    head_mask_np = np.zeros_like(cloth_mask)
                    face_clothing_hists.append(
                        extract_clothing_histogram(cur_rgb, cloth_mask, head_mask_np)
                    )
                    upper, lower = extract_clothing_colors(cur_rgb, cloth_mask, face.bbox)
                    face_clothing_colors.append((upper, lower))
                    u_str = f"H{upper[0]:.0f}/S{upper[1]:.0f}/V{upper[2]:.0f}" if upper is not None else "none"
                    l_str = f"H{lower[0]:.0f}/S{lower[1]:.0f}/V{lower[2]:.0f}" if lower is not None else "none"
                    print(f"  Face{fi+1} clothing: upper={u_str}, lower={l_str}")

            sim_matrix = build_appearance_matrix(
                face_sim_matrix, ref_hair_colors, face_hair_colors,
                ref_head_hists, face_head_hists, match_weights,
                ref_outfit_hists=ref_outfit_hists,
                face_clothing_hists=face_clothing_hists,
                ref_palette_colors=ref_palette_colors,
                face_clothing_colors=face_clothing_colors,
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

        assignments = self._assign_greedy(sim_matrix, effective_threshold, guaranteed_refs=guaranteed_refs)

        # Build per-face envelopes from person_mask (BiRefNet foreground split per face).
        # Preferred split: SAM body masks as seeds (distance transform) — SAM generates
        # per-person body masks with negative prompts for cross-person separation, so
        # each BiRefNet pixel gets assigned to the person whose actual body silhouette
        # is closest (not just whose face center is closest, which produces diagonal
        # Voronoi cuts for standing figures).
        # Fallback: face-center Voronoi when SAM isn't available.
        face_envelopes = {}  # fi -> [H,W] float32
        if person_mask_np is not None and face_count > 0:
            if sam_model is not None:
                # Generate per-face SAM body masks once, upfront, with negative prompts
                # at all other face centers for cross-person separation.
                seed_masks = []
                for fi, face in enumerate(cur_faces):
                    others = [f for j, f in enumerate(cur_faces) if j != fi]
                    sam_body = MaskGenerator.generate_body_mask(
                        cur_rgb, face, sam_model, other_faces=others)
                    sam_body = clean_mask_crumbs(sam_body, min_area_fraction=0.005)
                    # Fill holes in the SAM seed — SAM's negative prompts carve out
                    # the front person from the back person's mask, creating holes
                    # that would cause the distance-transform split to assign those
                    # pixels to the wrong person. Contour fill + morph close fixes this.
                    sam_uint8 = (sam_body * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(sam_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    filled = np.zeros_like(sam_uint8)
                    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
                    # Morphological close to bridge narrow gaps between SAM fragments
                    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, close_kernel)
                    sam_body = (filled / 255.0).astype(np.float32)
                    seed_masks.append(sam_body)
                envs = split_person_mask_by_seeds(person_mask_np, seed_masks)
                split_method = "SAM-seed distance transform"
            else:
                # Fallback: face-center anchors (Voronoi)
                anchors = []
                for face in cur_faces:
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    a = {"center": (cx, cy), "bbox": (x1, y1, x2, y2), "depth": None}
                    if depth_np is not None and y2 > y1 and x2 > x1:
                        a["depth"] = float(np.median(depth_np[y1:y2, x1:x2]))
                    anchors.append(a)
                envs = split_person_mask_by_anchors(person_mask_np, anchors, depth_np=depth_np)
                split_method = "face-center Voronoi (no SAM)"
            for fi, env in enumerate(envs):
                face_envelopes[fi] = env
            print(f"[PersonSelectorMulti] person_mask split into {len(envs)} envelopes "
                  f"via {split_method}")

        # Collect all mask types per reference
        from .utils.masker import ALL_MASK_TYPES
        masks_per_type = {mt: [] for mt in ALL_MASK_TYPES}

        bisenet_seeds_per_ref = {}  # {ri: [H,W] float32} for deconfliction priority
        for ri in range(num_refs):
            if ri in assignments:
                fi, sim = assignments[ri]
                others = [f for j, f in enumerate(cur_faces) if j != fi]
                masks = self._generate_all_masks(cur_rgb, cur_faces[fi], device, sam_model, mask_fill_holes, mask_blur,
                                                  depth_edges_data=depth_edges_data, depth_np=depth_np,
                                                  depth_carve_strength=depth_carve_strength, depth_grow=depth_grow,
                                                  other_faces=others, body_mask_mode=body_mask_mode,
                                                  person_mask_envelope=face_envelopes.get(fi))
                for mt in ALL_MASK_TYPES:
                    masks_per_type[mt].append(masks.get(mt, empty_mask(h, w)))
                # Store bisenet seed for deconfliction priority
                if "_bisenet_seed" in masks:
                    bisenet_seeds_per_ref[ri] = masks["_bisenet_seed"]
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
                                                      other_faces=others, body_mask_mode=body_mask_mode,
                                                      person_mask_envelope=face_envelopes.get(fi))
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
            "bisenet_seeds": bisenet_seeds_per_ref,
            "face_envelopes": face_envelopes,  # fi -> [H,W] when person_mask connected
        }

    def execute(self, sam_model, current_image, auto_threshold, threshold, guaranteed_refs,
                aggregation, mask_fill_holes, mask_blur, det_size, aux_mask_type="none",
                aux_model="none", aux_confidence=0.35, aux_label="",
                aux_fill_holes=False, aux_expand_pixels=0, aux_blend_pixels=0,
                match_weights="50/15/15/20",
                reference_1=None,
                outfit_palettes=None,
                person_mask=None,
                depth_map=None, depth_edge_threshold=0.05, depth_carve_strength=0.8, depth_grow_pixels=30,
                body_mask_mode="auto",
                depth_sort_order="front_last",
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

        ref_emb_sets = [self._extract_ref_embeddings(analyzer, rb) for rb in refs] if refs else []
        effective_threshold = self.AUTO_FLOOR if auto_threshold else threshold

        # Parse appearance weights (3 or 4 values)
        weights = parse_match_weights(match_weights)
        w_face, w_hair, w_head, w_outfit = weights
        use_appearance = w_hair > 0 or w_head > 0 or w_outfit > 0

        # Extract palette data from outfit_palettes batch (one per reference)
        ref_outfit_hists = None
        ref_palette_colors = None
        if outfit_palettes is not None and w_outfit > 0:
            ref_outfit_hists = []
            ref_palette_colors = []
            palette_batch_size = outfit_palettes.shape[0]
            for ri in range(num_refs):
                if ri < palette_batch_size:
                    palette_rgb = (outfit_palettes[ri].cpu().numpy() * 255).astype(np.uint8)
                    ref_outfit_hists.append(extract_palette_histogram(palette_rgb))
                    ref_palette_colors.append(extract_palette_colors(palette_rgb))
                else:
                    ref_outfit_hists.append(None)
                    ref_palette_colors.append([])
            print(f"[PersonSelectorMulti] Outfit palettes: {palette_batch_size} palette(s) for {num_refs} refs")
            for ri, colors in enumerate(ref_palette_colors):
                if colors:
                    color_strs = [f"H{c[0]:.0f}/S{c[1]:.0f}/V{c[2]:.0f}" for c in colors[:3]]
                    print(f"  Ref{ri+1} palette colors: {', '.join(color_strs)}")

        # Extract reference appearance features (hair color, head histogram)
        ref_appearances = None
        if use_appearance and refs:
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

        # Prepare person_mask (BiRefNet/RMBG foreground) per batch image.
        # Single mask broadcasts to full batch; otherwise per-image.
        person_mask_np_list = []
        if person_mask is not None:
            pm = person_mask
            if pm.dim() == 2:
                pm = pm.unsqueeze(0)
            pm_batch = pm.shape[0]
            for b in range(batch_size):
                idx = min(b, pm_batch - 1)
                arr = pm[idx].cpu().numpy().astype(np.float32)
                if arr.shape[0] != h or arr.shape[1] != w:
                    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
                person_mask_np_list.append(arr)
            fg_frac = np.mean([float((m > 0.5).mean()) for m in person_mask_np_list])
            print(f"[PersonSelectorMulti] person_mask connected: broadcast={pm_batch == 1 and batch_size > 1}, "
                  f"mean foreground = {fg_frac * 100:.1f}%")

        # Resolve auto body_mask_mode
        effective_body_mode = body_mask_mode
        if body_mask_mode == "auto":
            effective_body_mode = "seed_grow"
        guaranteed_str = f", guaranteed={guaranteed_refs}" if guaranteed_refs > 0 else ""
        print(f"[PersonSelectorMulti] batch_size={batch_size}, refs={num_refs}, "
              f"auto_threshold={auto_threshold}, effective={effective_threshold}, "
              f"body_mode={effective_body_mode}" + (" (auto)" if body_mask_mode == "auto" else "") + guaranteed_str)

        # Process each image in the batch
        batch_results = []
        for b in range(batch_size):
            single = current_image[b:b+1]
            result = self._process_single_image(
                single, analyzer, ref_emb_sets, num_refs,
                sam_model, aggregation, effective_threshold,
                mask_fill_holes, mask_blur, device,
                ref_appearances=ref_appearances, match_weights=weights,
                ref_outfit_hists=ref_outfit_hists, ref_palette_colors=ref_palette_colors,
                depth_edges_data=depth_edges_list[b] if use_depth else None,
                depth_np=depth_nps[b] if use_depth else None,
                depth_carve_strength=depth_carve_strength,
                depth_grow=depth_grow_pixels,
                body_mask_mode=effective_body_mode,
                guaranteed_refs=guaranteed_refs,
                person_mask_np=person_mask_np_list[b] if person_mask_np_list else None,
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

        # Note: cross-reference mask deconfliction (carving overlap from the farther
        # person) has been removed. It caused stripe/fragment artifacts on occluded
        # people (visible as horizontal cuts in the back person's body mask). This is
        # redundant because:
        #   1. PersonDetailer handles occlusion via back-to-front render order
        #      (depth_sort_order), painting each person in full and layering front-over-back.
        #   2. With person_mask (BiRefNet), the SAM-seed split already produces
        #      exclusive per-person envelopes with no overlap.
        #   3. Without person_mask, SAM with negative prompts minimizes overlap.
        # Each person keeps their FULL body mask including occluded parts so
        # PersonDetailer can inpaint them completely before layering.

        for b in range(batch_size):
            matches_for_image = [ri in batch_results[b]["assignments"] for ri in range(num_refs)]
            person_data_matches.append(matches_for_image)

        all_faces_masks = torch.cat([br["all_faces_mask"] for br in batch_results], dim=0)
        matched_faces_masks = torch.cat([br["matched_faces_mask"] for br in batch_results], dim=0)

        # Compute per-ref depth for rendering order (back-to-front)
        # Uses depth map if available, else face Y-position as proxy
        ref_depths_per_batch = []
        for b in range(batch_size):
            depths = {}
            br = batch_results[b]
            for ri, (fi, sim) in br["assignments"].items():
                if use_depth:
                    body_mask_np = person_data_masks["body"][ri][b].cpu().numpy()
                    masked = depth_nps[b][body_mask_np > 0.5] if body_mask_np.sum() > 0 else np.array([])
                    # Use 85th percentile instead of median so that limbs
                    # reaching toward the camera (arms in front of another
                    # person) pull the depth value forward.  Bright = near.
                    depths[ri] = float(np.percentile(masked, 85)) if len(masked) > 0 else 0.5
                else:
                    # Fallback: face Y center (higher Y = closer to camera in most perspectives)
                    face = br["cur_faces"][fi]
                    fy = (face.bbox[1] + face.bbox[3]) / 2
                    depths[ri] = fy / h  # normalize to 0-1 (0=top/far, 1=bottom/close)
            ref_depths_per_batch.append(depths)

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
            "ref_depths": ref_depths_per_batch,
            "depth_sort_order": depth_sort_order,
        }
        # Add all mask types: face_masks, head_masks, body_masks, hair_masks, etc.
        for mt in ALL_MASK_TYPES:
            person_data[f"{mt}_masks"] = person_data_masks[mt]

        # Built-in YOLO body-part detection (optional)
        if aux_model != "none":
            aux_per_ref = [[] for _ in range(num_refs)]  # list of [1,H,W] per ref per batch
            aux_unassigned = []  # [1,H,W] per batch
            aux_part_counts = []  # [{ri: count}, ...] per batch

            # Post-processing helper — applied to each merged YOLO mask so both
            # person_data["aux_masks"] (consumed by PersonDetailer) and the
            # aux_masks output port see the cleaned mask.
            def _aux_post(mask_2d):
                # Order: grow → fill → blur. Growing first can bridge small
                # gaps between adjacent detections (e.g. upper leg + foot),
                # and the subsequent fill_holes then closes the enclosed area,
                # yielding a solid silhouette before edge feathering.
                if aux_expand_pixels > 0:
                    mask_2d = expand_mask(mask_2d, aux_expand_pixels)
                if aux_fill_holes:
                    mask_2d = fill_mask_holes_2d(mask_2d)
                if aux_blend_pixels > 0:
                    mask_2d = feather_mask(mask_2d, aux_blend_pixels)
                return mask_2d

            for b in range(batch_size):
                single_image = current_image[b]  # [H, W, C]

                # Run YOLO detection on this image
                detections = detect_objects(single_image, aux_model,
                                            confidence=aux_confidence, label_filter=aux_label)

                # Assign detections to references via body mask overlap
                body_masks_list = person_data.get("body_masks", [])
                det_assignments = assign_detections_to_references(
                    detections, body_masks_list, num_refs, b)

                # Envelope clip mask for this batch image (union of all face envelopes
                # if person_mask was provided). Used to zero any YOLO detection that
                # straggles outside the BiRefNet foreground.
                env_clip_tensor = None
                if person_mask_np_list:
                    env_union = person_mask_np_list[b] > 0.5
                    env_clip_tensor = torch.from_numpy(env_union.astype(np.float32))

                counts = {}
                for ri in range(num_refs):
                    parts = det_assignments.get(ri, [])
                    counts[ri] = len(parts)
                    if parts:
                        merged = torch.max(torch.stack(parts), dim=0)[0]  # [H, W]
                        if env_clip_tensor is not None:
                            merged = merged * env_clip_tensor
                        merged = _aux_post(merged)
                        aux_per_ref[ri].append(merged.unsqueeze(0))  # [1, H, W]
                    else:
                        aux_per_ref[ri].append(torch.zeros(1, h, w, dtype=torch.float32))

                # Unassigned parts
                unassigned_parts = det_assignments.get(-1, [])
                if unassigned_parts:
                    merged_unassigned = torch.max(torch.stack(unassigned_parts), dim=0)[0]
                    if env_clip_tensor is not None:
                        merged_unassigned = merged_unassigned * env_clip_tensor
                    merged_unassigned = _aux_post(merged_unassigned)
                    aux_unassigned.append(merged_unassigned.unsqueeze(0))
                else:
                    aux_unassigned.append(torch.zeros(1, h, w, dtype=torch.float32))

                aux_part_counts.append(counts)

                total_parts = sum(counts.values()) + len(unassigned_parts)
                labels_found = set(d["label"] for d in detections)
                print(f"[PersonSelectorMulti] Image {b+1}: {total_parts} detections ({', '.join(labels_found) or 'none'}), "
                      f"{sum(1 for c in counts.values() if c > 0)} refs matched")

            # Store in PERSON_DATA
            person_data["aux_masks"] = [torch.cat(aux_per_ref[ri], dim=0) for ri in range(num_refs)]  # [B, H, W] per ref
            person_data["aux_unassigned_masks"] = torch.cat(aux_unassigned, dim=0)  # [B, H, W]
            person_data["aux_part_counts"] = aux_part_counts

            # Upgrade body masks if YOLO detections are substantial person silhouettes
            yolo_upgraded = 0
            for b in range(batch_size):
                for ri in range(num_refs):
                    aux_mask = person_data["aux_masks"][ri][b]
                    aux_area = aux_mask.sum().item()
                    current_body = person_data["body_masks"][ri][b]
                    current_area = current_body.sum().item()
                    if aux_area > current_area * 0.3 and aux_area > 1000:
                        person_data["body_masks"][ri][b] = aux_mask
                        yolo_upgraded += 1
            if yolo_upgraded > 0:
                print(f"[PersonSelectorMulti] Upgraded {yolo_upgraded} body masks from YOLO detector")

        # Legacy mask outputs: face, head, body stacked across batch.
        # With refs connected → one slot per ref. Without refs → fall back to
        # one slot per detected face (face_count mode) so outputs aren't empty.
        all_face_out = []
        all_head_out = []
        all_body_out = []
        if num_refs > 0:
            for b in range(batch_size):
                mpt = batch_results[b]["masks_per_type"]
                for ri in range(num_refs):
                    all_face_out.append(mpt["face"][ri])
                    all_head_out.append(mpt["head"][ri])
                    all_body_out.append(mpt["body"][ri])
        else:
            # No-refs fallback: use per-face masks so each detected face becomes a slot
            for b in range(batch_size):
                for pf in batch_results[b]["per_face_masks"]:
                    all_face_out.append(pf["face"])
                    all_head_out.append(pf["head"])
                    all_body_out.append(pf["body"])

        if all_face_out:
            face_masks_batch = torch.cat(all_face_out, dim=0)
            head_masks_batch = torch.cat(all_head_out, dim=0)
            body_masks_batch = torch.cat(all_body_out, dim=0)
        else:
            face_masks_batch = empty_mask(h, w)
            head_masks_batch = empty_mask(h, w)
            body_masks_batch = empty_mask(h, w)

        total_matched = sum(len(br["assignments"]) for br in batch_results)
        total_faces = sum(br["face_count"] for br in batch_results)

        # combined_* outputs: OR of all slots. In no-refs mode we still want this to
        # reflect "any detected face/body" so we use total_faces as the trigger.
        if total_matched > 0 or (num_refs == 0 and total_faces > 0):
            combined_face = torch.max(face_masks_batch, dim=0, keepdim=True)[0]
            combined_head = torch.max(head_masks_batch, dim=0, keepdim=True)[0]
            combined_body = torch.max(body_masks_batch, dim=0, keepdim=True)[0]
        else:
            combined_face = empty_mask(h, w)
            combined_head = empty_mask(h, w)
            combined_body = empty_mask(h, w)

        # Auxiliary masks output:
        #   1. If aux_mask_type is set → BiSeNet mask of that type
        #   2. Else if YOLO aux_model ran → merged YOLO detections per ref/face
        #   3. Else → empty
        if aux_mask_type != "none" and aux_mask_type in MASK_TYPE_LABELS:
            aux_list = []
            for b in range(batch_size):
                mpt = batch_results[b]["masks_per_type"]
                if num_refs > 0:
                    for ri in range(num_refs):
                        aux_list.append(mpt[aux_mask_type][ri])
                else:
                    for pf in batch_results[b]["per_face_masks"]:
                        aux_list.append(pf[aux_mask_type])
            aux_masks_batch = torch.cat(aux_list, dim=0) if aux_list else empty_mask(h, w)
        elif aux_model != "none" and "aux_masks" in person_data and num_refs > 0:
            aux_list = []
            for b in range(batch_size):
                for ri in range(num_refs):
                    aux_list.append(person_data["aux_masks"][ri][b:b+1])
            aux_masks_batch = torch.cat(aux_list, dim=0) if aux_list else empty_mask(h, w)
        else:
            aux_masks_batch = empty_mask(h, w)

        # Preview: mask overlay with render order (uses deconflicted masks).
        # In no-refs mode, source body masks from per_face_masks so each detected
        # face gets its own colored overlay and F-label.
        preview_parts = []
        for b in range(batch_size):
            if num_refs > 0:
                body_masks_for_preview = [person_data_masks["body"][ri][b:b+1] for ri in range(num_refs)]
            else:
                body_masks_for_preview = [pf["body"] for pf in batch_results[b]["per_face_masks"]]
            p = self._render_mask_layers(
                current_image[b:b+1], batch_results[b]["assignments"],
                batch_results[b]["cur_faces"], body_masks_for_preview, h, w, num_refs,
                ref_depths=ref_depths_per_batch[b],
                outfit_palettes=outfit_palettes,
                per_face_masks=batch_results[b]["per_face_masks"])
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
