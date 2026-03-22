import torch
import numpy as np

from .utils.face_analyzer import FaceAnalyzer
from .utils.matcher import find_best_match
from .utils.masker import MaskGenerator
from .utils.tensor_utils import tensor2np, tensor2cv2, np2tensor, mask2tensor, empty_mask, apply_gaussian_blur, fill_mask_holes


class PersonSelector:
    """ComfyUI node that identifies a reference person in images via face embedding matching."""

    # Class-level cache (shared with PersonSelectorMulti)
    _face_analyzer = None
    _last_det_size = None

    DESCRIPTION = (
        "Matches a reference person in the current image using InsightFace (ArcFace) embeddings.\n\n"
        "Outputs similarity score, match boolean, and optional mask (face/head/body).\n"
        "Body mask requires SAM model from Impact Pack SAMLoader (opt).\n"
        "Face/head masks use BiSeNet segmentation.\n\n"
        "threshold: Minimum cosine similarity for a match (0.6-0.7 recommended).\n"
        "aggregation: How to combine scores across multiple reference images.\n"
        "det_size: Face detection resolution — higher finds smaller faces but uses more VRAM."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_image": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Minimum cosine similarity to count as match (0.6-0.7 recommended)"}),
                "aggregation": (["max", "mean", "min"],
                                {"tooltip": "How to combine similarity scores across multiple reference images"}),
                "mask_mode": (["none", "face", "head", "body"],
                              {"tooltip": "none=no mask, face=skin only, head=face+hair, body=full body (needs SAM)"}),
                "mask_fill_holes": ("BOOLEAN", {"default": True,
                                               "tooltip": "Fill holes inside the mask (closes gaps in segmentation)"}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100,
                                      "tooltip": "Gaussian blur radius for mask edges"}),
                "det_size": (["320", "480", "640", "768"],
                             {"tooltip": "Face detection resolution — higher finds smaller faces but uses more VRAM"}),
            },
            "optional": {
                "sam_model": ("SAM_MODEL", {"tooltip": "(opt) SAM model from Impact Pack SAMLoader — required for body mask mode"}),
            },
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "MASK", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("similarity", "match", "mask", "best_reference", "face_count", "matched_face_index", "report")
    FUNCTION = "execute"
    CATEGORY = "FVM Tools/Face"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Matches a reference person in the current image using InsightFace (ArcFace) embeddings.\n\n"
        "Outputs similarity score, match boolean, and optional mask (face/head/body).\n"
        "Body mask requires SAM model from Impact Pack SAMLoader.\n"
        "Face/head masks use BiSeNet segmentation.\n\n"
        "threshold: minimum cosine similarity for a match (0.6-0.7 recommended)\n"
        "aggregation: how to combine scores across multiple reference images\n"
        "det_size: face detection resolution — higher finds smaller faces but uses more VRAM"
    )

    def execute(self, current_image, reference_images, threshold, aggregation,
                mask_mode, mask_fill_holes, mask_blur, det_size, sam_model=None):
        det_size_int = int(det_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if PersonSelector._face_analyzer is None or PersonSelector._last_det_size != det_size_int:
            PersonSelector._face_analyzer = FaceAnalyzer(det_size_int)
            PersonSelector._last_det_size = det_size_int
        analyzer = PersonSelector._face_analyzer

        h, w = current_image.shape[1], current_image.shape[2]

        # Extract reference embeddings
        ref_embs = []
        ref_count = reference_images.shape[0]
        best_ref_face_indices = []

        for i in range(ref_count):
            ref_frame = reference_images[i:i+1]
            ref_bgr = tensor2cv2(ref_frame)
            ref_faces = analyzer.detect_faces(ref_bgr)
            if ref_faces:
                emb = analyzer.get_embedding(ref_faces[0])
                ref_embs.append(emb)
                best_ref_face_indices.append(i)

        if not ref_embs:
            report = f"No faces detected in {ref_count} reference image(s)."
            return {
                "ui": {"text": ["0.0000|0|-1"]},
                "result": (0.0, False, empty_mask(h, w), reference_images[0:1], 0, -1, report),
            }

        # Detect faces in current image
        cur_bgr = tensor2cv2(current_image)
        cur_rgb = tensor2np(current_image)
        cur_faces = analyzer.detect_faces(cur_bgr)
        face_count = len(cur_faces)

        if face_count == 0:
            report = "No faces detected in current image."
            return {
                "ui": {"text": ["0.0000|0|-1"]},
                "result": (0.0, False, empty_mask(h, w), reference_images[0:1], 0, -1, report),
            }

        face_embs = [analyzer.get_embedding(f) for f in cur_faces]
        face_idx, similarity, ref_idx = find_best_match(face_embs, ref_embs, aggregation)
        is_match = similarity >= threshold

        best_ref_img_idx = best_ref_face_indices[ref_idx] if ref_idx >= 0 else 0
        best_reference = reference_images[best_ref_img_idx:best_ref_img_idx+1]

        # Generate mask
        if mask_mode == "none" or not is_match:
            mask = empty_mask(h, w)
        else:
            matched_face = cur_faces[face_idx]
            if mask_mode == "face":
                mask_np = MaskGenerator.generate_face_mask(cur_rgb, matched_face, device)
                mask = mask2tensor(mask_np)
            elif mask_mode == "head":
                mask_np = MaskGenerator.generate_head_mask(cur_rgb, matched_face, device)
                mask = mask2tensor(mask_np)
            elif mask_mode == "body":
                mask_np = MaskGenerator.generate_body_mask(cur_rgb, matched_face, sam_model)
                mask = mask2tensor(mask_np)
            else:
                mask = empty_mask(h, w)

        if mask_fill_holes and mask_mode != "none" and is_match:
            mask = fill_mask_holes(mask)
        if mask_blur > 0:
            mask = apply_gaussian_blur(mask, mask_blur)

        report_lines = [
            f"Faces in current image: {face_count}",
            f"Reference embeddings: {len(ref_embs)} (from {ref_count} images)",
            f"Best match: face #{face_idx} with similarity {similarity:.4f}",
            f"Threshold: {threshold} | Aggregation: {aggregation}",
            f"Match: {'YES' if is_match else 'NO'}",
            f"Mask mode: {mask_mode}",
        ]
        report = "\n".join(report_lines)

        return {
            "ui": {"text": [f"{similarity:.4f}|{face_count}|{face_idx}"]},
            "result": (similarity, is_match, mask, best_reference, face_count, face_idx, report),
        }
