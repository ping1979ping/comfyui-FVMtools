import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2

import folder_paths
from ...parsing import BiSeNet

try:
    from ...core.config import get_model_path
except ImportError:
    from core.config import get_model_path


# BiSeNet 19-class labels:
# 0=background, 1=skin, 2=left_brow, 3=right_brow, 4=left_eye, 5=right_eye,
# 6=glasses, 7=left_ear, 8=right_ear, 9=earrings, 10=nose,
# 11=mouth_interior, 12=upper_lip, 13=lower_lip, 14=neck,
# 15=necklace, 16=cloth, 17=hair, 18=hat

# Label groups
FACE_LABELS = {1, 2, 3, 4, 5, 10, 11, 12, 13}       # skin, brows, eyes, nose, mouth, lips
HEAD_EXTRA_LABELS = {6, 7, 8, 9, 14, 17, 18}          # glasses, ears, earrings, neck, hair, hat
HEAD_LABELS = FACE_LABELS | HEAD_EXTRA_LABELS

# Additional mask types (all derived from BiSeNet, no extra cost)
HAIR_LABELS = {17}
FACIAL_SKIN_LABELS = {1}                               # skin only (no eyes, brows, lips)
EYES_LABELS = {4, 5}
MOUTH_LABELS = {11, 12, 13}
NECK_LABELS = {14}
ACCESSORIES_LABELS = {6, 9, 15}                        # glasses, earrings, necklace

# Map from mask type name to label set
MASK_TYPE_LABELS = {
    "face": FACE_LABELS,
    "head": HEAD_LABELS,
    "hair": HAIR_LABELS,
    "facial_skin": FACIAL_SKIN_LABELS,
    "eyes": EYES_LABELS,
    "mouth": MOUTH_LABELS,
    "neck": NECK_LABELS,
    "accessories": ACCESSORIES_LABELS,
}

# All available mask type names (for dropdowns)
ALL_MASK_TYPES = ["face", "head", "body", "hair", "facial_skin", "eyes", "mouth", "neck", "accessories"]
BISENET_MASK_TYPES = [t for t in ALL_MASK_TYPES if t != "body"]  # types derivable from BiSeNet


def generate_all_masks_for_face(cur_rgb, face, device, sam_model, mask_fill_holes, mask_blur,
                                depth_edges_data=None, depth_np=None,
                                depth_carve_strength=0.8, depth_grow=30,
                                other_faces=None):
    """Generate all 9 mask types for a single face. Shared by PersonSelectorMulti and PersonDataRefiner.

    Args:
        cur_rgb: (H, W, 3) RGB numpy uint8
        face: InsightFace face object with .bbox
        device: torch device
        sam_model: SAM model for body mask
        mask_fill_holes: bool
        mask_blur: int (blur radius)
        depth_edges_data: optional tuple (edge_magnitude, edges_binary) from compute_depth_edges
        depth_np: optional (H, W) float32 depth map [0,1]
        depth_carve_strength: how strongly depth edges cut masks (0=off, 1=full)
        depth_grow: max gap fill pixels between edges

    Returns:
        dict {mask_type: torch.Tensor [1, H, W]} for all 9 types in ALL_MASK_TYPES
    """
    from .depth_refine import refine_mask_with_depth, DEPTH_REFINABLE_MASKS
    from .tensor_utils import mask2tensor, fill_mask_holes, apply_gaussian_blur
    from .mask_utils import clean_mask_crumbs

    use_depth = depth_edges_data is not None and depth_np is not None

    label_map = MaskGenerator._run_bisenet(cur_rgb, face, device)
    masks = {}

    for mask_type, labels in MASK_TYPE_LABELS.items():
        mask_np = np.isin(label_map, list(labels)).astype(np.float32)
        if use_depth and mask_type in DEPTH_REFINABLE_MASKS:
            mask_np = refine_mask_with_depth(mask_np, depth_edges_data, depth_np,
                                             depth_carve_strength, depth_grow)
        mask = mask2tensor(mask_np)
        if mask_fill_holes:
            mask = fill_mask_holes(mask)
        if mask_blur > 0:
            mask = apply_gaussian_blur(mask, mask_blur)
        masks[mask_type] = mask

    # Body mask via SAM (with negative prompts for other faces and head mask for scoring)
    head_mask_np = masks["head"][0].cpu().numpy() if "head" in masks else None
    body_mask_np = MaskGenerator.generate_body_mask(cur_rgb, face, sam_model,
                                                     other_faces=other_faces,
                                                     head_mask=head_mask_np)
    body_mask_np = clean_mask_crumbs(body_mask_np, min_area_fraction=0.005)
    if use_depth:
        body_mask_np = refine_mask_with_depth(body_mask_np, depth_edges_data, depth_np,
                                              depth_carve_strength, depth_grow)
    mask = mask2tensor(body_mask_np)
    if mask_fill_holes:
        mask = fill_mask_holes(mask)
    if mask_blur > 0:
        mask = apply_gaussian_blur(mask, mask_blur)
    masks["body"] = mask

    return masks


class MaskGenerator:
    """Generates face/head/body masks using BiSeNet and SAM."""

    _bisenet = None
    _bisenet_device = None

    @classmethod
    def _load_bisenet(cls, device):
        if cls._bisenet is not None and cls._bisenet_device == device:
            return cls._bisenet

        # Search for parsing_bisenet.pth in standard ComfyUI model directories
        weight_path = None
        search_dirs = [
            os.path.join(folder_paths.models_dir, "gfpgan"),
            os.path.join(folder_paths.models_dir, "facedetection"),
            os.path.join(folder_paths.models_dir, "facerestore_models"),
        ]

        # Also check folder_paths registered paths for face-related categories
        for category in ["gfpgan", "facedetection", "facerestore_models"]:
            try:
                for p in folder_paths.get_folder_paths(category):
                    if p not in search_dirs:
                        search_dirs.append(p)
            except Exception:
                pass

        for search_dir in search_dirs:
            candidate = os.path.join(search_dir, "parsing_bisenet.pth")
            if os.path.exists(candidate):
                weight_path = candidate
                break

        # INI fallback: check outfit_config.ini [models] bisenet_path
        if weight_path is None:
            ini_path = get_model_path("bisenet_path")
            if ini_path and os.path.isfile(ini_path):
                weight_path = ini_path

        if weight_path is None:
            raise FileNotFoundError(
                "parsing_bisenet.pth not found.\n"
                "Searched in: " + ", ".join(search_dirs) + "\n"
                "You can also set the path in outfit_config.ini under [models] bisenet_path.\n"
                "Download: https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth\n"
                "Mirror:   https://huggingface.co/leonelhs/facexlib/resolve/main/parsing_bisenet.pth"
            )

        net = BiSeNet(num_class=19)
        net.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        net.eval()
        net.to(device)
        cls._bisenet = net
        cls._bisenet_device = device
        print(f"[FVMTools] BiSeNet loaded from {weight_path}")
        return net

    @classmethod
    def _run_bisenet(cls, image_rgb: np.ndarray, face, device) -> np.ndarray:
        """Run BiSeNet on face crop, return label map at original image size."""
        h, w = image_rgb.shape[:2]
        net = cls._load_bisenet(device)

        # Get bbox with 30% padding
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.3), int(bh * 0.3)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = image_rgb[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return np.zeros((h, w), dtype=np.float32)

        # Preprocess: resize to 512x512, normalize
        crop_resized = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
        crop_t = torch.from_numpy(crop_resized.astype(np.float32) / 255.0)
        crop_t = crop_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 512, 512)
        crop_t = crop_t.to(device)

        with torch.no_grad():
            out = net(crop_t)[0]  # (1, 19, 512, 512)

        parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)  # (512, 512)

        # Scale back to crop size
        crop_h, crop_w = cy2 - cy1, cx2 - cx1
        parsing_resized = cv2.resize(parsing, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        # Place into full-size label map
        label_map = np.zeros((h, w), dtype=np.uint8)
        label_map[cy1:cy2, cx1:cx2] = parsing_resized
        return label_map

    @classmethod
    def generate_face_mask(cls, image_rgb: np.ndarray, face, device) -> np.ndarray:
        """Generate face-only mask using BiSeNet."""
        label_map = cls._run_bisenet(image_rgb, face, device)
        mask = np.isin(label_map, list(FACE_LABELS)).astype(np.float32)
        return mask

    @classmethod
    def generate_head_mask(cls, image_rgb: np.ndarray, face, device) -> np.ndarray:
        """Generate head mask (face + hair + ears etc.) using BiSeNet."""
        label_map = cls._run_bisenet(image_rgb, face, device)
        mask = np.isin(label_map, list(HEAD_LABELS)).astype(np.float32)
        return mask

    @classmethod
    def _get_sam_predictor(cls, sam_model):
        """Resolve SAM wrapper from sam_model."""
        if hasattr(sam_model, 'sam_wrapper'):
            return sam_model.sam_wrapper
        return sam_model

    @classmethod
    def _estimate_body_points(cls, face, h, w, other_faces=None):
        """Estimate body keypoints from face bbox for SAM prompts.

        Args:
            other_faces: list of other face objects — their centers become
                         negative SAM prompts to exclude other persons.
        """
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        face_w = x2 - x1
        face_h = y2 - y1
        cx = (x1 + x2) // 2

        points = [
            [cx, (y1 + y2) // 2],
            [cx, min(h - 1, y2 + int(face_h * 0.8))],
            [cx, min(h - 1, y2 + int(face_h * 2.0))],
            [cx, min(h - 1, y2 + int(face_h * 3.5))],
            [max(0, cx - face_w), min(h - 1, y2 + int(face_h * 1.5))],
            [min(w - 1, cx + face_w), min(h - 1, y2 + int(face_h * 1.5))],
        ]
        plabs = [1] * len(points)

        # Negative prompts at other face centers
        if other_faces:
            for of in other_faces:
                ox1, oy1, ox2, oy2 = [int(v) for v in of.bbox]
                ocx = (ox1 + ox2) // 2
                ocy = (oy1 + oy2) // 2
                points.append([ocx, ocy])
                plabs.append(0)  # negative prompt

        body_half_w = face_w * 2
        bbox = [
            max(0, cx - body_half_w),
            max(0, y1),
            min(w, cx + body_half_w),
            min(h, y2 + face_h * 6),
        ]
        return points, plabs, bbox

    @classmethod
    def generate_body_mask(cls, image_rgb: np.ndarray, face, sam_model=None,
                           other_faces=None, head_mask=None) -> np.ndarray:
        """Generate body mask using SAM with positive and negative prompts.

        Args:
            other_faces: list of other detected face objects — used as negative SAM prompts
            head_mask: optional [H,W] float32 head mask for better candidate selection
        """
        h, w = image_rgb.shape[:2]

        if sam_model is None:
            return cls.generate_body_bbox_fallback((h, w), face)

        try:
            predictor = cls._get_sam_predictor(sam_model)
            is_sam2 = hasattr(sam_model, 'image_predictor') or hasattr(sam_model, 'config')
            points, plabs, bbox = cls._estimate_body_points(face, h, w, other_faces=other_faces)

            sam_bbox = None if is_sam2 else bbox

            result_masks = predictor.predict(image_rgb, points, plabs, sam_bbox, 0.3)

            if result_masks is not None and len(result_masks) > 0:
                # Compute face/head center for scoring
                fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
                face_cx = (fx1 + fx2) / 2
                face_cy = (fy1 + fy2) / 2
                face_area = (fx2 - fx1) * (fy2 - fy1)
                max_body_area = face_area * 40  # body can't be more than 40x face area

                candidates = []
                for m in result_masks:
                    if isinstance(m, torch.Tensor):
                        m = m.cpu().numpy()
                    m = m.squeeze().astype(np.float32)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    area = np.sum(m > 0.5)
                    if area < face_area * 0.5:
                        continue  # too small
                    candidates.append((m, area))

                if not candidates:
                    print("[FVMTools] SAM: no valid candidates, using bbox fallback")
                    return cls.generate_body_bbox_fallback((h, w), face)

                # Score candidates: prefer mask that contains face, has reasonable size,
                # and best overlaps with head_mask if available
                best_mask = None
                best_score = -1
                for m, area in candidates:
                    # Must contain the face center
                    if m[int(face_cy), int(face_cx)] < 0.5:
                        continue

                    # Size penalty: prefer smaller masks (less likely to include others)
                    size_score = 1.0 - min(area / max_body_area, 1.0) if area <= max_body_area else 0.0

                    # Head overlap bonus
                    head_score = 0.0
                    if head_mask is not None:
                        head_pixels = np.sum(head_mask > 0.5)
                        if head_pixels > 0:
                            overlap = np.sum((m > 0.5) & (head_mask > 0.5))
                            head_score = overlap / head_pixels  # 1.0 = mask fully contains head

                    score = size_score * 0.4 + head_score * 0.6 + 0.01  # small base score
                    if score > best_score:
                        best_score = score
                        best_mask = m

                if best_mask is not None:
                    return best_mask

                # Fallback: smallest candidate that contains face
                for m, area in sorted(candidates, key=lambda x: x[1]):
                    if m[int(face_cy), int(face_cx)] > 0.5:
                        return m

                return candidates[0][0]

            print("[FVMTools] SAM returned no masks, using body bbox fallback")
            return cls.generate_body_bbox_fallback((h, w), face)
        except Exception as e:
            print(f"[FVMTools] SAM failed: {e}, using body bbox fallback")
            return cls.generate_body_bbox_fallback((h, w), face)

    @classmethod
    def generate_body_bbox_fallback(cls, shape: tuple, face) -> np.ndarray:
        """Estimate body region from face bbox."""
        h, w = shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        face_w = x2 - x1
        cx = (x1 + x2) // 2

        body_half_w = face_w * 2
        bx1 = max(0, cx - body_half_w)
        bx2 = min(w, cx + body_half_w)
        by1 = max(0, y1)
        by2 = h

        mask = np.zeros((h, w), dtype=np.float32)
        mask[by1:by2, bx1:bx2] = 1.0
        return mask

    @classmethod
    def generate_bbox_mask(cls, shape: tuple, face, expand: float = 1.0) -> np.ndarray:
        """Generate a simple bounding box mask, optionally expanded."""
        h, w = shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        bw, bh = x2 - x1, y2 - y1

        if expand != 1.0:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            new_bw, new_bh = bw * expand, bh * expand
            x1 = int(cx - new_bw / 2)
            y1 = int(cy - new_bh / 2)
            x2 = int(cx + new_bw / 2)
            y2 = int(cy + new_bh / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        mask = np.zeros((h, w), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        return mask
