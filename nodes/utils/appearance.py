"""Appearance-based matching signals: hair color distance and head crop histogram."""

import cv2
import numpy as np


def extract_hair_color(image_rgb: np.ndarray, label_map: np.ndarray, min_pixels: int = 100):
    """Extract average hair color from BiSeNet label map.

    Args:
        image_rgb: Full image as RGB numpy array (H, W, 3), uint8.
        label_map: BiSeNet label map (H, W), uint8. Label 17 = hair.
        min_pixels: Minimum hair pixels required for a valid result.

    Returns:
        np.ndarray of shape (3,) with mean HSV values [H, S, V], or None if too few pixels.
    """
    hair_mask = (label_map == 17)
    if hair_mask.sum() < min_pixels:
        return None

    hair_pixels = image_rgb[hair_mask]  # (N, 3) RGB
    # Convert to HSV for illumination-robust comparison
    hair_pixels_bgr = hair_pixels[:, ::-1].reshape(1, -1, 3).astype(np.uint8)
    hair_hsv = cv2.cvtColor(hair_pixels_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    # Use median instead of mean for robustness against highlights/shadows
    return np.median(hair_hsv, axis=0).astype(np.float32)


def hair_color_distance(hsv1, hsv2):
    """Compute normalized distance between two hair colors in HSV space.

    Returns: float in [0, 1] where 0 = identical, 1 = maximally different.
    Returns 1.0 if either input is None (no hair detected).
    """
    if hsv1 is None or hsv2 is None:
        return 1.0

    # Hue is circular (0-180 in OpenCV), weight it most
    h_diff = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0])) / 90.0
    s_diff = abs(hsv1[1] - hsv2[1]) / 255.0
    v_diff = abs(hsv1[2] - hsv2[2]) / 255.0

    # Weighted: hue matters most, then saturation, then value (brightness)
    return float(np.clip(h_diff * 0.5 + s_diff * 0.3 + v_diff * 0.2, 0.0, 1.0))


def hair_color_similarity(hsv1, hsv2):
    """Compute hair color similarity (1 = identical, 0 = maximally different)."""
    return 1.0 - hair_color_distance(hsv1, hsv2)


def extract_head_histogram(image_rgb: np.ndarray, face_bbox, expand: float = 0.4, bins=(30, 32)):
    """Compute HSV histogram of the head crop region.

    Args:
        image_rgb: Full image as RGB numpy array (H, W, 3), uint8.
        face_bbox: Face bounding box [x1, y1, x2, y2].
        expand: Expansion factor around the face bbox to capture hair/neck.
        bins: (hue_bins, saturation_bins) for the 2D histogram.

    Returns:
        np.ndarray — normalized 2D histogram, or None if crop is empty.
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    bw, bh = x2 - x1, y2 - y1

    # Expand bbox to include hair and upper body
    cx1 = max(0, int(x1 - bw * expand))
    cy1 = max(0, int(y1 - bh * expand))
    cx2 = min(w, int(x2 + bw * expand))
    cy2 = min(h, int(y2 + bh * expand * 0.5))  # less expansion downward

    crop = image_rgb[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    crop_bgr = crop[:, ::-1].copy()
    crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([crop_hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def head_histogram_similarity(hist1, hist2):
    """Compute similarity between two head histograms using correlation.

    Returns: float in [0, 1] where 1 = identical, 0 = completely different.
    Returns 0.0 if either histogram is None.
    """
    if hist1 is None or hist2 is None:
        return 0.0

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # CORREL returns [-1, 1], normalize to [0, 1]
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def parse_match_weights(weight_string: str):
    """Parse weight string like '60/20/20' or '0.6/0.2/0.2' into (face, hair, head) tuple.

    Accepts:
        - '60/20/20' (percentages, auto-normalized)
        - '0.6/0.2/0.2' (fractions)
        - '70/15/15' (any ratio)

    Returns: tuple of 3 floats summing to 1.0. Falls back to (1.0, 0.0, 0.0) on error.
    """
    try:
        parts = [float(p.strip()) for p in weight_string.strip().split("/")]
        if len(parts) != 3:
            return (1.0, 0.0, 0.0)

        total = sum(parts)
        if total <= 0:
            return (1.0, 0.0, 0.0)

        return tuple(p / total for p in parts)
    except (ValueError, ZeroDivisionError):
        return (1.0, 0.0, 0.0)


def combined_similarity(face_sim: float, hair_sim: float, head_sim: float,
                         weights: tuple) -> float:
    """Compute weighted combined similarity score.

    Args:
        face_sim: ArcFace cosine similarity [0, 1].
        hair_sim: Hair color similarity [0, 1].
        head_sim: Head histogram similarity [0, 1].
        weights: (face_weight, hair_weight, head_weight) summing to 1.0.

    Returns: float in [0, 1].
    """
    w_face, w_hair, w_head = weights
    return float(w_face * face_sim + w_hair * hair_sim + w_head * head_sim)
