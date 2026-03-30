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
    """Parse weight string into (face, hair, head, outfit) tuple.

    Accepts 3 or 4 values:
        - '60/20/20' → (face, hair, head, 0.0)  — backward compatible
        - '50/15/15/20' → (face, hair, head, outfit)
        - Any ratio, auto-normalized to sum 1.0

    Returns: tuple of 4 floats summing to 1.0. Falls back to (1.0, 0.0, 0.0, 0.0) on error.
    """
    try:
        parts = [float(p.strip()) for p in weight_string.strip().split("/")]
        if len(parts) == 3:
            parts.append(0.0)
        elif len(parts) != 4:
            return (1.0, 0.0, 0.0, 0.0)

        total = sum(parts)
        if total <= 0:
            return (1.0, 0.0, 0.0, 0.0)

        return tuple(p / total for p in parts)
    except (ValueError, ZeroDivisionError):
        return (1.0, 0.0, 0.0, 0.0)


def combined_similarity(face_sim: float, hair_sim: float, head_sim: float,
                         weights: tuple, outfit_sim: float = 0.0) -> float:
    """Compute weighted combined similarity score.

    Args:
        face_sim: ArcFace cosine similarity [0, 1].
        hair_sim: Hair color similarity [0, 1].
        head_sim: Head histogram similarity [0, 1].
        weights: (face_weight, hair_weight, head_weight, outfit_weight) summing to 1.0.
        outfit_sim: Outfit color similarity [0, 1].

    Returns: float in [0, 1].
    """
    w_face, w_hair, w_head = weights[0], weights[1], weights[2]
    w_outfit = weights[3] if len(weights) > 3 else 0.0
    return float(w_face * face_sim + w_hair * hair_sim + w_head * head_sim + w_outfit * outfit_sim)


def extract_palette_histogram(palette_image_rgb: np.ndarray, bins=(30, 32)):
    """Extract HSV histogram from a palette swatch image with primary/secondary weighting.

    The palette preview has color blocks arranged left-to-right (primary first).
    Splits into vertical columns and weights: primary=3x, secondary=2x, rest=1x.

    Args:
        palette_image_rgb: Palette preview as RGB numpy array (H, W, 3), uint8.
        bins: (hue_bins, saturation_bins) for the 2D histogram.

    Returns:
        np.ndarray — normalized 2D HSV histogram, or None if image is empty.
    """
    if palette_image_rgb is None or palette_image_rgb.size == 0:
        return None

    img_h = palette_image_rgb.shape[0]
    img_w = palette_image_rgb.shape[1]
    # Use top 70% to avoid text labels at the bottom
    swatch_h = int(img_h * 0.70)
    swatch = palette_image_rgb[:swatch_h, :, :]

    if swatch.size == 0:
        return None

    # Estimate number of color blocks from aspect ratio
    sw = swatch.shape[1]
    sh = swatch.shape[0]
    num_blocks = max(1, round(sw / max(1, sh)))

    if num_blocks <= 1:
        # Single block or can't split — use flat histogram (original behavior)
        swatch_bgr = swatch[:, ::-1].copy()
        swatch_hsv = cv2.cvtColor(swatch_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([swatch_hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    # Primary=3x, secondary=2x, rest=1x
    block_weights = [3.0, 2.0] + [1.0] * max(0, num_blocks - 2)
    combined_hist = np.zeros((bins[0], bins[1]), dtype=np.float32)
    block_w = sw // num_blocks

    for i in range(num_blocks):
        x_start = i * block_w
        x_end = (i + 1) * block_w if i < num_blocks - 1 else sw
        block = swatch[:, x_start:x_end, :]
        if block.size == 0:
            continue
        block_bgr = block[:, ::-1].copy()
        block_hsv = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2HSV)
        h_block = cv2.calcHist([block_hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
        cv2.normalize(h_block, h_block)
        combined_hist += h_block * block_weights[i]

    cv2.normalize(combined_hist, combined_hist)
    return combined_hist


def extract_clothing_histogram(image_rgb: np.ndarray, body_mask: np.ndarray,
                                head_mask: np.ndarray, min_pixels: int = 200,
                                bins=(30, 32)):
    """Extract HSV histogram from the clothing region (body minus head).

    Args:
        image_rgb: Full image as RGB numpy array (H, W, 3), uint8.
        body_mask: Binary body mask (H, W), float32 or uint8.
        head_mask: Binary head mask (H, W), float32 or uint8.
        min_pixels: Minimum clothing pixels required for a valid result.
        bins: (hue_bins, saturation_bins) for the 2D histogram.

    Returns:
        np.ndarray — normalized 2D HSV histogram, or None if too few pixels.
    """
    # Clothing = body AND NOT head
    body_bin = (body_mask > 0.5).astype(np.uint8)
    head_bin = (head_mask > 0.5).astype(np.uint8)
    clothing_mask = body_bin & (~head_bin.astype(bool)).astype(np.uint8)

    if clothing_mask.sum() < min_pixels:
        return None

    # Extract clothing pixels
    clothing_pixels = image_rgb[clothing_mask > 0]  # (N, 3) RGB
    clothing_bgr = clothing_pixels[:, ::-1].reshape(1, -1, 3).astype(np.uint8)
    clothing_hsv = cv2.cvtColor(clothing_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    # Build 2D histogram from clothing pixels
    hist = cv2.calcHist([clothing_hsv.reshape(-1, 1, 3)], [0, 1], None,
                        list(bins), [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def outfit_color_similarity(palette_hist, clothing_hist):
    """Compute similarity between palette and clothing HSV histograms.

    Uses histogram correlation (same approach as head_histogram_similarity).

    Returns: float in [0, 1] where 1 = identical distribution, 0 = completely different.
    Returns 0.0 if either histogram is None.
    """
    if palette_hist is None or clothing_hist is None:
        return 0.0

    score = cv2.compareHist(palette_hist, clothing_hist, cv2.HISTCMP_CORREL)
    # CORREL returns [-1, 1], normalize to [0, 1]
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


# ── Smart Outfit Matching: dominant color ranking ───────────────────────────

def _dominant_hsv(image_rgb, mask, min_pixels=50):
    """Extract median HSV color from masked image region.

    Returns: np.ndarray (3,) with [H, S, V] or None if too few pixels.
    """
    pixels = image_rgb[mask > 0.5]
    if len(pixels) < min_pixels:
        return None
    pixels_bgr = pixels[:, ::-1].reshape(1, -1, 3).astype(np.uint8)
    pixels_hsv = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return np.median(pixels_hsv, axis=0).astype(np.float32)


def _hsv_similarity(c1, c2):
    """Similarity between two HSV colors on the HSV cylinder.

    Hue wraps at 180 (OpenCV range). Returns float in [0, 1].
    """
    if c1 is None or c2 is None:
        return 0.0
    dh = min(abs(c1[0] - c2[0]), 180 - abs(c1[0] - c2[0]))
    ds = abs(c1[1] - c2[1])
    dv = abs(c1[2] - c2[2])
    dist_sq = (dh / 90.0) ** 2 + (ds / 255.0) ** 2 + (dv / 255.0) ** 2
    # max possible dist_sq = 1 + 1 + 1 = 3
    return float(np.clip(1.0 - dist_sq / 3.0, 0.0, 1.0))


def extract_palette_colors(palette_image_rgb):
    """Extract dominant HSV color per palette block (left to right).

    Splits palette swatch into vertical columns based on aspect ratio.
    Returns: list of np.ndarray [(H, S, V), ...] — primary first. Empty list on failure.
    """
    if palette_image_rgb is None or palette_image_rgb.size == 0:
        return []

    img_h, img_w = palette_image_rgb.shape[:2]
    swatch_h = int(img_h * 0.70)
    swatch = palette_image_rgb[:swatch_h, :, :]
    if swatch.size == 0:
        return []

    sw, sh = swatch.shape[1], swatch.shape[0]
    num_blocks = max(1, round(sw / max(1, sh)))
    block_w = sw // max(1, num_blocks)

    colors = []
    for i in range(num_blocks):
        x_start = i * block_w
        x_end = (i + 1) * block_w if i < num_blocks - 1 else sw
        block = swatch[:, x_start:x_end, :]
        if block.size == 0:
            continue
        block_bgr = block[:, ::-1].copy()
        block_hsv = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        dominant = np.median(block_hsv, axis=0).astype(np.float32)
        colors.append(dominant)
    return colors


def extract_clothing_colors(image_rgb, cloth_mask, face_bbox, min_pixels=100):
    """Extract dominant HSV from upper and lower clothing regions.

    Split point: face bottom (y2) + 40% of remaining cloth height.

    Args:
        image_rgb: Full image RGB (H, W, 3) uint8.
        cloth_mask: BiSeNet label 16 mask (H, W) float32.
        face_bbox: (x1, y1, x2, y2) face bounding box.
        min_pixels: Minimum pixels per region for valid extraction.

    Returns: (upper_hsv, lower_hsv) — each np.ndarray (3,) or None.
    """
    cloth_ys = np.where(cloth_mask > 0.5)[0]
    if len(cloth_ys) < min_pixels * 2:
        return None, None

    y2 = int(face_bbox[3])
    cloth_bottom = int(cloth_ys.max())
    split_y = int(y2 + (cloth_bottom - y2) * 0.4)

    upper_mask = cloth_mask.copy()
    upper_mask[split_y:, :] = 0
    lower_mask = cloth_mask.copy()
    lower_mask[:split_y, :] = 0

    upper_hsv = _dominant_hsv(image_rgb, upper_mask, min_pixels)
    lower_hsv = _dominant_hsv(image_rgb, lower_mask, min_pixels)
    return upper_hsv, lower_hsv


def outfit_ranking_similarity(palette_colors, upper_hsv, lower_hsv):
    """Score how well palette colors match upper/lower clothing regions.

    Scoring:
    - primary (palette[0]) vs upper body: weight 0.45
    - secondary (palette[1]) vs lower body: weight 0.35
    - presence bonus (best match of any palette color in any region): weight 0.20

    Returns: float in [0, 1].
    """
    if not palette_colors:
        return 0.0
    if upper_hsv is None and lower_hsv is None:
        return 0.0

    primary_sim = 0.0
    secondary_sim = 0.0
    presence_sim = 0.0

    # Primary vs upper body
    if len(palette_colors) >= 1 and upper_hsv is not None:
        primary_sim = _hsv_similarity(palette_colors[0], upper_hsv)

    # Secondary vs lower body
    if len(palette_colors) >= 2 and lower_hsv is not None:
        secondary_sim = _hsv_similarity(palette_colors[1], lower_hsv)

    # Presence bonus: best match of any palette color in either region
    best_match = 0.0
    for pc in palette_colors:
        if upper_hsv is not None:
            best_match = max(best_match, _hsv_similarity(pc, upper_hsv))
        if lower_hsv is not None:
            best_match = max(best_match, _hsv_similarity(pc, lower_hsv))
    presence_sim = best_match

    # Weighted combination
    score = 0.45 * primary_sim + 0.35 * secondary_sim + 0.20 * presence_sim
    return float(np.clip(score, 0.0, 1.0))
