"""Depth-guided mask reconstruction: fill gaps, remove overlaps, sharpen edges."""

import numpy as np
import cv2


# Mask types where depth refinement makes sense (semantic masks like hair/eyes need pixel precision)
DEPTH_REFINABLE_MASKS = {"body", "head", "face"}


def refine_mask_with_depth(mask, depth_map, grow_pixels=50, remove_overlap=True, tolerance=0.05):
    """Refine a person segmentation mask using depth map coherence.

    Uses the depth profile of the existing mask to:
    - Fill gaps by growing into nearby pixels with matching depth
    - Remove overlapping objects with mismatched depth
    - Sharpen edges at depth discontinuities

    Args:
        mask: [H, W] float32 numpy array [0,1]
        depth_map: [H, W] float32 numpy array [0,1], or None for passthrough
        grow_pixels: max dilation radius for gap filling (0 = no growing)
        remove_overlap: whether to remove depth-mismatched pixels
        tolerance: depth band expansion factor (higher = more permissive)

    Returns:
        [H, W] float32 numpy array [0,1]
    """
    if depth_map is None:
        return mask

    # Resize depth map if resolution doesn't match mask
    H, W = mask.shape
    if depth_map.shape[0] != H or depth_map.shape[1] != W:
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

    # Need enough mask pixels to estimate depth profile
    mask_pixels = np.sum(mask > 0.5)
    if mask_pixels < 50:
        return mask

    # Compute ROI for performance (only process mask region + padding)
    ys, xs = np.where(mask > 0.5)
    H, W = mask.shape
    pad = max(grow_pixels, 20)
    roi_y1 = max(0, int(ys.min()) - pad)
    roi_y2 = min(H, int(ys.max()) + pad + 1)
    roi_x1 = max(0, int(xs.min()) - pad)
    roi_x2 = min(W, int(xs.max()) + pad + 1)

    roi_mask = mask[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    roi_depth = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]

    # Phase 1: Estimate person depth profile
    profile = _estimate_depth_profile(roi_depth, roi_mask, tolerance)
    if profile is None:
        return mask

    depth_low, depth_high = profile

    # Phase 2: Region growing (fill gaps)
    if grow_pixels > 0:
        grown = _depth_constrained_grow(roi_mask, roi_depth, depth_low, depth_high, grow_pixels)
    else:
        grown = roi_mask.copy()

    # Phase 3: Overlap removal
    if remove_overlap:
        confidence = _depth_overlap_confidence(roi_depth, depth_low, depth_high)
    else:
        confidence = np.ones_like(roi_mask)

    # Phase 4: Merge and clean
    roi_result = _merge_and_clean(roi_mask, grown, confidence)

    # Write back to full-size mask
    result = mask.copy()
    result[roi_y1:roi_y2, roi_x1:roi_x2] = roi_result
    return result


def _estimate_depth_profile(depth_roi, mask_roi, tolerance_factor,
                            percentile_low=15.0, percentile_high=85.0,
                            min_tolerance=0.02):
    """Estimate the depth band of a person from their mask.

    Returns (depth_low, depth_high) or None if insufficient data.
    """
    masked_depths = depth_roi[mask_roi > 0.5]
    if len(masked_depths) < 50:
        return None

    low = np.percentile(masked_depths, percentile_low)
    high = np.percentile(masked_depths, percentile_high)
    iqr = high - low

    # Wider tolerance for extreme depth variation (arm toward camera)
    factor = tolerance_factor * 2.0 if iqr > 0.3 else tolerance_factor
    expansion = max(iqr * factor, min_tolerance)

    return (low - expansion, high + expansion)


def _depth_constrained_grow(mask, depth, depth_low, depth_high, max_pixels):
    """Iteratively dilate mask, only accepting pixels within depth band.

    Fills gaps where depth matches the person, stops at depth discontinuities.
    """
    current = (mask > 0.5).astype(np.uint8)
    depth_valid = ((depth >= depth_low) & (depth <= depth_high)).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    iterations = max_pixels // 2

    for _ in range(iterations):
        dilated = cv2.dilate(current, kernel, iterations=1)
        new_pixels = (dilated & ~current) & depth_valid
        if new_pixels.sum() == 0:
            break
        current = current | new_pixels

    return current.astype(np.float32)


def _depth_overlap_confidence(depth, depth_low, depth_high, softness=0.03):
    """Compute per-pixel confidence based on depth band membership.

    Pixels inside the band get confidence ~1.0, outside drops via sigmoid.
    """
    band_center = (depth_low + depth_high) / 2
    band_half = max((depth_high - depth_low) / 2, 1e-6)

    # Normalized distance: 0 at center, 1 at band edge
    distance = np.abs(depth - band_center) / band_half

    # Sigmoid: 1.0 inside band, smooth falloff outside
    scale = 1.0 / max(softness, 1e-6)
    confidence = 1.0 / (1.0 + np.exp(scale * (distance - 1.0)))

    return confidence.astype(np.float32)


def _merge_and_clean(original, grown, depth_confidence, closing_size=7, min_area_fraction=0.001):
    """Merge grown mask with depth confidence, clean up artifacts."""
    # Union of grown (fills gaps) weighted by depth confidence (removes overlaps)
    merged = np.maximum(grown, original) * depth_confidence

    # Morphological closing to smooth jagged edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
    merged_uint8 = (np.clip(merged, 0, 1) * 255).astype(np.uint8)
    closed = cv2.morphologyEx(merged_uint8, cv2.MORPH_CLOSE, kernel)

    # Remove tiny fragments
    result = closed.astype(np.float32) / 255.0
    H, W = result.shape
    min_area = int(H * W * min_area_fraction)
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (result > 0.5).astype(np.uint8), connectivity=8)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] < min_area:
                result[labels == label_id] = 0.0

    return result
