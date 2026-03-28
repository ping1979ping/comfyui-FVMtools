"""Depth-guided mask refinement v2: edge carving, gap filling, cross-mask deconfliction."""

import numpy as np
import cv2


# Mask types where depth refinement makes sense
DEPTH_REFINABLE_MASKS = {"body", "head", "face"}


# ── Phase 1: Depth Edge Map (once per image) ──

def compute_depth_edges(depth_map, threshold=0.05):
    """Compute depth discontinuity edges via Sobel gradients.

    Args:
        depth_map: [H, W] float32 [0,1]
        threshold: gradient magnitude threshold for edge detection

    Returns:
        (edge_magnitude [H,W] float32 normalized [0,1],
         edges_binary [H,W] bool)
    """
    # Sobel gradients
    sx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sx ** 2 + sy ** 2)

    # Normalize to [0, 1]
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max

    edges_binary = magnitude > threshold
    return magnitude, edges_binary


# ── Phase 2: Edge Carving (per mask) ──

def carve_mask_at_depth_edges(mask, edges_binary, depth_map, carve_strength=0.8, min_area=100):
    """Cut mask along depth edges, regroup components by depth similarity.

    Depth edges inside the mask create cuts. Then connected components are
    analyzed: components whose median depth matches the person's depth band
    are kept (handles the pole-in-front-of-person case where a person becomes
    a multi-part mask).

    Args:
        mask: [H, W] float32 [0,1]
        edges_binary: [H, W] bool — depth edge map
        depth_map: [H, W] float32 — for depth-based regrouping
        carve_strength: 0-1, how strongly edges cut (0=off)
        min_area: minimum component area in pixels to keep

    Returns:
        [H, W] float32 carved mask
    """
    if carve_strength <= 0 or np.sum(mask > 0.5) < min_area:
        return mask

    # Cut: zero out mask pixels on depth edges
    mask_binary = (mask > 0.5).astype(np.uint8)
    edge_cut = edges_binary.astype(np.uint8)

    # Thicken edges slightly for cleaner cuts (3x3 dilation)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_cut = cv2.dilate(edge_cut, edge_kernel, iterations=1)

    carved = mask_binary & ~edge_cut

    # Estimate person depth band from original mask
    masked_depths = depth_map[mask > 0.5]
    if len(masked_depths) < 50:
        return mask
    depth_low = np.percentile(masked_depths, 10)
    depth_high = np.percentile(masked_depths, 90)
    band_margin = max((depth_high - depth_low) * 0.3, 0.02)
    depth_low -= band_margin
    depth_high += band_margin

    # Regroup: keep all components whose depth matches the person
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(carved, connectivity=8)
    result = np.zeros_like(mask)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp_pixels = (labels == label_id)
        comp_depth = np.median(depth_map[comp_pixels])
        if depth_low <= comp_depth <= depth_high:
            result[comp_pixels] = 1.0

    # Blend with original based on carve_strength
    result = mask * (1.0 - carve_strength) + result * carve_strength
    return np.clip(result, 0, 1).astype(np.float32)


# ── Phase 3: Gap Filling (per mask) ──

def grow_mask_between_edges(mask, edges_binary, max_pixels=30):
    """Grow mask to fill gaps, but never cross depth edges.

    Args:
        mask: [H, W] float32 [0,1]
        edges_binary: [H, W] bool
        max_pixels: maximum dilation radius

    Returns:
        [H, W] float32
    """
    if max_pixels <= 0:
        return mask

    current = (mask > 0.5).astype(np.uint8)
    barrier = (~edges_binary).astype(np.uint8)  # 1 where growth is allowed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    iterations = max_pixels // 2

    for _ in range(iterations):
        dilated = cv2.dilate(current, kernel, iterations=1)
        new_pixels = (dilated & ~current) & barrier
        if new_pixels.sum() == 0:
            break
        current = current | new_pixels

    return current.astype(np.float32)


# ── Phase 4: Cross-Reference Deconfliction ──

def deconflict_masks(masks_dict, depth_map):
    """Resolve overlapping masks across references using depth proximity.

    Where masks overlap, the pixel goes to the person whose median depth
    is closer to that pixel's depth value.

    Args:
        masks_dict: {ri: [H,W] float32} for all references
        depth_map: [H,W] float32

    Returns:
        {ri: [H,W] float32} with overlaps resolved
    """
    if len(masks_dict) < 2:
        return masks_dict

    # Compute median depth per reference
    ref_depths = {}
    for ri, mask in masks_dict.items():
        pixels = depth_map[mask > 0.5]
        if len(pixels) > 0:
            ref_depths[ri] = np.median(pixels)
        else:
            ref_depths[ri] = 0.5  # fallback

    # Find all overlapping pixels (where 2+ masks are active)
    active_refs = list(masks_dict.keys())
    H, W = depth_map.shape
    overlap_count = np.zeros((H, W), dtype=np.int32)
    for ri in active_refs:
        overlap_count += (masks_dict[ri] > 0.5).astype(np.int32)

    overlap_mask = overlap_count >= 2
    if not overlap_mask.any():
        return masks_dict

    # For overlapping pixels: assign to nearest-depth reference
    result = {ri: masks_dict[ri].copy() for ri in active_refs}
    overlap_ys, overlap_xs = np.where(overlap_mask)

    if len(overlap_ys) == 0:
        return result

    pixel_depths = depth_map[overlap_ys, overlap_xs]

    for idx in range(len(overlap_ys)):
        y, x = overlap_ys[idx], overlap_xs[idx]
        pd = pixel_depths[idx]

        # Find which refs claim this pixel
        claimants = [ri for ri in active_refs if masks_dict[ri][y, x] > 0.5]
        if len(claimants) < 2:
            continue

        # Winner = closest depth to pixel
        best_ri = min(claimants, key=lambda ri: abs(ref_depths[ri] - pd))
        for ri in claimants:
            if ri != best_ri:
                result[ri][y, x] = 0.0

    return result


# ── Combined refinement (per mask, called from masker.py) ──

def refine_mask_with_depth(mask, depth_edges_data, depth_map,
                           carve_strength=0.8, grow_pixels=30):
    """Refine a single mask using precomputed depth edges.

    Args:
        mask: [H, W] float32
        depth_edges_data: (edge_magnitude, edges_binary) from compute_depth_edges
        depth_map: [H, W] float32
        carve_strength: 0-1
        grow_pixels: max gap fill radius

    Returns:
        [H, W] float32
    """
    if depth_edges_data is None or depth_map is None:
        return mask

    _, edges_binary = depth_edges_data

    # Resize if needed
    H, W = mask.shape
    if edges_binary.shape[0] != H or edges_binary.shape[1] != W:
        edges_binary = cv2.resize(edges_binary.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
    if depth_map.shape[0] != H or depth_map.shape[1] != W:
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

    if np.sum(mask > 0.5) < 50:
        return mask

    # Carve at edges + regroup by depth
    carved = carve_mask_at_depth_edges(mask, edges_binary, depth_map, carve_strength)

    # Fill gaps between edges
    if grow_pixels > 0:
        carved = grow_mask_between_edges(carved, edges_binary, grow_pixels)

    # Morphological closing to smooth
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx((carved * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return (result / 255.0).astype(np.float32)
