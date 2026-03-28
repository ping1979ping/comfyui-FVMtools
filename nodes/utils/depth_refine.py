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
    sx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sx ** 2 + sy ** 2)

    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max

    edges_binary = magnitude > threshold
    return magnitude, edges_binary


# ── Phase 2: Edge Carving (per mask) ──

def carve_mask_at_depth_edges(mask, edges_binary, depth_map, carve_strength=0.8, min_area=100):
    """Cut mask along depth edges, regroup components by depth similarity.

    Args:
        mask: [H, W] float32 [0,1]
        edges_binary: [H, W] bool
        depth_map: [H, W] float32
        carve_strength: 0-1
        min_area: minimum component area in pixels

    Returns:
        [H, W] float32 carved mask
    """
    if carve_strength <= 0 or np.sum(mask > 0.5) < min_area:
        return mask

    mask_binary = (mask > 0.5).astype(np.uint8)

    # Thicken edges for cleaner cuts
    edge_cut = edges_binary.astype(np.uint8)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_cut = cv2.dilate(edge_cut, edge_kernel, iterations=1)

    carved = mask_binary & ~edge_cut

    # Person depth band from original mask
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

    result = mask * (1.0 - carve_strength) + result * carve_strength
    return np.clip(result, 0, 1).astype(np.float32)


# ── Phase 3: Gap Filling (per mask) ──

def grow_mask_between_edges(mask, edges_binary, max_pixels=30):
    """Grow mask to fill gaps, but never cross depth edges."""
    if max_pixels <= 0:
        return mask

    current = (mask > 0.5).astype(np.uint8)
    barrier = (~edges_binary).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    iterations = max_pixels // 2

    for _ in range(iterations):
        dilated = cv2.dilate(current, kernel, iterations=1)
        new_pixels = (dilated & ~current) & barrier
        if new_pixels.sum() == 0:
            break
        current = current | new_pixels

    return current.astype(np.float32)


# ── Phase 4: Cross-Reference Deconfliction (vectorized) ──

def deconflict_masks(masks_dict, depth_map, edges_binary=None):
    """Resolve overlapping masks using depth-based winner-takes-all.

    For each pixel claimed by multiple references, the reference whose
    CORE depth (from non-overlapping region) is closest wins.
    After deconfliction, winning masks are extended along depth edges
    to fill any gaps at boundaries.

    Args:
        masks_dict: {ri: [H,W] float32} for all references
        depth_map: [H,W] float32
        edges_binary: optional [H,W] bool for edge-aware extension

    Returns:
        {ri: [H,W] float32} with overlaps resolved
    """
    if len(masks_dict) < 2:
        return masks_dict

    active_refs = sorted(masks_dict.keys())
    H, W = depth_map.shape

    # Stack all masks into array [N, H, W]
    N = len(active_refs)
    ri_to_idx = {ri: i for i, ri in enumerate(active_refs)}
    mask_stack = np.zeros((N, H, W), dtype=np.float32)
    for ri in active_refs:
        mask_stack[ri_to_idx[ri]] = masks_dict[ri]

    # Binary masks at low threshold to catch soft overlaps
    binary_stack = (mask_stack > 0.3).astype(np.int32)
    overlap_count = binary_stack.sum(axis=0)  # [H, W]

    has_overlap = (overlap_count >= 2).any()
    if not has_overlap:
        return masks_dict

    # Compute CORE depth per reference (from non-overlapping pixels only)
    non_overlap = (overlap_count <= 1)
    ref_core_depths = {}
    for ri in active_refs:
        i = ri_to_idx[ri]
        core_mask = (mask_stack[i] > 0.5) & non_overlap
        core_pixels = depth_map[core_mask]
        if len(core_pixels) > 20:
            ref_core_depths[ri] = np.median(core_pixels)
        else:
            # Fallback: use full mask
            all_pixels = depth_map[mask_stack[i] > 0.5]
            ref_core_depths[ri] = np.median(all_pixels) if len(all_pixels) > 0 else 0.5

    # Vectorized winner assignment for overlapping pixels
    overlap_mask = (overlap_count >= 2)
    oys, oxs = np.where(overlap_mask)

    if len(oys) > 0:
        pixel_depths = depth_map[oys, oxs]  # [K]

        # For each overlap pixel, compute distance to each ref's core depth
        # Shape: [N, K]
        distances = np.zeros((N, len(oys)), dtype=np.float32)
        claims = np.zeros((N, len(oys)), dtype=bool)
        for ri in active_refs:
            i = ri_to_idx[ri]
            claims[i] = mask_stack[i][oys, oxs] > 0.3
            distances[i] = np.abs(ref_core_depths[ri] - pixel_depths)

        # Set non-claimant distances to infinity
        distances[~claims] = np.inf

        # Winner = closest depth per pixel
        winners = np.argmin(distances, axis=0)  # [K]

        # Zero out losers
        for i in range(N):
            loser_pixels = (winners != i) & claims[i]
            if loser_pixels.any():
                ly = oys[loser_pixels]
                lx = oxs[loser_pixels]
                mask_stack[i][ly, lx] = 0.0

    # Edge-aware extension: grow winning masks slightly along depth edges
    # to fill gaps at boundaries where carving + deconfliction left thin strips
    if edges_binary is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Build combined "claimed" mask (any ref)
        any_claimed = (mask_stack.max(axis=0) > 0.3)

        for i in range(N):
            m = (mask_stack[i] > 0.3).astype(np.uint8)
            if m.sum() == 0:
                continue
            # Dilate slightly, but only into unclaimed territory near edges
            dilated = cv2.dilate(m, kernel, iterations=2)
            near_edge = edges_binary.astype(np.uint8)
            # New pixels: dilated, near an edge, not claimed by anyone else
            other_claimed = np.zeros((H, W), dtype=np.uint8)
            for j in range(N):
                if j != i:
                    other_claimed = np.maximum(other_claimed, (mask_stack[j] > 0.3).astype(np.uint8))
            new_pixels = dilated & ~m & near_edge & ~other_claimed
            mask_stack[i][new_pixels > 0] = 1.0

    # Build result
    result = {}
    for ri in active_refs:
        result[ri] = mask_stack[ri_to_idx[ri]]

    total_overlap_before = len(oys) if len(oys) > 0 else 0
    remaining = np.sum((mask_stack > 0.3).astype(np.int32).sum(axis=0) >= 2)
    print(f"    [Deconflict] {total_overlap_before} overlap pixels resolved, {remaining} remaining")

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

    H, W = mask.shape
    if edges_binary.shape[0] != H or edges_binary.shape[1] != W:
        edges_binary = cv2.resize(edges_binary.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
    if depth_map.shape[0] != H or depth_map.shape[1] != W:
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

    if np.sum(mask > 0.5) < 50:
        return mask

    carved = carve_mask_at_depth_edges(mask, edges_binary, depth_map, carve_strength)

    if grow_pixels > 0:
        carved = grow_mask_between_edges(carved, edges_binary, grow_pixels)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx((carved * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return (result / 255.0).astype(np.float32)
