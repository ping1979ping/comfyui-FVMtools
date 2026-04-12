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

    # Diagnostic: warn when depth is too flat for meaningful edge detection
    edge_count = int(edges_binary.sum())
    if edge_count < depth_map.size * 0.001:
        print(f"[FVMTools] compute_depth_edges: only {edge_count} edge pixels detected "
              f"(threshold={threshold}). Depth refinement will be a no-op for this image.")

    return magnitude, edges_binary


# ── Phase 1b: Fused Edges (color + depth agreement) ──

def compute_fused_edges(image_rgb, depth_map, depth_threshold=0.05):
    """Compute fused person-boundary edges requiring BOTH color AND depth agreement.

    Suppresses within-person depth changes (arm reaching forward, body tilt)
    where there's a depth edge but no visual edge (same skin/clothing color).
    Only fires where both depth AND color discontinuities agree — these are
    real person-to-person or person-to-background boundaries.

    Pipeline:
    1. Bilateral filter — smooth clothing texture/hair, preserve silhouettes
    2. Canny on LAB L-channel — perceptually uniform edge detection
    3. Morphological gradient on depth — robust to quantization
    4. Fuse: fused = color_edges & dilate(depth_edges, 5px)

    Args:
        image_rgb: [H, W, 3] uint8 RGB image
        depth_map: [H, W] float32 [0,1]
        depth_threshold: gradient magnitude threshold for depth edges

    Returns:
        (fused_edges [H,W] bool — for carving, only where both signals agree,
         depth_only_edges [H,W] bool — for grow_mask_between_edges,
         depth_magnitude [H,W] float32 — normalized gradient magnitude)
    """
    # 1. Bilateral filter — edge-preserving smoothing removes clothing patterns
    #    and hair detail while keeping the strong luminance jump at person boundaries.
    filtered = cv2.bilateralFilter(image_rgb, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Canny on LAB L-channel — LAB lightness is perceptually uniform, giving
    #    better skin-vs-clothing and person-vs-background separation than grayscale.
    lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    color_edges = cv2.Canny(L, 30, 90) > 0

    # 3. Morphological gradient on depth — more robust than Sobel on quantized
    #    depth maps (common from Depth Anything V2).
    depth_uint8 = (np.clip(depth_map, 0, 1) * 255).astype(np.uint8)
    depth_grad = cv2.morphologyEx(depth_uint8, cv2.MORPH_GRADIENT,
                                   np.ones((3, 3), dtype=np.uint8))
    depth_edges = depth_grad > int(depth_threshold * 255)

    # 4. Fuse: require BOTH signals. Dilate depth edges ~5px for alignment
    #    tolerance (depth and color edges rarely align pixel-perfectly).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    depth_dilated = cv2.dilate(depth_edges.astype(np.uint8), kernel).astype(bool)
    fused = color_edges & depth_dilated

    # Depth-only edges and magnitude for other consumers (grow, compute_depth_edges compat)
    magnitude = depth_grad.astype(np.float32) / 255.0
    depth_only_edges = magnitude > depth_threshold

    fused_count = int(fused.sum())
    depth_count = int(depth_only_edges.sum())
    color_count = int(color_edges.sum())
    print(f"    [FusedEdges] color={color_count}, depth={depth_count}, "
          f"fused={fused_count} ({fused_count / max(1, depth_count) * 100:.0f}% of depth edges confirmed by color)")

    return fused, depth_only_edges, magnitude


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
    """Grow mask to fill gaps, but never cross depth edges.

    Guards against whole-image explosions: bails out if there are no edges
    (barrier would be all-open) and caps growth at 3x starting area to
    contain partial barrier failures.
    """
    if max_pixels <= 0:
        return mask

    # Guard 1: no edges detected → barrier would be all-open, dilation would
    # fill the image. Bail out and return mask unchanged.
    if not edges_binary.any():
        return mask

    current = (mask > 0.5).astype(np.uint8)
    start_area = int(current.sum())
    if start_area == 0:
        return mask

    # Guard 2: hard cap at 3x starting area so a partial barrier failure
    # (edges present but not forming a closed boundary) can't produce
    # runaway growth either.
    max_area = start_area * 3

    barrier = (~edges_binary).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    iterations = max_pixels // 2

    for _ in range(iterations):
        dilated = cv2.dilate(current, kernel, iterations=1)
        new_pixels = (dilated & ~current) & barrier
        if new_pixels.sum() == 0:
            break
        current = current | new_pixels
        if int(current.sum()) > max_area:
            break

    return current.astype(np.float32)


# ── Phase 4: Cross-Reference Deconfliction (vectorized) ──

def deconflict_masks(masks_dict, depth_map, edges_binary=None, bisenet_seeds=None):
    """Resolve overlapping masks with BiSeNet priority + depth fallback.

    Priority order for each overlapping pixel:
    1. If exactly ONE ref has a BiSeNet seed there → that ref wins (identity-anchored)
    2. If MULTIPLE refs have BiSeNet seeds → closest depth wins
    3. If NO refs have BiSeNet seeds → closest depth wins

    Args:
        masks_dict: {ri: [H,W] float32} for all references
        depth_map: [H,W] float32
        edges_binary: optional [H,W] bool for edge-aware extension
        bisenet_seeds: optional {ri: [H,W] float32} — BiSeNet label seeds per ref

    Returns:
        {ri: [H,W] float32} with overlaps resolved
    """
    if len(masks_dict) < 2:
        return masks_dict

    active_refs = sorted(masks_dict.keys())
    H, W = depth_map.shape

    N = len(active_refs)
    ri_to_idx = {ri: i for i, ri in enumerate(active_refs)}
    mask_stack = np.zeros((N, H, W), dtype=np.float32)
    for ri in active_refs:
        mask_stack[ri_to_idx[ri]] = masks_dict[ri]

    # BiSeNet seed stack (if provided)
    has_seeds = bisenet_seeds is not None and len(bisenet_seeds) > 0
    seed_stack = np.zeros((N, H, W), dtype=bool)
    if has_seeds:
        for ri in active_refs:
            if ri in bisenet_seeds:
                seed_stack[ri_to_idx[ri]] = bisenet_seeds[ri] > 0.5

    binary_stack = (mask_stack > 0.3).astype(np.int32)
    overlap_count = binary_stack.sum(axis=0)

    has_overlap = (overlap_count >= 2).any()
    if not has_overlap:
        return masks_dict

    # Core depth per ref (non-overlapping pixels)
    non_overlap = (overlap_count <= 1)
    ref_core_depths = {}
    for ri in active_refs:
        i = ri_to_idx[ri]
        core_mask = (mask_stack[i] > 0.5) & non_overlap
        core_pixels = depth_map[core_mask]
        if len(core_pixels) > 20:
            ref_core_depths[ri] = np.median(core_pixels)
        else:
            all_pixels = depth_map[mask_stack[i] > 0.5]
            ref_core_depths[ri] = np.median(all_pixels) if len(all_pixels) > 0 else 0.5

    overlap_mask = (overlap_count >= 2)
    oys, oxs = np.where(overlap_mask)

    if len(oys) > 0:
        pixel_depths = depth_map[oys, oxs]

        # Claims and distances [N, K]
        claims = np.zeros((N, len(oys)), dtype=bool)
        distances = np.full((N, len(oys)), np.inf, dtype=np.float32)
        seed_claims = np.zeros((N, len(oys)), dtype=bool)

        for ri in active_refs:
            i = ri_to_idx[ri]
            claims[i] = mask_stack[i][oys, oxs] > 0.3
            distances[i] = np.where(claims[i], np.abs(ref_core_depths[ri] - pixel_depths), np.inf)
            if has_seeds:
                seed_claims[i] = seed_stack[i][oys, oxs] & claims[i]

        # Determine winners with BiSeNet priority
        seed_count = seed_claims.sum(axis=0)  # how many refs have seed at each pixel

        # Default: depth-based winner
        winners = np.argmin(distances, axis=0)

        if has_seeds:
            # Where exactly one ref has a seed → that ref wins
            single_seed = (seed_count == 1)
            if single_seed.any():
                seed_winner = np.argmax(seed_claims[:, single_seed], axis=0)
                winners[single_seed] = seed_winner

            # Where multiple seeds → depth among seed-holders only
            multi_seed = (seed_count >= 2)
            if multi_seed.any():
                seed_distances = np.where(seed_claims[:, multi_seed], distances[:, multi_seed], np.inf)
                winners[multi_seed] = np.argmin(seed_distances, axis=0)

        # Zero out losers
        for i in range(N):
            loser_pixels = (winners != i) & claims[i]
            if loser_pixels.any():
                mask_stack[i][oys[loser_pixels], oxs[loser_pixels]] = 0.0

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
