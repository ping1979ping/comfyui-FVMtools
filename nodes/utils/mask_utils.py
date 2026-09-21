import numpy as np
import torch
import cv2


def is_mask_empty(mask_2d: torch.Tensor, threshold: float = 1.0) -> bool:
    """Check if a 2D mask has insufficient substance."""
    return mask_2d.sum().item() < threshold


def expand_mask(mask_2d: torch.Tensor, pixels: int) -> torch.Tensor:
    """Dilate a [H,W] float32 mask by the given number of pixels."""
    if pixels <= 0:
        return mask_2d
    np_mask = (mask_2d.cpu().numpy() * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    dilated = cv2.dilate(np_mask, kernel, iterations=1)
    return torch.from_numpy(dilated.astype(np.float32) / 255.0)


# Largest inward feather relative to the mask's own half-thickness. Above ~0.5
# the ramp meets itself in the middle and the mask never reaches full strength.
MAX_INWARD_FEATHER_RATIO = 0.45


def feather_mask(mask_2d: torch.Tensor, pixels: int, direction: str = "both") -> torch.Tensor:
    """Soften mask edges for feathered blending.

    A plain Gaussian is symmetric: the ramp straddles the boundary, half of it
    inside the mask and half outside. On a compact mask that is harmless — the
    core is large. On a ring or a strand (a hair mask around a face, say) it is
    destructive in both directions at once: the subject loses strength while the
    neighbour gets painted. `direction` moves the ramp off the boundary.

    both    — ramp centred on the boundary, half in and half out (legacy)
    outward — ramp lies entirely outside; the mask itself stays at full strength
    inward  — ramp lies entirely inside; nothing outside the mask is touched

    The one-sided modes shift the mask by `pixels` first and blur afterwards.
    Clipping a symmetric blur instead would leave a step of ~0.5 at the
    boundary — the very hard edge feathering exists to remove.
    """
    if pixels <= 0:
        return mask_2d
    kernel_size = pixels * 2 + 1
    np_mask = mask_2d.cpu().numpy()

    if direction in ("inward", "outward"):
        binary = (np_mask * 255).astype(np.uint8)
        if direction == "inward":
            # An inward ramp of R eats R off every side, so a mask thinner than
            # about 2R is consumed by its own feather. Measured on a hair ring
            # (half-thickness 38px): R=24 already caps the mask at alpha 0.92,
            # R=32 at 0.41, R=48 erases it. Cap the radius against the mask's
            # own depth so a wide blend degrades to a narrow one instead of
            # deleting the region.
            depth = cv2.distanceTransform((binary > 127).astype(np.uint8), cv2.DIST_L2, 5).max()
            pixels = min(pixels, max(1, int(depth * MAX_INWARD_FEATHER_RATIO)))
            kernel_size = pixels * 2 + 1
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        shifted = (cv2.erode if direction == "inward" else cv2.dilate)(binary, morph_kernel)
        blurred = cv2.GaussianBlur(shifted.astype(np.float32) / 255.0,
                                   (kernel_size, kernel_size), 0)
        # The blur has already decayed to ~0 (or ~1) at the original boundary,
        # so this only pins the guarantee exactly; it is not a visible step.
        if direction == "inward":
            return torch.from_numpy(np.minimum(np_mask, blurred))
        return torch.from_numpy(np.maximum(np_mask, blurred))

    return torch.from_numpy(cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), 0))


# A hole larger than this fraction of the mask is structure, not a segmentation
# gap, and is left open. Sized so a face inside a hair ring (roughly a third of
# the hair area) survives, while speckle and small carve-outs still close.
MAX_FILLED_HOLE_FRACTION = 0.10


def fill_mask_holes_2d(mask_2d: torch.Tensor) -> torch.Tensor:
    """Close small holes in a [H,W] mask, leaving large structural ones open."""
    np_mask = (mask_2d.cpu().numpy() * 255).astype(np.uint8)
    binary = (np_mask > 127).astype(np.uint8)
    total = int(binary.sum())
    if total == 0:
        return torch.zeros_like(mask_2d)
    # RETR_CCOMP gives outer contours at hierarchy level 0 and their holes at
    # level 1. Filling every hole (what RETR_EXTERNAL + FILLED effectively did)
    # destroys ring-shaped masks: a hair mask that encircles the face has the
    # face as its hole, and blanket-filling turns hair into head. Only holes
    # small enough to be segmentation noise are closed.
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(np_mask)
    if hierarchy is None:
        return torch.zeros_like(mask_2d)
    hierarchy = hierarchy[0]
    max_hole_area = total * MAX_FILLED_HOLE_FRACTION
    for i, contour in enumerate(contours):
        if hierarchy[i][3] < 0:          # outer contour
            cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    for i, contour in enumerate(contours):
        if hierarchy[i][3] >= 0 and cv2.contourArea(contour) > max_hole_area:
            cv2.drawContours(filled, [contour], -1, 0, cv2.FILLED)   # keep the hole open
    return torch.from_numpy(filled.astype(np.float32) / 255.0)


def split_mask_to_components(mask_2d: torch.Tensor, min_area_fraction: float = 0.001) -> list:
    """Split a binary mask into connected components, filtering by size.

    Args:
        mask_2d: [H, W] float32 tensor, values in [0,1]
        min_area_fraction: minimum component area as fraction of image area

    Returns:
        List of [H, W] float32 tensors, one per component, sorted by area descending.
    """
    np_mask = (mask_2d.cpu().numpy() * 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(np_mask, connectivity=8)

    H, W = mask_2d.shape
    min_area = int(H * W * min_area_fraction)

    components = []
    for label_id in range(1, num_labels):  # skip background (0)
        component = (labels == label_id).astype(np.float32)
        if component.sum() >= min_area:
            components.append(torch.from_numpy(component))

    # Sort by area descending (largest face first)
    components.sort(key=lambda c: c.sum(), reverse=True)
    return components


def clean_mask_crumbs(mask_np, min_area_fraction=0.005):
    """Remove small disconnected blobs from a mask.

    Useful for body masks from SAM which often have small artifacts.

    Args:
        mask_np: [H, W] float32 numpy array, values in [0,1]
        min_area_fraction: minimum blob area as fraction of image area

    Returns:
        Cleaned float32 numpy array.
    """
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    total_area = mask_np.shape[0] * mask_np.shape[1]
    min_area = int(total_area * min_area_fraction)

    cleaned = np.zeros_like(mask_np)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 1.0
    return cleaned
