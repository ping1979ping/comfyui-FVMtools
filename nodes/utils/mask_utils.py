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


def feather_mask(mask_2d: torch.Tensor, pixels: int) -> torch.Tensor:
    """Apply Gaussian blur to mask edges for feathered blending."""
    if pixels <= 0:
        return mask_2d
    kernel_size = pixels * 2 + 1
    np_mask = mask_2d.cpu().numpy()
    blurred = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), 0)
    return torch.from_numpy(blurred)


def fill_mask_holes_2d(mask_2d: torch.Tensor) -> torch.Tensor:
    """Fill holes in a [H,W] mask using contour-based flood fill."""
    np_mask = (mask_2d.cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(np_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(np_mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
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
