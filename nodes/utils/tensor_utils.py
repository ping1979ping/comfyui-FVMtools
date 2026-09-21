import numpy as np
import torch
import cv2


def tensor2np(tensor: torch.Tensor) -> np.ndarray:
    """IMAGE tensor (B,H,W,C) float32 [0,1] -> RGB uint8 numpy (first frame).

    Extra channels are dropped. Most nodes hand over plain RGB, but some
    upstream nodes (GLSL/shader nodes, PNG loads that keep transparency) emit
    RGBA — and everything downstream here is a three-channel model: BiSeNet's
    first conv takes 3, SAM3's normalize subtracts a 3-vector mean. Both raise
    a shape error several frames deep, so the channels are trimmed at the one
    door they all come through.
    """
    img = tensor[0].cpu().numpy()
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[:, :, :3]
    return (img * 255).clip(0, 255).astype(np.uint8)


def tensor2cv2(tensor: torch.Tensor) -> np.ndarray:
    """IMAGE tensor -> BGR uint8 numpy (first frame)."""
    rgb = tensor2np(tensor)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def np2tensor(img: np.ndarray) -> torch.Tensor:
    """RGB uint8 numpy (H,W,C) -> IMAGE tensor (1,H,W,C)."""
    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)


def mask2tensor(mask: np.ndarray) -> torch.Tensor:
    """Float32 mask (H,W) [0,1] -> MASK tensor (1,H,W)."""
    return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)


def empty_mask(h: int, w: int) -> torch.Tensor:
    """Returns a zeros MASK tensor (1,H,W)."""
    return torch.zeros((1, h, w), dtype=torch.float32)


def apply_gaussian_blur(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply Gaussian blur to a MASK tensor. radius=0 means no blur."""
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    np_mask = mask[0].cpu().numpy()
    blurred = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), 0)
    return torch.from_numpy(blurred).unsqueeze(0)


# Mirrors MAX_FILLED_HOLE_FRACTION in mask_utils — see the note there.
MAX_FILLED_HOLE_FRACTION = 0.10


def fill_mask_holes(mask: torch.Tensor) -> torch.Tensor:
    """Close small holes in a [1,H,W] MASK, leaving large structural ones open."""
    np_mask = (mask[0].cpu().numpy() * 255).astype(np.uint8)
    binary = (np_mask > 127).astype(np.uint8)
    total = int(binary.sum())
    if total == 0:
        return torch.zeros_like(mask)
    # RETR_CCOMP gives outer contours at hierarchy level 0 and their holes at
    # level 1. Filling every hole (what RETR_EXTERNAL + FILLED effectively did)
    # destroys ring-shaped masks: a hair mask that encircles the face has the
    # face as its hole, and blanket-filling turns hair into head. Only holes
    # small enough to be segmentation noise are closed.
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(np_mask)
    if hierarchy is None:
        return torch.zeros_like(mask)
    hierarchy = hierarchy[0]
    max_hole_area = total * MAX_FILLED_HOLE_FRACTION
    for i, contour in enumerate(contours):
        if hierarchy[i][3] < 0:          # outer contour
            cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    for i, contour in enumerate(contours):
        if hierarchy[i][3] >= 0 and cv2.contourArea(contour) > max_hole_area:
            cv2.drawContours(filled, [contour], -1, 0, cv2.FILLED)   # keep the hole open
    return torch.from_numpy(filled.astype(np.float32) / 255.0).unsqueeze(0)
