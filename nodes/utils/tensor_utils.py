import numpy as np
import torch
import cv2


def tensor2np(tensor: torch.Tensor) -> np.ndarray:
    """IMAGE tensor (B,H,W,C) float32 [0,1] -> RGB uint8 numpy (first frame)."""
    img = tensor[0].cpu().numpy()
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


def fill_mask_holes(mask: torch.Tensor) -> torch.Tensor:
    """Fill holes in a MASK tensor using contour-based flood fill."""
    np_mask = (mask[0].cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(np_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(np_mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return torch.from_numpy(filled.astype(np.float32) / 255.0).unsqueeze(0)
