import torch
import numpy as np


def seg_to_full_mask(seg, image_shape):
    """Convert a SEG's cropped_mask + crop_region to a full-image [H, W] float32 mask.

    Args:
        seg: Impact Pack SEG namedtuple (cropped_mask, crop_region, ...)
        image_shape: (H, W) tuple of the full image

    Returns:
        torch.Tensor [H, W] float32 mask
    """
    h, w = image_shape
    mask = torch.zeros((h, w), dtype=torch.float32)

    crop_region = seg.crop_region  # (x1, y1, x2, y2)
    cropped_mask = seg.cropped_mask

    if cropped_mask is None:
        return mask

    if isinstance(cropped_mask, np.ndarray):
        cropped_mask = torch.from_numpy(cropped_mask)

    cropped_mask = cropped_mask.float()

    # Handle 3D masks [1, H, W] or [H, W]
    if cropped_mask.ndim == 3:
        cropped_mask = cropped_mask[0]

    x1, y1, x2, y2 = crop_region
    # Clamp to image bounds
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(w, x2)
    cy2 = min(h, y2)

    # Corresponding region in the cropped mask
    mx1 = cx1 - x1
    my1 = cy1 - y1
    mx2 = cropped_mask.shape[1] - (x2 - cx2)
    my2 = cropped_mask.shape[0] - (y2 - cy2)

    if cx2 > cx1 and cy2 > cy1 and mx2 > mx1 and my2 > my1:
        mask[cy1:cy2, cx1:cx2] = cropped_mask[my1:my2, mx1:mx2]

    return mask


def assign_segs_to_references(segs, body_masks, num_refs, batch_idx):
    """Assign each SEG to the reference person with highest body mask overlap.

    Args:
        segs: Impact Pack SEGS format: (shape, [SEG, SEG, ...])
        body_masks: list of [B, H, W] tensors, one per reference
        num_refs: number of active references
        batch_idx: current batch image index

    Returns:
        dict mapping ref_index → [list of full-image [H,W] masks]
        key -1 → unassigned body parts (for generic slot)
    """
    if not segs or len(segs) < 2:
        return {}

    shape = segs[0]  # (H, W) or (H, W, C)
    seg_list = segs[1]

    if not seg_list:
        return {}

    image_shape = (shape[0], shape[1])
    assignments = {}

    for seg in seg_list:
        full_mask = seg_to_full_mask(seg, image_shape)
        mask_area = full_mask.sum().item()

        if mask_area < 1.0:
            continue

        best_ref = -1
        best_overlap = 0.0
        min_overlap = mask_area * 0.05  # 5% threshold

        for ri in range(num_refs):
            if ri >= len(body_masks):
                continue
            body_mask = body_masks[ri][batch_idx]  # [H, W]
            overlap = (full_mask * body_mask).sum().item()
            if overlap > best_overlap:
                best_overlap = overlap
                best_ref = ri

        # Require minimum overlap to assign
        if best_overlap < min_overlap:
            best_ref = -1

        assignments.setdefault(best_ref, []).append(full_mask)

    return assignments


def run_detector(detector, image_tensor, threshold=0.3, dilation=10, crop_factor=3.0):
    """Run a BBOX_DETECTOR or SEGM_DETECTOR on a single image.

    Args:
        detector: Impact Pack detector (BBOX_DETECTOR or SEGM_DETECTOR)
        image_tensor: [1, H, W, C] or [H, W, C] tensor
        threshold: detection confidence threshold
        dilation: mask dilation in pixels
        crop_factor: crop expansion factor

    Returns:
        SEGS tuple: (shape, [SEG, ...])
    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Both BBOX_DETECTOR and SEGM_DETECTOR have a .detect() method
    # that returns SEGS format
    segs = detector.detect(image_tensor, threshold, dilation, crop_factor)
    return segs
