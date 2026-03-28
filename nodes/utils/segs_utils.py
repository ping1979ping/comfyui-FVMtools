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


def assign_segs_to_references(segs, body_masks, num_refs, batch_idx,
                               face_centers=None):
    """Assign each SEG to a reference person.

    Primary method: face_centers — assigns SEGS to the ref whose face center
    falls inside the SEG mask. This is identity-anchored and avoids the
    circular dependency of body-mask-based assignment.

    Fallback: body mask overlap (legacy, used when face_centers not provided).

    Args:
        segs: Impact Pack SEGS format: (shape, [SEG, SEG, ...])
        body_masks: list of [B, H, W] tensors, one per reference (fallback)
        num_refs: number of active references
        batch_idx: current batch image index
        face_centers: optional dict {ri: (cx, cy)} — face center per reference

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

    # Convert all SEGs to full masks once
    seg_masks = []
    for seg in seg_list:
        full_mask = seg_to_full_mask(seg, image_shape)
        mask_area = full_mask.sum().item()
        if mask_area >= 1.0:
            seg_masks.append(full_mask)

    if not seg_masks:
        return {}

    if face_centers:
        # Primary: assign by face center containment
        used_segs = set()

        for ri in range(num_refs):
            if ri not in face_centers:
                continue
            cx, cy = face_centers[ri]
            icx, icy = int(cx), int(cy)

            # Find the SEG that contains this face center
            best_seg_idx = None
            best_area = float('inf')  # prefer smallest containing mask (most specific)

            for si, mask in enumerate(seg_masks):
                if si in used_segs:
                    continue
                h_m, w_m = mask.shape
                if 0 <= icy < h_m and 0 <= icx < w_m and mask[icy, icx] > 0.5:
                    area = mask.sum().item()
                    if area < best_area:
                        best_area = area
                        best_seg_idx = si

            if best_seg_idx is not None:
                assignments.setdefault(ri, []).append(seg_masks[best_seg_idx])
                used_segs.add(best_seg_idx)

        # Remaining SEGs go to unassigned (-1)
        for si, mask in enumerate(seg_masks):
            if si not in used_segs:
                assignments.setdefault(-1, []).append(mask)

    else:
        # Fallback: body mask overlap (legacy)
        for full_mask in seg_masks:
            mask_area = full_mask.sum().item()
            best_ref = -1
            best_overlap = 0.0
            min_overlap = mask_area * 0.05

            for ri in range(num_refs):
                if ri >= len(body_masks):
                    continue
                body_mask = body_masks[ri][batch_idx]
                overlap = (full_mask * body_mask).sum().item()
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ref = ri

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

    segs = detector.detect(image_tensor, threshold, dilation, crop_factor)
    return segs
