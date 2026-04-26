"""Built-in YOLO detection for PersonSelectorMulti.

Loads Ultralytics YOLO models, runs inference per image,
returns per-object masks with labels. Handles model caching."""

import os
import torch
import numpy as np

import folder_paths

# ── Model path scanning ──────────────────────────────────────────────────

def get_available_yolo_models():
    """Scan ultralytics model folders for .pt files. Returns list of relative names.

    Deduplicates by basename so the same file isn't listed twice when it appears
    both via folder_paths registration (e.g. 'foo.pt' under 'ultralytics_segm')
    and via the fallback directory walk (e.g. 'segm/foo.pt' under 'ultralytics').
    """
    seen_basenames = set()
    models = []

    def _add(name):
        base = os.path.basename(name).lower()
        if base in seen_basenames:
            return
        seen_basenames.add(base)
        models.append(name)

    # Check registered folders
    for folder_key in ("ultralytics", "ultralytics_segm", "ultralytics_bbox"):
        try:
            files = folder_paths.get_filename_list(folder_key)
            for f in files:
                if f.endswith(".pt"):
                    _add(f)
        except Exception:
            pass

    # Also scan models/ultralytics/ directly as fallback
    base = os.path.join(folder_paths.models_dir, "ultralytics")
    if os.path.isdir(base):
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.endswith(".pt"):
                    rel = os.path.relpath(os.path.join(root, f), base)
                    _add(rel)

    return sorted(models)


def resolve_yolo_path(model_name):
    """Resolve a model name to absolute path."""
    # Try registered folder_paths first
    for folder_key in ("ultralytics_segm", "ultralytics_bbox", "ultralytics"):
        try:
            path = folder_paths.get_full_path(folder_key, model_name)
            if path and os.path.isfile(path):
                return path
        except Exception:
            pass
        # Also try just the filename without subdirectory prefix
        basename = os.path.basename(model_name)
        if basename != model_name:
            try:
                path = folder_paths.get_full_path(folder_key, basename)
                if path and os.path.isfile(path):
                    return path
            except Exception:
                pass

    # Direct path under models/ultralytics/
    base = os.path.join(folder_paths.models_dir, "ultralytics")
    direct = os.path.join(base, model_name)
    if os.path.isfile(direct):
        return direct

    # Try without subdirectory prefix
    direct2 = os.path.join(base, os.path.basename(model_name))
    if os.path.isfile(direct2):
        return direct2

    # Search recursively as last resort
    for root, dirs, files in os.walk(base):
        if os.path.basename(model_name) in files:
            return os.path.join(root, os.path.basename(model_name))

    print(f"[FVMTools] Could not resolve YOLO model path: {model_name}")
    return None


# ── Model cache ──────────────────────────────────────────────────────────

_model_cache = {}  # path → YOLO model
_classes_cache = {}  # path → list[str] of class names


def _load_model(model_path):
    """Load YOLO model with caching."""
    if model_path in _model_cache:
        return _model_cache[model_path]

    from ultralytics import YOLO
    model = YOLO(model_path)
    _model_cache[model_path] = model
    return model


def get_yolo_classes(model_name):
    """Return sorted list of class names for a YOLO model. Cached."""
    model_path = resolve_yolo_path(model_name)
    if model_path is None:
        return []
    if model_path in _classes_cache:
        return _classes_cache[model_path]
    try:
        model = _load_model(model_path)
        names = getattr(model, "names", None) or {}
        if isinstance(names, dict):
            classes = [str(v) for _, v in sorted(names.items())]
        else:
            classes = [str(v) for v in names]
    except Exception as e:
        print(f"[FVMTools] Failed to read classes from {model_name}: {e}")
        classes = []
    _classes_cache[model_path] = classes
    return classes


# ── Detection ────────────────────────────────────────────────────────────

def detect_objects(image_tensor, model_name, confidence=0.5, label_filter=""):
    """Run YOLO detection on a single image tensor.

    Args:
        image_tensor: [H, W, C] float32 0-1 (single image, no batch dim)
        model_name: model filename to load
        confidence: detection confidence threshold
        label_filter: comma-separated class names to keep (empty = all)

    Returns:
        list of dicts: [{"label": str, "mask": [H,W] float32, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
    """
    model_path = resolve_yolo_path(model_name)
    if model_path is None:
        print(f"[FVMTools] YOLO model not found: {model_name}")
        return []

    model = _load_model(model_path)

    # Convert to PIL
    from PIL import Image
    img_np = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_h, img_w = img_pil.height, img_pil.width

    # Run inference
    results = model.predict(img_pil, conf=confidence, verbose=False)

    # Log available class names from model
    if results and hasattr(results[0], 'names'):
        all_names = results[0].names
        print(f"[FVMTools] YOLO model classes: {all_names}")

    # Parse label filter (substring match — "leg" matches "Left-leg", "right_leg", etc.)
    filter_labels = [l.strip().lower() for l in label_filter.split(",") if l.strip()]
    if filter_labels:
        print(f"[FVMTools] Label filter (substring match): {filter_labels}")

    detections = []
    for r in results:
        has_masks = r.masks is not None
        num_detections = len(r.boxes) if r.boxes is not None else 0
        print(f"[FVMTools] YOLO detected {num_detections} objects, has_masks={has_masks}")

        if num_detections == 0:
            continue

        for j in range(num_detections):
            class_id = int(r.boxes.cls[j])
            class_name = r.names[class_id]

            # Apply label filter — substring match so "leg" hits "Left-leg", "right_leg", etc.
            if filter_labels:
                cn_lower = class_name.lower()
                if not any(f in cn_lower for f in filter_labels):
                    print(f"[FVMTools]   skipping '{class_name}' (not in filter)")
                    continue

            conf = r.boxes.conf[j].cpu().item()
            bbox = r.boxes.xyxy[j].cpu().numpy()

            # Get mask (segmentation model) or create from bbox
            is_bbox_only = not has_masks
            if has_masks:
                import cv2
                # Use r.masks.xy (polygon in ORIGINAL image coordinates) rather than
                # r.masks.data (which is at letterboxed model input size and would
                # introduce a vertical offset equal to the letterbox padding when
                # resized straight back to the image dimensions).
                mask = np.zeros((img_h, img_w), dtype=np.float32)
                try:
                    poly = r.masks.xy[j]
                    if poly is not None and len(poly) >= 3:
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1.0)
                except (AttributeError, IndexError):
                    # Fallback to legacy path if .xy isn't available
                    mask_raw = r.masks.data[j].cpu().numpy()
                    mask_resized = cv2.resize(mask_raw, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    mask = (mask_resized > 0.5).astype(np.float32)
            else:
                # Bbox-only model: create rectangular mask
                mask = np.zeros((img_h, img_w), dtype=np.float32)
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                mask[y1:y2, x1:x2] = 1.0

            detections.append({
                "label": class_name,
                "mask": torch.from_numpy(mask),  # [H, W] float32
                "bbox": bbox,
                "confidence": conf,
                "is_bbox_only": is_bbox_only,
            })

    return detections


def assign_detections_to_references(detections, body_masks, num_refs, batch_idx,
                                     min_overlap_fraction=0.10):
    """Assign YOLO detections to references using body mask overlap.

    For each detection, calculates overlap with each reference's body mask.
    Assigns to the reference with highest overlap (if above threshold).
    Unassigned detections go to key -1 (generic pool).

    Args:
        detections: list of dicts from detect_objects()
        body_masks: person_data["body_masks"] — list of [B, H, W] tensors per ref
        num_refs: number of references
        batch_idx: current batch image index
        min_overlap_fraction: minimum overlap as fraction of detection mask area

    Returns:
        dict: {ri: [mask_tensor, ...], -1: [mask_tensor, ...]}
    """
    assignments = {ri: [] for ri in range(num_refs)}
    assignments[-1] = []

    for det in detections:
        det_mask = det["mask"]  # [H, W]
        det_area = det_mask.sum().item()
        if det_area < 10:  # skip tiny detections
            continue

        best_ri = -1
        best_overlap = 0.0

        for ri in range(num_refs):
            if ri >= len(body_masks):
                continue
            body = body_masks[ri]
            if body.dim() == 3:
                body_2d = body[batch_idx]  # [H, W]
            else:
                body_2d = body

            # Calculate overlap: intersection / detection_area
            intersection = (det_mask * body_2d).sum().item()
            overlap_frac = intersection / det_area if det_area > 0 else 0

            if overlap_frac > best_overlap:
                best_overlap = overlap_frac
                best_ri = ri

        if best_overlap >= min_overlap_fraction and best_ri >= 0:
            assignments[best_ri].append(det_mask)
        else:
            assignments[-1].append(det_mask)

    return assignments
