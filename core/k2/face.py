"""K2 Lab — Gesichtserkennung und -verfeinerung für regionale Subjekte.

Nach dem Hauptdurchlauf sitzt jedes Subjekt in seiner Box, aber Gesichter sind
bei ganzkörperlichen Kompositionen klein und verlieren Identität. Dieser Pass
schneidet jedes erkannte Gesicht heraus, sampelt es mit *nur* den LoRAs seiner
Region nach und blendet es zurück.

Detektoren: Ultralytics-YOLO (bereits in FVMtools vorhanden) oder das NanoDet
``face_det.onnx`` aus FantasyPortrait — beides ohne neue Abhängigkeiten.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import PixelBox

DETECTOR_YOLO = "yolo"
DETECTOR_NANODET = "nanodet_onnx"
DETECTOR_BACKENDS = (DETECTOR_YOLO, DETECTOR_NANODET)

_NANODET_RELATIVE = (
    "custom_nodes/ComfyUI-WanVideoWrapper/fantasyportrait/models/face_det.onnx",
    "models/detection/face_det.onnx",
    "models/facedetection/face_det.onnx",
)


@dataclass
class FaceDetection:
    box: PixelBox
    score: float

    @property
    def center(self) -> tuple[float, float]:
        return self.box.center


@dataclass
class FaceTarget:
    region_id: str
    region_name: str
    prompt: str
    detection: FaceDetection
    lora_specs: tuple = ()


@dataclass
class FaceDetailSettings:
    enabled: bool = False
    steps: int = 8
    denoise: float = 0.15
    crop_size: int = 512
    padding: float = 2.0
    feather: float = 0.12
    blend: float = 0.5
    lora_scale: float = 0.5
    threshold: float = 0.4

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("Face-Detail Steps müssen positiv sein")
        if not 0.0 <= self.denoise <= 1.0:
            raise ValueError("Face-Detail Denoise muss zwischen 0 und 1 liegen")
        if self.crop_size < 64 or self.crop_size % 16:
            raise ValueError("Crop-Size muss ≥64 und ein Vielfaches von 16 sein")
        if self.padding < 1.0:
            raise ValueError("Padding muss ≥1.0 sein")
        if not 0.0 <= self.feather <= 0.5:
            raise ValueError("Feather muss zwischen 0 und 0.5 liegen")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("Blend muss zwischen 0 und 1 liegen")


# ── Detektoren ───────────────────────────────────────────────────────────


def discover_nanodet(base_path: str | Path) -> Path | None:
    base = Path(base_path)
    for relative in _NANODET_RELATIVE:
        candidate = base / relative
        if candidate.is_file():
            return candidate
    return None


def detect_faces_yolo(image_np: np.ndarray, model_name: str, threshold: float):
    """Ultralytics-Pfad. `image_np` ist HWC float 0..1."""
    from ...nodes.utils.yolo_detector import resolve_yolo_path

    path = resolve_yolo_path(model_name)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"YOLO-Modell nicht gefunden: {model_name}")

    model = _yolo_cache_get(path)
    array = (np.clip(image_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    results = model.predict(array[:, :, ::-1], conf=float(threshold), verbose=False)

    detections: list[FaceDetection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for index in range(len(boxes)):
            x0, y0, x1, y1 = (
                boxes.xyxy[index].detach().cpu().numpy().astype(float).tolist()
            )
            score = float(boxes.conf[index].item())
            if x1 <= x0 or y1 <= y0:
                continue
            detections.append(FaceDetection(PixelBox(x0, y0, x1, y1), score))
    return detections


_YOLO_CACHE: dict[str, object] = {}


def _yolo_cache_get(path: str):
    model = _YOLO_CACHE.get(path)
    if model is None:
        from ultralytics import YOLO

        model = YOLO(path)
        _YOLO_CACHE[path] = model
    return model


_ONNX_CACHE: dict[str, object] = {}


def detect_faces_nanodet(image_np: np.ndarray, model_path: str, threshold: float):
    """NanoDet-ONNX-Pfad (K2Lab-kompatibel), läuft auf der CPU."""
    import onnxruntime

    session = _ONNX_CACHE.get(model_path)
    if session is None:
        session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        _ONNX_CACHE[model_path] = session

    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    target_h = int(shape[2]) if isinstance(shape[2], int) else 320
    target_w = int(shape[3]) if isinstance(shape[3], int) else 320

    height, width = image_np.shape[:2]
    from PIL import Image

    pil = Image.fromarray(
        (np.clip(image_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    ).resize((target_w, target_h), Image.Resampling.BILINEAR)
    tensor = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    outputs = session.run(None, {input_meta.name: tensor})

    boxes, scores = _decode_nanodet(outputs, target_w, target_h)
    detections: list[FaceDetection] = []
    scale_x = width / float(target_w)
    scale_y = height / float(target_h)
    for (x0, y0, x1, y1), score in zip(boxes, scores):
        if score < threshold:
            continue
        box = (
            max(0.0, x0 * scale_x),
            max(0.0, y0 * scale_y),
            min(float(width), x1 * scale_x),
            min(float(height), y1 * scale_y),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append(FaceDetection(PixelBox(*box), float(score)))
    return _non_max_suppression(detections, 0.35)


def _decode_nanodet(outputs, width: int, height: int):
    """Liest die verbreiteten NanoDet-Ausgabeformen (boxes+scores oder kombiniert)."""
    arrays = [np.asarray(o) for o in outputs]
    for array in arrays:
        flat = array.reshape(-1, array.shape[-1]) if array.ndim >= 2 else array
        if flat.ndim == 2 and flat.shape[-1] >= 5:
            boxes = flat[:, :4].astype(float)
            scores = flat[:, 4].astype(float)
            if boxes.max(initial=0.0) <= 1.5:
                boxes = boxes * np.array([width, height, width, height], dtype=float)
            return boxes.tolist(), scores.tolist()
    if len(arrays) >= 2:
        boxes = arrays[0].reshape(-1, 4).astype(float)
        scores = arrays[1].reshape(-1).astype(float)
        if boxes.max(initial=0.0) <= 1.5:
            boxes = boxes * np.array([width, height, width, height], dtype=float)
        return boxes.tolist(), scores.tolist()
    raise RuntimeError("NanoDet-Ausgabeformat nicht erkannt")


def _non_max_suppression(detections: list[FaceDetection], iou_threshold: float):
    ordered = sorted(detections, key=lambda d: -d.score)
    kept: list[FaceDetection] = []
    for candidate in ordered:
        overlaps = False
        for existing in kept:
            if _iou(candidate.box, existing.box) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            kept.append(candidate)
    return kept


def _iou(a: PixelBox, b: PixelBox) -> float:
    ow = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    oh = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    intersection = ow * oh
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def detect_faces(
    image_np: np.ndarray,
    *,
    backend: str,
    model: str,
    threshold: float,
) -> list[FaceDetection]:
    if backend == DETECTOR_YOLO:
        return detect_faces_yolo(image_np, model, threshold)
    if backend == DETECTOR_NANODET:
        return detect_faces_nanodet(image_np, model, threshold)
    raise ValueError(f"Unbekannter Detektor-Backend: {backend!r}")


# ── Zuordnung und Compositing ────────────────────────────────────────────


def assign_faces(
    detections: list[FaceDetection],
    bound,
    routes=(),
    *,
    require_lora: bool = False,
) -> list[FaceTarget]:
    """Ordnet jede Erkennung der Region zu, in deren Box ihr Zentrum liegt.

    Regionen mit höherer Priorität (= früher im Plan) greifen zuerst zu.
    """
    plan = bound.plan
    lora_by_region: dict[str, list] = {}
    for route in routes:
        for region_id in route.region_ids:
            lora_by_region.setdefault(region_id, []).append(route)

    targets: list[FaceTarget] = []
    claimed: set[int] = set()
    for region in plan.regions:
        if region.role != "subject":
            continue
        region_loras = tuple(lora_by_region.get(region.region_id, ()))
        if require_lora and not region_loras:
            continue
        best_index = None
        best_score = -1.0
        for index, detection in enumerate(detections):
            if index in claimed:
                continue
            cx, cy = detection.center
            inside = (
                region.box.x0 <= cx < region.box.x1
                and region.box.y0 <= cy < region.box.y1
            )
            if not inside:
                continue
            if detection.score > best_score:
                best_score = detection.score
                best_index = index
        if best_index is None:
            continue
        claimed.add(best_index)
        # Im Crop ist das Gesicht das Motiv, deshalb steht die Identität hier
        # bewusst vorne — anders als im Gesamtprompt.
        identity = region.identity_prompt.strip()
        scene = region.prompt.strip()
        if identity and scene:
            crop_prompt = f"a close-up portrait of {identity}, {scene}"
        else:
            crop_prompt = identity or scene or region.name
        targets.append(
            FaceTarget(
                region_id=region.region_id,
                region_name=region.name,
                prompt=crop_prompt,
                detection=detections[best_index],
                lora_specs=region_loras,
            )
        )
    return targets


def expanded_square_crop(
    box: PixelBox, width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    """Quadratischer Ausschnitt um das Gesicht, an den Bildrändern eingerückt."""
    cx, cy = box.center
    size = max(box.width, box.height) * float(padding)
    size = min(size, float(min(width, height)))
    half = size / 2.0
    x0 = cx - half
    y0 = cy - half
    x0 = min(max(x0, 0.0), width - size)
    y0 = min(max(y0, 0.0), height - size)
    return (
        int(round(x0)),
        int(round(y0)),
        int(round(x0 + size)),
        int(round(y0 + size)),
    )


def feather_mask(size: tuple[int, int], feather: float) -> np.ndarray:
    """Rechteckmaske mit weichem Rand, Anteil `feather` der kürzeren Kante."""
    width, height = size
    mask = np.ones((height, width), dtype=np.float32)
    border = int(round(min(width, height) * float(feather)))
    if border <= 0:
        return mask
    ramp = np.linspace(0.0, 1.0, border, dtype=np.float32)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    mask[:border, :] *= ramp[:, None]
    mask[-border:, :] *= ramp[::-1][:, None]
    mask[:, :border] *= ramp[None, :]
    mask[:, -border:] *= ramp[::-1][None, :]
    return mask


def composite_crop(
    canvas: np.ndarray,
    refined: np.ndarray,
    crop_box: tuple[int, int, int, int],
    feather: float,
    blend: float,
) -> np.ndarray:
    """Blendet einen verfeinerten Ausschnitt weich in das Bild zurück."""
    x0, y0, x1, y1 = crop_box
    target_w = x1 - x0
    target_h = y1 - y0
    if refined.shape[0] != target_h or refined.shape[1] != target_w:
        from PIL import Image

        image = Image.fromarray(
            (np.clip(refined, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        ).resize((target_w, target_h), Image.Resampling.LANCZOS)
        refined = np.asarray(image, dtype=np.float32) / 255.0

    alpha = feather_mask((target_w, target_h), feather) * float(blend)
    region = canvas[y0:y1, x0:x1, :]
    canvas = canvas.copy()
    canvas[y0:y1, x0:x1, :] = region * (1.0 - alpha[..., None]) + refined * alpha[
        ..., None
    ]
    return np.clip(canvas, 0.0, 1.0)


__all__ = [
    "DETECTOR_BACKENDS",
    "DETECTOR_NANODET",
    "DETECTOR_YOLO",
    "FaceDetailSettings",
    "FaceDetection",
    "FaceTarget",
    "assign_faces",
    "composite_crop",
    "detect_faces",
    "discover_nanodet",
    "expanded_square_crop",
    "feather_mask",
]
