"""OCR backends for FVMtools - reading text from images (signs, labels, plates).

Two optional backends, both fully degradable:

  onnx     PP-OCRv4 detection + recognition ONNX models driven directly by
           onnxruntime (no paddle / rapidocr package needed).
  easyocr  the optional `easyocr` package, lazily imported, offline only.

Design rules for this module:
  * Nothing here is a hard dependency beyond numpy/cv2 (already required by
    FVMtools). onnxruntime and easyocr are probed, never imported at module
    import time.
  * Nothing is ever downloaded at runtime. Use scripts/fetch_ocr_models.py.
  * Every public function is safe to call when nothing at all is installed:
    run_ocr() returns [], ocr_region() returns the documented empty dict,
    get_available_backends() returns [].

Model location resolution mirrors the BiSeNet lookup in masker.py:
  Tier 1  <root>/ocr for every root in folder_paths.get_folder_paths("onnx")
  Tier 2  <ComfyUI>/models/onnx/ocr
  Tier 3  outfit_config.ini  [models] ocr_path
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 ships with ComfyUI
    cv2 = None

try:
    from ...core.config import get_model_path
except ImportError:  # pragma: no cover - flat import path (tests)
    try:
        from core.config import get_model_path
    except Exception:
        get_model_path = None


# ──── Model files ────

OCR_MODEL_FILES = {
    "det": "ch_PP-OCRv4_det_infer.onnx",
    "rec": "ch_PP-OCRv4_rec_infer.onnx",
    "cls": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "keys": "ppocr_keys_v1.txt",
}

EASYOCR_MODEL_FILES = {
    "detector": "craft_mlt_25k.pth",
    "recognizer": "english_g2.pth",
}

# "cls" is an angle classifier - nice to have, not required.
_ONNX_REQUIRED = ("det", "rec", "keys")

EASYOCR_SUBDIR = "easyocr"

_EMPTY_REGION = {"text": "", "conf": 0.0, "char_confs": [], "line_count": 0}

# Module level caches
_SESSIONS: dict = {}
_KEYS_CACHE: dict = {}
_EASYOCR_READER = None


# ──── Path resolution ────

def _import_folder_paths():
    """Import folder_paths defensively (it is MagicMock-ed under pytest)."""
    try:
        import folder_paths
        return folder_paths
    except Exception:
        return None


def _valid_dir(candidate) -> str | None:
    """Return candidate only if it is a real existing directory path string."""
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    try:
        return candidate if os.path.isdir(candidate) else None
    except Exception:
        return None


def resolve_ocr_dir() -> str | None:
    """Locate the directory holding the OCR model files. None if not found."""
    fp = _import_folder_paths()

    # Tier 1: registered "onnx" model roots -> <root>/ocr
    if fp is not None:
        try:
            roots = fp.get_folder_paths("onnx")
        except Exception:
            roots = None
        if isinstance(roots, (list, tuple)):
            for root in roots:
                if not isinstance(root, str):
                    continue
                found = _valid_dir(os.path.join(root, "ocr"))
                if found:
                    return found

    # Tier 2: <ComfyUI>/models/onnx/ocr
    if fp is not None:
        try:
            models_dir = getattr(fp, "models_dir", None)
            if isinstance(models_dir, str):
                found = _valid_dir(os.path.join(models_dir, "onnx", "ocr"))
                if found:
                    return found
        except Exception:
            pass

    # Tier 3: outfit_config.ini [models] ocr_path
    if get_model_path is not None:
        try:
            ini_path = get_model_path("ocr_path")
        except Exception:
            ini_path = ""
        found = _valid_dir(ini_path)
        if found:
            return found

    return None


def _search_hint() -> str:
    return ("no OCR model directory found - searched '<root>/ocr' for every registered "
            "'onnx' model root, '<ComfyUI>/models/onnx/ocr', and "
            "outfit_config.ini [models] ocr_path")


# ──── Backend status ────

def _spec_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _onnx_status() -> dict:
    st = {"available": False, "reason": "", "dir": None, "missing": []}
    if not _spec_available("onnxruntime"):
        st["reason"] = "onnxruntime is not installed"
        st["missing"] = list(OCR_MODEL_FILES.values())
        return st
    # DBNet box postprocessing needs these; they are not declared in
    # requirements.txt, so report them instead of failing silently later.
    deps = [d for d in ("pyclipper", "shapely") if not _spec_available(d)]
    if deps:
        st["reason"] = (f"detection postprocess needs {' and '.join(deps)} "
                        f"(pip install {' '.join(deps)})")
        st["missing"] = list(OCR_MODEL_FILES.values())
        return st
    ocr_dir = resolve_ocr_dir()
    st["dir"] = ocr_dir
    if ocr_dir is None:
        st["reason"] = _search_hint()
        st["missing"] = list(OCR_MODEL_FILES.values())
        return st
    missing = [f for f in OCR_MODEL_FILES.values()
               if not os.path.isfile(os.path.join(ocr_dir, f))]
    st["missing"] = missing
    req_missing = [OCR_MODEL_FILES[k] for k in _ONNX_REQUIRED if OCR_MODEL_FILES[k] in missing]
    if req_missing:
        st["reason"] = (f"missing model files in {ocr_dir}: {', '.join(req_missing)} "
                        f"- run scripts/fetch_ocr_models.py")
        return st
    st["available"] = True
    st["reason"] = f"ready - PP-OCRv4 ONNX models in {ocr_dir}"
    if missing:
        st["reason"] += " (angle classifier not installed, orientation fix disabled)"
    return st


def _easyocr_status() -> dict:
    st = {"available": False, "reason": "", "dir": None, "missing": []}
    if not _spec_available("easyocr"):
        st["reason"] = "the optional 'easyocr' package is not installed"
        st["missing"] = list(EASYOCR_MODEL_FILES.values())
        return st
    ocr_dir = resolve_ocr_dir()
    if ocr_dir is None:
        st["reason"] = _search_hint()
        st["missing"] = list(EASYOCR_MODEL_FILES.values())
        return st
    model_dir = os.path.join(ocr_dir, EASYOCR_SUBDIR)
    st["dir"] = model_dir
    missing = [f for f in EASYOCR_MODEL_FILES.values()
               if not os.path.isfile(os.path.join(model_dir, f))]
    st["missing"] = missing
    if missing:
        st["reason"] = (f"missing model files in {model_dir}: {', '.join(missing)} "
                        f"- run scripts/fetch_ocr_models.py --backend easyocr")
        return st
    st["available"] = True
    st["reason"] = f"ready - easyocr weights in {model_dir}"
    return st


def backend_status() -> dict:
    """Full diagnostic for every backend. Never raises."""
    out = {}
    for name, fn in (("onnx", _onnx_status), ("easyocr", _easyocr_status)):
        try:
            out[name] = fn()
        except Exception as exc:  # pragma: no cover - defensive
            out[name] = {"available": False, "reason": f"status probe failed: {exc}",
                         "dir": None, "missing": []}
    return out


def get_available_backends() -> list:
    """Backends that are both importable and have their model files on disk."""
    status = backend_status()
    return [n for n in ("onnx", "easyocr") if status.get(n, {}).get("available")]


# ──── ONNX / PP-OCRv4 implementation ────

def _get_session(path: str):
    """Cached onnxruntime session for a model path."""
    sess = _SESSIONS.get(path)
    if sess is not None:
        return sess
    import onnxruntime as ort
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(path, sess_options=opts, providers=providers or None)
    _SESSIONS[path] = sess
    return sess


def _load_keys(path: str) -> list:
    keys = _KEYS_CACHE.get(path)
    if keys is not None:
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        chars = [line.rstrip("\n").rstrip("\r") for line in fh]
    # PaddleOCR CTC label list: blank at index 0, space appended at the end.
    keys = ["blank"] + chars + [" "]
    _KEYS_CACHE[path] = keys
    return keys


def _det_preprocess(img_rgb: np.ndarray, limit_side_len: int = 960):
    h, w = img_rgb.shape[:2]
    ratio = 1.0
    if max(h, w) > limit_side_len:
        ratio = limit_side_len / float(max(h, w))
    rh = max(32, int(round(h * ratio / 32) * 32))
    rw = max(32, int(round(w * ratio / 32) * 32))
    resized = cv2.resize(img_rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = x.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x), w / float(rw), h / float(rh)


def _get_mini_boxes(contour):
    rect = cv2.minAreaRect(contour)
    pts = sorted(list(cv2.boxPoints(rect)), key=lambda p: p[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return [pts[i1], pts[i2], pts[i3], pts[i4]], min(rect[1])


def _box_score(bitmap: np.ndarray, box: np.ndarray) -> float:
    h, w = bitmap.shape
    xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = box.copy()
    shifted[:, 0] -= xmin
    shifted[:, 1] -= ymin
    cv2.fillPoly(mask, [shifted.reshape(-1, 1, 2).astype(np.int32)], 1)
    return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0])


def _unclip(box: np.ndarray, ratio: float = 1.5):
    import pyclipper
    from shapely.geometry import Polygon
    poly = Polygon(box)
    if poly.length <= 0:
        return None
    distance = poly.area * ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath([tuple(p) for p in box], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    return np.array(expanded[0]) if expanded else None


def _db_postprocess(prob: np.ndarray, ratio_w: float, ratio_h: float,
                    thresh: float = 0.3, box_thresh: float = 0.55) -> list:
    bitmap = (prob > thresh).astype(np.uint8)
    found = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    quads = []
    for contour in contours[:1000]:
        eps = 0.002 * cv2.arcLength(contour, True)
        points = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if points.shape[0] < 4:
            continue
        if _box_score(prob, points.astype(np.float32)) < box_thresh:
            continue
        expanded = _unclip(points.astype(np.float32))
        if expanded is None or len(expanded) < 4:
            continue
        quad, side = _get_mini_boxes(expanded.reshape(-1, 1, 2).astype(np.float32))
        if side < 3:
            continue
        quad = np.array(quad, dtype=np.float32)
        quad[:, 0] *= ratio_w
        quad[:, 1] *= ratio_h
        quads.append(quad)
    # reading order: top to bottom, then left to right
    quads.sort(key=lambda q: (round(float(q[:, 1].min()) / 10.0), float(q[:, 0].min())))
    return quads


def _crop_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    w = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3])))
    h = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if out.shape[0] * 1.0 / max(out.shape[1], 1) >= 1.5:
        out = np.rot90(out).copy()
    return out


def _rec_one(sess, keys: list, crop: np.ndarray, height: int = 48, max_w: int = 320):
    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return "", 0.0, []
    new_w = min(max_w, max(8, int(round(w * height / float(h)))))
    resized = cv2.resize(crop, (new_w, height), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    canvas = np.zeros((height, max_w, 3), dtype=np.float32)
    canvas[:, :new_w] = x
    inp = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
    preds = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 2:
        return "", 0.0, []
    if not (0.99 <= float(preds[0].sum()) <= 1.01):  # not softmaxed yet
        e = np.exp(preds - preds.max(axis=1, keepdims=True))
        preds = e / np.clip(e.sum(axis=1, keepdims=True), 1e-8, None)
    idx = preds.argmax(axis=1)
    conf = preds.max(axis=1)
    chars, char_confs = [], []
    prev = -1
    for t, k in enumerate(idx):
        k = int(k)
        if k != 0 and k != prev and k < len(keys):
            chars.append(keys[k])
            char_confs.append(float(conf[t]))
        prev = k
    text = "".join(chars)
    mean_conf = float(np.mean(char_confs)) if char_confs else 0.0
    return text, mean_conf, char_confs


def _run_onnx(image_rgb: np.ndarray) -> list:
    """PP-OCRv4 detection + CTC recognition. Every stage is guarded."""
    status = _onnx_status()
    if not status["available"]:
        print(f"[FVMTools] OCR onnx backend unavailable: {status['reason']}")
        return []
    ocr_dir = status["dir"]
    try:
        det_sess = _get_session(os.path.join(ocr_dir, OCR_MODEL_FILES["det"]))
        rec_sess = _get_session(os.path.join(ocr_dir, OCR_MODEL_FILES["rec"]))
        keys = _load_keys(os.path.join(ocr_dir, OCR_MODEL_FILES["keys"]))
    except Exception as exc:
        print(f"[FVMTools] OCR onnx model load failed: {exc}")
        return []

    try:
        inp, ratio_w, ratio_h = _det_preprocess(image_rgb)
        prob = det_sess.run(None, {det_sess.get_inputs()[0].name: inp})[0]
        prob = np.asarray(prob, dtype=np.float32).squeeze()
        if prob.ndim != 2:
            print(f"[FVMTools] OCR detection output has unexpected shape {prob.shape}")
            return []
        quads = _db_postprocess(prob, ratio_w, ratio_h)
    except Exception as exc:
        print(f"[FVMTools] OCR detection failed: {exc}")
        return []

    h, w = image_rgb.shape[:2]
    results = []
    for quad in quads:
        try:
            quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
            crop = _crop_quad(image_rgb, quad)
            text, conf, char_confs = _rec_one(rec_sess, keys, crop)
        except Exception as exc:
            print(f"[FVMTools] OCR recognition failed for one box: {exc}")
            continue
        if not text:
            continue
        results.append({
            "text": text,
            "conf": conf,
            "char_confs": char_confs,
            "bbox": [float(quad[:, 0].min()), float(quad[:, 1].min()),
                     float(quad[:, 0].max()), float(quad[:, 1].max())],
            "quad": [[float(p[0]), float(p[1])] for p in quad],
        })
    return results


# ──── EasyOCR implementation ────

def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    import easyocr
    model_dir = os.path.join(resolve_ocr_dir() or "", EASYOCR_SUBDIR)
    try:
        reader = easyocr.Reader(["en", "de"], model_storage_directory=model_dir,
                                download_enabled=False, gpu=True)
    except Exception as exc:
        print(f"[FVMTools] easyocr GPU init failed ({exc}) - falling back to CPU")
        reader = easyocr.Reader(["en", "de"], model_storage_directory=model_dir,
                                download_enabled=False, gpu=False)
    _EASYOCR_READER = reader
    return reader


def _run_easyocr(image_rgb: np.ndarray) -> list:
    status = _easyocr_status()
    if not status["available"]:
        print(f"[FVMTools] OCR easyocr backend unavailable: {status['reason']}")
        return []
    try:
        reader = _get_easyocr_reader()
        raw = reader.readtext(image_rgb)
    except Exception as exc:
        print(f"[FVMTools] easyocr run failed: {exc}")
        return []
    results = []
    for item in raw or []:
        try:
            quad, text, conf = item[0], item[1], float(item[2])
            pts = [[float(p[0]), float(p[1])] for p in quad]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
        except Exception:
            continue
        if not text:
            continue
        # easyocr does not expose per-character confidences
        results.append({"text": text, "conf": conf, "char_confs": [],
                        "bbox": [min(xs), min(ys), max(xs), max(ys)], "quad": pts})
    return results


_RUNNERS = {"onnx": _run_onnx, "easyocr": _run_easyocr}
_BACKEND_ORDER = ("onnx", "easyocr")


# ──── Public API ────

def _normalize(raw, min_conf: float) -> list:
    out = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            text = str(item.get("text", ""))
            conf = float(item.get("conf", 0.0))
        except Exception:
            continue
        if not text or conf < min_conf:
            continue
        bbox = item.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        try:
            bbox = [float(v) for v in bbox][:4]
        except Exception:
            bbox = [0.0, 0.0, 0.0, 0.0]
        quad = item.get("quad") or []
        try:
            quad = [[float(p[0]), float(p[1])] for p in quad][:4]
        except Exception:
            quad = []
        try:
            char_confs = [float(c) for c in (item.get("char_confs") or [])]
        except Exception:
            char_confs = []
        out.append({"text": text, "conf": conf, "char_confs": char_confs,
                    "bbox": bbox, "quad": quad})
    return out


def run_ocr(image_rgb: np.ndarray, backend: str = "auto", min_conf: float = 0.0) -> list:
    """Run OCR on an RGB uint8 image.

    Returns a list of dicts:
        {"text": str, "conf": float, "char_confs": [float],
         "bbox": [x1, y1, x2, y2], "quad": [[x, y] * 4]}
    Returns [] (never raises) when no backend is usable.
    """
    try:
        arr = np.asarray(image_rgb)
        if arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
            return []
        arr = np.ascontiguousarray(arr[:, :, :3])
        if arr.dtype != np.uint8:
            arr = np.clip(arr.astype(np.float32) * (255.0 if arr.max() <= 1.0 else 1.0),
                          0, 255).astype(np.uint8)
    except Exception:
        return []

    if backend and backend != "auto":
        candidates = [backend]
    else:
        candidates = get_available_backends() or []

    for name in candidates:
        runner = _RUNNERS.get(name)
        if runner is None:
            continue
        try:
            raw = runner(arr)
        except Exception as exc:
            print(f"[FVMTools] OCR backend '{name}' raised: {exc}")
            continue
        return _normalize(raw, min_conf)
    return []


def _region_bbox(region, shape) -> list | None:
    """Accept either an [x1,y1,x2,y2] bbox or a 2D mask and return a pixel bbox."""
    h, w = shape[:2]
    try:
        arr = np.asarray(region)
    except Exception:
        return None
    if arr.ndim == 1 and arr.size == 4:
        x1, y1, x2, y2 = [float(v) for v in arr]
    elif arr.ndim >= 2:
        m = arr
        while m.ndim > 2:
            m = m[0] if m.shape[0] == 1 else m.max(axis=0)
        ys, xs = np.nonzero(m > 0.5) if m.dtype != bool else np.nonzero(m)
        if xs.size == 0:
            return None
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    else:
        return None
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w, round(x2))))
    y2 = int(max(0, min(h, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def ocr_region(image_rgb: np.ndarray, mask_2d_or_bbox, backend: str = "auto",
               pad: int = 4, min_conf: float = 0.0) -> dict:
    """Crop to a region (mask or bbox), OCR it and aggregate all lines.

    Returns {"text", "conf", "char_confs", "line_count"} - plus "bbox" (union of
    all line boxes in full-image coordinates) and "lines" (the raw per-line
    dicts) when at least one line was read. When nothing is found or no backend
    exists the result is exactly
    {"text": "", "conf": 0.0, "char_confs": [], "line_count": 0}.
    """
    try:
        arr = np.asarray(image_rgb)
        if arr.ndim != 3 or arr.size == 0:
            return dict(_EMPTY_REGION)
        box = _region_bbox(mask_2d_or_bbox, arr.shape)
        if box is None:
            return dict(_EMPTY_REGION)
        h, w = arr.shape[:2]
        x1 = max(0, box[0] - pad)
        y1 = max(0, box[1] - pad)
        x2 = min(w, box[2] + pad)
        y2 = min(h, box[3] + pad)
        crop = arr[y1:y2, x1:x2]
        if crop.size == 0:
            return dict(_EMPTY_REGION)
    except Exception:
        return dict(_EMPTY_REGION)

    lines = run_ocr(crop, backend=backend, min_conf=min_conf)
    if not lines:
        return dict(_EMPTY_REGION)

    texts, confs, char_confs = [], [], []
    xs1, ys1, xs2, ys2 = [], [], [], []
    for line in lines:
        texts.append(line["text"])
        confs.append(float(line["conf"]))
        char_confs.extend(line.get("char_confs") or [])
        bb = line.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if len(bb) == 4:
            xs1.append(bb[0] + x1)
            ys1.append(bb[1] + y1)
            xs2.append(bb[2] + x1)
            ys2.append(bb[3] + y1)

    return {
        "text": " ".join(t for t in texts if t).strip(),
        "conf": float(sum(confs) / len(confs)) if confs else 0.0,
        "char_confs": char_confs,
        "line_count": len(lines),
        "bbox": [min(xs1), min(ys1), max(xs2), max(ys2)] if xs1 else None,
        "lines": lines,
    }
