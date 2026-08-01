"""GATE probe - does a real-text-trained DBNet detector fire on diffusion pseudo-glyphs?

Read-only. Touches no node code. It loads nodes/utils/ocr_backend.py by path and
calls its own detection primitives, but:

  * detection runs at NATIVE resolution (or upscaled), never at the module's
    hardcoded limit_side_len=960 which would squash 1280x768 to 0.75x,
  * the RAW detector quads are evaluated BEFORE any recognizer filter
    (ocr_backend lines 417/418 drop every box whose recognizer text is empty -
    exactly the case a pseudo-glyph produces).

Usage:
    python tests/live/gate_dbnet_probe.py --image <path> --tag street
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
OCR_BACKEND_PY = os.path.join(REPO, "nodes", "utils", "ocr_backend.py")
DEFAULT_MODEL_DIR = r"D:/AI/ComfyUI/ComfyUI/models/onnx/ocr"


def load_backend():
    """Import ocr_backend.py standalone (its relative imports fail soft)."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    spec = importlib.util.spec_from_file_location("gate_ocr_backend", OCR_BACKEND_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── geometry helpers ────────────────────────────────────────────────────────


def quad_sides(q: np.ndarray) -> tuple[float, float]:
    """(long side, short side) of a detector quad, in px."""
    a = float((np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[2] - q[3])) / 2.0)
    b = float((np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2.0)
    return (max(a, b), min(a, b))


def pct(values: list, p: float) -> float:
    return float(np.percentile(np.asarray(values, np.float64), p)) if values else 0.0


# ── detection ───────────────────────────────────────────────────────────────


def detect(
    ob,
    det_sess,
    img_rgb: np.ndarray,
    scale: float,
    thresh: float,
    box_thresh: float,
    unclip: float = 1.5,
) -> list:
    """Raw DBNet quads in ORIGINAL image coordinates. No recognizer involved."""
    h, w = img_rgb.shape[:2]
    if abs(scale - 1.0) > 1e-6:
        work = cv2.resize(
            img_rgb,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        work = img_rgb

    # limit_side_len above the working size => _det_preprocess never downscales.
    big = max(work.shape[:2]) + 64
    inp, ratio_w, ratio_h = ob._det_preprocess(work, limit_side_len=big)
    prob = det_sess.run(None, {det_sess.get_inputs()[0].name: inp})[0]
    prob = np.asarray(prob, dtype=np.float32).squeeze()
    if prob.ndim != 2:
        raise RuntimeError(f"unexpected det output shape {prob.shape}")

    if abs(unclip - 1.5) > 1e-6:
        orig_unclip = ob._unclip
        ob._unclip = lambda box, ratio=unclip, _f=orig_unclip: _f(box, ratio)
        try:
            quads = ob._db_postprocess(prob, ratio_w, ratio_h, thresh, box_thresh)
        finally:
            ob._unclip = orig_unclip
    else:
        quads = ob._db_postprocess(prob, ratio_w, ratio_h, thresh, box_thresh)

    out = []
    for q in quads:
        q = np.asarray(q, np.float32) / float(scale)
        q[:, 0] = np.clip(q[:, 0], 0, w - 1)
        q[:, 1] = np.clip(q[:, 1], 0, h - 1)
        out.append(q)
    detect.last_prob = prob
    return out, float(prob.max()), float((prob > thresh).mean())


def prob_overlay(img_rgb: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Raw DBNet shrink-map heat over the image - shows what the net SEES,
    independent of the quad-fitting / box_thresh postprocess."""
    h, w = img_rgb.shape[:2]
    p = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(
        (np.clip(p, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    a = np.clip(p, 0, 1)[..., None] * 0.85
    return (img_rgb.astype(np.float32) * (1 - a) + heat.astype(np.float32) * a).astype(
        np.uint8
    )


# ── drawing ─────────────────────────────────────────────────────────────────


def draw_overlay(img_rgb: np.ndarray, quads: list, thickness: int = 2) -> np.ndarray:
    vis = img_rgb.copy()
    for q in quads:
        cv2.polylines(
            vis,
            [q.astype(np.int32).reshape(-1, 1, 2)],
            True,
            (0, 255, 60),
            thickness,
            cv2.LINE_AA,
        )
    return vis


def contact_sheet(
    ob,
    img_rgb: np.ndarray,
    quads: list,
    labels: list,
    zoom: int = 3,
    per_row: int = 6,
    cell_h: int = 40,
) -> np.ndarray:
    """Grid of every detected line crop, upscaled, with index + recognizer text."""
    tiles = []
    ch = cell_h * zoom
    cw = int(ch * 5.0)
    for i, q in enumerate(quads):
        try:
            crop = ob._crop_quad(img_rgb, q)
        except Exception:
            crop = np.zeros((8, 8, 3), np.uint8)
        h, w = crop.shape[:2]
        if h < 1 or w < 1:
            crop = np.zeros((8, 8, 3), np.uint8)
            h, w = 8, 8
        sc = min(ch / float(h), cw / float(w))
        crop = cv2.resize(
            crop,
            (max(1, int(w * sc)), max(1, int(h * sc))),
            interpolation=cv2.INTER_CUBIC,
        )
        tile = np.full((ch + 22, cw, 3), 30, np.uint8)
        tile[: crop.shape[0], : crop.shape[1]] = crop
        cv2.putText(
            tile,
            f"#{i} {labels[i][:28]}",
            (2, ch + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return np.zeros((64, 256, 3), np.uint8)
    th, tw = tiles[0].shape[:2]
    rows = []
    for r in range(0, len(tiles), per_row):
        chunk = tiles[r : r + per_row]
        while len(chunk) < per_row:
            chunk.append(np.full((th, tw, 3), 30, np.uint8))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


def quadrant_zooms(vis_rgb: np.ndarray, base: str, outdir: str) -> list:
    h, w = vis_rgb.shape[:2]
    names = []
    for iy, (y0, y1) in enumerate(((0, h // 2), (h // 2, h))):
        for ix, (x0, x1) in enumerate(((0, w // 2), (w // 2, w))):
            sub = vis_rgb[y0:y1, x0:x1]
            sub = cv2.resize(
                sub, (sub.shape[1] * 2, sub.shape[0] * 2), interpolation=cv2.INTER_CUBIC
            )
            name = f"{base}_zoom{iy}{ix}.png"
            cv2.imwrite(
                os.path.join(outdir, name), cv2.cvtColor(sub, cv2.COLOR_RGB2BGR)
            )
            names.append(name)
    return names


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--tag", required=True, help="street | board")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--scale", type=float, default=1.0, help="primary run scale")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--box-thresh", type=float, default=0.55)
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    ob = load_backend()
    det_path = os.path.join(args.model_dir, ob.OCR_MODEL_FILES["det"])
    rec_path = os.path.join(args.model_dir, ob.OCR_MODEL_FILES["rec"])
    keys_path = os.path.join(args.model_dir, ob.OCR_MODEL_FILES["keys"])
    for p in (det_path, rec_path, keys_path):
        if not os.path.isfile(p):
            print(f"!! missing model file: {p}")
            return 2

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"!! cannot read {args.image}")
        return 2
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    det_sess = ob._get_session(det_path)
    print(f"image      : {args.image}  {w}x{h}")
    print(f"det model  : {det_path}")
    print(f"providers  : {det_sess.get_providers()}")

    # ── sweep: how much do scale / thresholds matter? ───────────────────────
    if not args.no_sweep:
        print("\n== SWEEP (raw detector boxes, no recognizer filter) ==")
        print(
            f"  {'scale':>6} {'thresh':>7} {'box_th':>7} {'boxes':>6} "
            f"{'medH':>6} {'minH':>6} {'maxH':>6}  {'probmax':>8} {'fg%':>6}"
        )
        for scale in (0.75, 1.0, 1.5, 2.0):
            for thr, bth in ((0.3, 0.55), (0.2, 0.4), (0.15, 0.3)):
                try:
                    quads, pmax, fg = detect(ob, det_sess, img, scale, thr, bth)
                except Exception as exc:
                    print(f"  {scale:>6} {thr:>7} {bth:>7}   ERROR {exc}")
                    continue
                hs = [quad_sides(q)[1] for q in quads]
                print(
                    f"  {scale:>6} {thr:>7} {bth:>7} {len(quads):>6} "
                    f"{pct(hs, 50):>6.1f} {(min(hs) if hs else 0):>6.1f} "
                    f"{(max(hs) if hs else 0):>6.1f}  {pmax:>8.3f} {fg * 100:>5.2f}%"
                )

    # ── primary run ─────────────────────────────────────────────────────────
    print(
        f"\n== PRIMARY  scale={args.scale} thresh={args.thresh} "
        f"box_thresh={args.box_thresh} =="
    )
    quads, pmax, fg = detect(
        ob, det_sess, img, args.scale, args.thresh, args.box_thresh
    )
    print(f"raw detector boxes : {len(quads)}")

    # recognizer pass - only to measure how many boxes the node's filter kills
    rec_sess = ob._get_session(rec_path)
    keys = ob._load_keys(keys_path)
    records, labels = [], []
    empty = 0
    for i, q in enumerate(quads):
        longs, shorts = quad_sides(q)
        try:
            crop = ob._crop_quad(img, q)
            text, conf, _ = ob._rec_one(rec_sess, keys, crop)
        except Exception as exc:
            text, conf = "", 0.0
            print(f"  rec failed on box {i}: {exc}")
        if not text:
            empty += 1
        labels.append(text if text else "<EMPTY>")
        records.append(
            {
                "i": i,
                "quad": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in q],
                "bbox": [
                    round(float(q[:, 0].min()), 1),
                    round(float(q[:, 1].min()), 1),
                    round(float(q[:, 0].max()), 1),
                    round(float(q[:, 1].max()), 1),
                ],
                "h_px": round(shorts, 1),
                "w_px": round(longs, 1),
                "rec_text": text,
                "rec_conf": round(float(conf), 3),
            }
        )

    hs = [r["h_px"] for r in records]
    area = sum(cv2.contourArea(q.astype(np.float32)) for q in quads)
    print(
        f"recognizer EMPTY   : {empty} / {len(quads)}  "
        f"({100.0 * empty / max(len(quads), 1):.1f}% would be dropped by "
        f"ocr_backend.py:417)"
    )
    print(f"box area coverage  : {100.0 * area / float(w * h):.2f}% of image")
    if hs:
        print(
            f"height px  min={min(hs):.1f}  p25={pct(hs, 25):.1f}  "
            f"med={pct(hs, 50):.1f}  p75={pct(hs, 75):.1f}  max={max(hs):.1f}"
        )
        for lo, hi in ((0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 10**6)):
            n = sum(1 for v in hs if lo <= v < hi)
            hi_s = "inf" if hi > 10**5 else str(hi)
            print(
                f"    {lo:>3}-{hi_s:<4} px : {n:>4}  "
                f"{'#' * int(round(40.0 * n / max(len(hs), 1)))}"
            )

    base = f"gate_dbnet_{args.tag}"
    os.makedirs(args.outdir, exist_ok=True)
    vis = draw_overlay(img, quads)
    p_overlay = os.path.join(args.outdir, f"{base}.png")
    cv2.imwrite(p_overlay, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    p_prob = os.path.join(args.outdir, f"{base}_probmap.png")
    cv2.imwrite(
        p_prob, cv2.cvtColor(prob_overlay(img, detect.last_prob), cv2.COLOR_RGB2BGR)
    )

    sheet = contact_sheet(ob, img, quads, labels)
    p_sheet = os.path.join(args.outdir, f"{base}_crops.png")
    cv2.imwrite(p_sheet, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    zooms = quadrant_zooms(vis, base, args.outdir)

    p_json = os.path.join(args.outdir, f"{base}.json")
    with open(p_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "image": args.image,
                "w": w,
                "h": h,
                "scale": args.scale,
                "thresh": args.thresh,
                "box_thresh": args.box_thresh,
                "n_boxes": len(quads),
                "n_rec_empty": empty,
                "coverage_pct": round(100.0 * area / float(w * h), 3),
                "boxes": records,
            },
            fh,
            indent=1,
        )

    print("\nwrote:")
    for p in [p_overlay, p_prob, p_sheet, p_json] + [
        os.path.join(args.outdir, z) for z in zooms
    ]:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
