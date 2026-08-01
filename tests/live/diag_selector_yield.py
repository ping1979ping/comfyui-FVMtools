"""Diagnose: does the selector LOSE regions at the cap, or never FIND them?

Runs the selector alone (no sampling) over one image with several settings and
reports how many regions survive each stage. The trick that makes the pre-cap
count observable without touching node code: `max_regions` is the only stage
that drops by count, and `min_height_px` drops nothing at all (it only sets the
`too_small` flag), so re-running the identical detection with max_regions=100
yields exactly the number that the small cap would have thrown away.

Also builds a union-of-masks overlay so coverage can be judged by eye.
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools")
import cv2
import numpy as np
import roundtrip as R

OUT = os.path.dirname(os.path.abspath(__file__))
HOST = "http://127.0.0.1:8189"


def fetch_bytes(filename, subfolder, ftype):
    q = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": ftype}
    )
    with urllib.request.urlopen(f"{HOST}/view?{q}", timeout=120) as r:
        return r.read()


def build(info, image_name, cfg, prefix):
    g = {}
    g["1"] = {
        "class_type": "LoadImage",
        "inputs": {**R.defaults_for(info, "LoadImage"), "image": image_name},
    }
    g["2"] = {
        "class_type": "LoadSAM3Model",
        "inputs": R.defaults_for(info, "LoadSAM3Model"),
    }
    sel = R.defaults_for(info, "FVM_SignSelectorSAM3")
    sel.update(
        {
            "sam3_model": ["2", 0],
            "image": ["1", 0],
            "slop_detection": "vlm",
            "cluster_similar": True,
        }
    )
    sel.update(cfg)
    g["3"] = {"class_type": "FVM_SignSelectorSAM3", "inputs": sel}
    g["4"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["3", 3], "filename_prefix": prefix},
    }
    g["5"] = {"class_type": "MaskToImage", "inputs": {"mask": ["3", 1]}}
    g["6"] = {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}}
    g["7"] = {"class_type": "PreviewAny", "inputs": {"source": ["3", 4]}}
    g["8"] = {"class_type": "PreviewAny", "inputs": {"source": ["3", 5]}}
    return g


def run(info, image_name, cfg, prefix):
    g = build(info, image_name, cfg, prefix)
    res = R.api("/prompt", {"prompt": g})
    if "error" in res:
        raise RuntimeError(json.dumps(res)[:900])
    pid = res["prompt_id"]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 900:
        time.sleep(2)
        hist = R.api(f"/history/{pid}")
        if pid not in hist:
            continue
        st = hist[pid].get("status", {})
        if st.get("status_str") == "error":
            for m in st.get("messages", []):
                if m[0] == "execution_error":
                    raise RuntimeError(json.dumps(m[1])[:1200])
            raise RuntimeError(json.dumps(st)[:900])
        outs = hist[pid].get("outputs", {})
        if not outs:
            continue
        secs = time.perf_counter() - t0
        preview = outs.get("4", {}).get("images", [])
        masks = outs.get("6", {}).get("images", [])
        count = outs.get("7", {}).get("text", [])
        report = outs.get("8", {}).get("text", [])
        if not preview:
            raise RuntimeError(f"finished without preview: {json.dumps(outs)[:600]}")
        return {
            "preview": preview[0],
            "masks": masks,
            "count": count,
            "report": report,
            "secs": secs,
        }
    raise RuntimeError("timeout")


def union_overlay(base_bgr, mask_imgs, out_path):
    """Colour every region mask onto a copy of the original and save it."""
    canvas = base_bgr.copy()
    tint = np.zeros_like(canvas)
    palette = [
        (60, 60, 255),
        (60, 220, 255),
        (60, 255, 60),
        (255, 200, 40),
        (255, 90, 255),
        (255, 255, 60),
        (150, 90, 255),
        (60, 160, 255),
    ]
    covered = np.zeros(canvas.shape[:2], dtype=np.uint8)
    for i, m in enumerate(mask_imgs):
        raw = fetch_bytes(m["filename"], m.get("subfolder", ""), m.get("type", "temp"))
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        if arr.shape != canvas.shape[:2]:
            arr = cv2.resize(
                arr, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        binm = (arr > 127).astype(np.uint8)
        if binm.sum() == 0:
            continue
        covered |= binm
        col = palette[i % len(palette)]
        tint[binm > 0] = col
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, col, 2)
    blended = cv2.addWeighted(tint, 0.40, canvas, 1.0, 0)
    blended[tint.sum(axis=2) == 0] = canvas[tint.sum(axis=2) == 0]
    cv2.imwrite(out_path, blended)
    # The two complements are what makes the coverage claim measurable rather
    # than a matter of eyesight: censusing them separately says how much of the
    # gibberish sits inside the selection and how much sits outside it.
    stem = os.path.splitext(out_path)[0]
    cv2.imwrite(stem + "_maskunion.png", covered * 255)
    inside = base_bgr.copy()
    inside[covered == 0] = 0
    cv2.imwrite(stem + "_inside.png", inside)
    outside = base_bgr.copy()
    outside[covered > 0] = 0
    cv2.imwrite(stem + "_outside.png", outside)
    pct = 100.0 * covered.sum() / covered.size
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="file name in ComfyUI/input or a local path")
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--suite-regions", type=int, default=4)
    ap.add_argument(
        "--overlay-for",
        default="wide_uncapped,suite_settings",
        help="comma list of config names to render an overlay for",
    )
    args = ap.parse_args()

    name = os.path.basename(args.image)
    src = args.image if os.path.exists(args.image) else R.scene_path(name)
    R.upload(src, name)
    base = cv2.imread(src)
    if base is None:
        raise SystemExit(f"cannot read {src}")
    print(f"image {name}  {base.shape[1]}x{base.shape[0]}")

    configs = [
        (
            "suite_settings",
            dict(threshold_scale=0.7, min_height_px=18, max_regions=args.suite_regions),
        ),
        (
            "suite_uncapped",
            dict(threshold_scale=0.7, min_height_px=18, max_regions=100),
        ),
        ("wide_capped40", dict(threshold_scale=0.35, min_height_px=8, max_regions=40)),
        ("wide_uncapped", dict(threshold_scale=0.35, min_height_px=8, max_regions=100)),
        (
            "wide_nomerge_noarea",
            dict(
                threshold_scale=0.35,
                min_height_px=8,
                max_regions=100,
                merge_iou=1.0,
                min_area_ratio=0.0,
            ),
        ),
    ]
    want_overlay = {s.strip() for s in args.overlay_for.split(",") if s.strip()}

    results = {}
    for cname, cfg in configs:
        prefix = f"{args.tag}_{cname}"
        try:
            r = run(R_INFO, name, cfg, prefix)
        except Exception as exc:
            print(f"\n### {cname}: FAILED {exc}")
            results[cname] = {"error": str(exc)}
            continue
        n_masks = len(r["masks"])
        cnt = r["count"][0] if r["count"] else "?"
        print(f"\n### {cname}  {json.dumps(cfg)}")
        print(f"    region_count={cnt}  mask_images={n_masks}  {r['secs']:.1f}s")
        rep = "\n".join(str(x) for x in r["report"])
        print("    " + rep.replace("\n", "\n    ")[:4000])

        local_prev = os.path.join(OUT, f"{args.tag}_{cname}_preview.png")
        with open(local_prev, "wb") as f:
            f.write(
                fetch_bytes(
                    r["preview"]["filename"],
                    r["preview"].get("subfolder", ""),
                    r["preview"].get("type", "output"),
                )
            )
        entry = {
            "count": cnt,
            "masks": n_masks,
            "secs": round(r["secs"], 1),
            "preview": local_prev,
            "cfg": cfg,
        }
        if cname in want_overlay and r["masks"]:
            ov = os.path.join(OUT, f"{args.tag}_{cname}_overlay.png")
            pct = union_overlay(base, r["masks"], ov)
            entry["overlay"] = ov
            entry["covered_pct"] = round(pct, 2)
            print(f"    overlay -> {ov}   {pct:.2f}% of image area covered")
        results[cname] = entry

    print("\n=== SUMMARY " + "=" * 50)
    print(f"{'config':<22}{'detected':>10}{'emitted':>10}{'secs':>8}")
    uncapped = results.get("suite_uncapped", {}).get("count")
    for cname, _ in configs:
        e = results.get(cname, {})
        if "error" in e:
            print(f"{cname:<22}  ERROR {e['error'][:60]}")
            continue
        print(
            f"{cname:<22}{str(e.get('count')):>10}{str(e.get('masks')):>10}"
            f"{str(e.get('secs')):>8}"
        )
    with open(os.path.join(OUT, f"{args.tag}_result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"json -> {os.path.join(OUT, f'{args.tag}_result.json')}")
    return 0


if __name__ == "__main__":
    R_INFO = R.api("/object_info", timeout=300)
    raise SystemExit(main())
