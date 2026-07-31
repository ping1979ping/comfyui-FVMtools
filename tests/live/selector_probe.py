"""Selector-only probe: how do the detected regions drift from pass to pass?

No sampling, so it is cheap. Runs Sign Selector SAM3 over a list of images and
saves the numbered preview for each, which carries class, size and cluster.
Use it to tell a detection problem apart from a rendering problem — the two
look identical in the final picture and need opposite fixes.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roundtrip as R


def probe(info, image_name, prefix, regions=6):
    g = {}
    g["1"] = {"class_type": "LoadImage",
              "inputs": {**R.defaults_for(info, "LoadImage"), "image": image_name}}
    g["2"] = {"class_type": "LoadSAM3Model", "inputs": R.defaults_for(info, "LoadSAM3Model")}
    sel = R.defaults_for(info, "FVM_SignSelectorSAM3")
    sel.update({"sam3_model": ["2", 0], "image": ["1", 0], "threshold_scale": 0.7,
                "min_height_px": 18, "max_regions": regions, "slop_detection": "vlm",
                "cluster_similar": True})
    g["3"] = {"class_type": "FVM_SignSelectorSAM3", "inputs": sel}
    g["4"] = {"class_type": "SaveImage",
              "inputs": {"images": ["3", 3], "filename_prefix": prefix}}
    # The masks come back too: with them the glyph step can be replayed offline,
    # which is the only way to tell a bad mask from a bad glyph layer.
    g["5"] = {"class_type": "MaskToImage", "inputs": {"mask": ["3", 1]}}
    g["6"] = {"class_type": "SaveImage",
              "inputs": {"images": ["5", 0], "filename_prefix": f"{prefix}_mask"}}

    res = R.api("/prompt", {"prompt": g})
    if "error" in res:
        raise RuntimeError(json.dumps(res)[:600])
    pid = res["prompt_id"]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 600:
        time.sleep(2)
        hist = R.api(f"/history/{pid}")
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        raise RuntimeError(json.dumps(m[1])[:900])
            outs = hist[pid].get("outputs", {})
            if outs:
                preview = outs.get("4", {}).get("images", [])
                masks = outs.get("6", {}).get("images", [])
                if preview:
                    return preview[0]["filename"], [m["filename"] for m in masks]
    raise RuntimeError("timeout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="file names already in ComfyUI/input, or local paths")
    ap.add_argument("--prefix", default="probe")
    ap.add_argument("--regions", type=int, default=6)
    args = ap.parse_args()

    info = R.api("/object_info", timeout=180)
    for i, item in enumerate(args.images):
        name = os.path.basename(item)
        if os.path.exists(item):
            R.upload(item, name)
        fn, masks = probe(info, name, f"{args.prefix}_{i}_{os.path.splitext(name)[0]}", args.regions)
        local = R.fetch(fn, os.path.join(R.OUT, f"{args.prefix}_{i}_{name}"))
        print(f"  {name:40} -> {local}")
        for j, mf in enumerate(masks):
            R.fetch(mf, os.path.join(R.OUT, f"{args.prefix}_{i}_mask{j}_{name}"))
        print(f"  {'':40}    {len(masks)} mask(s)")


if __name__ == "__main__":
    raise SystemExit(main())
