"""Acceptance suite: text there and back again, across every image type.

For each scene: render text A, then B, then A again, feeding each result into the
next pass. The vision model reads every result back. A pass means the target text
is what comes out, with no second set of letters showing through.
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
import numpy as np
import cv2
import roundtrip as R
from nodes.utils.lmstudio_client import chat_vision, parse_json_response

SCENES = [
    dict(tag="street", image="real_street.png", hint="european old town street",
         texts=("WEINHANDEL", "BAECKEREI"), regions=3, box=(40, 260, 380, 760)),
    dict(tag="shelf", image="real_shelf.png", hint="wine shop shelf",
         texts=("RIESLING", "SPAETBURGUNDER"), regions=6, box=(445, 585, 215, 1025)),
    dict(tag="board", image="real_noticeboard.png", hint="office kitchen noticeboard",
         texts=("RUHETAG", "TEEKUECHE"), regions=5, box=None),
    dict(tag="synth", image="rt1_scene.png", hint="german shopfront",
         texts=("WEINHANDEL", "BAECKEREI"), regions=2, box=(60, 280, 120, 740)),
]


def apply_overrides(widgets, assignments):
    """`--set glyph_denoise=0.45` on any detailer widget, typed from its default."""
    for item in assignments:
        name, _, value = item.partition("=")
        if name not in widgets:
            raise SystemExit(f"unknown detailer widget {name!r}")
        current = widgets[name]
        widgets[name] = (value.lower() in ("1", "true", "on") if isinstance(current, bool)
                         else type(current)(value))
        print(f"  override {name} = {widgets[name]!r}")
    return widgets


def render(info, image_name, text, prefix, regions, overrides=()):
    g = {}
    g["1"] = {"class_type": "LoadImage",
              "inputs": {**R.defaults_for(info, "LoadImage"), "image": image_name}}
    g["2"] = {"class_type": "LoadSAM3Model", "inputs": R.defaults_for(info, "LoadSAM3Model")}
    sel = R.defaults_for(info, "FVM_SignSelectorSAM3")
    sel.update({"sam3_model": ["2", 0], "image": ["1", 0], "threshold_scale": 0.7,
                "min_height_px": 18, "max_regions": regions, "slop_detection": "vlm",
                "cluster_similar": True})
    g["3"] = {"class_type": "FVM_SignSelectorSAM3", "inputs": sel}
    prop = R.defaults_for(info, "FVM_SignTextProposer")
    prop.update({"sign_data": ["3", 0], "image": ["1", 0], "enabled": False,
                 "manual_override": "\n".join(f"{i}: {text}" for i in range(1, regions + 1))})
    g["4"] = {"class_type": "FVM_SignTextProposer", "inputs": prop}
    g["5"] = {"class_type": "UNETLoader",
              "inputs": {"unet_name": "krea2\\krea2_turbo_fp8.safetensors", "weight_dtype": "default"}}
    g["6"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
                         "type": "krea2", "device": "default"}}
    g["7"] = {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}}
    g["8"] = {"class_type": "FVM_SignOptions", "inputs": R.defaults_for(info, "FVM_SignOptions")}
    det = R.defaults_for(info, "FVM_SignDetailer")
    det.update({"images": ["1", 0], "sign_data": ["4", 0], "model": ["5", 0],
                "clip": ["6", 0], "vae": ["7", 0], "sign_options": ["8", 0], "seed": 11})
    apply_overrides(det, overrides)
    g["9"] = {"class_type": "FVM_SignDetailer", "inputs": det}
    g["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}}

    res = R.api("/prompt", {"prompt": g})
    if "error" in res:
        raise RuntimeError(json.dumps(res)[:600])
    pid = res["prompt_id"]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 900:
        time.sleep(3)
        hist = R.api(f"/history/{pid}")
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        raise RuntimeError(json.dumps(m[1])[:900])
            for out in hist[pid].get("outputs", {}).values():
                for im in out.get("images", []):
                    return im["filename"], time.perf_counter() - t0
    raise RuntimeError("timeout")


JUDGE = (
    "You inspect a rendered image and report the state of its signage text. "
    "Answer with ONE JSON object only, keys exactly "
    '{"contains","ghosting","legibility"}. '
    '"contains" is true if the wording given by the user appears in the image, '
    "allowing for case and line breaks. "
    '"ghosting" is 0.0-1.0: how strongly a SECOND, different set of letters shows '
    "through behind or around the main text. Judge only overlapping leftover "
    "letters, not wear, texture or shadow. "
    '"legibility" is 0.0-1.0 for how cleanly the main text reads.'
)


def judge(path, target, box):
    img = cv2.imread(path)
    if box:
        y0, y1, x0, x1 = box
        img = img[y0:y1, x0:x1]
    h, w = img.shape[:2]
    if max(h, w) > 1100:
        s = 1100 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    res = chat_vision(base_url="http://localhost:1234/v1",
                      model_id="qwen3-8b-vl-instruct-abliterated",
                      system_prompt=JUDGE,
                      user_prompt=f'The wording should read "{target}". Report the JSON.',
                      images=[cv2.cvtColor(img, cv2.COLOR_BGR2RGB)],
                      temperature=0.1, max_tokens=300, timeout=240)
    if not res.get("ok"):
        return {"error": res.get("error")}
    return parse_json_response(res.get("content", "")) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    help="detailer widget override, e.g. --set glyph_surface_restyle=1.0")
    args = ap.parse_args()

    info = R.api("/object_info", timeout=180)
    rows = []
    for sc in SCENES:
        if args.only and sc["tag"] not in args.only:
            continue
        print(f"\n{'=' * 66}\n{sc['tag']}  ({sc['image']})")
        try:
            R.upload(R.scene_path(sc["image"]), sc["image"])
        except Exception as e:
            print(f"  upload failed: {e}")
            continue
        a, b = sc["texts"]
        current = sc["image"]
        steps = []
        for label, target in (("A", a), ("B", b), ("C", a)):
            try:
                fn, secs = render(info, current, target, f"suite_{sc['tag']}_{label}",
                                  sc["regions"], args.overrides)
            except Exception as e:
                print(f"  [{label}] RENDER FAILED: {str(e)[:200]}")
                steps.append((label, target, None, {"error": "render"}))
                break
            local = R.fetch(fn, os.path.join(R.OUT, f"suite_{sc['tag']}_{label}.png"))
            v = judge(local, target, sc["box"])
            steps.append((label, target, local, v))
            ok = v.get("contains")
            gh = v.get("ghosting")
            print(f"  [{label}] {target:<16} {secs:5.0f}s  enthalten={ok}  "
                  f"ghosting={gh}  lesbarkeit={v.get('legibility')}")
            R.upload(local, f"suite_{sc['tag']}_{label}_in.png")
            current = f"suite_{sc['tag']}_{label}_in.png"
        rows.append((sc["tag"], steps))

    print(f"\n{'=' * 66}\nZUSAMMENFASSUNG")
    total_pass = 0
    total = 0
    for tag, steps in rows:
        for label, target, path, v in steps:
            total += 1
            g = v.get("ghosting")
            try:
                g = float(g)
            except (TypeError, ValueError):
                g = None
            ok = bool(v.get("contains")) and (g is not None and g <= 0.2)
            total_pass += ok
            print(f"  {tag:8} {label}  {'PASS' if ok else 'FAIL'}  "
                  f"target={target!r} contains={v.get('contains')} ghosting={g}")
    print(f"\n{total_pass}/{total} Durchgaenge bestanden")
    if total == 0:
        # Nothing ran. Reporting that as a pass is how a broken harness gets
        # mistaken for a working pipeline.
        print("KEIN Durchgang ausgefuehrt — Szenen fehlen oder ComfyUI ist nicht erreichbar")
        return 2
    return 0 if total_pass == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
