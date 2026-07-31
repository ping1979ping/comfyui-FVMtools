"""End-to-end Sign Tools run on a live ComfyUI with Krea 2.

Builds the API-format graph from the server's own object_info, so widget
defaults and names can never drift out of sync with the node code.

Usage: krea_run.py [--image NAME] [--out TAG] [--set node.widget=value ...]
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.parse

HOST = "http://127.0.0.1:8189"


def api(path, payload=None):
    url = f"{HOST}{path}"
    if payload is None:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def defaults_for(info, node_type):
    """Widget defaults for a node, straight from the server."""
    spec = info[node_type]["input"]
    out = {}
    for section in ("required", "optional"):
        for name, definition in spec.get(section, {}).items():
            if not isinstance(definition, list) or not definition:
                continue
            typ = definition[0]
            opts = definition[1] if len(definition) > 1 and isinstance(definition[1], dict) else {}
            if isinstance(typ, list):                 # COMBO
                out[name] = opts.get("default", typ[0] if typ else "")
            elif typ in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                out[name] = opts.get("default", {"INT": 0, "FLOAT": 0.0,
                                                 "STRING": "", "BOOLEAN": False}[typ])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="int_before.png")
    ap.add_argument("--out", default="signtest")
    ap.add_argument("--set", action="append", default=[],
                    help="node.widget=value, e.g. detailer.glyph_denoise=0.45")
    args = ap.parse_args()

    info = api("/object_info")
    for needed in ("LoadSAM3Model", "FVM_SignSelectorSAM3", "FVM_SignTextProposer",
                   "FVM_SignDetailer", "FVM_SignOptions", "UNETLoader",
                   "CLIPLoader", "VAELoader"):
        if needed not in info:
            print(f"MISSING NODE: {needed}")
            return 1

    g = {}
    g["1"] = {"class_type": "LoadImage",
              "inputs": {**defaults_for(info, "LoadImage"), "image": args.image}}
    g["2"] = {"class_type": "LoadSAM3Model", "inputs": defaults_for(info, "LoadSAM3Model")}

    sel = defaults_for(info, "FVM_SignSelectorSAM3")
    sel.update({"sam3_model": ["2", 0], "image": ["1", 0],
                "threshold_scale": 0.7, "min_height_px": 20, "max_regions": 8,
                "slop_detection": "vlm"})
    g["3"] = {"class_type": "FVM_SignSelectorSAM3", "inputs": sel}

    prop = defaults_for(info, "FVM_SignTextProposer")
    prop.update({"sign_data": ["3", 0], "image": ["1", 0],
                 "model_id": "qwen3-8b-vl-instruct-abliterated",
                 "scene_hint": "German wine shop", "language": "de",
                 "max_tokens": 400, "timeout": 240})
    g["4"] = {"class_type": "FVM_SignTextProposer", "inputs": prop}

    g["5"] = {"class_type": "UNETLoader",
              "inputs": {"unet_name": "krea2\\krea2_turbo_fp8.safetensors",
                         "weight_dtype": "default"}}
    g["6"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
                         "type": "krea2", "device": "default"}}
    g["7"] = {"class_type": "VAELoader",
              "inputs": {"vae_name": "qwen_image_vae.safetensors"}}

    opt = defaults_for(info, "FVM_SignOptions")
    opt.update({"cfg": 1.0})
    g["8"] = {"class_type": "FVM_SignOptions", "inputs": opt}

    det = defaults_for(info, "FVM_SignDetailer")
    det.update({"images": ["1", 0], "sign_data": ["4", 0], "model": ["5", 0],
                "clip": ["6", 0], "vae": ["7", 0], "sign_options": ["8", 0],
                # Krea 2 Turbo: distilled, 8 steps at cfg 1, er_sde/simple
                "steps": 8, "sampler_name": "er_sde", "scheduler": "simple",
                "seed": 42})
    for assignment in args.set:
        node, _, value = assignment.partition("=")
        prefix, _, widget = node.partition(".")
        if prefix == "sel" and widget in sel:
            current = sel[widget]
            sel[widget] = (type(current)(value) if not isinstance(current, bool)
                           else value.lower() in ("1", "true", "on"))
            g["3"]["inputs"] = sel
            print(f"  override selector.{widget} = {sel[widget]!r}")
        elif widget in det:
            current = det[widget]
            det[widget] = (type(current)(value) if not isinstance(current, bool)
                           else value.lower() in ("1", "true", "on"))
            print(f"  override detailer.{widget} = {det[widget]!r}")
        elif widget in opt:
            opt[widget] = value
            print(f"  override options.{widget} = {value!r}")
    g["9"] = {"class_type": "FVM_SignDetailer", "inputs": det}

    g["10"] = {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0], "filename_prefix": args.out}}
    g["11"] = {"class_type": "SaveImage",
               "inputs": {"images": ["3", 5], "filename_prefix": args.out + "_preview"}}

    print(f"submitting graph ({len(g)} nodes) ...")
    res = api("/prompt", {"prompt": g})
    if "error" in res:
        print(json.dumps(res, indent=2)[:3000])
        return 1
    pid = res["prompt_id"]
    print(f"prompt_id {pid}")

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 900:
        time.sleep(3)
        hist = api(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            print(f"done in {time.perf_counter()-t0:.0f}s — {status.get('status_str')}")
            if status.get("status_str") == "error":
                for m in status.get("messages", []):
                    if m[0] in ("execution_error", "execution_interrupted"):
                        print(json.dumps(m[1], indent=2)[:2500])
                return 1
            for node_id, out in entry.get("outputs", {}).items():
                for im in out.get("images", []):
                    print(f"  image: {im['filename']} ({im['type']}/{im.get('subfolder','')})")
                for key in ("text", "string"):
                    if key in out:
                        print(f"  node {node_id} {key}: {str(out[key])[:400]}")
            return 0
    print("timed out")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
