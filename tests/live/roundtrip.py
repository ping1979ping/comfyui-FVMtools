"""Roundtrip test: text1 -> text2 -> text1, judged by the vision model.

Rendering the same region three times, changing the text and changing it back,
is the sharpest test for bleed-through there is: if anything of the previous
lettering survives, it accumulates visibly across the chain, and the final image
should otherwise match the first.

Each result is read back by the vision model, which reports what it actually
sees and how much ghosting is present — a judgement the pixel metrics cannot make.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, "D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools")
import numpy as np
import cv2

from nodes.utils.lmstudio_client import chat_vision, parse_json_response

HOST = "http://127.0.0.1:8189"
VLM = "qwen3-8b-vl-instruct-abliterated"
OUT = os.path.dirname(os.path.abspath(__file__))
# Scenes made by make_real.py are written here, but they are also already
# sitting in ComfyUI's input folder from the run that made them. Look there
# too, so a fresh clone does not have to regenerate them.
COMFY_INPUT = "D:/AI/ComfyUI/ComfyUI/input"


def scene_path(name):
    """Local copy of a test scene, wherever it happens to live."""
    local = os.path.join(OUT, name)
    if os.path.exists(local):
        return local
    return os.path.join(COMFY_INPUT, name)

JUDGE_SYSTEM = (
    "You inspect a crop of a sign in a rendered image and report what is actually "
    "there. Answer with ONE JSON object and nothing else, keys exactly: "
    '{"text","ghosting","legibility","artifacts"}. '
    '"text" is the lettering you can read, transcribed exactly as printed. '
    '"ghosting" is 0.0 to 1.0: how strongly a SECOND, different set of letters '
    "shows through behind or around the main text. 0.0 means perfectly clean, "
    "1.0 means two texts are equally visible. Judge only overlapping leftover "
    "letters, not normal wear, texture or shadow. "
    '"legibility" is 0.0 to 1.0 for how cleanly the main text reads. '
    '"artifacts" is a short phrase naming any distortion, or "none".'
)


def api(path, payload=None, timeout=60):
    url = f"{HOST}{path}"
    if payload is None:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def upload(path, name):
    """Multipart upload without external deps."""
    boundary = "----fvm" + str(len(name) * 7919)
    with open(path, "rb") as f:
        data = f.read()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode(),
        b"Content-Type: image/png\r\n\r\n", data, b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(f"{HOST}/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def defaults_for(info, node_type):
    spec = info[node_type]["input"]
    out = {}
    for section in ("required", "optional"):
        for name, d in spec.get(section, {}).items():
            if not isinstance(d, list) or not d:
                continue
            typ = d[0]
            opts = d[1] if len(d) > 1 and isinstance(d[1], dict) else {}
            if isinstance(typ, list):
                out[name] = opts.get("default", typ[0] if typ else "")
            elif typ in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                out[name] = opts.get("default", {"INT": 0, "FLOAT": 0.0,
                                                 "STRING": "", "BOOLEAN": False}[typ])
    return out


def render(info, image_name, text, out_prefix, detailer_overrides=None):
    """One pass: detect, force `text` on every region, re-render."""
    g = {}
    g["1"] = {"class_type": "LoadImage",
              "inputs": {**defaults_for(info, "LoadImage"), "image": image_name}}
    g["2"] = {"class_type": "LoadSAM3Model", "inputs": defaults_for(info, "LoadSAM3Model")}

    sel = defaults_for(info, "FVM_SignSelectorSAM3")
    sel.update({"sam3_model": ["2", 0], "image": ["1", 0], "threshold_scale": 0.7,
                "min_height_px": 20, "max_regions": 4, "slop_detection": "vlm",
                "cluster_similar": False})
    g["3"] = {"class_type": "FVM_SignSelectorSAM3", "inputs": sel}

    prop = defaults_for(info, "FVM_SignTextProposer")
    prop.update({"sign_data": ["3", 0], "image": ["1", 0], "enabled": False,
                 "manual_override": "\n".join(f"{i}: {text}" for i in range(1, 5))})
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
    g["8"] = {"class_type": "FVM_SignOptions", "inputs": opt}

    det = defaults_for(info, "FVM_SignDetailer")
    det.update({"images": ["1", 0], "sign_data": ["4", 0], "model": ["5", 0],
                "clip": ["6", 0], "vae": ["7", 0], "sign_options": ["8", 0], "seed": 7})
    det.update(detailer_overrides or {})
    g["9"] = {"class_type": "FVM_SignDetailer", "inputs": det}
    g["10"] = {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0], "filename_prefix": out_prefix}}

    res = api("/prompt", {"prompt": g})
    if "error" in res:
        raise RuntimeError(json.dumps(res)[:800])
    pid = res["prompt_id"]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 600:
        time.sleep(2)
        hist = api(f"/history/{pid}")
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        raise RuntimeError(json.dumps(m[1])[:1200])
            for out in hist[pid].get("outputs", {}).values():
                for im in out.get("images", []):
                    return im["filename"], time.perf_counter() - t0
            raise RuntimeError("finished but produced no image")
    raise RuntimeError("render timed out")


def fetch(filename, local):
    url = f"{HOST}/view?filename={urllib.parse.quote(filename)}&type=output"
    with urllib.request.urlopen(url, timeout=60) as r:
        with open(local, "wb") as f:
            f.write(r.read())
    return local


def judge(image_bgr, box):
    """Ask the vision model what it sees in the region."""
    y0, y1, x0, x1 = box
    crop = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    res = chat_vision(base_url="http://localhost:1234/v1", model_id=VLM,
                      system_prompt=JUDGE_SYSTEM,
                      user_prompt="Inspect this sign and answer with the JSON object.",
                      images=[crop], temperature=0.1, max_tokens=400, timeout=240)
    if not res.get("ok"):
        return {"error": res.get("error")}
    parsed = parse_json_response(res.get("content", "")) or {}
    return parsed


import urllib.parse  # noqa: E402  (used by fetch)
