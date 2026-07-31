"""Generate real photographic scenes with Krea 2 to test against.

Synthetic cv2 rectangles are not what the node will meet in practice: real
signage has texture, uneven lighting, motion blur, depth of field and lettering
that sits on a surface rather than on a flat fill.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.parse

HOST = "http://127.0.0.1:8189"
OUT = os.path.dirname(os.path.abspath(__file__))

SCENES = {
    "street": (
        "photograph of a narrow european old town street at dusk, several small "
        "shopfronts with hanging enamel signs and painted window lettering, warm "
        "shop lights, wet cobblestones, shallow depth of field, 35mm"),
    "shelf": (
        "photograph inside a small wine shop, wooden shelf filled with bottles, "
        "paper labels facing the camera, warm tungsten light, soft shadows, 50mm, "
        "shallow depth of field"),
    "noticeboard": (
        "photograph of an office kitchen corkboard with several pinned paper "
        "notices and sticky notes at slight angles, fluorescent light, 35mm"),
}


def api(path, payload=None, timeout=120):
    url = f"{HOST}{path}"
    if payload is None:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def generate(name, prompt, seed, width=1280, height=768):
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "krea2\\krea2_turbo_fp8.safetensors",
                         "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
                         "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "seed": seed, "steps": 8, "cfg": 1.0,
                         "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": f"real_{name}"}},
    }
    res = api("/prompt", {"prompt": g})
    if "error" in res:
        raise RuntimeError(json.dumps(res)[:800])
    pid = res["prompt_id"]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 600:
        time.sleep(3)
        hist = api(f"/history/{pid}")
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        raise RuntimeError(json.dumps(m[1])[:1000])
            for out in hist[pid].get("outputs", {}).values():
                for im in out.get("images", []):
                    return im["filename"], time.perf_counter() - t0
    raise RuntimeError("timed out")


def fetch(filename, local):
    url = f"{HOST}/view?filename={urllib.parse.quote(filename)}&type=output"
    with urllib.request.urlopen(url, timeout=120) as r:
        open(local, "wb").write(r.read())
    return local


if __name__ == "__main__":
    for i, (name, prompt) in enumerate(SCENES.items()):
        fn, secs = generate(name, prompt, 1000 + i * 17)
        local = fetch(fn, os.path.join(OUT, f"real_{name}.png"))
        print(f"{name:12} {secs:5.0f}s  {os.path.basename(local)}")
