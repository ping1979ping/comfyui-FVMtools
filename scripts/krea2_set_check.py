"""Render check for location / outfit sets on a running ComfyUI with Krea 2.

Builds each prompt with the real JB nodes (Outfit Block, Location Block,
Stitcher in ``sentences`` format), queues a plain Krea 2 txt2img graph, waits
for the images and writes one contact sheet per run plus a JSONL log with
the exact prompt of every tile. Judging the sheet is a human / multimodal
step — see ``.claude/commands/krea2-set-test.md``.

Examples::

    venv/Scripts/python.exe scripts/krea2_set_check.py \
        --location "indoor/hospital_campus/*" --outfit female/business/board_meeting_suit_with_heels \
        --seeds 11 22 --out D:/tmp/hospital.jpg

    venv/Scripts/python.exe scripts/krea2_set_check.py --outfit "female/business/*_with_heels" \
        --location outdoor/reading_pennsylvania_downtown_spot/penn_square_street_corner --seeds 1 2 3

``*`` in a set name is a glob over the sets on disk. Every location is paired
with every outfit and every seed; the seed drives the image and both blocks.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import conftest  # noqa: F401,E402  (mocks the comfy modules for standalone imports)
from core.location_engine import get_available_location_sets  # noqa: E402
from core.outfit_lists import get_available_sets  # noqa: E402
from nodes.jb.location_block import FVM_JB_LocationBlock  # noqa: E402
from nodes.jb.outfit_block import FVM_JB_OutfitBlock  # noqa: E402
from nodes.jb.stitcher import FVM_JB_Stitcher  # noqa: E402

PRESETS = {
    # The user's everyday Krea 2 stack (krea2_*_v19/v20 workflows).
    "cielbleu": dict(model="krea2\\cielbleuKrea2_v1.safetensors",
                     loras=[("krea2\\helper\\krea2_turbo_lora_rank_64_bf16.safetensors", 0.95),
                            ("krea2\\helper\\inline-skin-lora-krea-2-raw.safetensors", 0.35)],
                     steps=10, cfg=1.2, sampler="euler", scheduler="beta"),
    # Same checkpoint without the turbo LoRA: cielbleu already has it baked in,
    # so loading it again double-applies the turbo distillation.
    "cielbleu_native": dict(model="krea2\\cielbleuKrea2_v1.safetensors",
                            loras=[("krea2\\helper\\inline-skin-lora-krea-2-raw.safetensors", 0.35)],
                            steps=10, cfg=1.2, sampler="euler", scheduler="beta"),
    # Stock Krea 2 Turbo as published.
    "turbo": dict(model="krea2\\krea2_turbo_int8_convrot.safetensors", loras=[],
                  steps=8, cfg=1.0, sampler="er_sde", scheduler="simple"),
}
CLIP = "qwen3vl_4b_int8_convrot.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
DEFAULT_PERSON = ("A 45-year-old woman, a marketing director, stands relaxed with a "
                  "friendly, confident expression, seen full body from head to shoes.")


def expand(patterns, available):
    out = []
    for p in patterns:
        hits = [s for s in available if fnmatch.fnmatch(s, p)] if "*" in p else [p]
        if not hits:
            raise SystemExit(f"no set matches {p!r}")
        out += [h for h in hits if h not in out]
    return out


def location_text(name, seed, args):
    _, txt, _ = FVM_JB_LocationBlock().build(
        name, seed,
        enable_background=True, enable_midground=True,
        enable_architecture_detail=not args.no_detail, enable_props=not args.no_props,
        enable_foreground_element=True, enable_time_of_day=True, enable_weather=True,
        color_mood=args.location_mood, output_format="sentences")
    return txt


def outfit_text(name, seed, args):
    _, txt, _ = FVM_JB_OutfitBlock().build(
        name, seed, args.style, args.formality, 0.5,
        True, True, True, True, True, args.accessories, args.bag,
        0.15, "off", color_mood=args.outfit_mood, output_format="sentences")
    return txt


def stitch(*parts):
    kw = {f"input_{i + 1}": p for i, p in enumerate(parts) if p}
    return FVM_JB_Stitcher().stitch("scene", "sentences", **kw)[1]


def graph(text, seed, prefix, preset, w, h):
    p = PRESETS[preset]
    g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": p["model"], "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "krea2", "device": "default"}},
         "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}}
    mref = ["1", 0]
    for i, (lora, s) in enumerate(p["loras"]):
        g[f"L{i}"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": mref, "lora_name": lora, "strength_model": s}}
        mref = [f"L{i}", 0]
    g.update({
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": text}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": mref, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
            "seed": seed, "steps": p["steps"], "cfg": p["cfg"], "sampler_name": p["sampler"],
            "scheduler": p["scheduler"], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    })
    return g


def _req(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run(jobs, base, out_dir):
    client = str(uuid.uuid4())
    for j in jobs:
        res = _req(f"{base}/prompt", {"prompt": j.pop("graph"), "client_id": client})
        if res.get("node_errors"):
            raise SystemExit(json.dumps(res["node_errors"])[:2000])
        j["prompt_id"] = res["prompt_id"]
    for j in jobs:
        while True:
            hist = _req(f"{base}/history/{j['prompt_id']}")
            entry = hist.get(j["prompt_id"])
            if entry:
                if entry.get("status", {}).get("status_str") == "error":
                    raise SystemExit(json.dumps(entry["status"])[:2000])
                ims = [im for o in entry["outputs"].values() for im in o.get("images", [])]
                if ims:
                    j["path"] = os.path.join(out_dir, ims[0].get("subfolder", ""), ims[0]["filename"])
                    break
            time.sleep(1.5)
    return jobs


def sheet(jobs, path, cols, thumb_w):
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(j["path"]).convert("RGB") for j in jobs]
    th = int(thumb_w * ims[0].height / ims[0].width)
    cap = 34
    rows = (len(ims) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_w, rows * (th + cap)), "white")
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for i, (im, j) in enumerate(zip(ims, jobs)):
        x, y = (i % cols) * thumb_w, (i // cols) * (th + cap)
        canvas.paste(im.resize((thumb_w, th)), (x, y + cap))
        label = j["label"]
        n = max(8, thumb_w // 7)
        d.text((x + 3, y + 2), label[:n], fill="black", font=font)
        d.text((x + 3, y + 17), label[n:2 * n], fill="black", font=font)
    canvas.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--location", nargs="+", required=True, help="location set(s), * = glob")
    ap.add_argument("--outfit", nargs="+", default=["female/business/pantsuit_executive"],
                    help="outfit set(s), * = glob")
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22])
    ap.add_argument("--person", default=DEFAULT_PERSON)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="cielbleu_native")
    ap.add_argument("--formality", type=float, default=0.7)
    ap.add_argument("--style", default="business_executive", help="Outfit Block style_preset")
    ap.add_argument("--outfit-mood", default="neutral_basics", help="Outfit Block color_mood")
    ap.add_argument("--location-mood", default="everyday_muted", help="Location Block color_mood")
    ap.add_argument("--accessories", action="store_true")
    ap.add_argument("--bag", action="store_true")
    ap.add_argument("--no-props", action="store_true")
    ap.add_argument("--no-detail", action="store_true")
    ap.add_argument("--size", default="720x1280")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--comfy-output", default=os.path.normpath(os.path.join(ROOT, "..", "..", "output")))
    ap.add_argument("--prefix", default="fvm_krea2_check/run")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--thumb", type=int, default=300)
    ap.add_argument("--out", required=True, help="contact sheet .jpg (a .jsonl log is written next to it)")
    args = ap.parse_args()

    locs = expand(args.location, get_available_location_sets())
    outfits = expand(args.outfit, get_available_sets())
    w, h = (int(v) for v in args.size.lower().split("x"))
    jobs = []
    for loc in locs:
        for outfit in outfits:
            for seed in args.seeds:
                text = stitch(args.person, outfit_text(outfit, seed, args), location_text(loc, seed, args))
                label = f"{loc.split('/')[-1]} | {outfit.split('/')[-1]} | s{seed}"
                safe = f"{loc.split('/')[-1]}_{outfit.split('/')[-1]}_s{seed}"
                jobs.append(dict(label=label, location=loc, outfit=outfit, seed=seed, text=text,
                                 chars=len(text),
                                 graph=graph(text, seed, f"{args.prefix}_{safe}", args.preset, w, h)))
    long = [j for j in jobs if j["chars"] > 2400]
    if long:
        print(f"WARNING: {len(long)} prompt(s) over 2400 chars - Krea 2 may black out", file=sys.stderr)
    t0 = time.time()
    run(jobs, f"http://127.0.0.1:{args.port}", args.comfy_output)
    sheet(jobs, args.out, args.cols, args.thumb)
    log = os.path.splitext(args.out)[0] + ".jsonl"
    with open(log, "w", encoding="utf-8") as fh:
        for j in jobs:
            fh.write(json.dumps({k: j[k] for k in ("label", "location", "outfit", "seed", "chars", "text", "path")},
                                ensure_ascii=False) + "\n")
    print(f"{len(jobs)} images in {time.time() - t0:.0f}s -> {args.out}")
    print(f"prompts: {log}  (max {max(j['chars'] for j in jobs)} chars)")


if __name__ == "__main__":
    main()
