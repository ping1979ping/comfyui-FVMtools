"""Count every piece of text in a picture and say which of it is gibberish.

"The target text arrived and no second copy shows through" is too weak a test.
It passes a picture that still carries five lines of pseudo-writing beside the
one word that was replaced — which is the failure the tools exist to remove.

This asks the vision model to transcribe EVERY text element it can see, tile by
tile so small print is not skipped, and to judge each one on its own. The score
is the share of text elements that are real words.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools")
import cv2
import numpy as np

from nodes.utils.lmstudio_client import chat_vision, parse_json_response

VLM = "qwen3-8b-vl-instruct-abliterated"

CENSUS = (
    "You are auditing a rendered picture for fake writing. List EVERY separate "
    "piece of text you can see, however small, faint or partial — headings, "
    "body lines, handwriting, labels, numbers, single stray letters. "
    "Answer with ONE JSON object, key \"items\", an array. Each entry: "
    '{"text": what you read, "kind": one of "word" | "gibberish" | "unreadable"}. '
    '"word" means it is a real word, name, abbreviation or number in some real '
    "language. "
    '"gibberish" means letter shapes that pretend to be writing but spell '
    "nothing — invented strings, scrambled letters, a word broken into fragments, "
    "or the same word repeated down the page as filler. "
    '"unreadable" means you can tell text is there but cannot make out letters '
    "at all, the way distant writing looks in a photograph — that is normal and "
    "not a fault. "
    "Be exhaustive and be strict: if a string is not a word you know, it is "
    "gibberish. Report nothing else."
)


def tiles(img, rows, cols, overlap=0.12):
    h, w = img.shape[:2]
    th, tw = h / rows, w / cols
    for r in range(rows):
        for c in range(cols):
            y0 = int(max(0, r * th - th * overlap))
            y1 = int(min(h, (r + 1) * th + th * overlap))
            x0 = int(max(0, c * tw - tw * overlap))
            x1 = int(min(w, (c + 1) * tw + tw * overlap))
            yield img[y0:y1, x0:x1]


def ask(crop, scale=2):
    if scale != 1:
        crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                          interpolation=cv2.INTER_CUBIC)
    if max(crop.shape[:2]) > 1200:
        s = 1200 / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                          interpolation=cv2.INTER_AREA)
    res = chat_vision(base_url="http://localhost:1234/v1", model_id=VLM,
                      system_prompt=CENSUS,
                      user_prompt="List every piece of text in this crop as JSON.",
                      images=[cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)],
                      temperature=0.1, max_tokens=900, timeout=300)
    if not res.get("ok"):
        return []
    parsed = parse_json_response(res.get("content", "")) or {}
    items = parsed.get("items")
    return items if isinstance(items, list) else []


def census(path, rows=2, cols=2, targets=()):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    seen, items = set(), []
    for crop in tiles(img, rows, cols):
        for it in ask(crop):
            if not isinstance(it, dict):
                continue
            text = str(it.get("text", "")).strip()
            kind = str(it.get("kind", "")).strip().lower()
            key = "".join(text.upper().split())
            if not key or key in seen:
                continue
            seen.add(key)
            items.append({"text": text, "kind": kind})

    up = {"".join(t.upper().split()) for t in targets}
    good = [i for i in items if i["kind"] == "word"]
    bad = [i for i in items if i["kind"] == "gibberish"]
    blur = [i for i in items if i["kind"] == "unreadable"]
    hits = [i for i in good if "".join(i["text"].upper().split()) in up]
    return {"items": items, "word": good, "gibberish": bad, "unreadable": blur,
            "target_hits": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    out = {}
    for path in args.images:
        r = census(path, args.rows, args.cols, args.target)
        name = os.path.basename(path)
        total = len(r["word"]) + len(r["gibberish"])
        share = (len(r["gibberish"]) / total * 100) if total else 0.0
        print(f"\n{name}")
        print(f"  echte Woerter {len(r['word']):3d}   Kauderwelsch {len(r['gibberish']):3d}"
              f"   unlesbar {len(r['unreadable']):3d}   -> {share:.0f}% Slop")
        if r["target_hits"]:
            print(f"  Zieltext gefunden: {', '.join(i['text'] for i in r['target_hits'])}")
        for i in r["gibberish"]:
            print(f"    SLOP  {i['text']!r}")
        out[name] = r
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
