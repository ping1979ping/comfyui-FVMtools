"""Copy of slop_census with knobs, so the census can be measured against itself.

Nothing here is imported by the suite. It exists to answer one question: how
much of a slop count is the picture and how much is the vision model rolling
dice. `slop_census.py` is left untouched on purpose — a measuring instrument you
edit while measuring it tells you nothing.

Two variants are under test:
  * temperature 0.0 instead of 0.1 in `ask()`
  * majority-of-three: run the census three times and keep only strings that
    two of the three runs agreed were gibberish.
"""

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools")
import cv2

from nodes.utils.lmstudio_client import chat_vision, parse_json_response

import slop_census as SC

VLM = SC.VLM
CENSUS = SC.CENSUS
tiles = SC.tiles


def key_of(text):
    """The same normalisation slop_census uses for de-duplication."""
    return "".join(str(text).upper().split())


def ask(crop, scale=2, temperature=0.1, seed=None):
    """slop_census.ask with the temperature (and optional seed) exposed."""
    if scale != 1:
        crop = cv2.resize(
            crop,
            (crop.shape[1] * scale, crop.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
    if max(crop.shape[:2]) > 1200:
        s = 1200 / max(crop.shape[:2])
        crop = cv2.resize(
            crop,
            (int(crop.shape[1] * s), int(crop.shape[0] * s)),
            interpolation=cv2.INTER_AREA,
        )
    res = chat_vision(
        base_url="http://localhost:1234/v1",
        model_id=VLM,
        system_prompt=CENSUS,
        user_prompt="List every piece of text in this crop as JSON.",
        images=[cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)],
        temperature=temperature,
        max_tokens=900,
        timeout=300,
        seed=seed,
    )
    if not res.get("ok"):
        return [], res.get("error")
    parsed = parse_json_response(res.get("content", "")) or {}
    items = parsed.get("items")
    return (items if isinstance(items, list) else []), None


def census(path, rows=2, cols=2, targets=(), temperature=0.1, seed=None):
    """slop_census.census, plus timing, per-tile detail and transport errors."""
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    seen, items = set(), []
    per_tile, errors = [], []
    t0 = time.perf_counter()
    for ti, crop in enumerate(tiles(img, rows, cols)):
        raw, err = ask(crop, temperature=temperature, seed=seed)
        if err:
            errors.append({"tile": ti, "error": str(err)[:200]})
        kept = 0
        for it in raw:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text", "")).strip()
            kind = str(it.get("kind", "")).strip().lower()
            key = key_of(text)
            if not key or key in seen:
                continue
            seen.add(key)
            items.append({"text": text, "kind": kind, "tile": ti})
            kept += 1
        per_tile.append({"tile": ti, "raw": len(raw), "kept": kept})
    secs = time.perf_counter() - t0

    up = {key_of(t) for t in targets}
    good = [i for i in items if i["kind"] == "word"]
    bad = [i for i in items if i["kind"] == "gibberish"]
    blur = [i for i in items if i["kind"] == "unreadable"]
    hits = [i for i in good if key_of(i["text"]) in up]
    target_as_slop = [i for i in bad if key_of(i["text"]) in up]
    return {
        "items": items,
        "word": good,
        "gibberish": bad,
        "unreadable": blur,
        "target_hits": hits,
        "target_as_slop": target_as_slop,
        "secs": secs,
        "per_tile": per_tile,
        "errors": errors,
    }


def majority_of(runs, k=2):
    """Keys called gibberish by at least `k` of the given census results."""
    c = Counter()
    for r in runs:
        c.update({key_of(i["text"]) for i in r["gibberish"]})
    return {key for key, n in c.items() if n >= k}
