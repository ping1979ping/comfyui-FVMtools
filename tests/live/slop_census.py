"""Count every piece of text in a picture and say which of it is gibberish.

"The target text arrived and no second copy shows through" is too weak a test.
It passes a picture that still carries five lines of pseudo-writing beside the
one word that was replaced — which is the failure the tools exist to remove.

This asks the vision model to transcribe EVERY text element it can see, tile by
tile so small print is not skipped, and to judge each one on its own. The score
is the share of text elements that are real words.

Four things here are the way they are because they were measured — 100 repeats
of this census on unchanged pictures, written up in `noise_census.md`:

* **A failed tile is an error, never an empty tile.** `ask` used to return `[]`
  when the HTTP call failed, so a quarter of the picture vanished from the count
  and the run reported LESS slop instead of a fault. The bug makes a result look
  BETTER than it is. Seen in 4 of 100 runs; it explains a reported drop from 13
  to 4 in full. Tiles are now retried, and a tile that never answers is counted
  and reported, so the caller can refuse to compare that number.
* **temperature 0.0, not 0.1.** Costs nothing in runtime (583 s against 584 s
  over 50 runs) and takes the strings that appeared exactly once, across all
  pictures, from 22 down to 0.
* **The target word is not gibberish — but a fragment of it still is.** The
  vision model files a correctly rendered `RUHETAG` under "gibberish" in 10 of
  10 runs, and `KUHETAG` beside it in the same 10 — one rendered word, read
  twice. A target word therefore cost two points and its scene could never pass.
  Excused are: the target verbatim, and a reading of the same blob that is at
  least as long as the target. NOT excused is a SHORTER reading (`IESLIN`, `SLIN`
  for `RIESLING`): letters are missing there, so this tool's own lettering came
  out cut off. Length is what separates the two — a misread substitutes letters,
  a truncation loses them. Fragments count as faults and are reported in their
  own bucket, so the self-inflicted share of the slop stays visible.
* **Spelling variants of one scribble are one finding.** `PAKTRDE`, `PAXTRDE`,
  `PAXTRON` and `PAXTROS` are four readings of a single smudge; de-duplicating
  on the exact string counted them four times. The 32 "different strings" on
  board_A are 16 blobs.
"""
import argparse
import difflib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools")
import cv2
import numpy as np

from nodes.utils.lmstudio_client import chat_vision, parse_json_response

VLM = "qwen3-8b-vl-instruct-abliterated"

# Tries per tile before it is declared lost. The observed failure is a transient
# `HTTP 400 {"error":"terminated"}` from LM Studio at roughly 1 call in 100 — a
# second try has always been enough, the third is margin.
TILE_ATTEMPTS = 3

# How close a string must be to a target word before it is read as this tool's
# own lettering rather than someone else's invention. Same measure and same
# normalisation as `_fuzzy_match` in nodes/signs/detailer.py, so the two agree on
# what "nearly the same string" means:
#     KUHETAG vs RUHETAG   0.86      IESLIN vs RIESLING  0.86
#     ITZPLT  vs PUTZPLAN  0.57  ->  stays foreign invention
# Short fragments of long words fall below it (`SBURGU` against
# `WEISSBURGUNDER` is 0.60). Those are then counted as `gibberish` rather than
# as `target_fragment` — the total is unaffected, only the attribution is.
TARGET_SIMILARITY = 0.72

# Spelling variants of ONE scribble into one blob. Deliberately a shade below
# TARGET_SIMILARITY: the readings of the street_A smudge chain
# `PAKTRDE`-`PAXTRDE`-`PAXTRON`-`PAXTROS` through a 0.71 link, and 0.72 would
# cut that single blob into two counted findings.
CLUSTER_SIMILARITY = 0.70

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


class TileError(RuntimeError):
    """A tile the vision model never answered for.

    Raised instead of returning `[]`, because an empty list is indistinguishable
    from a tile with no text on it — and that is exactly how a broken run came to
    report less slop than a working one.
    """


def norm(text):
    """Comparison key: upper case, all whitespace removed."""
    return "".join(str(text).upper().split())


def similarity(a, b):
    """Similarity of two strings, case- and whitespace-insensitive."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


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


def ask(crop, scale=2, attempts=TILE_ATTEMPTS):
    """Transcribe one tile. Raises `TileError` when every attempt failed.

    A transport failure and an answer holding no JSON are the same accident:
    both used to turn into an empty tile. Both are retried, and both end as a
    reported error rather than as silence.
    """
    if scale != 1:
        crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                          interpolation=cv2.INTER_CUBIC)
    if max(crop.shape[:2]) > 1200:
        s = 1200 / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                          interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    problem = "no attempt made"
    for _ in range(max(1, int(attempts))):
        res = chat_vision(base_url="http://localhost:1234/v1", model_id=VLM,
                          system_prompt=CENSUS,
                          user_prompt="List every piece of text in this crop as JSON.",
                          images=[rgb],
                          temperature=0.0, max_tokens=900, timeout=300)
        if not res.get("ok"):
            problem = str(res.get("error") or "vision request failed")[:200]
            continue
        parsed = parse_json_response(res.get("content", ""))
        if parsed is None:
            problem = "unparsable answer: " + repr(res.get("content", ""))[:160]
            continue
        items = parsed.get("items")
        return items if isinstance(items, list) else []
    raise TileError(problem)


def cluster(entries, threshold=CLUSTER_SIMILARITY):
    """Group spelling variants of one scribble into one blob.

    Single linkage — an entry joins as soon as it is close enough to ANY member,
    not to a centre. The readings of one smudge form a chain, not a ball:
    `PAKTRDE`→`PAXTRDE`→`PAXTRON`→`PAXTROS` has no single string that every
    member is close to.
    """
    groups = []
    for entry in entries:
        key = norm(entry.get("text", ""))
        for group in groups:
            if any(similarity(key, m.get("text", "")) >= threshold for m in group):
                group.append(entry)
                break
        else:
            groups.append([entry])
    return groups


def entry_for(group, kind, **extra):
    """One counted finding: the blob's best reading plus its spelling variants."""
    found = {"text": representative(group)["text"], "kind": kind,
             "variants": [m.get("text", "") for m in group]}
    found.update(extra)
    return found


def representative(group):
    """The longest reading in a blob — the most complete one. Ties: first seen."""
    best = group[0]
    for member in group[1:]:
        if len(norm(member.get("text", ""))) > len(norm(best.get("text", ""))):
            best = member
    return best


def closest_target(group, targets, threshold=TARGET_SIMILARITY):
    """Best (target, ratio) over every member of the blob, or None if none is close."""
    best, score = None, 0.0
    for member in group:
        for target in targets or ():
            ratio = similarity(member.get("text", ""), target)
            if ratio > score:
                best, score = target, ratio
    return (best, round(score, 3)) if score >= threshold else None


def census(path, rows=2, cols=2, targets=()):
    """Transcribe the whole picture tile by tile and sort the findings.

    Besides the raw items it returns:
      `gibberish`        blobs of foreign invention                — counted
      `target_fragment`  blobs resembling a target word            — counted
      `target_exact`     the target word verbatim, and readings of the same
                         blob that are no shorter than it            — NOT counted
      `slop`             gibberish + target_fragment, the number that matters
      `raw_gibberish`    every gibberish string before clustering (the old count)
      `tile_errors`      tiles that never answered. While this is not 0 a piece
                         of the picture is missing from the count and the total
                         is not comparable with another run.
    """
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    seen, items = set(), []
    tile_errors = []
    for index, crop in enumerate(tiles(img, rows, cols)):
        try:
            raw = ask(crop)
        except TileError as exc:
            # Carry on, so the report still shows what the surviving tiles saw —
            # but the count is incomparable now, and it says so.
            tile_errors.append({"tile": index, "error": str(exc)[:200]})
            continue
        for it in raw:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text", "")).strip()
            kind = str(it.get("kind", "")).strip().lower()
            key = norm(text)
            if not key or key in seen:
                continue
            seen.add(key)
            items.append({"text": text, "kind": kind})

    up = {norm(t) for t in targets}
    good = [i for i in items if i["kind"] == "word"]
    raw_bad = [i for i in items if i["kind"] == "gibberish"]
    blur = [i for i in items if i["kind"] == "unreadable"]
    hits = [i for i in good if norm(i["text"]) in up]

    gibberish, fragments, excused = [], [], []
    for group in cluster(raw_bad):
        exact = [m for m in group if norm(m["text"]) in up]
        if exact:
            # The target word itself, filed under "gibberish" by the vision
            # model. It stands correctly in the picture, so it is no fault — and
            # neither is a misreading of it that is just as long, which is one
            # rendered word read twice (`KUHETAG` beside `RUHETAG`, in 10 of 10
            # runs).
            #
            # A SHORTER reading in the same blob is a different animal: letters
            # are missing, so the lettering itself came out cut off. Excusing the
            # whole blob would hide exactly that fault behind the correct word
            # standing next to it, so the blob is split and the short readings
            # stay counted.
            floor = min(len(norm(m["text"])) for m in exact)
            whole = [m for m in group if len(norm(m["text"])) >= floor]
            cut = [m for m in group if len(norm(m["text"])) < floor]
            excused.append(entry_for(whole, "target_exact", target=exact[0]["text"]))
            if cut:
                near = closest_target(cut, [exact[0]["text"]], threshold=0.0)
                fragments.append(entry_for(
                    cut, "target_fragment", target=exact[0]["text"],
                    similarity=near[1] if near else 0.0))
            continue
        near = closest_target(group, targets)
        if near:
            # Cut-off or mangled lettering of a word this tool wrote itself
            # (`IESLIN` for `RIESLING`). A real fault, so it counts — but it is
            # reported apart from foreign invention, because the two call for
            # completely different repairs.
            fragments.append(entry_for(group, "target_fragment",
                                       target=near[0], similarity=near[1]))
        else:
            gibberish.append(entry_for(group, "gibberish"))

    return {"items": items, "word": good, "unreadable": blur,
            "target_hits": hits,
            "gibberish": gibberish, "target_fragment": fragments,
            "target_exact": excused, "slop": gibberish + fragments,
            "raw_gibberish": raw_bad,
            "tile_errors": len(tile_errors), "tile_error_detail": tile_errors}


def variant_note(entry):
    extra = len(entry.get("variants", [])) - 1
    return f"   (+{extra} Schreibvariante{'n' if extra != 1 else ''})" if extra > 0 else ""


def report(name, r):
    """Print one picture's findings, the old count beside the new one."""
    total = len(r["word"]) + len(r["slop"])
    share = (len(r["slop"]) / total * 100) if total else 0.0
    print(f"\n{name}")
    print(f"  echte Woerter {len(r['word']):3d}   Slop {len(r['slop']):3d}"
          f"   unlesbar {len(r['unreadable']):3d}   -> {share:.0f}% Slop")
    print(f"  alt (exakte Strings) {len(r['raw_gibberish']):3d}"
          f"   neu (Blobs) {len(r['slop']):3d}"
          f"   davon fremd {len(r['gibberish'])}"
          f" / Zielfragment {len(r['target_fragment'])}"
          f"   Kachelfehler {r['tile_errors']}")
    if r["tile_errors"]:
        print("  ACHTUNG: Kachel(n) ohne Antwort — Bildflaeche fehlt in der "
              "Zaehlung, Zahl NICHT vergleichbar")
        for e in r["tile_error_detail"]:
            print(f"    KACHELFEHLER {e['tile']}: {e['error']}")
    if r["target_hits"]:
        print(f"  Zieltext gefunden: {', '.join(i['text'] for i in r['target_hits'])}")
    for i in r["gibberish"]:
        print(f"    SLOP      {i['text']!r}{variant_note(i)}")
    for i in r["target_fragment"]:
        print(f"    FRAGMENT  {i['text']!r} ~ {i['target']!r} {i['similarity']}"
              f"{variant_note(i)}")
    for i in r["target_exact"]:
        print(f"    ZIELWORT  {i['text']!r} (nicht gewertet){variant_note(i)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    out = {}
    broken = 0
    for path in args.images:
        r = census(path, args.rows, args.cols, args.target)
        name = os.path.basename(path)
        report(name, r)
        broken += r["tile_errors"]
        out[name] = r
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    if broken:
        print(f"\n{broken} Kachelfehler — dieser Lauf ist nicht vergleichbar")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
