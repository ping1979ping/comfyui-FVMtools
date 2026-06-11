"""FVM_Ideogram_BoxJitter — vary an Ideogram-4 caption's boxes per seed.

Perturbs each element's ``bbox`` (move / scale / independent w-h stretch) by
a ``mean ± delta`` around its current placement, always clamped to stay
**fully inside the 0-1000 frame**. Draw a layout once in the KJ node, then
generate many coherent variations just by changing the seed.

Rules live in a hidden ``jitter_rules`` JSON field owned by the JS widget:

    {
      "default":   {"pos": 0.06, "size": 0.12, "aspect": 0.08, "min": 0.03},
      "overrides": [ {"boxes": "logo", "pos": 0.12, "size": 0.03, "aspect": 0.0}, ... ]
    }

- ``default`` jitters every box. An ``overrides`` slot names one or more
  boxes (by the slot-key in their ``desc``; comma/space separated) and
  replaces the default for them — last matching slot wins. Set a slot's
  amounts to 0 to pin a box still.
- ``pos`` = ± fraction of the frame to move the center; ``size`` = ± uniform
  scale around 1.0; ``aspect`` = ± independent w/h stretch on top; ``min`` =
  smallest box edge as a fraction of the frame (collapse guard).

Box names are matched on ``desc``, so run this **before** the Assembler (which
overwrites ``desc`` with prose). Pixel-space consumers are unaffected — this
edits only the caption's normalized ``bbox``.
"""

from __future__ import annotations

import json
import random

try:
    from ...core.jb.serialize import emit_ideogram
except ImportError:  # pragma: no cover
    from core.jb.serialize import emit_ideogram


_DEFAULT_RULES = {
    "default": {"pos": 0.06, "size": 0.12, "aspect": 0.08, "min": 0.03},
    "overrides": [],
}
_FRAME = 1000.0


def _num(d, key, fallback):
    try:
        return float(d.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _rule_for(key: str, rules: dict) -> dict:
    """default, overridden by the last matching override slot."""
    default = rules.get("default", {})
    chosen = default
    for ov in rules.get("overrides", []):
        if not isinstance(ov, dict):
            continue
        targets = [t.strip().lstrip("@")
                   for t in str(ov.get("boxes", "")).replace(",", " ").split()]
        if key and key in targets:
            chosen = ov
    return {
        "pos": _num(chosen, "pos", _num(default, "pos", 0.0)),
        "size": _num(chosen, "size", _num(default, "size", 0.0)),
        "aspect": _num(chosen, "aspect", _num(default, "aspect", 0.0)),
    }


def jitter_caption(caption: dict, seed: int, rules: dict) -> dict:
    """Mutate ``caption`` in place; jitter every element bbox, clamp to frame."""
    cd = caption.get("compositional_deconstruction", {})
    floor = _num(rules.get("default", {}), "min", 0.03) * _FRAME
    floor = max(1.0, min(_FRAME, floor))
    for i, el in enumerate(cd.get("elements", [])):
        if not isinstance(el, dict):
            continue
        bb = el.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        try:
            ymin, xmin, ymax, xmax = (float(v) for v in bb)
        except (TypeError, ValueError):
            continue
        rule = _rule_for((el.get("desc") or "").strip().lstrip("@"), rules)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        w, h = abs(xmax - xmin), abs(ymax - ymin)

        rng = random.Random(f"{seed}|box{i}")
        s = 1 + rng.uniform(-rule["size"], rule["size"])
        ax = 1 + rng.uniform(-rule["aspect"], rule["aspect"])
        ay = 1 + rng.uniform(-rule["aspect"], rule["aspect"])
        w2 = min(_FRAME, max(floor, w * s * ax))
        h2 = min(_FRAME, max(floor, h * s * ay))
        cx2 = cx + rng.uniform(-rule["pos"], rule["pos"]) * _FRAME
        cy2 = cy + rng.uniform(-rule["pos"], rule["pos"]) * _FRAME
        # Clamp center so the (resized) box stays fully inside the frame.
        cx2 = min(_FRAME - w2 / 2, max(w2 / 2, cx2))
        cy2 = min(_FRAME - h2 / 2, max(h2 / 2, cy2))
        el["bbox"] = [max(0, min(1000, round(v)))
                      for v in (cy2 - h2 / 2, cx2 - w2 / 2, cy2 + h2 / 2, cx2 + w2 / 2)]
    return caption


class FVM_Ideogram_BoxJitter:
    CATEGORY = "FVM Tools/JB"
    FUNCTION = "jitter"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption_json",)
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Vary an Ideogram-4 caption's boxes per seed (move / scale / stretch),\n"
        "clamped inside the frame. A default rule jitters all boxes; add\n"
        "per-box override slots (by desc slot-key) for custom amounts.\n"
        "Run before the Assembler."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Hidden field owned by the JS widget (default block + + slots).
                "jitter_rules": ("STRING", {
                    "default": json.dumps(_DEFAULT_RULES), "multiline": True}),
                "caption_json": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Ideogram-4 caption JSON (KJ `prompt` output)."}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Same seed + same rows → identical layout variation."}),
            },
        }

    def jitter(self, jitter_rules, caption_json, seed):
        try:
            caption = json.loads(caption_json) if caption_json.strip() else {}
        except (json.JSONDecodeError, TypeError, AttributeError):
            return (caption_json,)
        if not isinstance(caption, dict):
            return (caption_json,)

        try:
            rules = json.loads(jitter_rules) if jitter_rules.strip() else {}
        except (json.JSONDecodeError, TypeError, AttributeError):
            rules = {}
        if not isinstance(rules, dict) or "default" not in rules:
            rules = _DEFAULT_RULES

        jitter_caption(caption, seed, rules)
        return (emit_ideogram(caption),)
