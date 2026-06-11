"""FVM_Ideogram_Assembler — marry a KJ Ideogram-4 layout with our JSONs.

KJNodes' *Ideogram 4 Prompt Builder* is a visual canvas: the user draws
boxes and types a short **slot-key** into each box's ``desc`` field. This
node takes that caption JSON and a keyed dict of per-box prompts (typically
the ``raw_json`` of an :class:`FVM_JB_Builder`) and fills each box from the
matching slot — then runs the whole caption through our wildcard engine.

Mapping (chosen design):
  - Box ↔ content by **slot-key in ``desc``** (``@`` prefix tolerated).
  - **Full field-map**: a slot's sub-dict writes ``desc`` / ``text`` /
    ``color_palette`` / ``type`` onto the element. ``bbox`` is never touched
    (KJ owns the layout). Setting ``text`` forces ``type: "text"``.
  - Reserved top-level keys (``background``, ``high_level_description``,
    ``style_description``) override the corresponding caption fields.
  - Final :func:`resolve_leaves` pass expands ``__wildcards__`` / ``{a|b}`` /
    ``__^var__`` in every string leaf (bbox ints / hex colors untouched).

Pipeline: ``KJ → [FVM_Ideogram_BoxJitter] → FVM_Ideogram_Assembler → Ideogram``.
Run jitter *before* this node — it overwrites ``desc`` with prose, after
which slot-keys are gone.
"""

from __future__ import annotations

import json

try:
    from ...core.jb.serialize import emit_ideogram, emit_strict_json
    from ...core.jb.resolve import resolve_leaves
except ImportError:  # pragma: no cover
    from core.jb.serialize import emit_ideogram, emit_strict_json
    from core.jb.resolve import resolve_leaves


_RESERVED = ("high_level_description", "background", "style_description")


def _slot_key(desc) -> str:
    return (desc or "").strip().lstrip("@")


def _apply_overrides(caption: dict, box_prompts: dict) -> dict:
    """Mutate ``caption`` in place; return a small report dict."""
    matched, unmatched_box = [], []
    used = set()

    # Reserved top-level overrides.
    cd = caption.setdefault("compositional_deconstruction", {})
    for rk in _RESERVED:
        if rk in box_prompts:
            used.add(rk)
            if rk == "background":
                cd["background"] = box_prompts[rk]
            else:
                caption[rk] = box_prompts[rk]

    for el in cd.get("elements", []):
        if not isinstance(el, dict):
            continue
        key = _slot_key(el.get("desc"))
        sub = box_prompts.get(key)
        if isinstance(sub, dict):
            matched.append(key)
            used.add(key)
            if "desc" in sub:
                el["desc"] = sub["desc"]
            if "text" in sub:
                el["text"] = sub["text"]
                el["type"] = "text"
            if "type" in sub:
                el["type"] = sub["type"]
            if "color_palette" in sub and isinstance(sub["color_palette"], list):
                el["color_palette"] = [str(c).upper() for c in sub["color_palette"]][:5]
        elif key:
            unmatched_box.append(key)

    unused = sorted(set(box_prompts) - used)
    return {"matched": matched, "unmatched_box": unmatched_box, "unused_slot": unused}


class FVM_Ideogram_Assembler:
    CATEGORY = "FVM Tools/JB"
    FUNCTION = "assemble"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "raw_json", "report")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Fill a KJ Ideogram-4 caption's boxes from our keyed JSON prompts.\n\n"
        "Each box is matched by the slot-key typed into its `desc`; the\n"
        "matching slot's sub-dict supplies desc/text/color_palette/type\n"
        "(bbox stays from KJ). Reserved keys background / "
        "high_level_description / style_description override the scene.\n"
        "A final wildcard pass expands tokens in every text field."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption_json": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Ideogram-4 caption JSON, e.g. the `prompt` output "
                               "of KJNodes' Ideogram 4 Prompt Builder (or our "
                               "BoxJitter)."}),
                "box_prompts": ("STRING", {
                    "default": "{}", "multiline": True, "forceInput": True,
                    "tooltip": "Keyed JSON: {slot: {desc,text,color_palette,type}, "
                               "..., background:\"...\"}. Wire an FVM_JB_Builder "
                               "raw_json here."}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed for the final wildcard/variable pass."}),
                "output_format": (("ideogram", "pretty_json", "compact_json"),
                                  {"default": "ideogram"}),
                "on_unmatched_box": (("keep typed desc", "clear desc"),
                                     {"default": "keep typed desc"}),
            },
            "optional": {
                "context_from_prompt_generator": ("DICT", {
                    "tooltip": "Optional adaptiveprompts context for `__^var__` recall."}),
            },
        }

    def assemble(self, caption_json, box_prompts, seed, output_format,
                 on_unmatched_box, context_from_prompt_generator=None):
        # Parse the caption. If it's unusable, pass it straight through so the
        # user sees their input rather than a silent empty node.
        try:
            caption = json.loads(caption_json) if caption_json.strip() else {}
        except (json.JSONDecodeError, TypeError, AttributeError):
            return (caption_json, "{}",
                    "ERROR: caption_json is not valid JSON — passed through unchanged.")
        if not isinstance(caption, dict):
            return (caption_json, "{}",
                    "ERROR: caption_json is not a JSON object — passed through unchanged.")

        try:
            bp = json.loads(box_prompts) if box_prompts.strip() else {}
        except (json.JSONDecodeError, TypeError, AttributeError):
            bp = {}
        if not isinstance(bp, dict):
            bp = {}

        report = _apply_overrides(caption, bp)

        if on_unmatched_box == "clear desc":
            for el in caption.get("compositional_deconstruction", {}).get("elements", []):
                if isinstance(el, dict) and _slot_key(el.get("desc")) in report["unmatched_box"]:
                    el["desc"] = ""

        # Final wildcard / variable pass over the whole caption.
        resolve_leaves(caption, seed, context_from_prompt_generator)

        if output_format == "pretty_json":
            prompt = emit_strict_json(caption, indent=2)
        elif output_format == "compact_json":
            prompt = emit_strict_json(caption, indent=None)
        else:
            prompt = emit_ideogram(caption)
        raw_json = emit_strict_json(caption, indent=2)

        report_str = (
            f"matched boxes : {report['matched']}\n"
            f"unmatched box : {report['unmatched_box']}\n"
            f"unused slots  : {report['unused_slot']}"
        )
        return (prompt, raw_json, report_str)
