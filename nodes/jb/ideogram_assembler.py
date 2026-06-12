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
    from ...core.jb.serialize import emit, emit_ideogram, emit_strict_json, parse_input
    from ...core.jb.resolve import resolve_leaves
except ImportError:  # pragma: no cover
    from core.jb.serialize import emit, emit_ideogram, emit_strict_json, parse_input
    from core.jb.resolve import resolve_leaves


_RESERVED = ("high_level_description", "background", "style_description")
# Keys that map onto an Ideogram element directly. Anything else in a slot's
# sub-dict is treated as nested *content* and folded into ``desc``.
_FIELDMAP_KEYS = {"desc", "text", "type", "color_palette"}


def _slot_key(desc) -> str:
    return (desc or "").strip().lstrip("@")


def _to_desc(value, nested_format: str) -> str:
    """An Ideogram ``desc`` must be a string. Strings pass through; any nested
    structure is serialised so a placeholder box can be replaced by a whole
    sub-tree (e.g. ``{age_desc, gender, hair:{...}, body:{...}}``)."""
    if isinstance(value, str):
        return value
    return emit(value, nested_format)


def _apply_slot(el: dict, sub, nested_format: str) -> None:
    """Write one matched slot's content onto an element.

    - bare string  → ``desc``.
    - dict with reserved keys (``text``/``type``/``color_palette``) → mapped.
    - ``desc`` present → used verbatim (serialised if itself nested).
    - no ``desc`` → every non-reserved key is folded into ``desc`` as a
      serialised sub-tree (this is the nested-placeholder case).
    """
    if not isinstance(sub, dict):
        el["desc"] = _to_desc(sub, nested_format)
        return
    if "text" in sub:
        el["text"] = sub["text"]
        el["type"] = "text"
    if "type" in sub:
        el["type"] = sub["type"]
    if "color_palette" in sub and isinstance(sub["color_palette"], list):
        el["color_palette"] = [str(c).upper() for c in sub["color_palette"]][:5]
    if "desc" in sub:
        el["desc"] = _to_desc(sub["desc"], nested_format)
    else:
        content = {k: v for k, v in sub.items() if k not in _FIELDMAP_KEYS}
        if content:
            el["desc"] = emit(content, nested_format)
        # else: pure field-map without a desc → leave the existing desc.


def _apply_overrides(caption: dict, box_prompts: dict, nested_format: str,
                     scene_overrides: bool) -> dict:
    """Mutate ``caption`` in place; return a small report dict.

    By default **every** top-level key in ``box_prompts`` is a box placeholder
    (matched against the slot-key in each box's ``desc``); the scene fields
    come from the KJ node. With ``scene_overrides`` on, the three reserved
    keys instead set the caption's scene and are not used to fill boxes.
    """
    matched, unmatched_box = [], []
    used = set()
    cd = caption.setdefault("compositional_deconstruction", {})

    if scene_overrides:
        for rk in _RESERVED:
            if rk in box_prompts:
                used.add(rk)
                if rk == "background":
                    cd["background"] = _to_desc(box_prompts[rk], nested_format)
                else:
                    caption[rk] = box_prompts[rk]

    for el in cd.get("elements", []):
        if not isinstance(el, dict):
            continue
        key = _slot_key(el.get("desc"))
        if scene_overrides and key in _RESERVED:
            continue  # reserved for the scene in this mode — not a box slot
        if key in box_prompts:
            matched.append(key)
            used.add(key)
            _apply_slot(el, box_prompts[key], nested_format)
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
        "Each box is matched by the slot-key typed into its `desc`. A slot's\n"
        "value may be:\n"
        " - a string → the box's desc;\n"
        " - a field-map dict (desc/text/color_palette/type);\n"
        " - a NESTED structure → folded into desc (serialised), so a\n"
        "   placeholder box can be replaced by a whole sub-tree.\n"
        "bbox always stays from KJ. By default EVERY top-level key is a box\n"
        "placeholder (scene comes from the KJ node); set scene_overrides=on to\n"
        "instead let background / high_level_description / style_description\n"
        "set the scene. A wildcard pass expands tokens in every text field."
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
                    "tooltip": "Keyed JSON: {slot: <string | field-map | nested "
                               "structure>, ..., background:\"...\"}. Wire an "
                               "FVM_JB_Builder raw_json here."}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed for the wildcard/variable pass."}),
                "output_format": (("ideogram", "pretty_json", "compact_json"),
                                  {"default": "ideogram"}),
                "nested_desc_format": (("loose_keys", "compact_json", "pretty_json"),
                                       {"default": "loose_keys",
                                        "tooltip": "How a nested slot structure is "
                                                   "serialised into the box's desc."}),
                "scene_overrides": (("off (all top-level keys are box slots)",
                                     "on (background/high_level_description/style_description set the scene)"),
                                    {"default": "off (all top-level keys are box slots)",
                                     "tooltip": "off: every top-level key in box_prompts is a "
                                                "box placeholder; the scene comes from the KJ "
                                                "node. on: the three reserved keys set the scene "
                                                "and are not used to fill boxes."}),
                "on_unmatched_box": (("keep typed desc", "clear desc"),
                                     {"default": "keep typed desc"}),
            },
            "optional": {
                "context_from_prompt_generator": ("DICT", {
                    "tooltip": "Optional adaptiveprompts context for `__^var__` recall."}),
            },
        }

    def assemble(self, caption_json, box_prompts, seed, output_format,
                 nested_desc_format, scene_overrides, on_unmatched_box,
                 context_from_prompt_generator=None):
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

        bp_warn = ""
        bp = {}
        if isinstance(box_prompts, str) and box_prompts.strip():
            try:
                bp = json.loads(box_prompts)
            except (json.JSONDecodeError, TypeError):
                bp = parse_input(box_prompts)  # best-effort for loose forms
            if not isinstance(bp, dict):
                bp = {}
                bp_warn = ("WARN: box_prompts is not valid JSON — connect the JB "
                           "Builder's `raw_json` output (the `string`/loose_keys "
                           "output cannot be parsed back into a dict).")

        # Resolve wildcards BEFORE merging — and never after. We resolve the
        # caption (KJ-typed fields like background; slot-key descs carry no
        # tokens) and the slot content per-leaf separately, then fold. Running
        # a pass over the *already-merged* caption would re-interpret the
        # ``{``/``|`` of a serialised nested desc as bracket-wildcard syntax
        # and corrupt the JSON, so we don't.
        resolve_leaves(caption, seed, context_from_prompt_generator)
        resolve_leaves(bp, seed, context_from_prompt_generator)

        report = _apply_overrides(caption, bp, nested_desc_format,
                                  scene_overrides.startswith("on"))

        if on_unmatched_box == "clear desc":
            for el in caption.get("compositional_deconstruction", {}).get("elements", []):
                if isinstance(el, dict) and _slot_key(el.get("desc")) in report["unmatched_box"]:
                    el["desc"] = ""

        if output_format == "pretty_json":
            prompt = emit_strict_json(caption, indent=2)
        elif output_format == "compact_json":
            prompt = emit_strict_json(caption, indent=None)
        else:
            prompt = emit_ideogram(caption)
        raw_json = emit_strict_json(caption, indent=2)

        lines = []
        if bp_warn:
            lines.append(bp_warn)
        lines.append(f"box_prompts keys : {sorted(bp.keys())}")
        lines.append(f"matched boxes : {report['matched']}")
        lines.append(f"unmatched box : {report['unmatched_box']}")
        lines.append(f"unused slots  : {report['unused_slot']}")
        return (prompt, raw_json, "\n".join(lines))
