"""Tests for the Ideogram interop nodes + shared helpers.

Covers FVM_Ideogram_Assembler (slot match, field map, reserved overrides,
wildcard pass, key order, reports) and FVM_Ideogram_BoxJitter (default vs
override precedence, in-frame clamp, seed reproducibility), plus the shared
resolve_leaves and emit_ideogram helpers.
"""

import json

from nodes.jb.ideogram_assembler import FVM_Ideogram_Assembler
from nodes.jb.ideogram_jitter import FVM_Ideogram_BoxJitter, jitter_caption
from core.jb.resolve import resolve_leaves
from core.jb.serialize import emit_ideogram


def _caption(elements, background="scene", **top):
    cap = dict(top)
    cap["compositional_deconstruction"] = {"background": background, "elements": elements}
    return json.dumps(cap)


def _el(desc, bbox=(100, 100, 300, 300), type="obj", **extra):
    e = {"type": type, "bbox": list(bbox), "desc": desc}
    e.update(extra)
    return e


# ─── shared helpers ────────────────────────────────────────────────────


def test_resolve_leaves_leaves_numbers_and_hex_untouched():
    tree = {"bbox": [10, 20, 30, 40], "color": "#AABBCC", "desc": "{cat|cat}"}
    out = resolve_leaves(tree, 1)
    assert out["bbox"] == [10, 20, 30, 40]      # ints untouched
    assert out["color"] == "#AABBCC"            # no token → verbatim
    assert out["desc"] == "cat"                 # bracket resolved


def test_resolve_leaves_walks_lists():
    tree = {"elements": [{"desc": "{a|a}"}, {"desc": "{b|b}"}]}
    out = resolve_leaves(tree, 7)
    assert out["elements"][0]["desc"] == "a"
    assert out["elements"][1]["desc"] == "b"


def test_emit_ideogram_inlines_scalar_arrays():
    s = emit_ideogram({"bbox": [1, 2, 3, 4], "palette": ["#FF0000", "#00FF00"]})
    assert "[1, 2, 3, 4]" in s
    assert '["#FF0000", "#00FF00"]' in s
    assert "\n" in s  # object itself is multiline


def test_emit_ideogram_preserves_key_order():
    s = emit_ideogram({"z": 1, "a": 2, "m": 3})
    assert s.index('"z"') < s.index('"a"') < s.index('"m"')


# ─── Assembler ─────────────────────────────────────────────────────────


def _assemble(caption, box_prompts, seed=0, fmt="ideogram", nested="loose_keys",
              scene="off", unmatched="keep typed desc", ctx=None):
    scene_opt = ("on (background/high_level_description/style_description set the scene)"
                 if scene == "on" else "off (all top-level keys are box slots)")
    return FVM_Ideogram_Assembler().assemble(
        caption_json=caption, box_prompts=json.dumps(box_prompts), seed=seed,
        output_format=fmt, nested_desc_format=nested, scene_overrides=scene_opt,
        on_unmatched_box=unmatched, context_from_prompt_generator=ctx)


def test_assembler_metadata():
    assert FVM_Ideogram_Assembler.RETURN_NAMES == ("prompt", "raw_json", "report")
    assert FVM_Ideogram_Assembler.CATEGORY.startswith("FVM Tools/JB")


def test_assembler_slot_match_and_field_map():
    cap = _caption([_el("woman"), _el("logo", type="text", text="")])
    bp = {
        "woman": {"desc": "a tall woman", "color_palette": ["#c0392b", "#2c3e50"]},
        "logo": {"type": "text", "text": "ACME", "desc": "neon sign"},
    }
    prompt, raw, report = _assemble(cap, bp)
    cd = json.loads(raw)["compositional_deconstruction"]
    woman, logo = cd["elements"]
    assert woman["desc"] == "a tall woman"
    assert woman["color_palette"] == ["#C0392B", "#2C3E50"]   # uppercased
    assert woman["bbox"] == [100, 100, 300, 300]              # bbox untouched
    assert logo["text"] == "ACME" and logo["type"] == "text"
    assert logo["desc"] == "neon sign"
    assert "woman" in report and "logo" in report


def test_assembler_text_field_forces_text_type():
    cap = _caption([_el("title", type="obj")])
    prompt, raw, _ = _assemble(cap, {"title": {"text": "HELLO"}})
    el = json.loads(raw)["compositional_deconstruction"]["elements"][0]
    assert el["type"] == "text" and el["text"] == "HELLO"


def test_assembler_reserved_background_override_when_scene_on():
    cap = _caption([_el("x")], background="PLACEHOLDER")
    _, raw, _ = _assemble(cap, {"background": "rainy street", "x": {"desc": "y"}}, scene="on")
    assert json.loads(raw)["compositional_deconstruction"]["background"] == "rainy street"


def test_assembler_default_all_top_level_are_box_slots():
    # Default (scene off): 'background' is NOT a scene override — it's a plain
    # box slot. The scene background stays from KJ; a box named 'background' fills.
    cap = _caption([_el("background"), _el("x")], background="KJ_SCENE")
    _, raw, rep = _assemble(cap, {"background": {"desc": "a filled box"}, "x": {"desc": "y"}})
    cd = json.loads(raw)["compositional_deconstruction"]
    assert cd["background"] == "KJ_SCENE"                       # scene untouched
    assert cd["elements"][0]["desc"] == "a filled box"         # box 'background' filled
    assert "background" in rep and "x" in rep                  # both matched


def test_assembler_scene_on_reserved_not_used_as_box():
    # With scene on, a box named 'background' is left alone (reserved for scene).
    cap = _caption([_el("background")], background="OLD")
    _, raw, _ = _assemble(cap, {"background": "new scene"}, scene="on")
    cd = json.loads(raw)["compositional_deconstruction"]
    assert cd["background"] == "new scene"
    assert cd["elements"][0]["desc"] == "background"           # box untouched


def test_assembler_color_palette_capped_at_five():
    cap = _caption([_el("p")])
    pal = [f"#{i:02x}0000" for i in range(8)]
    _, raw, _ = _assemble(cap, {"p": {"color_palette": pal}})
    assert len(json.loads(raw)["compositional_deconstruction"]["elements"][0]["color_palette"]) == 5


def test_assembler_wildcard_pass_runs():
    cap = _caption([_el("hero")])
    _, raw, _ = _assemble(cap, {"hero": {"desc": "a {red|red} coat"}}, seed=3)
    assert json.loads(raw)["compositional_deconstruction"]["elements"][0]["desc"] == "a red coat"


def test_assembler_unmatched_box_keep_vs_clear():
    cap = _caption([_el("ghost")])
    _, raw_keep, rep = _assemble(cap, {"other": {"desc": "z"}}, unmatched="keep typed desc")
    assert json.loads(raw_keep)["compositional_deconstruction"]["elements"][0]["desc"] == "ghost"
    assert "ghost" in rep and "other" in rep  # unmatched box + unused slot reported

    _, raw_clear, _ = _assemble(cap, {"other": {"desc": "z"}}, unmatched="clear desc")
    assert json.loads(raw_clear)["compositional_deconstruction"]["elements"][0]["desc"] == ""


def test_assembler_at_prefix_tolerated():
    cap = _caption([_el("@woman")])
    _, raw, _ = _assemble(cap, {"woman": {"desc": "ok"}})
    assert json.loads(raw)["compositional_deconstruction"]["elements"][0]["desc"] == "ok"


def test_assembler_bare_string_slot_sets_desc():
    cap = _caption([_el("woman")])
    _, raw, _ = _assemble(cap, {"woman": "a tall woman in red"})
    assert json.loads(raw)["compositional_deconstruction"]["elements"][0]["desc"] == "a tall woman in red"


def test_assembler_nested_structure_folds_into_desc():
    # The user's case: a placeholder replaced by a whole sub-tree (no `desc`
    # key) → the entire structure is serialised into the box's desc.
    cap = _caption([_el("woman")])
    woman = {
        "age_desc": "48yo", "gender": "female",
        "hair": {"color": "fiery red", "length": "over-shoulder"},
        "body": {"build": "fit", "height": "5ft5in"},
    }
    _, raw, _ = _assemble(cap, {"woman": woman}, nested="compact_json")
    desc = json.loads(raw)["compositional_deconstruction"]["elements"][0]["desc"]
    # compact_json → the desc is itself a JSON string carrying the sub-tree
    inner = json.loads(desc)
    assert inner["gender"] == "female"
    assert inner["hair"]["color"] == "fiery red"
    assert inner["body"]["height"] == "5ft5in"


def test_assembler_nested_loose_keys_default():
    cap = _caption([_el("woman")])
    _, raw, _ = _assemble(cap, {"woman": {"gender": "female", "hair": {"color": "red"}}})
    desc = json.loads(raw)["compositional_deconstruction"]["elements"][0]["desc"]
    assert "gender: female" in desc and "color: red" in desc  # loose_keys, no quotes


def test_assembler_nested_keeps_reserved_fields_alongside():
    # color_palette is reserved → pulled onto the element; the rest folds to desc.
    cap = _caption([_el("woman")])
    sub = {"gender": "female", "color_palette": ["#abcdef"]}
    _, raw, _ = _assemble(cap, {"woman": sub})
    el = json.loads(raw)["compositional_deconstruction"]["elements"][0]
    assert el["color_palette"] == ["#ABCDEF"]
    assert "gender: female" in el["desc"]
    assert "color_palette" not in el["desc"]  # reserved key not folded into desc


def test_assembler_explicit_desc_wins_over_siblings():
    cap = _caption([_el("woman")])
    _, raw, _ = _assemble(cap, {"woman": {"desc": "explicit", "gender": "female"}})
    assert json.loads(raw)["compositional_deconstruction"]["elements"][0]["desc"] == "explicit"


def test_assembler_nested_wildcards_resolve_per_leaf():
    cap = _caption([_el("woman")])
    sub = {"hair": {"color": "{red|red} hair"}, "mood": "{calm|calm}"}
    _, raw, _ = _assemble(cap, {"woman": sub}, seed=2)
    desc = json.loads(raw)["compositional_deconstruction"]["elements"][0]["desc"]
    # values resolved per-leaf; wildcard syntax ('|', '{red') gone — structural
    # loose_keys object braces are expected to remain.
    assert "red hair" in desc and "calm" in desc
    assert "|" not in desc and "{red" not in desc


def test_assembler_invalid_caption_passthrough():
    prompt, raw, report = _assemble("{not json", {"a": 1})
    assert prompt == "{not json" and "ERROR" in report


def test_assembler_loose_keys_box_prompts_warns():
    # Wiring the Builder's loose-keys `string` (not raw_json) → unparseable;
    # the box keeps its typed desc and the report flags the mistake loudly.
    cap = _caption([_el("woman")])
    loose = "{\n  woman: {\n    desc: a tall woman\n  }\n}"
    prompt, raw, report = FVM_Ideogram_Assembler().assemble(
        caption_json=cap, box_prompts=loose, seed=0, output_format="ideogram",
        nested_desc_format="loose_keys",
        scene_overrides="off (all top-level keys are box slots)",
        on_unmatched_box="keep typed desc")
    assert "WARN" in report and "raw_json" in report
    assert json.loads(raw)["compositional_deconstruction"]["elements"][0]["desc"] == "woman"


def test_assembler_report_lists_box_prompt_keys():
    cap = _caption([_el("a")])
    _, _, report = _assemble(cap, {"a": {"desc": "x"}, "b": {"desc": "y"}})
    assert "box_prompts keys" in report and "'a'" in report and "'b'" in report


def test_assembler_ideogram_format_inlines_bbox():
    cap = _caption([_el("a")])
    prompt, _, _ = _assemble(cap, {"a": {"desc": "x"}}, fmt="ideogram")
    assert "[100, 100, 300, 300]" in prompt


# ─── BoxJitter ─────────────────────────────────────────────────────────


def _rules(default=None, overrides=None):
    d = {"pos": 0.06, "size": 0.12, "aspect": 0.08, "min": 0.03}
    if default:
        d.update(default)
    return {"default": d, "overrides": overrides or []}


def _jit(caption, seed, rules):
    out, = FVM_Ideogram_BoxJitter().jitter(
        jitter_rules=json.dumps(rules), caption_json=caption, seed=seed)
    return json.loads(out)["compositional_deconstruction"]["elements"]


def test_jitter_metadata():
    assert FVM_Ideogram_BoxJitter.RETURN_NAMES == ("caption_json",)


def test_jitter_stays_inside_frame():
    cap = _caption([_el("a", (0, 0, 1000, 1000)), _el("b", (900, 900, 1000, 1000)),
                    _el("c", (500, 480, 520, 540))])
    for seed in range(20):
        for el in _jit(cap, seed, _rules({"pos": 0.2, "size": 0.3, "aspect": 0.2})):
            y0, x0, y1, x1 = el["bbox"]
            assert 0 <= y0 < y1 <= 1000
            assert 0 <= x0 < x1 <= 1000


def test_jitter_seed_reproducible():
    cap = _caption([_el("a"), _el("b")])
    assert _jit(cap, 42, _rules()) == _jit(cap, 42, _rules())


def test_jitter_zero_rule_keeps_box_still():
    cap = _caption([_el("frozen", (100, 100, 300, 300))])
    rules = _rules(overrides=[{"boxes": "frozen", "pos": 0, "size": 0, "aspect": 0}])
    el = _jit(cap, 5, rules)[0]
    assert el["bbox"] == [100, 100, 300, 300]


def test_jitter_override_precedence():
    # default would move 'still' too; an override pins it while others move.
    cap = _caption([_el("still", (100, 100, 300, 300)), _el("mover", (100, 100, 300, 300))])
    rules = _rules({"pos": 0.15, "size": 0.2, "aspect": 0.1},
                   overrides=[{"boxes": "still", "pos": 0, "size": 0, "aspect": 0}])
    els = _jit(cap, 9, rules)
    assert els[0]["bbox"] == [100, 100, 300, 300]   # pinned by override
    assert els[1]["bbox"] != [100, 100, 300, 300]   # moved by default


def test_jitter_multi_box_override():
    cap = _caption([_el("a", (100, 100, 200, 200)), _el("b", (100, 100, 200, 200))])
    rules = _rules({"pos": 0.2, "size": 0.2, "aspect": 0.2},
                   overrides=[{"boxes": "a, b", "pos": 0, "size": 0, "aspect": 0}])
    for el in _jit(cap, 3, rules):
        assert el["bbox"] == [100, 100, 200, 200]


def test_jitter_min_floor_prevents_collapse():
    cap = _caption([_el("tiny", (500, 500, 505, 505))])
    rules = _rules({"size": 0.9, "aspect": 0.9, "min": 0.05})  # min edge 50
    el = _jit(cap, 1, rules)[0]
    y0, x0, y1, x1 = el["bbox"]
    assert (y1 - y0) >= 49 and (x1 - x0) >= 49


def test_jitter_skips_non_bbox_elements():
    cap = _caption([{"type": "obj", "desc": "nobbox"}])
    out, = FVM_Ideogram_BoxJitter().jitter(
        jitter_rules=json.dumps(_rules()), caption_json=cap, seed=0)
    assert json.loads(out)["compositional_deconstruction"]["elements"][0] == {"type": "obj", "desc": "nobbox"}


def test_jitter_invalid_caption_passthrough():
    out, = FVM_Ideogram_BoxJitter().jitter(
        jitter_rules=json.dumps(_rules()), caption_json="{bad", seed=0)
    assert out == "{bad"
