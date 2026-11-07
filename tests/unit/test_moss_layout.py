"""MOSS layout from SAM3 boxes + JB prompt (core and node, SAM3 mocked)."""

import json

import numpy as np
import torch

from core.jb.moss_layout import (
    auto_rule, build_layout, centered_yxyx, fit_canvas, make_boxes_doc, merge_rules,
    number_slots, parse_box_rules, parse_vocabulary, rules_text, scale_canvas, select_hits,
    slot_name, split_prompt, to_ideogram,
)
import nodes.jb.moss_layout as node_mod
from nodes.jb.moss_layout import FVM_MOSS_SAM3Layout


def test_parse_box_rules_defaults_and_comments():
    rules = parse_box_rules(
        "# comment\nperson | character | character_{n}\ntable\nsign | text | s  # tail\n"
        "cup | weird | cup\n")
    assert rules == [
        {"prompt": "person", "label": "character", "slot": "character_{n}", "caption": ""},
        {"prompt": "table", "label": "object", "slot": "", "caption": ""},
        {"prompt": "sign", "label": "text", "slot": "s", "caption": ""},
        {"prompt": "cup", "label": "object", "slot": "cup", "caption": ""},
    ]


def test_select_hits_nms_floor_and_cap():
    hits = [
        {"bbox": [0, 0, 100, 100], "score": 0.9},
        {"bbox": [5, 5, 100, 100], "score": 0.8},      # duplicate of the first
        {"bbox": [200, 0, 300, 100], "score": 0.7},
        {"bbox": [400, 0, 402, 2], "score": 0.95},     # below min_area
        {"bbox": [500, 0, 600, 100], "score": 0.6},
    ]
    kept = select_hits(hits, max_hits=2, min_area=0.001, src_w=1000, src_h=1000)
    assert [h["score"] for h in kept] == [0.9, 0.7]


def test_number_slots_left_to_right():
    rule = {"prompt": "person", "label": "character", "slot": "character_{n}"}
    hits = [{"bbox": [600, 0, 800, 100], "score": 0.9},
            {"bbox": [0, 0, 200, 100], "score": 0.5}]
    out = number_slots(rule, hits)
    assert [(b["slot"], b["bbox"][0]) for b in out] == [("character_1", 0), ("character_2", 600)]


def test_fit_canvas_snaps_to_16_and_keeps_aspect():
    assert fit_canvas(3000, 3000, 1.0) == (1024, 1024)
    w, h = fit_canvas(832, 1216, 1.0)
    assert w % 16 == 0 and h % 16 == 0 and abs(w / h - 832 / 1216) < 0.02
    assert fit_canvas(1000, 600, 0) == (992, 592)
    assert scale_canvas(1024, 1024, 1.5) == (1536, 1536)
    assert scale_canvas(848, 1232, 1.5) == (1264, 1840)


def test_centered_yxyx_matches_moss_rule():
    # Full canvas on a 1024 canvas from a 512 source: [-512, -512, 512, 512].
    assert centered_yxyx([0, 0, 512, 512], 512, 512, 1024, 1024) == [-512, -512, 512, 512]
    assert centered_yxyx([10.4, 20.6, 30.2, 40.1], 100, 100, 100, 100) == [-30, -40, -9, -19]


def test_split_prompt_unwraps_stitcher_title():
    doc = {"scene": {"character_1": {"hair": "red"}, "location": {"place": "office"}}}
    claimed, rest = split_prompt(json.dumps(doc), {"character_1"})
    assert claimed == {"character_1": {"hair": "red"}}
    assert rest == {"location": {"place": "office"}}


def test_split_prompt_prose_is_all_summary():
    claimed, rest = split_prompt("A woman in an office.", {"character_1"})
    assert claimed == {} and rest == "A woman in an office."


def test_build_layout_fills_boxes_and_summary():
    boxes = [
        {"bbox": [0, 0, 50, 100], "label": "character", "slot": "character_1", "prompt": "person"},
        {"bbox": [50, 50, 100, 100], "label": "object", "slot": "table", "prompt": "table"},
    ]
    prompt = json.dumps({"scene": {"character_1": {"hair": "long red hair"},
                                   "location": {"background": "a quiet office"}}})
    layout, report = build_layout(boxes, 100, 100, 1024, 1024, prompt=prompt,
                                  high_level_description="Candid photo.", meta="photo, candid")
    assert layout["bbox_coord_space"] == "abs_centralized" and layout["bbox_format"] == "yxyx"
    assert layout["summary"]["meta"] == ["photo", "candid"]
    assert layout["summary"]["caption"].startswith("Candid photo.")
    assert "quiet office" in layout["summary"]["caption"]
    person, table = layout["items"]
    assert person["label"] == "character" and "long red hair" in person["caption"]
    assert table["caption"] == "table"                 # no slot in the prompt → SAM3 prompt
    assert person["bbox"] == [-512, -512, 512, 0]
    assert report == {"filled": ["character_1"], "fallback": ["table"],
                      "summary_keys": ["location"]}


def test_node_end_to_end_with_mocked_sam3(monkeypatch):
    monkeypatch.setattr(node_mod, "sam3_prepare", lambda cfg, img: ("proc", "state"))
    found = {
        "person": [(None, 0.9, [600, 100, 900, 1000]), (None, 0.8, [100, 100, 400, 1000])],
        "table": [],
    }
    monkeypatch.setattr(node_mod, "sam3_ground",
                        lambda p, s, shape, prompt, threshold: found[prompt])
    image = torch.rand(1, 1000, 1000, 3)
    prompt = json.dumps({"scene": {"character_1": {"hair": "black bob"},
                                   "character_2": {"hair": "grey curls"},
                                   "location": {"background": "a diner"}}})
    out = FVM_MOSS_SAM3Layout().build(
        sam3_model=object(), image=image,
        box_rules="person | character | character_{n}\ntable | object | table",
        threshold=0.4, max_per_rule=3, min_area=0.01, megapixels=1.0, upscale_by=1.5,
        high_level_description="", meta="photo", canvas="single", prompt=prompt)
    moss, moss_up, w, h, uw, uh, preview, report, doc, suggested, ideogram = out["result"]
    assert (w, h, uw, uh) == (1024, 1024, 1536, 1536)
    layout, layout_up = json.loads(moss), json.loads(moss_up)
    left, right = layout["items"]
    assert "black bob" in left["caption"] and "grey curls" in right["caption"]
    assert left["bbox"][1] < right["bbox"][1]
    assert all(abs(a - b * 1.5) <= 2 for a, b in zip(layout_up["items"][0]["bbox"], left["bbox"]))
    assert "a diner" in layout["summary"]["caption"]
    assert preview.shape == (1, 1024, 1024, 3)
    assert "'table': 0 found, 0 kept" in report


def test_node_reports_sam3_failure(monkeypatch):
    monkeypatch.setattr(node_mod, "sam3_prepare", lambda cfg, img: (None, None))
    monkeypatch.setattr(node_mod, "sam3_last_error", lambda: "no checkpoint")
    out = FVM_MOSS_SAM3Layout().build(
        sam3_model=object(), image=torch.rand(1, 64, 64, 3), box_rules="person",
        threshold=0.4, max_per_rule=3, min_area=0.0, megapixels=0.0, upscale_by=1.5,
        high_level_description="Scene.", meta="", canvas="single")["result"]
    assert "ERROR: SAM3 did not load — no checkpoint" in out[7]
    assert json.loads(out[0])["items"] == []


def test_node_wraps_native_sam3(monkeypatch):
    seen = {}

    def prepare(cfg, img):
        seen["cfg"] = cfg
        return None, None

    monkeypatch.setattr(node_mod, "sam3_prepare", prepare)
    monkeypatch.setattr(node_mod, "sam3_last_error", lambda: "x")
    model, clip = object(), object()
    FVM_MOSS_SAM3Layout().build(
        image=torch.rand(1, 64, 64, 3), box_rules="person", threshold=0.4, max_per_rule=3,
        min_area=0.0, megapixels=0.0, upscale_by=1.5, high_level_description="", meta="",
        canvas="single", sam3_native_model=model, sam3_native_clip=clip, sam3_model=object())
    assert isinstance(seen["cfg"], node_mod.NativeSAM3)
    assert seen["cfg"].model is model and seen["cfg"].clip is clip


def test_node_without_sam3_says_so():
    out = FVM_MOSS_SAM3Layout().build(
        image=torch.rand(1, 64, 64, 3), box_rules="person", threshold=0.4, max_per_rule=3,
        min_area=0.0, megapixels=0.0, upscale_by=1.5, high_level_description="Scene.",
        meta="", canvas="single")["result"]
    assert "ERROR: no SAM3 wired" in out[7]
    assert json.loads(out[0])["summary"]["caption"] == "Scene."


def test_parse_box_rules_caption_column():
    rules = parse_box_rules("plant | object | plant_{n} | a tall fern | in a pot\n"
                            "lamp | object | | a brass lamp\n")
    assert rules[0]["caption"] == "a tall fern | in a pot"
    assert rules[1] == {"prompt": "lamp", "label": "object", "slot": "", "caption": "a brass lamp"}


def test_rule_caption_wins_over_slot():
    rule = {"prompt": "person", "label": "character", "slot": "character_{n}",
            "caption": "man number {n} in a navy sweater"}
    boxes = number_slots(rule, [{"bbox": [0, 0, 50, 100], "score": 0.9}])
    assert boxes[0]["caption"] == "man number 1 in a navy sweater"
    prompt = json.dumps({"character_1": {"hair": "red"}})
    doc, report = make_boxes_doc(boxes, 100, 100, 1024, 1024, prompt=prompt)
    assert doc["boxes"][0]["caption"] == "man number 1 in a navy sweater"
    assert doc["boxes"][0]["slot"] == "character_1"
    assert report["filled"] == ["character_1"]
    # A rule without caption leaves the box uncaptioned (the JB slot fills it).
    plain = number_slots({**rule, "caption": ""}, [{"bbox": [0, 0, 50, 100], "score": 0.9}])
    assert "caption" not in plain[0]


def test_vocabulary_and_auto_rules():
    assert parse_vocabulary("person, table\n# c\nTable\n  flower   vase , \n") == \
        ["person", "table", "flower vase"]
    assert slot_name("Flower Vase") == "flower_vase"
    assert slot_name("t.v.") == "t_v"
    assert auto_rule("person") == {"prompt": "person", "label": "character",
                                   "slot": "character_{n}", "caption": ""}
    assert auto_rule("Flower Vase")["slot"] == "flower_vase_{n}"
    manual = parse_box_rules("Person | character | hero\n")
    rules, extra = merge_rules(manual, ["person", "plant"])
    assert extra == ["plant"]
    assert [r["slot"] for r in rules] == ["hero", "plant_{n}"]


def test_rules_text_round_trips():
    text = "person | character | character_{n}\nlamp | object |  | a brass lamp\n"
    rules = parse_box_rules(text)
    assert parse_box_rules(rules_text(rules)) == rules
    assert rules_text([auto_rule("plant")]) == "plant | object | plant_{n}\n"
    assert rules_text([]) == ""


def test_to_ideogram_boxes_and_desc():
    doc = {"summary": {"caption": "Candid photo. A quiet office."},
           "boxes": [
               {"slot": "character_1", "label": "character", "caption": "red hair",
                "prompt": "person", "rect": [0.0, 0.1, 0.5, 1.0]},
               {"slot": "", "label": "object", "caption": "a brass lamp", "prompt": "lamp",
                "rect": [0.6, 0.2, 0.8, 0.4]},
               {"slot": "sign_1", "label": "text", "caption": "sign", "text": "OPEN",
                "prompt": "sign", "rect": [0.12345, 0.0, 0.5, 0.05]},
           ]}
    cap = to_ideogram(doc, "Candid photo.")
    assert cap["high_level_description"] == "Candid photo."
    cd = cap["compositional_deconstruction"]
    assert cd["background"] == "A quiet office."
    person, lamp, sign = cd["elements"]
    assert person == {"type": "obj", "bbox": [100, 0, 1000, 500], "desc": "character_1"}
    assert lamp == {"type": "obj", "bbox": [200, 600, 400, 800], "desc": "a brass lamp"}
    assert sign == {"type": "text", "bbox": [0, 123, 50, 500], "text": "OPEN", "desc": "sign_1"}
    assert list(sign) == ["type", "bbox", "text", "desc"]          # KJ's key order
    # Without a high-level description the summary caption serves both fields.
    plain = to_ideogram(doc)
    assert plain["high_level_description"] == plain["compositional_deconstruction"]["background"]
    # A character without slot keeps its label through the caption prefix.
    doc["boxes"][0]["slot"] = ""
    el = to_ideogram(doc)["compositional_deconstruction"]["elements"][0]
    assert el["desc"] == "character: red hair"


def _mock_sam3(monkeypatch, found):
    calls = []

    def ground(p, s, shape, prompt, threshold):
        calls.append(prompt)
        return found.get(prompt, [])

    monkeypatch.setattr(node_mod, "sam3_prepare", lambda cfg, img: ("proc", "state"))
    monkeypatch.setattr(node_mod, "sam3_ground", ground)
    return calls


_ARGS = dict(threshold=0.4, max_per_rule=3, min_area=0.01, megapixels=1.0, upscale_by=1.5,
             high_level_description="", meta="photo", canvas="single")


def test_node_manual_mode_skips_vocabulary(monkeypatch):
    calls = _mock_sam3(monkeypatch, {"person": [(None, 0.9, [100, 100, 400, 1000])]})
    out = FVM_MOSS_SAM3Layout().build(
        sam3_model=object(), image=torch.rand(1, 1000, 1000, 3), box_rules=node_mod.DEFAULT_RULES,
        rules_mode="manual", vocabulary="person, plant, lamp", **_ARGS)
    assert calls == ["person"]
    assert out["ui"]["suggested_rules"] == [node_mod.DEFAULT_RULES]
    assert out["result"][9] == node_mod.DEFAULT_RULES
    assert len(out["result"]) == len(FVM_MOSS_SAM3Layout.RETURN_TYPES)


def test_node_auto_mode_suggests_rules_from_hits(monkeypatch):
    calls = _mock_sam3(monkeypatch, {
        "person": [(None, 0.9, [600, 100, 900, 1000]), (None, 0.8, [100, 100, 400, 1000])],
        "flower vase": [(None, 0.7, [450, 600, 550, 800])],
        "lamp": [(None, 0.6, [0, 0, 5, 5])],                  # below min_area → no hit
    })
    out = FVM_MOSS_SAM3Layout().build(
        sam3_model=object(), image=torch.rand(1, 1000, 1000, 3),
        box_rules="table | object | table", rules_mode="auto",
        vocabulary="person\nflower vase\nlamp\ndog", **_ARGS)
    assert calls == ["person", "flower vase", "lamp", "dog"]   # manual rules ignored
    report, doc, suggested, ideogram = out["result"][7:]
    assert suggested == "person | character | character_{n}\nflower vase | object | flower_vase_{n}\n"
    assert out["ui"]["suggested_rules"] == [suggested]
    assert "vocabulary: 2/4 words hit — person (2), flower vase (1)" in report
    assert [b["slot"] for b in doc["boxes"]] == ["character_1", "character_2", "flower_vase_1"]
    cap = json.loads(ideogram)
    elements = cap["compositional_deconstruction"]["elements"]
    assert [e["desc"] for e in elements] == ["character_1", "character_2", "flower_vase_1"]
    assert elements[0]["bbox"] == [100, 100, 1000, 400]


def test_node_manual_plus_auto(monkeypatch):
    calls = _mock_sam3(monkeypatch, {
        "person": [(None, 0.9, [100, 100, 400, 1000])],
        "plant": [(None, 0.7, [600, 500, 800, 900])],
    })
    rules = "# mine\nperson | character | hero | a tall man in a navy sweater\n"
    out = FVM_MOSS_SAM3Layout().build(
        sam3_model=object(), image=torch.rand(1, 1000, 1000, 3), box_rules=rules,
        rules_mode="manual + auto", vocabulary="person, plant, cup", **_ARGS)
    assert calls == ["person", "plant", "cup"]                 # person covered by the manual rule
    report, doc, suggested, _ideo = out["result"][7:]
    assert suggested == rules.rstrip() + "\nplant | object | plant_{n}\n"
    hero, plant = doc["boxes"]
    assert hero["slot"] == "hero" and hero["caption"] == "a tall man in a navy sweater"
    assert plant["slot"] == "plant_1" and plant["caption"] == "plant"
    assert "'person': 1 found, 1 kept" in report and "1/2 words hit — plant (1)" in report


def test_node_auto_without_sam3_suggests_nothing():
    out = FVM_MOSS_SAM3Layout().build(
        image=torch.rand(1, 64, 64, 3), box_rules="person", rules_mode="auto",
        vocabulary="person", **_ARGS)
    assert "ERROR: no SAM3 wired" in out["result"][7]
    assert out["result"][9] == "" and out["ui"]["suggested_rules"] == [""]


def test_pick_auto_hits_one_box_per_thing_and_cap():
    from core.jb.moss_layout import pick_auto_hits
    stool = {"prompt": "stool"}, [{"bbox": [0, 0, 10, 10], "score": 0.7}]
    chair = {"prompt": "chair"}, [{"bbox": [0, 0, 10, 11], "score": 0.9},
                                  {"bbox": [50, 50, 60, 60], "score": 0.65}]
    lamp = {"prompt": "lamp"}, [{"bbox": [20, 0, 30, 10], "score": 0.8}]
    person = [{"bbox": [20, 0, 30, 10], "score": 0.99}]          # manual hit covers the lamp
    out = pick_auto_hits([stool, chair, lamp], person, max_boxes=8)
    assert [(r["prompt"], [h["score"] for h in hits]) for r, hits in out] == [("chair", [0.9, 0.65])]
    capped = pick_auto_hits([stool, chair, lamp], [], max_boxes=2)
    assert [(r["prompt"], len(h)) for r, h in capped] == [("chair", 1), ("lamp", 1)]
