"""MOSS layout from SAM3 boxes + JB prompt (core and node, SAM3 mocked)."""

import json

import numpy as np
import torch

from core.jb.moss_layout import (
    build_layout, centered_yxyx, fit_canvas, number_slots, parse_box_rules,
    scale_canvas, select_hits, split_prompt,
)
import nodes.jb.moss_layout as node_mod
from nodes.jb.moss_layout import FVM_MOSS_SAM3Layout


def test_parse_box_rules_defaults_and_comments():
    rules = parse_box_rules(
        "# comment\nperson | character | character_{n}\ntable\nsign | text | s  # tail\n"
        "cup | weird | cup\n")
    assert rules == [
        {"prompt": "person", "label": "character", "slot": "character_{n}"},
        {"prompt": "table", "label": "object", "slot": ""},
        {"prompt": "sign", "label": "text", "slot": "s"},
        {"prompt": "cup", "label": "object", "slot": "cup"},
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
    moss, moss_up, w, h, uw, uh, preview, report = out
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
        high_level_description="Scene.", meta="", canvas="single")
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
        meta="", canvas="single")
    assert "ERROR: no SAM3 wired" in out[7]
    assert json.loads(out[0])["summary"]["caption"] == "Scene."
