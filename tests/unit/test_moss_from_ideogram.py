"""MOSS layout from a KJ Ideogram-4 caption (+ JB prompt), core and node."""

import json

import torch

from core.jb.moss_ideogram import (
    desc_key, element_rect, ideogram_doc, join_unique, parse_caption,
)
from core.jb.moss_layout import centered_yxyx, make_boxes_doc
from nodes.jb.moss_from_ideogram import FVM_MOSS_FromIdeogram


# What KJNodes' Ideogram 4 Prompt Builder emits (normalized 0-1000, yx).
KJ_CAPTION = {
    "high_level_description": "A candid evening photo in a cafe.",
    "style_description": {"aesthetics": "warm", "lighting": "tungsten", "photo": "35mm",
                          "medium": "photograph"},
    "compositional_deconstruction": {
        "background": "a warm cafe interior with brick walls",
        "elements": [
            {"type": "obj", "bbox": [100, 50, 950, 480], "desc": "character_1"},
            {"type": "obj", "bbox": [120, 520, 940, 960],
             "desc": "character: a tall man in a grey suit"},
            {"type": "obj", "bbox": [700, 400, 900, 600],
             "desc": "a small round coffee table with two cups"},
            {"type": "text", "bbox": [20, 300, 90, 700], "text": "OPEN",
             "desc": "red neon sign above the door", "color_palette": ["#FF2020"]},
            {"type": "obj", "desc": "a sleeping cat"},
        ],
    },
}

JB_PROMPT = json.dumps({"scene": {
    "character_1": {"hair": "long red hair", "outfit": "green summer dress"},
    "location": {"place": "a busy corner cafe"},
}})


def _build(caption, prompt="", **kw):
    args = dict(caption_json=json.dumps(caption), width=832, height=1216, upscale_by=1.5,
                high_level_description="", meta="photo", canvas="single", prompt=prompt)
    args.update(kw)
    return FVM_MOSS_FromIdeogram().build(**args)


def test_parse_caption_strict_loose_and_bad():
    assert parse_caption('{"a": 1}') == ({"a": 1}, "")
    assert parse_caption("{a: 1}")[0] == {"a": 1}
    assert parse_caption("")[0] is None
    assert parse_caption("[1, 2]")[0] is None


def test_desc_key_tolerates_at():
    assert desc_key(" @character_1 ") == "character_1"
    assert desc_key(None) == ""


def test_element_rect_spaces():
    assert element_rect([100, 50, 950, 480]) == [0.05, 0.1, 0.48, 0.95]
    assert element_rect([50, 100, 480, 950], "normalized xy") == [0.05, 0.1, 0.48, 0.95]
    assert element_rect([0, 0, 1024, 256], "pixels yx", 512, 1024) == [0.0, 0.0, 0.5, 1.0]
    assert element_rect([0, 0, 256, 512], "pixels xy", 512, 1024) == [0.0, 0.0, 0.5, 0.5]
    assert element_rect([-50, 900, 400, 1200]) == [0.9, 0.0, 1.0, 0.4]       # clamped
    assert element_rect([10, 10, 10, 50]) is None                             # empty
    assert element_rect([1, 2, 3]) is None and element_rect(["a", 0, 1, 1]) is None


def test_realistic_kj_caption_with_jb_prompt():
    moss, moss_up, w, h, uw, uh, preview, report, doc = _build(KJ_CAPTION, JB_PROMPT)
    assert (w, h, uw, uh) == (832, 1216, 1248, 1824)
    layout = json.loads(moss)
    assert layout["bbox_coord_space"] == "abs_centralized" and layout["bbox_format"] == "yxyx"

    labels = [(it["label"], it["caption"][:20]) for it in layout["items"]]
    assert [l for l, _ in labels] == ["character", "character", "object", "text"]
    jb, man, table, sign = layout["items"]
    # (a) slot key → JB sub-tree
    assert "long red hair" in jb["caption"] and "green summer dress" in jb["caption"]
    # (b) character: prefix
    assert man["caption"] == "a tall man in a grey suit"
    # (c) free text
    assert table["caption"] == "a small round coffee table with two cups"
    # text element keeps its text
    assert sign["text"] == "OPEN" and sign["caption"] == "red neon sign above the door"
    # bbox: normalized yx → MOSS centred pixels on 832x1216
    assert jb["bbox"] == [-487, -375, 548, -16]
    assert jb["bbox"] == centered_yxyx([0.05, 0.1, 0.48, 0.95], 1, 1, 832, 1216)

    # summary: own high-level description, background once, unclaimed JB keys
    cap = layout["summary"]["caption"]
    assert cap.startswith("A candid evening photo in a cafe.")
    assert cap.count("brick walls") == 1
    assert "busy corner cafe" in cap and "long red hair" not in cap
    assert layout["summary"]["meta"] == ["photo"]

    # report: JB-filled vs own text vs skipped
    assert "filled from prompt : ['#1 character_1']" in report
    assert "#2 character: a tall man" in report
    assert "#5 a sleeping cat: no bbox (unplaced)" in report
    assert "scene caption keys : ['location']" in report

    # MOSS_BOXES
    assert doc["source"] == "ideogram"
    assert all(b["source"] == "ideogram" for b in doc["boxes"])
    assert doc["boxes"][0]["slot"] == "character_1"
    assert doc["boxes"][0]["rect"] == [0.05, 0.1, 0.48, 0.95]
    assert (doc["width"], doc["height"], doc["up_width"], doc["up_height"]) == (832, 1216, 1248, 1824)
    assert preview.shape == (1, 1216, 832, 3)


def test_upscaled_prompt_scales_every_bbox():
    moss, moss_up, *_ = _build(KJ_CAPTION, JB_PROMPT)
    items, items_up = json.loads(moss)["items"], json.loads(moss_up)["items"]
    assert len(items) == len(items_up) == 4
    for a, b in zip(items, items_up):
        assert a["caption"] == b["caption"]
        assert all(abs(u - v * 1.5) <= 2 for u, v in zip(b["bbox"], a["bbox"]))
    assert items_up[0]["bbox"] == centered_yxyx([0.05, 0.1, 0.48, 0.95], 1, 1, 1248, 1824)


def test_without_prompt_keys_stay_slots_and_captions():
    cap = {"compositional_deconstruction": {"background": "a beach", "elements": [
        {"type": "obj", "bbox": [0, 0, 1000, 500], "desc": "@character_2"},
        {"type": "obj", "bbox": [0, 500, 1000, 1000], "desc": "lamp"},
    ]}}
    *_, report, doc = _build(cap, "", width=1024, height=1024)
    char, lamp = doc["boxes"]
    assert (char["slot"], char["label"], char["caption"]) == ("character_2", "character", "character_2")
    assert (lamp["slot"], lamp["label"], lamp["caption"]) == ("lamp", "object", "lamp")
    assert doc["summary"]["caption"] == "a beach"
    assert "filled from prompt : []" in report


def test_key_missing_from_prompt_is_reported():
    cap = {"compositional_deconstruction": {"background": "", "elements": [
        {"type": "obj", "bbox": [0, 0, 500, 500], "desc": "character_3"}]}}
    *_, report, doc = _build(cap, JB_PROMPT)
    assert "key not in prompt  : ['#1 character_3']" in report
    assert doc["boxes"][0]["caption"] == "character_3"
    # nothing claimed → the whole JB scene (incl. character_1) is summary
    assert "long red hair" in doc["summary"]["caption"]


def test_widget_description_wins_and_background_not_doubled():
    cap = json.loads(json.dumps(KJ_CAPTION))
    cap["compositional_deconstruction"]["background"] = "Night street."
    _, _, _, _, _, _, _, _, doc = _build(cap, "", high_level_description="Night street. Rain.")
    assert doc["summary"]["caption"] == "Night street. Rain."


def test_pixel_xy_space_and_snap():
    cap = {"compositional_deconstruction": {"background": "x", "elements": [
        {"type": "obj", "bbox": [0, 0, 500, 1000], "desc": "a door"}]}}
    *_, report, doc = _build(cap, width=1000, height=1000, bbox_space="pixels xy")
    assert (doc["width"], doc["height"]) == (992, 992)
    assert doc["boxes"][0]["rect"] == [0.0, 0.0, 0.5, 1.0]        # measured on the 1000 grid
    assert "(pixels xy)" in report


def test_normalized_space_warns_on_pixel_values():
    cap = {"compositional_deconstruction": {"background": "", "elements": [
        {"type": "obj", "bbox": [0, 0, 1216, 832], "desc": "a wall"}]}}
    report = _build(cap)[7]
    assert "WARN: 1 bbox(es) exceed the 0-1000 grid" in report


def test_bad_caption_json_reports_error():
    out = FVM_MOSS_FromIdeogram().build(
        caption_json="not json", width=1024, height=1024, upscale_by=1.5,
        high_level_description="Scene.", meta="", canvas="single")
    assert "ERROR: caption_json is not a JSON object" in out[7]
    layout = json.loads(out[0])
    assert layout["items"] == [] and layout["summary"]["caption"] == "Scene."


def test_preview_uses_image_background():
    image = torch.zeros(1, 300, 200, 3)
    image[..., 1] = 1.0                                   # green reference
    preview = _build(KJ_CAPTION, JB_PROMPT, image=image)[6]
    assert preview.shape == (1, 1216, 832, 3)
    assert float(preview[0, 600, 700, 1]) > 0.9           # green, not grey, outside boxes


def test_join_unique():
    assert join_unique("A b.", "a  B. C d.", "C d.") == "a  B. C d."    # swallowed in place
    assert join_unique("X.", "", None, "Y.", "x.") == "X. Y."
    assert join_unique("P1.", "P2.", "P1. P2. P3.") == "P1. P2. P3."


def test_caption_straight_from_sam3_node_is_not_doubled():
    # The SAM3 layout node's ideogram_json: background = its whole summary
    # caption (high level description + JB rest prose), desc = slot keys.
    sam3_doc, _ = make_boxes_doc(
        [{"bbox": [0, 0, 50, 100], "label": "character", "slot": "character_1",
          "prompt": "person"}],
        100, 100, 1024, 1024, prompt=JB_PROMPT, high_level_description="Candid photo.")
    background = sam3_doc["summary"]["caption"]
    assert background.startswith("Candid photo.") and "busy corner cafe" in background
    cap = {"compositional_deconstruction": {"background": background, "elements": [
        {"type": "obj", "bbox": [0, 0, 1000, 500], "desc": "character_1"}]}}

    for hld in ("", "Candid photo."):
        moss, *_, report, doc = _build(cap, JB_PROMPT, width=1024, height=1024,
                                       high_level_description=hld)
        summary = json.loads(moss)["summary"]["caption"]
        assert summary == background
        assert summary.count("busy corner cafe") == 1 and summary.count("Candid photo.") == 1
        assert "long red hair" in json.loads(moss)["items"][0]["caption"]
        assert "filled from prompt : ['#1 character_1']" in report


def test_ideogram_doc_core_direct():
    doc, report = ideogram_doc(KJ_CAPTION, 1024, 1024, prompt=JB_PROMPT)
    assert [b["label"] for b in doc["boxes"]] == ["character", "character", "object", "text"]
    assert report["from_prompt"] == ["#1 character_1"]
    assert len(report["skipped"]) == 1 and report["summary_keys"] == ["location"]
