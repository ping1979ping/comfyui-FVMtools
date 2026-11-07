"""MOSS boxes -> PERSON_DATA (SAM3 and BiSeNet mocked) and the PersonDetailer contract."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# person_detailer pulls in ComfyUI modules the shared conftest doesn't mock.
for _mod in (
    "comfy.model_base",
    "comfy.lora",
    "comfy.lora_convert",
    "comfy.model_detection",
    "comfy.k_diffusion",
    "comfy.k_diffusion.sampling",
    "node_helpers",
):
    sys.modules.setdefault(_mod, MagicMock())

import nodes.jb.moss_box_person as node_mod  # noqa: E402
from nodes.jb.moss_box_person import (  # noqa: E402
    FVM_MOSS_BoxPersonData, MASK_TYPES, assign_persons_to_slots, resolve_slots,
)
from nodes.person_detailer import PersonDetailer, INPAINT_DEFAULTS  # noqa: E402

H, W = 100, 200


def _rect(x1, y1, x2, y2):
    m = np.zeros((H, W), dtype=np.float32)
    m[y1:y2, x1:x2] = 1.0
    return m


def _person(x1, x2, score):
    """A standing person between x1..x2: body, face and head masks + SAM3 tuples."""
    cx = (x1 + x2) // 2
    body = (_rect(x1, 15, x2, 95), score, [x1, 15, x2, 95])
    face = (_rect(cx - 6, 20, cx + 6, 32), score, [cx - 6, 20, cx + 6, 32])
    head = (_rect(cx - 9, 16, cx + 9, 34), score, [cx - 9, 16, cx + 9, 34])
    return {"person": body, "face": face, "head": head}


# SAM3 results come back best score first; that order is the per_face order.
LEFT = _person(15, 85, 0.90)      # inside character_1's box
RIGHT = _person(115, 185, 0.80)   # inside character_2's box
MIDDLE = _person(92, 108, 0.70)   # between the boxes -> no slot


def _doc(*extra):
    boxes = [
        {"slot": "character_1", "label": "character", "caption": "a woman with a red bob",
         "rect": [0.05, 0.1, 0.45, 0.95]},
        {"slot": "character_2", "label": "character", "caption": "a man in a grey suit",
         "rect": [0.55, 0.1, 0.95, 0.95]},
        *extra,
    ]
    return {"version": 1, "width": 1024, "height": 512, "summary": {"caption": ""}, "boxes": boxes}


def _mock_sam3(monkeypatch, scenes, aux=None):
    """scenes: one list of persons per image, in SAM3's detection order."""
    calls = {"n": 0}

    def prepare(cfg, rgb):
        assert rgb.shape == (H, W, 3)
        calls["n"] += 1
        return "proc", calls["n"] - 1

    def ground(proc, state, shape, prompt, threshold):
        if prompt in ("person", "face", "head"):
            return [p[prompt] for p in scenes[state]]
        if aux and prompt == aux["prompt"]:
            return aux["hits"][state]
        return []

    monkeypatch.setattr(node_mod, "sam3_prepare", prepare)
    monkeypatch.setattr(node_mod, "sam3_ground", ground)
    monkeypatch.setattr(node_mod, "sam3_last_error", lambda: None)
    # BiSeNet: the eyes are the top half of the face box, everything else empty.
    def subtypes(rgb, face_box, body_np, device):
        x1, y1, x2, y2 = (int(v) for v in face_box)
        out = {mt: np.zeros((H, W), dtype=np.float32) for mt in node_mod.BISENET_SUBTYPES}
        out["eyes"][y1:(y1 + y2) // 2, x1:x2] = 1.0
        return out
    monkeypatch.setattr(node_mod, "_bisenet_subtypes", subtypes)


def _run(monkeypatch, scenes, doc=None, batch=1, aux=None, **kw):
    _mock_sam3(monkeypatch, scenes, aux)
    images = torch.rand(batch, H, W, 3)
    args = dict(images=images, moss_boxes=doc or _doc(), sam3_model=object())
    args.update(kw)
    return FVM_MOSS_BoxPersonData().build(**args)


def _same(t, m):
    return torch.equal(t, torch.from_numpy(m))


# ── pure helpers ──

def test_resolve_slots_explicit_dedup_and_cap():
    names, warn = resolve_slots("a, b,\nb, c, d, e, f", {})
    assert names == ["a", "b", "c", "d", "e"] and "dropped ['f']" in warn[0]


def test_resolve_slots_empty_takes_character_boxes_naturally_sorted():
    doc = _doc({"slot": "character_10", "label": "character", "rect": [0, 0, 1, 1]},
               {"slot": "table", "label": "object", "rect": [0, 0, 1, 1]})
    names, warn = resolve_slots("  ", doc)
    assert names == ["character_1", "character_2", "character_10"] and warn == []


def test_assign_best_first_one_person_per_slot():
    slots = [[0, 0, 100, 100], [100, 0, 200, 100], None]
    persons = [[110, 0, 200, 100], [0, 0, 90, 100], [0, 0, 100, 100]]
    out, scores = assign_persons_to_slots(persons, slots, min_overlap=0.1)
    assert out[0][0] == 2 and out[0][1] == pytest.approx(1.0)   # exact fit wins slot 0
    assert out[1][0] == 0                                         # right person -> slot 1
    assert 2 not in out and scores.shape == (3, 3)                # slot without box
    out, _ = assign_persons_to_slots(persons, slots, min_overlap=0.95)
    assert list(out) == [0]


# ── node ──

def test_two_persons_land_in_their_slots(monkeypatch):
    out = _run(monkeypatch, [[LEFT, RIGHT]])
    pd, p1, p2, p3, p4, p5, preview, report = out
    assert pd["num_references"] == 5 and pd["batch_size"] == 1
    assert (pd["image_height"], pd["image_width"]) == (H, W)
    assert _same(pd["body_masks"][0][0], LEFT["person"][0])
    assert _same(pd["face_masks"][1][0], RIGHT["face"][0])
    assert _same(pd["head_masks"][0][0], LEFT["head"][0])
    assert pd["eyes_masks"][0][0].sum() > 0                       # BiSeNet subtypes filled
    for mt in MASK_TYPES:
        assert len(pd[f"{mt}_masks"]) == 5
        assert pd[f"{mt}_masks"][2].shape == (1, H, W) and pd[f"{mt}_masks"][2].sum() == 0
    assert pd["matches"] == [[True, True, False, False, False]]
    assert pd["face_to_ref"] == [[0, 1]]
    assert (p1, p2, p3, p4, p5) == ("a woman with a red bob", "a man in a grey suit", "", "", "")
    assert "reference_1 (character_1): found P1" in report
    assert "reference_3 (character_3): slot missing in moss_boxes" in report
    assert preview.shape == (1, H, W, 3)
    assert set(pd["ref_depths"][0]) == {0, 1}
    assert "aux_masks" not in pd


def test_swapped_detection_order_keeps_slots(monkeypatch):
    """The right person is detected first (higher score) — slots follow the boxes."""
    right_first, left_second = _person(115, 185, 0.95), _person(15, 85, 0.60)
    pd = _run(monkeypatch, [[left_second, right_first]])[0]
    assert _same(pd["body_masks"][0][0], left_second["person"][0])
    assert _same(pd["body_masks"][1][0], right_first["person"][0])
    assert pd["face_to_ref"] == [[1, 0]]           # per_face order = SAM3 score order


def test_person_outside_every_box_is_unmatched(monkeypatch):
    pd, *_, report = _run(monkeypatch, [[LEFT, RIGHT, MIDDLE]])
    assert pd["face_to_ref"] == [[0, 1, None]]
    mid_face = torch.from_numpy(MIDDLE["face"][0])
    assert float((pd["all_faces_mask"][0] * mid_face).sum()) == mid_face.sum()
    assert float((pd["matched_faces_mask"][0] * mid_face).sum()) == 0
    assert len(pd["per_face_masks"][0]) == 3
    assert "P3: unassigned -> generic" in report


def test_missing_slot_keeps_reference_numbering(monkeypatch):
    out = _run(monkeypatch, [[LEFT, RIGHT]], slots="character_1, ghost, character_2")
    pd, p1, p2, p3 = out[:4]
    report = out[-1]
    assert pd["num_references"] == 3
    assert pd["body_masks"][1].sum() == 0 and p2 == ""
    assert _same(pd["body_masks"][2][0], RIGHT["person"][0]) and p3 == "a man in a grey suit"
    assert pd["matches"] == [[True, False, True]]
    assert "reference_2 (ghost): slot missing in moss_boxes" in report


def test_slot_box_without_person(monkeypatch):
    pd, *_, report = _run(monkeypatch, [[LEFT]])
    assert pd["matches"] == [[True, False, False, False, False]]
    assert "reference_2 (character_2): not found (best IoU 0.00)" in report


def test_batch_of_two(monkeypatch):
    pd = _run(monkeypatch, [[LEFT, RIGHT], [RIGHT, MIDDLE]], batch=2)[0]
    assert pd["batch_size"] == 2
    assert pd["body_masks"][0].shape == (2, H, W)
    assert pd["all_faces_mask"].shape == (2, H, W)
    assert _same(pd["body_masks"][0][0], LEFT["person"][0])
    assert pd["body_masks"][0][1].sum() == 0                  # no left person in image 2
    assert _same(pd["body_masks"][1][1], RIGHT["person"][0])
    assert pd["face_to_ref"] == [[0, 1], [1, None]]
    assert pd["matches"][1] == [False, True, False, False, False]
    assert len(pd["ref_depths"]) == 2


def test_other_resolution_same_aspect(monkeypatch):
    """Boxes are relative: a 2x canvas in the doc does not matter."""
    doc = _doc()
    doc["width"], doc["height"] = 2048, 1024
    pd, *_, report = _run(monkeypatch, [[LEFT, RIGHT]], doc=doc)
    assert pd["face_to_ref"] == [[0, 1]] and "WARN" not in report


def test_aux_prompt_assigns_hits(monkeypatch):
    hand_l = (_rect(20, 60, 30, 70), 0.5, [20, 60, 30, 70])     # on LEFT's body
    hand_m = (_rect(95, 60, 100, 65), 0.5, [95, 60, 100, 65])   # on MIDDLE (no slot)
    loose = (_rect(0, 0, 5, 5), 0.5, [0, 0, 5, 5])              # on nobody
    aux = {"prompt": "hand", "hits": [[hand_l, hand_m, loose]]}
    pd = _run(monkeypatch, [[LEFT, RIGHT, MIDDLE]], aux=aux, aux_prompt="hand")[0]
    assert _same(pd["aux_masks"][0][0], hand_l[0])
    assert pd["aux_masks"][1][0].sum() == 0
    assert pd["aux_part_counts"] == [{0: 1, 1: 0, 2: 0, 3: 0, 4: 0}]
    un = pd["aux_unassigned_masks"]
    assert un.shape == (1, H, W)
    assert _same(un[0], np.maximum(hand_m[0], loose[0]))


def test_without_sam3_says_so(monkeypatch):
    out = FVM_MOSS_BoxPersonData().build(images=torch.rand(1, H, W, 3), moss_boxes=_doc())
    pd, report = out[0], out[-1]
    assert "ERROR: no SAM3 wired" in report
    assert pd["per_face_masks"] == [[]] and pd["matches"] == [[False] * 5]
    assert out[1] == "a woman with a red bob"


def test_sam3_failure_is_reported(monkeypatch):
    monkeypatch.setattr(node_mod, "sam3_prepare", lambda cfg, img: (None, None))
    monkeypatch.setattr(node_mod, "sam3_last_error", lambda: "no checkpoint")
    out = FVM_MOSS_BoxPersonData().build(images=torch.rand(1, H, W, 3), moss_boxes=_doc(),
                                         sam3_model=object())
    assert "ERROR SAM3: no checkpoint" in out[-1]


def test_wraps_native_sam3(monkeypatch):
    seen = {}

    def prepare(cfg, img):
        seen["cfg"] = cfg
        return None, None

    monkeypatch.setattr(node_mod, "sam3_prepare", prepare)
    model, clip = object(), object()
    FVM_MOSS_BoxPersonData().build(images=torch.rand(1, H, W, 4), moss_boxes=_doc(),
                                   sam3_native_model=model, sam3_native_clip=clip,
                                   sam3_model=object())
    assert isinstance(seen["cfg"], node_mod.NativeSAM3)
    assert seen["cfg"].model is model and seen["cfg"].clip is clip


def test_schema_matches_build_signature():
    types = FVM_MOSS_BoxPersonData.INPUT_TYPES()
    assert types["required"]["moss_boxes"][0] == "MOSS_BOXES"
    assert types["optional"]["sam3_native_model"][0] == "MODEL"
    assert types["optional"]["sam3_model"][0] == "SAM3_MODEL_CONFIG"
    assert len(FVM_MOSS_BoxPersonData.RETURN_TYPES) == len(FVM_MOSS_BoxPersonData.RETURN_NAMES) == 8
    assert FVM_MOSS_BoxPersonData.RETURN_NAMES[1] == "reference_1_prompt"


# ── contract: the real PersonDetailer.execute, sampling stubbed ──

def _detail(pd, images, monkeypatch, ref_types, generic_type="head", enabled=(True, True),
            catch_unprocessed=True):
    calls = []

    def inpaint(self, current_image, mask, slot, **kw):
        calls.append((slot.get("label", "generic"), mask.clone()))
        return current_image, None

    monkeypatch.setattr(PersonDetailer, "_inpaint_mask", inpaint)
    monkeypatch.setattr(PersonDetailer, "_apply_lora", lambda self, m, c, n, s: (m, c))
    monkeypatch.setattr(PersonDetailer, "_encode_prompt", lambda self, c, t: [["cond", {}]])
    slots = {f"reference_{i}": {"mask_type": ref_types.get(i, "head"), "detail_daemon": True,
                                "rounds": 1} for i in range(1, 6)}
    slots["generic"] = {"mask_type": generic_type, "detail_daemon": True, "rounds": 1}
    opts = dict(INPAINT_DEFAULTS, slots=slots)
    refs = []
    for i in range(1, 6):
        refs += [i <= len(enabled) and enabled[i - 1], "None", 1.0, f"prompt {i}"]
    PersonDetailer().execute(
        images, pd, "model", "clip", "vae", 0, 4, 0.5, "euler", "simple",
        False, 0.2, True, 32, 0, 512, 512, *refs,
        True, catch_unprocessed, "None", 1.0, "", inpaint_options=opts)
    return calls


def test_person_detailer_details_each_slot_with_its_mask(monkeypatch):
    out = _run(monkeypatch, [[RIGHT, LEFT, MIDDLE]])
    pd = out[0]
    images = torch.rand(1, H, W, 3)
    calls = _detail(pd, images, monkeypatch, ref_types={1: "head", 2: "body"})
    labels = [c[0] for c in calls]
    assert labels == ["Ref1", "Ref2", "generic"]
    assert _same(calls[0][1], LEFT["head"][0])        # character_1 -> left person's head
    assert _same(calls[1][1], RIGHT["person"][0])     # character_2 -> right person's body
    assert _same(calls[2][1], MIDDLE["head"][0])      # nobody's person -> Generic


def test_person_detailer_generic_catches_disabled_slot(monkeypatch):
    pd = _run(monkeypatch, [[LEFT, RIGHT, MIDDLE]])[0]
    calls = _detail(pd, torch.rand(1, H, W, 3), monkeypatch, ref_types={},
                    generic_type="face", enabled=(False, True))
    assert [c[0] for c in calls] == ["Ref2", "generic", "generic"]
    generic = {int(np.argwhere(c[1].numpy() > 0)[0][1]) for c in calls if c[0] == "generic"}
    assert generic == {int(LEFT["face"][2][0]), int(MIDDLE["face"][2][0])}


def test_person_detailer_aux_slot(monkeypatch):
    hand = (_rect(20, 60, 30, 70), 0.5, [20, 60, 30, 70])
    pd = _run(monkeypatch, [[LEFT, RIGHT]], aux={"prompt": "hand", "hits": [[hand]]},
              aux_prompt="hand")[0]
    calls = _detail(pd, torch.rand(1, H, W, 3), monkeypatch, ref_types={1: "aux"},
                    enabled=(True, False))
    ref1 = [c for c in calls if c[0] == "Ref1"]
    assert len(ref1) == 1 and float(ref1[0][1].sum()) == float(hand[0].sum())
