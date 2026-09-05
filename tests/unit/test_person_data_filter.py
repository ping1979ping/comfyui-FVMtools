"""Tests for PersonDataFilter — the PERSON_DATA split node.

Covers the three stages (gate / rank / cut), both `source` modes, both `count`
semantics, and the structural contract of the two PERSON_DATA outputs: a filtered
output must be a valid, re-indexed PERSON_DATA that a PersonDetailer can consume.
"""

import sys
from unittest.mock import MagicMock

# ``nodes.utils.masker`` does ``from ...parsing import BiSeNet`` and
# ``nodes.utils.face_analyzer`` imports insightface — neither resolves outside
# ComfyUI's loader. Pre-stub both; the filter only calls into them for the
# SAM3 / InsightFace gate, which these tests exercise via injected fakes.
for _mod in ("nodes.utils.masker", "nodes.utils.face_analyzer"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import numpy as np
import pytest
import torch

try:
    import nodes.person_data_filter as pdf
except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
    pytest.skip(f"PersonDataFilter unavailable in this test environment: {e}",
                allow_module_level=True)


H, W = 64, 64
MASK_TYPES = ["face", "head", "body"]


def _rect_mask(x0, y0, x1, y1):
    """[1, H, W] mask with a filled rectangle."""
    m = torch.zeros(1, H, W, dtype=torch.float32)
    m[0, y0:y1, x0:x1] = 1.0
    return m


def _person(x0, y0, x1, y1):
    """One person's mask dict — face is the top strip of the body box."""
    body = _rect_mask(x0, y0, x1, y1)
    face_h = max(2, (y1 - y0) // 4)
    face = _rect_mask(x0, y0, x1, y0 + face_h)
    return {"face": face, "head": face.clone(), "body": body}


def make_person_data(boxes, matched=None, batch_size=1):
    """Build a PERSON_DATA dict with one person per box.

    boxes:   list of (x0, y0, x1, y1)
    matched: list of bool — which persons occupy a matched reference slot.
             Defaults to all matched.
    """
    if matched is None:
        matched = [True] * len(boxes)
    per_face = [_person(*b) for b in boxes]

    pd = {
        "batch_size": batch_size,
        "num_references": len(boxes),
        "image_height": H,
        "image_width": W,
        "depth_sort_order": "front_last",
        "matches": [list(matched) for _ in range(batch_size)],
        "per_face_masks": [[dict(p) for p in per_face] for _ in range(batch_size)],
        "face_to_ref": [[i for i in range(len(boxes))] for _ in range(batch_size)],
        "ref_depths": [{i: 0.1 * i for i in range(len(boxes))} for _ in range(batch_size)],
        "all_faces_mask": torch.zeros(batch_size, H, W),
        "matched_faces_mask": torch.zeros(batch_size, H, W),
    }
    for mt in MASK_TYPES:
        pd[f"{mt}_masks"] = [
            torch.cat([per_face[ri][mt] for _ in range(batch_size)], dim=0)
            for ri in range(len(boxes))
        ]
    return pd


# Three persons: small-left, big-middle, medium-right
BOXES = [
    (2, 10, 12, 30),    # P1: area 200, cx ~7
    (20, 4, 44, 60),    # P2: area 1344, cx ~32
    (48, 20, 60, 50),   # P3: area 360, cx ~54
]


def run(pd, **kwargs):
    node = pdf.PersonDataFilter()
    params = dict(source="all_detected", gender="any", sort_by="area",
                  order="descending", mode="top_n", count=1)
    params.update(kwargs)
    return node.execute(person_data=pd, **params)


class TestContract:
    def test_return_shape(self):
        assert len(pdf.PersonDataFilter.RETURN_TYPES) == len(pdf.PersonDataFilter.RETURN_NAMES)
        assert pdf.PersonDataFilter.RETURN_TYPES[:2] == ("PERSON_DATA", "PERSON_DATA")

    def test_input_types_widgets(self):
        req = pdf.PersonDataFilter.INPUT_TYPES()["required"]
        for key in ("person_data", "source", "gender", "sort_by", "order", "mode",
                    "count", "sam3_prompt"):
            assert key in req, f"missing widget: {key}"
        assert req["sort_by"][0] == pdf.SORT_CRITERIA
        assert req["count"][0] == "INT"

    def test_execute_returns_tuple(self):
        out = run(make_person_data(BOXES))
        assert isinstance(out, tuple)
        assert len(out) == len(pdf.PersonDataFilter.RETURN_TYPES)

    def test_split_is_complete_and_disjoint(self):
        filtered, remaining, *_ = run(make_person_data(BOXES), count=1)
        assert filtered["num_references"] == 1
        assert remaining["num_references"] == 2

    def test_filtered_person_data_is_reindexed(self):
        """Kept persons become slots 0..n-1 with matches all True."""
        filtered, _remaining, *_ = run(make_person_data(BOXES), count=2)
        assert filtered["num_references"] == 2
        assert filtered["matches"][0] == [True, True]
        assert filtered["face_to_ref"][0] == [0, 1]
        for mt in MASK_TYPES:
            lst = filtered[f"{mt}_masks"]
            assert len(lst) == 2
            for t in lst:
                assert t.shape == (1, H, W)

    def test_per_face_masks_stays_a_list_of_dicts(self):
        """`per_face_masks` also ends in `_masks` — it must not be mistaken
        for a per-slot tensor list and overwritten."""
        filtered, _r, *_ = run(make_person_data(BOXES), count=2)
        pfm = filtered["per_face_masks"]
        assert isinstance(pfm, list) and len(pfm) == 1
        assert len(pfm[0]) == 2
        for md in pfm[0]:
            assert isinstance(md, dict)
            assert md["body"].shape == (1, H, W)

    def test_all_faces_mask_covers_kept_only(self):
        filtered, _r, *_ = run(make_person_data(BOXES), count=1)
        # only P2's face strip (y 4..18, x 20..44) should be set
        m = filtered["all_faces_mask"][0]
        assert m[10, 30] > 0.5
        assert m[12, 7] < 0.5

    def test_batch_size_preserved(self):
        pd = make_person_data(BOXES, batch_size=3)
        filtered, remaining, *_ = run(pd, count=1)
        assert filtered["batch_size"] == 3
        assert remaining["batch_size"] == 3
        assert filtered["body_masks"][0].shape == (3, H, W)


class TestRankAndCut:
    def test_area_descending_picks_largest(self):
        filtered, _r, *_ = run(make_person_data(BOXES), sort_by="area",
                               order="descending", count=1)
        body = filtered["body_masks"][0][0]
        assert body[30, 30] > 0.5, "expected the big middle person"

    def test_area_ascending_picks_smallest(self):
        filtered, _r, *_ = run(make_person_data(BOXES), sort_by="area",
                               order="ascending", count=1)
        body = filtered["body_masks"][0][0]
        assert body[20, 7] > 0.5, "expected the small left person"

    def test_horizontal_ascending_is_left_to_right(self):
        filtered, _r, *_ = run(make_person_data(BOXES), sort_by="horizontal",
                               order="ascending", count=1)
        assert filtered["body_masks"][0][0][20, 7] > 0.5

    def test_horizontal_descending_is_right_to_left(self):
        filtered, _r, *_ = run(make_person_data(BOXES), sort_by="horizontal",
                               order="descending", count=1)
        assert filtered["body_masks"][0][0][30, 54] > 0.5

    def test_vertical_ascending_is_top_to_bottom(self):
        filtered, _r, *_ = run(make_person_data(BOXES), sort_by="vertical",
                               order="ascending", count=1)
        # P2 spans y 4..60 (cy 32), P1 y 10..30 (cy 20), P3 y 20..50 (cy 35)
        assert filtered["body_masks"][0][0][20, 7] > 0.5

    def test_index_order_is_detection_order(self):
        filtered, _r, *_ = run(make_person_data(BOXES), sort_by="index",
                               order="ascending", count=1)
        assert filtered["body_masks"][0][0][20, 7] > 0.5

    def test_top_n_keeps_first_n(self):
        filtered, remaining, *_ = run(make_person_data(BOXES), sort_by="area",
                                      order="descending", mode="top_n", count=2)
        assert filtered["num_references"] == 2
        assert remaining["num_references"] == 1
        # remaining is the smallest (P1)
        assert remaining["body_masks"][0][0][20, 7] > 0.5

    def test_count_zero_keeps_all(self):
        filtered, remaining, *_ = run(make_person_data(BOXES), mode="top_n", count=0)
        assert filtered["num_references"] == 3
        assert remaining["num_references"] == 0

    def test_nth_picks_exactly_one(self):
        filtered, remaining, *_ = run(make_person_data(BOXES), sort_by="area",
                                      order="descending", mode="nth", count=2)
        assert filtered["num_references"] == 1
        assert remaining["num_references"] == 2
        # 2nd largest is P3 (area 360)
        assert filtered["body_masks"][0][0][30, 54] > 0.5

    def test_nth_out_of_range_yields_empty_filtered(self):
        filtered, remaining, *_ = run(make_person_data(BOXES), mode="nth", count=9)
        assert filtered["num_references"] == 0
        assert remaining["num_references"] == 3

    def test_count_exceeding_persons_is_clamped(self):
        filtered, remaining, *_ = run(make_person_data(BOXES), mode="top_n", count=10)
        assert filtered["num_references"] == 3
        assert remaining["num_references"] == 0

    def test_invert_swaps_outputs(self):
        normal_f, normal_r, *_ = run(make_person_data(BOXES), count=1)
        inv_f, inv_r, *_ = run(make_person_data(BOXES), count=1, invert=True)
        assert inv_f["num_references"] == normal_r["num_references"]
        assert inv_r["num_references"] == normal_f["num_references"]


class TestSource:
    def test_matched_refs_skips_unmatched_slots(self):
        pd = make_person_data(BOXES, matched=[True, False, True])
        filtered, remaining, *_ = run(pd, source="matched_refs", count=0)
        assert filtered["num_references"] == 2, "unmatched slot must not be a candidate"
        assert remaining["num_references"] == 0

    def test_all_detected_uses_every_face(self):
        pd = make_person_data(BOXES, matched=[True, False, True])
        filtered, _r, *_ = run(pd, source="all_detected", count=0)
        assert filtered["num_references"] == 3

    def test_all_detected_works_without_references(self):
        """No reference images connected: num_references == 0, faces still filterable."""
        pd = make_person_data(BOXES)
        pd["num_references"] = 0
        pd["matches"] = [[]]
        pd["face_to_ref"] = [[None, None, None]]
        for mt in MASK_TYPES:
            pd[f"{mt}_masks"] = []
        filtered, remaining, *_ = run(pd, source="all_detected", sort_by="area",
                                      order="descending", count=1)
        assert filtered["num_references"] == 1
        assert remaining["num_references"] == 2
        assert filtered["body_masks"][0][0][30, 30] > 0.5


class TestGate:
    def test_gate_skipped_without_images(self):
        """gender != any but no image connected: rank/cut still applies, gate warns."""
        _f, _r, _fm, _rm, _prev, fcount, rcount, report = run(
            make_person_data(BOXES), gender="female", count=1)
        assert fcount == 1 and rcount == 2
        assert "gate skipped" in report

    def test_sam3_gender_gate(self, monkeypatch):
        """SAM3 grounds 'woman' onto P2 only -> only P2 passes a female gate."""
        pd_data = make_person_data(BOXES)
        image = torch.zeros(1, H, W, 3, dtype=torch.float32)

        def fake_prepare(_cfg, _rgb):
            return object(), {"backbone_out": None}

        def fake_ground(_proc, _state, _shape, text_prompt, _threshold=0.2):
            if text_prompt != "woman":
                return []
            m = np.zeros((H, W), dtype=np.float32)
            m[4:60, 20:44] = 1.0  # exactly P2's body box
            return [(m, 0.9, [20, 4, 44, 60])]

        monkeypatch.setattr(pdf, "sam3_prepare", fake_prepare)
        monkeypatch.setattr(pdf, "sam3_ground", fake_ground)

        filtered, remaining, *_ = run(pd_data, gender="female", count=0,
                                      images=image, sam3_model={"fake": True})
        assert filtered["num_references"] == 1
        assert filtered["body_masks"][0][0][30, 30] > 0.5, "expected P2 (the 'woman')"
        assert remaining["num_references"] == 2

    def test_free_text_prompt_overrides_gender(self, monkeypatch):
        """A non-empty sam3_prompt replaces the gender criterion entirely."""
        pd_data = make_person_data(BOXES)
        image = torch.zeros(1, H, W, 3, dtype=torch.float32)
        seen = []

        monkeypatch.setattr(pdf, "sam3_prepare", lambda _c, _r: (object(), {}))

        def fake_ground(_proc, _state, _shape, text_prompt, _threshold=0.2):
            seen.append(text_prompt)
            m = np.zeros((H, W), dtype=np.float32)
            m[20:50, 48:60] = 1.0  # P3's box
            return [(m, 0.8, [48, 20, 60, 50])]

        monkeypatch.setattr(pdf, "sam3_ground", fake_ground)

        filtered, _r, *_ = run(pd_data, gender="male", sam3_prompt="person with backpack",
                               count=0, images=image, sam3_model={"fake": True})
        assert seen == ["person with backpack"], "gender prompts must not run"
        assert filtered["num_references"] == 1
        assert filtered["body_masks"][0][0][30, 54] > 0.5, "expected P3"

    def test_min_overlap_rejects_weak_hits(self, monkeypatch):
        """A grounded mask that barely touches a body must not count as a hit."""
        pd_data = make_person_data(BOXES)
        image = torch.zeros(1, H, W, 3, dtype=torch.float32)

        monkeypatch.setattr(pdf, "sam3_prepare", lambda _c, _r: (object(), {}))

        def fake_ground(_proc, _state, _shape, _text, _threshold=0.2):
            m = np.zeros((H, W), dtype=np.float32)
            m[0:64, 0:64] = 1.0      # whole image
            return [(m, 0.9, [0, 0, 64, 64])]

        monkeypatch.setattr(pdf, "sam3_ground", fake_ground)

        filtered, _r, *_ = run(pd_data, sam3_prompt="anything", count=0,
                               min_overlap=0.9, images=image, sam3_model={"fake": True})
        assert filtered["num_references"] == 0

    def test_insightface_fallback_labels_unlabelled(self, monkeypatch):
        """Persons SAM3 could not label fall back to InsightFace genderage."""
        pd_data = make_person_data(BOXES)
        image = torch.zeros(1, H, W, 3, dtype=torch.float32)

        monkeypatch.setattr(pdf, "sam3_prepare", lambda _c, _r: (object(), {}))
        monkeypatch.setattr(pdf, "sam3_ground", lambda *_a, **_k: [])

        class FakeFace:
            def __init__(self, bbox, sex):
                self.bbox = bbox
                self.sex = sex

        class FakeAnalyzer:
            def __init__(self, _det_size):
                pass

            def detect_faces(self, _bgr):
                # face centers inside each person's face strip
                return [FakeFace([2, 10, 12, 15], "F"),
                        FakeFace([20, 4, 44, 18], "M"),
                        FakeFace([48, 20, 60, 27], "M")]

        monkeypatch.setattr(pdf, "FaceAnalyzer", FakeAnalyzer)
        pdf.PersonDataFilter._face_analyzer = None
        pdf.PersonDataFilter._last_det_size = None

        filtered, remaining, *_ = run(pd_data, gender="female", count=0,
                                      images=image, sam3_model={"fake": True})
        assert filtered["num_references"] == 1
        assert filtered["body_masks"][0][0][20, 7] > 0.5, "expected P1 (the only 'F')"
        assert remaining["num_references"] == 2


class TestEdgeCases:
    def test_empty_person_data(self):
        pd = make_person_data([])
        filtered, remaining, fm, rm, _prev, fc, rc, _report = run(pd, count=1)
        assert filtered["num_references"] == 0
        assert remaining["num_references"] == 0
        assert fc == 0 and rc == 0
        assert fm.shape == (1, H, W)

    def test_aux_masks_carry_through(self):
        pd = make_person_data(BOXES)
        aux = [_rect_mask(*b) for b in BOXES]
        pd["aux_masks"] = [a.clone() for a in aux]
        for b_faces in pd["per_face_masks"]:
            for i, face in enumerate(b_faces):
                face["aux"] = aux[i].clone()
        pd["aux_part_counts"] = [{0: 1, 1: 2, 2: 3}]
        pd["aux_unassigned_masks"] = torch.zeros(1, H, W)

        filtered, _r, *_ = run(pd, sort_by="area", order="descending", count=1)
        assert "aux_masks" in filtered
        assert len(filtered["aux_masks"]) == 1
        assert "aux_part_counts" in filtered
        assert "aux_unassigned_masks" in filtered

    def test_report_is_markdown(self):
        *_, report = run(make_person_data(BOXES), count=1)
        assert "## PersonDataFilter" in report
        assert "KEEP" in report

    def test_preview_is_image_tensor(self):
        pd = make_person_data(BOXES)
        image = torch.rand(1, H, W, 3, dtype=torch.float32)
        *_, preview, _fc, _rc, _report = run(pd, count=1, images=image)
        assert preview.dim() == 4 and preview.shape[-1] == 3
        assert preview.dtype == torch.float32
        assert 0.0 <= float(preview.min()) and float(preview.max()) <= 1.0

    def test_preview_without_images_is_placeholder(self):
        *_, preview, _fc, _rc, _report = run(make_person_data(BOXES), count=1)
        assert preview.shape == (1, 64, 64, 3)
