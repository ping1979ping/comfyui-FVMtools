"""Tests for the free-text aux channel of PersonSelectorSAM3Native.

aux_prompt grounds a second, independent SAM3 text prompt; its hits are assigned
to persons by body overlap and handed back as `aux_data` — a PERSON_DATA where
every mask type carries the aux region, plus an overlay on the preview.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

try:
    import nodes.person_selector_sam3_native as native
except (ImportError, ModuleNotFoundError):
    for _mod in ("nodes.utils.masker", "nodes.utils.face_analyzer",
                 "nodes.utils.yolo_detector"):
        sys.modules.setdefault(_mod, MagicMock())
    try:
        import nodes.person_selector_sam3_native as native
    except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
        pytest.skip(f"PersonSelectorSAM3Native unavailable: {e}", allow_module_level=True)


NODE = native.PersonSelectorSAM3Native
H = W = 64


def _rect(x0, y0, x1, y1):
    m = torch.zeros(1, H, W, dtype=torch.float32)
    m[0, y0:y1, x0:x1] = 1.0
    return m


def make_person_data(n=2):
    """Two persons: left half and right half."""
    bodies = [_rect(0, 0, 30, H), _rect(34, 0, W, H)][:n]
    pd = {
        "batch_size": 1, "num_references": n, "image_height": H, "image_width": W,
        "depth_sort_order": "front_last",
        "matches": [[True] * n],
        "face_to_ref": [list(range(n))],
        "ref_depths": [{i: 0.5 for i in range(n)}],
        "per_face_masks": [[{"body": b} for b in bodies]],
    }
    for mt in ("face", "head", "body"):
        pd[f"{mt}_masks"] = [b.clone() for b in bodies]
    return pd


def ground_returning(*rects):
    """Fake sam3_ground yielding one detection per rect (x0,y0,x1,y1)."""
    def _g(_state, _base, _shape, _prompt, _threshold=0.2):
        out = []
        for (x0, y0, x1, y1) in rects:
            m = np.zeros((H, W), dtype=np.float32)
            m[y0:y1, x0:x1] = 1.0
            out.append((m, 0.9, [x0, y0, x1, y1]))
        return out
    return _g


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(native, "sam3_prepare", lambda _b, _rgb: ("STATE", {}))
    return monkeypatch


class TestSchema:
    def test_aux_prompt_widget(self):
        req = NODE.INPUT_TYPES()["required"]
        assert req["aux_prompt"][0] == "STRING"
        assert req["aux_prompt"][1]["default"] == ""
        assert req["aux_prompt_threshold"][0] == "FLOAT"

    def test_aux_data_output_appended(self):
        from nodes.person_selector_sam3 import PersonSelectorSAM3
        assert NODE.RETURN_NAMES[-1] == "aux_data"
        assert NODE.RETURN_TYPES[-1] == "PERSON_DATA"
        # the parent's outputs keep their positions -> drop-in replaceable
        assert NODE.RETURN_NAMES[:len(PersonSelectorSAM3.RETURN_NAMES)] == \
            PersonSelectorSAM3.RETURN_NAMES


class TestBuildAux:
    def _run(self, monkeypatch, prompt, rects, n=2, images=None):
        node = NODE()
        pd = make_person_data(n)
        if rects is not None:
            monkeypatch.setattr(native, "sam3_ground", ground_returning(*rects))
        imgs = images if images is not None else torch.zeros(1, H, W, 3)
        preview = torch.zeros(1, H, W, 3)
        return node._build_aux(object(), imgs, pd, preview, prompt, 0.3)

    def test_empty_prompt_gives_empty_aux(self, wired):
        aux, preview = self._run(wired, "", None)
        assert aux["num_references"] == 2
        for ri in range(2):
            assert float(aux["body_masks"][ri].max()) == 0.0

    def test_hit_lands_on_the_overlapping_person(self, wired):
        # rect inside the LEFT person's body (x 0..30)
        aux, _p = self._run(wired, "necklace", [(5, 5, 20, 20)])
        assert float(aux["body_masks"][0].max()) > 0.5, "left person must own the hit"
        assert float(aux["body_masks"][1].max()) == 0.0

    def test_second_person_hit(self, wired):
        aux, _p = self._run(wired, "necklace", [(40, 5, 60, 20)])
        assert float(aux["body_masks"][0].max()) == 0.0
        assert float(aux["body_masks"][1].max()) > 0.5

    def test_every_mask_type_carries_the_aux_region(self, wired):
        aux, _p = self._run(wired, "shoes", [(5, 5, 20, 20)])
        for mt in ("face", "head", "body", "aux"):
            key = f"{mt}_masks"
            assert key in aux, f"missing {key}"
            assert float(aux[key][0].max()) > 0.5, f"{key} must carry the aux region"

    def test_unassigned_hit_is_kept_separate(self, wired):
        # gap between the two bodies (x 30..34) overlaps nobody
        aux, _p = self._run(wired, "ghost", [(31, 5, 33, 20)])
        assert float(aux["body_masks"][0].max()) == 0.0
        assert float(aux["body_masks"][1].max()) == 0.0
        assert float(aux["aux_unassigned_masks"].max()) > 0.5

    def test_person_data_contract(self, wired):
        aux, _p = self._run(wired, "hat", [(5, 5, 20, 20)])
        for key in ("batch_size", "num_references", "image_height", "image_width",
                    "matches", "face_to_ref", "per_face_masks", "all_faces_mask",
                    "matched_faces_mask", "aux_part_counts"):
            assert key in aux, f"PERSON_DATA key missing: {key}"
        assert aux["all_faces_mask"].shape == (1, H, W)
        assert len(aux["per_face_masks"][0]) == 2
        assert isinstance(aux["per_face_masks"][0][0], dict)

    def test_aux_part_counts_flag_hits(self, wired):
        aux, _p = self._run(wired, "hat", [(5, 5, 20, 20)])
        assert aux["aux_part_counts"][0][0] == 1
        assert aux["aux_part_counts"][0][1] == 0

    def test_two_hits_on_one_person_are_unioned(self, wired):
        aux, _p = self._run(wired, "tattoo", [(2, 2, 10, 10), (15, 30, 25, 40)])
        m = aux["body_masks"][0][0].numpy()
        assert m[5, 5] > 0.5 and m[35, 20] > 0.5, "both hits must survive"


class TestPreviewOverlay:
    def test_preview_is_marked_when_aux_hits(self, monkeypatch):
        monkeypatch.setattr(native, "sam3_prepare", lambda _b, _rgb: ("STATE", {}))
        monkeypatch.setattr(native, "sam3_ground", ground_returning((5, 5, 20, 20)))
        node = NODE()
        pd = make_person_data(2)
        preview = torch.zeros(1, H, W, 3)
        _aux, out = node._build_aux(object(), torch.zeros(1, H, W, 3), pd, preview,
                                    "necklace", 0.3)
        assert out.shape == preview.shape
        changed = int(((out - preview).abs().sum(dim=-1) > 0.01).sum())
        assert changed > 0, "aux region must be visible in the preview"

    def test_preview_untouched_without_hits(self, monkeypatch):
        monkeypatch.setattr(native, "sam3_prepare", lambda _b, _rgb: ("STATE", {}))
        monkeypatch.setattr(native, "sam3_ground", lambda *_a, **_k: [])
        node = NODE()
        preview = torch.rand(1, H, W, 3)
        _aux, out = node._build_aux(object(), torch.zeros(1, H, W, 3),
                                    make_person_data(2), preview, "nothing", 0.3)
        assert torch.equal(out, preview)

    def test_batch_preview_keeps_every_frame(self, monkeypatch):
        """Regression: the overlay used to collapse a batch preview to frame 0."""
        monkeypatch.setattr(native, "sam3_prepare", lambda _b, _rgb: ("STATE", {}))
        monkeypatch.setattr(native, "sam3_ground", ground_returning((5, 5, 20, 20)))
        pd = make_person_data(2)
        pd["batch_size"] = 3
        for mt in ("face", "head", "body"):
            pd[f"{mt}_masks"] = [m.repeat(3, 1, 1) for m in pd[f"{mt}_masks"]]
        preview = torch.stack([torch.full((H, W, 3), v) for v in (0.1, 0.5, 0.9)])
        _aux, out = NODE()._build_aux(object(), torch.zeros(3, H, W, 3), pd, preview,
                                      "necklace", 0.3)
        assert out.shape == (3, H, W, 3)
        # outside the aux region every frame keeps its own background
        for b, v in enumerate((0.1, 0.5, 0.9)):
            assert abs(float(out[b, 60, 60, 0]) - v) < 0.01

    def test_part_counts_count_every_hit(self, wired):
        node = NODE()
        monkeypatch = wired
        monkeypatch.setattr(native, "sam3_ground",
                            ground_returning((2, 2, 10, 10), (15, 30, 25, 40)))
        aux, _p = node._build_aux(object(), torch.zeros(1, H, W, 3), make_person_data(2),
                                  torch.zeros(1, H, W, 3), "hand", 0.3)
        assert aux["aux_part_counts"][0] == {0: 2, 1: 0}

    def test_overlay_failure_returns_original_preview(self, monkeypatch):
        preview = torch.rand(1, H, W, 3)
        out = NODE._draw_aux(preview, "not-a-list", None, "boom")
        assert torch.equal(out, preview), "a broken overlay must not kill the run"
