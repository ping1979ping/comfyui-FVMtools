"""Tests for PersonSelectorSAM3 PERSON_DATA contract completion.

Covers the three former contract gaps vs. PersonSelectorMulti:
1. BiSeNet facial subtypes (facial_skin/eyes/mouth/neck/accessories) are now
   derived per face and clipped to the person's SAM3 body mask.
2/3. ``_run_all_sam3_masks`` returns aux_stats (per-face counts + unassigned
   mask) feeding ``aux_part_counts`` / ``aux_unassigned_masks``.
"""

import sys
from unittest.mock import MagicMock

# The real ``nodes.utils.masker`` does ``from ...parsing import BiSeNet`` —
# resolvable only inside ComfyUI's loader. Pre-stub the module so importing
# ``nodes.person_selector_sam3`` succeeds under pytest (same pattern as
# test_person_data_refiner_sam3_aux.py). All masker symbols the node uses
# are replaced per-test on the imported module below.
if "nodes.utils.masker" not in sys.modules:
    sys.modules["nodes.utils.masker"] = MagicMock()

import numpy as np
import pytest
import torch

try:
    import nodes.person_selector_sam3 as psm3
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip(f"PersonSelectorSAM3 unavailable in this test environment: {e}",
                allow_module_level=True)


H, W = 96, 96

# Real label sets (mirror of nodes/utils/masker.py)
MASK_TYPE_LABELS = {
    "facial_skin": {1},
    "eyes": {4, 5},
    "mouth": {11, 12, 13},
    "neck": {14},
    "accessories": {6, 9, 15},
}


def _clip_labels_to_body(label_map, sam_body_mask):
    """Mirror of the real masker._clip_labels_to_body."""
    if sam_body_mask is None:
        return label_map
    clipped = label_map.copy()
    clipped[(clipped > 0) & (sam_body_mask < 0.5)] = 0
    return clipped


def _region_mask(y1, y2, x1, x2):
    m = np.zeros((H, W), dtype=np.float32)
    m[y1:y2, x1:x2] = 1.0
    return m


# Synthetic scene: one person on the left half, aux prompt finds two "hands" —
# one on the body (assigned), one far right outside the body (unassigned).
BODY_NP = _region_mask(0, H, 0, 64)
FACE_NP = _region_mask(20, 50, 30, 60)
HAIR_NP = _region_mask(10, 22, 28, 62)
HAND_IN_NP = _region_mask(60, 71, 10, 21)     # 121 px overlap with body
HAND_OUT_NP = _region_mask(60, 71, 80, 92)    # zero overlap → unassigned


def _fake_label_map():
    """BiSeNet label map: skin/eyes/mouth/neck inside the body, plus a mouth
    blob OUTSIDE the body that must be removed by body clipping."""
    lm = np.zeros((H, W), dtype=np.uint8)
    lm[25:45, 32:58] = 1        # facial skin
    lm[28:31, 36:42] = 4        # left eye
    lm[40:44, 40:52] = 12       # upper lip (inside body)
    lm[50:55, 40:52] = 14       # neck
    lm[40:44, 80:90] = 12       # mouth leakage OUTSIDE body → must be clipped
    return lm


class _FakeMaskGenerator:
    @classmethod
    def _run_bisenet(cls, image_rgb, face, device):
        return _fake_label_map()

    @classmethod
    def generate_face_mask(cls, image_rgb, face, device):
        return FACE_NP


def _fake_sam3_ground(processor, base_state, shape, prompt, threshold=0.3):
    results = {
        "person": [(BODY_NP, 0.9, (0, 0, 64, H))],
        "face": [(FACE_NP, 0.9, (30, 20, 60, 50))],
        "head": [],  # forces face+hair fallback
        "hair": [(HAIR_NP, 0.9, (28, 10, 62, 22))],
        "hand": [(HAND_IN_NP, 0.8, (10, 60, 21, 71)),
                 (HAND_OUT_NP, 0.7, (80, 60, 92, 71))],
    }
    return results.get(prompt, [])


def _fake_assign_by_overlap(mask_results, body_map, faces):
    """Mirror of the real assignment: max pixel overlap, one mask per face."""
    out, used = {}, set()
    for fi, body in body_map.items():
        best_mi, best_ov = None, 0
        for mi, (mask, score, bbox) in enumerate(mask_results):
            if mi in used:
                continue
            ov = float(((mask > 0.5) & (body > 0.5)).sum())
            if ov > best_ov:
                best_ov, best_mi = ov, mi
        if best_mi is not None and best_ov > 50:
            out[fi] = best_mi
            used.add(best_mi)
    return out


@pytest.fixture
def sam3_node(monkeypatch):
    """PersonSelectorSAM3 with all masker symbols replaced by functional fakes."""
    monkeypatch.setattr(psm3, "MaskGenerator", _FakeMaskGenerator)
    monkeypatch.setattr(psm3, "MASK_TYPE_LABELS", MASK_TYPE_LABELS)
    monkeypatch.setattr(psm3, "_clip_labels_to_body", _clip_labels_to_body)
    monkeypatch.setattr(psm3, "sam3_prepare", lambda cfg, rgb: (None, None))
    monkeypatch.setattr(psm3, "sam3_ground", _fake_sam3_ground)
    monkeypatch.setattr(psm3, "assign_masks_to_faces", lambda results, faces: {0: 0} if results else {})
    monkeypatch.setattr(psm3, "assign_masks_by_body_overlap", _fake_assign_by_overlap)
    psm3.PersonSelectorSAM3._bisenet_subtype_warned = False
    return psm3.PersonSelectorSAM3()


@pytest.fixture
def one_face():
    face = MagicMock()
    face.bbox = np.array([30, 20, 60, 50], dtype=np.float32)
    return [face]


CUR_RGB = np.zeros((H, W, 3), dtype=np.uint8)


class TestBisenetSubtypes:
    def test_subtypes_filled(self, sam3_node, one_face):
        per_face, _ = sam3_node._run_all_sam3_masks(None, CUR_RGB, one_face, "none", "", 0.3)
        masks = per_face[0]
        assert masks["mouth"][0, 42, 45] == 1.0
        assert masks["eyes"][0, 29, 38] == 1.0
        assert masks["facial_skin"][0, 35, 35] == 1.0
        assert masks["neck"][0, 52, 45] == 1.0
        # no accessory labels in the map → empty
        assert float(masks["accessories"].max()) == 0.0

    def test_subtypes_clipped_to_body(self, sam3_node, one_face):
        """Mouth leakage outside the SAM3 body mask must be zeroed."""
        per_face, _ = sam3_node._run_all_sam3_masks(None, CUR_RGB, one_face, "none", "", 0.3)
        assert float(per_face[0]["mouth"][0, 40:44, 80:90].max()) == 0.0

    def test_subtype_shapes(self, sam3_node, one_face):
        per_face, _ = sam3_node._run_all_sam3_masks(None, CUR_RGB, one_face, "none", "", 0.3)
        for mt in psm3.BISENET_SUBTYPES:
            assert per_face[0][mt].shape == (1, H, W)
            assert per_face[0][mt].dtype == torch.float32

    def test_bisenet_failure_falls_back_to_empty(self, sam3_node, one_face, monkeypatch):
        def _boom(cls, *a, **kw):
            raise RuntimeError("no weights")
        monkeypatch.setattr(_FakeMaskGenerator, "_run_bisenet", classmethod(_boom))
        per_face, _ = sam3_node._run_all_sam3_masks(None, CUR_RGB, one_face, "none", "", 0.3)
        for mt in psm3.BISENET_SUBTYPES:
            assert float(per_face[0][mt].max()) == 0.0


class TestAuxStats:
    def test_assigned_and_unassigned_split(self, sam3_node, one_face):
        per_face, aux_stats = sam3_node._run_all_sam3_masks(
            None, CUR_RGB, one_face, "custom", "hand", 0.3)
        # on-body hand assigned to face 0
        assert per_face[0]["aux"][0, 65, 15] == 1.0
        assert aux_stats["counts"] == {0: 1}
        # off-body hand lands in unassigned
        assert aux_stats["unassigned_count"] == 1
        assert aux_stats["unassigned_mask"][65, 85] == 1.0
        assert aux_stats["unassigned_mask"][65, 15] == 0.0

    def test_no_aux_preset(self, sam3_node, one_face):
        _, aux_stats = sam3_node._run_all_sam3_masks(None, CUR_RGB, one_face, "none", "", 0.3)
        assert aux_stats["counts"] == {0: 0}
        assert aux_stats["unassigned_count"] == 0
        assert float(aux_stats["unassigned_mask"].max()) == 0.0

    def test_headless_body_counts(self, sam3_node, one_face):
        _, aux_stats = sam3_node._run_all_sam3_masks(
            None, CUR_RGB, one_face, "headless_body", "", 0.3)
        assert aux_stats["counts"] == {0: 1}
        assert aux_stats["unassigned_count"] == 0
