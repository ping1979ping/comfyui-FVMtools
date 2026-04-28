"""Tests for the SAM3-text-aux fallback in PersonDataRefiner.

Covers the case where ``aux_model = "none"`` but ``aux_label`` is non-empty
and a ``sam3_model`` is connected — the refiner should run SAM3 grounded
text segmentation, build per-reference aux masks, and inject them into
``person_data["aux_masks"]``.
"""

import sys
from unittest.mock import MagicMock, patch

# The real ``nodes.utils.masker`` does ``from ...parsing import BiSeNet`` —
# resolvable only inside ComfyUI's loader. Pre-stub the module so importing
# ``nodes.person_data_refiner`` succeeds under pytest. We replace it with a
# functional shim that still routes ``run_sam3_grounding`` (the only function
# the new SAM3-text-aux path calls into masker for) so the test can patch
# that single symbol on ``nodes.person_data_refiner``.
if "nodes.utils.masker" not in sys.modules:
    _masker_stub = MagicMock()
    _masker_stub.MaskGenerator = MagicMock
    _masker_stub.ALL_MASK_TYPES = [
        "face", "head", "hair", "body", "facial_skin",
        "eyes", "mouth", "neck", "accessories",
    ]
    _masker_stub.generate_all_masks_for_face = MagicMock()
    _masker_stub.run_sam3_grounding = MagicMock(return_value=[])
    sys.modules["nodes.utils.masker"] = _masker_stub

import numpy as np
import pytest
import torch

try:
    from nodes.person_data_refiner import PersonDataRefiner
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip(f"PersonDataRefiner unavailable in this test environment: {e}",
                allow_module_level=True)


def _make_person_data(num_refs=2, h=64, w=64, batch_size=1):
    """Minimal PERSON_DATA fixture with two non-overlapping body masks."""
    body_a = torch.zeros(batch_size, h, w, dtype=torch.float32)
    body_b = torch.zeros(batch_size, h, w, dtype=torch.float32)
    body_a[:, :, : w // 2] = 1.0
    body_b[:, :, w // 2 :] = 1.0
    return {
        "batch_size": batch_size,
        "num_references": num_refs,
        "image_height": h,
        "image_width": w,
        "matches": [True, True],
        "all_faces_mask": torch.zeros(batch_size, h, w),
        "matched_faces_mask": torch.zeros(batch_size, h, w),
        "per_face_masks": [],
        "face_to_ref": [0, 1],
        "body_masks": [body_a, body_b],
        "face_masks": [torch.zeros(batch_size, h, w) for _ in range(num_refs)],
        "head_masks": [torch.zeros(batch_size, h, w) for _ in range(num_refs)],
    }


def _fake_sam3_two_legs(sam3_config, image_rgb, text_prompt, threshold=0.2):
    """Stub for run_sam3_grounding: emit one mask in the left half and one
    in the right half so each reference picks up exactly one detection."""
    h, w = image_rgb.shape[:2]
    left = np.zeros((h, w), dtype=np.float32)
    right = np.zeros((h, w), dtype=np.float32)
    left[:, : w // 4] = 1.0
    right[:, 3 * w // 4 :] = 1.0
    return [
        (left, 0.9, [0, 0, w // 4, h]),
        (right, 0.85, [3 * w // 4, 0, w, h]),
    ]


def _fake_sam3_no_detections(*_args, **_kwargs):
    return []


def _stub_sam3_config():
    return {"checkpoint_path": "<stub>", "bpe_path": "<stub>"}


def _common_kwargs(person_data, images, sam3_model, label="legs"):
    return dict(
        person_data=person_data,
        images=images,
        mask_fill_holes=True,
        mask_blur=0,
        det_size=480,
        sam_model=None,
        sam3_model=sam3_model,
        depth_map=None,
        depth_edge_threshold=0.05,
        depth_carve_strength=0.8,
        depth_grow_pixels=30,
        aux_model="none",
        aux_confidence=0.35,
        aux_label=label,
        aux_fill_holes=False,
        aux_expand_pixels=0,
        aux_blend_pixels=0,
        aux_yolo_sam_refine=True,
        aux_yolo_sam_bbox_expansion=0,
    )


def test_sam3_text_aux_smart_skip_path_runs():
    """Same-resolution chain → smart-skip path → SAM3-text-aux runs and
    populates per-reference aux_masks."""
    pd = _make_person_data(num_refs=2, h=64, w=64, batch_size=1)
    images = torch.rand(1, 64, 64, 3)
    with patch("nodes.person_data_refiner.run_sam3_grounding",
               side_effect=_fake_sam3_two_legs):
        new_pd, aux_masks_batch, report = PersonDataRefiner().execute(
            **_common_kwargs(pd, images, _stub_sam3_config(), "legs"))

    assert "aux_masks" in new_pd
    assert len(new_pd["aux_masks"]) == 2
    # Each ref should have caught exactly one of the two stub masks
    for ri in range(2):
        m = new_pd["aux_masks"][ri]
        assert m.shape == (1, 64, 64)
        assert m.sum() > 0, f"ref {ri} got empty aux mask"

    # Smart-skip report mentions the SAM3 source
    assert "SAM3" in report or "sam3" in report.lower()


def test_sam3_text_aux_disabled_when_label_empty():
    """No label, no YOLO → aux pipeline must skip entirely."""
    pd = _make_person_data(num_refs=2, h=64, w=64, batch_size=1)
    images = torch.rand(1, 64, 64, 3)
    with patch("nodes.person_data_refiner.run_sam3_grounding",
               side_effect=_fake_sam3_two_legs) as sam3_mock:
        kwargs = _common_kwargs(pd, images, _stub_sam3_config(), label="")
        new_pd, _, report = PersonDataRefiner().execute(**kwargs)
    sam3_mock.assert_not_called()
    # When neither YOLO nor SAM3-text fires, aux_masks stay as in input (no key)
    assert "aux_masks" not in new_pd or new_pd.get("aux_masks") == pd.get("aux_masks")


def test_sam3_text_aux_disabled_when_sam3_missing():
    """Label provided but no SAM3 model → aux pipeline must skip."""
    pd = _make_person_data(num_refs=2, h=64, w=64, batch_size=1)
    images = torch.rand(1, 64, 64, 3)
    with patch("nodes.person_data_refiner.run_sam3_grounding",
               side_effect=_fake_sam3_two_legs) as sam3_mock:
        kwargs = _common_kwargs(pd, images, sam3_model=None, label="legs")
        new_pd, _, _ = PersonDataRefiner().execute(**kwargs)
    sam3_mock.assert_not_called()


def test_sam3_text_aux_handles_no_detections():
    """SAM3 returns nothing → all aux_masks are zero, no exception."""
    pd = _make_person_data(num_refs=2, h=64, w=64, batch_size=1)
    images = torch.rand(1, 64, 64, 3)
    with patch("nodes.person_data_refiner.run_sam3_grounding",
               side_effect=_fake_sam3_no_detections):
        new_pd, _, _ = PersonDataRefiner().execute(
            **_common_kwargs(pd, images, _stub_sam3_config(), "legs"))
    for ri in range(2):
        assert float(new_pd["aux_masks"][ri].sum()) == 0.0


def test_sam3_text_aux_comma_separated_labels_run_as_unions():
    """`aux_label = 'arms,legs'` → SAM3 runs once per fragment, masks union."""
    pd = _make_person_data(num_refs=2, h=64, w=64, batch_size=1)
    images = torch.rand(1, 64, 64, 3)
    call_prompts = []

    def _record_prompts(sam3_config, image_rgb, text_prompt, threshold=0.2):
        call_prompts.append(text_prompt)
        return _fake_sam3_two_legs(sam3_config, image_rgb, text_prompt, threshold)

    with patch("nodes.person_data_refiner.run_sam3_grounding",
               side_effect=_record_prompts):
        PersonDataRefiner().execute(
            **_common_kwargs(pd, images, _stub_sam3_config(), "arms,legs"))

    assert call_prompts == ["arms", "legs"], (
        f"expected one SAM3 call per label fragment, got {call_prompts}"
    )
