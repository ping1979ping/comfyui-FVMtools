"""Hole filling must close segmentation gaps without swallowing ring masks.

The bug this pins: RETR_EXTERNAL + drawContours(FILLED) discards every inner
contour, so a hair mask that encircles the face had the face as its hole — and
the fill turned hair into head. The detailer then refined the face too.
"""
import cv2
import numpy as np
import pytest
import torch

from nodes.utils.mask_utils import fill_mask_holes_2d
from nodes.utils.tensor_utils import fill_mask_holes

H, W = 640, 512
FACE = ((256, 330), (110, 165))   # centre, axes


def _hair_ring():
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (256, 330), (170, 240), 0, 0, 360, 255, -1)
    cv2.ellipse(m, *FACE, 0, 0, 360, 0, -1)
    return torch.from_numpy(m.astype(np.float32) / 255.0)


def _face_with_speckle():
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, *FACE, 0, 0, 360, 255, -1)
    for cx, cy in [(220, 300), (290, 340), (256, 400)]:
        cv2.circle(m, (cx, cy), 9, 0, -1)
    return torch.from_numpy(m.astype(np.float32) / 255.0)


def _face_coverage(mask_np):
    probe = np.zeros((H, W), np.uint8)
    cv2.ellipse(probe, *FACE, 0, 0, 360, 255, -1)
    cv2.ellipse(probe, (256, 330), (80, 130), 0, 0, 360, 0, -1)  # ignore the rim
    inner = np.zeros((H, W), np.uint8)
    cv2.ellipse(inner, (256, 330), (80, 130), 0, 0, 360, 255, -1)
    return float(mask_np[inner > 0].mean())


@pytest.mark.parametrize("fill", [
    lambda m: fill_mask_holes_2d(m).numpy(),
    lambda m: fill_mask_holes(m.unsqueeze(0))[0].numpy(),
])
class TestHoleFill:
    def test_ring_keeps_its_hole(self, fill):
        """A hair mask around a face must not become a head mask."""
        assert _face_coverage(fill(_hair_ring())) == pytest.approx(0.0, abs=1e-6)

    def test_ring_area_is_preserved(self, fill):
        before = _hair_ring()
        after = fill(before)
        ratio = (after > 0.5).sum() / float((before.numpy() > 0.5).sum())
        assert 0.95 <= ratio <= 1.05, f"ring area changed by {(ratio - 1) * 100:.0f}%"

    def test_small_gaps_still_close(self, fill):
        before = _face_with_speckle()
        after = fill(before)
        assert (after > 0.5).sum() > (before.numpy() > 0.5).sum(), "speckle holes stayed open"

    def test_solid_mask_is_unchanged(self, fill):
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, *FACE, 0, 0, 360, 255, -1)
        t = torch.from_numpy(m.astype(np.float32) / 255.0)
        assert (fill(t) > 0.5).sum() == pytest.approx((m > 0).sum(), rel=0.01)

    def test_empty_mask_survives(self, fill):
        out = fill(torch.zeros(64, 64, dtype=torch.float32))
        assert float(out.max()) == 0.0
