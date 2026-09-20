"""The two mask rules that were derived from pictures, pinned so they stay.

Both are cheap to break by accident and expensive to rediscover — each cost a
measured run to find:

* `harden` exists because the soft mask rim rewrites pixels where it is
  invisible. Measured on a ten-region street pass: 4678 pixels changed outside
  the intended area. That is nothing in one frame and everything in a chain.
* `disjoint_groups` exists because two OVERLAPPING text boxes in one Ideogram
  caption garble each other. Seen directly: `KAESE` and `TABAK` came out clean
  while the sign wedged between them came back as nonsense.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "live", "ideogram"),
)
masks = pytest.importorskip("masks")


class TestHarden:
    def test_tail_becomes_exactly_zero(self):
        """The whole point: below the tail it must be 0.0, not merely small.

        `ImageCompositeMasked` computes orig*(1-m) + new*m, so m=0.004 still
        rewrites the pixel. Only an exact zero leaves the source byte alone.
        """
        m = np.array([[0.0, 0.004, 0.01, 0.02]], np.float32)
        assert masks.harden(m).tolist() == [[0.0, 0.0, 0.0, 0.0]]

    def test_full_coverage_survives(self):
        assert masks.harden(np.ones((3, 3), np.float32)).min() == pytest.approx(1.0)

    def test_middle_stays_a_gradient(self):
        """Not a threshold. A step at the rim is the hard edge the soft mask
        exists to avoid — DifferentialDiffusion reads the gradient as a per-pixel
        start time."""
        out = masks.harden(np.array([[0.3, 0.5, 0.7]], np.float32))[0]
        assert 0.0 < out[0] < out[1] < out[2] < 1.0

    def test_never_leaves_the_unit_range(self):
        m = np.array([[-0.5, 0.5, 1.5]], np.float32)
        out = masks.harden(m)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_a_real_feathered_blob_is_zero_far_away(self):
        m = masks.soft((60, 60), [("r", 20, 20, 40, 40)], feather=6)
        assert m[0, 0] > 0.0 or True  # blur may or may not reach; either is fine
        assert masks.harden(m)[0, 0] == 0.0  # but hardened it must not


class TestDisjointGroups:
    def test_overlapping_boxes_never_share_a_group(self):
        boxes = [(0, 0, 50, 50), (25, 25, 75, 75), (60, 60, 90, 90)]
        groups = masks.disjoint_groups(boxes, size=5)
        for g in groups:
            for i in g:
                for j in g:
                    if i != j:
                        assert not masks._overlaps(boxes[i], boxes[j], 2)

    def test_disjoint_boxes_may_share_a_group(self):
        boxes = [(0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50)]
        assert masks.disjoint_groups(boxes, size=5) == [[0, 1, 2]]

    def test_size_is_respected(self):
        boxes = [(i * 20, 0, i * 20 + 10, 10) for i in range(7)]
        assert all(len(g) <= 3 for g in masks.disjoint_groups(boxes, size=3))

    def test_every_box_lands_exactly_once(self):
        boxes = [
            (0, 0, 50, 50),
            (25, 25, 75, 75),
            (10, 10, 60, 60),
            (200, 200, 210, 210),
        ]
        got = sorted(i for g in masks.disjoint_groups(boxes, size=2) for i in g)
        assert got == [0, 1, 2, 3]

    def test_the_street_row_splits_the_way_the_picture_demanded(self):
        """The actual failure: three signs in a receding row, the middle one
        overlapping both neighbours, came back as nonsense while the outer two
        were clean."""
        blumen, kaese, brot = (
            (268, 262, 372, 348),
            (198, 320, 288, 395),
            (138, 352, 218, 412),
        )
        groups = masks.disjoint_groups([blumen, kaese, brot], size=5)
        home = {i: gi for gi, g in enumerate(groups) for i in g}
        assert home[0] != home[1], "BLUMEN und KAESE ueberlappen"
        assert home[1] != home[2], "KAESE und BROT ueberlappen"

    def test_empty_input(self):
        assert masks.disjoint_groups([], size=5) == []


class TestSoft:
    def test_ellipse_and_rectangle_both_fill(self):
        m = masks.soft(
            (100, 100), [("e", 10, 10, 40, 40), ("r", 60, 60, 90, 90)], feather=0
        )
        assert m[25, 25] == pytest.approx(1.0)
        assert m[75, 75] == pytest.approx(1.0)
        assert m[50, 50] == pytest.approx(0.0)

    def test_feather_softens_the_rim_without_leaving_the_range(self):
        m = masks.soft((100, 100), [("r", 30, 30, 70, 70)], feather=8)
        assert m.max() <= 1.0 and m.min() >= 0.0
        assert 0.0 < m[30, 50] < 1.0, "Rand muss ein Verlauf sein, keine Kante"
