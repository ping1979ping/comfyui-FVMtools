"""Polygon masks, and the two-sided failure they exist to avoid.

An axis-aligned box on a tilted surface is wrong in both directions at once, and
each direction has already produced a defect in this project:

* Too SMALL — the box clips the end of a line of text. Outside the mask the
  original scribble is restored byte for byte, so a clipped box guarantees a
  clipped word (`AHMENWERK`, `ANTIQUARIA`, `KUNSTAREALBL`).
* Too LARGE — the box takes in surroundings the model then repaints. A flat,
  even background grew a soft drop shadow around a sign that has none anywhere.

The quad is the only shape that forces neither.
"""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "live", "ideogram"),
)
masks = pytest.importorskip("masks")

# The synthetic scene's sign: a parallelogram that drops 60 px across the frame.
PLATTE = [(147, 83), (708, 143), (704, 252), (149, 227)]


class TestPolygonMask:
    def test_the_quad_is_filled(self):
        m = masks.soft((300, 800), [("p", PLATTE)], feather=0)
        assert m[150, 400] == pytest.approx(1.0)

    def test_the_corner_the_bounding_box_would_have_taken_is_not(self):
        """Top-right of the bounding box is background, not sign — this is the
        ground the model was repainting into a drop shadow."""
        m = masks.soft((300, 800), [("p", PLATTE)], feather=0)
        assert m[88, 700] == pytest.approx(0.0)
        assert m[248, 155] == pytest.approx(0.0)

    def test_a_box_would_have_covered_those_corners(self):
        """The counter-check, so the test above is about the quad and not about
        the coordinates happening to miss."""
        xs = [p[0] for p in PLATTE]
        ys = [p[1] for p in PLATTE]
        box = masks.soft(
            (300, 800), [("r", min(xs), min(ys), max(xs), max(ys))], feather=0
        )
        assert box[88, 700] == pytest.approx(1.0)
        assert box[248, 155] == pytest.approx(1.0)

    def test_polygons_and_boxes_mix_in_one_call(self):
        m = masks.soft((300, 800), [("p", PLATTE), ("r", 10, 260, 60, 290)], feather=0)
        assert m[150, 400] == pytest.approx(1.0)
        assert m[275, 35] == pytest.approx(1.0)

    def test_hardening_still_zeroes_the_far_field(self):
        m = masks.harden(masks.soft((300, 800), [("p", PLATTE)], feather=8))
        assert m[5, 5] == 0.0
        assert m[295, 795] == 0.0

    def test_a_triangle_works_too(self):
        """Not every found shape is four-sided; nothing here should assume it."""
        m = masks.soft((100, 100), [("p", [(10, 10), (90, 10), (10, 90)])], feather=0)
        assert m[20, 20] == pytest.approx(1.0)
        assert m[80, 80] == pytest.approx(0.0)


class TestShapeOf:
    """`pipeline.shape_of` must prefer the quad whenever one is present."""

    def test_quad_wins(self):
        pipeline = pytest.importorskip("pipeline")
        kind, pts = pipeline.shape_of(
            {"box": (0, 0, 10, 10), "kind": "r", "quad": PLATTE}
        )
        assert kind == "p" and pts == PLATTE

    def test_box_when_there_is_no_quad(self):
        pipeline = pytest.importorskip("pipeline")
        assert pipeline.shape_of({"box": (1, 2, 3, 4), "kind": "e"}) == (
            "e",
            1,
            2,
            3,
            4,
        )

    def test_none_quad_is_not_a_quad(self):
        """regions.py returns `quad: None` when it only found a box; treating
        that as a polygon would build an empty mask and silently skip a region."""
        pipeline = pytest.importorskip("pipeline")
        assert pipeline.shape_of({"box": (1, 2, 3, 4), "kind": "r", "quad": None}) == (
            "r",
            1,
            2,
            3,
            4,
        )
