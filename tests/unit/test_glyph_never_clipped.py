"""A word is never cut off.

The fit loop in :func:`render_glyph_layer` used to be a BUDGET: five attempts,
and whatever stood at the end was used, silhouette or no silhouette. The clip to
the mask that follows then took the ends off the word. Measured on the saved
selector masks of the street scene, with the legibility floor on:

    st_0_mask2 (a bowed painted shop window, contour mode)
        after five attempts 0.54 of the ink still outside,
        0.47 of the word removed by the clip
    st_0_mask0 (a shop front, contour mode)
        0.065 outside, 0.095 of the word removed

`SBURGU`, `SCHOENBURGE` and `LVANER` are what a census reads a result like that
back as, and it cannot tell a half word from invented text - rightly, because on
the page there is no difference.

So the walk is now a PROMISE. It steps the type down until the letters stand
inside, and when nothing that is still lettering does, it hands back an empty
layer so the caller can put the region through ``too_small_policy``. What it
never does is paint part of a word.

The promise is deliberately tied to ``min_legible_px > 0``. With the floor off
the type is already no larger than the lettering that was there before, the
margin walk is the only lever, and its reading is polluted by the warp border -
so that path is left exactly as it was, and a floor-off run is unchanged to the
byte. The tests below pin both halves.
"""

import glob
import os

import cv2
import numpy as np
import pytest

from nodes.utils.glyph import (
    GLYPH_CLIP_TOLERANCE,
    GLYPH_MAX_LINES,
    GLYPH_SHRINK_ATTEMPTS,
    GLYPH_SHRINK_STEP,
    MIN_FONT_SIZE,
    _fit_text,
    render_glyph_layer,
    resolve_font,
)

LIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live")
FACE = resolve_font("sans")
INK, PLATE = (255, 255, 255), (0, 0, 0)
LONG = "WEISSBURGUNDER"


# ── shapes ──


def rect_mask(w=260, h=90, canvas=(200, 400)):
    """A plain plate: the quad describes it exactly, so nothing falls outside."""
    m = np.zeros(canvas, np.float32)
    y0 = (canvas[0] - h) // 2
    x0 = (canvas[1] - w) // 2
    m[y0 : y0 + h, x0 : x0 + w] = 1.0
    return m


def bowed_mask(w=150, h=54, bow=16, canvas=(160, 260)):
    """A narrow label wrapped round a bottle: both long edges bow the same way."""
    m = np.zeros(canvas, np.float32)
    y0 = (canvas[0] - h) // 2
    x0 = (canvas[1] - w) // 2
    for i in range(w):
        t = (i - (w - 1) / 2.0) / ((w - 1) / 2.0)
        drop = int(round(bow * (1.0 - t * t)))
        m[y0 + drop : y0 + h + drop, x0 + i] = 1.0
    return m


def ring_mask(canvas=(200, 220), radius=95, thickness=26):
    """A hoop sign - lettering runs round a rim with nothing in the middle.

    The centre of its bounding quad lies in the HOLE, so every setting, however
    small, lands where there is no sign at all: measured, the un-floored path
    puts 100% of the word outside the silhouette and the clip then removes all
    of it. Shrinking cannot rescue a shape like this - a smaller setting sits
    more firmly outside, not less - which is exactly why the walk has to be
    allowed to fail instead of using its last attempt regardless.
    """
    m = np.zeros(canvas, np.float32)
    yy, xx = np.mgrid[0 : canvas[0], 0 : canvas[1]].astype(np.float32)
    d = np.hypot(yy - canvas[0] / 2.0, xx - canvas[1] / 2.0)
    m[(d > radius - thickness) & (d < radius)] = 1.0
    return m


# ── measurement ──


def letters(rgb):
    """The typeset letters, clipped or not.

    ``render_glyph_layer`` clips ALPHA to the mask but leaves RGB alone, so the
    returned rgb still carries the whole setting. Against a black plate the
    warp's own border is black too, so anything bright is a letter.
    """
    return np.asarray(rgb, np.float32).max(axis=2) > 0.5


def cut_share(rgb, mask):
    """Share of the typeset letters the clip to the mask takes off."""
    ink = letters(rgb)
    total = float(ink.sum())
    if total <= 0:
        return 0.0
    inside = np.asarray(mask, np.float32)
    if float(inside.max()) > 1.0:
        inside = inside / 255.0
    return float((ink & ~(inside > 0.5)).sum()) / total


def is_empty(alpha):
    return float(np.asarray(alpha).max()) <= 0.0


def layer(text, mask, floor, **kw):
    return render_glyph_layer(
        text, mask, font_path=FACE, fill=INK, bg=PLATE, min_legible_px=floor, **kw
    )


def real_masks(tag):
    return sorted(glob.glob(os.path.join(LIVE, tag + "_0_mask*.png")))


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        pytest.skip("missing " + path)
    ys, xs = np.where(m > 127)
    return m[max(0, ys.min() - 4) : ys.max() + 5, max(0, xs.min() - 4) : xs.max() + 5]


# ── the promise ──


class TestNoWordIsEverCutOff:
    """The core case: a long word on a narrow, bowed silhouette."""

    @pytest.mark.parametrize("text", [LONG, "SCHOENBURGER", "SILVANER"])
    def test_a_long_word_on_a_bowed_label_loses_no_ink_to_the_clip(self, text):
        mask = bowed_mask()
        rgb, alpha = layer(text, mask, 12, target_line_height=14)
        assert is_empty(alpha) or cut_share(rgb, mask) <= GLYPH_CLIP_TOLERANCE

    def test_it_is_set_smaller_rather_than_dropped(self):
        """The bowed label CAN carry the word - so it gets one, just smaller."""
        mask = bowed_mask()
        rgb, alpha = layer(LONG, mask, 12, target_line_height=14)
        assert not is_empty(alpha), "a label this size still carries the word"
        roomy = layer(LONG, rect_mask(), 12, target_line_height=14)
        assert letters(rgb).sum() < letters(roomy[0]).sum(), (
            "the narrow silhouette should have forced a smaller setting"
        )

    def test_every_typeset_letter_is_actually_painted(self):
        """Alpha carries the whole word, not the part that happened to fit."""
        mask = bowed_mask()
        rgb, alpha = layer(LONG, mask, 12, target_line_height=14)
        ink = letters(rgb)
        assert ink.sum() > 0
        carried = float((np.asarray(alpha)[ink] > 0.0).sum()) / float(ink.sum())
        assert carried >= 1.0 - GLYPH_CLIP_TOLERANCE

    def test_the_text_is_never_shortened_to_make_it_fit(self):
        """Wrapping and shrinking keep every character; nothing is dropped."""
        for box_w in (40, 80, 160, 320, 640):
            for box_h in (20, 60, 140):
                _font, lines = _fit_text(
                    "MITTAGESSEN IM HOF",
                    float(box_w),
                    float(box_h),
                    FACE,
                    GLYPH_MAX_LINES,
                )
                assert " ".join(lines) == "MITTAGESSEN IM HOF"
                assert len(lines) <= GLYPH_MAX_LINES

    def test_a_single_word_is_never_split(self):
        _font, lines = _fit_text(LONG, 30.0, 200.0, FACE, GLYPH_MAX_LINES)
        assert lines == [LONG]


class TestExhaustionDoesNotClip:
    """When the walk runs out, the answer is nothing - not half a word."""

    def test_a_hoop_sign_gives_back_an_empty_layer(self):
        mask = ring_mask()
        _rgb, alpha = layer(LONG, mask, 12, target_line_height=18)
        assert is_empty(alpha)

    def test_the_hoop_really_is_a_case_that_cannot_be_shrunk_into(self):
        """Not a vacuous test: no size stands inside, so giving up is the answer.

        Measured on the un-floored path, which sets the type at whatever height
        it is handed and does not shrink: from 6px to 30px the word lands wholly
        in the hole every time.
        """
        mask = ring_mask()
        best = min(
            cut_share(layer(LONG, mask, 0, target_line_height=h)[0], mask)
            for h in (6, 10, 18, 30)
        )
        assert best > GLYPH_CLIP_TOLERANCE

    @pytest.mark.parametrize("text", [LONG, "APOTHEKE", "MITTAGESSEN IM HOF"])
    def test_nothing_at_all_is_painted_rather_than_a_fragment(self, text):
        mask = ring_mask()
        rgb, alpha = layer(text, mask, 12, target_line_height=18)
        assert is_empty(alpha)
        assert not (letters(rgb) & (np.asarray(alpha) > 0.0)).any()

    def test_the_walk_is_bounded(self):
        """GLYPH_SHRINK_ATTEMPTS steps take a 700px cap under MIN_FONT_SIZE."""
        assert 700 * (GLYPH_SHRINK_STEP**GLYPH_SHRINK_ATTEMPTS) < MIN_FONT_SIZE

    def test_a_roomy_plate_is_still_written_on(self):
        """The give-up path must not swallow the ordinary case."""
        mask = rect_mask()
        rgb, alpha = layer(LONG, mask, 12, target_line_height=20)
        assert not is_empty(alpha)
        assert cut_share(rgb, mask) <= GLYPH_CLIP_TOLERANCE


class TestTheFloorOffPathIsUntouched:
    """min_legible_px=0 keeps the old behaviour, clipping included."""

    def test_the_hoop_still_gets_a_layer_without_the_floor(self):
        """The un-floored walk still lays the block out and still returns it."""
        mask = ring_mask()
        rgb, _alpha = layer(LONG, mask, 0, target_line_height=18)
        assert letters(rgb).any(), "the un-floored margin walk must be unchanged"

    def test_and_it_still_clips_there(self):
        """Stated, not hidden: without the floor the old outcome is kept."""
        mask = ring_mask()
        rgb, _alpha = layer(LONG, mask, 0, target_line_height=18)
        assert cut_share(rgb, mask) > GLYPH_CLIP_TOLERANCE

    def test_the_default_is_the_floor_off(self):
        mask = ring_mask()
        a = layer(LONG, mask, 0, target_line_height=18)
        b = render_glyph_layer(
            LONG, mask, font_path=FACE, fill=INK, bg=PLATE, target_line_height=18
        )
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ── the real selector masks ──


class TestOnRealSelectorMasks:
    @pytest.mark.parametrize("path", real_masks("st") or ["missing"])
    def test_no_street_region_comes_back_with_part_of_a_word(self, path):
        if path == "missing":
            pytest.skip("no saved street masks")
        mask = load_mask(path)
        rgb, alpha = layer("GOLDSCHMIED", mask, 12, target_line_height=16)
        assert is_empty(alpha) or cut_share(rgb, mask) <= GLYPH_CLIP_TOLERANCE

    @pytest.mark.parametrize("path", real_masks("sh") or ["missing"])
    def test_the_wine_labels_are_unaffected_and_still_written_on(self, path):
        if path == "missing":
            pytest.skip("no saved shelf masks")
        mask = load_mask(path)
        rgb, alpha = layer(LONG, mask, 12, target_line_height=18)
        assert not is_empty(alpha), "a bottle label carries its word as before"
        assert cut_share(rgb, mask) <= GLYPH_CLIP_TOLERANCE
