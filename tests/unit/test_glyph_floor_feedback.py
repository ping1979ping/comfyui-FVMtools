"""The legibility floor fed back through itself: the suite runs A -> B -> C.

Every pass reads the lettering the pass before it wrote. `measure_ink_height`
hands that back as `target_line_height`, and `_fit_text` turns it into a point
size — so the conversion between the two is a LOOP GAIN, not a formatting
detail. The rule of thumb it used, `size = height * 1.35`, is 1/0.74 and only
reproduces the size it was handed on a face whose capitals stand at 0.74 of the
point size. Measured over the faces on this machine, cap/size runs 0.65 to 0.80,
so the round trip loses 12 % per pass on one and gains 6-10 % on the other.

The gain is invisible while the type is small — integer rounding pins it — and
comes alive once something lifts the type, which is exactly what the floor does.
With a 12 px floor, Impact ratchets 15, 16, 18, 19, 20, 22, 23, 24 px and only
stops when the box binds, at which point the word stands as tall as the whole
plate. That is the `MITTAGESSEN` blown up to sheet height and the grey oversized
`WEISSBURG` standing over a sharp smaller setting of the same word.

These tests pin three things: the loop is a fixed point, the setting stands
wholly inside the region rather than being clipped at its edge, and a run with
`min_legible_px=0` is unchanged to the pixel.
"""

import numpy as np
import pytest
import cv2
import torch
from unittest.mock import patch

from nodes.utils import glyph as G
from nodes.utils.glyph import (
    MIN_FONT_SIZE,
    GLYPH_MAX_LINES,
    _cap_height,
    _fit_text,
    _load_font,
    _size_for_cap_height,
    existing_ink_mask,
    estimate_text_colors,
    glyph_ink_coverage,
    measure_ink_height,
    render_glyph_layer,
    render_text_block,
    resolve_font,
    reconstruct_surface,
    text_band,
)
from nodes.signs.detailer import SignDetailer


# Faces with deliberately different cap ratios. The condensed/display bucket is
# the one that ratchets; the others decay. A test that only saw one of them
# would have called the loop stable.
FACE_HINTS = ["clean sans-serif", "serif", "condensed", "mono", "bold"]
FACES = [resolve_font(h) for h in FACE_HINTS]
FACE_IDS = ["sans", "serif", "condensed", "mono", "bold"]

WORD = "WEISSBURGUNDER"


def _old_bound(inner_h, target_line_height):
    """The cap as it stood before this change — the byte-identity reference."""
    high = max(MIN_FONT_SIZE, int(inner_h) + 2)
    return max(MIN_FONT_SIZE, min(high, int(round(target_line_height * 1.35))))


def _sized(font_path, inner_w, inner_h, target_line_height, min_legible_px, text=WORD):
    font, _lines = _fit_text(
        text,
        inner_w,
        inner_h,
        font_path,
        GLYPH_MAX_LINES,
        target_line_height=target_line_height,
        min_legible_px=min_legible_px,
    )
    return font.size


# ── scenes ──


def label_scene(w=240, h=64, text=WORD, cap_px=6, font_path=None, pad=20):
    """A light plate carrying small dark lettering, with a darker surround."""
    from PIL import Image, ImageDraw

    img = np.full((h + 2 * pad, w + 2 * pad, 3), 190, np.uint8)
    img[pad : pad + h, pad : pad + w] = 234
    size = _size_for_cap_height(font_path, cap_px, 200)
    font = _load_font(font_path, size)
    pil = Image.fromarray(img)
    bbox = G._line_bbox(font, text)
    ImageDraw.Draw(pil).text(
        (pad + (w - (bbox[2] - bbox[0])) / 2.0 - bbox[0], pad + h / 2.0 - cap_px),
        text,
        font=font,
        fill=(30, 30, 32),
    )
    mask = np.zeros(np.array(pil).shape[:2], np.float32)
    mask[pad : pad + h, pad : pad + w] = 1.0
    return np.array(pil).astype(np.float32) / 255.0, mask


def diamond_mask(h=70, w=260):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(
        m,
        [
            np.array(
                [[w // 2, 2], [w - 3, h // 2], [w // 2, h - 3], [2, h // 2]], np.int32
            )
        ],
        1,
    )
    return m.astype(np.float32)


def tilted_mask(h=90, w=260, deg=25):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(
        m,
        [
            cv2.boxPoints(((w / 2.0, h / 2.0), (w * 0.78, h * 0.42), deg)).astype(
                np.int32
            )
        ],
        1,
    )
    return m.astype(np.float32)


def ink_on_rim(mask, font_path, target_line_height, min_legible_px, text=WORD):
    """Share of the new lettering sitting ON the region outline, plus its area.

    Black plate on purpose. The warp fades its own plate to black at the edge of
    the quad, and against a light plate every pixel of that fade reads as ink —
    which is what made the fitting loop chase a number that could not move. On a
    black plate the fade is black to black and only the letters answer.
    """
    rgb, alpha = render_glyph_layer(
        text,
        mask,
        font_path=font_path,
        fill=(255, 255, 255),
        bg=(0, 0, 0),
        uppercase=True,
        target_line_height=target_line_height,
        min_legible_px=min_legible_px,
    )
    coverage = glyph_ink_coverage(rgb, alpha, (0, 0, 0), (255, 255, 255))
    strokes = coverage > 0.15
    inside = (mask > 0.5).astype(np.uint8)
    rim = (inside > 0) & (
        cv2.erode(inside, np.ones((3, 3), np.uint8), iterations=1) == 0
    )
    total = float(strokes.sum())
    return (float((strokes & rim).sum()) / total if total else 0.0), int(total)


# ──────────────────────────────────────────────────────────────────────────────
# 1. The loop must not ratchet. This is the regression the change exists for.
# ──────────────────────────────────────────────────────────────────────────────


class TestTheLoopIsAFixedPoint:
    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    @pytest.mark.parametrize("size", [10, 15, 20, 28, 40])
    def test_a_measured_cap_maps_back_to_at_most_its_own_size(self, font_path, size):
        """The round trip size -> cap -> size may hold or shrink, never grow."""
        cap = _cap_height(_load_font(font_path, size))
        back = _size_for_cap_height(font_path, cap, 200)
        assert back <= size, (
            "converting a measured cap height back to a point size must not "
            "overshoot, or every pass sets the word larger than the last"
        )

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_repeated_passes_do_not_grow_the_type(self, font_path):
        """Feed each pass the cap height the pass before it set. Eight passes."""
        sizes = [7]
        for _ in range(8):
            cap = _cap_height(_load_font(font_path, sizes[-1]))
            sizes.append(_sized(font_path, 240, 74, cap, 12))
        settled = sizes[2:]
        assert settled == sorted(settled, reverse=True), (
            "the type grew from pass to pass: " + repr(sizes)
        )
        assert sizes[-1] <= sizes[1], (
            "eight passes must not end larger than the first floored one: "
            + repr(sizes)
        )

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_the_rule_of_thumb_never_undershoots_the_exact_conversion(self, font_path):
        """1.35 is only right at a cap ratio of 0.74; above it, it overshoots."""
        size = 20
        cap = _cap_height(_load_font(font_path, size))
        rule = int(round(cap * 1.35))
        exact = _size_for_cap_height(font_path, cap, 200)
        assert exact <= size
        if cap / size > 1.0 / 1.35:
            assert rule >= size, "this face is the ratcheting kind, by construction"

    def test_at_least_one_installed_face_would_have_ratcheted(self):
        """The bug is not hypothetical — it needs a face that actually overshoots."""
        offenders = []
        for hint, path in zip(FACE_IDS, FACES):
            for size in (15, 20, 28, 34):
                cap = _cap_height(_load_font(path, size))
                if int(round(cap * 1.35)) > size:
                    offenders.append((hint, size, cap))
                    break
        assert offenders, (
            "no face on this machine overshoots, so the regression above cannot "
            "be demonstrated here: "
            + repr(
                [(h, _cap_height(_load_font(p, 20))) for h, p in zip(FACE_IDS, FACES)]
            )
        )

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_the_type_reproduces_the_measured_height(self, font_path):
        """A roomy box, a floor well below: the setting comes back the size it read."""
        for wanted in (14, 20, 30):
            size = _sized(font_path, 900, 120, wanted, 8)
            assert abs(_cap_height(_load_font(font_path, size)) - wanted) <= 1

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_the_picture_does_not_grow_when_a_pass_reads_its_own_output(
        self, font_path
    ):
        """End to end: typeset, composite, measure the result, typeset again."""
        img, mask = label_scene(font_path=font_path)
        heights, sizes = [], []
        for _ in range(3):
            u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            ink_mask = existing_ink_mask(u8, mask, (128, 128, 128))
            ink, plate = estimate_text_colors(u8, mask, ink_mask=ink_mask)
            measured = measure_ink_height(u8, mask, plate, ink_mask=ink_mask)
            heights.append(measured)
            quad = G.mask_quad(mask)
            bw, bh = G.quad_size(quad)
            sizes.append(_sized(font_path, bw * 0.84, bh * 0.84, measured, 12))
            rgb, alpha = render_glyph_layer(
                WORD,
                mask,
                font_path=font_path,
                fill=ink,
                bg=plate,
                uppercase=True,
                target_line_height=measured,
                min_legible_px=12,
            )
            coverage = glyph_ink_coverage(rgb, alpha, plate, ink)
            band = text_band(
                mask, old_ink=ink_mask, new_ink=(coverage > 0.15), line_height=measured
            )
            base = img.astype(np.float32)
            if band is not None:
                surface = reconstruct_surface(img, band)
                if surface is not None:
                    keep = band[..., None].astype(np.float32)
                    base = base * (1.0 - keep) + surface * keep
            img = G.composite_glyph(base, rgb, coverage, strength=1.0)
        assert sizes[2] <= sizes[1], (
            "the second re-read set the word larger again: " + repr(sizes)
        )
        assert heights[2] is not None and heights[1] is not None
        assert heights[2] <= heights[1] + 1, (
            "the lettering measured taller on every pass: " + repr(heights)
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. A floored setting must stand inside the region, not be cut off at its edge.
# ──────────────────────────────────────────────────────────────────────────────


class TestNothingIsClipped:
    @pytest.mark.parametrize(
        "shape,mask",
        [("diamond", diamond_mask()), ("tilted", tilted_mask())],
        ids=["diamond", "tilted"],
    )
    def test_the_floored_setting_keeps_off_the_outline(self, shape, mask):
        """Either the word stands clear of the outline, or it is not drawn.

        Empty is a legitimate answer since the step-down stops at the floor: a
        shape that cannot carry the word at a readable size hands the region to
        too_small_policy instead of setting 5px mush the census reads back as an
        invented string. What must never happen is ink ON the outline, because
        that is where the clip to the mask takes the ends off the word.
        """
        share, area = ink_on_rim(mask, FACES[0], 6, 12)
        assert share <= 0.01, (
            "letters are sitting on the region outline, which is where the clip "
            "to the mask cuts them off: " + repr(share)
        )

    def test_shrinking_is_what_keeps_it_off_the_outline(self):
        """Disable the step-down and the same case can no longer be rescued.

        It used to come back with ink on the rim, because the loop spent its
        five attempts and then used the last one whatever it looked like. It now
        comes back EMPTY instead: a setting that will not stand inside is not
        drawn at all, and the region falls through to too_small_policy. Either
        way the step-down is what makes the difference between a written region
        and none, so the test still tells the two apart.
        """
        mask = diamond_mask()
        with_shrink, _area = ink_on_rim(mask, FACES[0], 6, 12)
        with patch.object(G, "GLYPH_SHRINK_STEP", 1.0):
            without, without_area = ink_on_rim(mask, FACES[0], 6, 12)
        assert with_shrink <= 0.01
        assert without_area == 0, (
            "the step-down is doing nothing; the test would pass for the wrong "
            "reason: " + repr((area, without_area))
        )

    @pytest.mark.parametrize(
        "shape,mask",
        [("diamond", diamond_mask()), ("tilted", tilted_mask())],
        ids=["diamond", "tilted"],
    )
    def test_the_floor_still_sets_larger_than_no_floor(self, shape, mask):
        """Where a word IS drawn under the floor, it must be drawn bigger.

        The floor may also decline the region outright — the step-down stops at
        it rather than walking below into unreadable type. What it must never do
        is quietly set the word at the un-floored size and call that a fit,
        because that is the 4-5px setting the floor exists to prevent.
        """
        _floored_share, floored = ink_on_rim(mask, FACES[0], 6, 12)
        _plain_share, plain = ink_on_rim(mask, FACES[0], 6, 0)
        if floored == 0:
            return  # declined: the region goes to too_small_policy, tested there
        assert floored > plain * 1.15, "the floor was shrunk away to nothing: " + repr(
            (plain, floored)
        )

    def test_a_setting_that_fits_is_left_at_full_size(self):
        """A rectangle with room to spare must not be stepped down at all."""
        mask = np.zeros((90, 300), np.float32)
        mask[8:82, 10:290] = 1.0
        share, area = ink_on_rim(mask, FACES[0], 6, 12)
        assert share == 0.0
        _s, unbounded = ink_on_rim(mask, FACES[0], 6, 0)
        assert area > unbounded, "the floor was not applied on a region that carries it"


# ──────────────────────────────────────────────────────────────────────────────
# 3. A floor is a floor. Once the ink clears it, it has nothing left to say.
# ──────────────────────────────────────────────────────────────────────────────


class TestTheFloorOnlyEverLifts:
    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_any_floor_below_the_measured_height_gives_the_same_size(self, font_path):
        """Old ink already above the floor: the floor makes no difference at all."""
        measured = 26
        sizes = {
            floor: _sized(font_path, 900, 120, measured, floor)
            for floor in (1, 4, 8, 12, 20, 25)
        }
        assert len(set(sizes.values())) == 1, (
            "the floor changed the size while the ink was already above it: "
            + repr(sizes)
        )

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_a_floor_above_the_measured_height_does_lift(self, font_path):
        small = _sized(font_path, 900, 120, 5, 0)
        lifted = _sized(font_path, 900, 120, 5, 16)
        assert lifted > small

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_the_box_still_has_the_last_word(self, font_path):
        """A floor larger than the plate must not push the word past the edge."""
        font, lines = _fit_text(
            WORD,
            70,
            14,
            font_path,
            GLYPH_MAX_LINES,
            target_line_height=4,
            min_legible_px=40,
        )
        widest = max(G._line_width(font, line) for line in lines)
        assert widest <= 70 or font.size == MIN_FONT_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# 4. min_legible_px = 0 is the pinned baseline. It must not move.
# ──────────────────────────────────────────────────────────────────────────────


class TestFloorOffIsUnchanged:
    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    @pytest.mark.parametrize("measured", [4, 6, 12, 22, 40])
    def test_the_cap_is_still_the_rule_of_thumb(self, font_path, measured):
        # A box with room to spare on purpose. The binary search takes the
        # largest size that fits, so a tight box would hide the cap behind its
        # own limit and the test would pass whatever the cap said.
        size = _sized(font_path, 1600, 400, measured, 0)
        assert size == _old_bound(400, measured), (
            "the baseline conversion moved; the next live run would measure two "
            "changes at once"
        )

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_the_font_is_never_asked_about_cap_heights(self, font_path):
        """Byte identity by construction: the new machinery is never reached."""
        with patch.object(
            G,
            "_size_for_cap_height",
            side_effect=AssertionError("consulted at min_legible_px=0"),
        ):
            _fit_text(
                WORD,
                400,
                90,
                font_path,
                GLYPH_MAX_LINES,
                target_line_height=7,
                min_legible_px=0,
            )
            render_text_block(
                WORD,
                300,
                70,
                font_path=font_path,
                target_line_height=7,
                min_legible_px=0,
            )
            render_glyph_layer(
                WORD,
                diamond_mask(),
                font_path=font_path,
                target_line_height=7,
                min_legible_px=0,
            )

    @pytest.mark.parametrize(
        "shape,mask",
        [("diamond", diamond_mask()), ("tilted", tilted_mask())],
        ids=["diamond", "tilted"],
    )
    def test_the_margin_walk_is_untouched(self, shape, mask):
        """Same layer, pixel for pixel, whichever way the loop is entered."""
        a = render_glyph_layer(
            WORD,
            mask,
            font_path=FACES[0],
            fill=(255, 255, 255),
            bg=(0, 0, 0),
            uppercase=True,
            target_line_height=6,
            min_legible_px=0,
        )
        b = render_glyph_layer(
            WORD,
            mask,
            font_path=FACES[0],
            fill=(255, 255, 255),
            bg=(0, 0, 0),
            uppercase=True,
            target_line_height=6,
        )
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_max_cap_px_defaults_to_off(self):
        plain = render_text_block(
            WORD, 300, 70, font_path=FACES[0], target_line_height=20
        )
        explicit = render_text_block(
            WORD, 300, 70, font_path=FACES[0], target_line_height=20, max_cap_px=None
        )
        assert np.array_equal(plain, explicit)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Separate, smaller fault: the erased patch had a hard rectangular edge.
# ──────────────────────────────────────────────────────────────────────────────


class TestErasedPatchIsBlendedIn:
    @staticmethod
    def _scene(h=192, w=256):
        """A lit plate in a dark surround, carrying dark bars that read as strokes."""
        img = np.full((h, w, 3), 0.12, np.float32)
        img[48:144, 64:192] = np.linspace(0.62, 0.88, 96, dtype=np.float32)[
            :, None, None
        ]
        for i in range(6):
            x = 76 + i * 18
            img[80:104, x : x + 8] = 0.10
        mask = np.zeros((h, w), np.float32)
        mask[48:144, 64:192] = 1.0
        return img, mask

    def _erased(self):
        img, mask = self._scene()
        node = SignDetailer()
        out, band = node._apply_erase(
            torch.from_numpy(img), {"mask": mask, "index": 0, "class": "sign"}
        )
        return img, mask, band, out.cpu().numpy()

    def test_something_was_erased(self):
        _img, _mask, band, _out = self._erased()
        assert band is not None and float(band.sum()) > 0

    def _hard(self, img, band):
        """The blend as it stood: the band used raw, 0 or 1 and nothing between.

        Compared against the blend rather than against the picture, because at
        the band's own outline the rebuilt surface still matches what was there
        — Telea starts from that outline — so a difference measured against the
        picture is smallest exactly where the ramp lives and cannot see it.
        """
        surface = reconstruct_surface(img, band)
        keep = np.asarray(band, np.float32)[..., None]
        return img.astype(np.float32) * (1.0 - keep) + surface * keep

    def test_the_patch_edge_is_a_ramp_not_a_step(self):
        img, _mask, band, out = self._erased()
        differs = np.abs(out - self._hard(img, band)).max(axis=2) > 1e-6
        assert int(differs.sum()) > 0, (
            "the blend is still the raw band, so the rebuilt patch keeps the "
            "hard rectangular edge seen on SIEGERREBE"
        )

    def test_the_ramp_stays_at_the_rim(self):
        """A feather wide enough to reach the middle would leave ink standing."""
        img, _mask, band, out = self._erased()
        differs = np.abs(out - self._hard(img, band)).max(axis=2) > 1e-6
        core = (
            cv2.erode(
                (band > 0.5).astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1
            )
            > 0
        )
        assert not bool((differs & core).any()), (
            "the feather reaches into the middle of the band, where it would let "
            "the very ink the band exists to remove show through"
        )

    def test_the_middle_of_the_patch_is_fully_rebuilt(self):
        """The ramp must live at the rim only, or ink survives inside the band."""
        img, _mask, band, out = self._erased()
        core = (
            cv2.erode(
                (band > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1
            )
            > 0
        )
        surface = reconstruct_surface(img, band)
        span = np.abs(surface - img).max(axis=2)
        moved = np.abs(out - img).max(axis=2)
        usable = core & (span > 0.02)
        assert usable.sum() > 0
        assert float(np.median(moved[usable] / span[usable])) > 0.9

    def test_nothing_outside_the_band_is_touched(self):
        img, _mask, band, out = self._erased()
        outside = band <= 0.0
        assert np.allclose(out[outside], img[outside], atol=1e-6)
