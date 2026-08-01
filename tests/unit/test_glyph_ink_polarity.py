"""Light lettering on a dark surface, and what happens when it is not found.

Sauvola scales the contrast it demands by the local mean OF THE IMAGE IT IS
HANDED. ``_threshold_ink`` finds light lettering by running it on the INVERTED
image, and inverting silently swaps that reference for ``255 - mean``. On a dark
surface the two passes then ask for wildly different things: measured on the
street scene's painted shopfronts, 41.3 grey levels upward against 7.0 downward,
a 5.9x asymmetry inside a region whose entire range is 6..102.

Gilt lettering on a dark fascia cannot clear that bar. ``st_0_mask3`` — the
``PAXTRES`` shopfront — came back with 68 ink pixels out of 13006 (0.52%), and
those 68 were a diagonal highlight, not the word; the morphological response on
the same region drew ``PAXTRES`` cleanly. One threshold, and then three symptoms
at once: ``text_band`` takes its seed from the same mask so the old writing was
never covered, ``_apply_erase`` handed the image back untouched, and
``measure_ink_height`` returned 4px for lettering standing 37px tall.

The fix scales the light pass by ``min(mean, 255 - mean)``, so neither polarity
is harder to satisfy than the other. On a light surface ``min()`` picks the old
value and nothing moves — the wine shelf's ten labels and the noticeboard's five
sheets come back bit-identical, only the four dark street regions change.

Two guards sit behind it, because no threshold catches everything:

* :func:`ink_detection_failed` — an empty mask on a surface that plainly carries
  structure is a detector failure, not a blank surface, and the region is handed
  to ``too_small_policy`` rather than quietly written over.
* :func:`fallback_line_height` — ``measure_ink_height() is None`` no longer
  drops the size cap altogether, which used to let the word fill the box.
"""

import glob
import os

import cv2
import numpy as np

from nodes.utils.glyph import (
    BAND_FALLBACK_HEIGHT,
    INK_EVIDENCE_SHARE,
    INK_UNRESOLVED_SHARE,
    _fit_text,
    existing_ink_mask,
    fallback_line_height,
    ink_detection_failed,
    ink_evidence_mask,
    measure_ink_height,
    text_band,
)

# ── a painted fascia: broad strokes, low contrast, out of focus ──
#
# The real thing is st_0_mask3. This stands in for it so the test does not
# depend on a 1.2 MB photograph: a dark plate at 30, gilt strokes at 85, 40px
# tall and 18px wide, blurred with an 8px sigma. Under the old rule the detector
# found 2.9% of the region and measured the lettering at 9px; it stands 40.

H, W = 140, 320
PLATE = (20, 20, 300, 120)  # x1, y1, x2, y2
STROKE_HEIGHT = 40


def _plate(level=30):
    img = np.full((H, W, 3), 12, np.uint8)
    x1, y1, x2, y2 = PLATE
    img[y1:y2, x1:x2] = level
    return img


def painted_fascia(plate=30, ink=85, blur=8):
    img = _plate(plate)
    face = np.zeros((H, W), np.uint8)
    for x in range(40, 280, 34):
        cv2.rectangle(face, (x, 52), (x + 18, 52 + STROKE_HEIGHT), 255, -1)
        cv2.rectangle(face, (x + 4, 58), (x + 14, 74), 0, -1)
    face = cv2.GaussianBlur(face, (blur * 2 + 1,) * 2, 0)
    alpha = (face.astype(np.float32) / 255.0)[..., None]
    return (img.astype(np.float32) * (1 - alpha) + ink * alpha).astype(np.uint8)


def blank_fascia():
    return _plate()


def grainy_fascia(sigma=5.0):
    rs = np.random.RandomState(3)
    img = _plate().astype(np.float32)
    x1, y1, x2, y2 = PLATE
    img[y1:y2, x1:x2] += rs.normal(0, sigma, (y2 - y1, x2 - x1, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def plate_mask():
    m = np.zeros((H, W), np.float32)
    x1, y1, x2, y2 = PLATE
    m[y1:y2, x1:x2] = 1.0
    return m


def _share(mask_2d, layer):
    inside = np.asarray(mask_2d) > 0.5
    if layer is None:
        return 0.0
    return float((np.asarray(layer)[inside] > 0).sum()) / float(inside.sum())


# ── the detection itself ──


class TestLightLetteringOnADarkSurfaceIsFound:
    def test_the_ink_mask_is_not_empty(self):
        mask = plate_mask()
        ink = existing_ink_mask(painted_fascia(), mask, (128, 128, 128))
        assert ink is not None
        assert _share(mask, ink) > INK_UNRESOLVED_SHARE * 5, (
            "the painted fascia used to come back at 2.9% of the region, which is "
            "indistinguishable from a blank plate"
        )

    def test_the_ink_sits_where_the_lettering_is(self):
        """Not on the plate around it: 68 stray highlight pixels also beat zero."""
        mask = plate_mask()
        ink = np.asarray(existing_ink_mask(painted_fascia(), mask, (128, 128, 128)))
        ys, xs = np.where(ink > 0)
        assert ys.size, "nothing found at all"
        # the strokes run y 52..92 and x 40..298
        assert 45 <= np.median(ys) <= 99
        assert 35 <= np.median(xs) <= 300

    def test_the_measured_height_matches_the_strokes(self):
        """4px was returned for 37px lettering on the real shopfront."""
        mask = plate_mask()
        img = painted_fascia()
        ink = existing_ink_mask(img, mask, (128, 128, 128))
        height = measure_ink_height(img, mask, (128, 128, 128), ink_mask=ink)
        assert height is not None
        assert abs(height - STROKE_HEIGHT) <= 0.35 * STROKE_HEIGHT, height

    def test_the_band_gets_its_seed(self):
        """The band comes from the same mask, so an empty mask leaves the old
        writing standing while the new word is drawn over it."""
        mask = plate_mask()
        img = painted_fascia()
        ink = existing_ink_mask(img, mask, (128, 128, 128))
        height = measure_ink_height(img, mask, (128, 128, 128), ink_mask=ink)
        band = text_band(mask, old_ink=ink, new_ink=None, line_height=height)
        assert band is not None
        assert _share(mask, band) > 0.15, "a band this thin cannot cover the word"


class TestALightSurfaceIsUnchanged:
    """min(mean, 255 - mean) picks the published value wherever mean > 127.5."""

    def test_dark_writing_on_paper_still_reads(self):
        img = np.full((H, W, 3), 235, np.uint8)
        for x in range(40, 280, 34):
            cv2.rectangle(img, (x, 52), (x + 6, 92), (40, 40, 40), -1)
        mask = plate_mask()
        ink = existing_ink_mask(img, mask, (128, 128, 128))
        assert _share(mask, ink) > 0.01
        height = measure_ink_height(img, mask, (128, 128, 128), ink_mask=ink)
        assert height is not None and height >= 20

    def test_the_saved_selector_masks_of_the_light_scenes_are_untouched(self):
        """Recorded before the change: the ten wine labels and the five
        noticeboard sheets come back bit-identical, so only dark surfaces move.

        Skipped when the recorded masks are not in the tree.
        """
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        board = os.path.join(here, "live", "bd_0_suite_board_A_00004_.png")
        if not os.path.exists(board):
            return
        img = cv2.cvtColor(cv2.imread(board, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        recorded = {0: 29.0, 1: 50.0, 2: 56.0, 3: 25.0, 4: 18.0}
        found = 0
        for path in sorted(glob.glob(os.path.join(here, "live", "bd_0_mask*.png"))):
            idx = int(os.path.basename(path).split("mask")[1][0])
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape[:2] != img.shape[:2]:
                continue
            mask = (m > 127).astype(np.float32)
            ink = existing_ink_mask(img, mask, (128, 128, 128))
            assert (
                measure_ink_height(img, mask, (128, 128, 128), ink_mask=ink)
                == (recorded[idx])
            ), f"bd_0_mask{idx} moved; the light path was supposed to stand still"
            found += 1
        assert found == len(recorded)


# ── the guard: an empty answer is not automatically an empty surface ──


class TestAnUnresolvedSurfaceIsReportedRatherThanIgnored:
    def test_structure_without_ink_is_a_detector_failure(self):
        """The detector is simulated as failing, which is what it did: 0.52% of
        the region, made of highlight specks rather than letters."""
        mask = plate_mask()
        empty = np.zeros((H, W), np.float32)
        assert ink_detection_failed(painted_fascia(), mask, empty) is True

    def test_the_evidence_is_what_decides(self):
        mask = plate_mask()
        inside = mask > 0.5
        evidence = ink_evidence_mask(painted_fascia(), mask)
        assert evidence is not None
        assert _share(mask, evidence) >= INK_EVIDENCE_SHARE
        assert (
            _share(mask, ink_evidence_mask(blank_fascia(), mask)) < INK_EVIDENCE_SHARE
        )
        assert inside.sum() > 0

    def test_a_region_that_did_resolve_is_not_flagged(self):
        mask = plate_mask()
        img = painted_fascia()
        ink = existing_ink_mask(img, mask, (128, 128, 128))
        assert ink_detection_failed(img, mask, ink) is False

    def test_a_precomputed_evidence_mask_is_used_as_given(self):
        mask = plate_mask()
        empty = np.zeros((H, W), np.float32)
        assert (
            ink_detection_failed(painted_fascia(), mask, empty, evidence=empty) is False
        )


class TestABlankSurfaceRaisesNoAlarm:
    """ "Nothing found" has to keep meaning "nothing there" on an empty plate."""

    def test_a_flat_plate_is_not_flagged(self):
        mask = plate_mask()
        empty = np.zeros((H, W), np.float32)
        assert ink_detection_failed(blank_fascia(), mask, empty) is False

    def test_photographic_grain_is_not_flagged(self):
        mask = plate_mask()
        empty = np.zeros((H, W), np.float32)
        assert ink_detection_failed(grainy_fascia(), mask, empty) is False

    def test_a_smooth_gradient_is_not_flagged(self):
        ramp = np.tile(np.linspace(60, 200, W, dtype=np.float32), (H, 1))
        img = ramp[..., None].repeat(3, 2).astype(np.uint8)
        mask = plate_mask()
        empty = np.zeros((H, W), np.float32)
        assert ink_detection_failed(img, mask, empty) is False

    def test_a_blank_plate_still_yields_no_ink_and_no_band(self):
        mask = plate_mask()
        img = blank_fascia()
        ink = existing_ink_mask(img, mask, (128, 128, 128))
        assert _share(mask, ink) == 0.0
        assert measure_ink_height(img, mask, (128, 128, 128), ink_mask=ink) is None
        assert text_band(mask, old_ink=ink, new_ink=None) is None

    def test_an_empty_mask_is_answered_without_a_verdict(self):
        empty_mask = np.zeros((H, W), np.float32)
        assert ink_detection_failed(painted_fascia(), empty_mask, None) is False, (
            "no region means no claim either way"
        )


# ── the guard: no measurement is not a licence to fill the box ──


class TestTheSizeCapSurvivesAnUnmeasurableSurface:
    def test_the_fallback_is_the_share_the_module_already_uses(self):
        mask = plate_mask()
        short_side = min(PLATE[3] - PLATE[1], PLATE[2] - PLATE[0])
        assert fallback_line_height(mask) == short_side * BAND_FALLBACK_HEIGHT

    def test_it_is_far_below_the_box(self):
        """The failure was a jump to quad height in a single pass."""
        mask = plate_mask()
        short_side = min(PLATE[3] - PLATE[1], PLATE[2] - PLATE[0])
        assert fallback_line_height(mask) < 0.25 * short_side

    def test_an_empty_mask_has_no_fallback(self):
        assert fallback_line_height(np.zeros((H, W), np.float32)) is None

    def test_it_actually_caps_the_setting(self):
        """Without a cap the binary search fills the box; with the fallback it
        does not. Both settings come from the same box, so this isolates the
        cap."""
        mask = plate_mask()
        inner_w, inner_h = 260.0, 90.0
        uncapped, _lines = _fit_text("BUCHLADEN", inner_w, inner_h, None, 3)
        capped, _lines = _fit_text(
            "BUCHLADEN",
            inner_w,
            inner_h,
            None,
            3,
            target_line_height=fallback_line_height(mask),
        )
        assert capped.size < uncapped.size, (uncapped.size, capped.size)

    def test_a_measurable_surface_is_not_pushed_to_the_fallback(self):
        """The fallback only ever stands in for a missing measurement."""
        mask = plate_mask()
        img = painted_fascia()
        ink = existing_ink_mask(img, mask, (128, 128, 128))
        measured = measure_ink_height(img, mask, (128, 128, 128), ink_mask=ink)
        assert measured is not None
        assert measured > fallback_line_height(mask)
