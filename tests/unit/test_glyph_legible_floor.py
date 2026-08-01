"""The legibility floor under the source-size cap (`min_legible_px`).

`glyph_match_source_size` ties the replacement lettering to the size the surface
already used. On a defocused label `measure_ink_height` reads 4-6px, so the new
word was typeset at 4-5px cap height on a surface that would carry three times
that, and the sampler turned those strokes to mush that reads back as a
fragment. The floor raises that cap — and only the cap. It never forces a size,
never drops a word and never lets one be clipped; the box keeps the last word.
"""

import numpy as np
import pytest

from nodes.utils import glyph as G
from nodes.utils.glyph import (
    MIN_FONT_SIZE,
    _cap_height,
    _line_height,
    _load_font,
    _fit_text,
    _size_for_cap_height,
    render_text_block,
    resolve_font,
)
from nodes.signs.detailer import SignDetailer


# Two faces on purpose: the bundled default and a real system TTF. They have very
# different em metrics (the system face is nearly three times its point size),
# and that is exactly why the floor is stated in cap height instead.
FACES = [None, resolve_font("clean sans-serif")]
FACE_IDS = ["default", "system"]


def _old_bound(inner_h, target_line_height):
    """The cap as it stood before the floor was added — the byte-identity reference."""
    high = max(MIN_FONT_SIZE, int(inner_h) + 2)
    return max(MIN_FONT_SIZE, min(high, int(round(target_line_height * 1.35))))


class TestCapHeight:
    """The floor is a cap height, not an em height. That distinction is the point."""

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    @pytest.mark.parametrize("size", [6, 10, 16, 24, 40])
    def test_cap_height_is_smaller_than_the_em_box(self, font_path, size):
        font = _load_font(font_path, size)
        assert _cap_height(font) < _line_height(font), (
            "measuring the floor on the em box would let strokes fall below it"
        )

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_cap_height_grows_with_the_point_size(self, font_path):
        heights = [_cap_height(_load_font(font_path, s)) for s in range(4, 64, 2)]
        assert heights == sorted(heights), (
            "the binary search relies on this being monotone"
        )
        assert heights[-1] > heights[0]

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    @pytest.mark.parametrize("target", [8, 12, 18, 24])
    def test_size_for_cap_height_is_the_smallest_size_that_reaches_it(
        self, font_path, target
    ):
        size = _size_for_cap_height(font_path, target, 400)
        assert _cap_height(_load_font(font_path, size)) >= target
        if size > MIN_FONT_SIZE:
            assert _cap_height(_load_font(font_path, size - 1)) < target, (
                "not the smallest"
            )

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_it_never_reaches_past_the_ceiling(self, font_path):
        """A floor the box cannot hold must not be answered with a size it cannot hold."""
        assert _size_for_cap_height(font_path, 500, 20) == 20

    def test_the_floor_of_twelve_really_is_about_three_times_the_old_size(self):
        """The claim the widget default rests on, pinned to a number."""
        small = _size_for_cap_height(None, 4, 400)
        floor = _size_for_cap_height(None, 12, 400)
        assert floor >= 2.5 * small


class TestFloorLiftsTinyLettering:
    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    @pytest.mark.parametrize("measured", [4.0, 5.0, 6.0])
    def test_a_word_on_four_pixel_lettering_is_no_longer_set_at_four_pixels(
        self, font_path, measured
    ):
        """The repair itself: a roomy plate whose existing ink measures 4-6px."""
        before, _ = _fit_text(
            "RIESLING",
            600.0,
            90.0,
            font_path,
            3,
            target_line_height=measured,
            min_legible_px=0,
        )
        after, _ = _fit_text(
            "RIESLING",
            600.0,
            90.0,
            font_path,
            3,
            target_line_height=measured,
            min_legible_px=12,
        )
        assert _cap_height(before) < 12, (
            "precondition: the old cap really was this small"
        )
        assert _cap_height(after) >= 12
        assert after.size > before.size

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_the_floor_is_reached_exactly_not_overshot(self, font_path):
        """A floor that quietly filled the box would be the poster bug all over again."""
        font, _ = _fit_text(
            "RIESLING",
            600.0,
            90.0,
            font_path,
            3,
            target_line_height=4.0,
            min_legible_px=12,
        )
        assert _cap_height(font) < 12 + 6, (
            "the cap is a floor, not a licence to fill the box"
        )


class TestFloorNeverLowersAnything:
    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    @pytest.mark.parametrize("measured", [14.0, 20.0, 30.0, 48.0])
    def test_a_cap_already_above_the_floor_is_untouched(self, font_path, measured):
        """Once the measured cap clears it, the floor's VALUE stops mattering.

        Stated against another floor rather than against `min_legible_px=0`,
        because the two no longer agree and should not: with the floor on, the
        measured height is converted to a point size by asking the face, and
        with it off by the 1.35 rule of thumb. The rule is 1/0.74 and overshoots
        on any face whose capitals stand taller than that, which is what let the
        lettering ratchet up over A -> B -> C. See test_glyph_floor_feedback.
        """
        low, lines_low = _fit_text(
            "RIESLING",
            900.0,
            300.0,
            font_path,
            3,
            target_line_height=measured,
            min_legible_px=4,
        )
        after, lines_after = _fit_text(
            "RIESLING",
            900.0,
            300.0,
            font_path,
            3,
            target_line_height=measured,
            min_legible_px=12,
        )
        assert _cap_height(after) >= 12, (
            "precondition: this cap already clears the floor"
        )
        assert after.size == low.size
        assert lines_after == lines_low
        assert abs(_cap_height(after) - measured) <= 1, (
            "the setting must come back the size that was measured, or a pass "
            "reading its own output drifts"
        )

    @pytest.mark.parametrize("measured", [14.0, 20.0, 30.0])
    def test_the_rendered_block_is_identical_pixel_for_pixel(self, measured):
        off = render_text_block(
            "RIESLING",
            480,
            160,
            font_path=FACES[1],
            fill=(255, 255, 255),
            bg=(20, 20, 20),
            target_line_height=measured,
            min_legible_px=4,
        )
        on = render_text_block(
            "RIESLING",
            480,
            160,
            font_path=FACES[1],
            fill=(255, 255, 255),
            bg=(20, 20, 20),
            target_line_height=measured,
            min_legible_px=12,
        )
        assert np.array_equal(off, on)


class TestZeroIsTheOldBehaviour:
    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    @pytest.mark.parametrize("measured", [4.0, 6.0, 12.0, 30.0])
    def test_zero_reproduces_the_old_cap_exactly(self, font_path, measured):
        """A wide box lets the search saturate, so the chosen size IS the cap."""
        font, _ = _fit_text(
            "HI",
            4000.0,
            400.0,
            font_path,
            1,
            target_line_height=measured,
            min_legible_px=0,
        )
        assert font.size == _old_bound(400.0, measured)

    def test_zero_never_even_consults_the_floor(self, monkeypatch):
        """Call structure, not just the result: at 0 the new code does not run."""
        calls = []

        def bomb(*args, **kwargs):
            calls.append(args)
            raise AssertionError("the floor must not be consulted at min_legible_px=0")

        monkeypatch.setattr(G, "_size_for_cap_height", bomb)
        _fit_text(
            "RIESLING",
            600.0,
            90.0,
            FACES[1],
            3,
            target_line_height=4.0,
            min_legible_px=0,
        )
        assert calls == []

    def test_a_positive_floor_does_consult_it(self, monkeypatch):
        """The negative above only means something if the positive is wired up.

        Two questions, not one: what size the face reaches the MEASURED height
        at, and what size it reaches the FLOOR at. The first is the conversion
        that used to be the 1.35 rule of thumb.
        """
        calls = []
        real = G._size_for_cap_height

        def spy(font_path, cap_px, ceiling):
            calls.append((font_path, cap_px, ceiling))
            return real(font_path, cap_px, ceiling)

        monkeypatch.setattr(G, "_size_for_cap_height", spy)
        _fit_text(
            "RIESLING",
            600.0,
            90.0,
            FACES[1],
            3,
            target_line_height=4.0,
            min_legible_px=12,
        )
        assert [c[1] for c in calls] == [4.0, 12]

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_without_a_measured_size_the_floor_changes_nothing(self, font_path):
        """It is a floor under the CAP. No cap, nothing to put a floor under."""
        off, _ = _fit_text(
            "RIESLING",
            600.0,
            90.0,
            font_path,
            3,
            target_line_height=None,
            min_legible_px=0,
        )
        on, _ = _fit_text(
            "RIESLING",
            600.0,
            90.0,
            font_path,
            3,
            target_line_height=None,
            min_legible_px=12,
        )
        assert on.size == off.size


class TestTheBoxKeepsTheLastWord:
    """A region that cannot carry the word at the floor still gets the word."""

    NARROW = (110.0, 22.0)  # roughly one wine label, in pixels

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_a_long_word_on_a_narrow_plate_still_shrinks_below_the_floor(
        self, font_path
    ):
        inner_w, inner_h = self.NARROW
        font, lines = _fit_text(
            "SCHWARZRIESLING",
            inner_w,
            inner_h,
            font_path,
            1,
            target_line_height=5.0,
            min_legible_px=12,
        )
        assert _cap_height(font) < 12, "the floor must not win against the box"
        assert G._line_width(font, lines[0]) <= inner_w

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_the_word_is_never_dropped_or_truncated(self, font_path):
        inner_w, inner_h = self.NARROW
        _font, lines = _fit_text(
            "SCHWARZRIESLING",
            inner_w,
            inner_h,
            font_path,
            1,
            target_line_height=5.0,
            min_legible_px=12,
        )
        assert "".join(lines).replace(" ", "") == "SCHWARZRIESLING"

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_every_line_stays_inside_the_box(self, font_path):
        """Clipping is the failure mode a floor could plausibly introduce."""
        inner_w, inner_h = self.NARROW
        font, lines = _fit_text(
            "MUELLER THURGAU",
            inner_w,
            inner_h,
            font_path,
            3,
            target_line_height=5.0,
            min_legible_px=12,
        )
        for line in lines:
            assert G._line_width(font, line) <= inner_w

    def test_no_ink_touches_the_edge_of_the_rendered_block(self):
        """The same claim on the pixels, where clipping would actually show."""
        block = render_text_block(
            "SCHWARZRIESLING",
            120,
            26,
            font_path=FACES[1],
            fill=(255, 255, 255),
            bg=(0, 0, 0),
            margin_ratio=0.08,
            max_lines=1,
            target_line_height=5.0,
            min_legible_px=12,
        )
        assert int(block.max()) > 0, "precondition: the word was actually typeset"
        edge = np.concatenate(
            [
                block[0].ravel(),
                block[-1].ravel(),
                block[:, 0].ravel(),
                block[:, -1].ravel(),
            ]
        )
        assert int(edge.max()) == 0, "ink on the border means the word was cut off"

    @pytest.mark.parametrize("font_path", FACES, ids=FACE_IDS)
    def test_a_short_word_on_the_same_plate_does_get_lifted(self, font_path):
        """The narrow plate is not blanket-exempt — only what genuinely cannot fit."""
        inner_w, inner_h = 110.0, 22.0
        off, _ = _fit_text(
            "REGENT",
            inner_w,
            inner_h,
            font_path,
            1,
            target_line_height=5.0,
            min_legible_px=0,
        )
        on, _ = _fit_text(
            "REGENT",
            inner_w,
            inner_h,
            font_path,
            1,
            target_line_height=5.0,
            min_legible_px=12,
        )
        assert on.size > off.size
        assert _cap_height(on) > _cap_height(off)


class TestDetailerWiring:
    def test_the_widget_is_off_by_default(self):
        """Off, and measured that way.

        A floor of 12 sharpens single words — RIESLING came back clean where it
        had read IESLIN/SLIN. But a full census run could not show a gain: larger
        type in pass A shifts the region layout the selector finds in pass B, so
        words land on other surfaces. Opt-in per job until that is understood.
        """
        spec = SignDetailer.INPUT_TYPES()["required"]["min_legible_px"]
        assert spec[0] == "INT"
        assert spec[1]["default"] == 0
        assert spec[1]["max"] >= 12, "the measured floor has to stay reachable"

    def test_execute_defaults_to_the_same_value_as_the_widget(self):
        import inspect

        widget = SignDetailer.INPUT_TYPES()["required"]["min_legible_px"][1]["default"]
        signature = inspect.signature(SignDetailer.execute)
        assert signature.parameters["min_legible_px"].default == widget

    def test_apply_glyph_hands_the_floor_to_the_renderer(self, monkeypatch):
        """Call structure: a widget nobody forwards is a widget that does nothing."""
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here — only the arguments are under test")

        monkeypatch.setattr("nodes.signs.detailer.render_glyph_layer", spy)
        mask = np.zeros((64, 64), np.float32)
        mask[20:44, 8:56] = 1.0
        region = {
            "index": 0,
            "class": "label",
            "mask": mask,
            "bbox": [8, 20, 56, 44],
            "proposal": {"text": "RIESLING"},
        }
        image = np.zeros((64, 64, 3), np.float32)
        SignDetailer()._apply_glyph(
            image,
            region,
            "RIESLING",
            "<auto>",
            1.0,
            True,
            False,
            0.08,
            min_legible_px=17,
        )
        assert seen.get("min_legible_px") == 17

    def test_the_soften_policy_still_carries_its_warning(self):
        """It stays available and stays flagged: measured, it invents sharp pseudo-text."""
        spec = SignDetailer.INPUT_TYPES()["required"]["too_small_policy"]
        assert spec[1]["default"] == "soften"
        assert "invents" in spec[1]["tooltip"]
