"""A region whose ink cannot be resolved goes to too_small_policy, not through.

``existing_ink_mask`` coming back empty used to mean the same thing whatever the
cause, and the cause matters. On a blank plate it is right and there is nothing
to do. On lettering the thresholds could not resolve — measured on the street
scene's painted shopfronts, 0.52% of the region and none of it on the word —
three things then go wrong at once, and all three go wrong quietly:

* ``text_band`` takes its seed from that same mask, so the band is empty and the
  old writing is never covered. The new word is set straight on top of it.
* ``_apply_erase`` finds no band and hands the image back untouched, while the
  report counts the region as treated.
* ``measure_ink_height`` returns ``None``, the size cap disappears entirely, and
  the setting jumps to the height of the whole plate — offline, 5px to 15-17px
  in one pass.

So the detailer now asks whether the detection succeeded, and routes a failure
to ``too_small_policy`` — the answer the pipeline already has for a surface it
cannot honestly write on. The verdict is deliberately NOT tied to
``min_legible_px``: the fault is present with the floor off as well.

The blank surface must keep its old answer, or every empty plate in a scene
would be routed to a policy meant for failures. :mod:`test_glyph_ink_polarity`
pins that side; here it is pinned again through the node.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from nodes.signs.detailer import SignDetailer
from nodes.utils.glyph import fallback_line_height

RECT = (64, 48, 192, 144)


def make_scene(h=192, w=256):
    x1, y1, x2, y2 = RECT
    img = np.full((h, w, 3), 0.25, np.float32)
    img[y1:y2, x1:x2] = 0.85
    for i in range(6):
        x = x1 + 12 + i * 18
        img[y1 + 22 : y1 + 46, x : x + 7] = 0.06
    return img


def make_mask(h=192, w=256):
    mask = np.zeros((h, w), np.float32)
    x1, y1, x2, y2 = RECT
    mask[y1:y2, x1:x2] = 1.0
    return mask


def make_region(index=0, text="WEISSBURGUNDER", **over):
    region = {
        "index": index,
        "class": "sign",
        "mask": make_mask(),
        "bbox": RECT,
        "height_px": 40,
        "too_small": False,
        "batch_index": 0,
        "cluster_id": -1,
        "proposal": {"text": text, "style": "photographic"},
    }
    region.update(over)
    return region


class Spies:
    def __init__(self):
        self.sampled = []
        self.glyphs = []
        self.erased = []


def _install(monkeypatch, unresolved):
    s = Spies()

    def fake_inpaint_slot(**kw):
        s.sampled.append(kw)
        return kw["image"], torch.zeros(1, 8, 8, 3)

    def fake_render_glyph_layer(**kw):
        s.glyphs.append(kw)
        mask = np.asarray(kw["mask_2d"], np.float32)
        return (
            np.zeros(mask.shape[:2] + (3,), np.float32),
            (mask > 0.5).astype(np.float32),
        )

    real_erase = SignDetailer._apply_erase

    def spy_erase(self, image_hwc, region, ink_mask=None):
        s.erased.append(ink_mask)
        return real_erase(self, image_hwc, region, ink_mask=ink_mask)

    monkeypatch.setattr("nodes.signs.detailer.inpaint_slot", fake_inpaint_slot)
    monkeypatch.setattr(
        "nodes.signs.detailer.render_glyph_layer", fake_render_glyph_layer
    )
    monkeypatch.setattr(
        "nodes.signs.detailer.ink_detection_failed", lambda *a, **k: unresolved
    )
    monkeypatch.setattr(SignDetailer, "_apply_erase", spy_erase)
    monkeypatch.setattr(
        SignDetailer, "_encode", lambda self, clip, text: [[torch.zeros(1, 4), {}]]
    )
    return s


@pytest.fixture
def unresolved(monkeypatch):
    """The detector came back empty from a surface that carries structure."""
    return _install(monkeypatch, unresolved=True)


@pytest.fixture
def resolved(monkeypatch):
    return _install(monkeypatch, unresolved=False)


def run(regions, policy="erase", floor=0, **over):
    node = SignDetailer()
    kw = dict(
        images=torch.from_numpy(make_scene())[None],
        sign_data={"regions": regions},
        model=MagicMock(),
        clip=MagicMock(),
        vae=MagicMock(),
        seed=11,
        steps=8,
        denoise=1.0,
        sampler_name="euler",
        scheduler="simple",
        target_width=1024,
        target_height=1024,
        max_upscale=8.0,
        too_small_policy=policy,
        min_legible_px=floor,
    )
    kw.update(over)
    return node.execute(**kw)


def report_of(result):
    return result[3]


class TestAnUnresolvedRegionFollowsThePolicy:
    def test_it_is_not_sampled_with_the_word_still_underneath(self, unresolved):
        result = run([make_region()], policy="erase")
        assert unresolved.sampled == []
        assert "1 erased" in report_of(result)

    def test_skip_leaves_it_alone_and_says_so(self, unresolved):
        result = run([make_region()], policy="skip")
        assert unresolved.sampled == []
        assert "1 skipped" in report_of(result)

    def test_the_reason_is_named(self, unresolved):
        result = run([make_region()], policy="skip")
        assert "no lettering could be resolved" in report_of(result)

    def test_soften_is_honoured(self, unresolved):
        result = run([make_region()], policy="soften")
        assert len(unresolved.sampled) == 1
        assert "1 softened" in report_of(result)

    def test_render_overrules_the_verdict(self, unresolved):
        result = run([make_region()], policy="render")
        assert len(unresolved.sampled) == 1
        assert "1 rendered" in report_of(result)

    def test_no_glyph_layer_is_typeset_for_it(self, unresolved):
        run([make_region()], policy="erase")
        assert unresolved.glyphs == [], (
            "typesetting a word for a region that is about to be erased is the "
            "detection cost the routing exists to avoid"
        )

    def test_the_verdict_holds_with_the_legibility_floor_off(self, unresolved):
        """The fault is present at min_legible_px=0, so the guard has to be too."""
        result = run([make_region()], policy="erase", floor=0)
        assert unresolved.sampled == []
        assert "1 erased" in report_of(result)

    def test_erase_is_seeded_from_the_evidence_not_the_failed_mask(self, unresolved):
        run([make_region()], policy="erase")
        assert len(unresolved.erased) == 1
        seed = unresolved.erased[0]
        assert seed is not None, (
            "erase must not be handed the mask that just failed, or it finds no "
            "band and returns the image untouched"
        )
        assert float(np.asarray(seed).sum()) > 0


class TestAResolvedRegionIsUntouched:
    def test_it_is_still_sampled(self, resolved):
        result = run([make_region()], policy="erase")
        assert len(resolved.sampled) == 1
        assert "1 rendered" in report_of(result)

    def test_it_is_still_typeset(self, resolved):
        run([make_region()], policy="erase")
        assert len(resolved.glyphs) == 1

    def test_nothing_is_erased(self, resolved):
        result = run([make_region()], policy="erase")
        assert "0 erased" in report_of(result)

    def test_a_region_with_no_text_is_still_just_skipped(self, resolved):
        result = run([make_region(text="")], policy="erase")
        assert "no text proposed" in report_of(result)

    def test_a_genuinely_small_region_still_reports_its_height(self, resolved):
        result = run([make_region(too_small=True, height_px=9)], policy="erase")
        assert "9px" in report_of(result)


class TestNoFalseAlarmOnARealBlankSurface:
    """The verdict is left unmocked here: a flat plate must route as it always did."""

    def test_a_blank_plate_is_not_routed_to_the_policy(self, monkeypatch):
        s = Spies()

        def fake_inpaint_slot(**kw):
            s.sampled.append(kw)
            return kw["image"], torch.zeros(1, 8, 8, 3)

        monkeypatch.setattr("nodes.signs.detailer.inpaint_slot", fake_inpaint_slot)
        monkeypatch.setattr(
            SignDetailer, "_encode", lambda self, clip, text: [[torch.zeros(1, 4), {}]]
        )
        blank = np.full((192, 256, 3), 0.55, np.float32)
        node = SignDetailer()
        result = node.execute(
            images=torch.from_numpy(blank)[None],
            sign_data={"regions": [make_region()]},
            model=MagicMock(),
            clip=MagicMock(),
            vae=MagicMock(),
            seed=11,
            steps=8,
            denoise=1.0,
            sampler_name="euler",
            scheduler="simple",
            target_width=1024,
            target_height=1024,
            max_upscale=8.0,
            too_small_policy="erase",
            min_legible_px=0,
        )
        assert "0 erased" in report_of(result)
        assert "no lettering could be resolved" not in report_of(result)
        assert len(s.sampled) == 1


class TestTheSizeCapNeverDisappears:
    def test_an_unmeasurable_surface_gets_the_fallback_not_the_box(self, monkeypatch):
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("only the arguments are under test")

        monkeypatch.setattr("nodes.signs.detailer.render_glyph_layer", spy)
        monkeypatch.setattr(
            "nodes.signs.detailer.measure_ink_height", lambda *a, **k: None
        )
        region = make_region()
        SignDetailer()._apply_glyph(
            make_scene(),
            region,
            "BUCHLADEN",
            "<auto>",
            1.0,
            True,
            False,
            0.08,
            match_source_size=True,
        )
        expected = fallback_line_height(region["mask"])
        assert seen["target_line_height"] == expected
        assert seen["target_line_height"] is not None, (
            "no cap means the box decides, and the box is the whole plate"
        )

    def test_the_fallback_is_a_fraction_of_the_region_not_its_height(self, monkeypatch):
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("only the arguments are under test")

        monkeypatch.setattr("nodes.signs.detailer.render_glyph_layer", spy)
        monkeypatch.setattr(
            "nodes.signs.detailer.measure_ink_height", lambda *a, **k: None
        )
        region = make_region()
        SignDetailer()._apply_glyph(
            make_scene(),
            region,
            "BUCHLADEN",
            "<auto>",
            1.0,
            True,
            False,
            0.08,
            match_source_size=True,
        )
        short_side = min(RECT[3] - RECT[1], RECT[2] - RECT[0])
        assert seen["target_line_height"] < 0.25 * short_side

    def test_filling_the_box_on_purpose_still_works(self, monkeypatch):
        """match_source_size=False asks for the box, and must keep getting it."""
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("only the arguments are under test")

        monkeypatch.setattr("nodes.signs.detailer.render_glyph_layer", spy)
        SignDetailer()._apply_glyph(
            make_scene(),
            make_region(),
            "BUCHLADEN",
            "<auto>",
            1.0,
            True,
            False,
            0.08,
            match_source_size=False,
        )
        assert seen["target_line_height"] is None

    def test_a_measurable_surface_keeps_its_own_measurement(self, monkeypatch):
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("only the arguments are under test")

        monkeypatch.setattr("nodes.signs.detailer.render_glyph_layer", spy)
        monkeypatch.setattr(
            "nodes.signs.detailer.measure_ink_height", lambda *a, **k: 31.0
        )
        SignDetailer()._apply_glyph(
            make_scene(),
            make_region(),
            "BUCHLADEN",
            "<auto>",
            1.0,
            True,
            False,
            0.08,
            match_source_size=True,
        )
        assert seen["target_line_height"] == 31.0
