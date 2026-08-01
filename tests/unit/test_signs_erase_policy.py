"""Unit tests for the `erase` too_small_policy on the Sign Detailer.

`erase` is the answer to a measured dead end. `soften` was supposed to be the
safe fallback for a surface too small to carry a readable word, but at denoise
0.35 the pass does not blur the strokes that are already there — it repaints
them as the model's own invention, sharp enough to transcribe (`TOKL CUBA ANY
RODA`, `TEGLING AVYAT`, `BUTORT DOATRKR`). Anything legible enough to read is
legible fantasy writing, so a census counts it, and rightly.

`erase` therefore does no rendering at all: it empties the text band and pulls
the surrounding surface inward over it. The load-bearing property is a NEGATIVE
one — that no sampler and no glyph layer ever touch the region — because a pass
over a swept surface writes its own notes on it whatever the denoise. That is
what these tests pin, at the call structure rather than at the pixels, since a
picture cannot prove which functions ran.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

import cv2

from nodes.signs.detailer import SignDetailer, SOFTEN_PROMPT
from nodes.signs.options import SIGN_DEFAULTS


# ── scene ──

RECT = (64, 48, 192, 144)  # x1, y1, x2, y2


def make_scene(rect=RECT, h=192, w=256, bars=6):
    """A lit plate inside a dark surround, carrying dark stroke-like bars.

    The plate has a vertical brightness gradient so a reconstruction that threw
    the lighting away and flooded the band with one flat colour would show up.
    """
    x1, y1, x2, y2 = rect
    img = np.full((h, w, 3), 0.25, np.float32)
    ramp = np.linspace(0.92, 0.72, y2 - y1, dtype=np.float32)
    img[y1:y2, x1:x2] = ramp[:, None, None]
    for i in range(bars):
        x = x1 + 12 + i * 18
        img[y1 + 22 : y1 + 46, x : x + 7] = 0.06
    return img


def make_mask(rect=RECT, h=192, w=256):
    mask = np.zeros((h, w), np.float32)
    x1, y1, x2, y2 = rect
    mask[y1:y2, x1:x2] = 1.0
    return mask


def make_region(index=0, rect=RECT, too_small=True, text="WEINHANDEL", **over):
    region = {
        "index": index,
        "class": "sign",
        "mask": make_mask(rect),
        "bbox": rect,
        "height_px": 12,
        "too_small": too_small,
        "batch_index": 0,
        "cluster_id": -1,
        "proposal": {"text": text, "style": "photographic"},
    }
    region.update(over)
    return region


def edge_density(rgb, sel):
    """Share of pixels carrying a strong gradient — strokes have many, plate none."""
    grey = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mag = np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(
        cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    )
    return float((mag[sel] > 40).mean())


def grey_std(rgb, sel):
    grey = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(grey[sel].std())


# ── harness ──


class Spies:
    """Records every call into the sampler and into glyph rendering."""

    def __init__(self):
        self.sampled = []
        self.glyphs = []
        self.prompts = []


@pytest.fixture
def spies(monkeypatch):
    s = Spies()

    def fake_inpaint_slot(**kw):
        s.sampled.append(kw)
        return kw["image"], torch.zeros(1, 8, 8, 3)

    def fake_render_glyph_layer(**kw):
        s.glyphs.append(kw)
        mask = np.asarray(kw["mask_2d"], np.float32)
        rgb = np.zeros(mask.shape[:2] + (3,), np.float32)
        return rgb, (mask > 0.5).astype(np.float32)

    def fake_encode(self, clip, text):
        s.prompts.append(text)
        return [[torch.zeros(1, 4), {}]]

    monkeypatch.setattr("nodes.signs.detailer.inpaint_slot", fake_inpaint_slot)
    monkeypatch.setattr(
        "nodes.signs.detailer.render_glyph_layer", fake_render_glyph_layer
    )
    monkeypatch.setattr(SignDetailer, "_encode", fake_encode)
    return s


def run(policy, regions, image=None):
    node = SignDetailer()
    img = make_scene() if image is None else image
    images = torch.from_numpy(img)[None]
    return node.execute(
        images=images,
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
    )


# ── the widget ──


class TestPolicyWidget:
    def test_erase_is_offered(self):
        options = SignDetailer.INPUT_TYPES()["required"]["too_small_policy"][0]
        assert "erase" in options

    def test_default_is_still_soften(self):
        """Not negotiable: `erase` ships switched off and is measured with --set.

        Changing the default would fold a policy change into every other
        measurement taken from here on, and there is no live number for it yet.
        """
        spec = SignDetailer.INPUT_TYPES()["required"]["too_small_policy"]
        assert spec[1]["default"] == "soften"

    def test_the_other_three_survive(self):
        options = SignDetailer.INPUT_TYPES()["required"]["too_small_policy"][0]
        assert options[:3] == ["soften", "skip", "render"]

    def test_tooltip_says_what_erase_does(self):
        tip = SignDetailer.INPUT_TYPES()["required"]["too_small_policy"][1]["tooltip"]
        assert "erase:" in tip
        assert "sampler" in tip.lower()

    def test_min_legible_px_default_untouched(self):
        """The floor stays off — this step changes the policy, nothing else."""
        assert (
            SignDetailer.INPUT_TYPES()["required"]["min_legible_px"][1]["default"] == 0
        )


# ── the negative property: nothing renders ──


class TestEraseRunsNoModel:
    def test_erase_calls_no_sampler(self, spies):
        run("erase", [make_region()])
        assert spies.sampled == [], (
            "erase must not run a diffusion pass over the region"
        )

    def test_erase_renders_no_glyph_layer(self, spies):
        run("erase", [make_region()])
        assert spies.glyphs == [], "erase must not typeset anything onto the surface"

    def test_erase_encodes_no_region_prompt(self, spies):
        """Only the run-wide negative prompt is encoded — no positive for the region."""
        run("erase", [make_region()])
        assert spies.prompts == [SIGN_DEFAULTS["negative_prompt"]]

    def test_the_spies_are_wired(self, spies):
        """Control: the same region under `render` does hit both of them.

        Without this, the two assertions above would also pass on a harness that
        simply never reached the code.
        """
        run("render", [make_region()])
        assert len(spies.sampled) == 1
        assert len(spies.glyphs) == 1

    def test_erase_is_reported_as_its_own_outcome(self, spies):
        _img, _crops, _glyph, report = run("erase", [make_region()])
        assert "erased" in report
        assert "1 erased" in report


# ── the picture ──


class TestErasedSurface:
    def setup_method(self):
        self.img = make_scene()
        self.node = SignDetailer()
        self.out, self.band = self.node._apply_erase(
            torch.from_numpy(self.img), make_region()
        )
        self.after = None if self.band is None else self.out.cpu().numpy()

    def test_a_band_was_found(self):
        assert self.band is not None
        assert float(self.band.sum()) > 0

    def test_no_stroke_structure_is_left(self):
        """The band must not read as lettering any more.

        Measured as gradient density rather than by eye: strokes are edges, and
        a surface pulled in from its own rim has none.
        """
        sel = self.band > 0.5
        before = edge_density(self.img, sel)
        after = edge_density(self.after, sel)
        assert before > 0.10, (
            "the fixture has to contain stroke-like structure to begin with"
        )
        assert after <= before * 0.10
        assert after <= 0.02

    def test_the_band_is_as_quiet_as_the_surface_around_it(self):
        """Variance inside the band falls to that of the untouched plate."""
        sel = self.band > 0.5
        plate = (make_mask() > 0.5) & (self.band <= 0.5)
        assert grey_std(self.after, sel) <= grey_std(self.img, sel) * 0.2
        assert grey_std(self.after, sel) <= grey_std(self.after, plate) + 6.0

    def test_the_lighting_is_kept(self):
        """Not a flat fill: the plate's gradient still runs through the band.

        Flooding it with one colour would throw away the lighting and hand the
        sampler a reason to invent it back.
        """
        sel = self.band > 0.5
        ys = np.where(sel.any(axis=1))[0]
        top = self.after[ys[0] : ys[0] + 3][sel[ys[0] : ys[0] + 3]].mean()
        bottom = self.after[ys[-1] - 2 : ys[-1] + 1][
            sel[ys[-1] - 2 : ys[-1] + 1]
        ].mean()
        assert top > bottom + 0.01, (
            "the top of the band must stay brighter than its foot"
        )

    def test_nothing_outside_the_band_moves(self):
        untouched = self.band <= 0.5
        assert np.allclose(self.after[untouched], self.img[untouched], atol=1e-6)

    def test_a_blank_region_comes_back_untouched(self):
        """No lettering to find means no change and no band, not a crash."""
        blank = np.full((192, 256, 3), 0.5, np.float32)
        out, band = self.node._apply_erase(torch.from_numpy(blank), make_region())
        assert band is None
        assert np.allclose(np.asarray(out), blank)


# ── the overlap ledger ──


class TestEraseCountsAsTreated:
    INNER = (80, 60, 170, 130)  # lies wholly inside RECT

    def test_a_later_region_may_not_write_over_an_erased_one(self, spies):
        regions = [
            make_region(index=0, rect=RECT, too_small=True),
            make_region(index=1, rect=self.INNER, too_small=False),
        ]
        _img, _crops, _glyph, report = run("erase", regions)
        assert "already rewritten" in report
        assert spies.sampled == [], (
            "the covered region must not be rendered after an erase"
        )

    def test_skip_leaves_the_surface_free(self, spies):
        """Control: `skip` does NOT claim the surface, so the later region runs.

        This is what makes the test above about erase's bookkeeping rather than
        about the two regions merely overlapping.
        """
        regions = [
            make_region(index=0, rect=RECT, too_small=True),
            make_region(index=1, rect=self.INNER, too_small=False),
        ]
        _img, _crops, _glyph, report = run("skip", regions)
        assert "already rewritten" not in report
        assert len(spies.sampled) == 1

    def test_an_untouched_neighbour_is_unaffected(self, spies):
        """Erasing claims its own surface only, not the whole image."""
        far = (10, 150, 60, 185)
        regions = [
            make_region(index=0, rect=RECT, too_small=True),
            make_region(index=1, rect=far, too_small=False),
        ]
        _img, _crops, _glyph, report = run("erase", regions)
        assert "already rewritten" not in report
        assert len(spies.sampled) == 1


# ── the policies that were already there ──


class TestSoftenAndSkipUnchanged:
    def test_soften_still_samples_the_region(self, spies):
        run("soften", [make_region()])
        assert len(spies.sampled) == 1

    def test_soften_still_uses_its_own_prompt_and_denoise(self, spies):
        run("soften", [make_region()])
        assert SOFTEN_PROMPT in spies.prompts
        assert spies.sampled[0]["denoise"] == pytest.approx(0.35)

    def test_soften_still_renders_no_glyph_layer(self, spies):
        run("soften", [make_region()])
        assert spies.glyphs == []

    def test_soften_is_still_counted_as_softened(self, spies):
        _img, _crops, _glyph, report = run("soften", [make_region()])
        assert "softened" in report
        assert "1 softened" in report

    def test_skip_still_touches_nothing(self, spies):
        _img, _crops, _glyph, report = run("skip", [make_region()])
        assert spies.sampled == []
        assert spies.glyphs == []
        assert "skipped (too small)" in report

    def test_skip_leaves_the_pixels_alone(self, spies):
        img = make_scene()
        out, _crops, _glyph, _report = run("skip", [make_region()], image=img)
        assert np.allclose(out[0].cpu().numpy(), img, atol=1e-6)

    def test_erase_leaves_a_big_enough_region_alone(self, spies):
        """The policy only ever applies to regions the selector flagged.

        A region that is not `too_small` goes down the normal path whatever the
        policy says, which is why turning `erase` on cannot quietly blank a sign
        that was renderable.
        """
        run("erase", [make_region(too_small=False)])
        assert len(spies.sampled) == 1
        assert len(spies.glyphs) == 1
