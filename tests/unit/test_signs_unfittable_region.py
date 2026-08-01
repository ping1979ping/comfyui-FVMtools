"""A region whose word cannot be typeset inside it goes to too_small_policy.

``render_glyph_layer`` now refuses to hand back a word with its ends cut off:
where no legible setting stands inside the silhouette it returns an empty layer
(see ``test_glyph_never_clipped.py``). The detailer has to answer for that, and
the honest answer is the one the pipeline already has for "this surface cannot
carry a readable word" - ``too_small_policy``.

The failure mode being closed is not obvious. An empty layer used to mean the
region was sampled with NO typeset guidance at all, while the prompt still named
the word - which is the condition under which the model invents its own
lettering, the exact thing the census counts. Erasing or skipping is strictly
better than that.

As with the layer itself, the verdict is only asked for when the legibility
floor is on. With ``min_legible_px=0`` the detailer behaves as it always did.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from nodes.signs.detailer import SignDetailer


# ── scene ──

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


# ── harness ──


class Spies:
    def __init__(self):
        self.sampled = []
        self.glyphs = []
        self.prompts = []


def _install(monkeypatch, glyph_returns_empty):
    s = Spies()

    def fake_inpaint_slot(**kw):
        s.sampled.append(kw)
        return kw["image"], torch.zeros(1, 8, 8, 3)

    def fake_render_glyph_layer(**kw):
        s.glyphs.append(kw)
        mask = np.asarray(kw["mask_2d"], np.float32)
        rgb = np.zeros(mask.shape[:2] + (3,), np.float32)
        if glyph_returns_empty:
            return rgb, np.zeros(mask.shape[:2], np.float32)
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


@pytest.fixture
def unfittable(monkeypatch):
    """The layer comes back empty: the word does not fit inside the outline."""
    return _install(monkeypatch, glyph_returns_empty=True)


@pytest.fixture
def fittable(monkeypatch):
    return _install(monkeypatch, glyph_returns_empty=False)


def run(regions, policy="erase", floor=12, **over):
    node = SignDetailer()
    images = torch.from_numpy(make_scene())[None]
    kw = dict(
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
        min_legible_px=floor,
    )
    kw.update(over)
    return node.execute(**kw)


def report_of(result):
    return result[3]


# ── the routing ──


class TestUnfittableGoesToThePolicy:
    def test_erase_is_applied_instead_of_sampling(self, unfittable):
        result = run([make_region()], policy="erase")
        assert unfittable.sampled == [], "no sampler may touch an unfittable region"
        assert "erased" in report_of(result)

    def test_it_is_counted_as_erased_not_rendered(self, unfittable):
        result = run([make_region()], policy="erase")
        assert "1 erased" in report_of(result)
        assert "0 rendered" in report_of(result)

    def test_skip_writes_nothing_and_samples_nothing(self, unfittable):
        result = run([make_region()], policy="skip")
        assert unfittable.sampled == []
        assert "1 skipped" in report_of(result)

    def test_the_reason_is_named_in_the_report(self, unfittable):
        result = run([make_region()], policy="skip")
        assert "does not fit inside the outline" in report_of(result)
        assert "WEISSBURGUNDER" in report_of(result)

    def test_soften_still_samples_but_without_a_glyph(self, unfittable):
        """The widget is honoured, whatever the journal thinks of soften."""
        result = run([make_region()], policy="soften")
        assert len(unfittable.sampled) == 1
        assert "1 softened" in report_of(result)

    def test_render_still_renders(self, unfittable):
        """`render` means render anyway - the policy is not overruled here."""
        result = run([make_region()], policy="render")
        assert len(unfittable.sampled) == 1
        assert "1 rendered" in report_of(result)

    def test_an_erased_region_blocks_a_later_one_on_the_same_surface(self, unfittable):
        result = run([make_region(0), make_region(1)], policy="erase")
        assert unfittable.sampled == []
        assert "already rewritten by an earlier region" in report_of(result)


class TestAFittableRegionIsUntouched:
    def test_it_is_still_sampled(self, fittable):
        result = run([make_region()], policy="erase")
        assert len(fittable.sampled) == 1
        assert "1 rendered" in report_of(result)

    def test_the_glyph_layer_is_used(self, fittable):
        run([make_region()], policy="erase")
        assert len(fittable.glyphs) == 1

    def test_the_layer_is_typeset_once_per_region_not_once_per_attempt(
        self, fittable, monkeypatch
    ):
        """Hoisted out of the retry loop: the image it is drawn on never moves.

        Three verification attempts, every read-back failing, so the loop runs
        to the end - and the region is still typeset exactly once.
        """
        monkeypatch.setattr(
            "nodes.signs.detailer.ocr_region", lambda *a, **k: {"text": "XXXX"}
        )
        run([make_region()], policy="erase", verify_after="ocr", max_attempts=3)
        assert len(fittable.sampled) == 3
        assert len(fittable.glyphs) == 1


class TestTheFloorOffRunIsUnchanged:
    def test_an_empty_layer_is_not_a_verdict_without_the_floor(self, unfittable):
        """min_legible_px=0 never asks the question, so nothing is re-routed."""
        result = run([make_region()], policy="erase", floor=0)
        assert len(unfittable.sampled) == 1
        assert "1 rendered" in report_of(result)

    def test_and_nothing_is_erased(self, unfittable):
        result = run([make_region()], policy="erase", floor=0)
        assert "0 erased" in report_of(result)

    def test_a_genuinely_small_region_still_follows_the_policy(self, fittable):
        result = run([make_region(too_small=True, height_px=9)], policy="erase")
        assert fittable.sampled == []
        assert "1 erased" in report_of(result)
        assert "9px" in report_of(result)
