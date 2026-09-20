"""The stitch delta clamp must guard the seam without ghosting the interior.

The bug this pins: a flat +/-0.35 cap across the whole mask writes part of the
ORIGINAL pixel back wherever the refine changed a lot. On a face that is exactly
the dark detail — lashes, mascara, brows on bright skin — so the blended image
grew grey smears the 'refined' crop preview never showed.
"""
import numpy as np
import pytest
import torch

from nodes.utils.inpaint_pipeline import _delta_limit_map, stitch_back


def _square_mask(size=200, margin=40):
    m = torch.zeros(size, size, dtype=torch.float32)
    m[margin:size - margin, margin:size - margin] = 1.0
    return m


class TestDeltaLimitMap:
    def test_edge_keeps_the_tight_cap(self):
        limits = _delta_limit_map(_square_mask().numpy(), 0.35, 24)
        # First row inside the mask is one pixel from the boundary.
        assert limits[40, 100] == pytest.approx(0.35, abs=0.06)

    def test_interior_is_unclamped(self):
        limits = _delta_limit_map(_square_mask().numpy(), 0.35, 24)
        assert limits[100, 100] == pytest.approx(1.0)

    def test_ramp_is_monotonic_inward(self):
        limits = _delta_limit_map(_square_mask().numpy(), 0.35, 24)
        column = limits[40:100, 100]
        assert np.all(np.diff(column) >= -1e-6)

    def test_small_mask_still_reaches_full_freedom(self):
        """A mask thinner than the falloff must not stay capped everywhere."""
        m = torch.zeros(60, 60, dtype=torch.float32)
        m[26:34, 26:34] = 1.0  # 8px square, far below the 24px falloff
        limits = _delta_limit_map(m.numpy(), 0.35, 24)
        assert limits.max() == pytest.approx(1.0)

    def test_empty_mask_returns_none(self):
        assert _delta_limit_map(np.zeros((32, 32), np.float32), 0.35, 24) is None


@pytest.fixture(autouse=True)
def real_upscale(monkeypatch):
    """conftest mocks the comfy package; stitch_back needs a working resize."""
    import comfy.utils

    def _upscale(samples, width, height, method, crop):
        if samples.shape[-1] == width and samples.shape[-2] == height:
            return samples
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="bilinear")

    monkeypatch.setattr(comfy.utils, "common_upscale", _upscale)


class TestStitchBackGhosting:
    def _stitch(self, delta_clamp):
        size = 200
        # Bright original with one very dark stroke in the middle — a lash.
        original = torch.full((size, size, 3), 0.72, dtype=torch.float32)
        original[95:105, 60:140, :] = 0.06
        # The refine removed the stroke: uniform skin.
        decoded = torch.full((1, size, size, 3), 0.72, dtype=torch.float32)
        mask = _square_mask(size)
        crop = {"x": 0, "y": 0, "w": size, "h": size}
        stitch_info = {"x": 0, "y": 0, "w": size, "h": size, "pad_l": 0, "pad_t": 0}
        return stitch_back(original, decoded, mask, crop, stitch_info,
                           denoise=0.5, delta_clamp=delta_clamp)

    def test_dark_stroke_is_gone_in_the_interior(self):
        out = self._stitch(0.35)
        stroke = out[95:105, 60:140, :]
        assert stroke.min().item() > 0.70, (
            f"lash ghost survived: min={stroke.min().item():.3f}, expected ~0.72"
        )

    def test_seam_still_capped(self):
        """A refine that goes far off-colour must not jump at the mask border."""
        size = 200
        original = torch.full((size, size, 3), 0.5, dtype=torch.float32)
        decoded = torch.full((1, size, size, 3), 1.0, dtype=torch.float32)
        mask = _square_mask(size)
        crop = {"x": 0, "y": 0, "w": size, "h": size}
        stitch_info = {"x": 0, "y": 0, "w": size, "h": size, "pad_l": 0, "pad_t": 0}
        out = stitch_back(original, decoded, mask, crop, stitch_info,
                          denoise=0.5, delta_clamp=0.35)
        # First row inside the mask (1 px from the boundary): the +0.5 the refine
        # wanted is held down to the edge budget, so no step at the seam.
        assert (out[40, 100, 0] - 0.5).item() <= 0.38
        # Deep inside, the same refine is allowed through in full.
        assert (out[100, 100, 0] - 0.5).item() > 0.49

    def test_clamp_of_one_disables_the_cap(self):
        out = self._stitch(1.0)
        stroke = out[95:105, 60:140, :]
        assert stroke.min().item() > 0.70
