"""A one-sided feather must not steal from the mask or bleed onto its neighbour.

The bug this pins: feather_mask was a plain Gaussian, symmetric by nature. On a
hair mask wrapped around a face that is destructive twice over — the hair loses
strength while the face gets repainted — which is how selecting 'hair' still
rendered the face.
"""
import cv2
import numpy as np
import pytest
import torch

from nodes.utils.mask_utils import feather_mask

H, W = 640, 512
FACE = ((256, 330), (110, 165))


def _hair_ring():
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (256, 330), (170, 240), 0, 0, 360, 255, -1)
    cv2.ellipse(m, *FACE, 0, 0, 360, 0, -1)
    return m


HAIR = _hair_ring() > 0
FACE_HOLE = np.zeros((H, W), np.uint8)
cv2.ellipse(FACE_HOLE, *FACE, 0, 0, 360, 255, -1)
FACE_HOLE = FACE_HOLE > 0


@pytest.fixture
def hair():
    return torch.from_numpy(_hair_ring().astype(np.float32) / 255.0)


class TestFeatherDirection:
    def test_inward_never_touches_the_neighbour(self, hair):
        """'hair' selected must mean the face is not sampled at all."""
        out = feather_mask(hair, 32, "inward").numpy()
        assert out[FACE_HOLE].max() == 0.0

    def test_inward_still_ramps(self, hair):
        out = feather_mask(hair, 32, "inward").numpy()
        ramp = out[HAIR]
        assert ramp.min() < 0.05, "no soft edge inside the mask"
        assert ramp.max() > 0.99, "the mask core lost strength"

    @pytest.mark.parametrize("direction", ["both", "inward", "outward"])
    def test_no_step_at_the_boundary(self, direction):
        """Every mode must ramp, not cliff.

        Clipping a symmetric blur (min/max against the original) looks one-sided
        but leaves a ~0.5 jump exactly at the boundary — a hard edge dressed up
        as a feather. The one-sided modes shift the mask before blurring so the
        gradient stays as gentle as the symmetric case.
        """
        m = np.zeros((64, 400), np.float32)
        m[:, :200] = 1.0
        profile = feather_mask(torch.from_numpy(m), 32, direction).numpy()[32]
        assert np.abs(np.diff(profile)).max() < 0.05

    @pytest.mark.parametrize("radius", [16, 32, 48, 64, 128])
    def test_inward_never_consumes_a_thin_mask(self, hair, radius):
        """A ring thinner than 2R would be eaten by its own feather.

        Measured before the cap on this ring (half-thickness 38px): R=32 left
        the hair at alpha 0.41 and R=48 erased it outright — selecting 'inward'
        for hair would have refined nothing at all.
        """
        out = feather_mask(hair, radius, "inward").numpy()
        assert out[HAIR].max() > 0.99, f"mask capped at {out[HAIR].max():.2f}"
        assert out[FACE_HOLE].max() == 0.0

    def test_inward_cap_scales_with_mask_depth(self):
        """A thick mask keeps the radius it asked for; a thin one gets less."""
        thick = np.zeros((400, 400), np.float32)
        thick[50:350, 50:350] = 1.0            # half-thickness 150px
        thin = np.zeros((400, 400), np.float32)
        thin[190:210, 50:350] = 1.0            # half-thickness 10px
        wide = feather_mask(torch.from_numpy(thick), 32, "inward").numpy()
        narrow = feather_mask(torch.from_numpy(thin), 32, "inward").numpy()
        wide_ramp = ((wide > 0.01) & (wide < 0.99)).sum() / (wide > 0.01).sum()
        narrow_ramp = ((narrow > 0.01) & (narrow < 0.99)).sum() / (narrow > 0.01).sum()
        assert wide_ramp < narrow_ramp        # thick mask: mostly solid core
        assert wide.max() > 0.99 and narrow.max() > 0.99

    def test_all_directions_share_one_ramp_steepness(self):
        """Same softness, only the position differs — so the widget is a choice
        of where the transition sits, not of how abrupt it is."""
        m = np.zeros((64, 400), np.float32)
        m[:, :200] = 1.0
        t = torch.from_numpy(m)
        slopes = [np.abs(np.diff(feather_mask(t, 32, d).numpy()[32])).max()
                  for d in ("both", "inward", "outward")]
        assert max(slopes) - min(slopes) < 0.005

    def test_outward_keeps_the_mask_at_full_strength(self, hair):
        out = feather_mask(hair, 32, "outward").numpy()
        assert out[HAIR].min() == pytest.approx(1.0)
        assert out[FACE_HOLE].max() > 0.1, "no ramp outside the mask"

    def test_both_is_centred_on_the_boundary(self):
        """'both' already splits the ramp evenly — half inside, half outside."""
        m = np.zeros((64, 400), np.float32)
        m[:, :200] = 1.0
        p = feather_mask(torch.from_numpy(m), 32, "both").numpy()[32]
        assert p[199] + p[200] == pytest.approx(1.0, abs=0.02)
        inside_ramp = ((p[:200] > 0.01) & (p[:200] < 0.99)).sum()
        outside_ramp = ((p[200:] > 0.01) & (p[200:] < 0.99)).sum()
        assert abs(inside_ramp - outside_ramp) <= 2

    def test_both_is_the_legacy_symmetric_blur(self, hair):
        out = feather_mask(hair, 32, "both").numpy()
        legacy = cv2.GaussianBlur(hair.numpy(), (65, 65), 0)
        assert np.allclose(out, legacy)

    def test_both_loses_half_the_hair_and_paints_the_face(self, hair):
        """Documents the behaviour the other two directions exist to avoid."""
        out = feather_mask(hair, 32, "both").numpy()
        assert (out[HAIR] > 0.95).mean() < 0.6      # measured ~0.50
        assert (out[FACE_HOLE] > 0.05).mean() > 0.2  # measured ~0.24

    def test_default_is_unchanged_behaviour(self, hair):
        assert np.allclose(feather_mask(hair, 32).numpy(),
                           feather_mask(hair, 32, "both").numpy())

    @pytest.mark.parametrize("direction", ["both", "inward", "outward"])
    def test_zero_radius_is_a_passthrough(self, hair, direction):
        assert torch.equal(feather_mask(hair, 0, direction), hair)


class TestBlendRadiusConversion:
    """mask_blend_pixels is defined in sampling resolution and converted back."""

    @staticmethod
    def _convert(mask_blend_pixels, target_width, crop_w):
        sampling_scale = target_width / max(1, crop_w)
        return min(256, max(1, int(round(mask_blend_pixels / sampling_scale))))

    def test_upscaled_crop_shrinks_the_image_side_radius(self):
        # 380px crop resized up to 800: 32 sampling px is ~15 image px
        assert self._convert(32, 800, 380) == 15

    def test_downscaled_crop_grows_it(self):
        # a large crop scaled down: the ramp covers more image pixels
        assert self._convert(32, 800, 1600) == 64

    def test_one_to_one_crop_is_identity(self):
        assert self._convert(32, 800, 800) == 32

    def test_never_rounds_away_to_zero(self):
        """A tiny ramp is still a ramp — rounding to 0 would leave a hard seam."""
        assert self._convert(1, 4096, 64) >= 1
