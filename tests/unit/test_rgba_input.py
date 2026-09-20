"""RGBA input must not reach the three-channel models.

The bug this pins: a GLSL/shader node upstream handed PersonSelectorSAM3 a
[1,H,W,4] IMAGE. SAM3's normalize subtracted a 3-vector mean from 4 channels,
BiSeNet's first conv got 4 channels where it wants 3, and both failed several
frames deep with a shape error that named neither the node nor the alpha.
"""
import numpy as np
import pytest
import torch

from nodes.utils.tensor_utils import tensor2np, tensor2cv2


@pytest.fixture
def rgba():
    img = torch.rand(1, 64, 48, 4, dtype=torch.float32)
    img[..., 3] = 1.0
    return img


@pytest.fixture
def rgb():
    return torch.rand(1, 64, 48, 3, dtype=torch.float32)


class TestTensor2np:
    def test_rgba_is_trimmed(self, rgba):
        assert tensor2np(rgba).shape == (64, 48, 3)

    def test_rgb_is_untouched(self, rgb):
        out = tensor2np(rgb)
        assert out.shape == (64, 48, 3)
        assert np.array_equal(out, (rgb[0].numpy() * 255).clip(0, 255).astype(np.uint8))

    def test_colour_channels_survive_the_trim(self, rgba):
        """Trimming must drop alpha, not reorder or rescale RGB."""
        expected = (rgba[0, :, :, :3].numpy() * 255).clip(0, 255).astype(np.uint8)
        assert np.array_equal(tensor2np(rgba), expected)

    def test_cv2_conversion_works_on_rgba(self, rgba):
        """cvtColor(RGB2BGR) would raise on a 4-channel array."""
        assert tensor2cv2(rgba).shape == (64, 48, 3)


class TestStitchNeedsThreeChannels:
    """Why inpaint_slot trims up front: stitch_back cannot take a 4-channel original.

    The VAE path hides the problem — it slices to 3 itself — so an RGBA image
    survives all the way to the blend, where a 3-channel delta meets a
    4-channel region.
    """

    @staticmethod
    def _stitch(original, monkeypatch):
        import comfy.utils
        monkeypatch.setattr(comfy.utils, "common_upscale", lambda s, w, h, m, c: s)
        from nodes.utils.inpaint_pipeline import stitch_back

        size = original.shape[0]
        decoded = torch.rand(1, size, size, 3, dtype=torch.float32)
        mask = torch.zeros(size, size, dtype=torch.float32)
        mask[16:48, 16:48] = 1.0
        return stitch_back(original, decoded,
                           mask, {"x": 0, "y": 0, "w": size, "h": size},
                           {"x": 0, "y": 0, "w": size, "h": size, "pad_l": 0, "pad_t": 0},
                           denoise=0.5)

    def test_four_channels_fail(self, monkeypatch):
        with pytest.raises((RuntimeError, ValueError)):
            self._stitch(torch.rand(64, 64, 4, dtype=torch.float32), monkeypatch)

    def test_the_guard_expression_makes_them_pass(self, monkeypatch):
        """Exactly what inpaint_slot does before Step 1."""
        image = torch.rand(64, 64, 4, dtype=torch.float32)
        if image.shape[-1] > 3:
            image = image[..., :3]
        assert self._stitch(image, monkeypatch).shape == (64, 64, 3)
