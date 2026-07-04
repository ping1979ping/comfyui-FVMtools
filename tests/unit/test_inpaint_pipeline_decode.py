"""Decode-shape normalization in the inpaint pipeline.

Video VAEs (WanVAE behind Wan/Krea checkpoints) decode to [B, T, H, W, C];
image VAEs (SDXL, Flux, Z-Image) to [B, H, W, C]. _flatten_video_frames must
map both onto the IMAGE layout [B, H, W, C] expected by stitch_back and the
latent-cycling re-encode.
"""

import torch

from nodes.utils.inpaint_pipeline import _flatten_video_frames


class TestFlattenVideoFrames:

    def test_wan_style_5d_single_frame(self):
        """WanVAE single image: [1, 1, H, W, C] -> [1, H, W, C], content intact."""
        decoded = torch.rand(1, 1, 96, 64, 3, dtype=torch.float32)
        out = _flatten_video_frames(decoded)
        assert out.shape == (1, 96, 64, 3)
        assert torch.equal(out[0], decoded[0, 0])

    def test_image_vae_4d_passthrough(self):
        """Z-Image/SDXL-style 4D output passes through unchanged (same object)."""
        decoded = torch.rand(1, 96, 64, 3, dtype=torch.float32)
        out = _flatten_video_frames(decoded)
        assert out is decoded

    def test_multi_frame_flattens_into_batch(self):
        """T > 1 flattens frames into the batch dim like core VAEDecode."""
        decoded = torch.rand(2, 3, 32, 48, 3, dtype=torch.float32)
        out = _flatten_video_frames(decoded)
        assert out.shape == (6, 32, 48, 3)
        assert torch.equal(out[0], decoded[0, 0])
        assert torch.equal(out[5], decoded[1, 2])

    def test_flattened_output_permutes_like_stitch_back(self):
        """The exact op that crashed (permute to BCHW) must work post-flatten."""
        decoded = torch.rand(1, 1, 96, 64, 3, dtype=torch.float32)
        out = _flatten_video_frames(decoded)
        bchw = out.permute(0, 3, 1, 2)
        assert bchw.shape == (1, 3, 96, 64)
