# Mocking and sys.path setup happens in root conftest.py
import pytest
import torch


# ──── Tensor Fixtures ────

@pytest.fixture
def dummy_image():
    """Standard ComfyUI IMAGE tensor: [B, H, W, C] float32 0-1."""
    return torch.rand(1, 512, 512, 3, dtype=torch.float32)


@pytest.fixture
def dummy_image_batch():
    """Batch of 4 images."""
    return torch.rand(4, 512, 512, 3, dtype=torch.float32)


@pytest.fixture
def dummy_mask():
    """Standard ComfyUI MASK tensor: [B, H, W] float32 0-1."""
    return torch.rand(1, 512, 512, dtype=torch.float32)


@pytest.fixture
def dummy_latent():
    """Standard ComfyUI LATENT dict."""
    return {"samples": torch.rand(1, 4, 64, 64, dtype=torch.float32)}


@pytest.fixture
def small_image():
    """Small image for fast tests."""
    return torch.rand(1, 64, 64, 3, dtype=torch.float32)
