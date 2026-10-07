"""``<type>+crop`` mask modes: the untouched input crop rides along as reference_latents."""
import pytest
import torch

from nodes.utils.mask_utils import split_mask_type
from nodes.utils import inpaint_pipeline as ip


class TestSplitMaskType:
    def test_suffix_is_split_off(self):
        assert split_mask_type("body+crop") == ("body", True)
        assert split_mask_type("face+crop") == ("face", True)

    def test_plain_types_pass_through(self):
        assert split_mask_type("body") == ("body", False)
        assert split_mask_type("aux") == ("aux", False)

    def test_non_strings_are_left_alone(self):
        assert split_mask_type(None) == (None, False)


class _FakeVae:
    """Encode returns a distinct latent per call, so cycling would be visible."""

    def __init__(self):
        self.calls = 0

    def encode(self, pixels):
        self.calls += 1
        h, w = pixels.shape[1] // 8, pixels.shape[2] // 8
        return torch.full((1, 4, h, w), float(self.calls))

    def decode(self, samples):
        return torch.full((1, samples.shape[-2] * 8, samples.shape[-1] * 8, 3), 0.5)


class _FakeModel:
    def get_model_object(self, name):
        return object()


@pytest.fixture
def captured(monkeypatch):
    seen = []

    class FakeGuider:
        def __init__(self, model):
            pass

        def set_conds(self, positive, negative):
            seen.append(positive)

        def set_cfg(self, cfg):
            pass

        def sample(self, noise, latent_image, sampler, sigmas, **kw):
            return latent_image

    monkeypatch.setattr(ip, "CFGGuider", FakeGuider)
    monkeypatch.setattr(ip.comfy.samplers, "sampler_object", lambda name: object())
    monkeypatch.setattr(ip.comfy.samplers, "calculate_sigmas",
                        lambda ms, sched, steps: torch.linspace(1.0, 0.0, steps + 1))
    monkeypatch.setattr(ip.comfy.sample, "prepare_noise", lambda lat, seed: torch.zeros_like(lat))
    monkeypatch.setattr(ip.comfy.sample, "fix_empty_latent_channels", lambda m, lat: lat)
    monkeypatch.setattr(ip.latent_preview, "prepare_callback", lambda m, n: None)
    monkeypatch.setattr(ip, "stitch_back", lambda image, decoded, *a, **k: image)  # blending is not under test
    monkeypatch.setattr(ip.comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ip.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    return seen


def _run(vae, **kw):
    image = torch.rand(128, 128, 3)
    mask = torch.zeros(128, 128)
    mask[32:96, 32:96] = 1.0
    cond = [[torch.zeros(1, 2, 3), {"pooled_output": None}]]
    return ip.inpaint_slot(
        image=image, mask_2d=mask, model=_FakeModel(), positive_cond=cond, negative_cond=cond,
        vae=vae, seed=1, steps=2, denoise=1.0, sampler_name="euler", scheduler="simple",
        target_width=64, target_height=64, mask_blend_pixels=0, **kw)


def test_reference_is_attached_when_requested(captured):
    _run(_FakeVae(), reference_from_crop=True)
    refs = captured[0][0][1]["reference_latents"]
    assert len(refs) == 1 and refs[0].shape[1] == 4


def test_no_reference_by_default(captured):
    _run(_FakeVae())
    assert "reference_latents" not in captured[0][0][1]


def test_later_rounds_keep_the_original_crop(captured):
    _run(_FakeVae(), reference_from_crop=True, repeat=2)
    assert len(captured) == 2
    first, second = (c[0][1]["reference_latents"][0] for c in captured)
    assert torch.equal(first, second)          # round 2 reference == original crop
    assert float(first.mean()) == 1.0          # i.e. the FIRST encode, not the cycled one


def test_input_conditioning_is_not_mutated(captured):
    cond = [[torch.zeros(1, 2, 3), {}]]
    ip_cond = cond
    image = torch.rand(128, 128, 3)
    mask = torch.zeros(128, 128)
    mask[32:96, 32:96] = 1.0
    ip.inpaint_slot(image=image, mask_2d=mask, model=_FakeModel(), positive_cond=ip_cond,
                    negative_cond=ip_cond, vae=_FakeVae(), seed=1, steps=2, denoise=1.0,
                    sampler_name="euler", scheduler="simple", target_width=64, target_height=64,
                    mask_blend_pixels=0, reference_from_crop=True)
    assert "reference_latents" not in cond[0][1]
