"""Qwen Image 2.1 reference head detailer: helpers and node wiring (no GPU)."""

import math
from types import SimpleNamespace

import pytest
import torch

import nodes.qwen_ref_detailer as qrd
from nodes.qwen_ref_detailer import FVM_QwenPipe, FVM_QwenRefHeadDetailer, resolve_pipe
from nodes.utils.qwen_ref import (
    DEFAULT_PROMPT_TEMPLATE,
    EXPRESSION_TEXTS,
    MAX_REFS,
    build_prompt,
    classify_expression,
    mouth_metrics,
    pick_face_for_mask,
    face_square_box,
    get_person_mask,
    make_grid,
    prepare_refs,
    qwen_dims,
)


@pytest.fixture(autouse=True)
def real_upscale(monkeypatch):
    """conftest mocks the comfy package; the helpers need a working resize."""
    import comfy.utils

    def _upscale(samples, width, height, method, crop):
        if samples.shape[-1] == width and samples.shape[-2] == height:
            return samples
        return torch.nn.functional.interpolate(
            samples, size=(height, width), mode="bilinear"
        )

    monkeypatch.setattr(comfy.utils, "common_upscale", _upscale)


def _core_dims(w, h, resolution):
    """Literal copy of the TextEncodeQwenImage21 formula (nodes_qwen.py)."""
    if resolution > 0:
        ratio = w / h
        width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
        height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
    else:
        width, height = round(w / 32) * 32, round(h / 32) * 32
    return max(32, width), max(32, height)


class FakeFace:
    def __init__(self, bbox):
        self.bbox = bbox


class FakeAnalyzer:
    def __init__(self, faces):
        self.faces = faces
        self.calls = 0

    def detect_faces(self, bgr):
        self.calls += 1
        assert bgr.dtype.name == "uint8" and bgr.ndim == 3
        return list(self.faces)


# ── qwen_dims ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "w,h", [(1024, 1024), (640, 480), (333, 777), (2656, 1984), (40, 3000), (17, 17)]
)
@pytest.mark.parametrize("res", [0, 512, 1024, 2048])
def test_qwen_dims_matches_core(w, h, res):
    assert qwen_dims(w, h, res) == _core_dims(w, h, res)


def test_qwen_dims_multiple_of_32():
    for w, h in [(517, 911), (1000, 600)]:
        tw, th = qwen_dims(w, h, 1024)
        assert tw % 32 == 0 and th % 32 == 0


# ── face_square_box ───────────────────────────────────────────────────────


def test_face_square_box_is_square_and_inside():
    x0, y0, x1, y1 = face_square_box((100, 100, 200, 220), 1000, 800, 2.2)
    assert x1 - x0 == y1 - y0 == round(2.2 * 120)
    assert 0 <= x0 and 0 <= y0 and x1 <= 1000 and y1 <= 800


def test_face_square_box_shifted_up():
    x0, y0, x1, y1 = face_square_box((400, 400, 500, 500), 1000, 1000, 2.0)
    assert (y0 + y1) / 2 == pytest.approx(450 - 10, abs=1)


def test_face_square_box_clamped_at_border():
    x0, y0, x1, y1 = face_square_box((0, 0, 50, 50), 300, 200, 4.0)
    assert x0 == 0 and y0 == 0 and x1 - x0 == 200 and y1 <= 200


def test_face_square_box_capped_to_short_side():
    x0, y0, x1, y1 = face_square_box((10, 10, 290, 190), 300, 200, 3.0)
    assert x1 - x0 == 200 and y1 - y0 == 200


# ── prepare_refs ──────────────────────────────────────────────────────────


def test_prepare_refs_splits_batch():
    refs, notes = prepare_refs(
        torch.rand(4, 300, 200, 3), max_refs=4, ref_crop="none", ref_resolution=512
    )
    assert len(refs) == 4
    for r in refs:
        assert r.shape[0] == 1 and r.shape[-1] == 3
        assert r.shape[1] % 32 == 0 and r.shape[2] % 32 == 0
        assert (r.shape[2], r.shape[1]) == qwen_dims(200, 300, 512)


def test_prepare_refs_cap_and_note():
    refs, notes = prepare_refs(torch.rand(6, 64, 64, 3), max_refs=2, ref_crop="none")
    assert len(refs) == 2
    assert any("first 2" in n for n in notes)


def test_prepare_refs_never_exceeds_total_budget():
    refs, _ = prepare_refs(
        torch.rand(12, 64, 64, 3), max_refs=50, ref_crop="none", ref_resolution=256
    )
    assert len(refs) == MAX_REFS == 9


def test_prepare_refs_first_only():
    batch = torch.rand(3, 64, 64, 3)
    refs, _ = prepare_refs(
        batch, max_refs=4, ref_mode="first_only", ref_crop="none", ref_resolution=256
    )
    assert len(refs) == 1


def test_prepare_refs_drops_alpha():
    refs, _ = prepare_refs(
        torch.rand(2, 64, 64, 4), max_refs=4, ref_crop="none", ref_resolution=256
    )
    assert all(r.shape[-1] == 3 for r in refs)


def test_prepare_refs_grid_single_image():
    refs, _ = prepare_refs(
        torch.rand(4, 100, 100, 3),
        max_refs=4,
        ref_mode="grid",
        ref_crop="none",
        ref_resolution=256,
    )
    assert len(refs) == 1
    assert refs[0].shape[1:3] == (512, 512)


def test_prepare_refs_face_crop_uses_largest_face():
    img = torch.zeros(1, 400, 600, 3)
    analyzer = FakeAnalyzer(
        [FakeFace((10, 10, 40, 40)), FakeFace((300, 150, 400, 250))]
    )
    refs, notes = prepare_refs(
        img, ref_crop="face", ref_crop_factor=2.0, ref_resolution=256, analyzer=analyzer
    )
    assert analyzer.calls == 1
    assert refs[0].shape[1:3] == (256, 256)  # square crop → square budget
    assert not notes


def test_prepare_refs_face_crop_region(monkeypatch):
    """The crop must come from the face box, not the whole image."""
    img = torch.zeros(1, 400, 600, 3)
    img[:, 100:300, 250:450, :] = 1.0  # white block where the face is
    analyzer = FakeAnalyzer([FakeFace((300, 150, 400, 250))])
    refs, _ = prepare_refs(
        img, ref_crop="face", ref_crop_factor=2.0, ref_resolution=256, analyzer=analyzer
    )
    assert refs[0].mean().item() > 0.9


def test_prepare_refs_face_fallback_whole_image():
    img = torch.rand(1, 200, 300, 3)
    refs, notes = prepare_refs(
        img, ref_crop="face", ref_resolution=256, analyzer=FakeAnalyzer([])
    )
    assert (refs[0].shape[2], refs[0].shape[1]) == qwen_dims(300, 200, 256)
    assert any("no face" in n for n in notes)


def test_prepare_refs_empty():
    refs, notes = prepare_refs(torch.zeros(0, 64, 64, 3))
    assert refs == []


# ── make_grid ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "n,cols,rows", [(1, 1, 1), (2, 2, 1), (3, 2, 2), (4, 2, 2), (5, 3, 2), (9, 3, 3)]
)
def test_make_grid_layout(n, cols, rows):
    grid = make_grid([torch.rand(1, 64, 96, 3) for _ in range(n)], cell=128)
    assert grid.shape == (1, rows * 128, cols * 128, 3)


def test_make_grid_white_padding_and_multiple_of_32():
    grid = make_grid([torch.zeros(1, 50, 100, 3)] * 3, cell=100)
    assert grid.shape[1] % 32 == 0 and grid.shape[2] % 32 == 0
    assert grid[0, -1, -1].tolist() == [1.0, 1.0, 1.0]  # empty 4th cell
    assert grid[0, 0, 0].tolist() == [1.0, 1.0, 1.0]  # letterbox above a wide tile


# ── build_prompt ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "n,tags",
    [
        (1, "<image2>"),
        (2, "<image2> and <image3>"),
        (3, "<image2>, <image3> and <image4>"),
        (4, "<image2>, <image3>, <image4> and <image5>"),
    ],
)
def test_build_prompt_ref_tags(n, tags):
    p = build_prompt(DEFAULT_PROMPT_TEMPLATE, n)
    assert p.startswith("Replace the face")
    assert f"with the face of the person shown in {tags}," in p
    assert "<image1>" in p and "{" not in p


def test_build_prompt_counts():
    assert "(these 4 photos)" in build_prompt(DEFAULT_PROMPT_TEMPLATE, 4)
    assert "(this photo)" in build_prompt(DEFAULT_PROMPT_TEMPLATE, 1)


def test_build_prompt_grid_and_first_only():
    g = build_prompt(DEFAULT_PROMPT_TEMPLATE, 4, "grid")
    assert "<image3>" not in g and "this photo grid" in g
    f = build_prompt(DEFAULT_PROMPT_TEMPLATE, 4, "first_only")
    assert "<image3>" not in f and "(this photo)" in f


def test_build_prompt_extra_with_braces():
    p = build_prompt(DEFAULT_PROMPT_TEMPLATE, 2, extra="  wearing {red} glasses ")
    assert p.endswith("wearing {red} glasses")


def test_build_prompt_empty_extra_no_trailing_space():
    assert not build_prompt(DEFAULT_PROMPT_TEMPLATE, 2).endswith(" ")


# ── get_person_mask ───────────────────────────────────────────────────────


def _pd(num_refs=2, B=1, H=32, W=32, matches=None):
    pd = {
        "num_references": num_refs,
        "batch_size": B,
        "image_height": H,
        "image_width": W,
        "matches": matches
        if matches is not None
        else [[True] * num_refs for _ in range(B)],
    }
    for mt in [
        "face",
        "head",
        "body",
        "hair",
        "facial_skin",
        "eyes",
        "mouth",
        "neck",
        "accessories",
    ]:
        pd[f"{mt}_masks"] = [torch.zeros(B, H, W) for _ in range(num_refs)]
    return pd


def test_get_person_mask_basic():
    pd = _pd()
    pd["head_masks"][1][0, 5:10, 5:10] = 1.0
    m, reason = get_person_mask(pd, 1, 0, "head", 32, 32)
    assert m.shape == (32, 32) and m.sum().item() == 25 and reason == ""


def test_get_person_mask_out_of_range():
    m, reason = get_person_mask(_pd(num_refs=2), 2, 0, "head", 32, 32)
    assert m is None and "not in PERSON_DATA" in reason


def test_get_person_mask_empty():
    m, reason = get_person_mask(_pd(), 0, 0, "head", 32, 32)
    assert m is None and "empty" in reason


def test_get_person_mask_not_matched():
    pd = _pd(matches=[[False, True]])
    pd["head_masks"][0][0, 1:5, 1:5] = 1.0
    m, reason = get_person_mask(pd, 0, 0, "head", 32, 32)
    assert m is None and "not matched" in reason


def test_get_person_mask_union():
    pd = _pd()
    pd["head_masks"][0][0, 0:4, 0:4] = 1.0
    pd["neck_masks"][0][0, 4:8, 0:4] = 1.0
    m, _ = get_person_mask(pd, 0, 0, "head+neck", 32, 32)
    assert m.sum().item() == 32
    pd["face_masks"][0][0, 10:12, 10:12] = 1.0
    pd["hair_masks"][0][0, 12:14, 10:12] = 1.0
    m, _ = get_person_mask(pd, 0, 0, "face+hair", 32, 32)
    assert m.sum().item() == 8


def test_get_person_mask_batch_index_and_clamp():
    pd = _pd(B=2, matches=[[True, True], [True, True]])
    pd["head_masks"][0][1, 0:2, 0:2] = 1.0
    assert get_person_mask(pd, 0, 0, "head", 32, 32)[0] is None
    assert get_person_mask(pd, 0, 1, "head", 32, 32)[0].sum().item() == 4
    assert (
        get_person_mask(pd, 0, 5, "head", 32, 32)[0].sum().item() == 4
    )  # clamped to last


def test_get_person_mask_resized():
    pd = _pd(H=16, W=16)
    pd["head_masks"][0][0, 4:12, 4:12] = 1.0
    m, reason = get_person_mask(pd, 0, 0, "head", 32, 32)
    assert m.shape == (32, 32) and "resized" in reason


# ── Pipe ──────────────────────────────────────────────────────────────────


def test_qwen_pipe_single_inputs():
    pipe, m, c, v = FVM_QwenPipe().execute(model="M", clip="C", vae="V")
    assert pipe == {"model": "M", "clip": "C", "vae": "V"} and (m, c, v) == (
        "M",
        "C",
        "V",
    )


def test_qwen_pipe_precedence():
    up = {"model": "pm", "clip": "pc", "vae": "pv"}
    bundle = {"Model": "bm", "clip": "bc"}
    pipe = resolve_pipe(up, bundle, model=None, clip="sc", vae=None)
    assert pipe == {"model": "bm", "clip": "sc", "vae": "pv"}


def test_qwen_pipe_missing_raises():
    with pytest.raises(ValueError, match="vae"):
        FVM_QwenPipe().execute(model="M", clip="C")


# ── Node execute (inpaint + encoder mocked) ───────────────────────────────


class Recorder:
    def __init__(self):
        self.slot_calls = []
        self.encode_calls = []


@pytest.fixture
def mocked(monkeypatch):
    rec = Recorder()

    def fake_encode(clip, prompt, negative_prompt, vae, images):
        rec.encode_calls.append(
            {"clip": clip, "vae": vae, "prompt": prompt, "images": images}
        )
        c = images["image_1"]
        latent = {"samples": torch.zeros(1, 64, c.shape[1] // 16, c.shape[2] // 16)}
        return "POS", "NEG", latent

    def fake_slot(**kw):
        rec.slot_calls.append(kw)
        crop = torch.rand(1, kw["target_height"], kw["target_width"], 3)
        m, pos, neg = kw["controlnet_apply_fn"](kw["model"], None, None, crop)
        rec.slot_calls[-1]["cond"] = (m, pos, neg)
        out = kw["image"].clone()
        out[kw["mask_2d"] > 0.5] = 0.5
        return out, torch.rand(1, kw["target_height"], kw["target_width"], 4)

    monkeypatch.setattr(qrd, "_encode_qwen21", fake_encode)
    monkeypatch.setattr(qrd, "inpaint_slot", fake_slot)
    return rec


def _args(**over):
    defaults = {}
    for name, spec in FVM_QwenRefHeadDetailer.INPUT_TYPES()["required"].items():
        if len(spec) > 1 and "default" in spec[1]:
            defaults[name] = spec[1]["default"]
    defaults.update(
        sampler_name="euler",
        scheduler="simple",
        ref_crop="none",
        expression_hint="off",
        skin_tone_match=0.0,
        candidates=1,
    )
    defaults.update(over)
    return defaults


def _scene(B=1, H=256, W=320, num_refs=2):
    image = torch.rand(B, H, W, 3)
    pd = _pd(
        num_refs=num_refs, B=B, H=H, W=W, matches=[[True] * num_refs for _ in range(B)]
    )
    pd["head_masks"][0][:, 60:120, 80:130] = 1.0
    pipe = {"model": "MODEL", "clip": "CLIP", "vae": "VAE"}
    return image, pd, pipe


def test_execute_passthrough_identity(mocked):
    image, pd, pipe = _scene()
    out = FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(4, 64, 64, 3),
        **_args(),
    )
    res = out["result"]
    assert res[1] is pipe and res[2] is pd
    assert res[0].shape == image.shape
    assert isinstance(res[6], str) and out["ui"]["text"] == [res[6]]
    assert len(res) == len(FVM_QwenRefHeadDetailer.RETURN_TYPES)


def test_execute_image_keys_and_resolution(mocked):
    image, pd, pipe = _scene()
    FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(4, 64, 64, 3),
        **_args(),
    )
    call = mocked.encode_calls[0]
    assert sorted(call["images"]) == [
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
    ]
    assert call["clip"] == "CLIP" and call["vae"] == "VAE"
    for img in call["images"].values():
        assert img.shape[1] % 32 == 0 and img.shape[2] % 32 == 0 and img.shape[-1] == 3
    kw = mocked.slot_calls[0]
    assert kw["target_width"] % 32 == 0 and kw["target_height"] % 32 == 0
    assert kw["mask_fill_holes"] is False and kw["mask_expand_pixels"] == 0
    assert kw["cond"] == ("MODEL", "POS", "NEG")
    assert "<image5>" in call["prompt"]


def test_encode_uses_resolution_zero(monkeypatch):
    """_encode_qwen21 must hand resolution=0 to the core node."""
    import sys
    import types

    seen = {}

    class FakeNode:
        @classmethod
        def execute(cls, **kw):
            seen.update(kw)
            return SimpleNamespace(args=("p", "n", {"samples": torch.zeros(1)}))

    mod = types.ModuleType("comfy_extras.nodes_qwen")
    mod.TextEncodeQwenImage21 = FakeNode
    monkeypatch.setitem(sys.modules, "comfy_extras", types.ModuleType("comfy_extras"))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_qwen", mod)
    out = qrd._encode_qwen21("c", "p", "n", "v", {"image_1": 1})
    assert seen["resolution"] == 0 and seen["images"] == {"image_1": 1}
    assert out[0] == "p"


def test_execute_no_match_bit_identical(mocked):
    image, pd, pipe = _scene()
    pd["matches"] = [[False, True]]
    out = FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        **_args(),
    )
    assert torch.equal(out["result"][0], image)
    assert mocked.slot_calls == []
    assert "not matched" in out["result"][6]


def test_execute_ref_index_beyond_references(mocked):
    image, pd, pipe = _scene(num_refs=2)
    out = FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        **_args(ref_index=4),
    )
    assert torch.equal(out["result"][0], image) and mocked.slot_calls == []


def test_execute_rgba_input_trimmed(mocked):
    image, pd, pipe = _scene()
    rgba = torch.cat([image, torch.ones_like(image[..., :1])], dim=-1)
    out = FVM_QwenRefHeadDetailer().execute(
        image=rgba,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        **_args(),
    )
    assert out["result"][0].shape[-1] == 3


@pytest.mark.parametrize("mode,expect_ones", [("full", True), ("masked", False)])
def test_execute_noise_mask(mocked, mode, expect_ones):
    image, pd, pipe = _scene()
    FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        **_args(latent_mode=mode),
    )
    nm = mocked.slot_calls[0]["noise_mask_2d"]
    if expect_ones:
        assert nm is not None and bool((nm == 1).all())
    else:
        assert nm is None


def test_execute_overrides(mocked):
    image, pd, pipe = _scene()
    out = FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        model_override="M2",
        clip_override="C2",
        vae_override="V2",
        **_args(),
    )
    assert mocked.slot_calls[0]["model"] == "M2" and mocked.slot_calls[0]["vae"] == "V2"
    assert (
        mocked.encode_calls[0]["clip"] == "C2" and mocked.encode_calls[0]["vae"] == "V2"
    )
    assert out["result"][1] is pipe


def test_execute_batch_and_seed(mocked):
    image, pd, pipe = _scene(B=2)
    out = FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        **_args(seed=7),
    )
    assert out["result"][0].shape == image.shape
    assert [c["seed"] for c in mocked.slot_calls] == [7, 8]
    assert out["result"][3].shape[0] == 2 and out["result"][3].shape[-1] == 3


def test_execute_mask_expand_scales_with_head(mocked):
    image, pd, pipe = _scene()
    FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        **_args(mask_expand_percent=0.2),
    )
    m = mocked.slot_calls[0]["mask_2d"]
    # 60 px tall head, 20% → 12 px dilation on every side
    ys = torch.nonzero(m[:, 105] > 0.5)[:, 0]
    assert ys.min().item() == 60 - 12 and ys.max().item() == 119 + 12


def test_execute_grid_mode_one_ref_image(mocked):
    image, pd, pipe = _scene()
    FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(4, 64, 64, 3),
        **_args(ref_mode="grid"),
    )
    call = mocked.encode_calls[0]
    assert sorted(call["images"]) == ["image_1", "image_2"]
    assert "this photo grid" in call["prompt"]


# ── Schema ────────────────────────────────────────────────────────────────


def test_schema_consistency():
    it = FVM_QwenRefHeadDetailer.INPUT_TYPES()
    req = it["required"]
    for name in [
        "image",
        "pipe",
        "person_data",
        "ref_index",
        "reference_images",
        "prompt_template",
        "extra_prompt",
        "negative_prompt",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
        "latent_mode",
        "ref_mode",
        "max_refs",
        "ref_crop",
        "ref_crop_factor",
        "canvas_resolution",
        "ref_resolution",
        "mask_type",
        "mask_expand_percent",
        "mask_blend_pixels",
        "context_expand_factor",
        "output_padding",
        "delta_clamp",
        "feather_direction",
        "expression_hint",
    ]:
        assert name in req, name
    assert set(it["optional"]) == {"model_override", "clip_override", "vae_override"}
    assert req["pipe"][0] == "FVM_QWEN_PIPE"
    assert (
        len(FVM_QwenRefHeadDetailer.RETURN_TYPES)
        == len(FVM_QwenRefHeadDetailer.RETURN_NAMES)
        == 7
    )
    assert FVM_QwenRefHeadDetailer.RETURN_TYPES[1] == FVM_QwenPipe.RETURN_TYPES[0]
    import inspect

    params = inspect.signature(FVM_QwenRefHeadDetailer.execute).parameters
    for name in list(req) + list(it["optional"]):
        assert name in params, name
    for combo in [
        "latent_mode",
        "ref_mode",
        "ref_crop",
        "mask_type",
        "feather_direction",
        "expression_hint",
    ]:
        assert req[combo][1]["default"] in req[combo][0]


def test_registration_mappings():
    assert (
        set(qrd.NODE_CLASS_MAPPINGS)
        == set(qrd.NODE_DISPLAY_NAME_MAPPINGS)
        == {"FVM_QwenPipe", "FVM_QwenRefHeadDetailer"}
    )


# ── Expression hint ───────────────────────────────────────────────────────


def _lm68(mouth_open=0.0, corner_lift=0.0):
    """Synthetic iBUG-68 landmarks: eyes at y=0 (iod 60), mouth around y=60."""
    import numpy as np

    L = np.zeros((68, 2))
    L[36:42] = [70, 100]
    L[42:48] = [130, 100]
    cy = 160.0
    L[48] = [75, cy - corner_lift]  # outer corners
    L[54] = [125, cy - corner_lift]
    L[51] = [100, cy - 4]
    L[57] = [100, cy + 4 + mouth_open]
    L[60] = [80, cy - corner_lift]  # inner corners
    L[64] = [120, cy - corner_lift]
    for top, bot, x in ((61, 67, 90), (62, 66, 100), (63, 65, 110)):
        L[top] = [x, cy - 1]
        L[bot] = [x, cy - 1 + mouth_open]
    return L


def test_mouth_metrics_open_vs_closed():
    closed = mouth_metrics(_lm68(0.0), None)
    opened = mouth_metrics(_lm68(10.0), None)
    assert closed["mar"] == pytest.approx(0.0)
    assert opened["mar"] == pytest.approx(10.0 / 40.0)
    assert closed["teeth"] == 0.0


def test_mouth_metrics_smile_sign_and_roll_invariance():
    import numpy as np

    up = mouth_metrics(_lm68(0.0, corner_lift=6.0), None)
    assert up["smile"] == pytest.approx(6.0 / 60.0)
    L = _lm68(8.0, corner_lift=6.0)
    a = np.deg2rad(20)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rot = mouth_metrics(L @ R.T, None)
    ref = mouth_metrics(L, None)
    assert rot["mar"] == pytest.approx(ref["mar"], abs=1e-6)
    assert rot["smile"] == pytest.approx(ref["smile"], abs=1e-6)


def test_mouth_metrics_teeth_counts_bright_pixels():
    import numpy as np

    bgr = np.zeros((300, 300, 3), np.uint8)
    bgr[150:180, 70:130] = 240  # white "teeth" inside the inner lip polygon
    m = mouth_metrics(_lm68(12.0), bgr)
    assert m["teeth"] > 0.05


@pytest.mark.parametrize(
    "m,kind",
    [
        ({"mar": 0.30, "smile": 0.07, "teeth": 0.09}, "laugh"),
        ({"mar": 0.17, "smile": 0.05, "teeth": 0.03}, "toothy_smile"),
        ({"mar": 0.02, "smile": 0.00, "teeth": 0.03}, "toothy_smile"),
        ({"mar": 0.02, "smile": 0.05, "teeth": 0.00}, "closed_smile"),
        ({"mar": 0.02, "smile": 0.01, "teeth": 0.00}, "neutral"),
    ],
)
def test_classify_expression(m, kind):
    assert classify_expression(m) == kind


def test_pick_face_for_mask_prefers_covered_face():
    mask = torch.zeros(100, 200)
    mask[10:60, 120:180] = 1.0
    a, b = FakeFace([10, 10, 60, 60]), FakeFace([125, 15, 175, 55])
    assert pick_face_for_mask([a, b], mask) is b
    assert pick_face_for_mask([a], mask) is None
    assert pick_face_for_mask([], mask) is None


def test_build_prompt_expression_placeholder_and_append():
    text = EXPRESSION_TEXTS["toothy_smile"]
    p = build_prompt(DEFAULT_PROMPT_TEMPLATE, 2, expression=text, extra="EXTRA")
    assert "In <image1> the person is smiling broadly" in p
    assert p.endswith("EXTRA") and "{" not in p
    q = build_prompt("Swap {canvas} with {refs}.", 1, expression=text)
    assert q.startswith("Swap <image1> with <image2>. In <image1> the person")
    assert "{expression}" not in build_prompt(DEFAULT_PROMPT_TEMPLATE, 2)


class _LmFace:
    def __init__(self, bbox, lm):
        self.bbox = bbox
        self.landmark_3d_68 = lm


class _SizeAnalyzer:
    """Laughing face on the scene image (256x320), `ref_face` on the 64px refs."""

    def __init__(self, ref_mouth):
        self.scene = _LmFace([85, 65, 125, 115], _lm68(14.0) * 0.25 + [80, 60])
        self.ref = _LmFace([10, 10, 50, 50], _lm68(ref_mouth) * 0.25 + [5, 5])

    def detect_faces(self, bgr):
        return [self.scene] if bgr.shape[0] == 256 else [self.ref]


def _run_hint(mocked, monkeypatch, hint, ref_mouth):
    image, pd, pipe = _scene()
    monkeypatch.setattr(qrd, "_get_face_analyzer", lambda: _SizeAnalyzer(ref_mouth))
    out = FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        **_args(expression_hint=hint),
    )
    return mocked.encode_calls[-1]["prompt"], out["result"][6]


def test_execute_expression_hint_auto_neutral_refs(mocked, monkeypatch):
    prompt, info = _run_hint(mocked, monkeypatch, "auto", ref_mouth=0.0)
    assert "In <image1> the person is laughing" in prompt
    assert "expression laugh" in info and "-> closed" in info


def test_execute_expression_hint_auto_skips_when_refs_match(mocked, monkeypatch):
    prompt, info = _run_hint(mocked, monkeypatch, "auto", ref_mouth=14.0)
    assert "In <image1> the person" not in prompt
    assert "no hint" in info


def test_execute_expression_hint_always(mocked, monkeypatch):
    prompt, _ = _run_hint(mocked, monkeypatch, "always", ref_mouth=14.0)
    assert "In <image1> the person is laughing" in prompt


def test_execute_expression_hint_off_and_no_face(mocked, monkeypatch):
    image, pd, pipe = _scene()
    monkeypatch.setattr(qrd, "_get_face_analyzer", lambda: FakeAnalyzer([]))
    FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        **_args(expression_hint="auto"),
    )
    FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        **_args(),
    )
    for call in mocked.encode_calls:
        assert "In <image1> the person" not in call["prompt"]


def test_tuned_defaults():
    """Defaults from the expression/identity/skin tuning (IMG_0014 / 0010 / 0001)."""
    req = FVM_QwenRefHeadDetailer.INPUT_TYPES()["required"]
    assert req["expression_hint"][1]["default"] == "auto"
    assert req["mask_expand_percent"][1]["default"] == pytest.approx(0.20)
    assert req["denoise"][1]["default"] == 1.0 and req["cfg"][1]["default"] == 1.0
    assert "{expression}" in req["prompt_template"][1]["default"]
    assert "{hair}" in req["prompt_template"][1]["default"]
    assert "but not their facial expression, skin tone" in DEFAULT_PROMPT_TEMPLATE
    assert "color grading" in DEFAULT_PROMPT_TEMPLATE
    assert req["candidates"][1]["default"] == 3
    assert req["candidates"][1]["min"] == 1 and req["candidates"][1]["max"] == 8
    assert req["skin_tone_match"][1]["default"] == pytest.approx(1.0)
    assert req["take_hairstyle"][1]["default"] == "yes"
    assert set(req["take_hairstyle"][0]) == {"yes", "no"}
# ── Round 2: hair placeholder, candidates, skin tone match ────────────────

import numpy as np  # noqa: E402

from nodes.utils import qwen_ref as qr  # noqa: E402


def test_build_prompt_hair_yes_no():
    yes = build_prompt(DEFAULT_PROMPT_TEMPLATE, 2, hair="yes")
    no = build_prompt(DEFAULT_PROMPT_TEMPLATE, 2, hair="no")
    assert "as well as the hairstyle and hair color from them" in yes
    assert "keep the hairstyle and hair color of the person in <image1>" in no
    assert "{hair}" not in yes and "{hair}" not in no and "{canvas}" not in no
    # default is "yes"; a template without {hair} is unaffected
    assert build_prompt(DEFAULT_PROMPT_TEMPLATE, 2) == yes
    assert build_prompt("x {canvas}", 1, hair="no") == "x <image1>"


@pytest.mark.parametrize(
    "seed,b,k,K,expect",
    [
        (7, 0, 0, 1, 7),
        (7, 2, 0, 1, 9),
        (7, 0, 2, 3, 9),
        (7, 1, 0, 3, 10),
        (7, 1, 2, 3, 12),
    ],
)
def test_candidate_seed(seed, b, k, K, expect):
    assert qr.candidate_seed(seed, b, k, K) == expect


def test_candidate_score_terms():
    assert qr.candidate_score(None, "laugh", "laugh") == qr.NO_FACE_SCORE
    assert qr.candidate_score(0.8, "toothy_smile", "laugh") == pytest.approx(0.8)
    assert qr.candidate_score(0.8, "toothy_smile", "neutral") == pytest.approx(
        0.8 - qr.MOUTH_MISMATCH_PENALTY
    )
    assert qr.candidate_score(0.8, None, "neutral") == pytest.approx(0.8)
    assert qr.candidate_score(0.8, "neutral", "neutral", d_pose=15.0) == pytest.approx(
        0.8 - qr.POSE_MISMATCH_PENALTY
    )
    assert qr.candidate_score(0.8, "neutral", "neutral", d_pose=5.0) == pytest.approx(
        0.8
    )
    assert qr.candidate_score(0.8, "neutral", "neutral", d_smile=0.02) == pytest.approx(
        0.8
    )
    assert qr.candidate_score(
        0.8, "neutral", "neutral", d_smile=-0.08
    ) == pytest.approx(0.8 - qr.SMILE_WEIGHT * 0.05)
    assert qr.candidate_score(
        0.8, "laugh", "toothy_smile", mar_ratio=0.4
    ) == pytest.approx(0.8 - qr.MAR_SHRINK_PENALTY)
    # closed originals: the mouth-opening ratio is irrelevant
    assert qr.candidate_score(
        0.8, "neutral", "neutral", mar_ratio=0.1
    ) == pytest.approx(0.8)


def test_pick_candidate_best_and_ties():
    assert qr.pick_candidate([0.5, 0.9, 0.7]) == 1
    assert qr.pick_candidate([0.7, 0.7, 0.6]) == 0
    assert qr.pick_candidate([qr.NO_FACE_SCORE, 0.1]) == 1
    with pytest.raises(ValueError):
        qr.pick_candidate([])


def test_pose_delta():
    assert qr.pose_delta(None, [0, 0, 0]) is None
    assert qr.pose_delta([1.0, 2.0, 0.0], [4.0, -8.0, 30.0]) == pytest.approx(10.0)


def test_format_candidates_text():
    rows = [
        {
            "seed": 3,
            "sim": 0.7,
            "kind": "laugh",
            "mismatch": False,
            "score": 0.7,
            "d_pose": 2.0,
        },
        {
            "seed": 4,
            "sim": None,
            "kind": None,
            "mismatch": False,
            "score": qr.NO_FACE_SCORE,
        },
        {
            "seed": 5,
            "sim": 0.9,
            "kind": "neutral",
            "mismatch": True,
            "score": 0.65,
            "d_smile": -0.05,
        },
    ]
    txt = qr.format_candidates(rows, 0)
    assert txt.startswith("candidates: *seed 3 sim 0.700 laugh")
    assert "seed 4 no face" in txt and "MOUTH CHANGED" in txt
    assert "dsmile -0.050" in txt
    assert txt.endswith("-> seed 3")


class _CandFace:
    def __init__(self):
        self.bbox = [85, 65, 125, 115]


class _CandAnalyzer:
    def detect_faces(self, bgr):
        return [_CandFace()]


def _cand_setup(monkeypatch, sims, kinds=None):
    """Mocked inpaint writes seed/100 into the head; _judge_face maps it back to a sim."""
    rec = {"slots": [], "encodes": 0}

    def fake_encode(clip, prompt, negative_prompt, vae, images):
        rec["encodes"] += 1
        c = images["image_1"]
        latent = {"samples": torch.zeros(1, 64, c.shape[1] // 16, c.shape[2] // 16)}
        return "POS", "NEG", latent

    def fake_slot(**kw):
        rec["slots"].append(kw["seed"])
        crop = torch.rand(1, kw["target_height"], kw["target_width"], 3)
        m, pos, neg = kw["controlnet_apply_fn"](kw["model"], None, None, crop)
        assert (pos, neg) == ("POS", "NEG")
        out = kw["image"].clone()
        out[kw["mask_2d"] > 0.5] = kw["seed"] / 100.0
        return out, torch.rand(1, kw["target_height"], kw["target_width"], 4)

    def fake_face_in_region(img, mask, box, analyzer):
        return img, None  # hand the image through to _judge_face

    metrics = {"mar": 0.2, "smile": 0.05, "teeth": 0.03}

    def fake_judge(img, bgr, ref_emb):
        if ref_emb is None:  # the original
            return None, "toothy_smile", [0.0, 0.0, 0.0], metrics
        k = int(round(float(img[90, 100, 0]) * 100)) - 10
        kind = (kinds or {}).get(k, "toothy_smile")
        return sims[k], kind, [0.0, 0.0, 0.0], metrics

    monkeypatch.setattr(qrd, "_encode_qwen21", fake_encode)
    monkeypatch.setattr(qrd, "inpaint_slot", fake_slot)
    monkeypatch.setattr(qrd, "_get_face_analyzer", lambda: _CandAnalyzer())
    monkeypatch.setattr(
        qrd, "_ref_embedding", lambda refs, an: np.ones(512) / np.sqrt(512)
    )
    monkeypatch.setattr(qrd, "_face_in_region", fake_face_in_region)
    monkeypatch.setattr(qrd, "_judge_face", fake_judge)
    return rec


def _run_cand(K, seed=10):
    image, pd, pipe = _scene()
    return FVM_QwenRefHeadDetailer().execute(
        image=image,
        pipe=pipe,
        person_data=pd,
        reference_images=torch.rand(2, 64, 64, 3),
        **_args(candidates=K, seed=seed),
    )


def test_candidates_pick_best_encode_once(monkeypatch):
    rec = _cand_setup(monkeypatch, sims=[0.5, 0.8, 0.6])
    out = _run_cand(3)
    assert rec["slots"] == [10, 11, 12]
    assert rec["encodes"] == 1  # conditioning cached across candidates
    res = out["result"][0]
    assert float(res[0, 90, 100, 0]) == pytest.approx(0.11)  # seed 11 won
    info = out["result"][6]
    assert "*seed 11 sim 0.800" in info and "-> seed 11" in info
    assert "3 candidate(s)" in info


def test_candidates_mouth_change_and_no_face(monkeypatch):
    # seed 10: best identity but mouth closed (-0.25); seed 11: no face at all
    rec = _cand_setup(monkeypatch, sims=[0.9, None, 0.7], kinds={0: "neutral"})
    out = _run_cand(3)
    assert float(out["result"][0][0, 90, 100, 0]) == pytest.approx(0.12)
    info = out["result"][6]
    assert "seed 11 no face" in info and "MOUTH CHANGED" in info
    assert len(rec["slots"]) == 3


def test_candidates_without_reference_embedding_render_one(monkeypatch):
    rec = _cand_setup(monkeypatch, sims=[0.5, 0.8, 0.6])
    monkeypatch.setattr(qrd, "_ref_embedding", lambda refs, an: None)
    out = _run_cand(3)
    assert rec["slots"] == [10]
    assert "rendering 1" in out["result"][6]


def test_single_candidate_keeps_old_seed(monkeypatch):
    rec = _cand_setup(monkeypatch, sims=[0.5])
    _run_cand(1, seed=42)
    assert rec["slots"] == [42]


def test_skin_tone_shift_and_apply_lab_shift():
    h, w = 40, 40
    orig = np.zeros((h, w, 3), np.float32)
    orig[...] = [0.80, 0.55, 0.40]  # warm skin
    new = orig.copy()
    new[5:35, 5:35] = [0.85, 0.72, 0.66]  # pale, cool re-render
    new[2:5, 2:5] = [0.2, 0.4, 0.9]  # blue "sky" the render also changed
    mask = np.zeros((h, w), np.uint8)
    mask[10:30, 10:30] = 1
    shift, key = qr.skin_tone_shift(orig, new, mask)
    assert shift is not None and shift[2] > 5  # needs more yellow
    weight = np.zeros((h, w), np.float32)
    weight[2:38, 2:38] = 1.0
    out = qr.apply_lab_shift(new, shift, weight, 1.0, key_lab=key, ref_rgb=orig)
    b_o = qr._rgb_to_lab(orig)[20, 20][2]
    b_new = qr._rgb_to_lab(new)[20, 20][2]
    b_out = qr._rgb_to_lab(out)[20, 20][2]
    assert abs(b_out - b_o) < abs(b_new - b_o) / 2
    assert np.array_equal(out[38:, :], new[38:, :])  # outside the weight: untouched
    assert np.allclose(out[2:4, 2:4], new[2:4, 2:4], atol=2e-3)  # blue: keyed out


def test_apply_lab_shift_gate_skips_unchanged_pixels():
    img = np.full((20, 20, 3), 0.6, np.float32)
    weight = np.ones((20, 20), np.float32)
    shift = np.array([0, 10, 10], np.float32)
    out = qr.apply_lab_shift(img, shift, weight, 1.0, ref_rgb=img)
    assert np.array_equal(out, img)  # render == original -> nothing to correct
    assert np.array_equal(qr.apply_lab_shift(img, None, weight), img)


def test_skin_tone_shift_separate_masks_and_too_few_pixels():
    a = np.full((30, 30, 3), 0.5, np.float32)
    b = a.copy()
    b[:, 15:] = 0.7
    m1 = np.zeros((30, 30), np.uint8)
    m1[:, :15] = 1
    m2 = np.zeros((30, 30), np.uint8)
    m2[:, 15:] = 1
    shift, key = qr.skin_tone_shift(a, b, m1, m2)
    assert shift[0] < -5  # new cheeks brighter -> darken
    assert qr.skin_tone_shift(a, b, np.zeros((30, 30), np.uint8)) == (None, None)


def _face68():
    """Full synthetic iBUG-68 face (jaw, brows, nose, eyes, mouth), iod 80."""
    L = np.zeros((68, 2))
    for i in range(17):  # jaw: half ellipse from ear to ear through the chin
        t = np.pi * i / 16
        L[i] = [150 - 90 * np.cos(t), 120 + 110 * np.sin(t)]
    L[17:27] = [[80 + 15 * i, 95] for i in range(5)] + [[160 + 15 * i, 95] for i in range(5)]
    L[27:31] = [[150, 115 + 12 * i] for i in range(4)]
    L[31:36] = [[130 + 10 * i, 165] for i in range(5)]
    for c, rng in ((110, range(36, 42)), (190, range(42, 48))):
        for j, i in enumerate(rng):
            t = 2 * np.pi * j / 6
            L[i] = [c + 14 * np.cos(t), 120 + 6 * np.sin(t)]
    for j, i in enumerate(range(48, 60)):
        t = 2 * np.pi * j / 12
        L[i] = [150 - 30 * np.cos(t), 200 + 10 * np.sin(t)]
    for j, i in enumerate(range(60, 68)):
        t = 2 * np.pi * j / 8
        L[i] = [150 - 20 * np.cos(t), 200 + 4 * np.sin(t)]
    return L


def test_cheek_mask_excludes_eyes_and_mouth():
    L = _face68()
    m = qr.cheek_mask(L, (300, 300))
    assert m.sum() > 500
    for i in (36, 39, 42, 45, 48, 54, 62, 30):  # eyes, mouth, nose tip
        x, y = int(round(L[i, 0])), int(round(L[i, 1]))
        assert m[y, x] == 0, i
    assert m[160, 90] == 1 and m[160, 210] == 1  # left and right cheek
