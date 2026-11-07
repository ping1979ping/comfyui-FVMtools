"""Regional LoRA: box masks and the hook/mask split (comfy.hooks faked)."""

import sys
import types

import torch


class _Group:
    def __init__(self, items=()):
        self.items = list(items)

    def clone_and_combine(self, other):
        return _Group(self.items + other.items)


_fake_hooks = types.ModuleType("comfy.hooks")
_fake_hooks.HookGroup = _Group
_fake_hooks.create_hook_lora = lambda lora, strength_model, strength_clip: _Group([(lora, strength_model)])
_fake_hooks.set_conds_props = lambda conds, strength, set_cond_area, mask=None, hooks=None: [
    [[c[0], {**c[1], "mask": mask, "hooks": hooks}] for c in conds[0]]]
sys.modules["comfy.hooks"] = _fake_hooks
sys.modules["comfy"].hooks = _fake_hooks

import nodes.jb.moss_regional_lora as mod  # noqa: E402
from nodes.jb.moss_regional_lora import (  # noqa: E402
    FVM_MOSS_RegionalLoRA, box_mask, parse_rows, slot_masks,
)


DOC = {"width": 100, "height": 50, "boxes": [
    {"slot": "character_1", "rect": [0.0, 0.0, 0.5, 1.0]},
    {"slot": "character_2", "rect": [0.6, 0.2, 0.9, 0.8]},
]}


def test_box_mask_hard_and_feathered():
    hard = box_mask([0.2, 0.2, 0.6, 0.6], 10, 10)
    assert hard.shape == (10, 10)
    assert hard[3, 3] == 1 and hard[0, 0] == 0 and hard.sum() == 16
    soft = box_mask([0.2, 0.2, 0.6, 0.6], 10, 10, feather=2)
    assert soft[3, 3] == 1 and 0 < soft[3, 1] < 1 and soft[3, 0] < soft[3, 1]
    grown = box_mask([0.2, 0.2, 0.6, 0.6], 10, 10, expand=1)
    assert grown.sum() == 36


def test_parse_rows_stacks_and_skips():
    rows = [("character_1", "a.safetensors", 0.75), ("character_1", "b.safetensors", 0.3),
            ("character_2", "None", 1.0), ("", "c.safetensors", 1.0), ("plant_1", "d.safetensors", 0)]
    assert parse_rows(rows) == {"character_1": [("a.safetensors", 0.75), ("b.safetensors", 0.3)]}


def test_slot_masks_rest_is_complement():
    masks, rest = slot_masks(DOC, ["character_1", "character_9"], feather=0, expand=0)
    assert list(masks) == ["character_1"]               # no box for character_9
    assert torch.equal(rest, 1 - masks["character_1"])
    assert rest[:, :50].sum() == 0 and rest[:, 50:].min() == 1


def test_apply_splits_positive_and_negative(monkeypatch):
    monkeypatch.setattr(mod.folder_paths, "get_full_path", lambda kind, name: "/l/" + name, raising=False)
    monkeypatch.setattr(mod._LoraFileCache, "get", classmethod(lambda cls, p: {"path": p}))
    pos = [[torch.zeros(1), {"tag": "pos"}]]
    neg = [[torch.zeros(1), {"tag": "neg"}]]
    out_pos, out_neg, preview, report = FVM_MOSS_RegionalLoRA().apply(
        positive=pos, negative=neg, moss_boxes=DOC, feather=4, expand=0,
        slot_1="character_1", lora_1="jg.safetensors", strength_1=0.75,
        slot_2="character_1", lora_2="body.safetensors", strength_2=0.3,
        slot_3="character_2", lora_3="None", strength_3=1.0,
        slot_4="", lora_4="None", strength_4=1.0)
    assert len(out_pos) == 2 and len(out_neg) == 2       # rest + one slot
    rest, box = out_pos
    assert rest[1]["hooks"] is None and rest[1]["tag"] == "pos"
    assert [h[0]["path"] for h in box[1]["hooks"].items] == ["/l/jg.safetensors", "/l/body.safetensors"]
    assert box[1]["mask"].shape == (50, 100) and out_neg[1][1]["tag"] == "neg"
    assert preview.shape == (1, 50, 100, 3)
    assert "character_1: jg.safetensors @0.75, body.safetensors @0.3" in report


def test_apply_passes_through_without_rows():
    pos = [[torch.zeros(1), {}]]
    out = FVM_MOSS_RegionalLoRA().apply(
        positive=pos, moss_boxes=DOC, feather=0, expand=0,
        **{f"{k}_{i}": v for i in range(1, 5) for k, v in (("slot", ""), ("lora", "None"), ("strength", 1.0))})
    assert out[0] is pos and out[1] is None
    assert "passed through" in out[3]
