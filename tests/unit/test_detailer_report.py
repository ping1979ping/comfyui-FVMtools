"""Tests for nodes/utils/detailer_report.py — per-part aux detailing, crop tiling,
per-image + batch status text."""

import torch

from nodes.utils.detailer_report import (
    split_aux_parts, tile_crops, detail_parts_individually, build_status_text,
)

H = W = 100


def _rect(y0, y1, x0, x1):
    m = torch.zeros(H, W)
    m[y0:y1, x0:x1] = 1.0
    return m


class TestSplitAuxParts:
    def test_empty(self):
        assert split_aux_parts(torch.zeros(H, W)) == []

    def test_single_part_returns_mask(self):
        parts = split_aux_parts(_rect(10, 30, 10, 30))
        assert len(parts) == 1

    def test_two_hands_split_in_reading_order(self):
        mask = torch.maximum(_rect(60, 80, 60, 80), _rect(60, 80, 5, 25))
        parts = split_aux_parts(mask)
        assert len(parts) == 2
        assert parts[0][70, 15] == 1.0 and parts[0][70, 70] == 0.0  # left first
        assert parts[1][70, 70] == 1.0

    def test_crumb_merged_into_nearest_part_no_pixel_lost(self):
        mask = torch.maximum(_rect(60, 80, 5, 25), _rect(60, 80, 60, 80))
        mask[85, 78] = 1.0  # 1px crumb next to the right part
        parts = split_aux_parts(mask)
        assert len(parts) == 2
        assert float(sum(p.sum() for p in parts)) == float(mask.sum())
        assert parts[1][85, 78] == 1.0

    def test_accepts_batched_mask(self):
        mask = torch.maximum(_rect(0, 10, 0, 10), _rect(50, 60, 50, 60)).unsqueeze(0)
        assert len(split_aux_parts(mask)) == 2


class TestTileCrops:
    def test_none_and_single(self):
        assert tile_crops([]) is None
        c = torch.rand(1, 64, 48, 3)
        assert tile_crops([c]) is c

    def test_grid_keeps_canvas_size_and_shows_every_crop(self):
        crops = [torch.full((1, 120, 80, 3), v) for v in (0.25, 0.5, 0.75)]
        tile = tile_crops(crops)
        assert tile.shape == (1, 120, 80, 3)
        vals = {round(float(v), 2) for v in tile.unique()}
        assert {0.25, 0.5, 0.75} <= vals


class TestDetailPartsIndividually:
    def test_one_inpaint_per_part_and_tiled(self):
        calls = []

        def fake_inpaint(image, mask, slot, cached_model=None, cached_cond=None, **kw):
            calls.append(float(mask.sum()))
            return image + 1, torch.full((1, 40, 40, 3), 0.5)

        mask = torch.maximum(_rect(60, 80, 5, 25), _rect(60, 80, 60, 80))
        img, tile, n = detail_parts_individually(fake_inpaint, torch.zeros(4), mask, {}, None)
        assert n == 2 and len(calls) == 2
        assert all(c == 400.0 for c in calls)
        assert float(img[0]) == 2.0, "second part must build on the first part's result"
        assert tile.shape == (1, 40, 40, 3)


class TestStatusText:
    def test_per_image_lines_and_batch_total(self):
        s1 = {"Ref1": {"status": "ok", "mask_type": "aux", "parts": 2},
              "Ref2": {"status": "ok", "mask_type": "head"}, "_refined": 3}
        s2 = {"Ref1": {"status": "no parts", "mask_type": "aux", "parts": 0},
              "Ref2": {"status": "ok", "mask_type": "head"}, "_refined": 1}
        text = build_status_text([s1, s2], 2, 4, elapsed_s=12)
        lines = text.split("\n")
        assert lines[0] == "Img 1/2: Ref1 aux(2) · Ref2 head → 3 refined"
        assert lines[1] == "Img 2/2: Ref1 aux(0) · Ref2 head → 1 refined"
        assert lines[2] == "Batch: 2/2 img · 4 refined · Ref1 1/2 (2 parts) · Ref2 2/2 · 12s"

    def test_running_image_and_cn_info(self):
        running = {"Ref1": {"status": "ok", "mask_type": "face"}}
        text = build_status_text([running], 3, 1, cn_info="depth s=0.5")
        assert text.split("\n")[0].endswith(" …")
        assert "Batch: 0/3 img" in text and "[depth s=0.5]" in text

    def test_long_batch_is_truncated(self):
        s = {"Ref1": {"status": "ok", "mask_type": "head"}, "_refined": 1}
        lines = build_status_text([dict(s) for _ in range(20)], 20, 20).split("\n")
        assert len(lines) == 9  # 7 images + "… +13 more" + batch line
        assert lines[7] == "… +13 more images"
        assert lines[-1].startswith("Batch: 20/20 img")
