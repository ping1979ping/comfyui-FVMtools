"""Unit tests for the Sign Tools node layer (selector / proposer / detailer / options).

These cover the pure helpers and the node contracts. The sampling path itself is
not exercised here — that needs a real ComfyUI with a loaded model.
"""

import numpy as np
import pytest
import torch

from nodes.signs.selector import (
    SignSelectorSAM3, _bbox_from_mask, _mask_iou, _short_side_px, _crop_to_canvas,
    CROP_CANVAS,
)
from nodes.signs.proposer import SignTextProposer, _parse_overrides, _parse_fallbacks
from nodes.signs.detailer import SignDetailer, _fuzzy_match, SOFTEN_PROMPT
from nodes.signs.options import SignOptions, SIGN_DEFAULTS, _parse_class_map
from nodes.signs import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from core.signs.classes import all_class_names


def _rect_mask(h=256, w=256, x1=40, y1=60, x2=180, y2=110):
    m = np.zeros((h, w), dtype=np.float32)
    m[y1:y2, x1:x2] = 1.0
    return m


class TestSelectorHelpers:

    def test_bbox_from_mask(self):
        m = _rect_mask()
        assert _bbox_from_mask(m) == [40, 60, 179, 109]

    def test_bbox_empty_mask_is_none(self):
        assert _bbox_from_mask(np.zeros((32, 32), dtype=np.float32)) is None

    def test_mask_iou_identical_and_disjoint(self):
        a = _rect_mask()
        assert _mask_iou(a, a) == pytest.approx(1.0)
        b = _rect_mask(x1=200, y1=200, x2=240, y2=240)
        assert _mask_iou(a, b) == pytest.approx(0.0)

    def test_mask_iou_partial_overlap(self):
        a = _rect_mask(x1=0, y1=0, x2=100, y2=100)
        b = _rect_mask(x1=50, y1=0, x2=150, y2=100)
        assert 0.3 < _mask_iou(a, b) < 0.4

    def test_mask_iou_two_empty_masks_is_zero_not_nan(self):
        z = np.zeros((16, 16), dtype=np.float32)
        assert _mask_iou(z, z) == 0.0

    def test_short_side_px_matches_rect_height(self):
        m = _rect_mask(y1=60, y2=110)  # 50 px tall, 140 px wide
        assert _short_side_px(m) == pytest.approx(50, abs=2)

    def test_short_side_px_empty_mask(self):
        assert _short_side_px(np.zeros((32, 32), dtype=np.float32)) == 0

    def test_crop_to_canvas_shape_and_content(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[60:110, 40:180] = 200
        crop = _crop_to_canvas(img, [40, 60, 179, 109])
        assert crop.shape == (CROP_CANVAS, CROP_CANVAS, 3)
        assert crop.max() > 150, "the bright region must survive the crop"

    def test_crop_to_canvas_degenerate_bbox(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        crop = _crop_to_canvas(img, [10, 10, 10, 10])
        assert crop.shape == (CROP_CANVAS, CROP_CANVAS, 3)

    def test_merge_overlaps_collapses_duplicates(self):
        node = SignSelectorSAM3()
        m = _rect_mask()
        raw = [
            {"class": "sign", "prompt": "sign", "mask": m, "score": 0.9, "bbox": None},
            {"class": "poster", "prompt": "poster", "mask": m.copy(), "score": 0.5, "bbox": None},
        ]
        kept = node._merge_overlaps(raw, merge_iou=0.5)
        assert len(kept) == 1
        assert kept[0]["class"] == "sign", "the higher-scoring detection keeps its class"
        assert "poster:poster" in kept[0]["also_matched"]

    def test_merge_overlaps_keeps_distinct_regions(self):
        node = SignSelectorSAM3()
        raw = [
            {"class": "sign", "prompt": "sign", "mask": _rect_mask(), "score": 0.9, "bbox": None},
            {"class": "label", "prompt": "label",
             "mask": _rect_mask(x1=200, y1=200, x2=250, y2=250), "score": 0.8, "bbox": None},
        ]
        assert len(node._merge_overlaps(raw, merge_iou=0.5)) == 2

    def test_collect_prompts_covers_enabled_classes_only(self):
        node = SignSelectorSAM3()
        toggles = {f"class_{n}": False for n in all_class_names()}
        toggles["class_sign"] = True
        jobs = node._collect_prompts(toggles, "", 1.0)
        assert jobs and all(j[0] == "sign" for j in jobs)

    def test_collect_prompts_threshold_scale_and_custom(self):
        node = SignSelectorSAM3()
        toggles = {f"class_{n}": False for n in all_class_names()}
        jobs = node._collect_prompts(toggles, "neon sign:0.2", 2.0)
        assert len(jobs) == 1
        assert jobs[0][0] == "custom"
        assert jobs[0][2] == pytest.approx(0.4)

    def test_collect_prompts_threshold_is_clamped(self):
        node = SignSelectorSAM3()
        toggles = {f"class_{n}": False for n in all_class_names()}
        toggles["class_plate"] = True
        jobs = node._collect_prompts(toggles, "", 2.0)
        assert all(0.05 <= j[2] <= 0.99 for j in jobs)

    def test_build_regions_flags_too_small(self):
        node = SignSelectorSAM3()
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        thin = _rect_mask(y1=60, y2=68)  # only 8 px tall
        raw = [{"class": "sign", "prompt": "sign", "mask": thin, "score": 0.9, "bbox": None}]
        regions = node._build_regions(raw, img, 0, 24, 0.0, None)
        assert len(regions) == 1
        assert regions[0]["too_small"] is True

    def test_build_regions_respects_restrict_mask(self):
        node = SignSelectorSAM3()
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        raw = [{"class": "sign", "prompt": "sign", "mask": _rect_mask(), "score": 0.9, "bbox": None}]
        elsewhere = _rect_mask(x1=200, y1=200, x2=250, y2=250)
        assert node._build_regions(raw, img, 0, 4, 0.0, elsewhere) == []

    def test_build_regions_area_ratio_gate(self):
        node = SignSelectorSAM3()
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        tiny = _rect_mask(x1=0, y1=0, x2=4, y2=4)
        raw = [{"class": "sign", "prompt": "sign", "mask": tiny, "score": 0.9, "bbox": None}]
        assert node._build_regions(raw, img, 0, 1, 0.5, None) == []

    def test_sort_orders(self):
        node = SignSelectorSAM3()
        regions = [
            {"area_px": 10, "score": 0.9, "bbox": [100, 5, 110, 15]},
            {"area_px": 99, "score": 0.1, "bbox": [5, 100, 15, 110]},
        ]
        assert node._sort_regions(regions, "area_desc")[0]["area_px"] == 99
        assert node._sort_regions(regions, "score_desc")[0]["score"] == 0.9
        assert node._sort_regions(regions, "left_right")[0]["bbox"][0] == 5
        assert node._sort_regions(regions, "top_down")[0]["bbox"][1] == 5

    def test_draw_preview_marks_the_image(self):
        node = SignSelectorSAM3()
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        regions = [{
            "class": "sign", "mask": _rect_mask(), "bbox": [40, 60, 179, 109],
            "cluster_id": -1, "too_small": False, "height_px": 50,
            "slop": {"verdict": "clean", "score": 0.1},
        }]
        out = node._draw_preview(img, regions)
        assert out.shape == img.shape
        assert out.max() > 0, "the preview must actually draw something"


class TestProposerHelpers:

    @pytest.mark.parametrize("line,idx,text", [
        ("3: ACHTUNG", 2, "ACHTUNG"),
        ("1 = Café Mozart", 0, "Café Mozart"),
        ("7 OPEN", 6, "OPEN"),
        ("  2 :  spaced  ", 1, "spaced"),
    ])
    def test_parse_overrides_accepted_forms(self, line, idx, text):
        assert _parse_overrides(line) == {idx: text}

    def test_parse_overrides_ignores_junk(self):
        assert _parse_overrides("") == {}
        assert _parse_overrides("# a comment\n\nnot-a-number: x") == {}
        assert _parse_overrides("0: zero") == {}, "1-based indices only"

    def test_parse_overrides_multiline(self):
        assert _parse_overrides("1: A\n2: B") == {0: "A", 1: "B"}

    def test_parse_fallbacks_keyed_and_plain(self):
        keyed, plain = _parse_fallbacks("sign: OPEN\nCLOSED\nlabel: Merlot")
        assert keyed == {"sign": "OPEN", "label": "Merlot"}
        assert plain == ["CLOSED"]

    def test_parse_fallbacks_empty(self):
        assert _parse_fallbacks("") == ({}, [])

    def test_neighbors_returns_closest_first(self):
        node = SignTextProposer()
        target = {"bbox": [0, 0, 10, 10]}
        near = {"bbox": [20, 0, 30, 10]}
        far = {"bbox": [500, 500, 510, 510]}
        result = node._neighbors([target, far, near], target, limit=2)
        assert result[0] is near


class TestDetailerHelpers:

    def test_fuzzy_match_exact_and_case_insensitive(self):
        assert _fuzzy_match("OPEN", "open") == pytest.approx(1.0)
        assert _fuzzy_match("COFFEE SHOP", "coffeeshop") == pytest.approx(1.0)

    def test_fuzzy_match_unrelated_is_low(self):
        assert _fuzzy_match("OPEN", "BAHNHOF") < 0.4

    def test_fuzzy_match_empty_is_zero(self):
        assert _fuzzy_match("", "OPEN") == 0.0
        assert _fuzzy_match("OPEN", "") == 0.0

    def test_clamp_target_leaves_large_regions_alone(self):
        node = SignDetailer()
        region = {"bbox": [0, 0, 512, 512]}
        assert node._clamp_target(region, 1024, 1024, 8.0) == (1024, 1024)

    def test_clamp_target_shrinks_for_tiny_regions(self):
        node = SignDetailer()
        region = {"bbox": [0, 0, 16, 16]}  # 64x upscale to 1024 without a cap
        tw, th = node._clamp_target(region, 1024, 1024, 8.0)
        assert tw < 1024 and th < 1024
        assert tw % 8 == 0 and th % 8 == 0
        assert tw / 16 == pytest.approx(8.0, abs=0.5)

    def test_clamp_target_never_below_floor(self):
        node = SignDetailer()
        region = {"bbox": [0, 0, 2, 2]}
        tw, th = node._clamp_target(region, 1024, 1024, 1.0)
        assert tw >= 64 and th >= 64

    def test_soften_prompt_mentions_unreadable(self):
        assert "read" in SOFTEN_PROMPT.lower()


class TestOptions:

    def test_parse_class_map_known_classes_only(self):
        parsed = _parse_class_map("plate: 0.95\nnonsense: 0.5\ngarment_print: 0.7")
        assert parsed == {"plate": 0.95, "garment_print": 0.7}

    def test_parse_class_map_bad_values_ignored(self):
        assert _parse_class_map("plate: not-a-number") == {}

    def test_parse_class_map_empty(self):
        assert _parse_class_map("") == {}

    def test_execute_returns_merged_dict(self):
        node = SignOptions()
        (opts,) = node.execute(
            cfg=1.5, negative_prompt="bad", context_expand_factor=1.4, output_padding=16,
            mask_fill_holes=False, denoise_progression="0.8|0.4", steps_progression="8|4",
            class_denoise="plate: 0.95", skip_classes="screen, bogus",
            uppercase=True, margin_ratio=0.12,
        )
        assert opts["cfg"] == 1.5
        assert opts["class_denoise"] == {"plate": 0.95}
        assert opts["class_skip"] == {"screen"}, "unknown class names are dropped"
        assert opts["uppercase"] is True

    def test_defaults_cover_every_key_the_detailer_reads(self):
        needed = {"cfg", "negative_prompt", "context_expand_factor", "output_padding",
                  "mask_fill_holes", "denoise_progression", "steps_progression",
                  "class_denoise", "class_skip", "uppercase", "margin_ratio"}
        assert needed <= set(SIGN_DEFAULTS)


class TestSurfaceControl:
    """Rendering the text AND the surface it sits on (a yellow post-it, say)."""

    @pytest.mark.parametrize("raw,expected", [
        ("#ffe680", (255, 230, 128)),
        ("ffe680", (255, 230, 128)),
        ("#FFE680", (255, 230, 128)),
        ("#fe8", (255, 238, 136)),
        ("255,230,128", (255, 230, 128)),
        (" 255 , 230 , 128 ", (255, 230, 128)),
    ])
    def test_parse_hex_rgb_accepted_forms(self, raw, expected):
        from nodes.signs.options import parse_hex_rgb
        assert parse_hex_rgb(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", None, "nonsense", "#12345", "1,2", "1,2,3,4"])
    def test_parse_hex_rgb_rejects_junk(self, raw):
        from nodes.signs.options import parse_hex_rgb
        assert parse_hex_rgb(raw) is None

    def test_parse_hex_rgb_honours_the_fallback(self):
        from nodes.signs.options import parse_hex_rgb
        assert parse_hex_rgb("", fallback=(1, 2, 3)) == (1, 2, 3)

    def test_parse_hex_rgb_clamps_out_of_range_triplets(self):
        from nodes.signs.options import parse_hex_rgb
        assert parse_hex_rgb("300,-20,128") == (255, 0, 128)

    def test_options_carries_the_prompt_suffix(self):
        node = SignOptions()
        (opts,) = node.execute(
            cfg=1.0, negative_prompt="", context_expand_factor=1.3, output_padding=32,
            mask_fill_holes=True, denoise_progression="", steps_progression="",
            class_denoise="", skip_classes="", uppercase=False, margin_ratio=0.08,
            prompt_suffix="  on a bright yellow post-it note  ")
        assert opts["prompt_suffix"] == "on a bright yellow post-it note"

    def test_prompt_suffix_defaults_to_empty(self):
        assert SIGN_DEFAULTS["prompt_suffix"] == ""

    def test_detailer_exposes_both_colour_overrides(self):
        req = SignDetailer.INPUT_TYPES()["required"]
        for name in ("glyph_plate_color", "glyph_ink_color"):
            assert name in req and req[name][0] == "STRING"
            assert req[name][1]["default"] == "", "an empty override must mean 'sample it'"

    def test_apply_glyph_prefers_the_override_over_the_sample(self):
        """The forced colour is what makes a grey scrap become a yellow post-it."""
        node = SignDetailer()
        img = torch.full((200, 300, 3), 0.55)
        region = {"index": 0, "class": "paper", "mask": _rect_mask(200, 300, 40, 40, 260, 160),
                  "proposal": {"font_hint": ""}}
        out, glyph = node._apply_glyph(
            img, region, "Telefon Nummer 1234", "<auto>", 1.0,
            autocolor=True, uppercase=False, margin_ratio=0.1,
            ink_override=(20, 20, 20), plate_override=(255, 230, 128))
        assert glyph is not None
        painted = out.numpy()[60:140, 60:240].reshape(-1, 3)
        # the forced plate is markedly warmer than the grey it replaced
        assert painted[:, 0].mean() > painted[:, 2].mean() + 0.15

    def test_apply_glyph_without_overrides_keeps_the_sampled_scheme(self):
        node = SignDetailer()
        img = torch.full((200, 300, 3), 0.55)
        region = {"index": 0, "class": "paper", "mask": _rect_mask(200, 300, 40, 40, 260, 160),
                  "proposal": {"font_hint": ""}}
        out, glyph = node._apply_glyph(
            img, region, "TEST", "<auto>", 1.0,
            autocolor=True, uppercase=False, margin_ratio=0.1)
        assert glyph is not None
        painted = out.numpy()[60:140, 60:240].reshape(-1, 3)
        spread = abs(painted[:, 0].mean() - painted[:, 2].mean())
        assert spread < 0.10, "a grey source must stay neutral when nothing is forced"


class TestAvoidRepeats:
    """Similar motifs must not all get the same text.

    Measured on four near-identical shopfronts against the real model:
    1/4 distinct texts with the flag off, 4/4 with it on.
    """

    def _regions(self, n=3, cluster=-1):
        out = []
        for i in range(n):
            out.append({
                "index": i, "class": "sign", "batch_index": 0,
                "bbox": [i * 50, 0, i * 50 + 40, 40], "mask": _rect_mask(),
                "crop": np.zeros((64, 64, 3), dtype=np.uint8),
                "cluster_id": cluster, "too_small": False,
                "slop": {"verdict": "slop", "score": 0.9, "ocr_text": ""},
                "proposal": None,
            })
        return out

    def _run(self, regions, monkeypatch, **kw):
        """Capture the avoid_texts handed to each call."""
        seen = []
        counter = {"n": 0}

        # Distinct on purpose: 'TEXT1'/'TEXT2' differ by one character and the
        # duplicate detector rightly treats them as the same answer.
        words = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]

        def fake_propose(**kwargs):
            seen.append(list(kwargs.get("avoid_texts") or []))
            counter["n"] += 1
            return {"text": words[(counter["n"] - 1) % len(words)],
                    "style": "", "font_hint": "",
                    "legible_original": 0.0, "confidence": 1.0,
                    "ok": True, "error": None, "source": "vlm"}

        monkeypatch.setattr("nodes.signs.proposer.propose_text", fake_propose)
        monkeypatch.setattr("nodes.signs.proposer.probe",
                            lambda *a, **k: {"reachable": True, "models": [], "error": None})
        node = SignTextProposer()
        data = {"regions": regions, "image_shape": (256, 256), "batch_size": 1}
        out, _, _ = node.execute(sign_data=data, image=torch.rand(1, 256, 256, 3), **kw)
        return out, seen

    def test_each_call_learns_the_previous_answers(self, monkeypatch):
        _, seen = self._run(self._regions(3), monkeypatch, avoid_repeats=True)
        assert seen[0] == []
        assert seen[1] == ["ALPHA"]
        assert sorted(seen[2]) == ["ALPHA", "BRAVO"]

    def test_flag_off_sends_nothing(self, monkeypatch):
        _, seen = self._run(self._regions(3), monkeypatch, avoid_repeats=False)
        assert all(s == [] for s in seen)

    def test_cluster_siblings_are_not_pushed_apart(self, monkeypatch):
        """Within a cluster the SAME text is the point — only strangers differ."""
        _, seen = self._run(self._regions(3, cluster=7), monkeypatch,
                            avoid_repeats=True, one_call_per_cluster=False)
        for s in seen:
            assert s == [], "a sibling must not be told to avoid its own cluster's text"

    def test_manual_overrides_join_the_avoid_list(self, monkeypatch):
        """Otherwise the model proposes exactly what you just typed next door."""
        _, seen = self._run(self._regions(2), monkeypatch,
                            avoid_repeats=True, manual_override="1: ACHTUNG")
        assert seen and "ACHTUNG" in seen[0]

    def test_kept_legible_text_joins_the_avoid_list(self, monkeypatch):
        regions = self._regions(2)
        regions[0]["slop"] = {"verdict": "clean", "score": 0.1, "ocr_text": "BAHNHOF"}
        _, seen = self._run(regions, monkeypatch, avoid_repeats=True, skip_legible=True)
        assert seen and "BAHNHOF" in seen[0]

    def test_default_is_on(self):
        spec = SignTextProposer.INPUT_TYPES()["required"]["avoid_repeats"]
        assert spec[0] == "BOOLEAN" and spec[1]["default"] is True


class TestStyleIsVisible:
    """style is what carries the surface into the diffusion prompt, so it has to
    be readable in the node's own output — otherwise an odd render is unexplainable.
    """

    def _region(self, idx=0, cluster=-1):
        return {
            "index": idx, "class": "paper", "batch_index": 0,
            "bbox": [0, 0, 40, 40], "mask": _rect_mask(),
            "crop": np.zeros((64, 64, 3), dtype=np.uint8),
            "cluster_id": cluster, "too_small": False,
            "slop": {"verdict": "slop", "score": 0.9, "ocr_text": ""},
            "proposal": None,
        }

    def test_texts_table_has_a_header_naming_the_columns(self):
        node = SignTextProposer()
        out, texts, _ = node.execute(
            sign_data={"regions": [self._region()], "image_shape": (256, 256), "batch_size": 1},
            image=torch.rand(1, 256, 256, 3), enabled=False, fallback_texts="paper: NOTIZ")
        header = texts.splitlines()[0].split("\t")
        assert header == ["#", "class", "source", "text", "style", "font_hint"]

    def test_texts_table_carries_the_style_column(self):
        node = SignTextProposer()
        region = self._region()
        out, texts, _ = node.execute(
            sign_data={"regions": [region], "image_shape": (256, 256), "batch_size": 1},
            image=torch.rand(1, 256, 256, 3), enabled=False, fallback_texts="paper: NOTIZ")
        # inject a style the way a model answer would, then re-render the table
        out["regions"][0]["proposal"]["style"] = "black ink on yellow sticky note"
        out2, texts2, _ = node.execute(
            sign_data=out, image=torch.rand(1, 256, 256, 3), enabled=False,
            manual_override="", fallback_texts="paper: NOTIZ")
        row = texts.splitlines()[1].split("\t")
        assert len(row) == 6, "every row must fill all six columns"

    def test_style_appears_in_the_report(self):
        node = SignTextProposer()
        region = self._region()
        region["proposal"] = None
        out, _, _ = node.execute(
            sign_data={"regions": [region], "image_shape": (256, 256), "batch_size": 1},
            image=torch.rand(1, 256, 256, 3), enabled=False, fallback_texts="paper: NOTIZ")
        # a fallback has no style; a cluster sibling inheriting one must show it
        out["regions"][0]["proposal"]["style"] = "black ink on yellow sticky note"
        out["regions"][0]["cluster_id"] = 3
        second = self._region(idx=1, cluster=3)
        data = {"regions": [out["regions"][0], second],
                "image_shape": (256, 256), "batch_size": 1}
        _, _, report = node.execute(
            sign_data=data, image=torch.rand(1, 256, 256, 3), enabled=False,
            fallback_texts="paper: NOTIZ")
        assert "sticky note" in report or "inherits cluster" in report

    def test_empty_region_list_still_yields_no_table(self):
        node = SignTextProposer()
        _, texts, _ = node.execute(
            sign_data={"regions": []}, image=torch.rand(1, 64, 64, 3), enabled=False)
        assert texts == "", "a header alone would be noise when there is nothing to show"


class TestPromptAssembly:
    """The model reports the surface in `style`; the template must not swallow it."""

    def test_every_template_separates_style_from_what_follows(self):
        """Without a separator two phrases merge into nonsense:
        '...on yellow sticky note clean printed typography on paper'.
        """
        from core.signs.classes import SIGN_CLASSES, FALLBACK_CLASS
        merged = []
        for name, cfg in list(SIGN_CLASSES.items()) + [("<fallback>", FALLBACK_CLASS)]:
            tpl = cfg["prompt_template"]
            if "{style}" not in tpl:
                continue
            tail = tpl.split("{style}", 1)[1]
            if tail.strip() and not tail.lstrip().startswith(","):
                merged.append(name)
        assert not merged, f"templates run {{style}} into the next phrase: {merged}"

    def test_style_carries_the_surface_into_the_prompt(self):
        from core.signs.classes import build_prompt
        prompt = build_prompt("paper", "Telefon Nummer 1234",
                              "black ink on yellow sticky note")
        assert "yellow sticky note" in prompt
        assert "Telefon Nummer 1234" in prompt
        assert "note clean" not in prompt and "note legible" not in prompt

    @pytest.mark.parametrize("cls", ["sign", "label", "paper", "screen", "poster"])
    def test_empty_style_leaves_no_double_separator(self, cls):
        """Without a language model there is no style, and the prompt must still read."""
        from core.signs.classes import build_prompt
        prompt = build_prompt(cls, "OPEN", "")
        assert ",," not in prompt
        assert ", ," not in prompt
        assert "  " not in prompt
        assert not prompt.strip().endswith(",")

    def test_paper_template_stays_neutral_about_the_medium(self):
        """The class covers sticky notes, receipts and handwriting too, so a fixed
        'printed typography' would contradict the surface the model reports."""
        from core.signs.classes import SIGN_CLASSES
        tpl = SIGN_CLASSES["paper"]["prompt_template"].lower()
        assert "printed" not in tpl

    def test_paper_instruction_asks_for_the_surface(self):
        from core.signs.classes import SIGN_CLASSES
        low = SIGN_CLASSES["paper"]["vlm_instruction"].lower()
        assert "style field" in low


class TestNodeContracts:

    @pytest.mark.parametrize("cls", [SignSelectorSAM3, SignTextProposer, SignDetailer, SignOptions])
    def test_required_class_attributes(self, cls):
        assert cls.CATEGORY.startswith("FVM Tools/Text")
        assert isinstance(cls.RETURN_TYPES, tuple)
        assert len(cls.RETURN_NAMES) == len(cls.RETURN_TYPES)
        assert hasattr(cls, cls.FUNCTION), "FUNCTION must name a real method"

    @pytest.mark.parametrize("cls", [SignSelectorSAM3, SignTextProposer, SignDetailer, SignOptions])
    def test_input_types_shape(self, cls):
        spec = cls.INPUT_TYPES()
        assert "required" in spec
        for section in ("required", "optional"):
            for name, definition in spec.get(section, {}).items():
                assert isinstance(definition, tuple) and len(definition) in (1, 2), \
                    f"{cls.__name__}.{name} has a malformed input definition"

    def test_selector_exposes_a_toggle_per_class(self):
        required = SignSelectorSAM3.INPUT_TYPES()["required"]
        for name in all_class_names():
            assert f"class_{name}" in required
            assert required[f"class_{name}"][0] == "BOOLEAN"

    def test_selector_pipes_sign_data_into_proposer_and_detailer(self):
        assert "SIGN_DATA" in SignSelectorSAM3.RETURN_TYPES
        assert SignTextProposer.INPUT_TYPES()["required"]["sign_data"][0] == "SIGN_DATA"
        assert SignDetailer.INPUT_TYPES()["required"]["sign_data"][0] == "SIGN_DATA"

    def test_options_type_matches_detailer_socket(self):
        assert SignOptions.RETURN_TYPES == ("SIGN_OPTIONS",)
        assert SignDetailer.INPUT_TYPES()["optional"]["sign_options"][0] == "SIGN_OPTIONS"

    def test_detailer_denoise_default_is_high(self):
        default = SignDetailer.INPUT_TYPES()["required"]["denoise"][1]["default"]
        assert default >= 0.8, "low denoise lets the original garbled strokes bleed through"

    def test_glyph_guidance_defaults_on(self):
        spec = SignDetailer.INPUT_TYPES()["required"]["glyph_guidance"]
        assert spec[1]["default"] == "init"
        assert "off" in spec[0]

    def test_registration_mappings_are_consistent(self):
        assert set(NODE_CLASS_MAPPINGS) == set(NODE_DISPLAY_NAME_MAPPINGS)
        assert len(NODE_CLASS_MAPPINGS) == 4
        for key in NODE_CLASS_MAPPINGS:
            assert key.startswith("FVM_Sign")


class TestProposerExecute:
    """The proposer must work end to end without LM Studio running."""

    def _sign_data(self, n=2):
        regions = []
        for i in range(n):
            regions.append({
                "index": i, "class": "sign", "batch_index": 0,
                "bbox": [10 * i, 10, 10 * i + 40, 40],
                "mask": _rect_mask(), "crop": np.zeros((64, 64, 3), dtype=np.uint8),
                "cluster_id": -1, "too_small": False,
                "slop": {"verdict": "slop", "score": 0.9, "ocr_text": "SHOPPINQ"},
                "proposal": None,
            })
        return {"regions": regions, "image_shape": (256, 256), "batch_size": 1}

    def test_disabled_model_uses_fallbacks(self):
        node = SignTextProposer()
        data = self._sign_data()
        image = torch.rand(1, 256, 256, 3)
        out_data, texts, report = node.execute(
            sign_data=data, image=image, enabled=False,
            fallback_texts="sign: OPEN", manual_override="",
        )
        assert all(r["proposal"]["text"] == "OPEN" for r in out_data["regions"])
        assert all(r["proposal"]["source"] == "fallback" for r in out_data["regions"])
        assert "OPEN" in texts

    def test_manual_override_wins(self):
        node = SignTextProposer()
        data = self._sign_data()
        image = torch.rand(1, 256, 256, 3)
        out_data, _, _ = node.execute(
            sign_data=data, image=image, enabled=False,
            fallback_texts="sign: OPEN", manual_override="1: ACHTUNG",
        )
        assert out_data["regions"][0]["proposal"]["text"] == "ACHTUNG"
        assert out_data["regions"][0]["proposal"]["source"] == "manual"
        assert out_data["regions"][1]["proposal"]["text"] == "OPEN"

    def test_falls_back_to_ocr_text_when_nothing_else_given(self):
        node = SignTextProposer()
        data = self._sign_data(n=1)
        image = torch.rand(1, 256, 256, 3)
        out_data, _, _ = node.execute(sign_data=data, image=image, enabled=False)
        assert out_data["regions"][0]["proposal"]["text"] == "SHOPPINQ"

    def test_skip_legible_keeps_existing_text(self):
        node = SignTextProposer()
        data = self._sign_data(n=1)
        data["regions"][0]["slop"] = {"verdict": "clean", "score": 0.1, "ocr_text": "BAHNHOF"}
        image = torch.rand(1, 256, 256, 3)
        out_data, _, _ = node.execute(
            sign_data=data, image=image, enabled=False, skip_legible=True)
        assert out_data["regions"][0]["proposal"]["source"] == "kept"
        assert out_data["regions"][0]["proposal"]["text"] == "BAHNHOF"

    def test_empty_sign_data_is_safe(self):
        node = SignTextProposer()
        image = torch.rand(1, 64, 64, 3)
        out_data, texts, report = node.execute(
            sign_data={"regions": []}, image=image, enabled=False)
        assert out_data["regions"] == []
        assert texts == ""
        assert "0 region" in report


class TestRegressions:
    """Bugs found by the end-to-end run — locked down so they cannot return."""

    def test_failed_proposal_is_not_cached_for_the_cluster(self):
        """A region that produced no text must not poison its cluster siblings.

        Seen live: the cluster representative timed out, the empty proposal was
        cached, and both sibling bottles inherited an empty text and were skipped.
        """
        node = SignTextProposer()
        regions = []
        for i in range(3):
            regions.append({
                "index": i, "class": "label", "batch_index": 0,
                "bbox": [10 * i, 10, 10 * i + 40, 40], "mask": _rect_mask(),
                "crop": np.zeros((64, 64, 3), dtype=np.uint8),
                "cluster_id": 0, "too_small": False,
                # no OCR text either, so nothing can rescue the first region
                "slop": {"verdict": "slop", "score": 0.9, "ocr_text": ""},
                "proposal": None,
            })
        image = torch.rand(1, 256, 256, 3)
        # First region gets nothing (no fallback for its class); the others must
        # still fall back on their own rather than inheriting the emptiness.
        out, _, _ = node.execute(
            sign_data={"regions": regions}, image=image, enabled=False,
            fallback_texts="", manual_override="2: MERLOT")
        assert out["regions"][0]["proposal"]["text"] == ""
        assert out["regions"][1]["proposal"]["text"] == "MERLOT"
        assert out["regions"][2]["proposal"]["source"] != "cluster" or \
            out["regions"][2]["proposal"]["text"] != "", \
            "an empty proposal must never be inherited"

    def test_successful_proposal_is_still_shared_across_the_cluster(self):
        node = SignTextProposer()
        regions = []
        for i in range(3):
            regions.append({
                "index": i, "class": "label", "batch_index": 0,
                "bbox": [0, 0, 40, 40], "mask": _rect_mask(),
                "crop": np.zeros((64, 64, 3), dtype=np.uint8),
                "cluster_id": 0, "too_small": False,
                "slop": {"verdict": "slop", "score": 0.9, "ocr_text": ""},
                "proposal": None,
            })
        image = torch.rand(1, 256, 256, 3)
        out, _, _ = node.execute(
            sign_data={"regions": regions}, image=image, enabled=False,
            fallback_texts="label: RESERVE 2019")
        texts = [r["proposal"]["text"] for r in out["regions"]]
        assert texts == ["RESERVE 2019"] * 3
        assert out["regions"][1]["proposal"]["source"] == "cluster"

    def test_missing_ocr_backend_does_not_mark_everything_as_slop(self):
        """'OCR is not installed' is not the same finding as 'OCR read nothing'.

        Scoring anyway makes every region look like a pseudo-glyph hit, because an
        absent backend returns empty text for all of them.
        """
        node = SignSelectorSAM3()
        regions = [{
            "class": "sign", "mask": _rect_mask(), "bbox": [40, 60, 179, 109],
            "slop": {"score": 0.0, "verdict": "unknown", "ocr_text": "",
                     "ocr_conf": 0.0, "signals": {}},
        }]
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        node._score_regions(regions, img, "ocr", "auto", 0.5, has_backend=False)
        assert regions[0]["slop"]["verdict"] == "unknown"
        assert regions[0]["slop"]["score"] == 0.0
        assert regions[0]["slop"]["needs_fix"] is True, \
            "unknown must still reach the detailer, just not as a confident slop verdict"

    def test_slop_mode_off_marks_regions_clean(self):
        node = SignSelectorSAM3()
        regions = [{"class": "sign", "mask": _rect_mask(), "bbox": [0, 0, 10, 10],
                    "slop": {"score": 0.0, "verdict": "unknown"}}]
        node._score_regions(regions, np.zeros((256, 256, 3), np.uint8),
                            "off", "auto", 0.5, has_backend=True)
        assert regions[0]["slop"]["verdict"] == "clean"
        assert regions[0]["slop"]["needs_fix"] is False

    def test_glyph_strength_defaults_to_full_coverage(self):
        """Measured: at 0.95 the original garbled lettering stays clearly readable
        under the new text and therefore enters the init latent."""
        spec = SignDetailer.INPUT_TYPES()["required"]["glyph_strength"]
        assert spec[1]["default"] == 1.0

    def test_proposer_temperature_default_stays_below_the_cliff(self):
        """Measured cliff, not a slope: 0/6 transcriptions at 0.2, 3/6 at 0.25.

        A widget default above it silently reintroduces the bug where the model
        returns a spell-corrected version of the gibberish instead of new text.
        """
        from nodes.utils.lmstudio_client import DEFAULT_TEMPERATURE
        assert DEFAULT_TEMPERATURE <= 0.2
        spec = SignTextProposer.INPUT_TYPES()["required"]["temperature"]
        assert spec[1]["default"] == DEFAULT_TEMPERATURE

    def test_proposer_warns_when_temperature_is_raised_past_the_cliff(self):
        node = SignTextProposer()
        data = {"regions": [], "image_shape": (64, 64), "batch_size": 1}
        image = torch.rand(1, 64, 64, 3)
        _, _, report = node.execute(sign_data=data, image=image, enabled=True,
                                    base_url="http://127.0.0.1:9", temperature=0.6)
        assert "above the measured cliff" in report

    def test_no_class_instruction_asks_the_model_to_transcribe(self):
        """The per-class instruction is appended to the anti-transcription system
        prompt. If it says 'transcribe', the two contradict each other and the
        model falls back to reading the garbled original — which is how a screen
        came back as 'SYSTEM READY' and a shirt print as its own gibberish.
        """
        from core.signs.classes import SIGN_CLASSES, FALLBACK_CLASS
        banned = ("transcribe", "read the current", "as it appears",
                  "exactly as", "as written")
        offenders = []
        for name, cfg in list(SIGN_CLASSES.items()) + [("<fallback>", FALLBACK_CLASS)]:
            low = cfg["vlm_instruction"].lower()
            # "do not read the current letters" is the opposite instruction
            for term in banned:
                if term in low and f"do not {term}" not in low and \
                        f"never {term}" not in low:
                    offenders.append(f"{name}: ...{term}...")
        assert not offenders, "class instructions must not ask for transcription: " + \
            "; ".join(offenders)

    def test_every_class_instruction_asks_for_invention(self):
        from core.signs.classes import SIGN_CLASSES, all_class_names
        for name in all_class_names():
            low = SIGN_CLASSES[name]["vlm_instruction"].lower()
            assert any(w in low for w in ("invent", "make up", "work out")), \
                f"{name} does not tell the model to produce new text"

    def test_screen_class_bans_the_generic_filler(self):
        """'SYSTEM READY' is the observed failure mode for unreadable screens."""
        from core.signs.classes import SIGN_CLASSES
        low = SIGN_CLASSES["screen"]["vlm_instruction"].lower()
        assert "system ready" in low and "never" in low

    def test_proposer_does_not_mutate_the_incoming_sign_data(self):
        """ComfyUI hands the same cached object to every downstream node.

        Two Proposers on one Selector must not overwrite each other's proposals,
        which is what in-place mutation would cause.
        """
        node = SignTextProposer()
        region = {
            "index": 0, "class": "sign", "batch_index": 0, "bbox": [0, 0, 40, 40],
            "mask": _rect_mask(), "crop": np.zeros((64, 64, 3), dtype=np.uint8),
            "cluster_id": -1, "too_small": False,
            "slop": {"verdict": "slop", "score": 0.9, "ocr_text": ""},
            "proposal": None,
        }
        upstream = {"regions": [region], "image_shape": (256, 256), "batch_size": 1}
        image = torch.rand(1, 256, 256, 3)

        out_a, _, _ = node.execute(sign_data=upstream, image=image, enabled=False,
                                   fallback_texts="sign: FIRST")
        assert region["proposal"] is None, "the upstream region must be untouched"
        assert upstream["regions"][0]["proposal"] is None

        out_b, _, _ = node.execute(sign_data=upstream, image=image, enabled=False,
                                   fallback_texts="sign: SECOND")
        assert out_a["regions"][0]["proposal"]["text"] == "FIRST", \
            "the second run must not reach back into the first run's result"
        assert out_b["regions"][0]["proposal"]["text"] == "SECOND"

    def test_proposer_shares_heavy_values_by_reference(self):
        """The copy must stay shallow — masks and crops are large."""
        node = SignTextProposer()
        mask = _rect_mask()
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        region = {
            "index": 0, "class": "sign", "batch_index": 0, "bbox": [0, 0, 40, 40],
            "mask": mask, "crop": crop, "cluster_id": -1, "too_small": False,
            "slop": {"verdict": "slop", "score": 0.9, "ocr_text": ""}, "proposal": None,
        }
        out, _, _ = node.execute(
            sign_data={"regions": [region], "image_shape": (256, 256), "batch_size": 1},
            image=torch.rand(1, 256, 256, 3), enabled=False, fallback_texts="sign: X")
        assert out["regions"][0]["mask"] is mask
        assert out["regions"][0]["crop"] is crop

    def test_proposer_stays_quiet_at_the_safe_temperature(self):
        from nodes.utils.lmstudio_client import DEFAULT_TEMPERATURE
        node = SignTextProposer()
        data = {"regions": [], "image_shape": (64, 64), "batch_size": 1}
        image = torch.rand(1, 64, 64, 3)
        _, _, report = node.execute(sign_data=data, image=image, enabled=True,
                                    base_url="http://127.0.0.1:9",
                                    temperature=DEFAULT_TEMPERATURE)
        assert "above the measured cliff" not in report


class TestMeasuredCeilings:
    """Values pinned by live renders on Krea 2 Turbo, not by taste."""

    def test_glyph_denoise_default_is_under_the_safe_ceiling(self):
        from nodes.signs.detailer import GLYPH_DENOISE_SAFE_MAX
        default = SignDetailer.INPUT_TYPES()["required"]["glyph_denoise"][1]["default"]
        assert default <= GLYPH_DENOISE_SAFE_MAX
        assert GLYPH_DENOISE_SAFE_MAX < 0.65, \
            "0.65 already produced a ghost copy of the word in a live render"

    def test_turbo_alone_does_not_flag_a_model_as_text_weak(self):
        """Krea 2 Turbo renders text well; matching on 'turbo' warned against it."""
        from nodes.signs.detailer import _TEXT_WEAK_HINTS
        assert "turbo" not in _TEXT_WEAK_HINTS
        assert any("z-image" in h or "zimage" in h for h in _TEXT_WEAK_HINTS), \
            "the genuinely text-weak families must still be caught"

    def test_krea2_turbo_config_is_not_warned_about(self):
        node = SignDetailer()

        class FakeInner:
            pass

        class FakeModel:
            def __init__(self):
                self.model = FakeInner()
                self.model.model_config = "krea2_turbo_fp8"

        assert node._warn_if_text_weak(FakeModel()) is None

    def test_z_image_is_still_warned_about(self):
        node = SignDetailer()

        class FakeInner:
            pass

        class FakeModel:
            def __init__(self):
                self.model = FakeInner()
                self.model.model_config = "z-image turbo lumina2"

        warning = node._warn_if_text_weak(FakeModel())
        assert warning and "WARNING" in warning
