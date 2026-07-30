"""Unit tests for nodes/utils/ocr_backend.py.

These tests are written for a machine with NO OCR models installed — the
degraded behaviour is asserted explicitly instead of being skipped. They must
also stay green once models are present, so anything state dependent is either
monkeypatched or expressed as a shape/type assertion.
"""

import numpy as np
import pytest

from nodes.utils import ocr_backend
from nodes.utils.ocr_backend import (
    EASYOCR_MODEL_FILES,
    OCR_MODEL_FILES,
    backend_status,
    get_available_backends,
    ocr_region,
    resolve_ocr_dir,
    run_ocr,
)


EMPTY_REGION = {"text": "", "conf": 0.0, "char_confs": [], "line_count": 0}


@pytest.fixture
def rgb_image():
    """Random RGB uint8 image — nothing readable in it."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 255, (128, 256, 3), dtype=np.uint8)


@pytest.fixture
def no_backends(monkeypatch):
    """Force the 'nothing installed' state regardless of the real machine."""
    monkeypatch.setattr(ocr_backend, "get_available_backends", lambda: [])
    return None


# ──── Constants ────

class TestModelConstants:

    def test_ocr_model_files_keys(self):
        assert set(OCR_MODEL_FILES) == {"det", "rec", "cls", "keys"}
        assert OCR_MODEL_FILES["det"] == "ch_PP-OCRv4_det_infer.onnx"
        assert OCR_MODEL_FILES["rec"] == "ch_PP-OCRv4_rec_infer.onnx"
        assert OCR_MODEL_FILES["cls"] == "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        assert OCR_MODEL_FILES["keys"] == "ppocr_keys_v1.txt"

    def test_easyocr_model_files_keys(self):
        assert set(EASYOCR_MODEL_FILES) == {"detector", "recognizer"}
        assert EASYOCR_MODEL_FILES["detector"] == "craft_mlt_25k.pth"
        assert EASYOCR_MODEL_FILES["recognizer"] == "english_g2.pth"

    def test_all_values_are_plain_filenames(self):
        for name in list(OCR_MODEL_FILES.values()) + list(EASYOCR_MODEL_FILES.values()):
            assert isinstance(name, str) and name
            assert "/" not in name and "\\" not in name


# ──── Path resolution ────

class TestResolveOcrDir:

    def test_returns_str_or_none_never_mock(self):
        """folder_paths is MagicMock-ed by conftest — a mock must not leak through."""
        result = resolve_ocr_dir()
        assert result is None or isinstance(result, str)
        assert type(result).__name__ not in ("MagicMock", "Mock")

    def test_result_is_an_existing_dir_if_not_none(self):
        result = resolve_ocr_dir()
        if result is not None:
            import os
            assert os.path.isdir(result)

    def test_does_not_raise_when_folder_paths_explodes(self, monkeypatch):
        class Boom:
            models_dir = object()  # not a str

            def get_folder_paths(self, _):
                raise RuntimeError("nope")

        monkeypatch.setattr(ocr_backend, "_import_folder_paths", lambda: Boom())
        assert resolve_ocr_dir() is None or isinstance(resolve_ocr_dir(), str)

    def test_tier1_picks_first_existing_onnx_root(self, monkeypatch, tmp_path):
        missing = tmp_path / "rootA"
        present = tmp_path / "rootB"
        (present / "ocr").mkdir(parents=True)

        class FP:
            models_dir = str(tmp_path / "models")

            def get_folder_paths(self, key):
                assert key == "onnx"
                return [str(missing), str(present)]

        monkeypatch.setattr(ocr_backend, "_import_folder_paths", lambda: FP())
        assert resolve_ocr_dir() == str(present / "ocr")

    def test_tier2_models_dir_fallback(self, monkeypatch, tmp_path):
        models = tmp_path / "models"
        (models / "onnx" / "ocr").mkdir(parents=True)

        class FP:
            models_dir = str(models)

            def get_folder_paths(self, key):
                return []

        monkeypatch.setattr(ocr_backend, "_import_folder_paths", lambda: FP())
        assert resolve_ocr_dir() == str(models / "onnx" / "ocr")

    def test_tier3_ini_fallback(self, monkeypatch, tmp_path):
        ini_dir = tmp_path / "from_ini"
        ini_dir.mkdir()
        monkeypatch.setattr(ocr_backend, "_import_folder_paths", lambda: None)
        monkeypatch.setattr(ocr_backend, "get_model_path", lambda key: str(ini_dir))
        assert resolve_ocr_dir() == str(ini_dir)

    def test_valid_dir_rejects_non_strings(self):
        from unittest.mock import MagicMock
        assert ocr_backend._valid_dir(MagicMock()) is None
        assert ocr_backend._valid_dir(None) is None
        assert ocr_backend._valid_dir("") is None
        assert ocr_backend._valid_dir("   ") is None
        assert ocr_backend._valid_dir(r"Z:\definitely\not\here") is None


# ──── Status / availability ────

class TestBackendStatus:

    def test_has_both_keys(self):
        status = backend_status()
        assert set(status) == {"onnx", "easyocr"}

    @pytest.mark.parametrize("name", ["onnx", "easyocr"])
    def test_documented_shape(self, name):
        entry = backend_status()[name]
        assert set(entry) == {"available", "reason", "dir", "missing"}
        assert isinstance(entry["available"], bool)
        assert isinstance(entry["reason"], str)
        assert entry["dir"] is None or isinstance(entry["dir"], str)
        assert isinstance(entry["missing"], list)
        assert all(isinstance(m, str) for m in entry["missing"])

    @pytest.mark.parametrize("name", ["onnx", "easyocr"])
    def test_reason_is_human_readable(self, name):
        reason = backend_status()[name]["reason"]
        assert len(reason) > 10, "reason must explain the state to a human"
        assert " " in reason

    @pytest.mark.parametrize("name", ["onnx", "easyocr"])
    def test_unavailable_backend_reports_missing_or_explains(self, name):
        entry = backend_status()[name]
        if not entry["available"]:
            assert entry["missing"] or "not installed" in entry["reason"] \
                or "not found" in entry["reason"]

    def test_get_available_backends_is_list(self):
        backends = get_available_backends()
        assert isinstance(backends, list)
        assert all(b in ("onnx", "easyocr") for b in backends)

    def test_get_available_backends_matches_status(self):
        status = backend_status()
        assert get_available_backends() == [n for n in ("onnx", "easyocr")
                                            if status[n]["available"]]

    def test_status_survives_probe_failure(self, monkeypatch):
        monkeypatch.setattr(ocr_backend, "_onnx_status",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        entry = backend_status()["onnx"]
        assert entry["available"] is False
        assert "boom" in entry["reason"]


# ──── Degraded behaviour with no backend ────

class TestNoBackendDegradation:

    def test_run_ocr_returns_empty_list(self, rgb_image, no_backends):
        assert run_ocr(rgb_image) == []

    def test_run_ocr_real_machine_state(self, rgb_image):
        """No monkeypatching — on a machine without models this must be []."""
        result = run_ocr(rgb_image)
        assert isinstance(result, list)
        if not get_available_backends():
            assert result == []

    def test_run_ocr_unknown_backend(self, rgb_image):
        assert run_ocr(rgb_image, backend="does_not_exist") == []

    def test_run_ocr_bad_input_never_raises(self, no_backends):
        assert run_ocr(None) == []
        assert run_ocr(np.zeros((10, 10), dtype=np.uint8)) == []
        assert run_ocr(np.zeros((0, 0, 3), dtype=np.uint8)) == []
        assert run_ocr("not an image") == []

    def test_run_ocr_float_image_accepted(self, no_backends):
        assert run_ocr(np.random.rand(32, 32, 3).astype(np.float32)) == []

    def test_run_ocr_backend_exception_is_swallowed(self, rgb_image, monkeypatch):
        def boom(_img):
            raise RuntimeError("backend exploded")

        monkeypatch.setattr(ocr_backend, "_RUNNERS", {"boom": boom})
        monkeypatch.setattr(ocr_backend, "get_available_backends", lambda: ["boom"])
        assert run_ocr(rgb_image) == []

    def test_ocr_region_empty_shape_with_bbox(self, rgb_image, no_backends):
        assert ocr_region(rgb_image, [10, 10, 100, 60]) == EMPTY_REGION

    def test_ocr_region_empty_shape_with_mask(self, rgb_image, no_backends):
        mask = np.zeros(rgb_image.shape[:2], dtype=np.float32)
        mask[20:60, 30:120] = 1.0
        assert ocr_region(rgb_image, mask) == EMPTY_REGION

    def test_ocr_region_empty_mask(self, rgb_image, no_backends):
        mask = np.zeros(rgb_image.shape[:2], dtype=np.float32)
        assert ocr_region(rgb_image, mask) == EMPTY_REGION

    def test_ocr_region_garbage_region(self, rgb_image, no_backends):
        assert ocr_region(rgb_image, None) == EMPTY_REGION
        assert ocr_region(rgb_image, [5, 5, 5, 5]) == EMPTY_REGION
        assert ocr_region(None, [0, 0, 10, 10]) == EMPTY_REGION

    def test_easyocr_and_onnx_runners_degrade(self, rgb_image):
        """Direct runner calls must also return [] instead of raising."""
        assert ocr_backend._run_onnx(rgb_image) == [] or get_available_backends()
        assert ocr_backend._run_easyocr(rgb_image) == [] or get_available_backends()


# ──── Aggregation logic with a fake backend ────

FAKE_LINES = [
    {"text": "HELLO", "conf": 0.9, "char_confs": [0.9] * 5,
     "bbox": [10, 5, 60, 25], "quad": [[10, 5], [60, 5], [60, 25], [10, 25]]},
    {"text": "WORLD", "conf": 0.7, "char_confs": [0.7] * 5,
     "bbox": [12, 30, 80, 50], "quad": [[12, 30], [80, 30], [80, 50], [12, 50]]},
]


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(ocr_backend, "_RUNNERS", {"fake": lambda img: [dict(l) for l in FAKE_LINES]})
    monkeypatch.setattr(ocr_backend, "get_available_backends", lambda: ["fake"])
    return FAKE_LINES


class TestRunOcrNormalization:

    def test_returns_documented_keys(self, rgb_image, fake_backend):
        results = run_ocr(rgb_image)
        assert len(results) == 2
        for item in results:
            assert set(item) == {"text", "conf", "char_confs", "bbox", "quad"}
            assert isinstance(item["text"], str)
            assert isinstance(item["conf"], float)
            assert len(item["bbox"]) == 4
            assert len(item["quad"]) == 4
            assert all(len(p) == 2 for p in item["quad"])

    def test_min_conf_filters(self, rgb_image, fake_backend):
        assert [r["text"] for r in run_ocr(rgb_image, min_conf=0.8)] == ["HELLO"]
        assert run_ocr(rgb_image, min_conf=0.95) == []
        assert len(run_ocr(rgb_image, min_conf=0.0)) == 2

    def test_explicit_backend_name_is_used(self, rgb_image, monkeypatch):
        monkeypatch.setattr(ocr_backend, "_RUNNERS",
                            {"fake": lambda img: [dict(l) for l in FAKE_LINES]})
        monkeypatch.setattr(ocr_backend, "get_available_backends", lambda: [])
        assert len(run_ocr(rgb_image, backend="fake")) == 2

    def test_empty_text_entries_dropped(self, rgb_image, monkeypatch):
        raw = [{"text": "", "conf": 0.99, "char_confs": [], "bbox": [0, 0, 1, 1], "quad": []},
               {"text": "OK", "conf": 0.5, "char_confs": [], "bbox": [0, 0, 2, 2], "quad": []}]
        monkeypatch.setattr(ocr_backend, "_RUNNERS", {"fake": lambda img: raw})
        monkeypatch.setattr(ocr_backend, "get_available_backends", lambda: ["fake"])
        assert [r["text"] for r in run_ocr(rgb_image)] == ["OK"]

    def test_malformed_entries_are_tolerated(self, rgb_image, monkeypatch):
        raw = ["garbage", None,
               {"text": "GOOD", "conf": 0.4, "char_confs": ["x"], "bbox": "bad", "quad": "bad"}]
        monkeypatch.setattr(ocr_backend, "_RUNNERS", {"fake": lambda img: raw})
        monkeypatch.setattr(ocr_backend, "get_available_backends", lambda: ["fake"])
        results = run_ocr(rgb_image)
        assert len(results) == 1
        assert results[0]["text"] == "GOOD"
        assert results[0]["bbox"] == [0.0, 0.0, 0.0, 0.0]
        assert results[0]["char_confs"] == []


class TestOcrRegionAggregation:

    def test_text_joining_and_conf_average(self, rgb_image, fake_backend):
        out = ocr_region(rgb_image, [0, 0, 128, 100], pad=0)
        assert out["text"] == "HELLO WORLD"
        assert out["line_count"] == 2
        assert out["conf"] == pytest.approx((0.9 + 0.7) / 2)

    def test_char_confs_concatenated(self, rgb_image, fake_backend):
        out = ocr_region(rgb_image, [0, 0, 128, 100], pad=0)
        assert len(out["char_confs"]) == 10
        assert out["char_confs"][:5] == [0.9] * 5
        assert out["char_confs"][5:] == [0.7] * 5

    def test_bbox_union_in_full_image_coords(self, rgb_image, fake_backend):
        out = ocr_region(rgb_image, [0, 0, 128, 100], pad=0)
        assert out["bbox"] == [10.0, 5.0, 80.0, 50.0]

    def test_bbox_union_offset_by_crop_origin(self, rgb_image, fake_backend):
        out = ocr_region(rgb_image, [40, 20, 128, 100], pad=0)
        assert out["bbox"] == [50.0, 25.0, 120.0, 70.0]

    def test_documented_keys_present(self, rgb_image, fake_backend):
        out = ocr_region(rgb_image, [0, 0, 128, 100], pad=0)
        for key in EMPTY_REGION:
            assert key in out
        assert out["lines"] == run_ocr(rgb_image)

    def test_mask_region_equivalent_to_bbox(self, rgb_image, fake_backend):
        mask = np.zeros(rgb_image.shape[:2], dtype=np.float32)
        mask[20:70, 40:120] = 1.0
        from_mask = ocr_region(rgb_image, mask, pad=0)
        # mask nonzero bbox is x 40..119, y 20..69
        from_bbox = ocr_region(rgb_image, [40, 20, 119, 69], pad=0)
        assert from_mask["text"] == from_bbox["text"]
        assert from_mask["bbox"] == from_bbox["bbox"]

    def test_min_conf_propagates(self, rgb_image, fake_backend):
        out = ocr_region(rgb_image, [0, 0, 128, 100], pad=0, min_conf=0.8)
        assert out["text"] == "HELLO"
        assert out["line_count"] == 1
        assert out["conf"] == pytest.approx(0.9)

    def test_all_lines_filtered_gives_empty_shape(self, rgb_image, fake_backend):
        assert ocr_region(rgb_image, [0, 0, 128, 100], pad=0, min_conf=0.99) == EMPTY_REGION


# ──── ONNX pipeline internals (synthetic data, no model files needed) ────

class TestOnnxPipelineInternals:
    """The PP-OCRv4 math is exercised with hand-built tensors so it is covered
    even though no .onnx file exists on this machine."""

    def test_det_preprocess_shape_and_normalisation(self):
        x, ratio_w, ratio_h = ocr_backend._det_preprocess(np.zeros((1300, 700, 3), np.uint8))
        assert x.shape[0] == 1 and x.shape[1] == 3
        assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0
        assert max(x.shape[2:]) <= 960
        assert ratio_w > 1 and ratio_h > 1  # network is smaller than the source

    def test_det_preprocess_small_image_not_upscaled(self):
        x, _, _ = ocr_backend._det_preprocess(np.zeros((100, 60, 3), np.uint8))
        assert max(x.shape[2:]) <= 128

    def test_db_postprocess_finds_boxes_in_reading_order(self):
        pytest.importorskip("pyclipper")
        pytest.importorskip("shapely")
        prob = np.zeros((160, 320), dtype=np.float32)
        prob[40:70, 60:200] = 0.95    # upper line
        prob[100:120, 20:90] = 0.90   # lower line
        quads = ocr_backend._db_postprocess(prob, ratio_w=2.0, ratio_h=2.0)
        assert len(quads) == 2
        assert quads[0][:, 1].min() < quads[1][:, 1].min(), "top line must come first"
        # unclip distance for a 140x30 box = area * 1.5 / perimeter = 18.53 px
        d = 140 * 30 * 1.5 / (2 * (140 + 30))
        assert float(quads[0][:, 0].min()) == pytest.approx((60 - d) * 2, abs=4)
        assert float(quads[0][:, 0].max()) == pytest.approx((200 + d) * 2, abs=4)

    def test_db_postprocess_ignores_low_probability(self):
        pytest.importorskip("pyclipper")
        prob = np.full((100, 100), 0.1, dtype=np.float32)
        assert ocr_backend._db_postprocess(prob, 1.0, 1.0) == []

    def test_crop_quad_rectifies(self):
        img = np.zeros((160, 320, 3), np.uint8)
        img[40:70, 60:200] = 255
        quad = np.array([[60, 40], [200, 40], [200, 70], [60, 70]], np.float32)
        crop = ocr_backend._crop_quad(img, quad)
        assert crop.shape[:2] == (30, 140)
        assert crop.mean() > 250

    def test_crop_quad_rotates_tall_boxes(self):
        img = np.zeros((200, 200, 3), np.uint8)
        quad = np.array([[60, 40], [90, 40], [90, 150], [60, 150]], np.float32)
        crop = ocr_backend._crop_quad(img, quad)
        assert crop.shape[1] > crop.shape[0], "vertical text must be rotated to landscape"

    @staticmethod
    def _fake_rec_session(matrix):
        import types

        class FakeSession:
            def get_inputs(self):
                return [types.SimpleNamespace(name="x")]

            def run(self, _outputs, feed):
                x = feed["x"]
                assert x.shape == (1, 3, 48, 320), x.shape
                assert x.min() >= -1.001 and x.max() <= 1.001
                return [matrix[None]]

        return FakeSession()

    def test_ctc_greedy_decode(self):
        keys = ["blank"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "]
        logits = np.zeros((12, len(keys)), np.float32)
        for t, k in enumerate([0, 1, 1, 0, 2, 0, 0, 3, 3, 3, 0, 0]):
            logits[t, k] = 8.0  # blank A A blank B blank blank C C C blank blank
        text, conf, char_confs = ocr_backend._rec_one(
            self._fake_rec_session(logits), keys, np.zeros((32, 100, 3), np.uint8))
        assert text == "ABC", "repeats collapsed and blanks dropped"
        assert len(char_confs) == 3
        assert 0.9 < conf <= 1.0

    def test_ctc_does_not_double_softmax(self):
        keys = ["blank"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "]
        logits = np.zeros((12, len(keys)), np.float32)
        for t, k in enumerate([0, 1, 1, 0, 2, 0, 0, 3, 3, 3, 0, 0]):
            logits[t, k] = 8.0
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        crop = np.zeros((32, 100, 3), np.uint8)
        raw_text, raw_conf, _ = ocr_backend._rec_one(self._fake_rec_session(logits), keys, crop)
        sm_text, sm_conf, _ = ocr_backend._rec_one(self._fake_rec_session(probs), keys, crop)
        assert raw_text == sm_text == "ABC"
        assert raw_conf == pytest.approx(sm_conf)

    def test_rec_rejects_degenerate_crop(self):
        keys = ["blank", "A", " "]
        out = ocr_backend._rec_one(self._fake_rec_session(np.zeros((2, 3), np.float32)),
                                   keys, np.zeros((1, 1, 3), np.uint8))
        assert out == ("", 0.0, [])
