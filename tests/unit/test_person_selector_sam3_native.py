"""Tests for PersonSelectorSAM3Native — the built-in-SAM3 variant.

Covers the slim front end (5 refs, no outfit/SAM2/YOLO), the threshold->auto
mapping, the match_weights migration from 4 to 3 terms, and that the native
backend is dispatched instead of the extension path.
"""

import sys
from unittest.mock import MagicMock

import pytest

# masker does `from ...parsing import BiSeNet` and face_analyzer imports
# insightface — neither resolves outside ComfyUI's loader. Try the REAL modules
# first and only stub what actually fails: installing stubs unconditionally
# poisons sys.modules for every later test that needs the real masker
# (test_person_data_refiner_sam3_aux does).
try:
    import nodes.person_selector_sam3_native as native
except (ImportError, ModuleNotFoundError):
    for _mod in ("nodes.utils.masker", "nodes.utils.face_analyzer",
                 "nodes.utils.yolo_detector"):
        sys.modules.setdefault(_mod, MagicMock())
    try:
        import nodes.person_selector_sam3_native as native
    except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
        pytest.skip(f"PersonSelectorSAM3Native unavailable: {e}", allow_module_level=True)


NODE = native.PersonSelectorSAM3Native


class TestSchema:
    def test_five_reference_slots(self):
        opt = NODE.INPUT_TYPES()["optional"]
        assert NODE.MAX_REFERENCES == 5
        for i in range(1, 6):
            assert f"reference_{i}" in opt
        assert "reference_6" not in opt

    def test_native_model_and_clip_inputs(self):
        req = NODE.INPUT_TYPES()["required"]
        assert req["sam3_model"][0] == "MODEL", "must take the native MODEL, not SAM3_MODEL_CONFIG"
        assert req["sam3_clip"][0] == "CLIP"

    def test_dropped_inputs_are_gone(self):
        it = NODE.INPUT_TYPES()
        everything = set(it["required"]) | set(it["optional"])
        for gone in ("outfit_palettes", "sam_model", "auto_threshold", "guaranteed_refs",
                     "aggregation", "aux_yolo_model", "aux_yolo_confidence",
                     "aux_yolo_label_filter", "aux_yolo_sam_refine",
                     "aux_yolo_sam_bbox_expansion"):
            assert gone not in everything, f"{gone} should not be an input any more"

    def test_input_count_is_slim(self):
        """19 inputs vs. the parent's 30 — the count is pinned so it can't creep back.

        required (12): sam3_model, sam3_clip, current_image, threshold, det_size,
                       match_weights, aux_preset, aux_custom_prompt, aux_threshold,
                       aux_prompt, aux_prompt_threshold, refine_iterations
        optional  (7): reference_1..5, depth_map, depth_sort_order
        """
        from nodes.person_selector_sam3 import PersonSelectorSAM3
        it = NODE.INPUT_TYPES()
        total = len(it["required"]) + len(it["optional"])
        parent = PersonSelectorSAM3.INPUT_TYPES()
        parent_total = len(parent["required"]) + len(parent["optional"])
        assert total == 19, f"input count changed: {total}"
        assert total < parent_total, f"native ({total}) must stay slimmer than parent ({parent_total})"

    def test_kept_inputs(self):
        it = NODE.INPUT_TYPES()
        everything = set(it["required"]) | set(it["optional"])
        for kept in ("current_image", "threshold", "det_size", "match_weights",
                     "aux_preset", "aux_custom_prompt", "aux_threshold",
                     "refine_iterations", "depth_map", "depth_sort_order"):
            assert kept in everything

    def test_contract_extends_parent(self):
        """The parent's outputs keep their slots; aux_data is appended.

        That ordering is what makes the node a drop-in swap for the old one in an
        existing workflow — every existing wire still lands on the same output.
        """
        from nodes.person_selector_sam3 import PersonSelectorSAM3
        n = len(PersonSelectorSAM3.RETURN_TYPES)
        assert NODE.RETURN_TYPES[:n] == PersonSelectorSAM3.RETURN_TYPES
        assert NODE.RETURN_NAMES[:n] == PersonSelectorSAM3.RETURN_NAMES
        assert NODE.RETURN_NAMES[n:] == ("aux_data",)
        assert len(NODE.RETURN_TYPES) == len(NODE.RETURN_NAMES)
        assert NODE.FUNCTION == "execute"

    def test_default_match_weights_has_three_terms(self):
        default = NODE.INPUT_TYPES()["required"]["match_weights"][1]["default"]
        assert len(default.split("/")) == 3, "no outfit term without outfit_palettes"


class TestExecuteWiring:
    """execute() only translates the front end, then defers to the parent."""

    @pytest.fixture
    def captured(self, monkeypatch):
        """Stand in for the inherited pipeline and record what it was handed.

        The fake returns a full-width result tuple because execute() now reaches
        into it (person_data at 0, preview at 5) to build the aux channel.
        """
        import torch
        calls = {}
        person_data = {"batch_size": 1, "num_references": 0,
                       "image_height": 8, "image_width": 8}
        preview = torch.zeros(1, 8, 8, 3)

        def fake_super_execute(self, **kwargs):
            calls.update(kwargs)
            n = len(native.PersonSelectorSAM3.RETURN_TYPES)
            out = [None] * n
            out[0] = person_data
            out[5] = preview
            return tuple(out)

        monkeypatch.setattr(native.PersonSelectorSAM3, "execute", fake_super_execute)
        return calls

    def _run(self, **over):
        node = NODE()
        import torch
        params = dict(sam3_model="MODEL_OBJ", sam3_clip="CLIP_OBJ",
                      current_image=torch.zeros(1, 8, 8, 3),
                      threshold=0.0, det_size="640")
        params.update(over)
        return node.execute(**params)

    def test_backend_is_native_bundle(self, captured, monkeypatch):
        made = {}

        class FakeNative:
            def __init__(self, model, clip, refine_iterations=2):
                made.update(model=model, clip=clip, refine=refine_iterations)

        monkeypatch.setattr(native, "NativeSAM3", FakeNative)
        self._run(refine_iterations=4)
        assert made == {"model": "MODEL_OBJ", "clip": "CLIP_OBJ", "refine": 4}
        assert isinstance(captured["sam3_model"], FakeNative), \
            "the parent must receive the native bundle in place of the config dict"

    def test_threshold_zero_means_auto(self, captured):
        self._run(threshold=0.0)
        assert captured["auto_threshold"] is True

    def test_threshold_above_zero_is_manual(self, captured):
        self._run(threshold=0.42)
        assert captured["auto_threshold"] is False
        assert captured["threshold"] == 0.42

    def test_three_term_weights_get_zero_outfit(self, captured):
        self._run(match_weights="70/15/15")
        assert captured["match_weights"] == "70/15/15/0"

    def test_four_term_weights_pass_through(self, captured):
        self._run(match_weights="50/15/15/20")
        assert captured["match_weights"] == "50/15/15/20"

    def test_dropped_features_are_disabled_downstream(self, captured):
        self._run()
        assert captured["outfit_palettes"] is None
        assert captured["sam_model"] is None
        assert captured["aux_yolo_model"] == "None"
        assert captured["guaranteed_refs"] == 0
        assert captured["aggregation"] == "max"

    def test_references_reach_the_parent(self, captured):
        self._run(reference_1="R1", reference_3="R3")
        assert captured["reference_1"] == "R1"
        assert captured["reference_3"] == "R3"

    def test_depth_passed_through(self, captured):
        self._run(depth_map="DEPTH", depth_sort_order="front_first")
        assert captured["depth_map"] == "DEPTH"
        assert captured["depth_sort_order"] == "front_first"
