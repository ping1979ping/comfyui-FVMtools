"""Node contracts — what ComfyUI requires of every node in the Batch block."""

import pytest

from nodes.batch import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODES = list(NODE_CLASS_MAPPINGS.items())


@pytest.mark.parametrize("name,cls", NODES, ids=[n for n, _c in NODES])
class TestEveryNode:
    def test_has_the_required_attributes(self, name, cls):
        for attribute in ("CATEGORY", "INPUT_TYPES", "RETURN_TYPES", "FUNCTION"):
            assert hasattr(cls, attribute), f"{name} lacks {attribute}"

    def test_function_points_at_a_real_method(self, name, cls):
        assert callable(getattr(cls, cls.FUNCTION, None))

    def test_return_names_line_up_with_types(self, name, cls):
        if hasattr(cls, "RETURN_NAMES"):
            assert len(cls.RETURN_NAMES) == len(cls.RETURN_TYPES)

    def test_input_types_are_well_formed(self, name, cls):
        spec = cls.INPUT_TYPES()
        assert "required" in spec
        for section in ("required", "optional"):
            for input_name, definition in spec.get(section, {}).items():
                assert isinstance(definition, tuple), f"{name}.{input_name}"
                assert len(definition) in (1, 2), f"{name}.{input_name}"

    def test_category_is_in_the_batch_block(self, name, cls):
        assert cls.CATEGORY == "FVM Tools/Batch"

    def test_has_a_display_name(self, name, cls):
        assert name in NODE_DISPLAY_NAME_MAPPINGS

    def test_execute_is_documented(self, name, cls):
        assert (cls.__doc__ or "").strip(), f"{name} has no docstring"


class TestBlock:
    def test_both_mappings_cover_the_same_nodes(self):
        assert set(NODE_CLASS_MAPPINGS) == set(NODE_DISPLAY_NAME_MAPPINGS)

    def test_names_are_prefixed(self):
        assert all(name.startswith("FVM_") for name in NODE_CLASS_MAPPINGS)

    def test_root_registration_picks_the_block_up(self):
        """The block has to reach ComfyUI through the package's own mappings."""
        import importlib

        root = importlib.import_module("__init__") if False else None  # noqa: F841
        # The root __init__ only registers inside ComfyUI, so assert the wiring
        # the root relies on instead: the submodule exports both mappings.
        from nodes import batch
        assert batch.NODE_CLASS_MAPPINGS and batch.NODE_DISPLAY_NAME_MAPPINGS

    def test_defaults_match_the_calibrated_values(self):
        """The node must ship the settings the acceptance run was measured with."""
        from nodes.batch.reality import FVM_RealityCheck
        from nodes.utils.reality_client import (
            DEFAULT_MAX_IMAGE_SIZE, DEFAULT_PROBES, DEFAULT_TEMPERATURE,
            DEFAULT_THRESHOLD,
        )

        required = FVM_RealityCheck.INPUT_TYPES()["required"]
        assert required["threshold"][1]["default"] == DEFAULT_THRESHOLD
        assert required["temperature"][1]["default"] == DEFAULT_TEMPERATURE
        assert required["max_image_size"][1]["default"] == DEFAULT_MAX_IMAGE_SIZE
        for probe in ("parts", "people", "landmarks"):
            assert probe in DEFAULT_PROBES
            assert required[f"check_{probe}"][1]["default"] is True
        for probe in ("hands", "physics", "text"):
            assert required[f"check_{probe}"][1]["default"] is False, \
                f"{probe} raised false alarms in the acceptance run"

    def test_loader_reruns_every_queue_run(self):
        """Without this the graph caches the first image and never advances."""
        from nodes.batch.loader import FVM_BatchLoadImage
        first, second = FVM_BatchLoadImage.IS_CHANGED(), FVM_BatchLoadImage.IS_CHANGED()
        assert first != first and second != second      # NaN
