"""Batch Save Multi — variants into folders, gated sorting, original handling."""

import os

import pytest
import torch
from PIL import Image

from nodes.batch.save_multi import MAX_SLOTS, FVM_BatchSaveMulti, gate_state


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "pic.jpg"
    Image.new("RGB", (8, 6), (200, 10, 10)).save(path, quality=90)
    return str(path)


def run(source_path="", **overrides):
    kwargs = dict(slots=2, route="all_matches", base_dir="", fallback_dir="unsorted",
                  original="keep", original_dir="done", format="keep", quality=95,
                  mask_min_area=0.001, overwrite=False, source_path=source_path)
    kwargs.update(overrides)
    result = FVM_BatchSaveMulti().execute(**kwargs)["result"]
    return dict(zip(FVM_BatchSaveMulti.RETURN_NAMES, result))


def mask(fraction, size=100):
    m = torch.zeros(1, size, size)
    m.view(-1)[: int(fraction * size * size)] = 1.0
    return m


class TestContract:
    def test_slot_inputs_declared(self):
        optional = FVM_BatchSaveMulti.INPUT_TYPES()["optional"]
        for i in range(1, MAX_SLOTS + 1):
            assert {f"image_{i}", f"gate_{i}", f"subdir_{i}"} <= set(optional)
        assert MAX_SLOTS == 8

    def test_return_shape(self):
        assert len(FVM_BatchSaveMulti.RETURN_TYPES) == len(FVM_BatchSaveMulti.RETURN_NAMES)


class TestGateState:
    @pytest.mark.parametrize("value,expected", [
        (None, None), (True, True), (False, False), (0, False), (3, True),
        (0.0, False), ("", False), ("false", False), ("yes", True),
    ])
    def test_scalars(self, value, expected):
        assert gate_state(value) is expected

    def test_mask_area_threshold(self):
        assert gate_state(mask(0.0)) is False
        assert gate_state(mask(0.05), mask_min_area=0.01) is True
        assert gate_state(mask(0.005), mask_min_area=0.01) is False


class TestVariants:
    def test_image_and_original_copy_to_own_folders(self, source, tmp_path):
        image = torch.rand(1, 6, 8, 3)
        out = run(source, subdir_1="refined", image_1=image, subdir_2="orig")
        paths = out["saved_paths"].splitlines()
        assert paths == [str(tmp_path / "refined" / "pic.jpg"),
                         str(tmp_path / "orig" / "pic.jpg")]
        # Slot 2 had no image: the original bytes, not a re-encode.
        assert open(paths[1], "rb").read() == open(source, "rb").read()
        assert os.path.isfile(source)                     # original = keep

    def test_slots_beyond_count_ignored(self, source, tmp_path):
        run(source, slots=1, subdir_1="a", subdir_2="b")
        assert (tmp_path / "a").is_dir() and not (tmp_path / "b").exists()

    def test_absolute_subdir(self, source, tmp_path):
        target = tmp_path / "elsewhere"
        out = run(source, slots=1, subdir_1=str(target))
        assert out["saved_paths"] == str(target / "pic.jpg")

    def test_collision_gets_counter(self, source, tmp_path):
        run(source, slots=1, subdir_1="a")
        out = run(source, slots=1, subdir_1="a")
        assert out["saved_paths"].endswith("pic_001.jpg")


class TestSorting:
    def test_all_matches(self, source, tmp_path):
        out = run(source, slots=3, subdir_1="glasses", gate_1=mask(0.1),
                  subdir_2="hat", gate_2=mask(0.0), subdir_3="dog", gate_3=2)
        assert out["matched"] == "glasses, dog"
        assert out["matched_count"] == 2
        assert (tmp_path / "glasses" / "pic.jpg").is_file()
        assert not (tmp_path / "hat").exists()
        assert not (tmp_path / "unsorted").exists()

    def test_first_match(self, source, tmp_path):
        out = run(source, slots=2, route="first_match", subdir_1="a", gate_1=True,
                  subdir_2="b", gate_2=True)
        assert out["matched"] == "a"
        assert not (tmp_path / "b").exists()

    def test_ungated_slot_always_writes_without_counting(self, source, tmp_path):
        out = run(source, slots=2, subdir_1="all", subdir_2="hat", gate_2=False)
        assert (tmp_path / "all" / "pic.jpg").is_file()
        assert out["matched_count"] == 0

    def test_fallback_when_nothing_matched(self, source, tmp_path):
        run(source, slots=2, subdir_1="a", gate_1=False, subdir_2="b", gate_2=0)
        assert (tmp_path / "unsorted" / "pic.jpg").is_file()

    def test_no_fallback_without_gates(self, source, tmp_path):
        run(source, slots=1, subdir_1="a")
        assert not (tmp_path / "unsorted").exists()


class TestOriginal:
    def test_move(self, source, tmp_path):
        run(source, slots=1, subdir_1="a", original="move")
        assert not os.path.exists(source)
        assert (tmp_path / "done" / "pic.jpg").is_file()

    def test_delete(self, source, tmp_path):
        run(source, slots=1, subdir_1="a", original="delete")
        assert not os.path.exists(source)
        assert (tmp_path / "a" / "pic.jpg").is_file()

    def test_error_keeps_original(self, source):
        # Slot 1 has neither an image nor a usable folder → error → no delete.
        out = run(source, slots=1, subdir_1="", original="delete")
        assert os.path.isfile(source)
        assert "left in place" in out["report"]

    def test_missing_source_is_harmless(self, tmp_path):
        out = run("", slots=1, subdir_1=str(tmp_path / "x"), original="delete")
        assert "neither image nor source" in out["report"]


class TestPersonDataGate:
    """SAM3 aux detection wired straight into a gate."""

    def pd(self, assigned=0.0, unassigned=0.0):
        return {"aux_masks": [mask(assigned)], "aux_unassigned_masks": mask(unassigned)}

    def test_fires_on_unassigned_parts(self):
        # No reference images: every hit is "unassigned".
        assert gate_state(self.pd(unassigned=0.05)) is True

    def test_fires_on_assigned_parts(self):
        assert gate_state(self.pd(assigned=0.05)) is True

    def test_silent_without_parts(self):
        assert gate_state(self.pd()) is False

    def test_person_data_without_aux_is_silent(self):
        # Selector ran with aux off: nothing searched, nothing found. This used
        # to fall through to dict truthiness and pass every picture.
        assert gate_state({"batch_size": 1, "num_references": 0, "face_masks": []}) is False

    def test_empty_aux_lists_are_silent(self):
        assert gate_state({"num_references": 0, "aux_masks": []}) is False

    def test_other_dicts_keep_truthiness(self):
        assert gate_state({"x": 1}) is True


class TestOriginalWhen:
    def test_on_match_leaves_misses_in_place(self, source, tmp_path):
        out = run(source, slots=1, subdir_1="hit", gate_1=False, fallback_dir="",
                  original="move", original_when="on_match", original_dir="hit")
        assert os.path.isfile(source)
        assert "no gate fired" in out["report"]
        assert not (tmp_path / "hit").exists()

    def test_on_match_moves_hits(self, source, tmp_path):
        run(source, slots=1, subdir_1="hit/replaced", gate_1=True, image_1=torch.rand(1, 6, 8, 3),
            fallback_dir="", original="move", original_when="on_match", original_dir="hit")
        assert not os.path.exists(source)
        assert (tmp_path / "hit" / "pic.jpg").is_file()
        assert (tmp_path / "hit" / "replaced" / "pic.jpg").is_file()
