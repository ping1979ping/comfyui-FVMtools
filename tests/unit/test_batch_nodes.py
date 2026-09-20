"""Loader, saver and router — the nodes that move files around."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from core.batch_state import get_done
from nodes.batch.loader import (
    BatchFinished, FVM_BatchLoadImage, make_bar, reset_tracker, tracker_status,
)
from nodes.batch.router import FVM_BatchRouter
from nodes.batch.saver import FVM_BatchSaveImage, unique_path


def write_image(path, size=(8, 6), color=(120, 60, 30)):
    Image.new("RGB", size, color).save(path)


@pytest.fixture
def folder(tmp_path):
    for name in ("a.png", "b.png", "c.png"):
        write_image(str(tmp_path / name))
    (tmp_path / "readme.txt").write_text("ignored")
    return str(tmp_path)


def load(folder, **overrides):
    settings = dict(directory=folder, pass_subdir="keep", fail_subdir="reject",
                    on_finish="stop", sort_by="name", include_subdirs=False,
                    create_dirs=True, tracker="default")
    settings.update(overrides)
    return FVM_BatchLoadImage().execute(**settings)


class TestLoader:
    def test_returns_every_declared_output(self, folder):
        result = load(folder)["result"]
        assert len(result) == len(FVM_BatchLoadImage.RETURN_TYPES)

    def test_image_is_comfy_shaped(self, folder):
        image, mask = load(folder)["result"][0:2]
        assert image.ndim == 4 and image.shape[0] == 1 and image.shape[3] == 3
        assert image.dtype == torch.float32
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
        assert mask.ndim == 3 and mask.shape[1:] == image.shape[1:3]

    def test_advances_one_file_per_run(self, folder):
        names = [load(folder)["result"][5] for _ in range(3)]
        assert names == ["a.png", "b.png", "c.png"]

    def test_stops_after_the_last_file(self, folder):
        for _ in range(3):
            load(folder)
        with pytest.raises(BatchFinished):
            load(folder)

    def test_loop_starts_over(self, folder):
        for _ in range(3):
            load(folder)
        assert load(folder, on_finish="loop")["result"][5] == "a.png"

    def test_non_images_are_ignored(self, folder):
        names = [load(folder)["result"][5] for _ in range(3)]
        assert "readme.txt" not in names

    def test_target_dirs_are_absolute_and_created(self, folder):
        _image, _mask, _src, pass_dir, fail_dir = load(folder)["result"][0:5]
        assert os.path.isabs(pass_dir) and os.path.isdir(pass_dir)
        assert os.path.basename(pass_dir) == "keep"
        assert os.path.basename(fail_dir) == "reject"

    def test_empty_subdir_name_falls_back_to_the_source(self, folder):
        pass_dir = load(folder, pass_subdir="")["result"][3]
        assert os.path.normpath(pass_dir) == os.path.normpath(folder)

    def test_counts_and_progress_string(self, folder):
        result = load(folder)["result"]
        assert result[6] == 1 and result[7] == 3
        assert "1/3" in result[8] and "33%" in result[8]

    def test_progress_reaches_full_on_the_last_file(self, folder):
        for _ in range(2):
            load(folder)
        result = load(folder)["result"]
        assert (result[6], result[7]) == (3, 3)
        assert "100%" in result[8]

    def test_trackers_walk_independently(self, folder):
        assert load(folder, tracker="one")["result"][5] == "a.png"
        assert load(folder, tracker="two")["result"][5] == "a.png"
        assert load(folder, tracker="one")["result"][5] == "b.png"

    def test_reset_starts_over(self, folder):
        load(folder)
        reset_tracker(folder, "default")
        assert load(folder)["result"][5] == "a.png"

    def test_rejects_a_missing_directory(self, tmp_path):
        with pytest.raises(ValueError, match="Not a directory"):
            load(str(tmp_path / "nope"))

    def test_quoted_path_is_accepted(self, folder):
        assert load(f'"{folder}"')["result"][5] == "a.png"

    def test_is_changed_forces_a_rerun(self):
        assert FVM_BatchLoadImage.IS_CHANGED() != FVM_BatchLoadImage.IS_CHANGED()

    def test_ui_payload_feeds_the_progress_bar(self, folder):
        payload = load(folder)["ui"]["fvm_batch"][0]
        assert payload["position"] == 1 and payload["total"] == 3
        assert 0.0 < payload["fraction"] <= 1.0

    def test_status_route_matches_the_node(self, folder):
        load(folder)
        status = tracker_status(folder)
        assert status["ok"] and status["total"] == 3 and status["remaining"] == 2

    def test_status_route_reports_a_bad_path(self, tmp_path):
        assert tracker_status(str(tmp_path / "nope"))["ok"] is False


class TestBar:
    def test_endpoints(self):
        assert make_bar(0.0, 10) == "░" * 10
        assert make_bar(1.0, 10) == "█" * 10

    def test_clamps_out_of_range(self):
        assert make_bar(2.0, 10) == "█" * 10
        assert make_bar(-1.0, 10) == "░" * 10


def save(directory, **overrides):
    settings = dict(directory=directory, mode="move", filename_prefix="",
                    format="keep", quality=95, overwrite=False, enabled=True)
    settings.update(overrides)
    return FVM_BatchSaveImage().execute(**settings)


class TestSaver:
    def test_move_takes_the_original_away(self, folder, tmp_path):
        source = os.path.join(folder, "a.png")
        target_dir = str(tmp_path / "keep")
        result = save(target_dir, source_path=source)["result"]
        assert result[2] is True
        assert not os.path.exists(source)
        assert os.path.isfile(os.path.join(target_dir, "a.png"))

    def test_copy_leaves_the_original(self, folder, tmp_path):
        source = os.path.join(folder, "a.png")
        save(str(tmp_path / "keep"), mode="copy", source_path=source)
        assert os.path.isfile(source)
        assert os.path.isfile(str(tmp_path / "keep" / "a.png"))

    def test_save_writes_the_tensor(self, tmp_path):
        image = torch.rand(1, 6, 8, 3, dtype=torch.float32)
        path = save(str(tmp_path / "out"), mode="save", image=image,
                    source_path="x/render.png")["result"][1]
        assert os.path.isfile(path)
        assert Image.open(path).size == (8, 6)

    def test_save_honours_the_format(self, tmp_path):
        image = torch.rand(1, 4, 4, 3, dtype=torch.float32)
        path = save(str(tmp_path / "out"), mode="save", format="jpg", image=image,
                    source_path="x/render.png")["result"][1]
        assert path.endswith(".jpg")

    def test_collision_gets_a_suffix(self, folder, tmp_path):
        target = str(tmp_path / "keep")
        os.makedirs(target)
        write_image(os.path.join(target, "a.png"))
        path = save(target, mode="copy",
                    source_path=os.path.join(folder, "a.png"))["result"][1]
        assert path.endswith("a_001.png")

    def test_overwrite_reuses_the_name(self, folder, tmp_path):
        target = str(tmp_path / "keep")
        os.makedirs(target)
        write_image(os.path.join(target, "a.png"))
        path = save(target, mode="copy", overwrite=True,
                    source_path=os.path.join(folder, "a.png"))["result"][1]
        assert path.endswith("a.png") and "_001" not in path

    def test_prefix_is_applied(self, folder, tmp_path):
        path = save(str(tmp_path / "keep"), mode="copy", filename_prefix="ok_",
                    source_path=os.path.join(folder, "a.png"))["result"][1]
        assert os.path.basename(path) == "ok_a.png"

    def test_disabled_does_nothing(self, folder, tmp_path):
        source = os.path.join(folder, "a.png")
        result = save(str(tmp_path / "keep"), enabled=False, source_path=source)["result"]
        assert result[2] is False
        assert os.path.isfile(source)

    def test_missing_source_reports_instead_of_raising(self, tmp_path):
        result = save(str(tmp_path / "keep"), mode="move",
                      source_path=str(tmp_path / "gone.png"))["result"]
        assert result[2] is False and result[1] == ""

    def test_empty_directory_reports_instead_of_raising(self, folder):
        result = save("", source_path=os.path.join(folder, "a.png"))["result"]
        assert result[2] is False

    def test_save_without_image_reports_instead_of_raising(self, tmp_path):
        assert save(str(tmp_path / "out"), mode="save")["result"][2] is False

    def test_report_lands_in_a_sidecar(self, folder, tmp_path):
        path = save(str(tmp_path / "keep"), mode="copy",
                    source_path=os.path.join(folder, "a.png"),
                    report="FAIL torso_twist")["result"][1]
        sidecar = os.path.splitext(path)[0] + ".txt"
        assert open(sidecar, encoding="utf-8").read() == "FAIL torso_twist"

    def test_image_passes_through(self, tmp_path):
        image = torch.rand(1, 4, 4, 3)
        out = save(str(tmp_path / "keep"), enabled=False, image=image)["result"][0]
        assert torch.equal(out, image)

    def test_unique_path_helper(self, tmp_path):
        first = str(tmp_path / "x.png")
        write_image(first)
        assert unique_path(first).endswith("x_001.png")
        assert unique_path(first, overwrite=True) == first


def route(**overrides):
    settings = dict(pass_dir="/keep", fail_dir="/reject", combine="all",
                    label_a="reality", label_b="identity", label_c="extra",
                    count_min=1, count_max=1, invert=False)
    settings.update(overrides)
    return FVM_BatchRouter().execute(**settings)["result"]


class TestRouter:
    def test_all_gates_true_passes(self):
        target, passed, failed, _report = route(gate_a=True, gate_b=True)
        assert (target, passed, failed) == ("/keep", True, False)

    def test_one_false_gate_fails(self):
        target, passed, failed, _report = route(gate_a=True, gate_b=False)
        assert (target, passed, failed) == ("/reject", False, True)

    def test_unconnected_gates_are_not_votes(self):
        """A gate nobody wired up must not reject the batch."""
        assert route(gate_a=True)[1] is True
        assert route()[1] is True

    def test_any_mode(self):
        assert route(combine="any", gate_a=False, gate_b=True)[1] is True
        assert route(combine="any", gate_a=False, gate_b=False)[1] is False

    def test_count_within_bounds(self):
        assert route(gate_a=True, count=1)[1] is True

    def test_count_outside_bounds_fails(self):
        assert route(gate_a=True, count=2)[1] is False
        assert route(gate_a=True, count=0)[1] is False

    def test_count_range_can_be_widened(self):
        assert route(count=3, count_min=1, count_max=4)[1] is True

    def test_invert_swaps_the_folders(self):
        assert route(gate_a=True, invert=True)[0] == "/reject"

    def test_report_names_the_failing_gate(self):
        report = route(gate_a=True, gate_b=False)[3]
        assert "identity fail" in report and "FAIL" in report

    def test_report_chains_upstream_text(self):
        report = route(gate_a=True, report_in="reality: score 0.10")[3]
        assert report.startswith("reality: score 0.10")

    def test_count_reports_the_bound_it_broke(self):
        assert "outside 1..1" in route(count=3)[3]

    def test_non_numeric_count_fails_closed(self):
        assert route(gate_a=True, count="two")[1] is False


class TestChainWiring:
    """The loader's outputs must line up with what the downstream nodes expect."""

    def test_loader_feeds_router_and_saver(self, folder, tmp_path):
        image, _mask, source_path, pass_dir, fail_dir, *_rest = load(folder)["result"]
        target, passed, _failed, report = FVM_BatchRouter().execute(
            pass_dir=pass_dir, fail_dir=fail_dir, combine="all", label_a="reality",
            label_b="identity", label_c="extra", count_min=1, count_max=1,
            invert=False, gate_a=True, count=1)["result"]
        assert passed and target == pass_dir

        result = FVM_BatchSaveImage().execute(
            directory=target, mode="move", filename_prefix="", format="keep",
            quality=95, overwrite=False, enabled=True, image=image,
            source_path=source_path, report=report)["result"]
        assert result[2] is True
        assert os.path.isfile(os.path.join(pass_dir, "a.png"))
        assert not os.path.exists(source_path)

    def test_moved_file_is_not_handed_out_again(self, folder):
        """The tidy-up loop: a moved picture must not come back around."""
        image, _mask, source_path, pass_dir, _fail, *_ = load(folder)["result"]
        FVM_BatchSaveImage().execute(
            directory=pass_dir, mode="move", filename_prefix="", format="keep",
            quality=95, overwrite=False, enabled=True, image=image,
            source_path=source_path, report="")
        assert load(folder)["result"][5] == "b.png"
        assert get_done(folder) == {"a.png", "b.png"}
