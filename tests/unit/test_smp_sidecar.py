"""P6 — tests for FVM_SMP_SidecarSaver."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def tmp_output_dir(tmp_path, monkeypatch):
    """Mock folder_paths to point output_dir at a temp directory."""
    fp = MagicMock()
    fp.get_output_directory = lambda: str(tmp_path)

    def _fake_save_image_path(prefix, output_dir, width=0, height=0):
        sub_path = prefix.replace("/", os.sep)
        full_folder = os.path.join(output_dir, os.path.dirname(sub_path) or "")
        filename = os.path.basename(sub_path)
        if not filename:
            filename = "img"
        return (full_folder, filename, 1, os.path.dirname(sub_path), prefix)

    fp.get_save_image_path = _fake_save_image_path
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    yield tmp_path


def _dummy_image(b=1, h=64, w=64):
    return torch.rand(b, h, w, 3, dtype=torch.float32)


def test_sidecar_writes_png_and_json(tmp_output_dir):
    from nodes.smp.sidecar_saver import FVM_SMP_SidecarSaver

    pd = {"meta": {"seed": 42}, "subjects": [{"id": "s1", "age_desc": "young"}]}
    result = FVM_SMP_SidecarSaver().save(
        images=_dummy_image(b=1),
        filename_prefix="SMP_test/img",
        prompt_dict=pd,
        structured=None,
        extra_metadata_json="{}",
    )
    assert "ui" in result
    images = result["ui"]["images"]
    assert len(images) == 1

    # PNG and sidecar JSON exist
    saved = list(Path(tmp_output_dir).rglob("*.png"))
    assert len(saved) == 1
    sidecars = list(Path(tmp_output_dir).rglob("*.prompt.json"))
    assert len(sidecars) == 1

    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert payload["prompt_dict"]["meta"]["seed"] == 42


def test_sidecar_handles_batch(tmp_output_dir):
    from nodes.smp.sidecar_saver import FVM_SMP_SidecarSaver

    pd = {"meta": {"seed": 1}}
    FVM_SMP_SidecarSaver().save(
        images=_dummy_image(b=3),
        filename_prefix="SMP_batch/img",
        prompt_dict=pd,
    )
    pngs = list(Path(tmp_output_dir).rglob("*.png"))
    sidecars = list(Path(tmp_output_dir).rglob("*.prompt.json"))
    assert len(pngs) == 3
    assert len(sidecars) == 3


def test_sidecar_includes_structured_block(tmp_output_dir):
    from nodes.smp.sidecar_saver import FVM_SMP_SidecarSaver

    pd = {"meta": {"seed": 1}}
    structured = {
        "face": "young woman",
        "outfit": "blazer in burgundy",
        "region_map": [{"region_id": "upper_body"}],
    }
    FVM_SMP_SidecarSaver().save(
        images=_dummy_image(b=1),
        filename_prefix="SMP_struct/img",
        prompt_dict=pd,
        structured=structured,
    )
    sidecars = list(Path(tmp_output_dir).rglob("*.prompt.json"))
    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert payload["structured_prompts"]["face"] == "young woman"
    assert payload["structured_prompts"]["outfit"] == "blazer in burgundy"


def test_sidecar_extra_metadata_merged(tmp_output_dir):
    from nodes.smp.sidecar_saver import FVM_SMP_SidecarSaver

    FVM_SMP_SidecarSaver().save(
        images=_dummy_image(b=1),
        filename_prefix="SMP_extra/img",
        prompt_dict={"meta": {"seed": 1}},
        extra_metadata_json='{"shoot": "spring_2026", "stylist": "claude"}',
    )
    sidecars = list(Path(tmp_output_dir).rglob("*.prompt.json"))
    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert payload["extras"] == {"shoot": "spring_2026", "stylist": "claude"}


def test_sidecar_metadata():
    from nodes.smp.sidecar_saver import FVM_SMP_SidecarSaver
    assert FVM_SMP_SidecarSaver.OUTPUT_NODE is True
    assert FVM_SMP_SidecarSaver.CATEGORY.startswith("FVM Tools/SMP")
    assert FVM_SMP_SidecarSaver.RETURN_TYPES == ()
