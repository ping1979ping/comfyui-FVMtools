"""Generic slot with mask_type 'aux': which aux hits it details."""

import sys
from unittest.mock import MagicMock

import torch

# person_detailer pulls in ComfyUI modules the shared conftest doesn't mock.
for _mod in (
    "comfy.model_base",
    "comfy.lora",
    "comfy.lora_convert",
    "comfy.model_detection",
    "comfy.k_diffusion",
    "comfy.k_diffusion.sampling",
    "node_helpers",
):
    sys.modules.setdefault(_mod, MagicMock())

from nodes.person_detailer import PersonDetailer  # noqa: E402

H = W = 16


def _mask(y0, y1, x0, x1):
    m = torch.zeros(H, W)
    m[y0:y1, x0:x1] = 1.0
    return m


def _person_data(assigned, unassigned=None):
    """assigned: one [H,W] aux mask per reference (single image batch)."""
    pd = {"aux_masks": [[m] for m in assigned]}
    if unassigned is not None:
        pd["aux_unassigned_masks"] = unassigned.unsqueeze(0)
    return pd


PHONE = _mask(2, 5, 2, 5)  # held by reference 1
LOOSE = _mask(10, 13, 10, 13)  # on the table, nobody's


def _slot(index, mask_type):
    return {"index": index, "mask_type": mask_type}


def test_unassigned_only_when_catch_unprocessed_off():
    pd = _person_data([PHONE], LOOSE)
    mask, claimed = PersonDetailer._generic_aux_mask(
        pd, 0, 1, [_slot(0, "head")], False
    )
    assert torch.equal(mask, LOOSE) and claimed == 0


def test_catch_unprocessed_takes_parts_of_ref_without_aux_slot():
    """Ref 1 details the head only -> the phone in its hand goes to Generic."""
    pd = _person_data([PHONE], LOOSE)
    mask, claimed = PersonDetailer._generic_aux_mask(pd, 0, 1, [_slot(0, "head")], True)
    assert torch.equal(mask, torch.max(PHONE, LOOSE)) and claimed == 1


def test_catch_unprocessed_takes_parts_of_disabled_ref():
    pd = _person_data([PHONE])  # no unassigned hits at all
    mask, claimed = PersonDetailer._generic_aux_mask(pd, 0, 1, [], True)
    assert torch.equal(mask, PHONE) and claimed == 1


def test_ref_with_aux_slot_keeps_its_parts():
    pd = _person_data([PHONE], LOOSE)
    mask, claimed = PersonDetailer._generic_aux_mask(pd, 0, 1, [_slot(0, "aux")], True)
    assert torch.equal(mask, LOOSE) and claimed == 0


def test_empty_assigned_masks_are_not_counted():
    pd = _person_data([torch.zeros(H, W)])
    mask, claimed = PersonDetailer._generic_aux_mask(pd, 0, 1, [], True)
    assert mask is None and claimed == 0
