import torch

from nodes.list_count import FVM_ListCount, analyse


def cond(n=1):
    return [[torch.zeros(1, 4, 8), {}] for _ in range(n)]


def test_list_of_conditionings_counts_the_list():
    count, kind, text = analyse([cond(), cond(), cond()])
    assert (count, kind) == (3, "LIST")
    assert "CONDITIONING" in text


def test_single_conditioning_counts_entries():
    count, kind, text = analyse([cond(2)])
    assert (count, kind) == (2, "BATCH")
    assert "CONDITIONING" in text


def test_image_batch_and_mask_batch():
    assert analyse([torch.rand(4, 8, 8, 3)])[:2] == (4, "BATCH")
    assert analyse([torch.rand(5, 8, 8)])[:2] == (5, "BATCH")


def test_latent_batch_vs_latent_list():
    assert analyse([{"samples": torch.rand(3, 4, 8, 8)}])[:2] == (3, "BATCH")
    count, kind, text = analyse([{"samples": torch.rand(1, 4, 8, 8)}] * 6)
    assert (count, kind) == (6, "LIST")
    assert "LATENT" in text


def test_single_and_empty_and_missing():
    assert analyse([torch.rand(1, 8, 8, 3)])[:2] == (1, "SINGLE")
    assert analyse(["hello"])[:2] == (1, "SINGLE")
    assert analyse([])[:2] == (0, "EMPTY")
    assert analyse(None)[:2] == (0, "EMPTY")


def test_mixed_list_and_node_output():
    count, kind, text = analyse(["a", 1, 2.0])
    assert (count, kind) == (3, "LIST") and "mixed" in text
    out = FVM_ListCount().doit(value=[torch.rand(2, 4, 4, 3)])
    assert out["result"][0] == 2
    assert out["ui"]["fvm_list_count"][0]["count"] == "2"
    assert FVM_ListCount.INPUT_IS_LIST is True
