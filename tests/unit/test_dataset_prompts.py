"""Tests for core/dataset_prompts.py and the FVM_DatasetPromptList node."""

import re

import pytest

from core import dataset_prompts as dp
from core.jb import wildcards as wc
from nodes.dataset_prompts import FVM_DatasetPromptList


@pytest.fixture(autouse=True)
def shipped_wildcards(monkeypatch):
    """Resolve against the shipped dataset wildcards, not the user's folder."""
    monkeypatch.setattr(wc, "wildcards_root", dp.shipped_wildcards_dir)
    wc.invalidate_cache()
    yield
    wc.invalidate_cache()


TEXT = """# comment
[close] face front, wearing __dataset/upper__

[half, back] waist up from behind, __dataset/setting__
[full] full body, wearing __dataset/outfit__
untagged line
"""


def test_parse_skips_comments_and_reads_tags():
    lines = dp.parse_lines(TEXT)
    assert [ln.index for ln in lines] == [0, 1, 2, 3]
    assert lines[0].tags == ("close",) and lines[0].text.startswith("face front")
    assert lines[1].tags == ("half", "back") and lines[1].shot == "half"
    assert lines[3].tags == () and lines[3].shot == ""


def test_filter_then_range():
    lines = dp.parse_lines(TEXT)
    assert [ln.index for ln in dp.select_lines(lines, "full")] == [2]
    assert [ln.index for ln in dp.select_lines(lines, "all", 1, 2)] == [1, 2]
    assert dp.select_lines(lines, "close", 5) == []


def test_variations_resolve_every_wildcard_and_differ():
    items = dp.expand(TEXT, variations=6, seed=7)
    assert len(items) == 24
    for it in items:
        assert "__" not in it["prompt"], it["prompt"]
    outfits = {it["prompt"] for it in items if it["index"] == 2}
    assert len(outfits) > 1


def test_same_seed_is_reproducible_and_filter_independent():
    a = dp.expand(TEXT, variations=3, seed=42)
    b = dp.expand(TEXT, variations=3, seed=42, shot_filter="full")
    full_a = [it["prompt"] for it in a if it["index"] == 2]
    assert full_a == [it["prompt"] for it in b]
    assert a != dp.expand(TEXT, variations=3, seed=43)


def test_order_modes():
    rounds = dp.expand(TEXT, variations=2, order="rounds", seed=1)
    assert [it["index"] for it in rounds] == [0, 1, 2, 3, 0, 1, 2, 3]
    per = dp.expand(TEXT, variations=2, order="per_prompt")
    assert [it["index"] for it in per] == [0, 0, 1, 1, 2, 2, 3, 3]
    shuf = dp.expand(TEXT, variations=2, order="shuffle", seed=1)
    assert sorted(it["prompt"] for it in shuf) == sorted(it["prompt"] for it in rounds)


def test_prefix_suffix_excluded_from_caption_and_share_variables():
    items = dp.expand(
        "[close] a {red|blue}^c hat and __^c__ scarf",
        prefix="TRIGGER, {cold|warm}^m",
        suffix="photo, __^m__ mood",
    )
    it = items[0]
    assert it["prompt"].startswith("TRIGGER") and it["prompt"].endswith("mood")
    assert "TRIGGER" not in it["caption"] and "photo" not in it["caption"]
    color = re.search(r"a (red|blue) hat and (\w+) scarf", it["caption"])
    assert color and color.group(1) == color.group(2)
    mood = re.search(r"TRIGGER, (cold|warm), .*photo, (\w+) mood", it["prompt"])
    assert mood and mood.group(1) == mood.group(2)


def test_tidy_cleans_empty_slots():
    assert dp._tidy("a ,  , b ,c..") == "a, b,c."


@pytest.mark.parametrize(
    "name", ["character_complete", "quick_20", "original_25_wildcards"]
)
def test_shipped_presets_resolve_fully(name):
    text = dp.read_preset(name)
    assert text
    lines = dp.parse_lines(text)
    assert all(ln.shot for ln in lines), "every preset line needs a shot tag"
    items = dp.expand(text, variations=4, seed=3)
    leftovers = [
        it["prompt"] for it in items if "__" in it["prompt"] or "{" in it["prompt"]
    ]
    assert not leftovers, leftovers[:3]
    texts = [ln.text for ln in lines]
    assert len(texts) == len(set(texts)), "duplicate lines in preset"


def test_complete_preset_shot_balance():
    lines = dp.parse_lines(dp.read_preset("character_complete"))
    counts = {s: sum(ln.shot == s for ln in lines) for s in dp.SHOT_SIZES}
    assert counts == {"close": 26, "half": 24, "full": 22}


def test_install_default_wildcards_never_overwrites(tmp_path):
    copied = dp.install_default_wildcards(str(tmp_path))
    assert "dataset/outfit" in copied
    target = tmp_path / "dataset" / "color.txt"
    target.write_text("mine\n", encoding="utf-8")
    assert dp.install_default_wildcards(str(tmp_path)) == []
    assert target.read_text(encoding="utf-8") == "mine\n"


def test_preset_name_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "presets_dir", lambda: str(tmp_path))
    assert dp.write_preset("ok_name-1", "x")
    assert not dp.write_preset("../evil", "x")
    assert dp.read_preset("..\\evil") is None
    assert dp.list_presets() == ["ok_name-1"]


def test_node_outputs():
    node = FVM_DatasetPromptList()
    prompt, caption, shot, index, count, listing = node.build(
        TEXT, "all", 0, 1000, 2, "rounds", 5, prefix="", suffix="__dataset/photo__"
    )
    assert count == 8 == len(prompt) == len(caption) == len(shot) == len(index)
    assert shot[:4] == ["close", "half", "full", ""]
    assert listing.startswith("8 prompts — close 2, half 2, full 2, untagged 2")
    assert FVM_DatasetPromptList.OUTPUT_IS_LIST == (
        True,
        True,
        True,
        True,
        False,
        False,
    )


def test_node_raises_on_empty_selection():
    with pytest.raises(ValueError):
        FVM_DatasetPromptList().build(TEXT, "close", 9, 10, 1, "rounds", 0)


def test_input_types_disable_frontend_dynamic_prompts():
    req = FVM_DatasetPromptList.INPUT_TYPES()["required"]
    assert req["prompt_text"][1]["dynamicPrompts"] is False
    assert all("tooltip" in opts for _, opts in req.values())


def test_tidy_fixes_articles_and_suffix_is_comma_joined():
    assert (
        dp._tidy("a off-white shirt, A olive coat, a navy tee")
        == "an off-white shirt, An olive coat, a navy tee"
    )
    it = dp.expand("[close] face", suffix="realistic photo")[0]
    assert it["prompt"] == "face, realistic photo"
