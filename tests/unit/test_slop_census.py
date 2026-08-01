"""The census IS the measuring instrument, so a fault in it is the worst kind.

100 repeats of the census on unchanged pictures (`tests/live/noise_census.md`)
found the PASS/FAIL verdict rock solid — it never flipped once — but turned up
two defects that move the *number*, and one of them moves it in the flattering
direction:

1. A tile whose HTTP call failed used to return `[]`, indistinguishable from a
   tile with nothing written on it. A quarter of the picture then dropped out of
   the count and the run reported LESS slop instead of a fault. Seen in 4 of 100
   runs; it explains a reported drop from a stable 13 down to 4 in full.
2. The vision model files a correctly rendered target word under "gibberish" —
   `RUHETAG` in 10 of 10 runs, plus `KUHETAG` in the same 10, the same word read
   once with a wrong first letter. A target word therefore cost two points and
   its scene could not pass.

The fix for (2) has to stay narrow, or it hides real faults: `IESLIN` and `SLIN`
for the target `RIESLING` are NOT the model misreading a good render, they are
this tool's own lettering coming out truncated. Those still count — in their own
bucket, so the self-inflicted share of the slop stays visible.

These tests need neither ComfyUI nor a vision model; the HTTP layer is a stub.
"""

import importlib.util
import json
import os
import sys

import cv2
import numpy as np
import pytest

LIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live"
)


def _load(name, filename):
    """Import a tests/live script by path — they are scripts, not package members."""
    if LIVE_DIR not in sys.path:
        sys.path.insert(0, LIVE_DIR)
    path = os.path.join(LIVE_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    SC = _load("live_slop_census", "slop_census.py")
    SUITE = _load("live_suite_for_census", "suite.py")
except Exception as exc:  # pragma: no cover - reported through the skip below
    SC = SUITE = None
    LOAD_ERROR = exc
else:
    LOAD_ERROR = None

pytestmark = pytest.mark.skipif(
    SC is None, reason=f"tests/live not importable: {LOAD_ERROR!r}"
)

#: One HTTP answer that failed the way LM Studio actually fails.
FAIL = object()


class FakeVLM:
    """Stands in for `chat_vision`. One entry per HTTP answer, in order.

    An entry is a list of items (a good answer), a raw string (an answer holding
    no JSON) or `FAIL` (transport failure). Running out of entries keeps failing,
    so a test that expects a retry cannot accidentally pass on a fresh success.
    """

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0) if self.answers else FAIL
        if answer is FAIL:
            return {
                "ok": False,
                "content": "",
                "error": 'HTTP 400 {"error":"terminated"}',
                "raw": None,
            }
        if isinstance(answer, str):
            return {"ok": True, "content": answer, "error": None, "raw": None}
        return {
            "ok": True,
            "content": json.dumps({"items": answer}),
            "error": None,
            "raw": None,
        }


def gib(*texts):
    return [{"text": t, "kind": "gibberish"} for t in texts]


def word(*texts):
    return [{"text": t, "kind": "word"} for t in texts]


@pytest.fixture
def picture(tmp_path):
    """A plain image on disk — its content never reaches the stubbed model."""
    path = tmp_path / "pic.png"
    cv2.imwrite(str(path), np.full((64, 64, 3), 200, np.uint8))
    return str(path)


@pytest.fixture
def crop():
    return np.full((32, 32, 3), 180, np.uint8)


# ──── 1. A failed tile is an error, not an empty tile ────


class TestFailedTileIsAnError:
    def test_transport_failure_is_retried_and_can_recover(self, monkeypatch, crop):
        vlm = FakeVLM(FAIL, gib("PAXTRON"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        assert SC.ask(crop) == gib("PAXTRON")
        assert len(vlm.calls) == 2, "the first failure must be retried"

    def test_an_answer_without_json_is_retried_too(self, monkeypatch, crop):
        vlm = FakeVLM("I am afraid I cannot do that", gib("PAXTRON"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        assert SC.ask(crop) == gib("PAXTRON")
        assert len(vlm.calls) == 2

    def test_every_attempt_failing_raises_instead_of_returning_empty(
        self, monkeypatch, crop
    ):
        vlm = FakeVLM(FAIL, FAIL, FAIL)
        monkeypatch.setattr(SC, "chat_vision", vlm)
        with pytest.raises(SC.TileError):
            SC.ask(crop)
        assert len(vlm.calls) == SC.TILE_ATTEMPTS

    def test_an_empty_tile_is_still_allowed_to_be_empty(self, monkeypatch, crop):
        vlm = FakeVLM([])
        monkeypatch.setattr(SC, "chat_vision", vlm)
        assert SC.ask(crop) == []
        assert len(vlm.calls) == 1, "a tile with no text is not an error"

    def test_census_counts_the_lost_tile(self, monkeypatch, picture):
        vlm = FakeVLM(gib("PAXTRON", "TOKLCUBA"), FAIL, FAIL, FAIL)
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=2)
        assert r["tile_errors"] == 1
        assert r["tile_error_detail"][0]["tile"] == 1
        assert "terminated" in r["tile_error_detail"][0]["error"]

    def test_a_lost_tile_shows_less_slop_and_must_say_so(self, monkeypatch, picture):
        """The whole point: the bug made a broken run look like a better one."""
        whole = FakeVLM(gib("PAXTRON", "TOKLCUBA"), gib("HOHIECSABL", "PIATDAZOOK"))
        monkeypatch.setattr(SC, "chat_vision", whole)
        good = SC.census(picture, rows=1, cols=2)

        broken = FakeVLM(gib("PAXTRON", "TOKLCUBA"), FAIL, FAIL, FAIL)
        monkeypatch.setattr(SC, "chat_vision", broken)
        lost = SC.census(picture, rows=1, cols=2)

        assert len(lost["slop"]) < len(good["slop"]), (
            "a lost tile lowers the count — that is exactly why it may not be silent"
        )
        assert good["tile_errors"] == 0
        assert lost["tile_errors"] == 1

    def test_verdict_is_error_not_fail(self):
        v = {"contains": False, "ghosting": 0.9, "slop": ["X"], "tile_errors": 1}
        assert SUITE.verdict(v) == "ERROR"

    def test_verdict_is_error_not_pass(self):
        v = {"contains": True, "ghosting": 0.0, "slop": [], "tile_errors": 1}
        assert SUITE.verdict(v) == "ERROR", (
            "a clean-looking count from an incomplete picture is not a pass"
        )

    def test_verdict_without_tile_errors_is_unchanged(self):
        clean = {"contains": True, "ghosting": 0.1, "slop": [], "tile_errors": 0}
        dirty = {"contains": True, "ghosting": 0.1, "slop": ["X"], "tile_errors": 0}
        ghosty = {"contains": True, "ghosting": 0.8, "slop": [], "tile_errors": 0}
        missing = {"contains": False, "ghosting": 0.0, "slop": [], "tile_errors": 0}
        assert SUITE.verdict(clean) == "PASS"
        assert SUITE.verdict(dirty) == "FAIL"
        assert SUITE.verdict(ghosty) == "FAIL"
        assert SUITE.verdict(missing) == "FAIL"

    def test_verdict_tolerates_a_missing_census(self):
        """`--fast` runs carry no slop key and no tile count."""
        assert SUITE.verdict({"contains": True, "ghosting": 0.1}) == "PASS"


# ──── 2. temperature 0.0 ────


def test_the_census_asks_at_temperature_zero(monkeypatch, crop):
    vlm = FakeVLM(gib("PAXTRON"))
    monkeypatch.setattr(SC, "chat_vision", vlm)
    SC.ask(crop)
    assert vlm.calls[0]["temperature"] == 0.0


# ──── 3. The target word, and the fragment that is not the target word ────


class TestTargetWords:
    def test_the_target_word_itself_is_not_slop(self, monkeypatch, picture):
        vlm = FakeVLM(gib("RUHETAG"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("RUHETAG", "PUTZPLAN"))
        assert len(r["slop"]) == 0
        assert [e["text"] for e in r["target_exact"]] == ["RUHETAG"]

    def test_case_and_spaces_do_not_matter(self, monkeypatch, picture):
        vlm = FakeVLM(gib("Muellel Thurgau"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("MUELLEL THURGAU",))
        assert len(r["slop"]) == 0

    def test_kuhetag_beside_ruhetag_costs_nothing(self, monkeypatch, picture):
        """Measured: both appear in 10 of 10 runs, so the target cost two points."""
        vlm = FakeVLM(gib("RUHETAG", "KUHETAG"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("RUHETAG", "PUTZPLAN"))
        assert len(r["slop"]) == 0, "one rendered word, read twice, is not two faults"
        assert len(r["target_exact"]) == 1
        assert sorted(r["target_exact"][0]["variants"]) == ["KUHETAG", "RUHETAG"]

    def test_kuhetag_is_close_enough_to_be_caught_by_the_threshold(self):
        assert SC.similarity("KUHETAG", "RUHETAG") >= SC.CLUSTER_SIMILARITY
        assert SC.similarity("KUHETAG", "RUHETAG") >= SC.TARGET_SIMILARITY

    def test_a_fragment_of_the_target_still_counts(self, monkeypatch, picture):
        """`IESLIN` is not a misreading of a good render — it is a cut-off render."""
        vlm = FakeVLM(gib("IESLIN"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("RIESLING", "SILVANER"))
        assert len(r["slop"]) == 1, "truncated lettering is a fault, not an excuse"
        assert [e["text"] for e in r["target_fragment"]] == ["IESLIN"]
        assert r["target_fragment"][0]["target"] == "RIESLING"
        assert not r["gibberish"], "it is the tool's own text, not foreign invention"

    def test_slin_and_ieslin_are_one_fragment_not_two(self, monkeypatch, picture):
        vlm = FakeVLM(gib("SLIN", "IESLIN"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("RIESLING",))
        assert len(r["slop"]) == 1
        assert r["target_fragment"][0]["text"] == "IESLIN", "longest reading wins"
        assert sorted(r["target_fragment"][0]["variants"]) == ["IESLIN", "SLIN"]

    def test_a_fragment_without_an_exact_sibling_is_never_excused(
        self, monkeypatch, picture
    ):
        vlm = FakeVLM(gib("SLIN", "IESLIN"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("RIESLING",))
        assert r["target_exact"] == []

    def test_a_cut_off_reading_is_not_excused_by_the_whole_word_beside_it(
        self, monkeypatch, picture
    ):
        """The trap: excusing the blob would hide the truncation behind the word.

        `RIESLING` and `IESLIN` are one blob (0.86). The verbatim reading is not
        a fault, the short one is — so the blob is split rather than excused.
        """
        vlm = FakeVLM(gib("RIESLING", "IESLIN"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("RIESLING",))
        assert [e["text"] for e in r["target_exact"]] == ["RIESLING"]
        assert [e["text"] for e in r["target_fragment"]] == ["IESLIN"]
        assert len(r["slop"]) == 1

    def test_length_is_what_separates_a_misread_from_a_truncation(self):
        """Similarity alone cannot: both pairs sit at exactly 0.86."""
        assert SC.similarity("KUHETAG", "RUHETAG") == SC.similarity(
            "IESLIN", "RIESLING"
        )
        assert len(SC.norm("KUHETAG")) == len(SC.norm("RUHETAG"))
        assert len(SC.norm("IESLIN")) < len(SC.norm("RIESLING"))

    def test_foreign_invention_stays_gibberish(self, monkeypatch, picture):
        vlm = FakeVLM(gib("TOKL CUBA ANY RODA"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("RIESLING", "WEISSBURGUNDER"))
        assert [e["text"] for e in r["gibberish"]] == ["TOKL CUBA ANY RODA"]
        assert r["target_fragment"] == []
        assert len(r["slop"]) == 1

    def test_the_two_kinds_of_fault_are_reported_apart(self, monkeypatch, picture):
        vlm = FakeVLM(gib("TOKL CUBA ANY RODA", "IESLIN", "RIESLING"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("RIESLING",))
        assert len(r["gibberish"]) == 1
        assert len(r["target_fragment"]) == 1
        assert len(r["target_exact"]) == 1
        assert len(r["slop"]) == 2, "both faults count; only the target word does not"

    def test_a_real_word_is_never_slop(self, monkeypatch, picture):
        vlm = FakeVLM(word("RIESLING", "APOTHEKE"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("RIESLING",))
        assert len(r["slop"]) == 0
        assert [i["text"] for i in r["target_hits"]] == ["RIESLING"]

    def test_no_targets_means_everything_foreign(self, monkeypatch, picture):
        vlm = FakeVLM(gib("IESLIN"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1)
        assert len(r["gibberish"]) == 1
        assert r["target_fragment"] == []


# ──── 4. Spelling variants are one blob ────


class TestClustering:
    def test_the_street_smudge_is_one_finding(self, monkeypatch, picture):
        """Measured on street_A: four readings of a single scribble."""
        vlm = FakeVLM(gib("PAKTRDE", "PAXTRDE", "PAXTRON", "PAXTROS"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1, targets=("WEINHANDEL",))
        assert len(r["slop"]) == 1
        assert len(r["gibberish"][0]["variants"]) == 4

    def test_the_chain_needs_single_linkage(self):
        """No single member is close to all the others — hence single linkage."""
        assert SC.similarity("PAKTRDE", "PAXTRON") < SC.CLUSTER_SIMILARITY
        assert SC.similarity("PAKTRDE", "PAXTRDE") >= SC.CLUSTER_SIMILARITY
        assert SC.similarity("PAXTRDE", "PAXTRON") >= SC.CLUSTER_SIMILARITY

    def test_unrelated_scribbles_stay_apart(self, monkeypatch, picture):
        vlm = FakeVLM(gib("PAKTRDE", "HOHIECSABL", "PIATDAZOOK"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1)
        assert len(r["slop"]) == 3

    def test_the_old_count_is_kept_for_comparison(self, monkeypatch, picture):
        vlm = FakeVLM(gib("PAKTRDE", "PAXTRDE", "PAXTRON", "PAXTROS"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=1)
        assert len(r["raw_gibberish"]) == 4
        assert len(r["slop"]) == 1

    def test_representative_is_the_longest_reading(self):
        group = [{"text": "SLIN"}, {"text": "IESLIN"}, {"text": "ESLIN"}]
        assert SC.representative(group)["text"] == "IESLIN"

    def test_identical_strings_are_deduplicated_before_clustering(
        self, monkeypatch, picture
    ):
        vlm = FakeVLM(gib("PAKTRDE"), gib("PAKTRDE"))
        monkeypatch.setattr(SC, "chat_vision", vlm)
        r = SC.census(picture, rows=1, cols=2)
        assert len(r["raw_gibberish"]) == 1
        assert len(r["slop"]) == 1


def test_normalisation_matches_the_detailer():
    """Both sides have to agree on what "nearly the same string" means."""
    from nodes.signs.detailer import _fuzzy_match

    for a, b in (("KUHETAG", "RUHETAG"), ("IESLIN", "RIESLING"), ("slin ", "SLIN")):
        assert SC.similarity(a, b) == pytest.approx(_fuzzy_match(a, b))
