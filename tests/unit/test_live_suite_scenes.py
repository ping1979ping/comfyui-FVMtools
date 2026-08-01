"""The live acceptance suite's scene table has to hold up on its own.

`tests/live/suite.py` cannot run here — it needs ComfyUI and a vision model —
but the part of it that silently ruins a measurement is pure data, and that part
is checkable in a second.

The failure these tests exist to catch: `suite.py` builds its manual override as
`texts[i % len(texts)]`, one line per region. Raise `regions` without lengthening
the word list and the same word gets written onto several surfaces. The census
in `slop_census.py` classifies "the same word repeated down the page as filler"
as gibberish, so the suite then scores its own filler as a pipeline fault and the
number moves for a reason that has nothing to do with the tools.

The A/B lists must also stay disjoint: the suite renders A, then B, then A again,
and a word shared between the two passes would let a leftover from pass A read as
a success in pass B.
"""

import importlib.util
import os
import sys

import pytest

LIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live"
)
SUITE_PATH = os.path.join(LIVE_DIR, "suite.py")


def _load_suite():
    """Import tests/live/suite.py by path — it is a script, not a package member."""
    if LIVE_DIR not in sys.path:
        sys.path.insert(0, LIVE_DIR)
    spec = importlib.util.spec_from_file_location("live_suite", SUITE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    SUITE = _load_suite()
except Exception as exc:  # pragma: no cover - reported through the skip below
    SUITE = None
    LOAD_ERROR = exc
else:
    LOAD_ERROR = None

pytestmark = pytest.mark.skipif(
    SUITE is None, reason=f"tests/live/suite.py not importable: {LOAD_ERROR!r}"
)

SCENE_IDS = [] if SUITE is None else [s["tag"] for s in SUITE.SCENES]
SCENES = [] if SUITE is None else list(SUITE.SCENES)


def _lists(scene):
    a, b = scene["texts"]
    return a, b


@pytest.mark.parametrize("scene", SCENES, ids=SCENE_IDS)
class TestSceneWordLists:
    def test_two_lists_per_scene(self, scene):
        assert len(scene["texts"]) == 2, "a scene needs an A list and a B list"

    def test_no_duplicates_within_a_list(self, scene):
        for label, words in zip("AB", _lists(scene)):
            dupes = sorted({w for w in words if words.count(w) > 1})
            assert not dupes, f"{scene['tag']} list {label} repeats {dupes}"

    def test_a_and_b_are_disjoint(self, scene):
        a, b = _lists(scene)
        shared = sorted(set(a) & set(b))
        assert not shared, (
            f"{scene['tag']}: A and B share {shared} — a leftover from pass A "
            f"would read as a success in pass B"
        )

    def test_list_is_at_least_as_long_as_regions(self, scene):
        for label, words in zip("AB", _lists(scene)):
            assert len(words) >= scene["regions"], (
                f"{scene['tag']} list {label} has {len(words)} words for "
                f"{scene['regions']} regions — the override would cycle and write "
                f"the same word onto several surfaces, which the census scores as "
                f"filler"
            )

    def test_override_gives_every_region_its_own_word(self, scene):
        """Rebuild the exact expression suite.render() uses and check it."""
        for label, words in zip("AB", _lists(scene)):
            assigned = [words[i % len(words)] for i in range(scene["regions"])]
            assert len(set(assigned)) == len(assigned), (
                f"{scene['tag']} list {label}: "
                f"{len(assigned) - len(set(assigned))} region(s) get a repeated word"
            )


def test_scene_tags_are_unique():
    tags = [s["tag"] for s in SCENES]
    assert len(set(tags)) == len(tags)


def test_every_scene_has_a_positive_region_count():
    for scene in SCENES:
        assert isinstance(scene["regions"], int) and scene["regions"] >= 1
