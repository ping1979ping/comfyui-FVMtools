"""Reality check scoring — the inference that turns model answers into a verdict.

No network: every test feeds the scorers the shape a model would return, because
that is where the actual decision is made. The live behaviour of the model is
covered by ``tests/live/reality/calibrate.py``.
"""

import numpy as np
import pytest

from nodes.utils import reality_client as rc


def parts(**counts):
    base = {"heads": 1, "arms": 2, "hands": 2, "legs": 2, "feet_or_shoes": 2}
    base.update(counts)
    return base


def landmarks(chest="breasts", pelvis="groin", navel=False, reason=""):
    return {"chest_shows": chest, "pelvis_shows": pelvis,
            "navel_visible": navel, "reason": reason}


class TestPartCounting:
    def test_a_clean_body_scores_nothing(self):
        assert rc._score_parts(parts(), {}) == {}

    def test_two_heads_is_a_second_person(self):
        found = rc._score_parts(parts(heads=2), {})
        assert "extra_heads" in found and found["extra_heads"][0] == 1.0

    def test_third_leg(self):
        assert "extra_legs" in rc._score_parts(parts(legs=3), {})

    def test_missing_limbs_are_occlusion_not_defects(self):
        """Fewer parts than expected means hidden, not absent — never a defect."""
        assert rc._score_parts(parts(arms=1, legs=1, hands=0, feet_or_shoes=0), {}) == {}

    def test_limits_scale_with_expected_people(self):
        assert rc._score_parts(parts(heads=2, arms=4), {"expected_people": 2}) == {}
        assert "extra_heads" in rc._score_parts(parts(heads=3), {"expected_people": 2})

    def test_unusable_count_is_ignored(self):
        assert rc._score_parts({"heads": "several"}, {}) == {}

    def test_note_names_the_count(self):
        note = rc._score_parts(parts(heads=2), {})["extra_heads"][1]
        assert "2 heads" in note


class TestPeople:
    def test_expected_count_is_clean(self):
        assert rc._score_people({"people": 1, "bodies_merged": False}, {}) == {}

    def test_extra_person(self):
        found = rc._score_people({"people": 2}, {})
        assert found["person_count"][0] == 1.0

    def test_missing_person_scores_lower_than_an_extra_one(self):
        assert rc._score_people({"people": 0}, {})["person_count"][0] < 1.0

    def test_merged_bodies(self):
        assert "bodies_merged" in rc._score_people(
            {"people": 1, "bodies_merged": True}, {})

    def test_string_boolean_is_understood(self):
        assert "bodies_merged" in rc._score_people(
            {"people": 1, "bodies_merged": "yes"}, {})


class TestLandmarks:
    def test_consistent_front_view_is_clean(self):
        assert rc._score_landmarks(landmarks(), {}) == {}

    def test_consistent_rear_view_is_clean(self):
        """A woman kneeling with her back to the camera is a normal photograph."""
        assert rc._score_landmarks(
            landmarks(chest="shoulder_blades", pelvis="buttock_cleft"), {}) == {}

    def test_front_chest_with_rear_pelvis_is_impossible(self):
        found = rc._score_landmarks(landmarks(pelvis="buttock_cleft"), {})
        assert found["torso_twist"][0] == 1.0

    def test_navel_above_a_buttock_cleft_is_impossible(self):
        found = rc._score_landmarks(
            landmarks(chest="neither", pelvis="buttock_cleft", navel=True), {})
        assert "navel_and_buttocks" in found

    def test_reverse_twist(self):
        found = rc._score_landmarks(
            landmarks(chest="shoulder_blades", pelvis="groin"), {})
        assert "torso_twist_reverse" in found

    def test_edge_on_views_are_not_contradictions(self):
        assert rc._score_landmarks(landmarks(chest="neither", pelvis="neither"), {}) == {}

    def test_answer_wrapped_in_prose_still_parses(self):
        found = rc._score_landmarks(
            {"chest_shows": "  Breasts ", "pelvis_shows": '"buttock_cleft"'}, {})
        assert "torso_twist" in found


class TestAggregation:
    def test_median_ignores_a_single_outlier(self):
        assert rc.aggregate_passes([0.0, 0.0, 1.0]) == 0.0
        assert rc.aggregate_passes([1.0, 1.0, 0.0]) == 1.0

    def test_max_takes_the_outlier(self):
        assert rc.aggregate_passes([0.0, 0.0, 1.0], "max") == 1.0

    def test_mean(self):
        assert rc.aggregate_passes([0.0, 1.0], "mean") == 0.5

    def test_majority_needs_more_than_half(self):
        assert rc.aggregate_passes([1.0, 0.0, 0.0], "majority") == 0.0
        assert rc.aggregate_passes([1.0, 1.0, 0.0], "majority") == 1.0

    def test_empty_is_clean(self):
        assert rc.aggregate_passes([]) == 0.0

    def test_single_pass_passes_through(self):
        assert rc.aggregate_passes([0.85]) == 0.85


class TestCombine:
    def test_worst_finding_wins_not_the_sum(self):
        """Four mild doubts must not add up to a conviction."""
        violations = {f"v{i}": {"severity": 0.3, "probe": "parts"} for i in range(4)}
        assert rc.combine_violations(violations) == pytest.approx(0.3)

    def test_one_conclusive_finding_condemns(self):
        assert rc.combine_violations(
            {"extra_heads": {"severity": 1.0, "probe": "parts"}}) == 1.0

    def test_weights_scale_a_probe_down(self):
        violations = {"hand_malformed": {"severity": 1.0, "probe": "hands"}}
        assert rc.combine_violations(violations, {"hands": 0.2}) == pytest.approx(0.2)

    def test_nothing_found_is_zero(self):
        assert rc.combine_violations({}) == 0.0


class TestConfirmation:
    def test_rear_answer_upholds(self):
        assert rc._confirmation_verdict([{"ok": True, "data": {"choice": "B"}}]) is True

    def test_front_answer_overturns(self):
        assert rc._confirmation_verdict([{"ok": True, "data": {"choice": "A"}}]) is False

    def test_prose_around_the_letter_still_counts(self):
        assert rc._confirmation_verdict(
            [{"ok": True, "data": {"choice": "B — the rear, two cheeks"}}]) is True

    def test_majority_of_passes_decides(self):
        answers = [{"ok": True, "data": {"choice": c}} for c in ("B", "B", "A")]
        assert rc._confirmation_verdict(answers) is True

    def test_no_usable_answer_abstains(self):
        assert rc._confirmation_verdict([{"ok": False}]) is None
        assert rc._confirmation_verdict([]) is None


class TestCoercion:
    @pytest.mark.parametrize("value,expected", [
        (1, 1), (2.0, 2), ("3", 3), ("about 4 legs", 4), (None, None),
        (True, None), ("many", None), ([1, 2], 2),
    ])
    def test_counts(self, value, expected):
        assert rc._as_count(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (0.5, 0.5), ("0.5", 0.5), (85, 0.85), ("85%", 0.85), (True, 1.0),
        (-1, 0.0), (2.0, 0.02), ("nonsense", 0.0),
    ])
    def test_unit_floats(self, value, expected):
        assert rc._clamp01(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value,expected", [
        (True, True), ("yes", True), ("true", True), ("no", False),
        ("false", False), (None, False), ("maybe", False),
    ])
    def test_booleans(self, value, expected):
        assert rc._as_bool(value) is expected


class TestAssessWithoutAServer:
    """Every failure path has to degrade, not raise — a batch must keep running."""

    def test_unreachable_model_passes_and_says_so(self, monkeypatch):
        monkeypatch.setattr(rc, "chat_vision", lambda **kwargs: {
            "ok": False, "content": "", "error": "Cannot reach LM Studio", "raw": None})
        result = rc.assess(np.zeros((8, 8, 3), dtype=np.uint8))
        assert result["ok"] is False
        assert result["passed"] is True, "a blind inspector must not reject a batch"
        assert "unavailable" in result["report"]

    def test_unparsable_answers_abstain(self, monkeypatch):
        monkeypatch.setattr(rc, "chat_vision", lambda **kwargs: {
            "ok": True, "content": "I'm not sure, sorry!", "error": None, "raw": {}})
        result = rc.assess(np.zeros((8, 8, 3), dtype=np.uint8))
        assert result["ok"] is False and result["passed"] is True

    def test_clean_answers_pass(self, monkeypatch):
        monkeypatch.setattr(rc, "chat_vision", self._answers({
            "parts": parts(), "people": {"people": 1, "bodies_merged": False},
            "landmarks": landmarks(),
        }))
        result = rc.assess(np.zeros((8, 8, 3), dtype=np.uint8))
        assert result["passed"] is True and result["score"] == 0.0

    def test_two_heads_fails(self, monkeypatch):
        monkeypatch.setattr(rc, "chat_vision", self._answers({
            "parts": parts(heads=2), "people": {"people": 2},
            "landmarks": landmarks(),
        }))
        result = rc.assess(np.zeros((8, 8, 3), dtype=np.uint8))
        assert result["passed"] is False
        assert any("extra_heads" in issue for issue in result["issues"])

    def test_twist_upheld_by_the_confirmation(self, monkeypatch):
        monkeypatch.setattr(rc, "chat_vision", self._answers({
            "parts": parts(), "people": {"people": 1},
            "landmarks": landmarks(pelvis="buttock_cleft"),
            "pelvis_confirm": {"choice": "B"},
        }))
        result = rc.assess(np.zeros((8, 8, 3), dtype=np.uint8))
        assert result["passed"] is False
        assert "torso_twist" in result["violations"]

    def test_twist_overturned_by_the_confirmation(self, monkeypatch):
        """The measured false alarm: a seated open-legged pose read as a rear view."""
        monkeypatch.setattr(rc, "chat_vision", self._answers({
            "parts": parts(), "people": {"people": 1},
            "landmarks": landmarks(pelvis="buttock_cleft"),
            "pelvis_confirm": {"choice": "A"},
        }))
        result = rc.assess(np.zeros((8, 8, 3), dtype=np.uint8))
        assert result["passed"] is True
        assert "torso_twist" not in result["violations"]
        assert "overturned" in result["report"]

    def test_confirmation_is_skipped_when_nothing_to_confirm(self, monkeypatch):
        asked = []

        def record(**kwargs):
            asked.append(kwargs["user_prompt"])
            return {"ok": True, "error": None, "raw": {},
                    "content": '{"heads":1,"arms":2,"hands":2,"legs":2,'
                               '"feet_or_shoes":2,"people":1,'
                               '"chest_shows":"breasts","pelvis_shows":"groin"}'}

        monkeypatch.setattr(rc, "chat_vision", record)
        rc.assess(np.zeros((8, 8, 3), dtype=np.uint8))
        assert not any("THE REAR" in prompt for prompt in asked), \
            "the confirmation stage should not run on a clean picture"

    def test_confirmation_cannot_condemn_on_its_own(self, monkeypatch):
        """It only ever clears a suspicion; a lone 'B' must not fail a clean picture."""
        monkeypatch.setattr(rc, "chat_vision", self._answers({
            "parts": parts(), "people": {"people": 1}, "landmarks": landmarks(),
            "pelvis_confirm": {"choice": "B"},
        }))
        assert rc.assess(np.zeros((8, 8, 3), dtype=np.uint8))["passed"] is True

    def test_json_summary_is_machine_readable(self, monkeypatch):
        import json
        monkeypatch.setattr(rc, "chat_vision", self._answers({
            "parts": parts(heads=2), "people": {"people": 2}, "landmarks": landmarks(),
        }))
        data = json.loads(rc.verdict_to_json(rc.assess(np.zeros((8, 8, 3), dtype=np.uint8))))
        assert data["passed"] is False
        assert data["violations"]["extra_heads"] == 1.0

    @staticmethod
    def _answers(by_probe):
        """Fake chat_vision that answers by matching the prompt to a probe."""
        import json as _json

        def fake(**kwargs):
            prompt = kwargs["user_prompt"]
            for name, payload in by_probe.items():
                if rc.PROBES[name]["prompt"] == prompt:
                    return {"ok": True, "content": _json.dumps(payload),
                            "error": None, "raw": {}}
            return {"ok": False, "content": "", "error": "unexpected probe", "raw": None}
        return fake
