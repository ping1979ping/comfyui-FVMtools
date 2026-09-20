"""Unit-Tests: Song verlaengern (ABC-Abschnitte, Ausrichtung, Session, Nodes-Logik)."""

import json
import os
from fractions import Fraction

import numpy as np
import pytest

from nodes.music import extend_session as es
from nodes.music.abc_score import (
    AbcError,
    RepairReport,
    bar_duration,
    extract_abc_block,
    format_score,
    onset_times,
    outline,
    parse_body,
    parse_score,
    repair_bar,
    repair_sections,
    section_tag,
    select_sections,
    silence_voice,
    validate_sections,
    voice_is_silent,
)
from nodes.music.audio_stitch import (
    Segment,
    arrangement_segments,
    fit_score_timing,
    join_segments,
    parse_arrangement,
)
from nodes.music.melody_template import (
    build_template,
    key_alterations,
    parse_llm_json,
    parse_llm_sections,
    pitch_to_abc,
    shift_octave,
)
from nodes.music.song_extend import (
    MODES,
    apply_plan,
    build_plan,
    commit_part,
    merge_session,
    plan_labels,
    read_production_json,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "music", "sloop_cover.abc")


@pytest.fixture
def abc_text():
    with open(FIXTURE, encoding="utf-8") as handle:
        return handle.read()


@pytest.fixture
def score(abc_text):
    return parse_score(abc_text)


# ---------------------------------------------------------------- ABC

class TestAbcScore:
    def test_roundtrip_is_exact(self, abc_text, score):
        assert format_score(score).strip() == abc_text.strip()

    def test_header_values(self, score):
        assert score.voices == ["Vocal", "Ins"]
        assert score.bar_units == 32
        assert score.tempo_bpm == 129
        assert score.key == "Ab"

    def test_duration_matches_toolkit_timeline(self, score):
        # Toolkit-Timeline fuer dieselbe Partitur: 179.0698 s
        assert score.total_seconds() == pytest.approx(179.07, abs=0.01)

    def test_inline_meter_change_is_valid(self, score):
        assert validate_sections(score, score.sections) == []
        assert score.end_meters() == {"Vocal": Fraction(1), "Ins": Fraction(1)}

    def test_outline(self, score):
        o = outline(score)
        assert [x["label"] for x in o] == ["intro", "verse", "chorus", "verse", "chorus", "verse", "chorus"]
        assert o[1]["bars"] == 16

    def test_vocal_voice_is_silent_in_instrumental_cover(self, score):
        assert voice_is_silent(score, "Vocal")
        assert not voice_is_silent(score, "Ins")

    @pytest.mark.parametrize("bar,expected", [("c8c8c8c8", 32), ('"G#"z32', 32), ("Z", 32),
                                               ("c'4-c'4z24", 32), ("^F/2z", Fraction(3, 2))])
    def test_bar_duration(self, bar, expected):
        assert bar_duration(bar, Fraction(32)) == expected

    def test_repair_short_bar_pads_rest(self):
        bar, note = repair_bar("c8c8", Fraction(32))
        assert bar_duration(bar, Fraction(32)) == 32 and bar.endswith("z16") and note

    def test_repair_long_bar_trims_end(self):
        bar, note = repair_bar("c16d16e8", Fraction(32))
        assert bar_duration(bar, Fraction(32)) == 32 and bar == "c16d16" and note

    def test_repair_drops_invalid_chord(self):
        bar, note = repair_bar('"xyz"c32', Fraction(32))
        assert bar == "c32" and "Akkord" in note

    def test_unsupported_notation_raises(self):
        with pytest.raises(AbcError):
            bar_duration("(3cde c8", Fraction(32))

    def test_select_sections(self, score):
        assert [s.label for s in select_sections(score, "last2")] == ["verse", "chorus"]
        assert select_sections(score, "chorus")[0] is score.sections[-1]
        assert [s.label for s in select_sections(score, "2-3")] == ["verse", "chorus"]
        with pytest.raises(AbcError):
            select_sections(score, "bridge")

    def test_parse_llm_variants(self, score):
        text = "X:1\nM:4/4\n% bridge\n[V:1] z32|z32|\n[V:2] c32|c32|\n"
        secs = parse_body(extract_abc_block(text), score.voices)
        assert len(secs) == 1 and secs[0].label == "bridge"
        assert validate_sections(score, secs) == []

    def test_unknown_voice_raises(self, score):
        with pytest.raises(AbcError):
            parse_body("% x\nV: Drums\nz32|\n", score.voices)

    def test_silence_voice_keeps_chords(self, score):
        secs = parse_body('% x\nV: Vocal\n"G#"c8d8e8f8|\nV: Ins\nz32|\n', score.voices)
        out, changed = silence_voice(score, secs, "Vocal")
        assert changed == 4
        assert out[0].groups[0]["Vocal"] == '"G#"z8z8z8z8|'

    def test_group_bar_count_mismatch_repaired(self, score):
        secs = parse_body("% x\nV: Vocal\nz32|z32|\nV: Ins\nc32|\n", score.voices)
        rep = RepairReport()
        fixed = repair_sections(score, secs, report=rep)
        assert validate_sections(score, fixed) == []
        assert rep.bars_repaired == 1

    def test_section_tags(self):
        assert section_tag("intro") == "[Intro]"
        assert section_tag("chorus") == "[Instrumental]"
        assert section_tag("chorus", instrumental=False) == "[Chorus]"


# ---------------------------------------------------------------- Plan / Apply

class TestPlanApply:
    def test_auto_labels_follow_song_form(self, score):
        assert plan_labels(score, 3, "auto") == ["verse", "chorus", "verse"]
        assert plan_labels(score, 2, "bridge") == ["bridge", "bridge"]

    def test_llm_plan_prompts(self, abc_text):
        plan = build_plan(abc_text, MODES[0], 1, "verse", 0, "last", "build up", 1)
        assert plan["needs_llm"] and plan["labels"] == ["verse"]
        assert plan["template_sources"] == [5]                      # letzter verse (Abschnitt 6)
        assert '"Ins": [[' in plan["system_prompt"]                 # nur die Melodiestimme
        assert "Ab major" in plan["system_prompt"]
        assert '"Db5"' in plan["user_prompt"]                        # Tonart aufgeloest
        assert '"Ins": [["?", "?"' in plan["user_prompt"]            # Antwortgeruest
        assert "notes per bar = [" in plan["user_prompt"]
        assert "build up" in plan["user_prompt"]
        assert "Vocal\" range" not in plan["user_prompt"]           # Vocal ist stumm

    def test_loop_needs_no_llm(self, abc_text):
        plan = build_plan(abc_text, MODES[2], 2, "auto", 0, "chorus", "", 1)
        assert not plan["needs_llm"] and plan["system_prompt"] == ""

    def test_apply_loop(self, abc_text, score):
        plan = build_plan(abc_text, MODES[2], 2, "auto", 0, "chorus", "", 1)
        part = apply_plan(plan)
        assert part["source"] == "loop" and part["labels"] == ["chorus", "chorus"]
        render = parse_score(part["render_abc"])
        # Ueberlapp + 2 Wiederholungen + Ausklang (nur im Render)
        assert [s.label for s in render.sections] == ["chorus", "chorus", "chorus", "outro"]
        # letzter Chorus ohne seine zwei Schluss-Pausentakte (Z|): 19 statt 21 Takte
        assert part["overlap_seconds"] == pytest.approx(35.35, abs=0.01)
        assert part["material_seconds"] == pytest.approx(3 * 35.35, abs=0.05)
        assert any("Schluss-Pausentakt" in n for n in part["notes"])
        assert part["render_lyrics"].split("\n\n") == ["[Instrumental]"] * 3 + ["[Outro]"]
        after = parse_score(part["full_abc_after"])
        assert len(after.sections) == len(score.sections) + 2
        assert validate_sections(after, after.sections) == []

    def _json_answer(self, plan, abc_text, transform=lambda p: p, chords=None, extra=None):
        score = parse_score(abc_text)
        out = []
        for label, src in zip(plan["labels"], plan["template_sources"]):
            tpl = build_template(score, score.sections[src], label, plan["bars"])
            entry = {"label": label, "Ins": [[transform(p) for p in bar] for bar in tpl.pitches("Ins")]}
            if chords is not None:
                entry["chords"] = chords
            out.append(entry)
        if extra:
            out[0].update(extra)
        return "Sure!\n```json\n" + json.dumps({"sections": out}) + "\n```"

    def test_apply_valid_llm_json(self, abc_text):
        plan = build_plan(abc_text, MODES[0], 1, "verse", 0, "last", "", 1)
        part = apply_plan(plan, self._json_answer(plan, abc_text, lambda p: "F5"))
        assert part["source"] == "llm" and part["labels"] == ["verse"]
        assert part["warnings"] == []
        # Rhythmus bleibt: gleiche Dauer wie der Vorlagen-Verse (8 Takte)
        assert part["new_seconds"] == pytest.approx(8 * 1.86, abs=0.05)
        ins = parse_body(part["new_abc"], ["Vocal", "Ins"])[0].groups[0]["Ins"]
        assert ins.startswith("f8") and "c" not in ins

    def test_llm_wrong_counts_are_adjusted(self, abc_text):
        plan = build_plan(abc_text, MODES[0], 1, "verse", 0, "last", "", 1)
        text = json.dumps({"sections": [{"Ins": [["C5"], ["D5", "Eb5", "F5", "G5", "Ab5", "Bb5", "C6"]]}]})
        part = apply_plan(plan, text)
        assert part["source"] == "llm" and part["warnings"] == []
        assert any("Tonzahl angepasst" in n for n in part["notes"])

    def test_llm_garbage_falls_back_to_template(self, abc_text, score):
        plan = build_plan(abc_text, MODES[0], 1, "verse", 0, "last", "", 1)
        part = apply_plan(plan, "Sorry, I cannot write music.")
        assert part["source"] == "fallback" and part["labels"] == ["verse"]
        assert part["warnings"] == []

    def test_llm_chords_and_octave_folding(self, abc_text):
        plan = build_plan(abc_text, MODES[1], 1, "verse", 0, "last", "", 1)
        part = apply_plan(plan, self._json_answer(plan, abc_text, lambda p: "C8", chords=["C#"] * 16))
        assert part["source"] == "llm"
        assert "c''''" not in part["new_abc"]
        assert any("Oktave" in x for x in part["notes"])
        assert '"C#"' in part["new_abc"]

    def test_llm_output_stays_instrumental(self, abc_text):
        plan = build_plan(abc_text, MODES[0], 1, "verse", 0, "last", "", 1)
        text = self._json_answer(plan, abc_text, extra={"Vocal": [["c", "d"]] * 8})
        part = apply_plan(plan, text)
        assert voice_is_silent(parse_score(part["full_abc_after"]), "Vocal")


class TestMelodyTemplate:
    def test_ties_form_one_slot_across_barline(self, score):
        secs = parse_body("% x\nV: Vocal\nz32|z32|\nV: Ins\nc16B16-|B8z24|\n", score.voices)
        tpl = build_template(score, secs[0], "x")
        assert tpl.counts("Ins") == [2, 0]
        sec, _ = tpl.fill({"Ins": [["Eb5", "F5"], []]}, None)   # Es steht in As-Dur
        assert sec.groups[0]["Ins"] == "e16f16-|f8z24|"

    def test_key_alterations(self):
        assert sorted(l for l, a in key_alterations("Ab").items() if a == -1) == ["A", "B", "D", "E"]
        assert key_alterations("Em")["F"] == 1 and key_alterations("C")["B"] == 0

    def test_accidentals_follow_key_and_bar(self):
        alts = key_alterations("Ab")
        state = {}
        assert pitch_to_abc(("B", 4, -1), alts, state) == "B"       # Bb steht in der Tonart
        assert pitch_to_abc(("B", 4, 0), alts, state) == "=B"       # Aufloesung noetig
        assert pitch_to_abc(("B", 4, 0), alts, state) == "B"        # gilt bis Taktende
        assert pitch_to_abc(("E", 5, -1), alts, {}) == "e"

    def test_broken_json_still_yields_valid_sections(self):
        # wie bei mistral: im zweiten Abschnitt fehlt ein schliessendes Anfuehrungszeichen
        text = '{"sections": [{"label": "verse", "Ins": [["C5"]]}, {"label": "chorus", "Ins": [["D5]]}]}'
        secs = parse_llm_sections(text)
        assert secs and secs[0]["label"] == "verse"

    def test_template_drops_irregular_bars_and_fits_length(self, score):
        chorus = score.sections[2]                       # enthaelt den 1/4-Takt
        tpl = build_template(score, chorus, "chorus", 8)
        assert tpl.bar_count == 8
        assert any("unregelmaessig" in n for n in tpl.notes)
        sec, _ = tpl.fill({}, None)
        assert validate_sections(score, [sec]) == []

    def test_parse_llm_json_variants(self):
        assert parse_llm_json('x {"a": 1} y')["a"] == 1
        assert parse_llm_json('```json\n{"sections": []}\n```') == {"sections": []}
        with pytest.raises(AbcError):
            parse_llm_json("no json here")

    @pytest.mark.parametrize("head,octaves,expected", [("c", 1, "c'"), ("c'", -1, "c"),
                                                       ("C", -1, "C,"), ("^f", -1, "^F")])
    def test_shift_octave(self, head, octaves, expected):
        assert shift_octave(head, octaves) == expected


# ---------------------------------------------------------------- Audio

SR = 8000


def click_track(seconds, seed=0, sr=SR):
    """Zufaellige Onsets (Rauschbursts), reproduzierbar - gut ausrichtbar."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    wave = np.zeros(n, dtype=np.float32)
    t = 0
    while t < n - 400:
        burst = rng.standard_normal(400).astype(np.float32) * np.exp(-np.arange(400) / 80.0)
        wave[t:t + 400] += burst * rng.uniform(0.3, 1.0)
        t += int(rng.uniform(0.08, 0.4) * sr)
    return np.stack([wave, wave])


def render_from_score(score, offset, scale=1.0, tail=8.0, seed=0, sr=SR):
    """Synthetisches 'Rendering': Bursts an den Partitur-Einsaetzen, mit Vorlauf,
    Tempo-Faktor, etwas Timing-Jitter und Ausklang - wie YuE2, nur messbar."""
    rng = np.random.default_rng(seed)
    n = int((offset + scale * score.total_seconds() + tail) * sr)
    wave = np.zeros(n, dtype=np.float32)
    for t, w in onset_times(score):
        i = int((offset + scale * t + rng.normal(0, 0.008)) * sr)
        if 0 <= i < n - 400:
            wave[i:i + 400] += rng.standard_normal(400).astype(np.float32) * np.exp(-np.arange(400) / 80.0) * w * 0.4
    wave += rng.standard_normal(n).astype(np.float32) * 0.01
    return np.stack([wave, wave])


class TestAudioStitch:
    @pytest.mark.parametrize("offset,scale", [(2.47, 1.0), (0.3, 1.02), (5.0, 0.98)])
    def test_score_timing_fit_recovers_offset_and_tempo(self, score, offset, scale):
        wave = render_from_score(score, offset, scale, seed=int(offset * 10))
        t = fit_score_timing(wave, SR, onset_times(score), score.total_seconds())
        assert t.offset == pytest.approx(offset, abs=0.03)
        assert t.scale == pytest.approx(scale, abs=0.0025)
        assert t.contrast > 3.0

    def test_repetitive_audio_vs_audio_would_be_ambiguous_but_score_is_not(self, score):
        # Die ganze Partitur als Muster ist eindeutig, obwohl sich die Refrains wiederholen
        wave = render_from_score(score, 1.0)
        t = fit_score_timing(wave, SR, onset_times(score), score.total_seconds())
        assert t.contrast > 3.0

    def test_onset_times_follow_score(self, score):
        ons = onset_times(score)
        assert ons[0][0] >= 0 and ons[-1][0] < score.total_seconds()
        assert sum(1 for _, w in ons if w == 1.0) > 100

    def test_join_segments_length_and_crossfade(self):
        a = np.ones((2, 1000), np.float32)
        b = np.ones((2, 1000), np.float32) * 0.5
        out = join_segments([Segment(a, 0, 600), Segment(b, 300, None)], sr=1000, crossfade_ms=100)
        assert out.shape[-1] == 600 + 700
        assert out[0, 0] == 1.0 and out[0, -1] == 0.5

    def test_parse_arrangement(self):
        assert parse_arrangement("alle", 3) == [0, 1, 2]
        assert parse_arrangement("0,1,2,2,3", 4) == [0, 1, 2, 2, 3]
        assert parse_arrangement("0-1,2x3", 3) == [0, 1, 2, 2, 2]
        with pytest.raises(ValueError):
            parse_arrangement("0,5", 3)

    def test_arrangement_segments_rules(self):
        w = np.zeros((2, 100), np.float32)
        parts = [(w, {"start": 0, "end": 80, "score_end": 80}),
                 (w, {"start": 10, "end": 70, "score_end": 60}),
                 (w, {"start": 15, "end": None, "score_end": 50})]
        segs = arrangement_segments(parts, [0, 1, 1, 2])
        assert [(s.start, s.end) for s in segs] == [(0, 80), (10, 60), (10, 70), (15, None)]


# ---------------------------------------------------------------- Session end-to-end

class TestSession:
    def test_append_undo_reset_merge(self, tmp_path, abc_text, score):
        d = str(tmp_path / "sess")
        base = render_from_score(score, offset=2.47, tail=10.0, seed=3)      # wie der echte Song
        state = es.init_or_load(d, "t", abc_text, base, SR, "style", "lyrics", "Title", "test")
        assert os.path.exists(os.path.join(d, "part_000.flac"))

        plan = build_plan(abc_text, MODES[2], 1, "auto", 0, "chorus", "", 1)
        part = apply_plan(plan)
        render_score = parse_score(part["render_abc"])
        render = render_from_score(render_score, offset=0.27, tail=6.0, seed=4)
        rec = commit_part(d, part, render, SR)
        assert rec["index"] == 1 and rec["file"] == "part_001.flac"
        # Schnitt an einem Taktstrich im Ueberlapp, >= 8 s vor dessen Ende (dort spielt
        # der Vorgaenger schon "Schluss")
        cut = rec["cut_in_overlap_s"]
        bar = 60 / 129 * 4
        assert cut <= part["overlap_seconds"] - 8.0
        assert (cut / bar) == pytest.approx(round(cut / bar), abs=0.01)
        # Vorgaenger: Vorlauf + Beginn des Ueberlapps (Partitur ohne Schluss-Pausen) + cut
        overlap_start = 179.07 - 2 * 1.8605 - part["overlap_seconds"]
        assert rec["prev_cut"] / SR == pytest.approx(2.47 + overlap_start + cut, abs=0.03)
        # Neuer Teil: Vorlauf + dieselbe Stelle im Ueberlapp
        assert rec["start"] / SR == pytest.approx(0.27 + cut, abs=0.03)
        state = es.load_state(d)
        assert len(state["parts"]) == 2
        assert len(parse_score(state["full_abc"]).sections) == 8

        wave, sr, order, report = merge_session(d, "alle", crossfade_ms=50)
        assert order == [0, 1]
        expected = rec["prev_cut"] / SR + (render.shape[-1] - rec["start"]) / SR
        assert wave.shape[-1] / sr == pytest.approx(expected, abs=0.1)

        wave2, *_ = merge_session(d, "0,1,1", crossfade_ms=50)
        assert wave2.shape[-1] > wave.shape[-1]

        es.undo_last(d, es.load_state(d))
        state = es.load_state(d)
        assert len(state["parts"]) == 1 and state["full_abc"].strip() == abc_text.strip()
        assert not os.path.exists(os.path.join(d, "part_001.flac"))

        es.reset(d, state)
        assert es.load_state(d)["parts"][0]["end"] is None

    def test_other_song_with_parts_is_refused(self, tmp_path, abc_text):
        d = str(tmp_path / "sess")
        base = click_track(10)
        state = es.init_or_load(d, "t", abc_text, base, SR, "", "", "A", "test")
        state["parts"].append({"index": 1, "file": "x.flac"})
        es.save_state(d, state)
        with pytest.raises(ValueError):
            es.init_or_load(d, "t", abc_text + "\n% x\n", base, SR, "", "", "B", "test")


def test_read_production_json(tmp_path):
    data = {"title": "T", "generation": {"abc": "X:1\nK:C\n", "cover_conditioning": {
        "native_style": "S", "native_lyrics": "[Intro]"}},
        "outputs": {"original_audio": {"path": "/x/a.flac"}}}
    p = tmp_path / "log.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    info = read_production_json(str(p))
    assert info == {"abc": "X:1\nK:C\n", "style": "S", "lyrics": "[Intro]", "title": "T",
                    "audio_path": "/x/a.flac", "seed": ""}


class TestKeySnapping:
    def test_out_of_key_notes_are_pulled_into_scale(self):
        from nodes.music.melody_template import respell
        alts = key_alterations("Ab")
        assert respell(("D", 5, 0), alts, True) == (("D", 5, -1), True)     # D -> Db
        assert respell(("A", 4, 1), alts, True) == (("B", 4, -1), False)    # A# = Bb, nur umbenannt
        assert respell(("G", 4, 1), alts, True) == (("A", 4, -1), False)    # G# = Ab
        assert respell(("D", 5, 0), alts, False) == (("D", 5, 0), False)    # chromatisch erlaubt

    def test_strict_chord_grammar(self):
        from nodes.music.melody_template import CHORD_STRICT
        for ok in ("G#", "D#7", "A#m7", "Cmaj7", "Bbsus4", "F#m7b5", "C/E"):
            assert CHORD_STRICT.match(ok), ok
        for bad in ("B4", "E#4x", "Eb4", "Ab5", "xyz", ""):
            assert not CHORD_STRICT.match(bad), bad


class TestSeamRefine:
    @staticmethod
    def grid(seconds, period, shift=0.0, seed=0):
        rng = np.random.default_rng(seed)
        n = int(seconds * SR)
        w = np.zeros(n, np.float32)
        t = shift
        while t < seconds - 0.1:
            i = int(t * SR)
            w[i:i + 300] += rng.standard_normal(300).astype(np.float32) * np.exp(-np.arange(300) / 60.0)
            t += period * rng.choice([1, 1, 2])          # Achtel und Viertel gemischt
        return np.stack([w, w])

    def test_refine_finds_grid_offset(self):
        from nodes.music.audio_stitch import refine_cut
        period = 60 / 129 / 2
        # dieselbe Partiturstelle in zwei Renderings: gleicher Rhythmus, 50 ms versetzt
        prev = self.grid(20, period, 0.0, seed=1)
        new = self.grid(20, period, 0.05, seed=1)
        shift, conf = refine_cut(prev, new, SR, 10 * SR, 10 * SR, period)
        assert shift / SR == pytest.approx(0.05, abs=0.012)
        assert conf > 0.8

    def test_refine_leaves_cut_alone_without_clear_grid(self):
        from nodes.music.audio_stitch import refine_cut
        rng = np.random.default_rng(0)
        noise = rng.standard_normal((2, 20 * SR)).astype(np.float32) * 0.1
        shift, conf = refine_cut(noise, noise[:, ::-1].copy(), SR, 10 * SR, 10 * SR, 60 / 129 / 2)
        assert shift == 0 and conf < 0.6


class TestLoudness:
    def test_body_level_ignores_silence(self):
        from nodes.music.audio_stitch import body_level_db
        tone = np.sin(np.linspace(0, 2000 * np.pi, 10 * SR)).astype(np.float32) * 0.5
        with_gap = np.concatenate([tone, np.zeros(3 * SR, np.float32)])     # 3 s Stille / Ausklang
        assert body_level_db(np.stack([with_gap] * 2), SR) == pytest.approx(
            body_level_db(np.stack([tone] * 2), SR), abs=0.2)

    def test_reference_gain(self):
        from nodes.music.audio_stitch import reference_gain
        assert reference_gain(-15.0, -21.0) == pytest.approx(10 ** (6 / 20), rel=1e-3)
        assert reference_gain(-15.0, -40.0) == 2.0            # begrenzt
        assert reference_gain(-15.0, -15.0) == pytest.approx(1.0)

    def test_session_stores_reference_and_levels_parts(self, tmp_path, abc_text, score):
        d = str(tmp_path / "sess")
        base = render_from_score(score, offset=2.47, tail=10.0, seed=3)
        es.init_or_load(d, "t", abc_text, base, SR, "", "", "T", "test")
        part = apply_plan(build_plan(abc_text, MODES[2], 1, "auto", 0, "chorus", "", 1))
        render = render_from_score(parse_score(part["render_abc"]), offset=0.27, tail=6.0, seed=4) * 0.25
        rec = commit_part(d, part, render, SR)
        state = es.load_state(d)
        assert {"*", "verse", "chorus"} <= set(state["loudness_by_label"])
        assert rec["gain"] == 2.0          # 12 dB leiser -> Faktor 4, begrenzt auf 2.0
        assert rec["gain_note"]            # und das steht im Bericht


class TestOverlap:
    def test_auto_overlap_long_enough_after_short_section(self, abc_text):
        from nodes.music.song_extend import open_score
        # Verse 6 (8 Takte, 14.9 s) anhaengen -> danach ist der letzte Abschnitt zu kurz
        part = apply_plan(build_plan(abc_text, MODES[2], 1, "auto", 0, "6", "", 1))
        plan = build_plan(part["full_abc_after"], MODES[2], 1, "auto", 0, "last", "", 0)
        assert plan["overlap"] == 2
        nxt = apply_plan(plan)
        assert nxt["overlap_seconds"] >= 22.0

    def test_auto_overlap_single_long_section(self, abc_text):
        plan = build_plan(abc_text, MODES[2], 1, "auto", 0, "last", "", 0)
        assert plan["overlap"] == 1                     # Chorus mit 35 s reicht

    def test_manual_overlap_is_respected(self, abc_text):
        plan = build_plan(abc_text, MODES[2], 1, "auto", 0, "last", "", 2)
        assert plan["overlap"] == 2


class TestChooseCut:
    def test_avoids_spot_where_one_render_drops_out(self):
        from nodes.music.audio_stitch import choose_cut
        rng = np.random.default_rng(0)
        prev = (rng.standard_normal((2, 30 * SR)) * 0.2).astype(np.float32)
        new = (rng.standard_normal((2, 30 * SR)) * 0.2).astype(np.float32)
        new[:, 10 * SR:11 * SR] = 0.0                    # im neuen Rendering fehlt hier 1 s Musik
        candidates = [(10 * SR, 10 * SR), (20 * SR, 20 * SR)]
        pick, dist = choose_cut(prev, new, SR, candidates)
        assert pick == 1


class TestFinalBar:
    def test_final_cadence_bar_is_held_not_rested(self, score):
        from nodes.music.abc_score import smooth_final_bar, strip_trailing_rests
        secs, removed = strip_trailing_rests(score, score.sections)
        assert split_last(secs, "Ins").endswith("c'8z24|")
        smoothed, changed = smooth_final_bar(score, secs)
        assert split_last(smoothed, "Ins").endswith("c'32|")
        assert changed == 1                                   # Vocal ist nur Pause -> unveraendert
        assert validate_sections(score, smoothed) == []
        assert score.section_durations(smoothed) == score.section_durations(secs)


def split_last(sections, voice):
    return sections[-1].groups[-1][voice]
