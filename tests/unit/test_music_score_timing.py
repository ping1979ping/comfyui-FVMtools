"""Unit-Tests: Partitur-Timing als JSON (FVM_ScoreTimingExport), synthetisches Audio, ohne GPU."""

import json
import os

import numpy as np
import pytest
import torch

from nodes.music import score_timing as st
from nodes.music.abc_score import bar_starts, onset_times, parse_score

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "music", "sloop_cover.abc")
SR = 8000

KURZ = """X:1
T:
M:4/4
L:1/8
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% intro
V: Vocal
z8|z8|
V: Ins
C2E2G2c2|G2E2C2E2|
% verse
V: Vocal
"C"C2D2E2F2|"G"G2A2B2c2|"F"A2G2F2E2|"C"D2C2D2E2|
V: Ins
C2C2E2E2|G2G2B2B2|A2A2F2F2|D2D2C2C2|
% chorus
V: Vocal
"C"c2B2A2G2|"F"F2G2A2B2|"G"c2A2F2D2|"C"C4E4|
V: Ins
c2c2A2A2|F2F2A2A2|c2c2F2F2|C4E4|
"""


def render_from_score(score, offset, scale=1.0, tail=4.0, seed=0, sr=SR, cut_at=None):
    """Synthetisches 'Rendering': Rauschbursts an den Partitur-Einsaetzen (wie YuE2, nur messbar)."""
    rng = np.random.default_rng(seed)
    n = int((offset + scale * score.total_seconds() + tail) * sr)
    if cut_at is not None:
        n = int(cut_at * sr)
    wave = np.zeros(n, dtype=np.float32)
    for t, w in onset_times(score):
        i = int((offset + scale * t + rng.normal(0, 0.006)) * sr)
        if 0 <= i < n - 400:
            wave[i:i + 400] += rng.standard_normal(400).astype(np.float32) * np.exp(-np.arange(400) / 80.0) * w * 0.4
    wave += rng.standard_normal(n).astype(np.float32) * 0.01
    return np.stack([wave, wave])


@pytest.fixture
def sloop():
    with open(FIXTURE, encoding="utf-8") as handle:
        return handle.read()


class TestBuildTiming:
    @pytest.mark.parametrize("offset", [0.4, 1.7, 2.4])
    def test_offset_and_tempo_recovered(self, sloop, offset):
        score = parse_score(sloop)
        wave = render_from_score(score, offset, seed=int(offset * 10))
        data = st.build_timing(wave, SR, sloop)
        assert data["offset_s"] == pytest.approx(offset, abs=0.03)
        assert data["tempo_factor"] == pytest.approx(1.0, abs=0.0025)
        assert data["tempo_bpm"] == pytest.approx(129.0)
        assert data["meter"] == "4/4"
        assert data["warnings"] == []

    def test_sections_contiguous_and_cover_music(self, sloop):
        score = parse_score(sloop)
        wave = render_from_score(score, 1.2)
        data = st.build_timing(wave, SR, sloop)
        secs = data["sections"]
        assert [s["label"] for s in secs] == [s.label for s in score.sections]
        assert secs[0]["start_s"] == pytest.approx(data["offset_s"], abs=1e-3)
        for a, b in zip(secs, secs[1:]):
            assert b["start_s"] == pytest.approx(a["end_s"], abs=2e-3)
        assert secs[-1]["end_s"] == pytest.approx(data["music_end_s"], abs=2e-3)
        assert data["music_end_s"] <= data["audio_duration_s"]
        assert all(s["in_audio"] and not s["truncated"] for s in secs)
        assert sum(s["bars"] for s in secs) == len(data["bars"])

    def test_bars_follow_score_grid(self):
        score = parse_score(KURZ)
        wave = render_from_score(score, 0.9, seed=3)
        data = st.build_timing(wave, SR, KURZ)
        bars = data["bars"]
        assert len(bars) == 10                         # 2 intro + 4 verse + 4 chorus
        assert data["bar_s"] == pytest.approx(2.0, abs=0.01)   # 4/4 bei 120 BPM
        starts = bar_starts(score, score.sections)[:-1]
        for bar, s in zip(bars, starts):
            assert bar["start_s"] == pytest.approx(data["offset_s"] + data["tempo_factor"] * s, abs=2e-3)
        assert [b["section"] for b in bars] == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        assert [s["first_bar"] for s in data["sections"]] == [0, 2, 6]

    def test_beats_four_per_bar(self):
        score = parse_score(KURZ)
        wave = render_from_score(score, 0.9, seed=3)
        data = st.build_timing(wave, SR, KURZ)
        beats = data["beats"]
        assert len(beats) == 40
        assert [b["beat"] for b in beats[:5]] == [1, 2, 3, 4, 1]
        assert beats[1]["t"] - beats[0]["t"] == pytest.approx(0.5, abs=0.01)
        assert beats[0]["t"] == data["bars"][0]["start_s"]

    def test_without_beats(self):
        score = parse_score(KURZ)
        data = st.build_timing(render_from_score(score, 0.9), SR, KURZ, include_beats=False)
        assert data["beats"] is None

    def test_truncated_audio_is_marked(self):
        # max_duration hat das Rendering mitten im Chorus gekappt
        score = parse_score(KURZ)
        wave = render_from_score(score, 0.9, seed=5, cut_at=0.9 + 15.0)
        data = st.build_timing(wave, SR, KURZ)
        chorus = data["sections"][2]
        assert chorus["in_audio"] and chorus["truncated"]
        assert chorus["end_s"] == pytest.approx(data["audio_duration_s"], abs=1e-3)
        assert not data["bars"][-1]["in_audio"]
        assert all(b["t"] < data["audio_duration_s"] for b in data["beats"])
        assert any("Audio endet vor dem letzten Ton" in w for w in data["warnings"])
        assert data["complete"] is False

    def test_missing_trailing_rest_bar_is_no_warning(self):
        # YuE2 endet am letzten Ton; der Pausentakt am Schluss fehlt im Audio - Song ist vollstaendig
        abc = KURZ.replace('"C"C4E4|\nV: Ins\nc2c2A2A2|F2F2A2A2|c2c2F2F2|C4E4|',
                           '"C"C4E4|"C"z8|\nV: Ins\nc2c2A2A2|F2F2A2A2|c2c2F2F2|C4E4|z8|')
        score = parse_score(abc)
        assert len(bar_starts(score, score.sections)) == 12
        wave = render_from_score(score, 0.9, seed=4, cut_at=0.9 + 20.0 + 0.1)
        data = st.build_timing(wave, SR, abc)
        assert data["sections"][-1]["truncated"]
        assert data["material_end_s"] == pytest.approx(0.9 + 20.0, abs=0.05)
        assert data["complete"] is True
        assert not any("Audio endet" in w for w in data["warnings"])

    def test_unusual_offset_warns(self):
        score = parse_score(KURZ)
        data = st.build_timing(render_from_score(score, 6.0, seed=2), SR, KURZ)
        assert data["offset_s"] == pytest.approx(6.0, abs=0.03)
        assert any("Versatz" in w for w in data["warnings"])

    def test_empty_score_raises(self):
        with pytest.raises(ValueError):
            st.build_timing(np.zeros((2, SR)), SR, "X:1\nM:4/4\nL:1/8\nQ:1/4=120\nV: Vocal\nK:C\n")

    def test_json_serialisable(self, sloop):
        score = parse_score(sloop)
        data = st.build_timing(render_from_score(score, 1.0), SR, sloop)
        again = json.loads(json.dumps(data))
        assert again["schema"] == st.SCHEMA
        assert "Tempo 129" in st.report_text(data)


# Gleichmaessige Viertel ohne Akkorde: jede Verschiebung um einen Schlag passt fast genauso gut
GLEICH = """X:1
T:
M:4/4
L:1/4
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
K:C
% verse
V: Vocal
C D E F|G A B c|C D E F|G A B c|C D E F|G A B c|
"""

# echte Laeufe (StudioSuite V1) im ComfyUI-Ausgabeordner neben dem Pack
RUNS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "output",
                                    "studiosuite", "test", "music_song_yue2"))


def _echter_lauf(run):
    import soundfile as sf
    base = os.path.join(RUNS, run)
    wave, sr = sf.read(os.path.join(base, "song_00001.flac"), always_2d=True, dtype="float32")
    with open(os.path.join(base, "score_00001.txt"), encoding="utf-8") as handle:
        return wave.T, sr, handle.read()


class TestRunnerUp:
    def test_runner_up_always_reported(self, sloop):
        score = parse_score(sloop)
        data = st.build_timing(render_from_score(score, 1.3, seed=1), SR, sloop)
        ru = data["fit"]["runner_up"]
        assert ru is not None
        assert abs(ru["shift_beats"]) >= st.RUNNER_UP_MIN_BEATS - 0.02
        assert 0.0 < ru["ratio"] < st.RUNNER_UP_CLOSE
        assert ru["shift_s"] == pytest.approx(ru["shift_beats"] * data["beat_s"], abs=0.01)
        # sichere Ausrichtung: keine Alternative, alte Felder wie bisher
        assert data["alignment_unsure"] is False
        assert data["alternative"] is None
        assert data["offset_s"] == pytest.approx(1.3, abs=0.03)
        assert set(data["fit"]) >= {"corr", "contrast", "search_seconds"}
        assert "Zweitbeste Spitze" in st.report_text(data)

    def test_periodic_score_is_unsure_with_alternative(self):
        score = parse_score(GLEICH)
        data = st.build_timing(render_from_score(score, 1.5, seed=2), SR, GLEICH)
        assert data["alignment_unsure"] is True
        alt = data["alternative"]
        assert alt is not None
        assert abs(abs(alt["shift_beats"]) - 1.0) < 0.1          # Nachbarschlag
        assert alt["offset_s"] == pytest.approx(data["offset_s"] + alt["shift_s"], abs=0.02)
        assert [s["label"] for s in alt["sections"]] == [s["label"] for s in data["sections"]]
        assert alt["sections"][0]["start_s"] == pytest.approx(alt["offset_s"], abs=1e-3)
        assert any("Zweitbeste Ausrichtung" in w for w in data["warnings"])
        assert any(w.startswith("Ausrichtung unsicher") for w in data["warnings"])

    def test_runner_up_function_respects_min_shift(self):
        from nodes.music.audio_stitch import fit_score_timing, runner_up_timing
        score = parse_score(KURZ)
        wave = render_from_score(score, 0.8, seed=6)
        best = fit_score_timing(wave, SR, onset_times(score), score.total_seconds())
        ru = runner_up_timing(wave, SR, onset_times(score), score.total_seconds(), best, 0.4)
        mid = score.total_seconds() / 2
        assert abs((ru.offset + ru.scale * mid) - (best.offset + best.scale * mid)) >= 0.4 - 1e-6
        assert ru.corr <= best.corr
        assert 0.0 < ru.contrast <= 1.0

    @pytest.mark.skipif(not os.path.isdir(RUNS), reason="echte YuE2-Laeufe nicht vorhanden")
    def test_real_run02_minus_one_beat(self):
        wave, sr, abc = _echter_lauf("run02")
        data = st.build_timing(wave, sr, abc)
        assert data["fit"]["onset_offset_s"] == pytest.approx(0.6, abs=0.02)   # Einsatz-Spitze wie bisher
        ru = data["fit"]["runner_up"]
        assert ru["offset_s"] == pytest.approx(0.04, abs=0.03)
        assert ru["shift_beats"] == pytest.approx(-1.0, abs=0.05)
        assert ru["ratio"] == pytest.approx(0.89, abs=0.02)

    @pytest.mark.skipif(not os.path.isdir(RUNS), reason="echte YuE2-Laeufe nicht vorhanden")
    def test_real_run01_close_runner_up(self):
        wave, sr, abc = _echter_lauf("run01")
        data = st.build_timing(wave, sr, abc)
        assert data["fit"]["onset_offset_s"] == pytest.approx(1.12, abs=0.02)
        assert data["fit"]["contrast"] >= st.WEAK_CONTRAST                 # alte Eindeutigkeit: sicher
        assert data["fit"]["runner_up"]["ratio"] >= st.RUNNER_UP_CLOSE     # aber Nachbarschlag fast gleich
        assert data["fit"]["runner_up"]["shift_beats"] == pytest.approx(1.0, abs=0.05)

    # Whisper-Versatz (Zeilenabgleich, siehe test_music_lyric_align): run01 -0,74 s, run02 -1,55 s
    @pytest.mark.skipif(not os.path.isdir(RUNS), reason="echte YuE2-Laeufe nicht vorhanden")
    @pytest.mark.parametrize("run,whisper", [("run01", -0.74), ("run02", -1.55), ("run03", -0.77)])
    def test_real_runs_vocal_alignment_matches_whisper(self, run, whisper):
        wave, sr, abc = _echter_lauf(run)
        data = st.build_timing(wave, sr, abc)
        assert data["alignment_method"] == "vocal"
        assert data["offset_s"] == pytest.approx(whisper, abs=max(0.5 * data["beat_s"], 0.35))
        assert data["intro_trimmed_s"] == pytest.approx(-data["offset_s"] / data["tempo_factor"], abs=0.01)
        assert data["alignment_unsure"] is False and data["alternative"] is None
        intro = data["sections"][0]
        assert intro["start_s"] == 0.0 and intro["trimmed_s"] == pytest.approx(-data["offset_s"], abs=0.01)
        assert all(b["t"] >= 0 for b in data["beats"])
        assert any("Gesangs-Einmessung" in w for w in data["warnings"])
        assert not any("Vorlaufs" in w for w in data["warnings"])


def render_with_voice(score, offset, seed=0, sr=SR, tail=3.0):
    """Wie render_from_score, dazu ein 1,5-kHz-Ton fuer jeden Gesangston (die "Stimme").

    ``offset`` darf negativ sein: dann fehlt der Anfang der Partitur (gekuerztes Intro).
    """
    rng = np.random.default_rng(seed)
    n = int((offset + score.total_seconds() + tail) * sr)
    wave = np.zeros(n, dtype=np.float32)
    for t, w in onset_times(score):
        i = int((offset + t + rng.normal(0, 0.006)) * sr)
        if 0 <= i < n - 400:
            wave[i:i + 400] += rng.standard_normal(400).astype(np.float32) * np.exp(-np.arange(400) / 80.0) * w * 0.4
    for a, b in st.vocal_spans(score):
        i0, i1 = max(0, int((offset + a) * sr)), min(n, int((offset + b - 0.03) * sr))
        if i1 > i0:
            wave[i0:i1] += 0.2 * np.sin(2 * np.pi * 1500.0 * np.arange(i1 - i0) / sr).astype(np.float32)
    wave += rng.standard_normal(n).astype(np.float32) * 0.01
    return np.stack([wave, wave])


class TestVocalAlignment:
    def test_vocal_spans_vocal_voice_only(self):
        score = parse_score(KURZ)
        spans = st.vocal_spans(score)
        assert len(spans) == 16 + 14                      # Verse 16 Achtel, Chorus 14 Toene
        assert spans[0] == pytest.approx((4.0, 4.5))      # nach zwei Pausentakten Intro
        assert st.vocal_voice(score) == "Vocal"

    def test_fit_vocal_offset_negative(self):
        score = parse_score(KURZ)
        wave = render_with_voice(score, -1.5, seed=4)
        v = st.fit_vocal_offset(wave, SR, score)
        assert v["offset"] == pytest.approx(-1.5, abs=0.1)
        assert v["ratio"] < st.VOCAL_MAX_RATIO

    def test_trimmed_intro_uses_vocal(self):
        score = parse_score(KURZ)
        data = st.build_timing(render_with_voice(score, -1.5, seed=4), SR, KURZ)
        assert data["alignment_method"] == "vocal"
        assert data["offset_s"] == pytest.approx(-1.5, abs=0.1)
        assert data["intro_trimmed_s"] == pytest.approx(1.5, abs=0.1)
        assert data["fit"]["vocal"]["used"] is True
        intro, verse = data["sections"][:2]
        assert intro["start_s"] == 0.0 and intro["trimmed_s"] == pytest.approx(1.5, abs=0.1)
        assert verse["start_s"] == pytest.approx(4.0 - 1.5, abs=0.1)
        assert data["bars"][0]["in_audio"]                # erster Intro-Takt (0-2 s) ragt noch hinein
        assert all(b["t"] >= 0 for b in data["beats"])
        assert "Gesangs-Einmessung" in st.report_text(data)

    def test_normal_render_keeps_onsets(self):
        score = parse_score(KURZ)
        data = st.build_timing(render_with_voice(score, 1.1, seed=5), SR, KURZ)
        assert data["alignment_method"] == "onsets"
        assert data["offset_s"] == pytest.approx(1.1, abs=0.03)
        assert data["fit"]["vocal"]["offset_s"] == pytest.approx(1.1, abs=0.1)   # stimmt ueberein
        assert data["fit"]["vocal"]["used"] is False
        assert data["intro_trimmed_s"] == 0.0

    def test_no_vocal_notes_no_vocal_fit(self):
        abc = KURZ.replace('"C"C2D2E2F2|"G"G2A2B2c2|"F"A2G2F2E2|"C"D2C2D2E2|', "z8|z8|z8|z8|").replace(
            '"C"c2B2A2G2|"F"F2G2A2B2|"G"c2A2F2D2|"C"C4E4|', "z8|z8|z8|z8|")
        score = parse_score(abc)
        assert st.fit_vocal_offset(render_from_score(score, 1.0), SR, score) is None
        data = st.build_timing(render_from_score(score, 1.0), SR, abc)
        assert data["fit"]["vocal"] is None and data["alignment_method"] == "onsets"


class TestSaveJson:
    def test_counter_and_subfolder(self, tmp_path):
        p1, n1, sub = st.save_json("{}", "studiosuite/test/run1/timing", str(tmp_path))
        p2, n2, _ = st.save_json("{}", "studiosuite/test/run1/timing", str(tmp_path))
        assert (n1, n2) == ("timing_00001.json", "timing_00002.json")
        assert sub == "studiosuite/test/run1"
        assert os.path.exists(p1) and os.path.exists(p2)

    def test_prefix_without_folder(self, tmp_path):
        _, name, sub = st.save_json("{}", "song", str(tmp_path))
        assert (name, sub) == ("song_00001.json", "")

    def test_escape_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            st.save_json("{}", "../../evil", str(tmp_path / "out"))


class TestNode:
    def test_schema(self):
        inputs = st.FVM_ScoreTimingExport.INPUT_TYPES()["required"]
        assert inputs["audio"][0] == "AUDIO"
        assert inputs["abc"][0] == "STRING"
        assert "filename_prefix" in inputs
        assert st.FVM_ScoreTimingExport.OUTPUT_NODE is True
        assert len(st.FVM_ScoreTimingExport.RETURN_TYPES) == len(st.FVM_ScoreTimingExport.RETURN_NAMES)

    def test_export_writes_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(st, "_output_dir", lambda: str(tmp_path))
        score = parse_score(KURZ)
        wave = render_from_score(score, 1.1, seed=7)
        audio = {"waveform": torch.from_numpy(wave[:1].copy())[None], "sample_rate": SR}   # mono [1,1,N]
        out = st.FVM_ScoreTimingExport().export(audio, KURZ, "studiosuite/test/x/timing")
        text, report = out["result"]
        data = json.loads(text)
        assert data["offset_s"] == pytest.approx(1.1, abs=0.03)
        files = out["ui"]["files"]
        assert files == [{"filename": "timing_00001.json", "subfolder": "studiosuite/test/x", "type": "output"}]
        saved = tmp_path / "studiosuite" / "test" / "x" / "timing_00001.json"
        assert json.loads(saved.read_text(encoding="utf-8")) == data
        assert "Gespeichert" in report

    def test_export_without_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(st, "_output_dir", lambda: str(tmp_path))
        score = parse_score(KURZ)
        audio = {"waveform": torch.from_numpy(render_from_score(score, 1.0))[None], "sample_rate": SR}
        out = st.FVM_ScoreTimingExport().export(audio, KURZ, "x", save_file=False)
        assert "files" not in out["ui"]
        assert list(tmp_path.iterdir()) == []
