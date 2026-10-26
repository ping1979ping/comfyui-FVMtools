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
