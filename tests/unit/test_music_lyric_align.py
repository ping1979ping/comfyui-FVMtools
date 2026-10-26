"""Unit-Tests: Lyrics-Zeilen gegen Whisper-Zeilen und Partitur-Timing (FVM_LyricTimingAlign), ohne GPU."""

import copy
import json
import os

import numpy as np
import pytest

from nodes.music import lyric_align as la
from nodes.music import score_timing as st
from nodes.music.abc_score import onset_times, parse_score

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "music")
SR = 8000

# Vier Lyrics-Zeilen, je ein Ton pro Silbe: Verse 8 + 8, Chorus 7 + 7 Toene
ABC = """X:1
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

LYRICS = """[Verse]
Haus Baum Weg Licht Stern Mond Wind Meer
Brot Wein Salz Gold Glas Holz Stein Tisch

[Chorus]
Hand Fuss Kopf Herz Mund Ohr Arm
Tag Nacht Jahr Zeit Welt Feld Wald
"""

# Partitur-Einsatz jeder Zeile (s): Verse ab 4,0 (8 Achtel je Zeile), Chorus ab 12,0 (8. Ton bei 15,5)
EINSAETZE = [4.0, 8.0, 12.0, 15.5]
OFFSET = 0.9


def render(score, offset, seed=0, tail=3.0):
    rng = np.random.default_rng(seed)
    n = int((offset + score.total_seconds() + tail) * SR)
    wave = np.zeros(n, dtype=np.float32)
    for t, w in onset_times(score):
        i = int((offset + t + rng.normal(0, 0.006)) * SR)
        if 0 <= i < n - 400:
            wave[i:i + 400] += rng.standard_normal(400).astype(np.float32) * np.exp(-np.arange(400) / 80.0) * w * 0.4
    wave += rng.standard_normal(n).astype(np.float32) * 0.01
    return np.stack([wave, wave])


@pytest.fixture(scope="module")
def timing():
    data = st.build_timing(render(parse_score(ABC), OFFSET, seed=3), SR, ABC)
    assert data["offset_s"] == pytest.approx(OFFSET, abs=0.03)
    return data


def chunks_at(starts, texts, dauer=1.6):
    """Whisper-artige Passagen: Anfang WHISPER_VORLAUF vor dem Gesang."""
    return {"chunks": [{"text": t, "timestamp": [round(s - la.WHISPER_VORLAUF, 2), round(s + dauer, 2)]}
                       for s, t in zip(starts, texts)]}


def zeilen_texte():
    return [z["text"] for z in la.lyrics_zeilen(LYRICS)]


def gesungen(offset):
    return [offset + t for t in EINSAETZE]


# ---------------------------------------------------------------- Text

class TestText:
    def test_normalform(self):
        assert la.normalform("Die Straßen, werden WACH!") == "die strassen werden wach"
        assert la.normalform("Über   Brücken…") == "uber brucken"

    def test_aehnlichkeit(self):
        assert la.aehnlichkeit("was uns heut noch gehalten hat", "Was uns heute noch gehalten hat") > 0.95
        assert la.aehnlichkeit("Am Morgen geht die Sonne auf", "Ho-ho-ho-ho-ho") < la.MIN_AEHNLICHKEIT - 0.1
        assert la.aehnlichkeit("", "x") == 0.0

    def test_lyrics_zeilen_mit_marken(self):
        zeilen = la.lyrics_zeilen(LYRICS)
        assert [z["tag"] for z in zeilen] == ["Verse", "Verse", "Chorus", "Chorus"]
        assert [z["block"] for z in zeilen] == [0, 0, 1, 1]


# ---------------------------------------------------------------- Abgleich

class TestZeilenAbgleich:
    def test_eins_zu_eins(self):
        t = zeilen_texte()
        g = la.zeilen_abgleich(t, list(t))
        assert [(li, wi) for li, wi, _ in g] == [([0], [0]), ([1], [1]), ([2], [2]), ([3], [3])]

    def test_zwei_zeilen_in_einer_passage(self):
        t = zeilen_texte()
        g = la.zeilen_abgleich(t, [t[0] + ", " + t[1], t[2], t[3]])
        assert g[0][:2] == ([0, 1], [0])
        assert g[0][2] > 0.9

    def test_ganze_strophe_in_einer_passage(self):
        t = zeilen_texte()
        g = la.zeilen_abgleich(t, [" ".join(t)])
        assert g == [([0, 1, 2, 3], [0], pytest.approx(1.0))]

    def test_zeile_ueber_zwei_passagen(self):
        t = zeilen_texte()
        g = la.zeilen_abgleich(t, [t[0], "Brot Wein Salz Gold", "Glas Holz Stein Tisch", t[2], t[3]])
        assert g[1][:2] == ([1], [1, 2])

    def test_fehlend_und_zusaetzlich(self):
        t = zeilen_texte()
        g = la.zeilen_abgleich(t, ["Ho-ho-ho-ho ho ho", t[0], t[2], t[3]])
        assert ([], [0]) in [(li, wi) for li, wi, _ in g]          # Vokalise: zusaetzlich
        assert ([1], []) in [(li, wi) for li, wi, _ in g]          # Zeile 2 fehlt
        assert ([0], [1]) in [(li, wi) for li, wi, _ in g]

    def test_whisper_tippfehler_bleiben_zugeordnet(self):
        g = la.zeilen_abgleich(["ich lauf mit dir den Fluss hinauf"], ["Ich laufe mit dir den Fluß hinauf"])
        assert g[0][:2] == ([0], [0])

    def test_leer(self):
        assert la.zeilen_abgleich([], []) == []
        assert la.zeilen_abgleich(["a b c"], []) == [([0], [], 0.0)]


# ---------------------------------------------------------------- Partitur

class TestEinsaetze:
    def test_zeilen_einsaetze_aus_silben(self):
        zeilen = la.lyrics_zeilen(LYRICS)
        e = la.zeilen_einsaetze(parse_score(ABC), zeilen)
        assert [x[0] for x in e] == [1, 1, 2, 2]
        assert [x[1] for x in e] == pytest.approx(EINSAETZE)

    def test_unbekannte_marke_bleibt_offen(self):
        zeilen = la.lyrics_zeilen("[Bridge]\nEins zwei drei vier\n[Chorus]\nHand Fuss Kopf Herz Mund Ohr Arm")
        e = la.zeilen_einsaetze(parse_score(ABC), zeilen)
        assert e[0] is None
        assert e[1] == (2, pytest.approx(12.0))

    def test_gesangseinsaetze_ohne_haltebogen(self):
        abc = ABC.replace('"C"C4E4|', '"C"C4-C4|')
        n_vorher = len(la.gesangseinsaetze(parse_score(ABC)))
        assert len(la.gesangseinsaetze(parse_score(abc))) == n_vorher - 1


# ---------------------------------------------------------------- Ganzer Abgleich

class TestAlignLyrics:
    def test_passt_zur_besten_ausrichtung(self, timing):
        w = chunks_at(gesungen(OFFSET), zeilen_texte())
        d = la.align_lyrics(timing, LYRICS, w, ABC)
        assert d["alignment"]["choice"] == "best"
        assert d["alignment"]["candidates"][0]["residual_s"] == pytest.approx(0.0, abs=0.05)
        assert d["summary"]["matched"] == 4 and d["summary"]["missing"] == 0 and d["summary"]["extra"] == 0
        assert [z["section_label"] for z in d["lines"]] == ["verse", "verse", "chorus", "chorus"]
        assert all(z["section_match"] for z in d["lines"])
        assert [z["bar"] for z in d["lines"]] == [2, 4, 6, 7]
        assert d["lines"][3]["beat"] == 4                       # 15,5 s: vierter Schlag im 8. Takt
        assert all(z["confidence"] == pytest.approx(1.0) for z in d["lines"])
        assert d["warnings"] == []

    def test_waehlt_die_alternative(self, timing):
        t = copy.deepcopy(timing)
        beat = t["beat_s"]
        t["alternative"] = {"offset_s": t["offset_s"] - beat, "tempo_factor": t["tempo_factor"]}
        w = chunks_at(gesungen(t["offset_s"] - beat), zeilen_texte())
        d = la.align_lyrics(t, LYRICS, w, ABC)
        assert d["alignment"]["choice"] == "alternative"
        assert d["alignment"]["closest_candidate"] == "alternative"
        assert d["offset_s"] == pytest.approx(t["offset_s"] - beat, abs=1e-3)
        assert d["sections"][1]["start_s"] == pytest.approx(t["offset_s"] - beat + 4.0, abs=0.01)
        assert any("Zweitbeste" in x for x in d["warnings"])

    def test_bleibt_bei_best_wenn_die_besser_passt(self, timing):
        t = copy.deepcopy(timing)
        t["alternative"] = {"offset_s": t["offset_s"] + t["beat_s"], "tempo_factor": t["tempo_factor"]}
        d = la.align_lyrics(t, LYRICS, chunks_at(gesungen(t["offset_s"]), zeilen_texte()), ABC)
        assert d["alignment"]["choice"] == "best"

    def test_gesang_weit_daneben_nimmt_whisper_versatz(self, timing):
        verschoben = OFFSET - 2.0                                # YuE2 hat das Intro gekuerzt
        d = la.align_lyrics(timing, LYRICS, chunks_at(gesungen(verschoben), zeilen_texte()), ABC)
        assert d["alignment"]["choice"] == "whisper"
        assert d["offset_s"] == pytest.approx(timing["offset_s"] - 2.0, abs=0.05)
        assert [z["section_label"] for z in d["lines"]] == ["verse", "verse", "chorus", "chorus"]
        assert any("neben der Partitur" in x for x in d["warnings"])

    def test_passage_mit_zwei_zeilen_wird_nach_partitur_geteilt(self, timing):
        t = zeilen_texte()
        s = gesungen(OFFSET)
        w = {"chunks": [{"text": t[0] + " " + t[1], "timestamp": [s[0] - 0.3, s[1] + 2.0]},
                        {"text": t[2], "timestamp": [s[2] - 0.3, s[2] + 2.0]},
                        {"text": t[3], "timestamp": [s[3] - 0.3, s[3] + 2.0]}]}
        d = la.align_lyrics(timing, LYRICS, w, ABC)
        l0, l1 = d["lines"][:2]
        assert l0["timing_source"] == l1["timing_source"] == "whisper_split_score"
        assert l0["shared_passage"] and l1["shared_passage"]
        assert l1["start_s"] - l0["start_s"] == pytest.approx(4.0, abs=0.02)
        assert l0["end_s"] == l1["start_s"]

    def test_ohne_partitur_nach_textlaenge(self, timing):
        t = zeilen_texte()
        w = {"chunks": [{"text": t[0] + " " + t[1], "timestamp": [5.0, 9.0]},
                        {"text": t[2], "timestamp": [12.6, 14.0]}, {"text": t[3], "timestamp": [15.1, 17.0]}]}
        d = la.align_lyrics(timing, LYRICS, w, "")
        assert d["lines"][0]["timing_source"] == "whisper_split_text"
        assert 6.5 < d["lines"][1]["start_s"] < 7.5
        assert d["alignment"]["choice"] == "best" and d["alignment"]["anchors"] == 0
        assert any("Ohne Partitur" in x for x in d["warnings"])

    def test_fehlende_zeile_wird_geschaetzt_und_vokalise_ist_extra(self, timing):
        t = zeilen_texte()
        s = gesungen(OFFSET)
        w = {"chunks": [{"text": "Ho-ho-ho-ho-ho-ho", "timestamp": [1.2, 3.0]},
                        {"text": t[0], "timestamp": [s[0] - 0.3, s[0] + 2.0]},
                        {"text": t[2], "timestamp": [s[2] - 0.3, s[2] + 2.0]},
                        {"text": t[3], "timestamp": [s[3] - 0.3, s[3] + 2.0]}]}
        d = la.align_lyrics(timing, LYRICS, w, ABC)
        fehlt = d["lines"][1]
        assert fehlt["status"] == "missing" and fehlt["confidence"] == 0.0
        assert fehlt["timing_source"] == "estimated_score"
        assert fehlt["start_s"] == pytest.approx(s[1] - la.WHISPER_VORLAUF, abs=0.05)
        assert d["extra"][0]["section_label"] == "intro"
        assert d["summary"]["missing"] == 1 and d["summary"]["extra"] == 1
        assert any("nicht erkannt: 2" in x for x in d["warnings"])

    def test_anschliessende_passage_bekommt_partitur_anfang(self, timing):
        # Whisper laesst die Passage direkt an die vorige anschliessen - Anfang 2 s zu frueh
        t = zeilen_texte()
        s = gesungen(OFFSET)
        w = {"chunks": [{"text": t[0], "timestamp": [s[0] - 0.3, s[0] + 2.0]},
                        {"text": t[1], "timestamp": [s[1] - 0.3, s[1] + 2.0]},
                        {"text": t[2], "timestamp": [s[2] - 0.3, s[2] + 2.0]},
                        {"text": t[3], "timestamp": [s[2] + 2.0, s[3] + 2.0]}]}
        d = la.align_lyrics(timing, LYRICS, w, ABC)
        z = d["lines"][3]
        assert z["timing_source"] == "whisper+score_start"
        assert z["start_s"] == pytest.approx(s[3] - la.WHISPER_VORLAUF, abs=0.05)

    def test_json_serialisierbar_und_bericht(self, timing):
        d = la.align_lyrics(timing, LYRICS, chunks_at(gesungen(OFFSET), zeilen_texte()), ABC)
        again = json.loads(json.dumps(d))
        assert again["schema"] == la.SCHEMA
        assert "Ausrichtung: best" in la.report_text(d)

    def test_ohne_lyrics_fehler(self, timing):
        with pytest.raises(ValueError):
            la.align_lyrics(timing, "  \n", {"chunks": []}, ABC)


# ---------------------------------------------------------------- Echte Laeufe (YuE2, Whisper-small)

LYRICS_V1 = ("[Verse]\nAm Morgen geht die Sonne auf\ndie Straßen werden wach\nich lauf mit dir den Fluss hinauf\n"
             "und lache hundertfach\n\n[Chorus]\nWir sind das Licht in dieser Stadt\nwir singen laut und frei\n"
             "was uns heut noch gehalten hat\nist morgen schon vorbei")

# Wortgenaue Whisper-Marken (Cross-Attention) fuer den Chorus-Anfang jeder Zeile, run01/run03
WORTMARKEN_CHORUS = [27.48, 30.24, 32.22, 34.76]


def _nur_einsaetze(timing):
    """Timing wie vor der Gesangs-Einmessung: Einsatz-Spitze als best, Nebenspitze als alternative."""
    t = copy.deepcopy(timing)
    t["offset_s"] = t["fit"]["onset_offset_s"]
    ru = t["fit"]["runner_up"]
    t["alternative"] = {"offset_s": ru["offset_s"], "tempo_factor": ru["tempo_factor"]}
    return t


def _lauf(run, nur_einsaetze=False):
    abc_run = "run02" if run == "run02" else "run01"     # run03 = gleicher Seed, gleiche Partitur wie run01
    with open(os.path.join(FIXTURES, f"yue2_{run}_timing.json"), encoding="utf-8") as h:
        timing = json.load(h)
    if nur_einsaetze:
        timing = _nur_einsaetze(timing)
    with open(os.path.join(FIXTURES, f"yue2_{run}_whisper.json"), encoding="utf-8") as h:
        whisper = json.load(h)
    with open(os.path.join(FIXTURES, f"yue2_{abc_run}.abc"), encoding="utf-8") as h:
        abc = h.read()
    return la.align_lyrics(timing, LYRICS_V1, whisper, abc)


class TestEchteLaeufe:
    @pytest.mark.parametrize("run", ["run01", "run02", "run03"])
    def test_alle_zeilen_erkannt_vokalise_extra(self, run):
        d = _lauf(run)
        assert d["summary"]["matched"] == 8 and d["summary"]["missing"] == 0
        assert d["summary"]["min_confidence"] >= 0.95
        assert [e["section_label"] for e in d["extra"]] == ["intro"]          # "Ho-ho"/"Ah ah" im Intro
        starts = [z["start_s"] for z in d["lines"]]
        assert starts == sorted(starts)
        assert [z["section_label"] for z in d["lines"]] == ["verse"] * 4 + ["chorus"] * 4
        assert all(z["section_match"] for z in d["lines"])

    @pytest.mark.parametrize("run", ["run01", "run02", "run03"])
    def test_gesangs_einmessung_passt_zu_whisper(self, run):
        # Timing mit Gesangs-Einmessung (gekuerztes Intro): "best" liegt innerhalb der Toleranz
        d = _lauf(run)
        w = d["alignment"]
        assert w["choice"] == "best"
        assert abs(w["candidates"][0]["residual_s"]) <= w["tolerance_s"]
        assert d["offset_s"] < 0                                             # Intro im Audio gekuerzt
        assert d["warnings"] == []

    @pytest.mark.parametrize("run,naeher", [("run01", "best"), ("run02", "alternative"), ("run03", "best")])
    def test_reine_einsatz_ausrichtung_liegt_daneben(self, run, naeher):
        # Vorher (nur Einsaetze, Versatz >= 0): der Gesang lag 1,6-2,5 s vor beiden Kandidaten
        d = _lauf(run, nur_einsaetze=True)
        w = d["alignment"]
        assert w["choice"] == "whisper"
        assert w["closest_candidate"] == naeher
        assert all(k["residual_s"] < -1.0 for k in w["candidates"])

    def test_run02_alternative_naeher_um_einen_schlag(self):
        w = _lauf("run02", nur_einsaetze=True)["alignment"]
        best, alt = w["candidates"]
        assert alt["name"] == "alternative"
        assert best["residual_s"] - alt["residual_s"] == pytest.approx(-0.56, abs=0.02)

    @pytest.mark.parametrize("run", ["run01", "run03"])
    def test_chorus_zeilen_treffen_die_wortmarken(self, run):
        # Whisper fasst den Chorus in zwei Passagen, die direkt anschliessen (Anfang 1,8 s zu frueh);
        # Teilung und Anfang ueber die Partitur liegen auf 0,25 s an den Wortmarken
        d = _lauf(run)
        for z, wort in zip(d["lines"][4:], WORTMARKEN_CHORUS):
            assert z["start_s"] + la.WHISPER_VORLAUF == pytest.approx(wort, abs=0.25), z


# ---------------------------------------------------------------- Node

class TestNode:
    def test_schema(self):
        req = la.FVM_LyricTimingAlign.INPUT_TYPES()
        assert req["required"]["timing_json"][0] == "STRING"
        assert req["optional"]["whisper_chunks"][0] == "WHISPER_CHUNKS"
        assert "abc" in req["optional"] and "whisper_json" in req["optional"]
        assert la.FVM_LyricTimingAlign.OUTPUT_NODE is True
        assert len(la.FVM_LyricTimingAlign.RETURN_TYPES) == len(la.FVM_LyricTimingAlign.RETURN_NAMES)

    def test_align_mit_json_und_datei(self, timing, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "_output_dir", lambda: str(tmp_path))
        w = chunks_at(gesungen(OFFSET), zeilen_texte())
        out = la.FVM_LyricTimingAlign().align(json.dumps(timing), LYRICS, "studiosuite/test/x/lyrics",
                                              whisper_json=json.dumps(w), abc=ABC)
        text, report = out["result"]
        data = json.loads(text)
        assert data["summary"]["matched"] == 4
        assert out["ui"]["files"] == [{"filename": "lyrics_00001.json", "subfolder": "studiosuite/test/x",
                                       "type": "output"}]
        assert "Gespeichert" in report

    def test_whisper_chunks_vor_json(self, timing, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "_output_dir", lambda: str(tmp_path))
        w = chunks_at(gesungen(OFFSET), zeilen_texte())
        out = la.FVM_LyricTimingAlign().align(json.dumps(timing), LYRICS, "x", save_file=False,
                                              whisper_chunks=w["chunks"], whisper_json="kaputt", abc=ABC)
        assert json.loads(out["result"][0])["summary"]["matched"] == 4
        assert list(tmp_path.iterdir()) == []

    def test_ohne_whisper_fehler(self, timing):
        with pytest.raises(ValueError):
            la.FVM_LyricTimingAlign().align(json.dumps(timing), LYRICS, "x", save_file=False)
