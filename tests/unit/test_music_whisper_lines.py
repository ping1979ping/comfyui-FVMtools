"""Tests fuer die Zeilenbildung aus Whisper-Passagen (mit Zeitmarken)."""

from nodes.music.lyrics_prosody import analysiere
from nodes.music.whisper_lines import (
    MIN_ZEILEN_SEKTION,
    FVM_WhisperChunksToLyricLines,
    bericht,
    chunks_zu_zeilen,
    token_text,
)

# So sieht WHISPER_CHUNKS aus comfy_mtb aus: Passage plus [start, ende].
CHUNKS = [
    {"text": "It's not in the way that you hold me", "timestamp": [0.0, 2.4]},
    {"text": "It's not in the way you say you care", "timestamp": [2.6, 5.0]},
    {"text": "Love isn't always on time", "timestamp": [7.0, 9.2]},
    {"text": "Love isn't always on time", "timestamp": [9.4, 11.6]},
    {"text": "It's not in the words that you told me", "timestamp": [14.0, 16.5]},
    # Zweite Schlusszeile: Ein einzelner Vers waere sonst eine Ein-Zeilen-
    # Sektion und wuerde regelkonform an den Refrain gehaengt.
    {"text": "It's not in the way you say you're mine", "timestamp": [16.7, 19.0]},
]

CHUNKS_FETZEN = [
    {"text": "Hold the line", "timestamp": [0.0, 1.2]},
    {"text": "girl", "timestamp": [1.25, 1.4]},
    {"text": "Love isn't always on time", "timestamp": [3.0, 5.0]},
]

# Genau so kamen die Chunks im Lauf vom 16.09. an: BPE-Tokens mit Leerzeichen.
CHUNKS_BPE = [
    {"text": "ĠIt 's Ġnot Ġin Ġthe Ġway Ġthat Ġyou Ġhold Ġme", "timestamp": [0.0, 2.4]},
    {
        "text": "ĠIt 's Ġnot Ġin Ġthe Ġway Ġyou 've Ġbeen Ġtreat in '",
        "timestamp": [2.5, 5.0],
    },
    {"text": "ĠLove Ġisn 't Ġalways Ġon Ġtime", "timestamp": [7.0, 9.0]},
    {"text": "ĠLove Ġisn 't Ġalways Ġon Ġtime", "timestamp": [9.1, 11.0]},
]


class TestTokenText:
    def test_bpe_wird_zu_lesbarem_text(self):
        roh = "ĠIt 's Ġnot Ġin Ġthe Ġway"
        assert token_text(roh) == "It's not in the way"

    def test_wortteile_kleben_zusammen(self):
        assert token_text("Ġtreat in '") == "treatin'"
        assert token_text("ĠLove Ġisn 't") == "Love isn't"

    def test_normaler_text_bleibt(self):
        assert token_text("Love isn't always on time") == "Love isn't always on time"

    def test_leer_und_none(self):
        assert token_text("") == ""
        assert token_text(None) == ""


class TestChunksZuZeilen:
    def test_eine_zeile_pro_passage(self):
        text = chunks_zu_zeilen(CHUNKS, mit_sektionen=False)
        assert text.splitlines() == [c["text"] for c in CHUNKS]

    def test_bpe_chunks_ergeben_saubere_zeilen(self):
        zeilen = chunks_zu_zeilen(CHUNKS_BPE, mit_sektionen=False).splitlines()
        assert zeilen[0] == "It's not in the way that you hold me"
        assert zeilen[1].endswith("treatin'")
        assert "Ġ" not in "\n".join(zeilen)

    def test_kurze_pause_verschmilzt_nur_auf_wunsch(self):
        ohne = chunks_zu_zeilen(CHUNKS_FETZEN, mit_sektionen=False).splitlines()
        assert ohne[0] == "Hold the line girl"  # "girl" ist ein Fetzen
        mit = chunks_zu_zeilen(
            CHUNKS_FETZEN, pause_zeile=2.0, mit_sektionen=False
        ).splitlines()
        assert len(mit) == 1

    def test_dict_form_wird_akzeptiert(self):
        text = chunks_zu_zeilen({"chunks": CHUNKS, "text": "egal"}, mit_sektionen=False)
        assert len(text.splitlines()) == len(CHUNKS)

    def test_sektionen_aus_wiederholung(self):
        text = chunks_zu_zeilen(CHUNKS, pause_sektion=1.5)
        assert "[Verse]" in text and "[Chorus]" in text
        assert text.count("Love isn't always on time") == 2

    def test_keine_ein_zeilen_sektionen(self):
        """Der gemessene Fehler: [Chorus] mit einer einzigen Zeile, im Wechsel."""
        text = chunks_zu_zeilen(CHUNKS_BPE, pause_sektion=1.5)
        bloecke, aktuell = [], []
        for zeile in text.splitlines():
            if zeile.startswith("["):
                if aktuell:
                    bloecke.append(aktuell)
                aktuell = []
            elif zeile.strip():
                aktuell.append(zeile)
        if aktuell:
            bloecke.append(aktuell)
        assert bloecke, text
        assert all(len(b) >= MIN_ZEILEN_SEKTION for b in bloecke), text

    def test_leere_eingabe(self):
        assert chunks_zu_zeilen([]) == ""
        assert chunks_zu_zeilen({"chunks": []}) == ""

    def test_passagen_ohne_zeitmarken(self):
        ohne = [{"text": "Zeile eins hier"}, {"text": "Zeile zwei hier"}]
        assert chunks_zu_zeilen(ohne, mit_sektionen=False).splitlines() == [
            "Zeile eins hier",
            "Zeile zwei hier",
        ]


class TestNode:
    def test_zwei_ausgaben_und_info(self):
        text, info = FVM_WhisperChunksToLyricLines().execute(whisper_chunks=CHUNKS)
        assert "[Chorus]" in text
        assert "Whisper-Passagen: 6" in info and "erkannte Laenge: 19.0s" in info

    def test_ergebnis_ist_ohne_raten_messbar(self):
        """Der Zweck: Die Analyse muss dann nicht mehr segmentieren."""
        text, _ = FVM_WhisperChunksToLyricLines().execute(whisper_chunks=CHUNKS)
        daten = analysiere(text, "en", "wie eingegeben")
        assert daten["segmented"] is False
        assert daten["total_lines"] == 6
        chorus = [s for s in daten["sections"] if s["tag"] == "Chorus"]
        assert chorus and chorus[0]["rhyme_scheme"] == "AA"
        for sektion in daten["sections"]:
            for zeile in sektion["line_details"]:
                assert 4 <= zeile["syllables"] <= 12, zeile

    def test_bpe_kette_bis_zur_messung(self):
        text, info = FVM_WhisperChunksToLyricLines().execute(whisper_chunks=CHUNKS_BPE)
        daten = analysiere(text, "en", "wie eingegeben")
        # Ohne Token-Zusammensetzung waeren die Endwoerter Bruchstuecke wie "'"
        endwoerter = [
            z["end_word"] for s in daten["sections"] for z in s["line_details"]
        ]
        assert all(len(w) > 1 for w in endwoerter), endwoerter
        assert "time" in endwoerter
        assert bericht(CHUNKS_BPE, text).startswith("Whisper-Passagen: 4")
