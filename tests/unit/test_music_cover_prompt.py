"""Unit-Tests fuer die Cover-Text-Nodes (Prosodie messen, Prompt bauen)."""

import json

from nodes.music.cover_prompt import (
    FVM_CoverLyricsPromptBuilder,
    baue_bloecke,
    system_zusatz,
)
from nodes.music.lyrics_prosody import (
    FVM_LyricsProsodyAnalyze,
    _ist_flowtext,
    _sektionen_aus_zeilen,
    _woerterbuch,
    analysiere,
    kurzfassung,
    reimschluessel,
    segmentiere_flowtext,
    zeilensilben,
)

DE_TEXT = """[Verse]
Ich halte die Linie fuer dich
Der Regen faellt schwer auf das Dach
Ich warte am Bahnhof auf dich
Und bleibe die ganze Nacht wach

[Chorus]
Komm mit mir ins Licht
Komm mit mir nach Haus
"""

EN_TEXT = """[Verse]
Hold the line tonight
Love is hard to find
Waiting in the light
Leaving you behind
"""

# So kommt ein Transkript aus HeartMuLa: ein Absatz, keine Umbrueche, Refrain
# mehrfach enthalten.
TRANSKRIPT = (
    "It's not in the way that you hold me It's not in the way you say you care "
    "Love isn't always on time Whoa whoa whoa It's not in the words that you told me "
    "It's not in the way that you came back to me Love isn't always on time "
    "Whoa whoa whoa You can try but you can't"
)


class TestReimschluessel:
    def test_deutsche_reime(self):
        assert reimschluessel("Nacht") == reimschluessel("gebracht")
        assert reimschluessel("Dach") == reimschluessel("wach")

    def test_stummes_e_englisch(self):
        assert reimschluessel("time") == reimschluessel("rhyme")
        assert reimschluessel("love") == reimschluessel("above")

    def test_ungleiche_endungen_reimen_nicht(self):
        assert reimschluessel("line") != reimschluessel("time")

    def test_leeres_wort(self):
        assert reimschluessel("") == ""
        assert reimschluessel("...") == ""


class TestAnalyse:
    def test_sektionen_und_zeilen(self):
        daten = analysiere(DE_TEXT, "de")
        assert [s["tag"] for s in daten["sections"]] == ["Verse", "Chorus"]
        assert daten["sections"][0]["lines"] == 4
        assert daten["total_lines"] == 6
        assert daten["segmented"] is False

    def test_silben_plausibel(self):
        daten = analysiere(DE_TEXT, "de")
        for zeile in daten["sections"][0]["line_details"]:
            assert 5 <= zeile["syllables"] <= 12, zeile

    def test_reimschema_abab(self):
        daten = analysiere(DE_TEXT, "de")
        schema = daten["sections"][0]["rhyme_scheme"]
        assert schema[0] == schema[2] and schema[1] == schema[3]
        assert schema[0] != schema[1]

    def test_englisch_wird_erkannt(self):
        assert analysiere(EN_TEXT, "auto")["language"] == "en"

    def test_sektionsmarken_sind_keine_zeilen(self):
        daten = analysiere(DE_TEXT, "de")
        texte = [z["text"] for s in daten["sections"] for z in s["line_details"]]
        assert not any(t.startswith("[") for t in texte)

    def test_leerer_text(self):
        daten = analysiere("", "auto")
        assert daten["total_lines"] == 0 and daten["sections"] == []


class TestFlowtext:
    """Der Kern des Transkript-Problems: ein Absatz muss zu Zeilen werden."""

    def test_flowtext_erkennung(self):
        assert _ist_flowtext(TRANSKRIPT) is True
        assert _ist_flowtext(DE_TEXT) is False

    def test_segmentierung_bricht_um(self):
        woerterbuch, _ = _woerterbuch("en")
        zeilen = segmentiere_flowtext(TRANSKRIPT, woerterbuch, 8)
        assert len(zeilen) >= 6
        for zeile in zeilen:
            assert zeilensilben(zeile, woerterbuch) <= 14, zeile

    def test_segmentierung_nutzt_wiederkehrende_phrase_als_anker(self):
        """Eine wiederkehrende Phrase muss mehrfach eine Zeile eroeffnen.

        Welche Phrase das ist, gibt der Text vor — eine bestimmte Melodiezeile zu
        erwarten waere geraten. Genau daran ist die reine Textheuristik gemessen
        worden: Sie findet die haeufigste Wortfolge, nicht die Gesangszeile.
        Verlaessliche Zeilen liefert nur der Whisper-Weg mit Zeitmarken
        (FVM_WhisperChunksToLyricLines).
        """
        from nodes.music.lyrics_prosody import _refrain_phrasen

        woerter = TRANSKRIPT.split()
        phrasen = _refrain_phrasen(woerter)
        assert len(phrasen) >= 2, phrasen
        # Die Anker duerfen sich nicht ueberlappen, sonst entstehen Wortzeilen.
        starts = sorted(phrasen)
        for links, rechts in zip(starts, starts[1:]):
            assert rechts >= links + phrasen[links], (links, rechts, phrasen)

    def test_keine_zu_kurzen_zeilen(self):
        """Der gemessene Fehler: Zeilen mit einer Silbe ("the", "For", "life")."""
        from nodes.music.lyrics_prosody import MIN_SILBEN_ZEILE

        woerterbuch, _ = _woerterbuch("en")
        zeilen = segmentiere_flowtext(TRANSKRIPT, woerterbuch, 8)
        kurz = [
            z for z in zeilen[1:] if zeilensilben(z, woerterbuch) < MIN_SILBEN_ZEILE
        ]
        assert kurz == [], kurz

    def test_wiederholte_zeilen_werden_chorus(self):
        sektionen = _sektionen_aus_zeilen(
            ["Strophe eins", "Refrain hier", "Strophe zwei", "Refrain hier"]
        )
        tags = [s["tag"] for s in sektionen]
        assert "Chorus" in tags and "Verse" in tags

    def test_auto_segmentiert_transkript(self):
        daten = analysiere(TRANSKRIPT, "en", "auto")
        assert daten["segmented"] is True
        assert daten["total_lines"] >= 6
        # Genau das war vorher kaputt: eine Zeile mit 239 Silben.
        assert max(max(s["syllables_per_line"]) for s in daten["sections"]) <= 14

    def test_wie_eingegeben_laesst_flowtext_in_ruhe(self):
        daten = analysiere(TRANSKRIPT, "en", "wie eingegeben")
        assert daten["segmented"] is False
        assert daten["total_lines"] == 1

    def test_ziel_silben_wirkt(self):
        kurz = analysiere(TRANSKRIPT, "en", "fliesstext segmentieren", 5)["total_lines"]
        lang = analysiere(TRANSKRIPT, "en", "fliesstext segmentieren", 12)[
            "total_lines"
        ]
        assert kurz > lang

    def test_kurzfassung_weist_auf_schaetzung_hin(self):
        kurz = kurzfassung(analysiere(TRANSKRIPT, "en", "auto"))
        assert "estimated" in kurz


class TestNodeProsodie:
    def test_vier_ausgaben(self):
        ergebnis = FVM_LyricsProsodyAnalyze().execute(lyrics=DE_TEXT, language="de")
        assert isinstance(ergebnis, tuple) and len(ergebnis) == 4
        assert json.loads(ergebnis[0])["total_lines"] == 6
        assert "rhyme scheme" in ergebnis[1]
        assert ergebnis[2].startswith("# Prosodie")
        assert ergebnis[3] == "de"

    def test_node_segmentiert_transkript(self):
        ergebnis = FVM_LyricsProsodyAnalyze().execute(
            lyrics=TRANSKRIPT, language="en", struktur="auto", ziel_silben=8
        )
        daten = json.loads(ergebnis[0])
        assert daten["segmented"] is True and daten["total_lines"] >= 6
        assert "geschaetzt" in ergebnis[2]

    def test_kurzfassung_nennt_jede_sektion(self):
        kurz = kurzfassung(analysiere(DE_TEXT, "de"))
        assert "[Verse]" in kurz and "[Chorus]" in kurz


class TestPromptBuilder:
    def _bau(self, **over):
        args = dict(
            user_prompt="Schreibe einen Song.",
            system_prompt="Du bist Komponist.",
            prosody_brief="[Verse] 4 lines | syllables per line: 8/8/8/8 | rhyme scheme: ABAB",
            flavor_art="thema",
            flavor_text="Abschied am Bahnhof",
            strenge="nah",
            referenz_mitgeben=False,
        )
        args.update(over)
        return FVM_CoverLyricsPromptBuilder().execute(**args)

    def test_prompts_bleiben_erhalten(self):
        user, system, _ = self._bau()
        assert user.startswith("Schreibe einen Song.")
        assert system.startswith("Du bist Komponist.")

    def test_formvorgaben_landen_im_user_prompt(self):
        user, _, _ = self._bau()
        assert "COVER FORM TARGET" in user and "rhyme scheme: ABAB" in user
        assert "Abschied am Bahnhof" in user

    def test_system_prompt_haelt_ausgabeformat_fest(self):
        _, system, _ = self._bau()
        assert "COVER MODE" in system and "four-section output contract" in system

    def test_toleranz_folgt_strenge(self):
        assert "+/-0" in self._bau(strenge="streng")[0]
        assert "+/-2" in self._bau(strenge="frei")[0]

    def test_kopierverbot_immer_enthalten(self):
        user, system, _ = self._bau()
        assert "Do not reuse lines" in user
        assert "never reuse phrases" in system

    def test_referenztext_nur_auf_wunsch(self):
        ohne, _, info = self._bau(reference_lyrics=DE_TEXT)
        assert "REFERENCE LYRICS" not in ohne and "Referenztext im Prompt: nein" in info
        mit, _, info_mit = self._bau(reference_lyrics=DE_TEXT, referenz_mitgeben=True)
        assert "REFERENCE LYRICS" in mit and "Referenztext im Prompt: ja" in info_mit

    def test_flavor_arten_formulieren_unterschiedlich(self):
        assert "theme to write about" in self._bau(flavor_art="thema")[0]
        assert "work these terms" in self._bau(flavor_art="begriffe")[0]
        assert "rewrite the following text" in self._bau(flavor_art="umdichten")[0]

    def test_leerer_flavor_fuegt_keinen_block_ein(self):
        assert "NEW LYRIC CONTENT" not in self._bau(flavor_text="   ")[0]

    def test_sprache_wird_uebernommen(self):
        assert "Keep the language of the new lyrics: de" in self._bau(language="de")[0]

    def test_bloecke_ohne_node(self):
        block = baue_bloecke("kurz", "thema", "Sommer", "de", "streng")
        assert "COVER FORM TARGET" in block and "Sommer" in block and "+/-0" in block
        assert "COVER MODE" in system_zusatz("streng")

    def test_kette_analyse_zu_prompt(self):
        """Vollstaendiger Weg: Transkript -> Analyse -> Prompt."""
        _, kurz, _, sprache = FVM_LyricsProsodyAnalyze().execute(
            lyrics=TRANSKRIPT, language="auto", struktur="auto", ziel_silben=8
        )
        user, system, info = self._bau(prosody_brief=kurz, language=sprache)
        assert "syllables per line" in user and "COVER MODE" in system
        assert "Strenge: nah" in info
