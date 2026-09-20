"""Messe die Form eines Songtexts: Sektionen, Silben pro Zeile, Reimschema.

Wofuer das gebraucht wird: Fuer ein Cover soll ein LLM einen NEUEN Text
schreiben, der auf die bestehende Melodie passt. Damit das singbar wird, muss
der neue Text die Form des Originals treffen — gleiche Sektionsfolge, gleiche
Zeilenzahl, gleiche Silbenzahl pro Zeile, gleiches Reimschema. Ein LLM schaetzt
Silben notorisch schlecht, also wird hier gemessen und dem Modell als Vorgabe
mitgegeben.

Was hier NICHT passiert: Kein BPM, kein Takt, keine Notenwerte. Die kommen aus
dem Audio (SheetSage2 macht daraus die ABC-Partitur), nicht aus dem Text. Das
Silbenprofil pro Zeile ist das textseitige Gegenstueck dazu.

**Fliesstext-Problem.** Beide verfuegbaren Transcriber liefern einen Absatz ohne
Zeilenumbrueche: HeartMuLa gibt nur ``result["text"]`` zurueck, Whisper ueber
comfy_mtb koennte Zeitmarken liefern, der Lyrics-Pfad nutzt sie aber nicht. Ohne
Zeilen ist die Messung wertlos — gemessen an einem Toto-Transkript ergab das
*eine* Zeile mit 239 Silben. Deshalb kann dieser Node Fliesstext selbst
segmentieren (``struktur = "fliesstext segmentieren"``):

1. Wiederkehrende Wortfolgen finden. In einem Songtext ist die haeufigste lange
   Phrase der Refrain, und ihre Vorkommen sind verlaessliche Zeilenanfaenge —
   verlaesslicher als jede Interpunktion, die in Transkripten ohnehin fehlt.
2. Zwischen diesen Ankern nach Silbenbudget umbrechen (Ziel ~8, hart bei ~12),
   weil gesungene Zeilen selten laenger sind.
3. Zeilen, die mehrfach identisch vorkommen, als ``[Chorus]`` markieren, den
   Rest als ``[Verse]``.

Das ist eine Schaetzung und wird als solche im Bericht benannt. Wer es genau
will, korrigiert das Transkript von Hand und stellt ``struktur`` auf
``"wie eingegeben"`` — dann wird nur gemessen, nicht geraten.

Silben: pyphen (Woerterbuecher fuer de/en/es/fr/it/nl im venv), sonst
Vokalgruppen. Reime: orthografisch ab der letzten Vokalgruppe, y->i, ohne
Diakritika, stummes Schluss-e abgeschnitten (time -> im, rhyme -> im).
Phonem-basiert waere genauer, aber g2p_en braucht hier fehlende NLTK-Daten und
espeak eine Binaerdatei.
"""

import json
import re
import unicodedata
from collections import Counter

# Eine Zeile wie "[Verse]" oder "[Chorus 2]" ist eine Sektionsmarke, keine
# Textzeile. Laengere Klammerinhalte sind Regieanweisungen und bleiben Text.
SEKTION_RE = re.compile(r"^\s*\[([^\]]{1,40})\]\s*$")

# Woerter inklusive Apostroph (isn't, don't), ohne Zahlen und Satzzeichen.
WORT_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)

VOKALE = set("aeiouy")

MIN_LAENGE_STUMMES_E = 4

PYPHEN_SPRACHEN = {
    "de": "de_DE",
    "en": "en_US",
    "es": "es",
    "fr": "fr",
    "it": "it_IT",
    "nl": "nl_NL",
    "pt": "pt_PT",
    "sv": "sv",
}

# Phrasenlaenge fuer die Refrain-Suche. Kuerzer als 3 Woerter trifft Floskeln,
# laenger als 7 findet in kurzen Texten nichts mehr.
PHRASE_MIN = 3
PHRASE_MAX = 7

ZIEL_SILBEN_STANDARD = 8
HART_SILBEN_FAKTOR = 1.5

# Kuerzere Zeilen sind keine Gesangszeilen, sondern Reste einer Fehltrennung.
MIN_SILBEN_ANTEIL = 0.5
MIN_SILBEN_ZEILE = 3


def _ohne_diakritika(wort: str) -> str:
    zerlegt = unicodedata.normalize("NFD", wort.lower())
    return "".join(z for z in zerlegt if unicodedata.category(z) != "Mn")


def _vokalgruppen(wort: str) -> int:
    """Silbenzahl als Zahl der Vokalgruppen — der sprachlose Rueckfall."""
    rein = _ohne_diakritika(wort)
    gruppen = 0
    davor_vokal = False
    for zeichen in rein:
        ist_vokal = zeichen in VOKALE
        if ist_vokal and not davor_vokal:
            gruppen += 1
        davor_vokal = ist_vokal
    if rein.endswith("e") and gruppen > 1:
        gruppen -= 1
    return max(1, gruppen)


def _silben(wort: str, woerterbuch) -> int:
    if woerterbuch is None:
        return _vokalgruppen(wort)
    getrennt = woerterbuch.inserted(wort)
    return max(1, len([teil for teil in getrennt.split("-") if teil]))


def zeilensilben(zeile: str, woerterbuch) -> int:
    return sum(_silben(wort, woerterbuch) for wort in WORT_RE.findall(zeile))


def reimschluessel(wort: str) -> str:
    """Endung ab der letzten Vokalgruppe, normalisiert.

    Nacht/gebracht -> acht, time/rhyme -> im, love/above -> ov.
    """
    rein = re.sub(r"[^a-z]", "", _ohne_diakritika(wort)).replace("y", "i")
    if not rein:
        return ""
    if (
        len(rein) >= MIN_LAENGE_STUMMES_E
        and rein.endswith("e")
        and rein[-2] not in VOKALE
    ):
        rein = rein[:-1]
    letzte = -1
    for index, zeichen in enumerate(rein):
        if zeichen in VOKALE:
            letzte = index
    if letzte < 0:
        return rein[-2:]
    return rein[letzte:]


def _sprache_erkennen(text: str) -> str:
    probe = " ".join(text.split())[:2000]
    if len(probe) < 12:
        return "en"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # sonst wechselt das Ergebnis pro Aufruf
        return detect(probe)[:2]
    except Exception:
        pass
    woerter = {w.lower() for w in WORT_RE.findall(probe)}
    deutsch = {
        "und",
        "nicht",
        "ich",
        "der",
        "die",
        "das",
        "mit",
        "ist",
        "du",
        "wir",
        "nacht",
    }
    englisch = {"and", "the", "you", "not", "with", "is", "we", "night", "love", "time"}
    return "de" if len(woerter & deutsch) > len(woerter & englisch) else "en"


def _woerterbuch(sprache: str):
    try:
        import pyphen

        code = PYPHEN_SPRACHEN.get(sprache, sprache)
        if code not in pyphen.LANGUAGES:
            code = PYPHEN_SPRACHEN.get(sprache.split("_")[0], "")
        if code and code in pyphen.LANGUAGES:
            return pyphen.Pyphen(lang=code), code
    except Exception:
        pass
    return None, ""


def _refrain_phrasen(woerter: list) -> dict:
    """Startindex -> Laenge wiederkehrender Phrasen, ohne Ueberlappungen.

    Ueberlappungen sind der Grund, warum ein naiver Ansatz hier scheitert: Ein
    haeufiges 7-Gramm erzeugt an sieben aufeinanderfolgenden Positionen einen
    Treffer, und wer an jedem Treffer umbricht, bekommt Zeilen mit einer Silbe.
    Nach einem Treffer werden die Positionen innerhalb der Phrase deshalb
    uebersprungen.
    """
    klein = [w.lower() for w in woerter]
    for laenge in range(PHRASE_MAX, PHRASE_MIN - 1, -1):
        if len(klein) < laenge * 2:
            continue
        zaehler = Counter(
            tuple(klein[i : i + laenge]) for i in range(len(klein) - laenge + 1)
        )
        haeufige = {phrase for phrase, anzahl in zaehler.items() if anzahl >= 2}
        if not haeufige:
            continue
        treffer, index = {}, 0
        while index <= len(klein) - laenge:
            if tuple(klein[index : index + laenge]) in haeufige:
                treffer[index] = laenge
                index += laenge
            else:
                index += 1
        if treffer:
            return treffer
    return {}


def segmentiere_flowtext(
    text: str, woerterbuch, ziel_silben: int = ZIEL_SILBEN_STANDARD
) -> list:
    """Zerlege einen Absatz in singbare Zeilen (Refrain-Phrasen + Silbenbudget).

    Eine erkannte Phrase bleibt zusammen — sie ist die Refrainzeile. Zwischen
    den Phrasen wird nach Silbenbudget umbrochen, und Zeilen unter der
    Mindestlaenge werden am Ende mit ihrer Vorgaengerin verschmolzen, damit
    keine Ein-Wort-Zeilen stehen bleiben.
    """
    woerter = WORT_RE.findall(text)
    if not woerter:
        return []
    phrasen = _refrain_phrasen(woerter)
    hart = max(ziel_silben + 2, int(ziel_silben * HART_SILBEN_FAKTOR))
    mindest = max(MIN_SILBEN_ZEILE, int(ziel_silben * MIN_SILBEN_ANTEIL))
    zeilen, aktuell, silben = [], [], 0
    phrase_endet = -1

    def abschliessen():
        nonlocal aktuell, silben
        if aktuell:
            zeilen.append(" ".join(aktuell))
            aktuell, silben = [], 0

    for index, wort in enumerate(woerter):
        in_phrase = index <= phrase_endet
        if index in phrasen and not in_phrase:
            if silben >= mindest:
                abschliessen()
            phrase_endet = index + phrasen[index] - 1
            in_phrase = True
        elif not in_phrase and silben >= ziel_silben:
            abschliessen()
        elif in_phrase and silben >= hart:
            abschliessen()
        aktuell.append(wort)
        silben += _silben(wort, woerterbuch)
        if index == phrase_endet:
            abschliessen()
    abschliessen()

    verschmolzen = []
    for zeile in zeilen:
        if verschmolzen and zeilensilben(zeile, woerterbuch) < mindest:
            verschmolzen[-1] = f"{verschmolzen[-1]} {zeile}"
        else:
            verschmolzen.append(zeile)
    return verschmolzen


def _sektionen_aus_zeilen(zeilen: list) -> list:
    """Mehrfach vorkommende Zeilen sind Refrain, der Rest Strophe."""
    normal = [" ".join(z.lower().split()) for z in zeilen]
    haeufig = {z for z, anzahl in Counter(normal).items() if anzahl >= 2}
    sektionen, aktuell, art = [], [], None
    for zeile, norm in zip(zeilen, normal):
        neue_art = "Chorus" if norm in haeufig else "Verse"
        if art is not None and neue_art != art:
            sektionen.append({"tag": art, "zeilen": aktuell})
            aktuell = []
        art = neue_art
        aktuell.append(zeile)
    if aktuell:
        sektionen.append({"tag": art or "Verse", "zeilen": aktuell})
    return sektionen


def _sektionen_aus_text(lyrics: str) -> list:
    sektionen = []
    aktuell = {"tag": "", "zeilen": []}
    for rohzeile in (lyrics or "").replace("\r\n", "\n").split("\n"):
        treffer = SEKTION_RE.match(rohzeile)
        if treffer:
            if aktuell["zeilen"] or aktuell["tag"]:
                sektionen.append(aktuell)
            aktuell = {"tag": treffer.group(1).strip(), "zeilen": []}
            continue
        if rohzeile.strip():
            aktuell["zeilen"].append(rohzeile.strip())
    if aktuell["zeilen"] or aktuell["tag"]:
        sektionen.append(aktuell)
    return sektionen


def _ist_flowtext(lyrics: str) -> bool:
    """Ein Absatz ohne verwertbare Umbrueche — typisch fuer ein Transkript."""
    zeilen = [z for z in (lyrics or "").splitlines() if z.strip()]
    if len(zeilen) <= 1:
        return True
    lang = [z for z in zeilen if len(WORT_RE.findall(z)) > 20]
    return len(lang) >= max(1, len(zeilen) // 2)


def analysiere(
    lyrics: str,
    sprache: str = "auto",
    struktur: str = "auto",
    ziel_silben: int = ZIEL_SILBEN_STANDARD,
) -> dict:
    """Zerlege den Text in Sektionen und messe jede Zeile."""
    erkannt = _sprache_erkennen(lyrics) if sprache in ("auto", "", None) else sprache
    woerterbuch, woerterbuch_code = _woerterbuch(erkannt)

    segmentiert = struktur == "fliesstext segmentieren" or (
        struktur == "auto" and _ist_flowtext(lyrics)
    )
    if segmentiert:
        marken_frei = "\n".join(
            z for z in (lyrics or "").splitlines() if not SEKTION_RE.match(z)
        )
        zeilen = segmentiere_flowtext(marken_frei, woerterbuch, ziel_silben)
        sektionen = _sektionen_aus_zeilen(zeilen)
    else:
        sektionen = _sektionen_aus_text(lyrics)

    ergebnis = {
        "language": erkannt,
        "syllable_dictionary": woerterbuch_code or "vowel-groups",
        "segmented": bool(segmentiert),
        "target_syllables": ziel_silben if segmentiert else None,
        "sections": [],
        "total_lines": 0,
        "total_syllables": 0,
    }

    for sektion in sektionen:
        zeilen_info = []
        schluessel_zu_buchstabe = {}
        schema = []
        for zeile in sektion["zeilen"]:
            woerter = WORT_RE.findall(zeile)
            silben = sum(_silben(wort, woerterbuch) for wort in woerter)
            endwort = woerter[-1] if woerter else ""
            schluessel = reimschluessel(endwort)
            if schluessel:
                buchstabe = schluessel_zu_buchstabe.setdefault(
                    schluessel, chr(ord("A") + len(schluessel_zu_buchstabe) % 26)
                )
            else:
                buchstabe = "-"
            schema.append(buchstabe)
            zeilen_info.append(
                {
                    "text": zeile,
                    "syllables": silben,
                    "words": len(woerter),
                    "end_word": endwort,
                    "rhyme": buchstabe,
                }
            )
        silbenzahlen = [z["syllables"] for z in zeilen_info]
        ergebnis["sections"].append(
            {
                "tag": sektion["tag"] or "(ohne Marke)",
                "lines": len(zeilen_info),
                "syllables_per_line": silbenzahlen,
                "syllable_span": [min(silbenzahlen), max(silbenzahlen)]
                if silbenzahlen
                else [0, 0],
                "rhyme_scheme": "".join(schema),
                "line_details": zeilen_info,
            }
        )
        ergebnis["total_lines"] += len(zeilen_info)
        ergebnis["total_syllables"] += sum(silbenzahlen)

    ergebnis["section_order"] = [s["tag"] for s in ergebnis["sections"]]
    return ergebnis


def kurzfassung(daten: dict) -> str:
    """Kompakte Form fuer den LLM-Prompt — pro Sektion eine Zeile."""
    zeilen = [
        f"Language of the reference lyrics: {daten.get('language', '?')}",
        f"Sections in order: {' -> '.join(daten.get('section_order') or []) or '(none)'}",
    ]
    for sektion in daten.get("sections", []):
        zeilen.append(
            f"[{sektion['tag']}] {sektion['lines']} lines | "
            f"syllables per line: {'/'.join(str(z) for z in sektion['syllables_per_line'])} | "
            f"rhyme scheme: {sektion['rhyme_scheme'] or '-'}"
        )
    zeilen.append(
        f"Total: {daten.get('total_lines', 0)} lines, "
        f"{daten.get('total_syllables', 0)} syllables"
    )
    if daten.get("segmented"):
        zeilen.append(
            "Note: line breaks were estimated from a transcript without line breaks "
            "(repeated phrases plus a syllable budget). Treat the line lengths as a target, "
            "the exact split as an approximation."
        )
    return "\n".join(zeilen)


def bericht(daten: dict) -> str:
    """Lesbarer Markdown-Bericht zum Nachvollziehen im Canvas."""
    teile = [
        "# Prosodie der Referenz\n",
        f"- Sprache: **{daten.get('language', '?')}**",
        f"- Silbenzaehlung: `{daten.get('syllable_dictionary', '?')}`",
        f"- Zeilen: {'geschaetzt (Fliesstext segmentiert, Ziel ' + str(daten.get('target_syllables')) + ' Silben)' if daten.get('segmented') else 'wie eingegeben'}",
        f"- Gesamt: {daten.get('total_lines', 0)} Zeilen, {daten.get('total_syllables', 0)} Silben\n",
    ]
    for sektion in daten.get("sections", []):
        teile.append(
            f"## [{sektion['tag']}] — {sektion['lines']} Zeilen, "
            f"Reimschema `{sektion['rhyme_scheme'] or '-'}`"
        )
        for index, zeile in enumerate(sektion["line_details"], 1):
            teile.append(
                f"{index}. `{zeile['syllables']:>2}` Silben, Reim `{zeile['rhyme']}` "
                f"(Endwort *{zeile['end_word']}*) — {zeile['text']}"
            )
        teile.append("")
    teile.append(
        "> Silben und Reime sind heuristisch gemessen (pyphen bzw. Endsilben-Vergleich). "
        "Takt und Tempo kommen aus dem Audio, nicht aus dem Text. Fuer eine exakte Form "
        "das Transkript von Hand in Zeilen und Sektionen bringen und `struktur` auf "
        "`wie eingegeben` stellen."
    )
    return "\n".join(teile)


class FVM_LyricsProsodyAnalyze:
    """Sektionen, Silben pro Zeile und Reimschema eines Songtexts messen."""

    CATEGORY = "FVMtools/music"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Originaltext, z. B. aus einem Transcriber.",
                    },
                ),
                "language": (
                    ["auto", "de", "en", "es", "fr", "it", "nl", "pt", "sv"],
                    {"default": "auto"},
                ),
                "struktur": (
                    ["auto", "wie eingegeben", "fliesstext segmentieren"],
                    {
                        "default": "auto",
                        "tooltip": "auto: segmentiert nur, wenn der Text keine Zeilen hat.",
                    },
                ),
                "ziel_silben": (
                    "INT",
                    {
                        "default": ZIEL_SILBEN_STANDARD,
                        "min": 4,
                        "max": 20,
                        "tooltip": "Angestrebte Silben pro Zeile beim Segmentieren.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prosody_json", "prosody_brief", "report_md", "language")
    FUNCTION = "execute"

    def execute(
        self, lyrics, language="auto", struktur="auto", ziel_silben=ZIEL_SILBEN_STANDARD
    ):
        daten = analysiere(lyrics, language, struktur, ziel_silben)
        return (
            json.dumps(daten, ensure_ascii=False),
            kurzfassung(daten),
            bericht(daten),
            daten.get("language", ""),
        )
