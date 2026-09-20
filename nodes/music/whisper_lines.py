"""Aus Whisper-Passagen singbare Zeilen machen — mit echten Zeitmarken.

Warum das noetig ist: Die Zeilen eines Songtexts lassen sich aus reinem Text
nicht verlaesslich rekonstruieren. Gemessen an einem Toto-Transkript brach eine
reine Textheuristik mitten in der Gesangszeile um ("hold me It's not in the way
You say"), weil die haeufigste Wortfolge nicht die Melodiezeile ist. Damit wird
das Reimschema — der Zweck der Messung — Zufall.

Whisper weiss es besser: Es liefert Zeitstempel pro Passage, und eine Passage
ist in der Praxis meist eine Gesangszeile. Diese Information steckt in
comfy_mtb's ``WHISPER_CHUNKS``; der String-Ausgang von ``Process Whisper
Output`` ist dagegen derselbe Fliesstext wie vorher.

**Zwei Eigenheiten der mtb-Chunks, beide an echten Daten gemessen:**

1. Der Text ist aus BPE-Tokens mit Leerzeichen zusammengesetzt:
   ``ĠIt 's Ġnot Ġin Ġthe Ġway`` und ``Ġtreat in '``. ``Ġ`` markiert einen
   Wortanfang, Tokens ohne ``Ġ`` gehoeren an das vorherige Wort. Ohne
   Zusammensetzen zaehlt der Silbenzaehler Bruchstuecke und jedes Reimwort ist
   falsch. ``token_text`` macht daraus ``It's not in the way`` und ``treatin'``.
2. Passagen sind schon segmentiert und liegen oft ohne Pause aneinander, also
   wird standardmaessig NICHT verschmolzen (``pause_zeile = 0``). Nur Fetzen
   unter ``MIN_WOERTER_ZEILE`` Woertern wandern an die Vorgaengerin.

Sektionen: Wiederkehrende Zeilen sind der Refrain, eine lange Pause beginnt
einen Abschnitt. Der Vergleich laeuft ueber eine normalisierte Form (klein, ohne
Satzzeichen), weil Whisper dieselbe Zeile zweimal leicht unterschiedlich
schreibt. Ein Abschnitt braucht mindestens ``MIN_ZEILEN_SEKTION`` Zeilen, sonst
entstehen Ein-Zeilen-Sektionen, die keine Form beschreiben.
"""

import re
from collections import Counter

MIN_WOERTER_ZEILE = 2
MIN_ZEILEN_SEKTION = 2

# BPE-Marker aus GPT-2/Whisper-Tokenizern.
WORTANFANG = "Ġ"  # Ġ
ZEILENUMBRUCH = "Ċ"  # Ċ


def token_text(rohtext: str) -> str:
    """Setze BPE-Tokens zu lesbarem Text zusammen (siehe Modul-Docstring)."""
    text = str(rohtext or "")
    if WORTANFANG not in text and ZEILENUMBRUCH not in text:
        return " ".join(text.split())
    stuecke = []
    for token in text.replace(ZEILENUMBRUCH, f" {WORTANFANG}").split(" "):
        if not token:
            continue
        if token.startswith(WORTANFANG):
            stuecke.append(" " + token[len(WORTANFANG) :])
        else:
            stuecke.append(token)
    return " ".join("".join(stuecke).split())


def _chunkliste(chunks):
    """Die Chunks liegen als dict mit 'chunks' oder als Liste vor."""
    if isinstance(chunks, dict):
        return chunks.get("chunks") or []
    return chunks or []


def _zeitpaar(chunk):
    zeit = chunk.get("timestamp") or [None, None]
    start = zeit[0] if len(zeit) > 0 else None
    ende = zeit[1] if len(zeit) > 1 else None
    return start, ende


def _vergleichsform(zeile: str) -> str:
    """Klein, ohne Satzzeichen — damit zwei Schreibweisen derselben Zeile treffen."""
    return " ".join(re.sub(r"[^\w\s']", " ", zeile.lower()).split())


def chunks_zu_zeilen(
    chunks,
    pause_zeile: float = 0.0,
    pause_sektion: float = 1.6,
    mit_sektionen: bool = True,
) -> str:
    """Baue Text mit Zeilenumbruechen und optionalen Sektionsmarken."""
    liste = _chunkliste(chunks)
    zeilen, luecken, letztes_ende = [], [], None
    for chunk in liste:
        text = token_text(chunk.get("text"))
        if not text:
            continue
        start, ende = _zeitpaar(chunk)
        luecke = None
        if start is not None and letztes_ende is not None:
            luecke = max(0.0, float(start) - float(letztes_ende))
        zu_kurz = len(text.split()) < MIN_WOERTER_ZEILE
        dicht = luecke is not None and pause_zeile > 0 and luecke < pause_zeile
        if zeilen and (zu_kurz or dicht):
            zeilen[-1] = f"{zeilen[-1]} {text}"
        else:
            zeilen.append(text)
            luecken.append(luecke if luecke is not None else 0.0)
        if ende is not None:
            letztes_ende = ende
    if not zeilen:
        return ""
    if not mit_sektionen:
        return "\n".join(zeilen)

    normal = [_vergleichsform(z) for z in zeilen]
    haeufig = {z for z, anzahl in Counter(normal).items() if anzahl >= 2 and z}
    # Erst die Rohfolge bilden, dann zu kurze Abschnitte mit dem Nachbarn
    # verschmelzen — sonst wechselt die Marke bei jeder zweiten Zeile.
    gruppen = []
    for index, (zeile, norm) in enumerate(zip(zeilen, normal)):
        art = "Chorus" if norm in haeufig else "Verse"
        pause = luecken[index] if index < len(luecken) else 0.0
        neue_gruppe = (
            not gruppen
            or gruppen[-1]["art"] != art
            or (
                pause >= pause_sektion
                and len(gruppen[-1]["zeilen"]) >= MIN_ZEILEN_SEKTION
            )
        )
        if neue_gruppe:
            gruppen.append({"art": art, "zeilen": [zeile]})
        else:
            gruppen[-1]["zeilen"].append(zeile)
    verdichtet = []
    for gruppe in gruppen:
        if (
            verdichtet
            and len(gruppe["zeilen"]) < MIN_ZEILEN_SEKTION
            and gruppe["art"] != verdichtet[-1]["art"]
        ):
            verdichtet[-1]["zeilen"].extend(gruppe["zeilen"])
        elif verdichtet and gruppe["art"] == verdichtet[-1]["art"]:
            verdichtet[-1]["zeilen"].extend(gruppe["zeilen"])
        else:
            verdichtet.append(gruppe)

    ausgabe = []
    for gruppe in verdichtet:
        if ausgabe:
            ausgabe.append("")
        ausgabe.append(f"[{gruppe['art']}]")
        ausgabe.extend(gruppe["zeilen"])
    return "\n".join(ausgabe)


def bericht(chunks, text: str) -> str:
    liste = _chunkliste(chunks)
    zeilen = [
        z for z in text.splitlines() if z.strip() and not re.match(r"^\[.+\]$", z)
    ]
    dauer = 0.0
    for chunk in liste:
        _, ende = _zeitpaar(chunk)
        if ende is not None:
            dauer = max(dauer, float(ende))
    marken = len(re.findall(r"^\[.+\]$", text, re.M))
    return (
        f"Whisper-Passagen: {len(liste)} | daraus Zeilen: {len(zeilen)} | "
        f"Sektionen: {marken} | erkannte Laenge: {dauer:.1f}s"
    )


class FVM_WhisperChunksToLyricLines:
    """Whisper-Passagen (mit Zeitmarken) in Zeilen und Sektionen umsetzen."""

    CATEGORY = "FVMtools/music"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "whisper_chunks": ("WHISPER_CHUNKS",),
                "pause_zeile": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": "0 = jede Passage bleibt eine Zeile (Whisper hat "
                        "schon segmentiert). Groesser 0 klebt Passagen "
                        "zusammen, deren Pause kuerzer ist.",
                    },
                ),
                "pause_sektion": (
                    "FLOAT",
                    {
                        "default": 1.6,
                        "min": 0.2,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Laengere Pause = neuer Abschnitt.",
                    },
                ),
                "mit_sektionen": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lyrics", "info")
    FUNCTION = "execute"

    def execute(
        self, whisper_chunks, pause_zeile=0.0, pause_sektion=1.6, mit_sektionen=True
    ):
        text = chunks_zu_zeilen(
            whisper_chunks, pause_zeile, pause_sektion, mit_sektionen
        )
        return (text, bericht(whisper_chunks, text))
