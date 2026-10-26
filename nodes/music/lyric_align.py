"""Lyrics-Zeilen gegen Whisper-Zeilen abgleichen: je Zeile Zeit, Abschnitt, Takt, Konfidenz.

Die Partitur sagt, wo Abschnitte und Takte im Audio liegen (``score_timing``),
Whisper sagt, wann welche Zeile gesungen wird. Dieser Node fuehrt beides fuer
die StudioSuite (Musikvideo) zusammen:

1. **Zeilenabgleich.** Die Zeilen des Prompts (mit ``[Verse]``-Marken) und die
   Whisper-Zeilen werden monoton gegeneinander ausgerichtet (dynamische
   Programmierung ueber die Text-Aehnlichkeit, ``difflib``). Whisper fasst oft
   mehrere Zeilen in eine Passage (run01: die ganze Strophe in einer), deshalb
   duerfen bis zu ``MAX_GRUPPE`` Zeilen auf eine Passage fallen und umgekehrt.
   Ohne Partner bleiben: fehlende Zeilen (nicht erkannt) und zusaetzliche
   Passagen (Vokalisen wie "Ho-ho-ho", Halluzinationen).
2. **Zeiten** (im Whisper-Sinn: Passagenanfang, im Mittel ``WHISPER_VORLAUF``
   vor dem ersten gesungenen Wort). Eine Zeile mit eigener Passage bekommt deren Zeitmarken. Teilen
   sich mehrere Zeilen eine Passage, wird sie nach dem Abstand der
   Gesangseinsaetze in der Partitur geteilt - die Abstaende stimmen gemessen
   auf 0,1-0,2 s (run01: 4,8 s Whisper gegen 5,0 s Partitur), die absolute Lage
   der Partitur dagegen nicht (siehe 3.). Fehlende Zeilen werden von der
   naechsten bekannten Zeile aus geschaetzt (``estimated``).
3. **Ausrichtung waehlen.** Ist die Partitur-Ausrichtung unsicher, liefert
   ``score_timing`` eine ``alternative``. Entschieden wird ueber den Versatz
   der Gesangseinsaetze: Whisper-Anfang minus Partitur-Einsatz der Zeile, im
   Median ueber verlaessliche Zeilen (Passage nach einer Pause). Mit der reinen
   Einsatz-Ausrichtung lag der Gesang an run01/run02 1,6-2,5 s VOR beiden
   Kandidaten (YuE2 kuerzt das Intro); seit der Gesangs-Einmessung in
   ``score_timing`` passt "best" auf 0,1-0,15 s. Passt keiner auf einen halben
   Schlag (plus Whisper-Ungenauigkeit), wird der aus Whisper gemessene Versatz
   genommen (``choice: "whisper"``) und gewarnt.

Abnehmer ist die Suite: sie liest die Datei (Ausgang ``#out_lyrics_timing``)
und legt Szenen- und Untertitelwechsel auf Zeilen, Takte auf Schnitte.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import unicodedata
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .abc_score import parse_score, tokenize_bar, walk_bars, AbcError
from .lyrics_prosody import _sektionen_aus_text, _sprache_erkennen, _woerterbuch, zeilensilben
from .score_timing import save_json, vocal_voice
from .song_extend import _output_dir
from .whisper_lines import zeilen_mit_zeiten

logger = logging.getLogger("FVMtools.LyricAlign")

CATEGORY = "FVMtools/music"

SCHEMA = "fvm.lyric_align/1"
MIN_AEHNLICHKEIT = 0.5      # darunter gilt ein Paar nicht als dieselbe Zeile (fehlt + zusaetzlich)
MAX_GRUPPE = 4               # bis zu 4 Zeilen in einer Passage (run01: ganze Strophe) bzw. umgekehrt
ANKER_AEHNLICHKEIT = 0.6     # nur gut erkannte Zeilen bestimmen den Versatz
ANKER_PAUSE = 0.3            # Passagen direkt nach der vorigen sind zu frueh (run01: 1,8 s) - kein Anker
WHISPER_VORLAUF = 0.3        # Passagenanfang liegt im Mittel 0,3 s vor dem ersten Wort (run02, Wortmarken)
TOLERANZ_SCHLAEGE = 0.5      # Kandidat passt, wenn der Gesang hoechstens so weit daneben liegt
TOLERANZ_MIN_S = 0.35        # ... aber nie enger als die Whisper-Streuung


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 3)


# ---------------------------------------------------------------- Text

def normalform(text: str) -> str:
    """Klein, ohne Satzzeichen und Akzente, ss fuer ß - fuer den Textvergleich."""
    text = unicodedata.normalize("NFKD", str(text or "").casefold().replace("ß", "ss"))
    text = "".join(z for z in text if unicodedata.category(z) != "Mn")
    return " ".join(re.sub(r"[^\w\s]", " ", text).replace("_", " ").split())


def aehnlichkeit(a: str, b: str) -> float:
    """0..1, ``difflib``-Quote auf der Normalform (wie ein normierter Levenshtein)."""
    a, b = normalform(a), normalform(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _abschnittsname(tag: str) -> str:
    """``[Verse 2]`` -> ``verse``, ``Pre-Chorus`` -> ``prechorus``."""
    return re.sub(r"[^a-z]", "", (tag or "").lower())


def lyrics_zeilen(lyrics: str) -> List[Dict[str, Any]]:
    """Prompt-Lyrics als flache Zeilenliste mit Abschnittsmarke und Abschnittsnummer."""
    zeilen: List[Dict[str, Any]] = []
    for nr, sektion in enumerate(_sektionen_aus_text(lyrics)):
        for text in sektion["zeilen"]:
            zeilen.append({"tag": sektion["tag"], "block": nr, "text": text})
    return zeilen


# ---------------------------------------------------------------- Abgleich

def zeilen_abgleich(lyrics: Sequence[str], whisper: Sequence[str],
                    min_aehnlichkeit: float = MIN_AEHNLICHKEIT,
                    max_gruppe: int = MAX_GRUPPE) -> List[Tuple[List[int], List[int], float]]:
    """Monotone Zuordnung Lyrics-Zeilen <-> Whisper-Zeilen.

    Rueckgabe: Gruppen ``(lyrics_indizes, whisper_indizes, aehnlichkeit)``; eine
    Seite ist leer fuer fehlende bzw. zusaetzliche Zeilen. Gewinn einer Gruppe:
    ``(Aehnlichkeit - Schwelle) * Zahl der Lyrics-Zeilen`` - so lohnt das
    Zusammenfassen mehrerer Zeilen nur, wenn die Passage wirklich alle enthaelt.
    """
    n, m = len(lyrics), len(whisper)
    minus = float("-inf")
    best = [[minus] * (m + 1) for _ in range(n + 1)]
    back: List[List[Optional[Tuple[int, int, float]]]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    cache: Dict[Tuple[int, int, int, int], float] = {}

    def sim(i0, i1, j0, j1):
        key = (i0, i1, j0, j1)
        if key not in cache:
            cache[key] = aehnlichkeit(" ".join(lyrics[i0:i1]), " ".join(whisper[j0:j1]))
        return cache[key]

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            cand: List[Tuple[float, Tuple[int, int, float]]] = []
            if i > 0 and best[i - 1][j] > minus:
                cand.append((best[i - 1][j], (1, 0, 0.0)))          # Zeile fehlt
            if j > 0 and best[i][j - 1] > minus:
                cand.append((best[i][j - 1], (0, 1, 0.0)))          # Passage zusaetzlich
            for a in range(1, min(max_gruppe, i) + 1):
                for b in range(1, min(max_gruppe, j) + 1):
                    if a > 1 and b > 1:
                        continue
                    prev = best[i - a][j - b]
                    if prev == minus:
                        continue
                    s = sim(i - a, i, j - b, j)
                    if s < min_aehnlichkeit:
                        continue
                    cand.append((prev + (s - min_aehnlichkeit) * a + 1e-6, (a, b, s)))
            if cand:
                best[i][j], back[i][j] = max(cand, key=lambda c: c[0])
    gruppen: List[Tuple[List[int], List[int], float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        a, b, s = back[i][j]
        gruppen.append((list(range(i - a, i)), list(range(j - b, j)), s))
        i, j = i - a, j - b
    gruppen.reverse()
    return gruppen


# ---------------------------------------------------------------- Partitur

def gesangseinsaetze(score) -> List[Tuple[int, float]]:
    """(Abschnitt, Partitur-Sekunde) jedes Gesangstons (ohne Haltebogen-Fortsetzungen)."""
    stimme = vocal_voice(score)
    bars, _ = walk_bars(score, score.sections)
    t = Fraction(0)
    out: List[Tuple[int, float]] = []
    for b in bars:
        if b.voice != stimme:
            continue
        try:
            toks = tokenize_bar(b.bar, b.bar_units)
        except AbcError:
            t += b.bar_units
            continue
        tied = False
        for tok in toks:
            if tok.kind == "tie":
                tied = True
            elif tok.kind in ("note", "mrest"):
                if tok.kind == "note" and not tok.head.startswith("z") and not tied:
                    out.append((b.section, score.units_to_seconds(t)))
                t += tok.length
                tied = False
    return out


def zeilen_einsaetze(score, zeilen: List[Dict[str, Any]], woerterbuch=None) -> List[Optional[Tuple[int, float]]]:
    """Partitur-Abschnitt und -Einsatz (Sekunde) jeder Lyrics-Zeile, oder None.

    Lyrics-Bloecke werden in Reihenfolge den Partitur-Abschnitten gleichen
    Namens zugeordnet (``[Verse]`` -> ``verse``). Im Abschnitt beginnt Zeile k
    beim Ton Nummer "Silben der Zeilen davor" - YuE2 setzt einen Ton je Silbe;
    weicht die Tonzahl ab, wird anteilig umgerechnet.
    """
    toene: Dict[int, List[float]] = {}
    for sec, t in gesangseinsaetze(score):
        toene.setdefault(sec, []).append(t)
    labels = [_abschnittsname(s.label) for s in score.sections]
    bloecke: Dict[int, List[int]] = {}
    for i, z in enumerate(zeilen):
        bloecke.setdefault(z["block"], []).append(i)
    out: List[Optional[Tuple[int, float]]] = [None] * len(zeilen)
    cursor = 0
    for block in sorted(bloecke):
        idx = bloecke[block]
        name = _abschnittsname(zeilen[idx[0]]["tag"])
        treffer = next((k for k in range(cursor, len(labels)) if name and labels[k] == name), None)
        if treffer is None or not toene.get(treffer):
            continue
        cursor = treffer + 1
        noten = toene[treffer]
        silben = [max(1, zeilensilben(zeilen[i]["text"], woerterbuch)) for i in idx]
        gesamt = float(sum(silben))
        kum = 0
        for i, s in zip(idx, silben):
            k = min(len(noten) - 1, int(round(kum * len(noten) / gesamt)))
            out[i] = (treffer, noten[k])
            kum += s
    return out


# ---------------------------------------------------------------- Ausrichtung

def _kandidaten(timing: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = [{"name": "best", "offset_s": float(timing["offset_s"]), "tempo_factor": float(timing["tempo_factor"])}]
    alt = timing.get("alternative")
    if alt:
        out.append({"name": "alternative", "offset_s": float(alt["offset_s"]),
                    "tempo_factor": float(alt["tempo_factor"])})
    return out


def _median(werte: List[float]) -> float:
    w = sorted(werte)
    n = len(w)
    return w[n // 2] if n % 2 else 0.5 * (w[n // 2 - 1] + w[n // 2])


def ausrichtung_waehlen(timing: Dict[str, Any], anker: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Kandidat waehlen, dessen Gesangseinsaetze am besten zu den Whisper-Anfaengen passen.

    ``anker``: (Whisper-Anfang, Partitur-Einsatz) verlaesslicher Zeilen.
    Abweichung = Median(Whisper-Anfang + Vorlauf - Kandidat(Partitur-Einsatz)).
    """
    beat_s = float(timing.get("beat_s") or 0.5)
    toleranz = max(TOLERANZ_SCHLAEGE * beat_s, TOLERANZ_MIN_S)
    kandidaten = _kandidaten(timing)
    for k in kandidaten:
        if anker:
            res = [w + WHISPER_VORLAUF - (k["offset_s"] + k["tempo_factor"] * s) for w, s in anker]
            k["residual_s"] = _r(_median(res))
            k["residual_beats"] = round(k["residual_s"] / beat_s, 2)
        else:
            k["residual_s"] = k["residual_beats"] = None
    wahl: Dict[str, Any] = {"candidates": kandidaten, "anchors": len(anker), "tolerance_s": _r(toleranz)}
    if not anker:
        k = kandidaten[0]
        wahl.update(choice="best", offset_s=k["offset_s"], tempo_factor=k["tempo_factor"], decided_by="none")
        return wahl
    best = min(kandidaten, key=lambda k: abs(k["residual_s"]))
    vocal = best["offset_s"] + best["residual_s"]
    wahl["vocal_offset_s"] = _r(vocal)
    wahl["closest_candidate"] = best["name"]
    if abs(best["residual_s"]) <= toleranz:
        wahl.update(choice=best["name"], offset_s=best["offset_s"], tempo_factor=best["tempo_factor"],
                    decided_by="whisper_lines")
    else:
        wahl.update(choice="whisper", offset_s=_r(vocal), tempo_factor=best["tempo_factor"],
                    decided_by="whisper_lines")
    return wahl


def _abschnitte(timing: Dict[str, Any], offset: float, scale: float) -> List[Dict[str, Any]]:
    dauer = float(timing["audio_duration_s"])
    out = []
    for s in timing["sections"]:
        a = offset + scale * float(s["score_start_s"])
        b = offset + scale * float(s["score_end_s"])
        out.append({"index": s["index"], "label": s["label"], "start_s": _r(max(0.0, a)),
                    "end_s": _r(min(max(0.0, b), dauer)), "in_audio": a < dauer and b > 0})
    return out


def _takt(timing: Dict[str, Any], offset: float, scale: float, t: float) -> Optional[Dict[str, Any]]:
    """Takt (und Schlag), in dem Audio-Zeit ``t`` liegt."""
    bars = timing.get("bars") or []
    treffer = None
    for b in bars:
        if offset + scale * float(b["score_s"]) <= t + 1e-6:
            treffer = b
        else:
            break
    if treffer is None:
        return None
    start = offset + scale * float(treffer["score_s"])
    beat_s = float(timing.get("beat_s") or 0.0)
    schlag = int((t - start) // beat_s) + 1 if beat_s > 0 else None
    return {"bar": treffer["index"], "beat": schlag, "section": treffer["section"], "bar_start_s": _r(start)}


# ---------------------------------------------------------------- Hauptfunktion

def align_lyrics(timing: Dict[str, Any], lyrics: str, whisper_chunks, abc: str = "",
                 pause_zeile: float = 0.0, min_aehnlichkeit: float = MIN_AEHNLICHKEIT) -> Dict[str, Any]:
    """Alles zusammen: Abgleich, Zeiten, Ausrichtungswahl, Zuordnung zu Abschnitt und Takt."""
    zeilen = lyrics_zeilen(lyrics)
    if not zeilen:
        raise ValueError("Keine Lyrics-Zeilen - nichts abzugleichen.")
    gehoert = [z for z in zeilen_mit_zeiten(whisper_chunks, pause_zeile) if z["start"] is not None]
    gruppen = zeilen_abgleich([z["text"] for z in zeilen], [w["text"] for w in gehoert], min_aehnlichkeit)

    einsaetze: List[Optional[Tuple[int, float]]] = [None] * len(zeilen)
    if abc and abc.strip():
        score = parse_score(abc)
        woerterbuch, _ = _woerterbuch(_sprache_erkennen(lyrics))
        einsaetze = zeilen_einsaetze(score, zeilen, woerterbuch)

    # Verlaessliche Anker: gut erkannte Zeile mit Partitur-Einsatz, Passage nach einer Pause
    def anker_aus(gruppen_, mit_pause):
        out = []
        for li, wi, s in gruppen_:
            if not (li and wi and s >= ANKER_AEHNLICHKEIT and einsaetze[li[0]] is not None):
                continue
            gap = gehoert[wi[0]]["gap"]
            if mit_pause and (gap is None or gap < ANKER_PAUSE):
                continue
            out.append((gehoert[wi[0]]["start"], einsaetze[li[0]][1]))
        return out

    anker = anker_aus(gruppen, True)
    schwache_anker = not anker
    if schwache_anker:   # Notnagel: auch anschliessende Passagen
        anker = anker_aus(gruppen, False)
    wahl = ausrichtung_waehlen(timing, anker)
    offset, scale = float(wahl["offset_s"]), float(wahl["tempo_factor"])

    def erwartet(i):
        """Whisper-artiger Anfang der Zeile laut gewaehlter Ausrichtung (oder None)."""
        if einsaetze[i] is None or not anker:
            return None
        return offset + scale * einsaetze[i][1] - WHISPER_VORLAUF

    # Zeiten je Zeile aus den Gruppen
    start: List[Optional[float]] = [None] * len(zeilen)
    ende: List[Optional[float]] = [None] * len(zeilen)
    quelle: List[str] = ["missing"] * len(zeilen)
    konfidenz = [0.0] * len(zeilen)
    passagen: List[List[int]] = [[] for _ in zeilen]
    extra: List[int] = []
    for li, wi, s in gruppen:
        if not li:
            extra.extend(wi)
            continue
        if not wi:
            continue
        a, b = gehoert[wi[0]]["start"], gehoert[wi[-1]]["end"]
        b = a if b is None else b
        for i in li:
            passagen[i] = list(wi)
            konfidenz[i] = round(s, 3)
        suffix = ""
        # Passage direkt nach der vorigen: Whisper setzt den Anfang zu frueh (run01: 1,8 s) -
        # dann den Anfang aus der Partitur nehmen, solange er in der Passage liegt
        gap = gehoert[wi[0]]["gap"]
        soll = erwartet(li[0])
        if (gap is None or gap < ANKER_PAUSE) and soll is not None and a < soll < b - 0.5:
            a, suffix = soll, "+score_start"
        if len(li) == 1:
            start[li[0]], ende[li[0]], quelle[li[0]] = a, b, "whisper" + suffix
            continue
        # Passage teilen: nach Partitur-Abstaenden, sonst nach Textlaenge
        rel = None
        if all(einsaetze[i] is not None for i in li):
            rel = [scale * (einsaetze[i][1] - einsaetze[li[0]][1]) for i in li]
            if rel[-1] >= (b - a) or any(y <= x for x, y in zip(rel, rel[1:])):
                rel = None
        src = "whisper_split_score"
        if rel is None:
            laengen = [max(1, len(normalform(zeilen[i]["text"]))) for i in li]
            ges = float(sum(laengen))
            rel, kum = [], 0
            for n_ in laengen:
                rel.append((b - a) * kum / ges)
                kum += n_
            src = "whisper_split_text"
        for k, i in enumerate(li):
            start[i] = a + rel[k]
            ende[i] = a + rel[k + 1] if k + 1 < len(li) else b
            quelle[i] = src + suffix

    # Fehlende Zeilen schaetzen: ueber die Partitur von der naechsten bekannten Zeile aus,
    # ohne Partitur zwischen den Nachbarn verteilt
    bekannt = [i for i in range(len(zeilen)) if start[i] is not None]
    for i in range(len(zeilen)):
        if start[i] is not None:
            continue
        if einsaetze[i] is not None:
            start[i] = offset + scale * einsaetze[i][1] - WHISPER_VORLAUF
            nach = next((k for k in range(i + 1, len(zeilen)) if einsaetze[k] is not None), None)
            ende[i] = (offset + scale * einsaetze[nach][1] - WHISPER_VORLAUF) if nach is not None else None
            quelle[i] = "estimated_score"
            continue
        vor = max((k for k in bekannt if k < i), default=None)
        nach = min((k for k in bekannt if k > i), default=None)
        if vor is not None and nach is not None:
            luecke = nach - vor
            a = ende[vor] if ende[vor] is not None else start[vor]
            start[i] = a + (start[nach] - a) * (i - vor - 1) / luecke
            ende[i] = a + (start[nach] - a) * (i - vor) / luecke
            quelle[i] = "estimated_neighbours"

    dauer = float(timing["audio_duration_s"])
    labels = [s["label"] for s in timing["sections"]]
    zeilen_out: List[Dict[str, Any]] = []
    for i, z in enumerate(zeilen):
        bezug = None if start[i] is None else start[i] + WHISPER_VORLAUF
        takt = _takt(timing, offset, scale, bezug) if bezug is not None else None
        sec = takt["section"] if takt else None
        soll = einsaetze[i]
        eintrag = {
            "index": i, "tag": z["tag"], "block": z["block"], "text": z["text"],
            "status": "matched" if passagen[i] else "missing",
            "heard": " / ".join(gehoert[w]["text"] for w in passagen[i]) or None,
            "whisper_lines": passagen[i],
            "shared_passage": len([k for k in range(len(zeilen)) if passagen[k] and passagen[k] == passagen[i]]) > 1,
            "start_s": _r(start[i]), "end_s": _r(ende[i]), "timing_source": quelle[i],
            "confidence": konfidenz[i],
            "section": sec, "section_label": labels[sec] if sec is not None else None,
            "bar": takt["bar"] if takt else None, "beat": takt["beat"] if takt else None,
            "score_section": soll[0] if soll else None,
            "score_onset_s": _r(soll[1]) if soll else None,
            "expected_start_s": _r(offset + scale * soll[1]) if soll else None,
        }
        eintrag["section_match"] = (sec == soll[0]) if (soll and sec is not None) else None
        zeilen_out.append(eintrag)

    extra_out = []
    for w in extra:
        g = gehoert[w]
        takt = _takt(timing, offset, scale, g["start"] + WHISPER_VORLAUF)
        sec = takt["section"] if takt else None
        extra_out.append({"whisper_index": w, "text": g["text"], "start_s": _r(g["start"]), "end_s": _r(g["end"]),
                          "section": sec, "section_label": labels[sec] if sec is not None else None,
                          "bar": takt["bar"] if takt else None})

    gefunden = [z for z in zeilen_out if z["status"] == "matched"]
    warnings: List[str] = []
    fehlend = [z["index"] for z in zeilen_out if z["status"] == "missing"]
    if fehlend:
        warnings.append(f"{len(fehlend)} Zeile(n) nicht erkannt: {', '.join(str(i + 1) for i in fehlend)}.")
    if wahl["choice"] == "whisper":
        warnings.append(f"Gesang liegt {wahl['candidates'][0]['residual_s']:+.2f} s neben der Partitur-Ausrichtung "
                        f"(kein Kandidat innerhalb {wahl['tolerance_s']} s) - Versatz aus Whisper genommen "
                        f"({wahl['offset_s']} s).")
    elif wahl["choice"] == "alternative":
        warnings.append("Zweitbeste Partitur-Ausrichtung passt besser zu den Whisper-Zeilen - gewaehlt.")
    if schwache_anker and anker:
        warnings.append("Keine Whisper-Passage nach einer Pause - Versatz aus anschliessenden Passagen (ungenauer).")
    if not abc or not abc.strip():
        warnings.append("Ohne Partitur: keine Gesangseinsaetze, Ausrichtung nicht geprueft.")
    nicht_passend = [z["index"] for z in zeilen_out if z["section_match"] is False]
    if nicht_passend:
        warnings.append(f"Zeile(n) {', '.join(str(i + 1) for i in nicht_passend)} liegen in einem anderen "
                        f"Partitur-Abschnitt als ihre Lyrics-Marke.")

    konf = [z["confidence"] for z in gefunden]
    return {
        "schema": SCHEMA,
        "alignment": wahl,
        "offset_s": _r(offset),
        "tempo_factor": round(scale, 5),
        "audio_duration_s": _r(dauer),
        "sections": _abschnitte(timing, offset, scale),
        "lines": zeilen_out,
        "extra": extra_out,
        "summary": {
            "lines": len(zeilen_out), "matched": len(gefunden), "missing": len(fehlend), "extra": len(extra_out),
            "whisper_lines": len(gehoert),
            "mean_confidence": round(sum(konf) / len(konf), 3) if konf else 0.0,
            "min_confidence": min(konf) if konf else 0.0,
        },
        "warnings": warnings,
    }


def report_text(data: Dict[str, Any]) -> str:
    """Kurzbericht fuer die Node-Anzeige."""
    w = data["alignment"]
    s = data["summary"]
    lines = [f"Ausrichtung: {w['choice']} (Versatz {data['offset_s']} s, Anker {w['anchors']})"
             + "".join(f", {k['name']} {k['residual_s']:+.2f} s" for k in w["candidates"]
                       if k.get("residual_s") is not None),
             f"Zeilen {s['lines']}: erkannt {s['matched']}, fehlend {s['missing']}, zusaetzlich {s['extra']}, "
             f"Konfidenz im Mittel {s['mean_confidence']:.2f}"]
    for z in data["lines"]:
        zeit = f"{z['start_s']:.2f}-{z['end_s']:.2f} s" if z["start_s"] is not None and z["end_s"] is not None else "?"
        ort = f"{z['section_label']} T{z['bar'] + 1}" if z["bar"] is not None else "-"
        lines.append(f"  {z['index'] + 1}. [{z['tag']}] {zeit} {ort} {z['confidence']:.2f} {z['timing_source']}: "
                     f"{z['text']}")
    lines += ["WARNUNG: " + x for x in data["warnings"]]
    return "\n".join(lines)


class FVM_LyricTimingAlign:
    """Lyrics-Zeilen des Prompts gegen Whisper-Zeilen und Partitur-Timing abgleichen, als JSON."""

    CATEGORY = CATEGORY
    FUNCTION = "align"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lyrics_timing_json", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timing_json": ("STRING", {"forceInput": True,
                                           "tooltip": "Ausgang timing_json von 'Partitur-Timing exportieren'."}),
                "lyrics": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Die Lyrics aus dem YuE2-Prompt, mit [Verse]/[Chorus]-Marken."}),
                "filename_prefix": ("STRING", {"default": "audio/music/timing/lyrics",
                                               "tooltip": "Datei: <prefix>_00001.json im ComfyUI-Ausgabeordner."}),
                "min_similarity": ("FLOAT", {"default": MIN_AEHNLICHKEIT, "min": 0.2, "max": 0.95, "step": 0.05,
                                             "tooltip": "Darunter gilt eine Whisper-Zeile nicht als diese "
                                                        "Lyrics-Zeile."}),
                "pause_zeile": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05,
                                          "tooltip": "Wie bei 'Whisper-Passagen zu Zeilen': 0 = jede Passage "
                                                     "bleibt eine Zeile."}),
                "save_file": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "whisper_chunks": ("WHISPER_CHUNKS", {"tooltip": "Von mtb 'Process Whisper Output'."}),
                "whisper_json": ("STRING", {"forceInput": True,
                                            "tooltip": "Alternativ: Chunks als JSON ({'chunks': [{text, "
                                                       "timestamp}]} oder Liste)."}),
                "abc": ("STRING", {"forceInput": True,
                                   "tooltip": "Die Partitur des Songs - fuer Gesangseinsaetze und die Wahl "
                                              "der Ausrichtung."}),
            },
        }

    def align(self, timing_json, lyrics, filename_prefix, min_similarity=MIN_AEHNLICHKEIT, pause_zeile=0.0,
              save_file=True, whisper_chunks=None, whisper_json=None, abc=""):
        if whisper_chunks is None:
            if not whisper_json or not str(whisper_json).strip():
                raise ValueError("Weder whisper_chunks noch whisper_json angeschlossen.")
            whisper_chunks = json.loads(whisper_json)
        timing = json.loads(timing_json)
        data = align_lyrics(timing, lyrics, whisper_chunks, abc or "", pause_zeile, min_similarity)
        text = json.dumps(data, ensure_ascii=False, indent=1)
        report = report_text(data)
        ui: Dict[str, Any] = {"text": [report]}
        if save_file:
            path, name, sub = save_json(text, filename_prefix, _output_dir())
            ui["files"] = [{"filename": name, "subfolder": sub, "type": "output"}]
            report += f"\nGespeichert: {path}"
            ui["text"] = [report]
        logger.info("[LyricAlign] %s", report.replace("\n", " | "))
        return {"ui": ui, "result": (text, report)}
