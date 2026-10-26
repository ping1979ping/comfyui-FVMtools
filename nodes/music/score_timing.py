"""Timing eines YuE2-Songs aus seiner Partitur: Abschnitte, Takte, Schlaege als JSON.

YuE2 rendert die ABC-Partitur im notierten Tempo, nur mit einem Vorlauf
(gemessen 0,3-2,5 s). ``fit_score_timing`` legt die Noteneinsaetze der
Partitur ueber die Onset-Huellkurve des Renderings und liefert Versatz und
Tempo-Faktor. Daraus folgt fuer jeden Abschnitt, Takt und Schlag die Zeit im
Audio - ohne Beat-Tracker und ohne Whisper.

Abnehmer ist die StudioSuite (Musikvideo, Schritte 4 und 5): sie liest die
Datei aus dem Prompt-Verlauf (Ausgang ``#out_timing``) und rastet Szenen an
Takte und Abschnittsgrenzen.

Die Rechnung ist dieselbe wie beim Verlaengern (``song_extend._fit`` und
``_section_spans``); hier wird sie nur vollstaendig ausgegeben.

Zweitbeste Ausrichtung: In repetitiver Musik liegt oft eine Nebenspitze einen
Schlag daneben (run02: -1 Schlag mit 89 % der besten Spitze, run01: +1 Schlag
mit 97 % - trotz Eindeutigkeit 4,4, denn die misst nur gegen andere
Tempo-Faktoren). ``fit.runner_up`` nennt sie immer; ist die Ausrichtung
unsicher (``alignment_unsure``: Eindeutigkeit < 3 oder Nebenspitze >= 90 %),
steht sie zusaetzlich mit ihren Abschnittszeiten unter ``alternative``. Welche
passt, entscheidet dann der Zeilenabgleich mit Whisper (``lyric_align``).

Gesangs-Einmessung (gekuerztes Intro): YuE2 rendert das Intro oft kuerzer
als notiert - an run01-run03 fehlten 0,7-1,6 s am Anfang, der Gesang lag 1,6-2,5 s
vor der Einsatz-Ausrichtung (gemessen mit Whisper-Zeilen). Die Einsatz-Suche
kann das nicht finden, denn sie laesst nur positive Versaetze zu und sieht in
Partituren ohne Begleitstimme (run01: Ins nur Pausen) kaum Muster.
``fit_vocal_offset`` legt deshalb zusaetzlich die Gesangsaktivitaet der
Partitur (Ton = 1, Pause = 0) ueber die Energie im Stimmband 1-4 kHz und
erlaubt negative Versaetze. Weicht sie deutlich von der Einsatz-Ausrichtung ab
und ist eindeutig, gilt sie (``alignment_method: "vocal"``); ``offset_s`` ist
dann negativ und ``intro_trimmed_s`` sagt, wie viel Partitur am Anfang fehlt.
Gemessen: run01 -0,84 s (Whisper -0,74), run02 -1,64 s (Whisper -1,55).
"""

from __future__ import annotations

import json
import logging
import os
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .abc_score import AbcError, bar_starts, onset_times, parse_score, strip_trailing_rests, tokenize_bar, walk_bars
from .audio_stitch import Timing, runner_up_timing
from .song_extend import _audio_to_np, _fit, _output_dir

logger = logging.getLogger("FVMtools.ScoreTiming")

CATEGORY = "FVMtools/music"

SCHEMA = "fvm.score_timing/1"
WEAK_CONTRAST = 3.0          # wie beim Verlaengern: darunter ist die Ausrichtung unsicher
EXPECTED_OFFSET = (0.3, 2.5)  # gemessener Vorlauf echter YuE2-Renderings
RUNNER_UP_MIN_BEATS = 0.5    # Nebenspitze erst ab einem halben Schlag Abstand (sonst dieselbe Spitze)
RUNNER_UP_CLOSE = 0.9        # Nebenspitze ab 90 % der besten: ebenfalls unsicher (run01: +1 Schlag, 97 %)
VOCAL_BAND = (1000.0, 4000.0)  # Stimmband: 300-3000 Hz trennte run02 schlechter (0,303 gegen 0,301)
VOCAL_HOP = 0.02
VOCAL_DETREND_S = 3.0        # gleitender Mittelwert raus: nur "lauter als die Umgebung" zaehlt
VOCAL_MIN_CORR = 0.2         # darunter ist die Gesangs-Einmessung nicht verwertbar
VOCAL_MAX_RATIO = 0.9        # zweite Spitze (>= 1 s daneben) muss deutlich schwaecher sein
VOCAL_SECOND_GAP_S = 1.0
VOCAL_MIN_COVER = 0.75       # mindestens 3/4 der Gesangstoene muessen im Audio liegen (sonst Zufallstreffer
                             # auf kleiner Ueberlappung: run01 sonst 0,34 bei +29 s)


def _r(x: float) -> float:
    return round(float(x), 3)


def _meter(score) -> Tuple[int, int]:
    """Taktart wie notiert (Fraction kuerzt 4/4 zu 1/1, der Schlag waere dann die Ganze)."""
    for line in score.header:
        if line.startswith("M:"):
            text = line[2:].strip()
            if text == "C":
                return 4, 4
            if text == "C|":
                return 2, 2
            m = re.match(r"^(\d+)\s*/\s*(\d+)$", text)
            if m:
                return int(m.group(1)), int(m.group(2))
    return score.meter.numerator, score.meter.denominator


def vocal_voice(score) -> str:
    """Stimme mit dem Gesang (``Vocal``), sonst die erste."""
    for v in score.voices:
        if v.lower().startswith("voc") or "vocal" in v.lower():
            return v
    return score.voices[0]


def vocal_spans(score) -> List[Tuple[float, float]]:
    """(Anfang, Ende) jedes Gesangstons in Partitur-Sekunden."""
    voice = vocal_voice(score)
    bars, _ = walk_bars(score, score.sections)
    t = Fraction(0)
    spans: List[Tuple[float, float]] = []
    for b in bars:
        if b.voice != voice:
            continue
        try:
            toks = tokenize_bar(b.bar, b.bar_units)
        except AbcError:
            t += b.bar_units
            continue
        for tok in toks:
            if tok.kind in ("note", "mrest"):
                if tok.kind == "note" and not tok.head.startswith("z"):
                    spans.append((score.units_to_seconds(t), score.units_to_seconds(t + tok.length)))
                t += tok.length
    return spans


def vocal_band_envelope(wave: np.ndarray, sr: int, hop_s: float = VOCAL_HOP) -> np.ndarray:
    """Energie im Stimmband (dB) je Hop, ohne den gleitenden Mittelwert ueber ``VOCAL_DETREND_S``."""
    mono = np.asarray(wave, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=0)
    hop = max(1, int(round(sr * hop_s)))
    n_fft = 1
    while n_fft < 2 * hop:
        n_fft *= 2
    mono = np.pad(mono, (n_fft // 2, n_fft // 2))
    frames = 1 + (mono.shape[0] - n_fft) // hop
    if frames < 2:
        return np.zeros(max(frames, 0), dtype=np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(mono[idx] * np.hanning(n_fft)[None, :], axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= VOCAL_BAND[0]) & (freqs < min(VOCAL_BAND[1], sr / 2.0))
    env = 10.0 * np.log10(spec[:, band].sum(axis=1) + 1e-9)
    k = max(1, int(VOCAL_DETREND_S / hop_s))
    env = env - np.convolve(env, np.ones(k) / k, "same")
    return (env - env.mean()).astype(np.float32)


def fit_vocal_offset(wave: np.ndarray, sr: int, score, scale: float = 1.0,
                     max_offset_s: float = 30.0) -> Optional[Dict[str, float]]:
    """Versatz (auch negativ) aus Gesangsaktivitaet der Partitur gegen das Stimmband.

    Korrelation nur ueber den Bereich, in dem Audio und Partitur ueberlappen
    (ein gekuerztes Intro liegt vor dem Audio-Anfang). Kleinster Versatz:
    der erste Gesangston liegt gerade noch im Audio.
    Rueckgabe ``{offset, corr, second_offset, second_corr, ratio}`` oder None
    (kein Gesang in der Partitur).
    """
    spans = vocal_spans(score)
    if not spans:
        return None
    env = vocal_band_envelope(wave, sr)
    hop = VOCAL_HOP
    tpl = np.zeros(int(scale * score.total_seconds() / hop) + 2)
    for a, b in spans:
        tpl[int(round(scale * a / hop)):max(int(round(scale * b / hop)), int(round(scale * a / hop)) + 1)] = 1.0
    active = np.concatenate([[0.0], np.cumsum(tpl)])
    tpl = tpl - tpl.mean()
    lo = -int(scale * spans[0][0] / hop)
    hi = int(max_offset_s / hop)
    lags = np.arange(lo, hi + 1)
    corr = np.full(len(lags), -1.0)
    for k, lag in enumerate(lags):
        a, b = max(0, lag), min(len(env), lag + len(tpl))
        if b - a < int(2.0 / hop) or active[b - lag] - active[a - lag] < VOCAL_MIN_COVER * active[-1]:
            continue
        x, y = env[a:b], tpl[a - lag:b - lag]
        den = np.linalg.norm(x) * np.linalg.norm(y)
        if den > 0:
            corr[k] = float(np.dot(x, y) / den)
    i = int(np.argmax(corr))
    far = np.abs(lags - lags[i]) * hop >= VOCAL_SECOND_GAP_S
    peak = np.zeros(len(corr), dtype=bool)
    peak[1:-1] = (corr[1:-1] >= corr[:-2]) & (corr[1:-1] >= corr[2:])
    cand = np.where(far & peak, corr, -np.inf)
    j = int(np.argmax(cand))
    second = float(cand[j]) if np.isfinite(cand[j]) else 0.0
    best = float(corr[i])
    return {"offset": float(lags[i] * hop), "corr": best,
            "second_offset": float(lags[j] * hop) if np.isfinite(cand[j]) else None,
            "second_corr": second, "ratio": second / best if best > 0 else 1.0}


def build_timing(wave: np.ndarray, sr: int, abc: str, search_seconds: float = 30.0,
                 include_beats: bool = True) -> Dict[str, Any]:
    """Partitur gegen das Rendering einmessen und alle Zeiten im Audio ausgeben.

    Zeit im Audio = offset + tempo_factor * Partitur-Zeit. Abschnitte und Takte,
    die hinter dem Audio-Ende liegen (``max_duration`` hat das Rendering
    gekappt), bleiben drin, sind aber mit ``in_audio: false`` markiert; das
    letzte hoerbare Stueck wird bei ``end_s`` auf das Audio-Ende begrenzt.
    """
    score = parse_score(abc)
    if not score.sections:
        raise ValueError("Partitur ohne Abschnitte (keine '% label'-Zeilen) - nichts einzumessen.")
    wave = np.asarray(wave, dtype=np.float32)
    duration = wave.shape[-1] / float(sr)
    timing = _fit(wave, sr, score, search_seconds)
    offset, scale = float(timing["offset"]), float(timing["scale"])
    m_num, m_den = _meter(score)
    beat_s = score.units_to_seconds(Fraction(1, m_den) / score.unit)
    runner = runner_up_timing(wave, sr, onset_times(score), score.total_seconds(), Timing(**timing),
                              RUNNER_UP_MIN_BEATS * beat_s * scale, max_offset_s=search_seconds)
    close = runner is not None and runner.contrast >= RUNNER_UP_CLOSE
    unsure = timing["contrast"] < WEAK_CONTRAST or close

    # Gesangs-Einmessung: ersetzt die Einsatz-Ausrichtung nur, wenn sie eindeutig ist und
    # mehr als einen halben Schlag (mind. 0,35 s) daneben liegt - sonst ist die Einsatz-Ausrichtung genauer
    onset_offset = offset
    vocal = fit_vocal_offset(wave, sr, score, scale, search_seconds)
    tol = max(0.5 * beat_s * scale, 0.35)
    use_vocal = (vocal is not None and vocal["corr"] >= VOCAL_MIN_CORR and vocal["ratio"] < VOCAL_MAX_RATIO
                 and abs(vocal["offset"] - onset_offset) > tol)
    if use_vocal:
        offset = vocal["offset"]
        unsure = False          # Nebenspitzen der Einsatz-Suche betreffen die verworfene Ausrichtung
    method = "vocal" if use_vocal else "onsets"
    intro_trimmed = max(0.0, -offset / scale) if scale else 0.0

    def at(score_s: float) -> float:
        return offset + scale * score_s

    durs = score.section_durations()
    score_total = float(sum(durs))
    # Schluss-Pausentakte (z. B. "z16|" am Ende) rendert YuE2 oft nicht mehr aus -
    # fehlen nur sie im Audio, ist der Song vollstaendig.
    material, _ = strip_trailing_rests(score, score.sections)
    material_total = float(sum(score.section_durations(material))) if material else score_total
    # Vollstaendig = der letzte Ton setzt im Audio noch ein (YuE2 laesst ihn oft kurz ausklingen
    # und hoert dann auf; die Ausrichtung kann zudem um einen Schlag schwanken)
    notes = [t for t, w in onset_times(score) if w >= 1.0]
    last_note = notes[-1] if notes else material_total

    # Takte: Anfaenge (erste Stimme) ueber alle Abschnitte, mit Abschnittszuordnung
    sec_bounds, t = [], 0.0
    for d in durs:
        sec_bounds.append((t, t + d))
        t += d
    starts = bar_starts(score, score.sections)          # inkl. Ende als letzter Eintrag
    bars: List[Dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        sec_i = next((k for k, (s0, s1) in enumerate(sec_bounds) if s0 - 1e-6 <= a < s1 - 1e-6),
                     len(sec_bounds) - 1)
        bars.append({"index": i, "section": sec_i, "start_s": _r(at(a)), "end_s": _r(at(b)),
                     "score_s": _r(a), "in_audio": at(a) < duration and at(b) > 0})

    sections: List[Dict[str, Any]] = []
    for k, (sec, (s0, s1)) in enumerate(zip(score.sections, sec_bounds)):
        a, b = at(s0), at(s1)
        sec_bars = [x["index"] for x in bars if x["section"] == k]
        sections.append({
            "index": k, "label": sec.label, "start_s": _r(max(a, 0.0)), "end_s": _r(max(min(b, duration), 0.0)),
            "score_start_s": _r(s0), "score_end_s": _r(s1),
            "bars": len(sec_bars), "first_bar": sec_bars[0] if sec_bars else None,
            "in_audio": a < duration and b > 0, "truncated": b > duration + 1e-3,
            "trimmed_s": _r(max(0.0, min(b, 0.0) - a)),     # vorne fehlend (gekuerztes Intro)
        })

    beats: Optional[List[Dict[str, Any]]] = None
    if include_beats and beat_s > 0:
        beats = []
        for bar, (a, b) in zip(bars, zip(starts[:-1], starts[1:])):
            n = max(1, int(round((b - a) / beat_s)))
            for j in range(n):
                t_s = at(a + j * beat_s)
                if t_s >= duration:
                    break
                if t_s < 0:                                 # gekuerztes Intro: Schlag liegt vor dem Audio
                    continue
                beats.append({"bar": bar["index"], "beat": j + 1, "t": _r(t_s)})

    # Zweitbeste Ausrichtung: Abstand in der Mitte der Partitur (wie bei der Suche)
    runner_up: Optional[Dict[str, Any]] = None
    if runner is not None:
        mid = score_total / 2.0
        shift = (runner.offset + runner.scale * mid) - (onset_offset + scale * mid)   # gegen die Einsatz-Spitze
        runner_up = {"offset_s": _r(runner.offset), "tempo_factor": round(runner.scale, 5),
                     "corr": round(runner.corr, 3), "ratio": round(runner.contrast, 3),
                     "shift_s": _r(shift), "shift_beats": round(shift / (beat_s * scale), 2)}
    alternative: Optional[Dict[str, Any]] = None
    if unsure and runner_up is not None:
        alt_at = runner.to_audio
        alternative = dict(runner_up)
        alternative["sections"] = [
            {"index": s["index"], "label": s["label"], "start_s": _r(alt_at(s0)),
             "end_s": _r(min(alt_at(s1), duration)), "in_audio": alt_at(s0) < duration}
            for s, (s0, s1) in zip(sections, sec_bounds)]

    warnings: List[str] = []
    if use_vocal:
        warnings.append(f"Gesangs-Einmessung statt Einsatz-Ausrichtung: Versatz {offset:.2f} s statt "
                        f"{onset_offset:.2f} s (Gesang {vocal['corr']:.2f}, Nebenspitze "
                        f"{vocal['ratio'] * 100:.0f} %)"
                        + (f"; Intro um {intro_trimmed:.2f} s gekuerzt gerendert." if intro_trimmed > 0 else "."))
    elif timing["contrast"] < WEAK_CONTRAST:
        warnings.append(f"Ausrichtung unsicher (Eindeutigkeit {timing['contrast']} < {WEAK_CONTRAST}).")
    elif close:
        warnings.append(f"Ausrichtung unsicher (Nebenspitze {runner.contrast * 100:.0f} % >= "
                        f"{RUNNER_UP_CLOSE * 100:.0f} % der besten).")
    if unsure:
        if runner_up is not None:
            warnings.append(f"Zweitbeste Ausrichtung: Versatz {runner_up['offset_s']:.2f} s "
                            f"({runner_up['shift_beats']:+.1f} Schlag), {runner_up['ratio'] * 100:.0f} % "
                            f"der besten Spitze - siehe 'alternative', Wahl per Whisper-Zeilenabgleich.")
    if not use_vocal and not EXPECTED_OFFSET[0] <= offset <= EXPECTED_OFFSET[1]:
        warnings.append(f"Versatz {offset:.2f} s liegt ausserhalb des ueblichen YuE2-Vorlaufs "
                        f"{EXPECTED_OFFSET[0]}-{EXPECTED_OFFSET[1]} s.")
    if abs(scale - 1.0) > 0.01:
        warnings.append(f"Tempo-Faktor {scale:.3f} weicht von 1,000 ab (YuE2 haelt sonst das Tempo).")
    complete = at(last_note) + 0.1 <= duration
    if not complete:
        cut = sum(1 for s in sections if s["truncated"])
        warnings.append(f"Audio endet vor dem letzten Ton der Partitur ({duration:.1f} s, letzter "
                        f"Einsatz bei {at(last_note):.1f} s): {cut} Abschnitt(e) gekappt oder fehlend - "
                        f"max_duration zu klein oder YuE2 hat frueh aufgehoert.")

    return {
        "schema": SCHEMA,
        "tempo_bpm": round(score.tempo_bpm, 3),
        "tempo_beat": str(score.tempo_beat),
        "effective_bpm": round(score.tempo_bpm / scale, 3) if scale else None,
        "meter": f"{m_num}/{m_den}",
        "key": score.key,
        "beat_s": _r(beat_s * scale),
        "bar_s": _r(score.seconds_per_bar() * scale),
        "offset_s": _r(offset),
        "tempo_factor": round(scale, 5),
        "fit": {"corr": timing["corr"], "contrast": timing["contrast"], "search_seconds": search_seconds,
                "runner_up": runner_up,
                "onset_offset_s": _r(onset_offset),
                "vocal": None if vocal is None else {
                    "offset_s": _r(vocal["offset"]), "corr": round(vocal["corr"], 3),
                    "ratio": round(vocal["ratio"], 3),
                    "second_offset_s": _r(vocal["second_offset"]) if vocal["second_offset"] is not None else None,
                    "used": use_vocal}},
        "alignment_method": method,
        "intro_trimmed_s": _r(intro_trimmed),
        "alignment_unsure": unsure,
        "alternative": alternative,
        "audio_duration_s": _r(duration),
        "sample_rate": int(sr),
        "score_seconds": _r(score_total),
        "music_end_s": _r(at(score_total)),
        "material_end_s": _r(at(material_total)),
        "last_note_s": _r(at(last_note)),
        "complete": complete,
        "sections": sections,
        "bars": bars,
        "beats": beats,
        "warnings": warnings,
    }


def report_text(data: Dict[str, Any]) -> str:
    """Kurzbericht fuer die Node-Anzeige."""
    lines = [f"Tempo {data['tempo_bpm']} BPM ({data['meter']}), Versatz {data['offset_s']} s, "
             f"Tempo-Faktor x{data['tempo_factor']}, Eindeutigkeit {data['fit']['contrast']}",
             f"Audio {data['audio_duration_s']} s, Musik laut Partitur bis {data['music_end_s']} s, "
             f"{len(data['bars'])} Takte" + (f", {len(data['beats'])} Schlaege" if data.get("beats") else "")]
    for s in data["sections"]:
        if not s["in_audio"]:
            mark = "  (fehlt)"
        elif s["truncated"]:
            mark = "  (nur Schlusspause gekappt)" if data["complete"] else "  (gekappt)"
        else:
            mark = ""
        lines.append(f"  {s['index'] + 1}. {s['label']}: {s['start_s']:.2f}-{s['end_s']:.2f} s, "
                     f"{s['bars']} Takte{mark}")
    if data.get("alignment_method") == "vocal":
        lines.append(f"Gesangs-Einmessung: Einsatz-Ausrichtung {data['fit']['onset_offset_s']} s verworfen, "
                     f"Intro um {data['intro_trimmed_s']} s gekuerzt")
    runner_up = data["fit"].get("runner_up")
    if runner_up:
        lines.append(f"Zweitbeste Spitze: Versatz {runner_up['offset_s']} s ({runner_up['shift_beats']:+} Schlag), "
                     f"{runner_up['ratio'] * 100:.0f} %")
    lines += ["WARNUNG: " + w for w in data["warnings"]]
    return "\n".join(lines)


def save_json(text: str, filename_prefix: str, output_dir: str) -> Tuple[str, str, str]:
    """``<prefix>_00001.json`` im Ausgabeordner, Zaehler wie bei SaveText.

    Rueckgabe (voller Pfad, Dateiname, Unterordner relativ zu ``output_dir``).
    Ein Praefix, der aus dem Ausgabeordner herausfuehrt, wird abgelehnt.
    """
    prefix = (filename_prefix or "timing").replace("\\", "/").strip("/")
    sub, base = os.path.split(prefix)
    base = base or "timing"
    root = os.path.abspath(output_dir)
    folder = os.path.abspath(os.path.join(root, sub))
    if os.path.commonpath([root, folder]) != root:
        raise ValueError(f"filename_prefix '{filename_prefix}' zeigt aus dem Ausgabeordner heraus.")
    os.makedirs(folder, exist_ok=True)
    pat = re.compile(re.escape(base) + r"_(\d{5})\.json$")
    used = [int(m.group(1)) for f in os.listdir(folder) for m in [pat.match(f)] if m]
    name = f"{base}_{(max(used) + 1 if used else 1):05}.json"
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    rel = os.path.relpath(folder, root).replace("\\", "/")
    return path, name, "" if rel == "." else rel


class FVM_ScoreTimingExport:
    """Abschnitte, Takte und Schlaege eines YuE2-Songs aus seiner Partitur, als JSON-Datei."""

    CATEGORY = CATEGORY
    FUNCTION = "export"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("timing_json", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Das YuE2-Rendering (ungemastert, nicht geschnitten)."}),
                "abc": ("STRING", {"multiline": True, "default": "",
                                   "tooltip": "Die Partitur, aus der genau dieses Audio gerendert wurde "
                                              "(Ausgang von YuE2 Generate ABC)."}),
                "filename_prefix": ("STRING", {"default": "audio/music/timing/song",
                                               "tooltip": "Datei: <prefix>_00001.json im ComfyUI-Ausgabeordner."}),
                "search_seconds": ("FLOAT", {"default": 30.0, "min": 2.0, "max": 120.0, "step": 1.0,
                                             "tooltip": "Maximaler Vorlauf von YuE2 vor der ersten Note."}),
                "include_beats": ("BOOLEAN", {"default": True,
                                              "tooltip": "Auch jeden Schlag ausgeben (sonst nur Takte)."}),
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    def export(self, audio, abc, filename_prefix, search_seconds=30.0, include_beats=True, save_file=True):
        wave, sr = _audio_to_np(audio)
        data = build_timing(wave, sr, abc, search_seconds, include_beats)
        text = json.dumps(data, ensure_ascii=False, indent=1)
        report = report_text(data)
        ui: Dict[str, Any] = {"text": [report]}
        if save_file:
            path, name, sub = save_json(text, filename_prefix, _output_dir())
            ui["files"] = [{"filename": name, "subfolder": sub, "type": "output"}]
            report += f"\nGespeichert: {path}"
            ui["text"] = [report]
        logger.info("[ScoreTiming] %s", report.replace("\n", " | "))
        return {"ui": ui, "result": (text, report)}
