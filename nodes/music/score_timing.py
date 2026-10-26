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
"""

from __future__ import annotations

import json
import logging
import os
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .abc_score import bar_starts, onset_times, parse_score, strip_trailing_rests
from .song_extend import _audio_to_np, _fit, _output_dir

logger = logging.getLogger("FVMtools.ScoreTiming")

CATEGORY = "FVMtools/music"

SCHEMA = "fvm.score_timing/1"
WEAK_CONTRAST = 3.0          # wie beim Verlaengern: darunter ist die Ausrichtung unsicher
EXPECTED_OFFSET = (0.3, 2.5)  # gemessener Vorlauf echter YuE2-Renderings


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
                     "score_s": _r(a), "in_audio": at(a) < duration})

    sections: List[Dict[str, Any]] = []
    for k, (sec, (s0, s1)) in enumerate(zip(score.sections, sec_bounds)):
        a, b = at(s0), at(s1)
        sec_bars = [x["index"] for x in bars if x["section"] == k]
        sections.append({
            "index": k, "label": sec.label, "start_s": _r(a), "end_s": _r(min(b, duration)),
            "score_start_s": _r(s0), "score_end_s": _r(s1),
            "bars": len(sec_bars), "first_bar": sec_bars[0] if sec_bars else None,
            "in_audio": a < duration, "truncated": b > duration + 1e-3,
        })

    beats: Optional[List[Dict[str, Any]]] = None
    m_num, m_den = _meter(score)
    beat_s = score.units_to_seconds(Fraction(1, m_den) / score.unit)
    if include_beats and beat_s > 0:
        beats = []
        for bar, (a, b) in zip(bars, zip(starts[:-1], starts[1:])):
            n = max(1, int(round((b - a) / beat_s)))
            for j in range(n):
                t_s = at(a + j * beat_s)
                if t_s >= duration:
                    break
                beats.append({"bar": bar["index"], "beat": j + 1, "t": _r(t_s)})

    warnings: List[str] = []
    if timing["contrast"] < WEAK_CONTRAST:
        warnings.append(f"Ausrichtung unsicher (Eindeutigkeit {timing['contrast']} < {WEAK_CONTRAST}).")
    if not EXPECTED_OFFSET[0] <= offset <= EXPECTED_OFFSET[1]:
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
        "fit": {"corr": timing["corr"], "contrast": timing["contrast"], "search_seconds": search_seconds},
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
