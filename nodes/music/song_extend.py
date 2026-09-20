"""Song verlaengern, Stueck fuer Stueck: Quelle -> Session -> Plan -> Pruefen -> Anhaengen -> Zusammenfuegen.

Ein Queue-Lauf = ein neuer Teil. Die Partitur des fertigen Songs (YuE2-ABC)
wird um Abschnitte ergaenzt - vom LLM neu geschrieben, als Variation oder
einfach wiederholt (Loop). Gerendert wird jeweils nur der neue Teil plus ein
Abschnitt Ueberlapp; der Ueberlapp dient als Anker, um den Teil sauber an das
bisherige Audio zu setzen. YuE2 kann eine immer laengere Partitur nicht am
Stueck rendern (Kontextfenster), gestueckelt geht es beliebig weit.

Rendering und LLM sind normale Nodes im Workflow (YuE2 core, LLM-Chat des
Toolkits); diese Nodes liefern ihnen Partitur, Lyrics-Tags und Prompts und
nehmen das Ergebnis wieder an. Sie importieren nichts aus anderen Extensions.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from fractions import Fraction
from typing import Any, Dict, List, Tuple

import numpy as np

from .abc_score import (
    AbcError,
    RepairReport,
    Score,
    Section,
    extract_abc_block,
    format_score,
    bar_starts,
    format_section,
    onset_times,
    outline,
    parse_body,
    parse_score,
    repair_sections,
    section_tag,
    select_sections,
    silence_voice,
    smooth_final_bar,
    strip_trailing_rests,
    validate_sections,
    voice_is_silent,
    with_sections,
)
from .audio_stitch import (
    arrangement_segments,
    choose_cut,
    fit_score_timing,
    refine_cut,
    join_segments,
    body_level_db,
    reference_gain,
    rms,
    parse_arrangement,
    resample,
    to_stereo,
)
from .melody_template import build_template, parse_llm_sections, skeleton, template_brief
from . import extend_session as es

logger = logging.getLogger("FVMtools.SongExtend")

CATEGORY = "FVMtools/music/extend"

ACTIONS = ["anhaengen", "letzten Teil entfernen", "zuruecksetzen", "nur zusammenfuegen"]
MODES = ["LLM: neuer Abschnitt", "LLM: Variation", "Loop: Abschnitte wiederholen"]
LABELS = ["auto", "verse", "chorus", "pre-chorus", "bridge", "solo", "breakdown", "build-up", "outro"]
MANUAL = "(manuell - Eingaenge unten)"


# ---------------------------------------------------------------- Hilfen

def _audio_to_np(audio: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    wave = audio["waveform"]
    if hasattr(wave, "detach"):
        wave = wave.detach().float().cpu().numpy()
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 3:
        wave = wave[0]
    return to_stereo(wave), int(audio["sample_rate"])


def _np_to_audio(wave: np.ndarray, sr: int) -> Dict[str, Any]:
    import torch
    return {"waveform": torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))[None], "sample_rate": int(sr)}


def _vocal_voice(score: Score) -> str:
    for v in score.voices:
        if "voc" in v.lower():
            return v
    return score.voices[0]


def _output_dir() -> str:
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return os.path.join(os.getcwd(), "output")


# ---------------------------------------------------------------- Quelle

def scan_production_logs(root: str, limit: int = 300) -> List[str]:
    """Production-JSONs des Music Production Toolkits, neueste zuerst (relativ zu root)."""
    found = []
    base = os.path.join(root, "audio", "music")
    for dirpath, _dirs, files in os.walk(base):
        if os.path.basename(dirpath) != "log":
            continue
        for f in files:
            if f.lower().endswith(".json"):
                full = os.path.join(dirpath, f)
                try:
                    found.append((os.path.getmtime(full), os.path.relpath(full, root)))
                except OSError:
                    pass
    found.sort(reverse=True)
    return [p.replace("\\", "/") for _, p in found[:limit]]


def read_production_json(path: str) -> Dict[str, str]:
    """ABC, Style, Lyrics, Titel und Original-Audio aus einem Toolkit-Production-JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    gen = data.get("generation") or data.get("models") or {}
    cond = gen.get("cover_conditioning") or {}
    abc = gen.get("abc") or cond.get("native_abc") or ""
    style = cond.get("native_style") or data.get("style") or data.get("caption") or ""
    lyrics = cond.get("native_lyrics") or data.get("lyrics") or ""
    outputs = data.get("outputs") or {}
    audio_path = ((outputs.get("original_audio") or {}).get("path")
                  or (outputs.get("release_flac") or {}).get("path") or "")
    return {"abc": abc, "style": style, "lyrics": lyrics, "title": data.get("title") or "",
            "audio_path": audio_path, "seed": str(data.get("generation_seed") or "")}


class FVM_SongExtendSource:
    """Ausgangs-Song laden: am einfachsten aus einem Production-JSON des Toolkits."""

    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "abc", "style", "lyrics", "title", "info")

    @classmethod
    def INPUT_TYPES(cls):
        logs = scan_production_logs(_output_dir())
        return {
            "required": {
                "production_json": ([MANUAL] + logs, {
                    "default": logs[0] if logs else MANUAL,
                    "tooltip": "Log-JSON eines fertigen Toolkit-Laufs (output/audio/music/.../log). "
                               "Liefert die exakt gerenderte Partitur, den Style und das "
                               "ungemasterte Original-Audio."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Ueberschreibt das Audio aus dem JSON."}),
                "abc_override": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": "Eigene YuE2-ABC-Partitur (leer = aus JSON)."}),
                "style_override": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "Eigener Style (leer = aus JSON)."}),
                "lyrics_override": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, production_json, **kwargs):
        path = os.path.join(_output_dir(), production_json)
        try:
            return f"{production_json}:{os.path.getmtime(path)}"
        except OSError:
            return production_json

    def load(self, production_json, audio=None, abc_override="", style_override="", lyrics_override=""):
        info = {}
        if production_json != MANUAL:
            info = read_production_json(os.path.join(_output_dir(), production_json))
        abc = (abc_override or "").strip() or info.get("abc", "")
        style = (style_override or "").strip() or info.get("style", "")
        lyrics = (lyrics_override or "").strip() or info.get("lyrics", "")
        title = info.get("title") or "song"
        if not abc:
            raise ValueError("Keine Partitur: Production-JSON waehlen oder abc_override fuellen.")
        parse_score(abc)  # frueh scheitern, wenn das keine YuE2-Partitur ist
        if audio is None:
            path = info.get("audio_path", "")
            if not path or not os.path.exists(path):
                raise ValueError(f"Audio nicht gefunden ({path!r}). Audio-Eingang verbinden.")
            wave, sr = es.read_audio(path)
            audio = _np_to_audio(to_stereo(wave), sr)
        wave, sr = _audio_to_np(audio)
        text = (f"Quelle: {production_json}\nTitel: {title}\nAudio: {wave.shape[-1] / sr:.1f} s @ {sr} Hz\n"
                f"Partitur: {len(abc)} Zeichen, {len(parse_score(abc).sections)} Abschnitte")
        return (audio, abc, style, lyrics, title, text)


# ---------------------------------------------------------------- Session

class FVM_SongExtendSession:
    """Oeffnet die Session und fuehrt Rueckgaengig/Zuruecksetzen aus."""

    CATEGORY = CATEGORY
    FUNCTION = "open"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("session_json", "full_abc", "overview", "parts")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_audio": ("AUDIO",),
                "base_abc": ("STRING", {"forceInput": True}),
                "style": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "title": ("STRING", {"forceInput": True}),
                "action": (ACTIONS, {"default": ACTIONS[0], "tooltip":
                           "anhaengen: dieser Queue-Lauf haengt einen Teil an. "
                           "letzten Teil entfernen / zuruecksetzen: wirkt sofort, rendert nichts. "
                           "nur zusammenfuegen: nichts aendern, nur das Ergebnis neu bauen."}),
                "session_name": ("STRING", {"default": "", "tooltip":
                                 "Leer = Songtitel. Verschiedene Namen = unabhaengige Versionen."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")   # jeder Queue-Lauf ist ein Schritt

    def open(self, base_audio, base_abc, style, lyrics, title, action, session_name=""):
        name = es.safe_name(session_name or title)
        directory = es.session_dir(name)
        wave, sr = _audio_to_np(base_audio)
        state = es.init_or_load(directory, name, base_abc, wave, sr, style, lyrics, title, "source")
        msg = ""
        if action == "letzten Teil entfernen":
            msg = es.undo_last(directory, state)
        elif action == "zuruecksetzen":
            msg = es.reset(directory, state)
        state = es.load_state(directory)
        score = parse_score(state["full_abc"])
        lines = [f"Session '{name}' - {len(state['parts']) - 1} angehaengte(r) Teil(e)"]
        if msg:
            lines.append(msg)
        lines.append(f"Partitur: {round(score.total_seconds(), 1)} s")
        for o in outline(score):
            lines.append(f"  {o['index']:>2}. {o['label']:<12} {o['bars']:>3} Takte  {o['seconds']:>6.1f} s")
        payload = {"dir": directory, "name": name, "action": action,
                   "parts": len(state["parts"]), "abc_sha": es.sha(state["full_abc"]),
                   "nonce": time.time()}
        return (json.dumps(payload), state["full_abc"], "\n".join(lines), len(state["parts"]))


# ---------------------------------------------------------------- Plan

NEXT_LABEL = {"intro": "verse", "verse": "chorus", "pre-chorus": "chorus", "chorus": "verse",
              "bridge": "chorus", "solo": "chorus", "breakdown": "build-up", "build-up": "chorus"}


def plan_labels(score: Score, count: int, label: str) -> List[str]:
    if label != "auto":
        return [label] * count
    labels, last = [], (score.sections[-1].label if score.sections else "verse").lower()
    for _ in range(count):
        last = NEXT_LABEL.get(last, "verse")
        labels.append(last)
    return labels


# Gemessen: YuE2 beginnt ein Rendering oft wie ein Intro (bis ~14 s bei -30 dB statt -17 dB)
# und spielt sein Ende wie einen Schluss. Geschnitten wird deshalb frueh genug vor dem
# Ueberlapp-Ende und spaet genug nach dem Render-Beginn - dafuer muss der Ueberlapp lang sein.
SETTLE_S = 12.0
MIN_OVERLAP_S = 22.0
MIN_SEGMENT_S = 4.0


def auto_overlap(score: Score, requested: int) -> int:
    """Anzahl Ueberlapp-Abschnitte: gewuenscht, oder (0) so viele, bis >= MIN_OVERLAP_S."""
    n_max = min(3, len(score.sections))
    if requested > 0:
        return max(1, min(requested, n_max))
    durs = score.section_durations()
    n, total = 0, 0.0
    while n < n_max and total < MIN_OVERLAP_S:
        n += 1
        total += durs[-n]
    return max(1, n)


def open_score(full_abc: str) -> Tuple[Score, int]:
    """Partitur ohne Schluss-Pausentakte - an diesem Ende wird weitergeschrieben."""
    score = parse_score(full_abc)
    secs, removed = strip_trailing_rests(score, score.sections)
    secs, _ = smooth_final_bar(score, secs)
    return with_sections(score, secs), removed


def template_sources(score: Score, mode: str, labels: List[str], source_spec: str) -> List[int]:
    """0-basierter Index des Vorlagen-Abschnitts je neuem Abschnitt.

    Variation: die gewaehlten Abschnitte (zyklisch). Neu: der letzte Abschnitt
    mit demselben Label, sonst der letzte - der Rhythmus passt dann zur Rolle.
    """
    if mode == MODES[1]:
        chosen = select_sections(score, source_spec or "last")
        idx = [next(i for i, s in enumerate(score.sections) if s is c) for c in chosen]
        return [idx[i % len(idx)] for i in range(len(labels))]
    out = []
    for label in labels:
        match = [i for i, s in enumerate(score.sections) if s.label.lower() == label.lower()]
        out.append(match[-1] if match else len(score.sections) - 1)
    return out


SYSTEM_PROMPT = """You are a composer extending an existing song. You do NOT write rhythm or note
lengths - a rhythm template from the song fixes them. You choose PITCHES and CHORDS.

Answer with ONE JSON object and nothing else:
{{"sections": [ {{"label": "<label>", "chords": ["<chord bar 1>", ...], {voice_keys} }} ]}}

RULES
- One entry in "sections" per requested section, in the requested order.
- For every voice listed, give one list per bar, and in each bar EXACTLY as many pitches as
  "notes per bar" says for that bar (a bar with 0 notes gets []).
- Pitches are written in scientific pitch notation as sounding notes: letter, optional # or b,
  octave number - e.g. "Ab4", "C5", "Eb5", "F#4". C4 is middle C. The song is in {key_name};
  the reference pitches show which notes belong to it.
- Stay inside the song's range shown for each voice. Prefer stepwise motion with a few leaps,
  land on chord tones on strong beats, and give the melody a clear shape and a hook.
- "chords": one chord symbol per bar in the same naming style as the song (e.g. {chord_examples}).
{vocal_rule}"""

VOCAL_SILENT_RULE = ("- The {vocal} voice is silent in this song; the melody is played by {ins}. "
                     "Only write pitches for the voices listed.")
VOCAL_ACTIVE_RULE = "- Both voices carry music, as in the song. Keep their roles."

MODE_TASK = {
    MODES[0]: ("Write a NEW melody for each section: it must belong to this song (same scale, "
               "similar motifs and energy) but must not copy the reference pitches. You may "
               "re-harmonise tastefully; the chord change rhythm should stay close to the reference."),
    MODES[1]: ("Write a VARIATION of each reference section: keep the chords (or change at most a "
               "few), keep the melodic outline recognisable, but vary the pitches clearly - "
               "ornaments, answers an octave higher, new turns at phrase ends."),
}


def _key_name(key: str) -> str:
    m = re.match(r"^\s*([A-Ga-g][#b]?)\s*([A-Za-z]*)", key or "C")
    if not m:
        return key
    tonic = m.group(1)[0].upper() + m.group(1)[1:]
    mode = "minor" if m.group(2).lower() in ("m", "min", "minor") else (m.group(2) or "major")
    return f"{tonic} {mode}"


def build_plan(full_abc: str, mode: str, count: int, label: str, bars: int, source_spec: str,
               instructions: str, overlap: int) -> Dict[str, Any]:
    score, _ = open_score(full_abc)
    if not score.sections:
        raise AbcError("Die Partitur hat keine Abschnitte.")
    vocal = _vocal_voice(score)
    others = [v for v in score.voices if v != vocal]
    silent = voice_is_silent(score, vocal)
    needs_llm = not mode.startswith("Loop")
    labels = plan_labels(score, count, label) if needs_llm else []
    plan = {"mode": mode, "labels": labels, "bars": bars, "source_spec": source_spec,
            "overlap": auto_overlap(score, overlap), "vocal": vocal,
            "vocal_silent": silent, "full_abc": full_abc, "needs_llm": needs_llm,
            "count": count, "system_prompt": "", "user_prompt": ""}
    if not needs_llm:
        return plan

    sources = template_sources(score, mode, labels, source_spec)
    plan["template_sources"] = sources
    briefs, voice_keys, templates = [], set(), []
    for label_i, src_i in zip(labels, sources):
        tpl = build_template(score, score.sections[src_i], label_i, bars)
        templates.append(tpl)
        voice_keys.update(tpl.melody_voices)
        briefs.append(f'--- new section {len(briefs) + 1}: label "{label_i}" '
                      f'(rhythm from song section {src_i + 1} "{score.sections[src_i].label}")\n'
                      + template_brief(tpl))
    chords = []
    for s in score.sections:
        for g in s.groups:
            for v in score.voices:
                chords += re.findall(r'"([^"]+)"', g.get(v, ""))
    examples = ", ".join(f'"{c}"' for c in dict.fromkeys(chords)) or '"C", "G7", "Am"'
    keys = ", ".join(f'"{v}": [[...bar 1 pitches...], ...]' for v in sorted(voice_keys))
    plan["system_prompt"] = SYSTEM_PROMPT.format(
        voice_keys=keys, key_name=_key_name(score.key), chord_examples=examples,
        vocal_rule=(VOCAL_SILENT_RULE.format(vocal=vocal, ins=", ".join(others) or "the other voice")
                    if silent else VOCAL_ACTIVE_RULE))
    shape = ", ".join(f"{o['label']} ({o['bars']} bars)" for o in outline(score))
    user = [f"SONG: key {score.key}, tempo {round(score.tempo_bpm)} bpm, form so far: {shape}.",
            f"\nTASK: {MODE_TASK[mode]}",
            f"\nWrite {len(labels)} section(s): " + ", ".join(f'"{l}"' for l in labels) + ".",
            "\n" + "\n\n".join(briefs)]
    if (instructions or "").strip():
        user.append(f"\nADDITIONAL DIRECTION: {instructions.strip()}")
    user.append(f"\nFill in this exact structure (replace every ? - keep the list lengths), "
                f"reply with the JSON only:\n{skeleton(templates)}")
    plan["user_prompt"] = "\n".join(user)
    return plan


class FVM_SongExtendPlan:
    """Legt fest, was angehaengt wird, und baut die LLM-Prompts dafuer."""

    CATEGORY = CATEGORY
    FUNCTION = "plan"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("system_prompt", "user_prompt", "plan_json", "needs_llm")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_json": ("STRING", {"forceInput": True}),
                "full_abc": ("STRING", {"forceInput": True}),
                "mode": (MODES, {"default": MODES[0], "tooltip":
                         "LLM neu: neue Melodie/Akkorde ueber den Rhythmus des letzten Abschnitts "
                         "mit gleichem Label. LLM Variation: variiert source_sections. "
                         "Loop: wiederholt source_sections ohne LLM (wird neu eingespielt). "
                         "Rhythmus und Taktlaengen kommen immer aus dem Song - das LLM waehlt nur Toene."}),
                "sections_to_add": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip":
                                    "Abschnitte pro Queue-Lauf (bei Loop: Wiederholungen)."}),
                "section_label": (LABELS, {"default": "auto"}),
                "bars_per_section": ("INT", {"default": 0, "min": 0, "max": 32, "tooltip":
                                     "0 = wie der letzte Abschnitt mit diesem Label."}),
                "source_sections": ("STRING", {"default": "last", "tooltip":
                                    "Fuer Loop/Variation: last, last2, chorus, 3, 2-4 (1-basiert)."}),
                "instructions": ("STRING", {"multiline": True, "default": "", "tooltip":
                                 "Freie Regie fuer das LLM, z. B. 'steigern, Hook im hohen Register'."}),
                "overlap_sections": ("INT", {"default": 0, "min": 0, "max": 3, "tooltip":
                                     "So viele Abschnitte des bisherigen Endes werden mitgerendert "
                                     "(Anker fuer die Ausrichtung). 0 = automatisch, mindestens "
                                     f"{MIN_OVERLAP_S:.0f} s - YuE2 braucht am Render-Anfang Anlauf."}),
            },
        }

    def plan(self, session_json, full_abc, mode, sections_to_add, section_label, bars_per_section,
             source_sections, instructions, overlap_sections):
        plan = build_plan(full_abc, mode, sections_to_add, section_label, bars_per_section,
                          source_sections, instructions, overlap_sections)
        plan["session"] = json.loads(session_json)
        return (plan["system_prompt"], plan["user_prompt"], json.dumps(plan), plan["needs_llm"])


# ---------------------------------------------------------------- Pruefen & uebernehmen

def _copy_sections(sections: List[Section], labels: List[str] = None) -> List[Section]:
    out = []
    for i, s in enumerate(sections):
        out.append(Section(label=(labels[i] if labels and i < len(labels) else s.label),
                           groups=[dict(g) for g in s.groups]))
    return out


def _llm_sections(score: Score, plan: Dict[str, Any], llm_text: str, notes: List[str],
                  snap_to_key: bool = True) -> Tuple[List[Section], str]:
    """LLM-Toene in die Vorlagen setzen. Rueckgabe (Abschnitte, Quelle)."""
    templates = [build_template(score, score.sections[i], label, plan.get("bars", 0))
                 for label, i in zip(plan["labels"], plan["template_sources"])]
    for t in templates:
        notes += t.notes
    try:
        entries = parse_llm_sections(llm_text)
    except AbcError as exc:
        notes.append(f"LLM-Antwort unbrauchbar ({exc}) -> Vorlage unveraendert (Loop).")
        return [t.fill({}, None)[0] for t in templates], "fallback"
    out, used = [], 0
    for i, tpl in enumerate(templates):
        entry = entries[i] if i < len(entries) and isinstance(entries[i], dict) else None
        if entry is None:
            notes.append(f"Abschnitt {i + 1}: vom LLM nicht geliefert -> Vorlage unveraendert.")
            out.append(tpl.fill({}, None)[0])
            continue
        lower = {str(k).lower(): v for k, v in entry.items()}
        melodies = {}
        for v in tpl.melody_voices:
            val = lower.get(v.lower())
            if val is None and len(tpl.melody_voices) == 1:
                val = lower.get("melody") or lower.get("pitches")
            if isinstance(val, list):
                melodies[v] = val
        chords = lower.get("chords") if isinstance(lower.get("chords"), list) else None
        section, fill_notes = tpl.fill(melodies, chords, snap_to_key)
        notes += [f"{tpl.label}: {n}" for n in fill_notes]
        if melodies:
            used += 1
        out.append(section)
    return out, ("llm" if used else "fallback")


def _ending_section(score: Score, bars: int) -> Section:
    """Abschnitt 'outro' aus reinen Pausentakten (Ausklang im Render)."""
    rest = f"z{score.bar_units}" if score.bar_units != 1 else "z"
    return Section(label="outro", groups=[{v: (rest + "|") * bars for v in score.voices}])


def apply_plan(plan: Dict[str, Any], llm_text: str = "", snap_to_key: bool = True) -> Dict[str, Any]:
    """Neue Abschnitte bestimmen (LLM-Toene in Vorlage oder Loop), pruefen, Render-Partitur bauen."""
    score, stripped = open_score(plan["full_abc"])
    meters = score.end_meters()
    notes: List[str] = []
    if stripped:
        notes.append(f"{stripped} Schluss-Pausentakt(e) entfernt - dort geht es weiter.")

    if plan["needs_llm"]:
        new, source = _llm_sections(score, plan, llm_text, notes, snap_to_key)
    else:
        source = "loop"
        try:
            block = select_sections(score, plan.get("source_spec") or "last")
        except AbcError as exc:
            notes.append(f"{exc} -> letzter Abschnitt")
            block = score.sections[-1:]
        new = []
        for _ in range(max(1, plan["count"])):
            new += _copy_sections(block)

    if plan.get("vocal_silent"):
        new, changed = silence_voice(score, new, plan["vocal"])
        if changed:
            notes.append(f"{changed} Noten in {plan['vocal']} zu Pausen gemacht (Instrumental bleibt instrumental).")

    # Sicherheitsnetz: Takte, die trotz Vorlage nicht stimmen, reparieren
    rep = RepairReport()
    new = repair_sections(score, new, meters, rep)
    if rep.bars_repaired:
        notes.append(f"{rep.bars_repaired} Takt(e) nachrepariert: " + "; ".join(rep.notes[:4]))
    errors = validate_sections(score, new, meters)

    overlap = score.sections[-plan["overlap"]:]
    # Ausklang: hinter dem neuen Material Pausentakte, damit YuE2 natuerlich ausklingt
    # statt am letzten Ton abzubrechen. Nur im Render - die gespeicherte Partitur
    # bleibt ohne, denn dort geht es beim naechsten Teil weiter.
    render_sections = overlap + new + [_ending_section(score, max(2, stripped))]
    render_abc = format_score(with_sections(score, render_sections))
    durs_overlap = score.section_durations(overlap, score.end_meters(before=overlap[0]))
    durs_new = score.section_durations(new, meters)
    full_after = format_score(with_sections(score, score.sections + new))
    instrumental = bool(plan.get("vocal_silent"))
    lyrics = "\n\n".join(section_tag(s.label, instrumental) for s in render_sections)
    overlap_s, new_s = float(sum(durs_overlap)), float(sum(durs_new))
    return {
        "source": source,
        "labels": [s.label for s in new],
        "mode": plan["mode"],
        "sections_added": len(new),
        "new_abc": "\n".join(format_section(s, score.voices) for s in new),
        "render_abc": render_abc,
        "render_lyrics": lyrics,
        "full_abc_after": full_after,
        "overlap_seconds": overlap_s,
        "overlap_sections": len(overlap),
        "new_seconds": new_s,
        "material_seconds": overlap_s + new_s,
        "max_seconds": round((overlap_s + new_s) * 1.25 + 15.0, 2),
        "warnings": errors,
        "notes": notes,
        "session": plan.get("session", {}),
    }


class FVM_SongExtendApply:
    """Prueft den LLM-Entwurf (oder baut den Loop) und liefert die Render-Partitur."""

    CATEGORY = CATEGORY
    FUNCTION = "apply"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("render_abc", "render_lyrics", "max_seconds", "part_json", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {"forceInput": True}),
                "tonart_strikt": ("BOOLEAN", {"default": True, "tooltip":
                                  "Tonartfremde LLM-Toene auf den naechsten Ton der Tonleiter ziehen. "
                                  "Aus = chromatische Toene bleiben (mutiger, aber riskanter)."}),
            },
            "optional": {
                "llm_text": ("STRING", {"forceInput": True, "lazy": True}),
            },
        }

    def check_lazy_status(self, plan_json, tonart_strikt=True, llm_text=None):
        try:
            needs = json.loads(plan_json).get("needs_llm", False)
        except ValueError:
            needs = False
        return ["llm_text"] if needs and llm_text is None else []

    def apply(self, plan_json, tonart_strikt=True, llm_text=""):
        plan = json.loads(plan_json)
        part = apply_plan(plan, llm_text or "", tonart_strikt)
        report = [f"Quelle: {part['source']} | neu: {', '.join(part['labels'])} "
                  f"({part['new_seconds']:.1f} s) | Ueberlapp {part['overlap_seconds']:.1f} s"]
        report += part["notes"]
        if part["warnings"]:
            report.append("Warnungen: " + "; ".join(part["warnings"][:5]))
        part["report"] = "\n".join(report)
        logger.info("[SongExtend] %s", part["report"].replace("\n", " | "))
        return (part["render_abc"], part["render_lyrics"], float(part["max_seconds"]),
                json.dumps(part), part["report"])


# ---------------------------------------------------------------- Anhaengen

def _fit(wave: np.ndarray, sr: int, score: Score, max_offset_s: float = 30.0) -> Dict[str, Any]:
    timing = fit_score_timing(wave, sr, onset_times(score), score.total_seconds(),
                              max_offset_s=max_offset_s)
    return timing.as_dict()


def _to_samples(timing: Dict[str, Any], score_seconds: float, sr: int) -> int:
    return int(round((timing["offset"] + timing["scale"] * score_seconds) * sr))


def _section_spans(score: Score, timing: Dict[str, Any], sr: int) -> List[Tuple[str, int, int]]:
    """(Label, Start, Ende) jedes Abschnitts im Audio, ueber die Partitur-Ausrichtung."""
    spans, t = [], 0.0
    for sec, dur in zip(score.sections, score.section_durations()):
        spans.append((sec.label.lower(), _to_samples(timing, t, sr), _to_samples(timing, t + dur, sr)))
        t += dur
    return spans


def _levels_by_label(wave: np.ndarray, sr: int, score: Score, timing: Dict[str, Any]) -> Dict[str, float]:
    """Typische Lautheit je Abschnittstyp im Original (Verse sind meist leiser als Chorusse)."""
    out: Dict[str, float] = {"*": body_level_db(wave, sr, 0, _section_spans(score, timing, sr)[-1][2])}
    for label in {s.label.lower() for s in score.sections}:
        pieces = [wave[:, a:b] for l, a, b in _section_spans(score, timing, sr) if l == label and b > a]
        if pieces:
            out[label] = body_level_db(np.concatenate(pieces, axis=-1), sr)
    return out


def _part_gain(wave: np.ndarray, sr: int, render: Score, timing: Dict[str, Any],
               n_overlap: int, levels: Dict[str, float]) -> float:
    """Gain fuer einen neuen Teil: jeder neue Abschnitt gegen denselben Typ im Original.

    Gemessen: gegen die Gesamt-Lautheit wurde ein neuer Verse um +4.8 dB zu laut
    gemacht, weil der Song-Median von den lauteren Chorussen bestimmt wird.
    """
    spans = _section_spans(render, timing, sr)[n_overlap:-1]   # ohne Ueberlapp und Ausklang-Pausen
    total_w, acc = 0.0, 0.0
    for label, a, b in spans:
        b = min(b, wave.shape[-1])
        if b - a < sr:
            continue
        ref = levels.get(label, levels["*"])
        acc += (ref - body_level_db(wave, sr, a, b)) * (b - a)
        total_w += b - a
    if total_w == 0:
        return 1.0
    return reference_gain(0.0, -acc / total_w)


def commit_part(directory: str, part: Dict[str, Any], new_wave: np.ndarray, new_sr: int,
                crossfade_ms: float = 120.0, search_seconds: float = 30.0,
                loudness_match: bool = True, min_new_share: float = 0.9) -> Dict[str, Any]:
    """Neuen Teil an der Abschnittsgrenze ansetzen - Positionen aus der Partitur.

    Vorgaenger: Ende seines notierten Materials (Basis: Partitur ohne
    Schluss-Pausen; angehaengter Teil: Ende seiner Render-Partitur).
    Neuer Teil: Ende des Ueberlapps in seiner Render-Partitur.
    """
    state = es.load_state(directory)
    sr = int(state["sample_rate"])
    new_wave = resample(to_stereo(new_wave), new_sr, sr)
    prev_rec = state["parts"][-1]
    prev_wave, prev_sr = es.read_audio(os.path.join(directory, prev_rec["file"]))
    prev_wave = resample(to_stereo(prev_wave), prev_sr, sr)

    if "timing" not in prev_rec:                      # Basis: einmal einmessen
        base = parse_score(state["base_abc"])
        prev_rec["timing"] = _fit(prev_wave, sr, base, search_seconds)
        prev_rec["material_end_s"] = float(sum(open_score(state["base_abc"])[0].section_durations()))
    render = parse_score(part["render_abc"])
    new_timing = _fit(new_wave, sr, render, search_seconds)
    overlap_s = float(part["overlap_seconds"])
    material_s = float(part.get("material_seconds", render.total_seconds()))   # ohne Ausklang-Pausen

    # Schnitt an einem Taktstrich im Ueberlapp, ~8 s vor seinem Ende: dort spielt der
    # Vorgaenger schon "Schluss" (leiser, ausklingend) - das soll nicht ins Ergebnis.
    n_overlap = int(part.get("overlap_sections", 1))
    starts = bar_starts(render, render.sections[:n_overlap])
    back = min(8.0, overlap_s / 2)
    candidates = [b for b in starts if SETTLE_S <= b <= overlap_s - back]
    if not candidates:                      # zu kurzer Ueberlapp (manuell gewaehlt)
        candidates = [b for b in starts if 2.0 <= b <= overlap_s - back]
    if not candidates:
        candidates = [overlap_s]

    def at_prev(b: float) -> int:
        return min(prev_wave.shape[-1],
                   _to_samples(prev_rec["timing"], prev_rec["material_end_s"] - overlap_s + b, sr))

    # Der Vorgaenger behaelt nach seinem eigenen Schnitt mindestens MIN_SEGMENT_S -
    # sonst wuerde sein neues Material ganz ersetzt und zwei Naehte laegen dicht beieinander.
    keep = [b for b in candidates if at_prev(b) >= int(prev_rec.get("start", 0)) + int(MIN_SEGMENT_S * sr)]
    candidates = keep or candidates[-1:]

    pick, cut_distance = choose_cut(prev_wave, new_wave, sr,
                                    [(at_prev(b), _to_samples(new_timing, b, sr)) for b in candidates])
    cut_s = candidates[pick]
    prev_cut = at_prev(cut_s)
    new_cut = _to_samples(new_timing, cut_s, sr)
    eighth_s = render.units_to_seconds(Fraction(1, 8) / render.unit) * float(new_timing["scale"])
    shift, refine_corr = refine_cut(prev_wave, new_wave, sr, prev_cut, new_cut, eighth_s)
    new_cut += shift
    score_end = _to_samples(new_timing, material_s, sr) + shift
    have = (new_wave.shape[-1] - _to_samples(new_timing, overlap_s, sr)) / sr
    need = material_s - overlap_s
    if have < min_new_share * need * new_timing["scale"]:
        raise ValueError(
            f"YuE2 hat zu frueh aufgehoert: vom neuen Teil sind nur {have:.1f} s von ~{need:.1f} s "
            f"gerendert. Nichts angehaengt - einfach nochmal queuen (neuer Seed).")

    gain, gain_note = 1.0, ""
    gain_start = None
    ramp_end = _to_samples(new_timing, overlap_s, sr) + shift
    if loudness_match:
        if "loudness_by_label" not in state:        # einmal am Original messen
            base_rec = state["parts"][0]
            if len(state["parts"]) == 1:
                base_wave = prev_wave
            else:
                raw, base_sr = es.read_audio(os.path.join(directory, base_rec["file"]))
                base_wave = resample(to_stereo(raw), base_sr, sr)
            state["loudness_by_label"] = _levels_by_label(
                base_wave, sr, open_score(state["base_abc"])[0], base_rec["timing"])
        gain = _part_gain(new_wave, sr, render, new_timing, n_overlap, state["loudness_by_label"])
        # Am Schnitt an den Vorgaenger angleichen (gleiche Musik, mitten im Song), dann
        # bis zum Beginn des neuen Materials zum Zielpegel gleiten - keine Stufe an der
        # Naht, kein Wegdriften ueber viele Teile.
        w = int(min(6.0, cut_s) * sr)
        local = (rms(prev_wave[:, max(0, prev_cut - w):prev_cut])
                 / rms(new_wave[:, max(0, new_cut - w):new_cut]))
        gain_start = float(min(2.0, max(0.5, float(prev_rec.get("gain", 1.0)) * local)))
        if gain in (0.5, 2.0) or gain_start in (0.5, 2.0):
            gain_note = (f"Pegelangleich an der Grenze (Schnitt {gain_start:.2f}, Ziel {gain:.2f}) "
                         f"- Uebergang anhoeren")
    weak = [name for name, t in (("Vorgaenger", prev_rec["timing"]), ("neuer Teil", new_timing))
            if t["contrast"] < 3.0]
    record = {"kind": "render", "start": new_cut, "end": None, "score_end": min(score_end, new_wave.shape[-1]),
              "gain": gain, "labels": part["labels"], "mode": part["mode"], "source": part["source"],
              "sections_added": part["sections_added"], "overlap_seconds": overlap_s,
              "new_seconds": part["new_seconds"], "timing": new_timing, "material_end_s": material_s,
              "gain_note": gain_note, "weak_timing": weak,
              "prev_cut": prev_cut, "render_seconds": round(new_wave.shape[-1] / sr, 2),
              "refine_ms": round(shift / sr * 1000), "refine_corr": round(refine_corr, 3),
              "gain_start": gain_start, "ramp_end": ramp_end,
              "cut_in_overlap_s": round(cut_s, 2), "cut_candidates": len(candidates),
              "cut_distance_db": round(cut_distance, 2)}
    return es.append_part(directory, state, new_wave, sr, record, prev_cut, part["full_abc_after"])


class FVM_SongExtendCommit:
    """Haengt den gerenderten Teil an (nur bei Aktion 'anhaengen')."""

    CATEGORY = CATEGORY
    FUNCTION = "commit"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("session_json", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_json": ("STRING", {"forceInput": True}),
                "search_seconds": ("FLOAT", {"default": 30.0, "min": 2.0, "max": 120.0, "step": 1.0,
                                   "tooltip": "Maximaler Versatz zwischen Partitur und Rendering "
                                              "(Vorlauf von YuE2). 30 s reichen praktisch immer."}),
                "loudness_match": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "part_json": ("STRING", {"forceInput": True, "lazy": True}),
                "rendered_audio": ("AUDIO", {"lazy": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def check_lazy_status(self, session_json, search_seconds=30.0, loudness_match=True,
                          part_json=None, rendered_audio=None):
        action = json.loads(session_json).get("action")
        if action != ACTIONS[0]:
            return []
        return [n for n, v in (("part_json", part_json), ("rendered_audio", rendered_audio)) if v is None]

    def commit(self, session_json, search_seconds=30.0, loudness_match=True,
               part_json=None, rendered_audio=None):
        sess = json.loads(session_json)
        if sess.get("action") != ACTIONS[0]:
            return (session_json, f"Aktion '{sess.get('action')}': nichts angehaengt.")
        part = json.loads(part_json)
        wave, sr = _audio_to_np(rendered_audio)
        rec = commit_part(sess["dir"], part, wave, sr, search_seconds=search_seconds,
                          loudness_match=loudness_match)
        sr_s = float(es.load_state(sess["dir"])["sample_rate"])
        prev_t = es.load_state(sess["dir"])["parts"][rec["index"] - 1].get("timing", {})
        t = rec["timing"]
        lines = [part.get("report", ""),
                 f"Teil {rec['index']} angehaengt: Rendering {rec['render_seconds']} s, "
                 f"Vorgaenger geschnitten bei {rec['prev_cut'] / sr_s:.2f} s, neuer Teil ab "
                 f"{rec['start'] / sr_s:.2f} s, Pegel am Schnitt x{rec['gain_start'] or 1:.2f} -> Ziel x{rec['gain']:.2f}",
                 f"Naht: Taktstrich {rec.get('cut_in_overlap_s')} s in den Ueberlapp (bester von "
                 f"{rec.get('cut_candidates')}, Klangabweichung {rec.get('cut_distance_db')} dB), Feinjustage "
                 f"{rec.get('refine_ms')} ms (Rasterkonsistenz {rec.get('refine_corr')})",
                 f"Partitur-Ausrichtung: Vorgaenger Versatz {prev_t.get('offset')} s, Tempo x{prev_t.get('scale')}, "
                 f"Eindeutigkeit {prev_t.get('contrast')} | neu Versatz {t['offset']} s, Tempo x{t['scale']}, "
                 f"Eindeutigkeit {t['contrast']}"]
        if rec.get("gain_note"):
            lines.append("HINWEIS: " + rec["gain_note"])
        if rec.get("weak_timing"):
            lines.append("WARNUNG: Partitur-Ausrichtung unsicher (" + ", ".join(rec["weak_timing"]) +
                         ") - Uebergang anhoeren, ggf. 'letzten Teil entfernen' und neu anhaengen.")
        out = dict(sess)
        out["committed"] = rec["index"]
        report = "\n".join(l for l in lines if l)
        logger.info("[SongExtend] %s", report.replace("\n", " | "))
        return (json.dumps(out), report)


# ---------------------------------------------------------------- Zusammenfuegen

def merge_session(directory: str, arrangement: str = "alle", crossfade_ms: float = 120.0,
                  end_fade_s: float = 0.0) -> Tuple[np.ndarray, int, List[int], str]:
    state = es.load_state(directory)
    parts = es.load_waves(directory, state)
    sr = int(state["sample_rate"])
    order = parse_arrangement(arrangement, len(parts))
    segs = arrangement_segments(parts, order)
    wave = join_segments(segs, sr, crossfade_ms, end_fade_s)
    peak = float(np.abs(wave).max()) if wave.size else 0.0
    if peak > 0.999:
        wave = wave * (0.999 / peak)
    lines = [f"Reihenfolge: {', '.join(map(str, order))}  ->  {wave.shape[-1] / sr:.1f} s"]
    for pos, (seg, idx) in enumerate(zip(segs, order)):
        end = seg.wave.shape[-1] if seg.end is None else seg.end
        lines.append(f"  {pos + 1}. Teil {idx} ({', '.join(state['parts'][idx].get('labels') or ['Basis'])}): "
                     f"{(end - seg.start) / sr:.1f} s, Gain {seg.gain:.2f}")
    return wave, sr, order, "\n".join(lines)


class FVM_SongExtendMerge:
    """Fuegt alle (oder ausgewaehlte, auch wiederholte) Teile zu einem Song zusammen."""

    CATEGORY = CATEGORY
    FUNCTION = "merge"
    OUTPUT_NODE = True
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "report", "saved_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_json": ("STRING", {"forceInput": True}),
                "arrangement": ("STRING", {"default": "alle", "tooltip":
                                "alle = 0,1,2,... in Reihenfolge. Eigene Folge z. B. 0,1,2,2,3 "
                                "oder 0-2,2x3 (Teil 2 dreimal). Teil 0 ist der Ausgangs-Song."}),
                "crossfade_ms": ("FLOAT", {"default": 120.0, "min": 0.0, "max": 2000.0, "step": 10.0}),
                "end_fade_seconds": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 30.0, "step": 0.5, "tooltip": "Ausblenden am Songende. YuE2 hoert bei angehaengten Teilen am letzten Ton auf; fuer ein echtes Ende zuletzt einen Abschnitt mit Label outro anhaengen."}),
                "save_final": ("BOOLEAN", {"default": False, "tooltip":
                               "Ergebnis als FLAC nach output/audio/music/extended speichern."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def merge(self, session_json, arrangement, crossfade_ms, end_fade_seconds, save_final):
        sess = json.loads(session_json)
        wave, sr, order, report = merge_session(sess["dir"], arrangement, crossfade_ms, end_fade_seconds)
        saved = ""
        if save_final:
            state = es.load_state(sess["dir"])
            name = es.safe_name(f"{state.get('title') or sess['name']}-extended-{len(order)}teile")
            path = os.path.join(_output_dir(), "audio", "music", "extended",
                                f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.flac")
            es.write_audio(path, wave, sr)
            saved = path
            report += f"\nGespeichert: {path}"
        return {"ui": {"text": [report]}, "result": (_np_to_audio(wave, sr), report, saved)}
