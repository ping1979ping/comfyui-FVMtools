"""Zustand einer Song-Verlaengerung ueber mehrere Queue-Laeufe.

Jeder Queue-Lauf haengt hoechstens einen Teil an. Was bisher entstanden ist,
liegt in einem Ordner je Session::

    <output>/fvm_song_extend/<session>/
        state.json          Partitur, Teile, Schnittpunkte
        part_000.flac       Basis (Kopie des Ausgangs-Songs)
        part_001.flac       erster angehaengter Teil (komplettes Rendering inkl. Ueberlapp)
        ...

Ein Teil-Datensatz haelt nur Positionen in seiner eigenen Datei (``start``,
``end``, ``score_end``); zusammengesetzt wird immer frisch aus den Dateien.
Dadurch sind Rueckgaengig und Umsortieren ohne Qualitaetsverlust moeglich.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np

STATE_FILE = "state.json"
STATE_VERSION = 1


def session_root() -> str:
    try:
        import folder_paths  # ComfyUI
        base = folder_paths.get_output_directory()
    except Exception:  # Tests / ausserhalb von ComfyUI
        base = os.path.join(os.getcwd(), "output")
    return os.path.join(base, "fvm_song_extend")


def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", (name or "").strip()).strip(" .")
    return name or "song"


def session_dir(name: str, root: Optional[str] = None) -> str:
    return os.path.join(root or session_root(), safe_name(name))


def sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def load_state(directory: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(directory, STATE_FILE), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(directory: str, state: Dict[str, Any]) -> None:
    os.makedirs(directory, exist_ok=True)
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".state_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(directory, STATE_FILE))
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------- Audio-Dateien

def write_audio(path: str, wave: np.ndarray, sr: int) -> None:
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.clip(np.asarray(wave, dtype=np.float32), -1.0, 1.0).T
    sf.write(path, data, int(sr), subtype="PCM_24")


def read_audio(path: str) -> tuple:
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.T.copy(), int(sr)


# ---------------------------------------------------------------- Session

def new_state(name: str, base_abc: str, base_audio_file: str, sr: int, style: str,
              lyrics: str, title: str, source: str) -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "name": name,
        "title": title,
        "source": source,
        "base_abc_sha": sha(base_abc),
        "sample_rate": int(sr),
        "style": style,
        "lyrics": lyrics,
        "base_abc": base_abc,
        "full_abc": base_abc,
        "parts": [{
            "index": 0, "kind": "base", "file": base_audio_file,
            "start": 0, "end": None, "score_end": None, "gain": 1.0,
            "sections_added": 0, "labels": [],
        }],
        "log": [],
    }


def init_or_load(directory: str, name: str, base_abc: str, base_wave: np.ndarray, sr: int,
                 style: str, lyrics: str, title: str, source: str) -> Dict[str, Any]:
    """Session oeffnen; beim ersten Mal die Basis als part_000.flac anlegen.

    Gehoert eine vorhandene Session zu einer anderen Basis-Partitur, wird das
    als Fehler gemeldet statt still ueberschrieben - sonst verliert ein
    vertippter Session-Name die Arbeit einer anderen Session.
    """
    state = load_state(directory)
    if state and state.get("base_abc_sha") != sha(base_abc):
        if len(state.get("parts", [])) > 1:
            raise ValueError(
                f"Session '{name}' gehoert zu einem anderen Song ({state.get('title')!r}, "
                f"{len(state['parts']) - 1} angehaengte Teile). Neuen Session-Namen waehlen "
                f"oder die Session mit Aktion 'zuruecksetzen' leeren.")
        state = {}
    if not state:
        base_file = "part_000.flac"
        write_audio(os.path.join(directory, base_file), base_wave, sr)
        state = new_state(name, base_abc, base_file, sr, style, lyrics, title, source)
        state["log"].append("Session angelegt")
        save_state(directory, state)
    return state


def undo_last(directory: str, state: Dict[str, Any]) -> str:
    parts = state["parts"]
    if len(parts) <= 1:
        return "Nichts zu entfernen - nur die Basis ist vorhanden."
    last = parts.pop()
    if len(parts) == 1:
        state["full_abc"] = state["base_abc"]      # Original samt Schluss zurueck
    else:
        _drop_sections(state, int(last.get("sections_added", 0)))
    parts[-1]["end"] = None
    parts[-1]["gain_next"] = None
    try:
        os.remove(os.path.join(directory, last["file"]))
    except OSError:
        pass
    msg = f"Teil {last['index']} entfernt ({', '.join(last.get('labels', []))})."
    state["log"].append(msg)
    save_state(directory, state)
    return msg


def reset(directory: str, state: Dict[str, Any]) -> str:
    removed = len(state["parts"]) - 1
    for p in state["parts"][1:]:
        try:
            os.remove(os.path.join(directory, p["file"]))
        except OSError:
            pass
    state["parts"] = state["parts"][:1]
    state["parts"][0]["end"] = None
    state["full_abc"] = state["base_abc"]
    state["log"].append(f"Zurueckgesetzt ({removed} Teile entfernt)")
    save_state(directory, state)
    return f"Session zurueckgesetzt, {removed} Teil(e) entfernt."


def _drop_sections(state: Dict[str, Any], count: int) -> None:
    if count <= 0:
        return
    from .abc_score import format_score, parse_score, with_sections
    score = parse_score(state["full_abc"])
    state["full_abc"] = format_score(with_sections(score, score.sections[:-count]))


def append_part(directory: str, state: Dict[str, Any], wave: np.ndarray, sr: int,
                record: Dict[str, Any], prev_end: int, full_abc: str) -> Dict[str, Any]:
    """Neuen Teil speichern, Schnitt am Vorgaenger setzen, Partitur fortschreiben."""
    idx = len(state["parts"])
    record = dict(record)
    record["index"] = idx
    record["file"] = f"part_{idx:03d}.flac"
    write_audio(os.path.join(directory, record["file"]), wave, sr)
    prev = state["parts"][-1]
    prev["end"] = int(prev_end)
    if prev.get("score_end") is None:
        prev["score_end"] = int(prev_end)
    state["parts"].append(record)
    state["full_abc"] = full_abc
    state["log"].append(f"Teil {idx} angehaengt: {', '.join(record.get('labels', []))} "
                        f"({record.get('mode', '')})")
    save_state(directory, state)
    return record


def load_waves(directory: str, state: Dict[str, Any]) -> List[tuple]:
    """(wave, record) je Teil, auf die Session-Samplerate gebracht."""
    from .audio_stitch import resample, to_stereo
    out = []
    target = int(state["sample_rate"])
    for rec in state["parts"]:
        wave, sr = read_audio(os.path.join(directory, rec["file"]))
        out.append((resample(to_stereo(wave), sr, target), rec))
    return out
