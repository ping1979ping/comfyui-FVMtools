"""Neue Melodie ueber eine Rhythmus-Vorlage: das LLM waehlt Toene, der Code haelt den Takt.

Gemessen an echten lokalen Modellen (mistral-small-24b, qwen3.8-27b): frei
geschriebenes ABC scheitert fast immer an der Taktarithmetik - zu kurze Takte,
falsche Laengen, zu wenige Takte. Tonhoehen und Akkorde waehlen koennen die
Modelle dagegen gut.

Deshalb bekommt das LLM hier keine Notenlaengen zu sehen. Ein vorhandener
Abschnitt des Songs liefert Rhythmus, Taktzahl und Stimmenaufbau; das LLM
liefert pro Takt genau so viele Tonhoehen, wie die Vorlage Noten hat (plus
optional einen Akkord je Takt), als JSON. Der Code setzt sie ein - das Ergebnis
hat damit per Konstruktion gueltige Takte.

Tonhoehen laufen in wissenschaftlicher Notation (``Ab4``, ``C5``): JSON-sicher
(ABC-Oktavstriche ``c'`` zerlegen LLM-JSON nachweislich), Tonart aufgeloest,
und jedem Modell vertraut. Der Code rechnet nach ABC zurueck - inklusive der
ABC-Regel, dass ein Vorzeichen bis zum Taktende gilt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .abc_score import (
    AbcError,
    Score,
    Section,
    Tok,
    _emit,
    join_bars,
    split_bars,
    tokenize_bar,
)

STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LETTERS = "CDEFGAB"
SHARP_ORDER = "FCGDAEB"
MAJOR_FIFTHS = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
                "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7}
MODE_SHIFT = {"": 0, "maj": 0, "ion": 0, "m": -3, "min": -3, "aeo": -3, "dor": -2,
              "phr": -4, "lyd": 1, "mix": -1, "loc": -5}
ACC_TEXT = {-2: "__", -1: "_", 0: "=", 1: "^", 2: "^^"}

Pitch = Tuple[str, int, int]          # (Buchstabe, Oktave, Alteration)

CHORD_STRICT = re.compile(
    r"^[A-G][#b]?(maj7|maj9|maj|m7b5|m7|m9|m6|m11|min7|min|m|dim7|dim|aug|sus2|sus4|7sus4|"
    r"add9|6|7|9|11|13)?(/[A-G][#b]?)?$")


# ---------------------------------------------------------------- Tonart / Tonhoehen

def key_alterations(key: str) -> Dict[str, int]:
    """Vorzeichen der Tonart je Buchstabe, z. B. K:Ab -> B,E,A,D = -1."""
    m = re.match(r"^\s*([A-Ga-g])([#b]?)\s*([A-Za-z]*)", key or "C")
    if not m:
        return {}
    tonic = m.group(1).upper() + m.group(2)
    mode = m.group(3).lower()[:3]
    mode = "m" if m.group(3).lower() in ("m", "min", "minor") else mode
    fifths = MAJOR_FIFTHS.get(tonic)
    if fifths is None:        # enharmonisch ungewoehnlich (z. B. A#): ueber Halbtoene annaehern
        fifths = 0
    fifths += MODE_SHIFT.get(mode, 0)
    fifths = max(-7, min(7, fifths))
    alts = {l: 0 for l in LETTERS}
    if fifths > 0:
        for l in SHARP_ORDER[:fifths]:
            alts[l] = 1
    elif fifths < 0:
        for l in SHARP_ORDER[::-1][:-fifths]:
            alts[l] = -1
    return alts


def midi(p: Pitch) -> int:
    letter, octave, alter = p
    return 12 * (octave + 1) + STEP[letter] + alter


def pitch_value(head: str) -> int:
    """Grobe Tonhoehe eines ABC-Kopfs (Vorzeichen ignoriert) - fuer Sortierung/Anzeige."""
    p = abc_head_to_pitch(head, {}, {})
    return midi(p) if p else 0


def abc_head_to_pitch(head: str, key_alts: Dict[str, int], bar_state: Dict[Tuple[str, int], int]) -> Optional[Pitch]:
    """ABC-Notenkopf -> (Buchstabe, Oktave, Alteration); ``bar_state`` wird fortgeschrieben."""
    m = re.match(r"^([\^_=]*)([A-Ga-g])([,']*)$", head or "")
    if not m:
        return None
    acc, letter, octs = m.groups()
    octave = (5 if letter.islower() else 4) + octs.count("'") - octs.count(",")
    L = letter.upper()
    if acc:
        alter = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}.get(acc, 0)
        bar_state[(L, octave)] = alter
    else:
        alter = bar_state.get((L, octave), key_alts.get(L, 0))
    return (L, octave, alter)


def pitch_to_abc(p: Pitch, key_alts: Dict[str, int], bar_state: Dict[Tuple[str, int], int]) -> str:
    """(Buchstabe, Oktave, Alteration) -> ABC-Kopf; Vorzeichen nur wenn noetig."""
    L, octave, alter = p
    current = bar_state.get((L, octave), key_alts.get(L, 0))
    acc = ""
    if alter != current:
        acc = ACC_TEXT.get(alter, "")
        bar_state[(L, octave)] = alter
    if octave >= 5:
        return acc + L.lower() + "'" * (octave - 5)
    return acc + L + "," * (4 - octave)


def diatonic_spelling(key_alts: Dict[str, int]) -> Dict[int, Tuple[str, int]]:
    """Tonklasse -> (Buchstabe, Alteration) der Tonleiter."""
    if not key_alts:
        return {STEP[l]: (l, 0) for l in LETTERS}
    return {(STEP[l] + a) % 12: (l, a) for l, a in key_alts.items()}


def respell(p: Pitch, key_alts: Dict[str, int], snap: bool) -> Tuple[Pitch, bool]:
    """In die Schreibweise der Tonart bringen; mit ``snap`` tonartfremde Toene auf den
    naechsten leitereigenen Ton ziehen. Rueckgabe (Ton, gezogen?)."""
    scale = diatonic_spelling(key_alts)
    m = midi(p)
    pc = m % 12
    moved = False
    if pc not in scale:
        if not snap:
            return p, False
        for delta in (-1, 1, -2, 2):          # halbtonweise, abwaerts zuerst
            if (pc + delta) % 12 in scale:
                m, pc, moved = m + delta, (pc + delta) % 12, True
                break
    letter, alter = scale[pc]
    octave = (m - STEP[letter] - alter) // 12 - 1
    return (letter, octave, alter), moved


def sci_name(p: Pitch) -> str:
    L, octave, alter = p
    return L + {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}.get(alter, "") + str(octave)


def parse_pitch(text: str, key_alts: Dict[str, int], default_octave: int) -> Optional[Pitch]:
    """``Ab4`` / ``C#5`` / ``Eb`` (Oktave ergaenzt) - notfalls auch ein ABC-Kopf wie ``c'``."""
    t = str(text or "").strip().replace("♭", "b").replace("♯", "#")
    m = re.match(r"^([A-Ga-g])(##|bb|#|b)?(-?\d)?$", t)
    if m:
        L = m.group(1).upper()
        alter = {"##": 2, "#": 1, "b": -1, "bb": -2}.get(m.group(2) or "", 0)
        octave = int(m.group(3)) if m.group(3) is not None else default_octave
        return (L, octave, alter)
    return abc_head_to_pitch(t, key_alts, {})


def shift_octave(head: str, octaves: int) -> str:
    """ABC-Kopf um Oktaven verschieben (Vorzeichen bleibt)."""
    m = re.match(r"^([\^_=]*)([A-Ga-g])([,']*)$", head)
    if not m or octaves == 0:
        return head
    acc, letter, octs = m.groups()
    level = (1 if letter.islower() else 0) + octs.count("'") - octs.count(",") + octaves
    if level >= 1:
        return acc + letter.lower() + "'" * (level - 1)
    return acc + letter.upper() + "," * (-level)


# ---------------------------------------------------------------- Vorlage

@dataclass
class Template:
    label: str
    voices: List[str]
    bars: Dict[str, List[List[Tok]]]            # Stimme -> Takte -> Token
    melody_voices: List[str]
    chord_voice: Optional[str]
    key_alts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def bar_count(self) -> int:
        return len(self.bars[self.voices[0]]) if self.voices else 0

    def slots(self, voice: str) -> List[List[Tuple[int, int]]]:
        """Je Slot die (takt, token)-Positionen: eine Note oder per Bindebogen verbundene."""
        out: List[List[Tuple[int, int]]] = []
        tied = False
        for bi, bar in enumerate(self.bars[voice]):
            for ti, t in enumerate(bar):
                if t.kind == "tie":
                    tied = bool(out)
                    continue
                if t.kind != "note":
                    continue
                if t.head.startswith("z"):
                    tied = False
                    continue
                if tied and out:
                    out[-1].append((bi, ti))
                else:
                    out.append([(bi, ti)])
                tied = False
        return out

    def counts(self, voice: str) -> List[int]:
        c = [0] * self.bar_count
        for slot in self.slots(voice):
            c[slot[0][0]] += 1
        return c

    def resolved(self, voice: str) -> Dict[Tuple[int, int], Pitch]:
        """Klingende Tonhoehe jeder Note (Tonart + Taktvorzeichen aufgeloest)."""
        out = {}
        for bi, bar in enumerate(self.bars[voice]):
            state: Dict[Tuple[str, int], int] = {}
            for ti, t in enumerate(bar):
                if t.kind == "note" and not t.head.startswith("z"):
                    p = abc_head_to_pitch(t.head, self.key_alts, state)
                    if p:
                        out[(bi, ti)] = p
        return out

    def pitches(self, voice: str) -> List[List[str]]:
        """Referenz-Toene je Takt in wissenschaftlicher Notation (Slot-Anfaenge)."""
        res = self.resolved(voice)
        p: List[List[str]] = [[] for _ in range(self.bar_count)]
        for slot in self.slots(voice):
            if slot[0] in res:
                p[slot[0][0]].append(sci_name(res[slot[0]]))
        return p

    def chords(self) -> List[str]:
        if not self.chord_voice:
            return [""] * self.bar_count
        return [next((t.raw.strip('"') for t in bar if t.kind == "chord"), "")
                for bar in self.bars[self.chord_voice]]

    def midi_range(self, voice: str) -> Tuple[int, int]:
        vals = [midi(p) for p in self.resolved(voice).values()]
        return (min(vals), max(vals)) if vals else (60, 84)

    def range_names(self, voice: str) -> Tuple[str, str]:
        res = list(self.resolved(voice).values())
        if not res:
            return "C4", "C6"
        return sci_name(min(res, key=midi)), sci_name(max(res, key=midi))

    def fill(self, melodies: Dict[str, List[List[str]]], chords: Optional[List[str]],
             snap_to_key: bool = True) -> Tuple[Section, List[str]]:
        notes: List[str] = []
        bars = {v: [[Tok(t.kind, t.raw, t.head, t.length) for t in bar] for bar in self.bars[v]]
                for v in self.voices}
        for voice in self.melody_voices:
            wanted = melodies.get(voice)
            if not wanted:
                notes.append(f"{voice}: keine Toene geliefert - Vorlage bleibt")
                continue
            lo, hi = self.midi_range(voice)
            ref = self.resolved(voice)
            slots = self.slots(voice)
            counts = self.counts(voice)
            targets: List[Pitch] = []
            adjusted = invalid = 0
            si = 0
            for bi, n in enumerate(counts):
                row = wanted[bi] if bi < len(wanted) and isinstance(wanted[bi], list) else []
                default_oct = ref[slots[si][0]][1] if si < len(slots) and slots[si][0] in ref else 5
                parsed = []
                for x in row:
                    p = parse_pitch(x, self.key_alts, default_oct)
                    if p is None:
                        invalid += 1
                    else:
                        parsed.append(p)
                        default_oct = p[1]
                if len(parsed) != n:
                    adjusted += 1
                if not parsed:   # Takt leer/unbrauchbar: Vorlage behalten
                    parsed = [ref.get(slots[si + k][0], ("C", 5, 0)) for k in range(n)]
                parsed = (parsed + [parsed[-1]] * n)[:n] if n else []
                targets += parsed
                si += n
            if adjusted:
                notes.append(f"{voice}: in {adjusted} von {len(counts)} Takten Tonzahl angepasst")
            if invalid:
                notes.append(f"{voice}: {invalid} unlesbare Tonangabe(n) ignoriert")
            folded = snapped = 0
            assign: Dict[Tuple[int, int], Pitch] = {}
            for slot, p in zip(slots, targets):
                p, moved = respell(p, self.key_alts, snap_to_key)
                snapped += moved
                shift = 0
                while midi(p) + 12 * shift > hi + 5:
                    shift -= 1
                while midi(p) + 12 * shift < lo - 5:
                    shift += 1
                if shift:
                    folded += 1
                    p = (p[0], p[1] + shift, p[2])
                for pos in slot:
                    assign[pos] = p
            if folded:
                notes.append(f"{voice}: {folded} Toene per Oktave in den Tonumfang geklappt")
            if snapped:
                notes.append(f"{voice}: {snapped} tonartfremde Toene auf die Tonleiter gezogen")
            for bi, bar in enumerate(bars[voice]):
                state: Dict[Tuple[str, int], int] = {}
                for ti, t in enumerate(bar):
                    if (bi, ti) in assign:
                        t.head = pitch_to_abc(assign[(bi, ti)], self.key_alts, state)
        if chords and self.chord_voice:
            changed = rejected = 0
            for bi, ch in enumerate(chords[:self.bar_count]):
                ch = str(ch or "").strip().strip('"')
                bar = bars[self.chord_voice][bi]
                if not ch or not any(t.kind == "chord" for t in bar):
                    continue
                if not CHORD_STRICT.match(ch):
                    rejected += 1
                    continue
                bars[self.chord_voice][bi] = [Tok("chord", f'"{ch}"')] + [t for t in bar if t.kind != "chord"]
                changed += 1
            if changed:
                notes.append(f"{changed} Akkorde gesetzt")
            if rejected:
                notes.append(f"{rejected} unbrauchbare Akkordangabe(n) verworfen - Vorlage bleibt dort")
        return self._section(bars), notes

    def _section(self, bars: Dict[str, List[List[Tok]]]) -> Section:
        groups = []
        for start in range(0, self.bar_count, 4):
            groups.append({v: join_bars([_emit(b) for b in bars[v][start:start + 4]]) for v in self.voices})
        return Section(label=self.label, groups=groups)


def build_template(score: Score, source: Section, label: str, bars: int = 0) -> Template:
    """Vorlage aus einem Abschnitt: regulaere Takte flach, auf ``bars`` Takte gebracht.

    Takte mit abweichender Laenge (Inline-Taktwechsel wie ein 1/4-Takt) fliegen
    raus, damit die Vorlage durchgehend im Grundtakt steht.
    """
    voices = score.voices
    units = score.bar_units
    flat: Dict[str, List[List[Tok]]] = {v: [] for v in voices}
    dropped = 0
    for g in source.groups:
        per = {v: split_bars(g.get(v, "")) for v in voices}
        n = min(len(b) for b in per.values())
        for i in range(n):
            toks = {v: tokenize_bar(per[v][i], units) for v in voices}
            if any(sum((t.length for t in toks[v]), Fraction(0)) != units for v in voices):
                dropped += 1
                continue
            for v in voices:
                flat[v].append(toks[v])
    if not flat[voices[0]]:
        raise AbcError(f"Abschnitt '{source.label}' hat keine regulaeren Takte als Vorlage.")
    have = len(flat[voices[0]])
    want = bars if bars > 0 else have
    for v in voices:
        seq = flat[v]
        flat[v] = [[Tok(t.kind, t.raw, t.head, t.length) for t in seq[i % have]] for i in range(want)]
    melody = [v for v in voices if any(t.kind == "note" and not t.head.startswith("z")
                                        for bar in flat[v] for t in bar)]
    chord_voice = next((v for v in voices if any(t.kind == "chord" for bar in flat[v] for t in bar)), None)
    tpl = Template(label=label, voices=voices, bars=flat, melody_voices=melody,
                   chord_voice=chord_voice, key_alts=key_alterations(score.key))
    if dropped:
        tpl.notes.append(f"{dropped} unregelmaessige(r) Takt(e) aus der Vorlage genommen")
    if want != have:
        tpl.notes.append(f"Vorlage von {have} auf {want} Takte gebracht")
    return tpl


def template_brief(tpl: Template) -> str:
    """Prompt-Teil: Taktzahl, Toene je Takt, Referenz-Toene, Akkorde, Tonumfang."""
    lines = [f'SECTION "{tpl.label}": {tpl.bar_count} bars.']
    if tpl.chord_voice:
        lines.append("Reference chords per bar: " + json.dumps(tpl.chords()))
    for v in tpl.melody_voices:
        lo, hi = tpl.range_names(v)
        lines.append(f'Voice "{v}": notes per bar = {json.dumps(tpl.counts(v))}')
        lines.append(f'Voice "{v}" reference pitches per bar: {json.dumps(tpl.pitches(v))}')
        lines.append(f'Voice "{v}" range: {lo} to {hi}')
    return "\n".join(lines)


def skeleton(templates: List[Template]) -> str:
    """Leeres Antwortgeruest mit den richtigen Laengen - Modelle fuellen das zuverlaessiger aus."""
    secs = []
    for t in templates:
        entry = {"label": t.label}
        if t.chord_voice:
            entry["chords"] = ["?"] * t.bar_count
        for v in t.melody_voices:
            entry[v] = [["?"] * n for n in t.counts(v)]
        secs.append(entry)
    return json.dumps({"sections": secs})


def _balanced_objects(text: str) -> List[Tuple[int, dict]]:
    found = []
    for start in [i for i, ch in enumerate(text) if ch == "{"]:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            found.append((start, obj))
                    except ValueError:
                        pass
                    break
    return found


def parse_llm_sections(text: str) -> List[dict]:
    """Abschnitts-Objekte aus einer LLM-Antwort.

    Bevorzugt ``{"sections": [...]}``; ist das Gesamt-JSON kaputt, werden die
    einzeln gueltigen Abschnitts-Objekte (mit ``label``) in Reihenfolge gesammelt.
    """
    objs = _balanced_objects(text or "")
    for _, obj in objs:
        if isinstance(obj.get("sections"), list):
            return [s for s in obj["sections"] if isinstance(s, dict)]
    parts = [obj for _, obj in objs if "label" in obj]
    if parts:
        return parts
    if objs:
        return [objs[0][1]]
    raise AbcError("Kein gueltiges JSON in der LLM-Antwort.")


def parse_llm_json(text: str) -> Dict:
    """Groesstes gueltiges JSON-Objekt (Kompatibilitaet)."""
    objs = _balanced_objects(text or "")
    if not objs:
        raise AbcError("Kein gueltiges JSON in der LLM-Antwort.")
    return max(objs, key=lambda o: len(json.dumps(o[1])))[1]
