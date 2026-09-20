"""YuE2-ABC auf Abschnittsebene: lesen, pruefen, reparieren, zusammensetzen.

YuE2 rendert aus einer ABC-Partitur in einem festen, schmalen Dialekt (so wie
SheetSage2 ihn schreibt und das Music Production Toolkit ihn durchreicht):

    X:1 / T: / M:4/4 / L:1/32 / Q:1/4=129
    V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
    V: Ins clef=treble name="Ins Melody" snm="Inst."
    K:Ab
    % intro
    V: Vocal
    Z|"G#"z32|"G#"z24z4z4|
    V: Ins
    Z|z32|z24z4E4|
    % verse
    ...

Ein Abschnitt beginnt mit ``% label``; darin stehen Gruppen, je Stimme eine
``V:``-Zeile und eine Zeile mit bis zu vier Takten. Jeder Takt muss exakt die
Taktlaenge in L-Einheiten haben (bei 4/4 und L:1/32 also 32).

Fuers Verlaengern eines Songs reicht diese Abschnittssicht: Abschnitte
anhaengen, wiederholen, den Schluss als Ueberlapp mitrendern. Das Modul ist
bewusst eigenstaendig (keine Imports aus dem Toolkit), prueft aber genau das,
woran ein LLM-Entwurf typischerweise scheitert: Taktlaengen, Stimmenfolge,
fremde Notation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

HEADER_KEYS = ("X", "T", "M", "L", "Q", "K", "V")

TOKEN = re.compile(
    r"""
     (?P<chord>"[^"]*")
    |(?P<deco>![^!]*!)
    |(?P<note>(?P<acc>[\^_=]*)(?P<pitch>[A-Ga-gz])(?P<oct>[,']*)(?P<len>\d*(?:/\d*)?))
    |(?P<mrest>Z\d*)
    |(?P<tie>-)
    |(?P<space>\s+)
    """,
    re.X,
)

NL = "\n"

CHORD_OK = re.compile(r"^[A-G][#b]?[A-Za-z0-9#b+/()]*$")


class AbcError(ValueError):
    """Partitur passt nicht zum Dialekt."""


# ---------------------------------------------------------------- Kopf / Masse

@dataclass
class Score:
    header: List[str]
    voices: List[str]
    sections: List["Section"]
    meter: Fraction = Fraction(4, 4)
    unit: Fraction = Fraction(1, 32)
    tempo_beat: Fraction = Fraction(1, 4)
    tempo_bpm: float = 120.0
    key: str = "C"

    @property
    def bar_units(self) -> Fraction:
        return self.meter / self.unit

    def seconds_per_bar(self) -> float:
        return float(self.meter / self.tempo_beat) * 60.0 / self.tempo_bpm

    def units_to_seconds(self, units: Fraction) -> float:
        return float(units * self.unit / self.tempo_beat) * 60.0 / self.tempo_bpm

    def section_durations(self, sections: Optional[List["Section"]] = None,
                          meters: Optional[Dict[str, Fraction]] = None) -> List[float]:
        """Sekunden je Abschnitt aus den tatsaechlichen Taktlaengen der ersten Stimme."""
        secs = self.sections if sections is None else sections
        out = [0.0] * len(secs)
        bars, _ = walk_bars(self, secs, meters)
        v0 = self.voices[0]
        for b in bars:
            if b.voice == v0:
                try:
                    d = bar_duration(b.bar, b.bar_units)
                except AbcError:
                    d = b.bar_units
                out[b.section] += self.units_to_seconds(d)
        return out

    def section_seconds(self, section: "Section") -> float:
        return self.section_durations([section], self.end_meters(before=section))[0]

    def total_seconds(self) -> float:
        return sum(self.section_durations())

    def end_meters(self, before: Optional["Section"] = None) -> Dict[str, Fraction]:
        """Taktart je Stimme am Ende der Partitur (oder vor einem ihrer Abschnitte)."""
        secs = self.sections
        if before is not None:
            idx = next((i for i, s in enumerate(secs) if s is before), len(secs))
            secs = secs[:idx]
        return walk_bars(self, secs)[1]


@dataclass
class Section:
    label: str
    groups: List[Dict[str, str]] = field(default_factory=list)

    def bar_count(self, voices: List[str]) -> int:
        if not voices:
            return 0
        return sum(len(split_bars(g.get(voices[0], ""))) for g in self.groups)


def _fraction(text: str, default: Fraction) -> Fraction:
    text = (text or "").strip()
    if text == "C":
        return Fraction(4, 4)
    if text == "C|":
        return Fraction(2, 2)
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", text)
    return Fraction(int(m.group(1)), int(m.group(2))) if m else default


def _tempo(text: str) -> Tuple[Fraction, float]:
    text = (text or "").strip()
    m = re.match(r"^(\d+)\s*/\s*(\d+)\s*=\s*(\d+(?:\.\d+)?)$", text)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2))), float(m.group(3))
    m = re.match(r"^(\d+(?:\.\d+)?)$", text)
    if m:
        return Fraction(1, 4), float(m.group(1))
    return Fraction(1, 4), 120.0


def _voice_id(line: str) -> str:
    rest = line.split(":", 1)[1].strip()
    return rest.split()[0] if rest else ""


def voice_parts(text: str) -> Tuple[List[str], str]:
    """Stimmblock -> (Inline-Kopfzeilen wie ``M:1/4``, Notenzeile)."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return [], ""
    return lines[:-1], lines[-1]


def split_bars(line: str) -> List[str]:
    """Takte einer Stimmzeile; der abschliessende Taktstrich erzeugt kein Leerfeld."""
    parts = voice_parts(line)[1].split("|")
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return parts


def join_bars(bars: List[str]) -> str:
    return "".join(b + "|" for b in bars)


# ---------------------------------------------------------------- Parser

def parse_score(text: str) -> Score:
    """Komplette Partitur lesen. Kopf endet mit der K:-Zeile."""
    lines = [l.rstrip() for l in (text or "").replace("\r\n", "\n").split("\n")]
    header: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip():
            continue
        header.append(line)
        if line.startswith("K:"):
            break
    if not header or not header[-1].startswith("K:"):
        raise AbcError("Kein K:-Kopf gefunden - das ist keine vollstaendige YuE2-Partitur.")

    fields = {}
    voices: List[str] = []
    for line in header:
        key = line.split(":", 1)[0].strip()
        if key == "V":
            voices.append(_voice_id(line))
        elif key in HEADER_KEYS:
            fields[key] = line.split(":", 1)[1].strip()
    if not voices:
        raise AbcError("Keine V:-Stimmen im Kopf.")
    beat, bpm = _tempo(fields.get("Q", ""))
    score = Score(header=header, voices=voices, sections=[],
                  meter=_fraction(fields.get("M", ""), Fraction(4, 4)),
                  unit=_fraction(fields.get("L", ""), Fraction(1, 8)),
                  tempo_beat=beat, tempo_bpm=bpm, key=fields.get("K", "C"))
    score.sections = parse_body("\n".join(lines[i:]), voices)
    return score


def _match_voice(name: str, voices: List[str]) -> Optional[str]:
    name = (name or "").strip()
    for v in voices:
        if v.lower() == name.lower():
            return v
    if name.isdigit() and 1 <= int(name) <= len(voices):
        return voices[int(name) - 1]
    for v in voices:  # "Inst" / "Instrument" / "Vocals"
        if name.lower().startswith(v.lower()[:3]):
            return v
    return None


def parse_body(text: str, voices: List[str], default_label: str = "section") -> List[Section]:
    """Abschnitte aus dem Rumpf (ohne Kopf) lesen.

    Tolerant gegen LLM-Eigenheiten: ``V:1``/``V:Vocal``-Varianten, Kopfzeilen
    mitten im Text (ignoriert), fehlendes ``% label`` vor der ersten Gruppe.
    """
    sections: List[Section] = []
    current: Optional[Section] = None
    group: Dict[str, str] = {}
    voice: Optional[str] = None
    pending: Dict[str, List[str]] = {}

    def flush_group():
        nonlocal group
        if group and current is not None:
            current.groups.append(group)
        group = {}

    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            flush_group()
            label = line.lstrip("%").strip() or default_label
            current = Section(label=label)
            sections.append(current)
            voice = None
            continue
        m = re.match(r"^\[?V:\s*([^\s\]]+)", line)
        if m:
            v = _match_voice(m.group(1), voices)
            if v is None:
                raise AbcError(f"Unbekannte Stimme '{m.group(1)}' (erlaubt: {', '.join(voices)}).")
            if v in group:
                flush_group()
            voice = v
            rest = line[m.end():].lstrip("]").strip()
            if rest and "|" in rest:          # "[V:Ins] c4c4|..." in einer Zeile
                if current is None:
                    current = Section(label=default_label)
                    sections.append(current)
                group[voice] = rest
                voice = None
            continue
        if re.match(r"^[A-Za-z]:", line) and "|" not in line:
            if voice is not None and line[0] in "MLK":
                pending.setdefault(voice, []).append(line)   # Taktart-/Tonartwechsel
            continue                           # sonst: Kopfzeile im LLM-Text
        if voice is None:
            raise AbcError(f"Notenzeile ohne vorherige V:-Zeile: '{line[:40]}'")
        if current is None:
            current = Section(label=default_label)
            sections.append(current)
        group[voice] = NL.join(pending.pop(voice, []) + [line])
        voice = None
    flush_group()
    return [s for s in sections if s.groups]


# ---------------------------------------------------------------- Takte

@dataclass
class Tok:
    kind: str              # chord | deco | note | mrest | tie
    raw: str
    head: str = ""         # Note ohne Laenge
    length: Fraction = Fraction(0)


def _parse_len(text: str) -> Fraction:
    if not text:
        return Fraction(1)
    if "/" in text:
        num, den = text.split("/", 1)
        return Fraction(int(num) if num else 1, int(den) if den else 2)
    return Fraction(int(text))


def tokenize_bar(bar: str, bar_units: Fraction) -> List[Tok]:
    toks: List[Tok] = []
    pos = 0
    while pos < len(bar):
        m = TOKEN.match(bar, pos)
        if not m or m.end() == pos:
            raise AbcError(f"Nicht unterstuetzte Notation '{bar[pos:pos + 12]}' in Takt '{bar}'.")
        pos = m.end()
        if m.group("space"):
            continue
        if m.group("note"):
            toks.append(Tok("note", m.group("note"),
                            head=m.group("acc") + m.group("pitch") + m.group("oct"),
                            length=_parse_len(m.group("len"))))
        elif m.group("mrest"):
            toks.append(Tok("mrest", m.group("mrest"), length=bar_units))
        elif m.group("chord"):
            toks.append(Tok("chord", m.group("chord")))
        elif m.group("deco"):
            toks.append(Tok("deco", m.group("deco")))
        elif m.group("tie"):
            toks.append(Tok("tie", "-"))
    return toks


def bar_duration(bar: str, bar_units: Fraction) -> Fraction:
    return sum((t.length for t in tokenize_bar(bar, bar_units)), Fraction(0))


def _len_text(length: Fraction) -> str:
    if length == 1:
        return ""
    if length.denominator == 1:
        return str(length.numerator)
    if length.numerator == 1:
        return f"/{length.denominator}"
    return f"{length.numerator}/{length.denominator}"


def _emit(toks: List[Tok]) -> str:
    out = []
    for t in toks:
        if t.kind == "note":
            out.append(t.head + _len_text(t.length))
        else:
            out.append(t.raw)
    return "".join(out)


def rest_bar(bar_units: Fraction, chord: str = "") -> str:
    return f'{chord}z{_len_text(bar_units)}'


def repair_bar(bar: str, bar_units: Fraction) -> Tuple[str, Optional[str]]:
    """Takt auf exakte Laenge bringen. Rueckgabe (takt, notiz|None).

    Zu kurz: mit Pause auffuellen. Zu lang: von hinten kuerzen. Ungueltige
    Akkordsymbole fliegen raus. Fremde Notation ist nicht reparierbar (AbcError).
    """
    toks = tokenize_bar(bar, bar_units)
    note = None
    cleaned = []
    for t in toks:
        if t.kind == "chord" and not CHORD_OK.match(t.raw.strip('"')):
            note = f"Akkord {t.raw} entfernt"
            continue
        cleaned.append(t)
    toks = cleaned
    total = sum((t.length for t in toks), Fraction(0))
    if total == bar_units:
        return _emit(toks), note
    if total < bar_units:
        toks.append(Tok("note", "", head="z", length=bar_units - total))
        return _emit(toks), f"Takt um {bar_units - total} Einheiten mit Pause aufgefuellt"
    excess = total - bar_units
    while excess > 0 and toks:
        t = toks[-1]
        if t.length == 0:
            toks.pop()
            continue
        if t.kind == "mrest":
            toks[-1] = Tok("note", "", head="z", length=bar_units)
            t = toks[-1]
        if t.length > excess:
            t.length -= excess
            excess = Fraction(0)
        else:
            excess -= t.length
            toks.pop()
    while toks and toks[-1].kind == "tie":
        toks.pop()
    return _emit(toks), f"Takt um {total - bar_units} Einheiten gekuerzt"


# ---------------------------------------------------------------- Abschnitte

@dataclass
class BarRef:
    section: int
    group: int
    voice: str
    index: int
    bar: str
    bar_units: Fraction


def _meter_after(inline: List[str], meter: Fraction) -> Fraction:
    for line in inline:
        if line.startswith("M:"):
            meter = _fraction(line[2:], meter)
    return meter


def walk_bars(score: "Score", sections: List[Section],
              meters: Optional[Dict[str, Fraction]] = None) -> Tuple[List[BarRef], Dict[str, Fraction]]:
    """Alle Takte der Reihe nach, mit der je Stimme gueltigen Taktlaenge.

    Inline-``M:``-Zeilen (SheetSage2 schreibt z. B. einen 1/4-Takt mitten in den
    Chorus) gelten ab ihrer Stelle fuer diese Stimme weiter.
    """
    meters = dict(meters) if meters else {v: score.meter for v in score.voices}
    out: List[BarRef] = []
    for si, sec in enumerate(sections):
        for gi, g in enumerate(sec.groups):
            for v in score.voices:
                if v not in g:
                    continue
                inline, _ = voice_parts(g[v])
                meters[v] = _meter_after(inline, meters.get(v, score.meter))
                units = meters[v] / score.unit
                for bi, bar in enumerate(split_bars(g[v])):
                    out.append(BarRef(si, gi, v, bi, bar, units))
    return out, meters


@dataclass
class RepairReport:
    bars_total: int = 0
    bars_repaired: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def repaired_share(self) -> float:
        return self.bars_repaired / self.bars_total if self.bars_total else 0.0


def _rebuild(inline: List[str], bars: List[str]) -> str:
    return NL.join(inline + [join_bars(bars)])


def repair_sections(score: "Score", sections: List[Section],
                    meters: Optional[Dict[str, Fraction]] = None,
                    report: Optional[RepairReport] = None) -> List[Section]:
    """Stimmen vervollstaendigen, Taktzahl je Gruppe angleichen, Takte reparieren."""
    report = report if report is not None else RepairReport()
    meters = dict(meters) if meters else {v: score.meter for v in score.voices}
    out = []
    for sec in sections:
        groups = []
        for gi, g in enumerate(sec.groups):
            parts = {v: voice_parts(g.get(v, "")) for v in score.voices}
            n = max(len(split_bars(g.get(v, ""))) for v in score.voices)
            if n == 0:
                continue
            new_g = {}
            for v in score.voices:
                inline, _ = parts[v]
                meters[v] = _meter_after(inline, meters.get(v, score.meter))
                units = meters[v] / score.unit
                bars = split_bars(g.get(v, ""))
                if len(bars) < n:
                    report.notes.append(f"{sec.label} G{gi + 1}: Stimme {v} um {n - len(bars)} "
                                        f"Pausentakt(e) ergaenzt")
                    report.bars_repaired += n - len(bars)
                    bars = bars + [rest_bar(units)] * (n - len(bars))
                fixed = []
                for bi, bar in enumerate(bars):
                    report.bars_total += 1
                    new_bar, note = repair_bar(bar, units)
                    if note:
                        report.bars_repaired += 1
                        report.notes.append(f"{sec.label} G{gi + 1} {v} T{bi + 1}: {note}")
                    fixed.append(new_bar)
                new_g[v] = _rebuild(inline, fixed)
            groups.append(new_g)
        out.append(Section(label=sec.label, groups=groups))
    return out


def silence_voice(score: "Score", sections: List[Section], voice: str) -> Tuple[List[Section], int]:
    """Alle Noten einer Stimme zu Pausen gleicher Laenge (Akkordsymbole bleiben).

    Instrumental-Cover halten die Vocal-Stimme leer; ein LLM schreibt dort gern
    Melodie hinein, die YuE2 dann als Gesang interpretieren wuerde.
    """
    changed = 0
    out = []
    for sec in sections:
        groups = []
        for g in sec.groups:
            new_g = dict(g)
            if voice in g:
                inline, _ = voice_parts(g[voice])
                bars = []
                for bar in split_bars(g[voice]):
                    toks = tokenize_bar(bar, Fraction(1))
                    for t in toks:
                        if t.kind == "note" and not t.head.startswith("z"):
                            t.head = "z"
                            changed += 1
                        if t.kind == "tie":
                            t.raw = ""
                    bars.append(_emit(toks))
                new_g[voice] = _rebuild(inline, bars)
            groups.append(new_g)
        out.append(Section(label=sec.label, groups=groups))
    return out, changed


def voice_is_silent(score: "Score", voice: str) -> bool:
    for sec in score.sections:
        for g in sec.groups:
            for bar in split_bars(g.get(voice, "")):
                for t in tokenize_bar(bar, Fraction(1)):
                    if t.kind == "note" and not t.head.startswith("z"):
                        return False
    return True


def format_section(section: Section, voices: List[str]) -> str:
    out = [f"% {section.label}"]
    for g in section.groups:
        for v in voices:
            if v in g:
                out += [f"V: {v}", g[v]]
    return "\n".join(out)


def format_score(score: Score, sections: Optional[List[Section]] = None) -> str:
    secs = score.sections if sections is None else sections
    body = [format_section(s, score.voices) for s in secs]
    return "\n".join(score.header + body) + "\n"


def with_sections(score: Score, sections: List[Section]) -> Score:
    return Score(header=list(score.header), voices=list(score.voices), sections=list(sections),
                 meter=score.meter, unit=score.unit, tempo_beat=score.tempo_beat,
                 tempo_bpm=score.tempo_bpm, key=score.key)


def outline(score: Score) -> List[Dict]:
    """Abschnittsliste mit Taktzahl und Sekunden (fuer Prompts und Berichte)."""
    secs = score.section_durations()
    return [{"index": i + 1, "label": s.label, "bars": s.bar_count(score.voices),
             "seconds": round(secs[i], 2)}
            for i, s in enumerate(score.sections)]


def select_sections(score: Score, spec: str) -> List[Section]:
    """Abschnitte fuer Loop/Variation waehlen.

    ``last`` / ``last2`` = die letzten n; ``3`` / ``2,3`` / ``2-4`` = 1-basierte
    Indizes; ein Label (``chorus``) = das letzte Vorkommen dieses Labels.
    """
    spec = (spec or "last").strip().lower()
    secs = score.sections
    if not secs:
        return []
    m = re.match(r"^last(\d*)$", spec)
    if m:
        n = int(m.group(1) or 1)
        return secs[-n:]
    if re.match(r"^[\d,\s\-]+$", spec):
        picked = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                picked += list(range(a, b + 1))
            else:
                picked.append(int(part))
        return [secs[i - 1] for i in picked if 1 <= i <= len(secs)]
    for s in reversed(secs):
        if s.label.lower() == spec:
            return [s]
    raise AbcError(f"Abschnitt '{spec}' nicht gefunden (vorhanden: "
                   f"{', '.join(s.label for s in secs)}).")


def validate_sections(score: Score, sections: List[Section],
                      meters: Optional[Dict[str, Fraction]] = None) -> List[str]:
    """Harte Fehler (leer = gueltig): Stimmen, Taktzahlen, Taktlaengen, Notation."""
    errors = []
    for s in sections:
        for gi, g in enumerate(s.groups):
            counts = {v: len(split_bars(g.get(v, ""))) for v in score.voices}
            if len(set(counts.values())) != 1 or 0 in counts.values():
                errors.append(f"{s.label} G{gi + 1}: Taktzahlen je Stimme {counts}")
    bars, _ = walk_bars(score, sections, meters)
    for b in bars:
        label = sections[b.section].label
        try:
            d = bar_duration(b.bar, b.bar_units)
        except AbcError as exc:
            errors.append(f"{label} G{b.group + 1} {b.voice} T{b.index + 1}: {exc}")
            continue
        if d != b.bar_units:
            errors.append(f"{label} G{b.group + 1} {b.voice} T{b.index + 1}: "
                          f"Laenge {d} statt {b.bar_units}")
    return errors


def extract_abc_block(text: str) -> str:
    """ABC aus einer LLM-Antwort holen (Codeblock bevorzugt)."""
    text = text or ""
    blocks = re.findall(r"```(?:abc)?\s*\n(.*?)```", text, re.S | re.I)
    if blocks:
        return max(blocks, key=len)
    start = re.search(r"^\s*(%|\[?V:)", text, re.M)
    return text[start.start():] if start else text


def section_tag(label: str, instrumental: bool = True) -> str:
    """YuE2-Lyrics-Tag je Abschnitt (gleiche Anzahl/Reihenfolge wie die ABC-Abschnitte)."""
    low = (label or "").lower()
    if low.startswith("intro"):
        return "[Intro]"
    if low.startswith("outro") or low.startswith("ending"):
        return "[Outro]"
    if instrumental:
        return "[Instrumental]"
    return "[" + (label[:1].upper() + label[1:] if label else "Verse") + "]"


def _is_rest_bar(bar: str) -> bool:
    try:
        toks = tokenize_bar(bar, Fraction(1))
    except AbcError:
        return False
    return all(t.kind != "note" or t.head.startswith("z") for t in toks)


def strip_trailing_rests(score: Score, sections: List[Section]) -> Tuple[List[Section], int]:
    """Schluss-Pausentakte (in allen Stimmen leer) vom Ende entfernen.

    Ein fertiger Song endet oft mit ``Z|``-Takten - das ist sein Ende. Wird
    dieser Schluss als Ueberlapp mitgerendert, spielt YuE2 das Ende aus und hoert
    auf, statt in den neuen Teil ueberzuleiten. Rueckgabe: (Abschnitte, entfernte Takte).
    """
    secs = [Section(label=s.label, groups=[dict(g) for g in s.groups]) for s in sections]
    removed = 0
    while secs:
        sec = secs[-1]
        if not sec.groups:
            secs.pop()
            continue
        g = sec.groups[-1]
        parts = {v: voice_parts(g.get(v, "")) for v in score.voices}
        bars = {v: split_bars(g.get(v, "")) for v in score.voices}
        if any(not b for b in bars.values()):
            sec.groups.pop()
            continue
        if not all(_is_rest_bar(bars[v][-1]) for v in score.voices):
            break
        for v in score.voices:
            bars[v].pop()
        removed += 1
        if any(not b for b in bars.values()):
            sec.groups.pop()
        else:
            sec.groups[-1] = {v: NL.join(parts[v][0] + [join_bars(bars[v])]) for v in score.voices}
    return [s for s in secs if s.groups], removed


def onset_times(score: Score, sections: Optional[List[Section]] = None) -> List[Tuple[float, float]]:
    """(Sekunde, Gewicht) jedes Noteneinsatzes (1.0) und Akkordwechsels (0.5) laut Partitur.

    Grundlage der Partitur-Ausrichtung: YuE2 haelt das notierte Tempo, also
    liegen die Einsaetze im Audio bis auf Versatz (und ggf. Tempo-Faktor) dort,
    wo die Partitur sie hinschreibt.
    """
    secs = score.sections if sections is None else sections
    bars, _ = walk_bars(score, secs)
    t_voice = {v: Fraction(0) for v in score.voices}
    out: List[Tuple[float, float]] = []
    for b in bars:
        t = t_voice[b.voice]
        tied = False
        try:
            toks = tokenize_bar(b.bar, b.bar_units)
        except AbcError:
            t_voice[b.voice] = t + b.bar_units
            continue
        for tok in toks:
            if tok.kind == "chord":
                out.append((score.units_to_seconds(t), 0.5))
            elif tok.kind == "tie":
                tied = True
            elif tok.kind in ("note", "mrest"):
                if tok.kind == "note" and not tok.head.startswith("z") and not tied:
                    out.append((score.units_to_seconds(t), 1.0))
                t += tok.length
                tied = False
        t_voice[b.voice] = t
    out.sort()
    return out


def bar_starts(score: Score, sections: List[Section],
               meters: Optional[Dict[str, Fraction]] = None) -> List[float]:
    """Sekunden jedes Taktanfangs (erste Stimme) ab Beginn von ``sections``, plus Ende."""
    bars, _ = walk_bars(score, sections, meters)
    v0 = score.voices[0]
    out, t = [0.0], Fraction(0)
    for b in bars:
        if b.voice != v0:
            continue
        try:
            t += bar_duration(b.bar, b.bar_units)
        except AbcError:
            t += b.bar_units
        out.append(score.units_to_seconds(t))
    return out


def smooth_final_bar(score: Score, sections: List[Section]) -> Tuple[List[Section], int]:
    """Den Schlusstakt entschaerfen: endet er mit >= halbem Takt Pause, den letzten Ton halten.

    Gemessen: YuE2 spielt einen Schlusstakt wie ``c'8z24|`` (Schlusston, dann Pause)
    als Songende - ausblenden, gut 1 s digitale Stille, dann Neustart. Beim
    Weiterschreiben ist dieser Takt kein Ende mehr. Die Taktlaenge bleibt gleich,
    alle Zeitberechnungen stimmen weiter. Rueckgabe: (Abschnitte, geaenderte Stimmen).
    """
    if not sections or not sections[-1].groups:
        return sections, 0
    secs = [Section(label=s.label, groups=[dict(g) for g in s.groups]) for s in sections]
    group = secs[-1].groups[-1]
    changed = 0
    for v in score.voices:
        inline, _ = voice_parts(group.get(v, ""))
        bars = split_bars(group.get(v, ""))
        if not bars:
            continue
        toks = tokenize_bar(bars[-1], Fraction(1))
        notes = [i for i, t in enumerate(toks) if t.kind == "note" and not t.head.startswith("z")]
        if not notes:
            continue
        last = notes[-1]
        tail = sum((t.length for t in toks[last + 1:] if t.kind in ("note", "mrest")), Fraction(0))
        total = sum((t.length for t in toks if t.kind in ("note", "mrest")), Fraction(0))
        if total == 0 or tail * 2 < total:
            continue
        toks[last].length += tail
        toks = toks[:last + 1]
        bars[-1] = _emit(toks)
        group[v] = NL.join(inline + [join_bars(bars)])
        changed += 1
    return secs, changed
