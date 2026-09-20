"""Gerenderte Song-Teile nahtlos aneinanderhaengen.

Jeder neue Teil wird mit einem Ueberlapp gerendert: vorne steht der letzte
Abschnitt des bisherigen Songs, danach das neue Material.

Wo im Audio welcher Abschnitt liegt, kommt aus der Partitur: jedes Rendering
wird gegen seine eigene Partitur eingemessen (Versatz + Tempo-Faktor, siehe
``fit_score_timing``). Geschnitten wird an der Abschnittsgrenze "Ende
Ueberlapp" - im Vorgaenger ist das das Ende seines notierten Materials, im
neuen Teil der Beginn des neuen Abschnitts. Ein Equal-Power-Crossfade und ein
Pegelangleich ueber die Sekunden vor dem Schnitt verdecken den Rest.

Audio-gegen-Audio-Suche wurde verworfen: zwei YuE2-Renderings sind
verschiedene Einspielungen, und in repetitiver Musik aehneln sich alle
Refrains - die Suche lag im Livetest 14 s daneben.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

HOP_SECONDS = 0.01


# ---------------------------------------------------------------- Grundlagen

def to_stereo(wave: np.ndarray) -> np.ndarray:
    """[C, N] -> [2, N]."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 1:
        wave = wave[None, :]
    if wave.shape[0] == 1:
        wave = np.repeat(wave, 2, axis=0)
    return wave[:2]


def resample(wave: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return wave
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr_from), int(sr_to))
        return resample_poly(wave, sr_to // g, sr_from // g, axis=-1).astype(np.float32)
    except ImportError:  # lineare Interpolation als Notnagel
        n_out = int(round(wave.shape[-1] * sr_to / sr_from))
        x_old = np.linspace(0.0, 1.0, wave.shape[-1], endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.stack([np.interp(x_new, x_old, ch) for ch in wave]).astype(np.float32)


def onset_envelope(wave: np.ndarray, sr: int, hop_s: float = HOP_SECONDS) -> np.ndarray:
    """Spektraler Fluss (log-Betrag, halbwellengleichgerichtet), eine Zahl je Hop."""
    mono = np.asarray(wave, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=0)
    hop = max(1, int(round(sr * hop_s)))
    n_fft = 1
    while n_fft < 4 * hop:
        n_fft *= 2
    # Fenster zentrieren: Frame k beschreibt dann die Zeit k * hop (sonst meldet
    # sich jeder Einsatz um ein halbes Fenster zu frueh)
    mono = np.pad(mono, (n_fft // 2, n_fft // 2))
    if mono.shape[0] < n_fft:
        mono = np.pad(mono, (0, n_fft - mono.shape[0]))
    frames = 1 + (mono.shape[0] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(mono[idx] * np.hanning(n_fft)[None, :], axis=1))
    logmag = np.log1p(100.0 * spec)
    flux = np.maximum(0.0, np.diff(logmag, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    return flux.astype(np.float32)


@dataclass
class Timing:
    """Audio-Zeit = offset + scale * Partitur-Zeit."""
    offset: float
    scale: float
    corr: float
    contrast: float        # Spitze / Median ueber alle Tempo-Faktoren: Eindeutigkeit

    def to_audio(self, score_seconds: float) -> float:
        return self.offset + self.scale * score_seconds

    def as_dict(self) -> dict:
        return {"offset": round(self.offset, 3), "scale": round(self.scale, 5),
                "corr": round(self.corr, 3), "contrast": round(self.contrast, 1)}


def fit_score_timing(wave: np.ndarray, sr: int, onsets: Sequence[Tuple[float, float]],
                     score_seconds: float, scales: Optional[np.ndarray] = None,
                     max_offset_s: float = 30.0) -> Timing:
    """Partitur-Einsaetze gegen die Onset-Huellkurve des Renderings legen.

    Gemessen an echten YuE2-Renderings: Tempo-Faktor 1.000, Versatz 0.3-2.5 s,
    danach ein natuerlicher Ausklang. Audio-gegen-Audio scheitert dagegen an
    repetitiver Musik (jeder Chorus sieht gleich aus); die ganze Partitur als
    Muster ist eindeutig.
    """
    env = onset_envelope(wave, sr)
    env = env - env.mean()
    env_norm = np.linalg.norm(env) + 1e-9
    if scales is None:
        scales = np.arange(0.94, 1.0601, 0.002)
    kernel = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    max_lag = int(max_offset_s / HOP_SECONDS)
    best = (-1.0, 1.0, 0)
    peaks = []
    for s in scales:
        n = int(score_seconds * s / HOP_SECONDS) + 50
        tpl = np.zeros(n)
        for t, w in onsets:
            i = int(round(t * s / HOP_SECONDS))
            if 0 <= i < n:
                tpl[i] += w
        tpl = np.convolve(tpl, kernel, "same")
        tpl = tpl - tpl.mean()
        L = len(env) + n
        corr = np.fft.irfft(np.fft.rfft(env, L) * np.conj(np.fft.rfft(tpl, L)), L)
        corr = corr[:max(1, min(max_lag, len(env)))] / (np.linalg.norm(tpl) * env_norm + 1e-9)
        i = int(np.argmax(corr))
        peaks.append(corr[i])
        if corr[i] > best[0]:
            best = (float(corr[i]), float(s), i)
    med = float(np.median(np.abs(peaks))) + 1e-9
    return Timing(offset=best[2] * HOP_SECONDS, scale=best[1], corr=best[0], contrast=best[0] / med)


def grid_phase(wave: np.ndarray, sr: int, end: int, period_s: float,
               window_s: float = 6.0) -> Tuple[float, float]:
    """Lage der Anschlaege auf dem Raster (z. B. Achtel) in den Sekunden vor ``end``.

    Rueckgabe (Phase in Sekunden relativ zu ``end``, Konsistenz 0..1). Die Phase
    ist der Kreismittelwert der Anschlagszeiten modulo Rasterlaenge.
    """
    start = max(0, end - int(window_s * sr))
    env = onset_envelope(wave[:, start:end], sr)
    if env.size < 10:
        return 0.0, 0.0
    thr = np.percentile(env, 85)
    edge = 5   # 50 ms: am Fensterrand erzeugt das Padding Schein-Einsaetze
    peaks = [i for i in range(edge, len(env) - edge)
             if env[i] > thr and env[i] >= env[i - 1] and env[i] >= env[i + 1]]
    if len(peaks) < 4:
        return 0.0, 0.0
    # Spitzen per Parabel zwischen die 10-ms-Frames legen, nach Staerke gewichten
    pos, weight = [], []
    for i in peaks:
        a, b, c = env[i - 1], env[i], env[i + 1]
        den = a - 2 * b + c
        pos.append(i + (0.5 * (a - c) / den if den < 0 else 0.0))
        weight.append(b)
    t = np.array(pos) * HOP_SECONDS - (end - start) / sr
    w = np.array(weight) / (np.sum(weight) + 1e-12)
    z = np.sum(w * np.exp(2j * np.pi * (t % period_s) / period_s))
    return float(np.angle(z) / (2 * np.pi) * period_s), float(abs(z))


def refine_cut(prev: np.ndarray, new: np.ndarray, sr: int, prev_cut: int, new_cut: int,
               period_s: float, window_s: float = 6.0, min_consistency: float = 0.6) -> Tuple[int, float]:
    """Neuen Schnitt so nachziehen, dass die Anschlaege auf demselben Raster weiterlaufen.

    Die Partitur-Ausrichtung ist global genau, aber YuE2 spielt lokal leicht vor
    oder hinter dem Raster (gemessen: 60 ms an einer Naht - hoerbar als Stolpern).
    Verglichen wird die Rasterphase der letzten Sekunden vor dem Schnitt - in
    beiden Renderings dieselbe Partiturstelle, also derselbe Rhythmus. Die
    Korrektur ist auf eine halbe Rasterlaenge begrenzt, kann also nicht auf den
    Nachbarschlag rutschen; bei unklarem Raster (Konsistenz zu klein) bleibt der
    Schnitt, wo die Partitur ihn hinlegt.
    Rueckgabe: (Verschiebung von new_cut in Samples, kleinere der beiden Konsistenzen).
    """
    pp, cp = grid_phase(prev, sr, prev_cut, period_s, window_s)
    pn, cn = grid_phase(new, sr, new_cut, period_s, window_s)
    conf = min(cp, cn)
    if conf < min_consistency:
        return 0, conf
    shift = ((pn - pp + period_s / 2) % period_s) - period_s / 2
    return int(round(shift * sr)), conf


def band_profile(wave: np.ndarray, sr: int, start: int, end: int, bands: int = 24) -> np.ndarray:
    """Klangfarbe eines Ausschnitts: log. Energie in Frequenzbaendern, pegelnormiert (dB)."""
    mono = np.asarray(wave, dtype=np.float32)
    mono = mono.mean(axis=0) if mono.ndim == 2 else mono
    seg = mono[max(0, start):max(0, end)]
    if seg.size < 2048:
        return np.zeros(bands)
    n = 2048
    frames = seg[: (len(seg) // n) * n].reshape(-1, n) * np.hanning(n)[None, :]
    spec = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    edges = np.geomspace(60.0, min(16000.0, sr / 2 - 1), bands + 1)
    prof = np.array([spec[(freqs >= a) & (freqs < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    prof = 10 * np.log10(prof + 1e-12)
    return prof - prof.max()


def choose_cut(prev: np.ndarray, new: np.ndarray, sr: int,
               candidates: Sequence[Tuple[int, int]], window_s: float = 2.0) -> Tuple[int, float]:
    """Unter mehreren Taktstrichen den waehlen, an dem beide Einspielungen gleich klingen.

    Zwei YuE2-Renderings arrangieren dieselben Takte unterschiedlich dicht
    (gemessen: im neuen Teil fielen an einer Stelle die Instrumente fuer 1 s weg,
    waehrend das Original durchspielte). Verglichen wird die Klangfarbe je
    ``window_s`` vor und nach dem Kandidaten; der Pegel selbst wird spaeter
    angeglichen und zaehlt hier nicht. ``candidates`` = [(prev_sample, new_sample)].
    Rueckgabe: (Index des besten Kandidaten, seine Abweichung in dB).
    """
    w = int(window_s * sr)
    best_i, best_d = 0, float("inf")
    for i, (p, q) in enumerate(candidates):
        d = 0.0
        for (a0, a1), (b0, b1) in (((p - w, p), (q - w, q)), ((p, p + w), (q, q + w))):
            d += float(np.mean(np.abs(band_profile(prev, sr, a0, a1) - band_profile(new, sr, b0, b1))))
            # Dichte: Anteil sehr leiser 50-ms-Stuecke (Loecher) muss zusammenpassen
            d += 20.0 * abs(_hole_share(prev, sr, a0, a1) - _hole_share(new, sr, b0, b1))
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


def _hole_share(wave: np.ndarray, sr: int, start: int, end: int) -> float:
    mono = np.asarray(wave, dtype=np.float32)
    mono = mono.mean(axis=0) if mono.ndim == 2 else mono
    seg = mono[max(0, start):max(0, end)]
    hop = max(1, int(0.05 * sr))
    if seg.size < hop * 4:
        return 0.0
    r = np.array([np.sqrt(np.mean(seg[i:i + hop] ** 2)) for i in range(0, len(seg) - hop, hop)])
    ref = np.median(r) + 1e-9
    return float(np.mean(r < ref * 0.05))      # mehr als 26 dB unter dem Median


def rms(wave: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)) + 1e-12))


def body_level_db(wave: np.ndarray, sr: int, start: int = 0, end: Optional[int] = None) -> float:
    """Typische Lautheit eines Abschnitts: Median der 0,4-s-RMS-Werte, ohne die leisesten 20 %.

    Pausen und Ausklang zaehlen so nicht mit - verglichen wird "wie laut spielt die
    Musik", nicht "wie viel Stille ist drin".
    """
    mono = np.asarray(wave, dtype=np.float32)
    mono = mono.mean(axis=0) if mono.ndim == 2 else mono
    mono = mono[start:end]
    hop = max(1, int(0.4 * sr))
    vals = np.array([np.sqrt(np.mean(mono[i:i + hop] ** 2)) for i in range(0, len(mono) - hop, hop)])
    if vals.size == 0:
        return -120.0
    vals = vals[vals >= np.percentile(vals, 20)]
    return float(20 * np.log10(np.median(vals) + 1e-12))


def reference_gain(ref_db: float, part_db: float, lo: float = 0.5, hi: float = 2.0) -> float:
    """Verstaerkung, die einen Teil auf die Referenz-Lautheit des Original-Songs bringt.

    Gemessen: ein lokaler Angleich an den Vorgaenger (wenige Sekunden vor dem
    Schnitt) erzeugte +2.8 dB Spruenge, weil YuE2 jeden Teil mit eigener Dynamik
    spielt (das Ende eines Renderings leiser, dieselbe Stelle als Ueberleitung
    lauter). Der Bezug auf die Lautheit des Originals ergab +0.5/-1.5 dB, im
    Bereich der natuerlichen Taktgrenzen im Original, und schaukelt sich ueber
    viele Teile nicht auf.
    """
    return float(min(hi, max(lo, 10 ** ((ref_db - part_db) / 20.0))))


# ---------------------------------------------------------------- Zusammensetzen

@dataclass
class Segment:
    wave: np.ndarray       # [2, N], bereits im Ziel-Samplerate
    start: int
    end: Optional[int]     # None = bis Dateiende
    gain: float = 1.0      # Zielpegel des Teils
    gain_start: Optional[float] = None   # Pegel am Schnitt (an den Vorgaenger angeglichen)
    ramp_end: Optional[int] = None       # Sample (Dateiposition), ab dem ``gain`` gilt

    def gain_curve(self, s: int, e: int) -> np.ndarray:
        """Gain je Sample in [s, e): am Schnitt ``gain_start``, bis ``ramp_end`` linear in dB zu ``gain``."""
        if self.gain_start is None or self.ramp_end is None or self.ramp_end <= self.start:
            return np.full(e - s, self.gain, dtype=np.float32)
        pos = np.arange(s, e, dtype=np.float64)
        frac = np.clip((pos - self.start) / (self.ramp_end - self.start), 0.0, 1.0)
        db = 20 * np.log10(self.gain_start) * (1 - frac) + 20 * np.log10(self.gain) * frac
        return (10 ** (db / 20)).astype(np.float32)

    def slice(self, pad_before: int = 0, pad_after: int = 0) -> np.ndarray:
        n = self.wave.shape[-1]
        s = max(0, self.start - pad_before)
        e = n if self.end is None else min(n, self.end + pad_after)
        return self.wave[:, s:e] * self.gain_curve(s, e)[None, :]


def join_segments(segments: Sequence[Segment], sr: int, crossfade_ms: float = 120.0,
                  end_fade_s: float = 0.0) -> np.ndarray:
    """Segmente mit Equal-Power-Crossfade zentriert auf den Schnittpunkten verbinden."""
    xf = max(0, int(round(sr * crossfade_ms / 1000.0)))
    half = xf // 2
    out: Optional[np.ndarray] = None
    for i, seg in enumerate(segments):
        pad_before = half if i > 0 else 0
        pad_after = half if i < len(segments) - 1 else 0
        piece = seg.slice(pad_before, pad_after)
        if out is None:
            out = piece
            continue
        n = min(xf, out.shape[-1], piece.shape[-1])
        if n <= 0:
            out = np.concatenate([out, piece], axis=-1)
            continue
        t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
        fade_out, fade_in = np.cos(t), np.sin(t)
        mixed = out[:, -n:] * fade_out + piece[:, :n] * fade_in
        out = np.concatenate([out[:, :-n], mixed, piece[:, n:]], axis=-1)
    if out is None:
        return np.zeros((2, 0), dtype=np.float32)
    if end_fade_s > 0:
        n = min(out.shape[-1], int(round(sr * end_fade_s)))
        if n > 0:
            out = out.copy()
            out[:, -n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32) ** 2
    return out.astype(np.float32)


def parse_arrangement(text: str, count: int) -> List[int]:
    """``alle`` / ``all`` / leer -> 0..count-1; sonst ``0,1,2,2,3`` oder ``0-3,2``."""
    text = (text or "").strip().lower()
    if text in ("", "alle", "all"):
        return list(range(count))
    order: List[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        m = part.split("x")
        rep = 1
        if len(m) == 2 and m[1].strip().isdigit():    # "2x3" = Teil 2 dreimal
            part, rep = m[0].strip(), int(m[1])
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            items = list(range(a, b + 1))
        else:
            items = [int(part)]
        for _ in range(rep):
            order += items
    bad = [i for i in order if not 0 <= i < count]
    if bad:
        raise ValueError(f"Arrangement nennt Teile {bad}, vorhanden sind 0..{count - 1}.")
    return order


def arrangement_segments(parts: Sequence[Tuple[np.ndarray, dict]], order: Sequence[int]) -> List[Segment]:
    """Segmente fuer eine Reihenfolge.

    ``parts[i] = (wave, record)``; ``record`` hat ``start`` (Schnitt vom
    Vorgaenger), ``end`` (Schnitt zum echten Nachfolger, None beim letzten Teil),
    ``score_end`` (Ende des notierten Materials) und ``gain``.

    - Folgt in der Reihenfolge der echte Nachfolger, wird am gemessenen Schnitt
      getrennt.
    - Die letzte Position laeuft bis Dateiende: dort liegt das natuerliche Ende,
      das YuE2 dem Teil gerendert hat.
    - Bei Spruengen/Loops endet ein Teil am Ende seines notierten Materials, damit
      kein Ausklang in die Wiederholung hineinklingt.
    """
    segs: List[Segment] = []
    for pos, idx in enumerate(order):
        wave, rec = parts[idx]
        if pos == len(order) - 1:
            end = None
        elif order[pos + 1] == idx + 1 and rec.get("end") is not None:
            end = rec["end"]
        else:
            end = rec.get("score_end", rec.get("end"))
        natural = pos > 0 and order[pos - 1] == idx - 1     # echter Vorgaenger davor
        segs.append(Segment(wave=wave, start=int(rec.get("start", 0)),
                            end=None if end is None else int(end),
                            gain=float(rec.get("gain", 1.0)),
                            # Rampe nur hinter dem echten Vorgaenger - an ihn ist gain_start angeglichen
                            gain_start=rec.get("gain_start") if natural else None,
                            ramp_end=rec.get("ramp_end") if natural else None))
    return segs
