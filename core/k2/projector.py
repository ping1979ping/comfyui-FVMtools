"""K2 Lab — Kontrolle über Krea 2s ``txtfusion.projector``.

Krea 2 konditioniert auf 12 Hidden-State-Ebenen von Qwen3-VL. ``txtfusion.projector``
ist ein ``Linear(12 → 1)`` ohne Bias, das diese 12 Ebenen zu einem Textvektor
mischt. Ein Delta auf diese 12 Zahlen verschiebt, *welche* Sprachebene das Bild
dominiert — das ist der stärkste Einzelhebel für Stil- und Semantikverhalten und
die Grundlage der bekannten „projector"-LoRAs.

Die Basisgewichte werden nie überschrieben; das Delta wird als Patch bzw. als
tokenselektiver Bypass-Adapter angewandt.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import numpy as np

PROJECTOR_LENGTH = 12
PROJECTOR_TARGET = "diffusion_model.txtfusion.projector.weight"
CUSTOM_PRESET = "custom"

# Referenztabelle aus veröffentlichten K2Lab-Projekten. Sie ist bewusst als
# Kompatibilitätsschicht erhalten — die exakten Werte einer Projector-LoRA
# liefert `projector_delta_from_lora()`.
PROJECTOR_PRESETS: dict[str, tuple[float, ...]] = {
    "none": (0.0,) * PROJECTOR_LENGTH,
    "filter_bypass2": (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5117, -0.8906, 0.0, 0.0,
    ),
    "filter_bypass3": (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5117, -0.8906, -0.6094, 0.0,
    ),
    "skc3vo": (
        -5.4400, -16.1100, -37.1100, -50.3900, -70.7000, -39.4500,
        -39.8400, -143.7511, -51.1700, -89.0600, -60.9400, -11.2800,
    ),
    "z0jglf": (
        -13.6000, -40.2750, -92.7750, -159.7500, -176.7500, -98.6250,
        -99.6000, -359.3778, -127.9250, -222.6500, -152.3500, -28.2000,
    ),
}

PROJECTOR_PRESET_NAMES = tuple(PROJECTOR_PRESETS) + (CUSTOM_PRESET,)

# Schlüssel, unter denen Projector-Deltas in LoRA-Dateien auftauchen.
_PROJECTOR_KEY_HINTS = (
    "txtfusion.projector",
    "text_fusion.projector",
)


def preset_values(preset: str) -> tuple[float, ...]:
    try:
        return PROJECTOR_PRESETS[preset]
    except KeyError as error:
        raise ValueError(f"Unbekanntes Projector-Preset: {preset!r}") from error


def validate_values(values) -> tuple[float, ...]:
    vector = tuple(float(v) for v in values)
    if len(vector) != PROJECTOR_LENGTH:
        raise ValueError(f"Projector-Vektor braucht {PROJECTOR_LENGTH} Werte")
    if not all(isfinite(v) for v in vector):
        raise ValueError("Projector-Werte müssen endlich sein")
    return vector


def scaled_values(values, multiplier: float) -> tuple[float, ...]:
    vector = validate_values(values)
    scale = float(multiplier)
    if not isfinite(scale):
        raise ValueError("Projector-Multiplier muss endlich sein")
    return tuple(v * scale for v in vector)


def parse_values(text: str) -> tuple[float, ...]:
    """Liest 12 Zahlen aus einem Freitextfeld (Komma, Leerzeichen oder JSON)."""
    cleaned = text.strip().strip("[]()")
    if not cleaned:
        raise ValueError("Projector-Werte sind leer")
    parts = [p for p in cleaned.replace(",", " ").split() if p]
    return validate_values(parts)


def token_delta_mask(
    text_token_count: int,
    protected_spans: tuple[tuple[int, int], ...],
    protection: float,
) -> np.ndarray:
    """Skaliert das Projector-Delta pro Texttoken.

    1.0 = volles Delta, 0.0 = unveränderte Basis-Mischung. Identitäts-Tokens
    (Gesichtsbeschreibungen) lassen sich so vom Projector ausnehmen, damit ein
    aggressiver Stil-Shift kein Gesicht mitverbiegt.
    """
    if text_token_count <= 0:
        raise ValueError("Projector-Tokenmaske braucht eine positive Tokenanzahl")
    amount = float(protection)
    if not 0.0 <= amount <= 1.0:
        raise ValueError("Identity protection muss zwischen 0 und 1 liegen")
    mask = np.ones(text_token_count, dtype=np.float32)
    for start, end in protected_spans:
        if start < 0 or end <= start or end > text_token_count:
            raise ValueError("Geschützte Projector-Spanne liegt außerhalb der Sequenz")
        mask[start:end] = np.minimum(mask[start:end], 1.0 - amount)
    return mask


def projector_delta_from_lora(path: str | Path) -> tuple[float, ...]:
    """Liest das exakte 12-Werte-Delta aus einer Projector-LoRA.

    Unterstützt beide verbreiteten Formate:
    ``…txtfusion.projector.diff`` (direktes Delta) und
    ``…projector.lora_A/​lora_B`` bzw. ``lora_down/​lora_up`` (Rang-1-Adapter).
    """
    import comfy.utils

    state = comfy.utils.load_torch_file(str(path), safe_load=True)
    candidates = {
        key: value
        for key, value in state.items()
        if any(hint in key for hint in _PROJECTOR_KEY_HINTS)
    }
    if not candidates:
        raise ValueError(
            f"{Path(path).name}: kein txtfusion.projector-Tensor gefunden — das ist "
            "keine Projector-LoRA"
        )

    for key, value in candidates.items():
        if key.endswith(".diff"):
            flat = value.flatten().float().tolist()
            if len(flat) != PROJECTOR_LENGTH:
                raise ValueError(
                    f"{key}: erwartet {PROJECTOR_LENGTH} Werte, gefunden {len(flat)}"
                )
            return validate_values(flat)

    down = next(
        (v for k, v in candidates.items() if ".lora_A" in k or ".lora_down" in k), None
    )
    up = next(
        (v for k, v in candidates.items() if ".lora_B" in k or ".lora_up" in k), None
    )
    if down is None or up is None:
        raise ValueError(
            f"{Path(path).name}: unvollständiges Projector-Adapterpaar "
            f"({sorted(candidates)})"
        )
    alpha = next((v for k, v in candidates.items() if k.endswith(".alpha")), None)
    delta = (up.float() @ down.float()).flatten()
    if alpha is not None:
        rank = float(down.shape[0])
        delta = delta * (float(alpha.flatten()[0]) / max(rank, 1.0))
    values = delta.tolist()
    if len(values) != PROJECTOR_LENGTH:
        raise ValueError(
            f"{Path(path).name}: Projector-Delta hat {len(values)} statt "
            f"{PROJECTOR_LENGTH} Werte"
        )
    return validate_values(values)


def looks_like_projector_lora(path: str | Path) -> bool:
    """Header-Schnelltest, ohne die Tensoren zu laden."""
    import json
    import struct

    try:
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
    except Exception:
        return False
    return any(
        any(hint in key for hint in _PROJECTOR_KEY_HINTS)
        for key in header
        if key != "__metadata__"
    )


__all__ = [
    "CUSTOM_PRESET",
    "PROJECTOR_LENGTH",
    "PROJECTOR_PRESETS",
    "PROJECTOR_PRESET_NAMES",
    "PROJECTOR_TARGET",
    "looks_like_projector_lora",
    "parse_values",
    "preset_values",
    "projector_delta_from_lora",
    "scaled_values",
    "token_delta_mask",
    "validate_values",
]
