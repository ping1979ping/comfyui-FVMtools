"""K2 Lab — Canvas-Geometrie und Bildtoken-Felder.

Krea 2 patchifiziert das Latent mit patch=2 auf einem VAE mit Faktor 8, also
entspricht **ein Bildtoken genau 16x16 Ausgabepixeln**. Alle Regionen werden in
Ausgabepixeln definiert und hier auf dieses Tokenraster abgebildet.

Reine Mathematik — kein torch, kein ComfyUI. Damit unit-testbar ohne Server.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

VAE_SCALE = 8
PATCH_SIZE = 2
TOKEN_PIXELS = VAE_SCALE * PATCH_SIZE  # 16


def align_up(value: int, alignment: int) -> int:
    if value <= 0:
        raise ValueError("Wert muss positiv sein")
    if alignment <= 0:
        raise ValueError("Alignment muss positiv sein")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class PixelBox:
    """Halboffenes Rechteck in Ausgabepixeln: [x0, x1) x [y0, y1)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not all(isfinite(v) for v in (self.x0, self.y0, self.x1, self.y1)):
            raise ValueError("Box-Koordinaten müssen endlich sein")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(
                f"Box braucht positive Breite/Höhe, bekam ({self.x0},{self.y0})-({self.x1},{self.y1})"
            )

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def clipped(self, width: int, height: int) -> "PixelBox":
        if width <= 0 or height <= 0:
            raise ValueError("Canvas-Maße müssen positiv sein")
        x0 = min(max(self.x0, 0.0), float(width))
        y0 = min(max(self.y0, 0.0), float(height))
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Box liegt vollständig außerhalb der Canvas")
        return PixelBox(x0, y0, x1, y1)

    def grown(self, pixels: float) -> "PixelBox":
        return PixelBox(
            self.x0 - pixels, self.y0 - pixels, self.x1 + pixels, self.y1 + pixels
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "PixelBox":
        return cls(float(x), float(y), float(x) + float(w), float(y) + float(h))


@dataclass(frozen=True)
class CanvasGeometry:
    """Bildmaße plus abgeleitetes Krea-Bildtokenraster."""

    requested_width: int
    requested_height: int
    aligned_width: int
    aligned_height: int

    @classmethod
    def resolve(cls, width: int, height: int) -> "CanvasGeometry":
        return cls(
            requested_width=int(width),
            requested_height=int(height),
            aligned_width=align_up(int(width), TOKEN_PIXELS),
            aligned_height=align_up(int(height), TOKEN_PIXELS),
        )

    @property
    def token_width(self) -> int:
        return self.aligned_width // TOKEN_PIXELS

    @property
    def token_height(self) -> int:
        return self.aligned_height // TOKEN_PIXELS

    @property
    def token_count(self) -> int:
        return self.token_width * self.token_height

    def token_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-Mittelpunkte aller Bildtoken als (ys, xs) im Rasterlayout."""
        ys = (np.arange(self.token_height, dtype=np.float64) + 0.5) * TOKEN_PIXELS
        xs = (np.arange(self.token_width, dtype=np.float64) + 0.5) * TOKEN_PIXELS
        return ys, xs

    # ── Rasterisierung ────────────────────────────────────────────────────

    def rasterize_box(self, box: PixelBox) -> np.ndarray:
        """Harte Boxmaske: Überdeckungsanteil jedes 16x16-Tokens (0..1), flach."""
        clipped = box.clipped(self.aligned_width, self.aligned_height)
        mask = np.zeros((self.token_height, self.token_width), dtype=np.float32)

        col0 = max(0, int(clipped.x0 // TOKEN_PIXELS))
        col1 = min(self.token_width - 1, int((clipped.x1 - 1e-9) // TOKEN_PIXELS))
        row0 = max(0, int(clipped.y0 // TOKEN_PIXELS))
        row1 = min(self.token_height - 1, int((clipped.y1 - 1e-9) // TOKEN_PIXELS))
        if col1 < col0 or row1 < row0:
            return mask.reshape(-1)

        cols = np.arange(col0, col1 + 1)
        rows = np.arange(row0, row1 + 1)
        tx0 = cols * TOKEN_PIXELS
        tx1 = tx0 + TOKEN_PIXELS
        ty0 = rows * TOKEN_PIXELS
        ty1 = ty0 + TOKEN_PIXELS
        ow = np.clip(np.minimum(tx1, clipped.x1) - np.maximum(tx0, clipped.x0), 0, None)
        oh = np.clip(np.minimum(ty1, clipped.y1) - np.maximum(ty0, clipped.y0), 0, None)
        mask[row0 : row1 + 1, col0 : col1 + 1] = np.outer(oh, ow) / (
            TOKEN_PIXELS * TOKEN_PIXELS
        )
        return mask.reshape(-1)

    def _outside_distance(self, box: PixelBox) -> np.ndarray:
        """Euklidischer Abstand jedes Tokenmittelpunkts zur Box (0 innerhalb)."""
        ys, xs = self.token_centers()
        dx = np.maximum.reduce([box.x0 - xs, np.zeros_like(xs), xs - box.x1])
        dy = np.maximum.reduce([box.y0 - ys, np.zeros_like(ys), ys - box.y1])
        return np.hypot(dy[:, None], dx[None, :])

    @staticmethod
    def _smoothstep(u: np.ndarray) -> np.ndarray:
        return u * u * (3.0 - 2.0 * u)

    def soft_box_field(self, box: PixelBox, falloff_pixels: float) -> np.ndarray:
        """Weiches Feld: 1.0 in der Box, smoothstep-Abfall über `falloff_pixels`.

        Für Hintergrundregionen — sie dürfen über ihre Box hinaus ausfransen.
        """
        clipped = box.clipped(self.aligned_width, self.aligned_height)
        distance = self._outside_distance(clipped)
        field = np.zeros_like(distance)
        inside = distance == 0.0
        field[inside] = 1.0
        if falloff_pixels > 0.0:
            edge = (~inside) & (distance < falloff_pixels)
            u = 1.0 - distance[edge] / falloff_pixels
            field[edge] = self._smoothstep(u)
        return field.reshape(-1).astype(np.float32)

    def subject_target_field(
        self, box: PixelBox, falloff_pixels: float, *, edge_weight: float
    ) -> np.ndarray:
        """Subjekt-Feld: Peak in der Boxmitte, `edge_weight` am Boxrand.

        `edge_weight` nahe 1.0 (subject_fill) hält das Feld bis zum Rand stark und
        vermeidet unbeanspruchte Leerflächen im Subjektkasten.
        """
        clipped = box.clipped(self.aligned_width, self.aligned_height)
        ys, xs = self.token_centers()
        distance = self._outside_distance(clipped)
        field = np.zeros_like(distance)

        mid_x, mid_y = clipped.center
        half_w = max(clipped.width / 2.0, 1e-9)
        half_h = max(clipped.height / 2.0, 1e-9)
        norm = np.maximum(
            np.abs(xs[None, :] - mid_x) / half_w, np.abs(ys[:, None] - mid_y) / half_h
        )
        inside = distance == 0.0
        u_in = np.clip(norm[inside], 0.0, 1.0)
        field[inside] = 1.0 - (1.0 - edge_weight) * self._smoothstep(u_in)

        if falloff_pixels > 0.0:
            edge = (~inside) & (distance < falloff_pixels)
            u = 1.0 - distance[edge] / falloff_pixels
            field[edge] = edge_weight * self._smoothstep(u)
        return field.reshape(-1).astype(np.float32)


def apply_subject_competition(
    fields: list[np.ndarray], roles: list[str]
) -> list[np.ndarray]:
    """Überlappende Subjektfelder teilen sich Token proportional zur Feldstärke².

    Ohne diesen Schritt beanspruchen zwei überlappende Subjekte dasselbe Token
    vollständig und verschmelzen zu einer Person.
    """
    subject_idx = [i for i, role in enumerate(roles) if role == "subject"]
    if len(subject_idx) < 2:
        return fields
    stacked = np.stack([fields[i] for i in subject_idx], axis=0).astype(np.float64)
    squared = stacked**2
    denom = squared.sum(axis=0)
    safe = denom > 0.0
    ownership = np.zeros_like(squared)
    ownership[:, safe] = squared[:, safe] / denom[safe]
    result = list(fields)
    for slot, index in enumerate(subject_idx):
        result[index] = (stacked[slot] * ownership[slot]).astype(np.float32)
    return result


def spatial_pair_bias(
    field: np.ndarray, strength: float, outside_penalty: float
) -> np.ndarray:
    """Weiches Feld → additiver Attention-Logit-Bias.

    Feldwert 1.0 → +strength, Feldwert 0.0 → -outside_penalty. Dazwischen linear.
    """
    return (strength + outside_penalty) * field - outside_penalty


__all__ = [
    "PATCH_SIZE",
    "TOKEN_PIXELS",
    "VAE_SCALE",
    "CanvasGeometry",
    "PixelBox",
    "align_up",
    "apply_subject_competition",
    "spatial_pair_bias",
]
