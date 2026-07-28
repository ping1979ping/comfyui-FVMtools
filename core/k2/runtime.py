"""K2 Lab — Laufzeitobjekt, das an das gepatchte MODEL gehängt wird.

Der Sampler holt es sich über ``model.get_attachment(K2_ATTACHMENT)`` und meldet
den Denoising-Fortschritt zurück. Darüber laufen die späte Lockerung der
räumlichen Bindung und die LoRA-Delta-Adaption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .attention import K2SpatialAttention
from .binding import BoundPlan
from .lora import LoraDeltaStatistics, LoraRoute

K2_ATTACHMENT = "fvm_k2_runtime"


@dataclass
class K2Runtime:
    bound: BoundPlan
    attention: K2SpatialAttention | None = None
    statistics: LoraDeltaStatistics | None = None
    routes: tuple[LoraRoute, ...] = ()
    lora_reports: list[dict] = field(default_factory=list)
    projector_report: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    release_callbacks: tuple[Callable[[], None], ...] = ()

    # ── Sampler-Rückmeldung ──────────────────────────────────────────────

    def update_step(self, completed: int, total: int) -> None:
        if self.attention is None:
            return
        self.attention.set_denoising_progress(completed, total)
        if self.attention.lora_delta_adaptation and self.statistics is not None:
            self.attention.set_region_scales(
                self.statistics.region_scales(
                    self.attention.lora_delta_adaptation_gain
                )
            )
            self.statistics.reset_step()

    def release(self) -> None:
        """Nach dem Sampling GPU-Zustand freigeben (verhindert VRAM-Wachstum).

        Die Delta-Statistik bleibt bewusst erhalten — sie hält nur Skalare und
        wird direkt danach für den Lauf-Report gebraucht.
        """
        if self.attention is not None:
            self.attention.clear()
        for callback in self.release_callbacks:
            try:
                callback()
            except Exception:  # pragma: no cover — Aufräumen darf nie werfen
                pass

    # ── Berichte ─────────────────────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "plan": self.bound.plan.summary(),
            "binding": self.bound.summary(),
            "settings": self.settings,
            "lora_reports": self.lora_reports,
            "projector": self.projector_report,
        }
        if self.attention is not None:
            data["spatial_attention"] = self.attention.summary()
        else:
            data["spatial_attention"] = {"status": "disabled"}
        if self.statistics is not None:
            data["lora_delta_statistics"] = self.statistics.summary()
        return data

    def sanity_warnings(self) -> list[str]:
        """Prüft nach dem Lauf, ob die Hooks tatsächlich gegriffen haben."""
        warnings: list[str] = []
        if self.attention is None:
            return warnings
        if self.attention.main_calls == 0:
            warnings.append(
                "Der räumliche Attention-Router wurde nie aufgerufen — Sequenzlänge "
                "passt nicht (anderes Modell, anderer CLIP oder Auflösung geändert?)."
            )
        if self.attention.strict_isolation and self.attention.refiner_calls == 0:
            warnings.append(
                "Die Textpartition im txtfusion-Refiner wurde nie erreicht; regionale "
                "LoRA-Isolation ist dadurch schwächer."
            )
        return warnings


def as_image_latent(latent):
    """VAE-Encode-Ergebnis auf das 4D-Bildlayout ``[B, C, H, W]`` bringen.

    Krea 2 teilt sich den Autoencoder mit Qwen-Image — und der ist ein 3D-VAE:
    ``encode()`` liefert ``[B, C, 1, H, W]``. Für ein reines Bildmodell muss die
    Zeitachse weg, sonst wandert sie durch Sampler und Decode und das Bild
    kommt als Farbmüll heraus.
    """
    if latent.ndim == 5:
        batch, channels, frames = latent.shape[0], latent.shape[1], latent.shape[2]
        if frames == 1:
            return latent.squeeze(2)
        return latent.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, *latent.shape[3:]
        )
    return latent


def as_image_batch(decoded):
    """VAE-Decode-Ergebnis auf ``[B, H, W, 3]`` bringen (siehe as_image_latent)."""
    if decoded.ndim == 5:
        batch, frames = decoded.shape[0], decoded.shape[1]
        decoded = decoded.reshape(batch * frames, *decoded.shape[2:])
    if decoded.ndim == 4 and decoded.shape[-1] not in (1, 3, 4):
        decoded = decoded.movedim(1, -1)
    return decoded


def union_mask_tensor(bound: BoundPlan, width: int, height: int):
    """Vereinigungsmaske aller Regionen als normaler ComfyUI-MASK-Tensor."""
    import torch

    geometry = bound.plan.geometry
    values = bound.plan.union_field()
    token_mask = torch.from_numpy(
        np.ascontiguousarray(
            values.reshape(geometry.token_height, geometry.token_width)
        )
    ).float()
    return torch.nn.functional.interpolate(
        token_mask.unsqueeze(0).unsqueeze(0),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def region_mask_tensor(mask_values: np.ndarray, bound: BoundPlan, width: int, height: int):
    """Einzelne Regionsmaske (Tokenraster) als MASK in Bildauflösung."""
    import torch

    geometry = bound.plan.geometry
    token_mask = torch.from_numpy(
        np.ascontiguousarray(
            mask_values.reshape(geometry.token_height, geometry.token_width)
        )
    ).float()
    return torch.nn.functional.interpolate(
        token_mask.unsqueeze(0).unsqueeze(0),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


__all__ = [
    "K2_ATTACHMENT",
    "K2Runtime",
    "as_image_batch",
    "as_image_latent",
    "region_mask_tensor",
    "union_mask_tensor",
]
