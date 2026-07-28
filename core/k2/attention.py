"""K2 Lab — räumlicher Attention-Router für Krea 2.

Krea 2 ist ein Single-Stream-MMDiT: Text- und Bildtoken liegen in *einer*
Sequenz ``[text | image]``. Dadurch lässt sich regionales Prompting ohne zweiten
Sampling-Pass erreichen, indem die Attention-Logits gezielt verschoben werden:

* **weicher Bias** — Tokenpaare (Regionsklausel ↔ Bildtoken in ihrer Box) werden
  angehoben, Paare außerhalb abgesenkt;
* **harte Partition** (strict isolation) — subjektspezifische Texttoken sind für
  fremde Subjekte und für Bildtoken außerhalb ihrer Box gesperrt. Damit kann ein
  regionaler LoRA-Delta auf dem Textpfad nicht in eine andere Region auslaufen.

Bild-zu-Bild-Attention bleibt unangetastet, sonst entstünden Kachelkanten.

Der Router hängt sich in ComfyUIs ``optimized_attention_override`` ein und baut
eine additive Maske, die die installierte Attention-Implementierung selbst
verrechnet (SDPA verarbeitet additive Float-Masken nativ). Für sehr große
Sequenzen gibt es einen speicherschonenden Chunk-Pfad.
"""

from __future__ import annotations

import numpy as np
import torch

from .binding import BoundPlan
from .geometry import spatial_pair_bias
from .prompt import ROLE_SUBJECT

# Ab dieser Maskengröße (in Elementen) wird auf den Chunk-Pfad gewechselt.
DENSE_MASK_ELEMENT_LIMIT = 160_000_000  # ~320 MB bei fp16


def text_region_owners(bound: BoundPlan) -> np.ndarray:
    """0 = geteilter Text, k = exklusiv Subjekt k."""
    owners = np.zeros(bound.text_token_count, dtype=np.int16)
    owner = 0
    for span in bound.spans:
        if span.role != ROLE_SUBJECT:
            continue
        owner += 1
        owners[span.start : span.end] = owner
    return owners


def image_region_owners(bound: BoundPlan) -> np.ndarray:
    """Jedes Bildtoken in einer Subjektbox gehört dem erstplatzierten Subjekt."""
    owners = np.zeros(bound.image_token_count, dtype=np.int16)
    owner = 0
    for span in bound.spans:
        if span.role != ROLE_SUBJECT:
            continue
        owner += 1
        claim = (span.mask > 0.0) & (owners == 0)
        owners[claim] = owner
    return owners


class K2SpatialAttention:
    """Callable für ``transformer_options["optimized_attention_override"]``."""

    def __init__(
        self,
        bound: BoundPlan,
        *,
        strict_isolation: bool = True,
        lora_delta_adaptation: bool = False,
        lora_delta_adaptation_gain: float = 0.35,
        query_chunk_size: int = 256,
        dense_limit: int = DENSE_MASK_ELEMENT_LIMIT,
    ) -> None:
        self.bound = bound
        self.strict_isolation = bool(strict_isolation)
        self.lora_delta_adaptation = bool(lora_delta_adaptation)
        self.lora_delta_adaptation_gain = float(lora_delta_adaptation_gain)
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size muss positiv sein")
        self.query_chunk_size = int(query_chunk_size)
        self.dense_limit = int(dense_limit)

        self.text_count = bound.text_token_count
        self.image_count = bound.image_token_count
        self.sequence_length = bound.sequence_length

        self.text_owners = text_region_owners(bound)
        self.image_owners = image_region_owners(bound)

        self.step_scale = 1.0
        self.region_scales: dict[str, float] = {}
        self._scale_version = 0

        self.main_calls = 0
        self.refiner_calls = 0
        self.dense_path_used = False
        self.chunk_path_used = False

        self._main_cache: tuple | None = None
        self._refiner_cache: tuple | None = None
        self._np_cache: dict[str, np.ndarray] = {}

    # ── Laufzeitsteuerung ────────────────────────────────────────────────

    def set_denoising_progress(self, completed: int, total: int) -> None:
        """Platzierung früh hart, spät gelockert — sonst wirken Kanten aufgeklebt."""
        if total <= 0:
            raise ValueError("Gesamtschrittzahl muss positiv sein")
        progress = min(1.0, max(0.0, completed / total))
        relaxation_start = 0.55
        if progress <= relaxation_start:
            new_scale = 1.0
        else:
            fraction = (progress - relaxation_start) / (1.0 - relaxation_start)
            new_scale = 1.0 + fraction * (self.bound.plan.late_step_scale - 1.0)
        if new_scale != self.step_scale:
            self.step_scale = new_scale
            self._scale_version += 1

    def set_region_scales(self, scales: dict[str, float]) -> None:
        if not self.lora_delta_adaptation:
            return
        known = {span.region_id for span in self.bound.spans}
        clamped = {
            region_id: min(1.5, max(0.5, float(value)))
            for region_id, value in scales.items()
            if region_id in known
        }
        if clamped != self.region_scales:
            self.region_scales = clamped
            self._scale_version += 1

    def clear(self) -> None:
        self._main_cache = None
        self._refiner_cache = None
        self._np_cache.clear()

    # ── Attention-Hook ───────────────────────────────────────────────────

    def __call__(self, original, *args, **kwargs):
        q, k = args[0], args[1]
        q_len = int(q.shape[-2])
        k_len = int(k.shape[-2])

        main_stream = q_len == self.sequence_length and k_len == self.sequence_length
        # Die ersten beiden txtfusion-Blöcke attendieren über die 12 Qwen-Layer;
        # dort ist die Sequenz die Layerachse und der Prompt liegt im Batch.
        folded_layerwise = (
            q_len == 12
            and k_len == 12
            and (q_len != self.text_count or int(q.shape[0]) % max(self.text_count, 1) == 0)
        )
        refiner = (
            self.strict_isolation
            and q_len == self.text_count
            and k_len == self.text_count
            and not folded_layerwise
        )
        if not main_stream and not refiner:
            return original(*args, **kwargs)
        if kwargs.get("mask") is not None or (len(args) > 4 and args[4] is not None):
            # Ein anderer Knoten hält bereits eine Maske auf diesem Zweig.
            return original(*args, **kwargs)
        if q.ndim != 4:
            return original(*args, **kwargs)

        v = args[2]
        heads = args[3] if len(args) > 3 else kwargs.get("heads")
        dense_elements = q_len * k_len
        use_dense = dense_elements <= self.dense_limit

        if main_stream:
            self.main_calls += 1
        else:
            self.refiner_calls += 1

        if use_dense:
            mask = (
                self._main_mask(q, kwargs.get("transformer_options"))
                if main_stream
                else self._refiner_mask(q)
            )
            self.dense_path_used = True
            new_kwargs = dict(kwargs)
            new_kwargs["mask"] = mask
            return original(*args, **new_kwargs)

        self.chunk_path_used = True
        scale = float(kwargs.get("scale", q.shape[-1] ** -0.5))
        out = self._chunked_attention(q, k, v, scale, main_stream=main_stream)
        if kwargs.get("skip_output_reshape", False):
            return out
        return out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)

    # ── Maskenbau ────────────────────────────────────────────────────────

    def _neg(self, dtype: torch.dtype) -> float:
        return torch.finfo(dtype).min / 2.0

    def _region_bias_arrays(self) -> list[np.ndarray]:
        """Additiver Bias pro Region über alle Bildtoken (ohne Skalen)."""
        cached = self._np_cache.get("region_bias")
        if cached is not None:
            return list(cached)
        plan = self.bound.plan
        arrays = []
        for span in self.bound.spans:
            penalty = plan.outside_penalty * (1.0 if span.role == ROLE_SUBJECT else 0.25)
            arrays.append(
                spatial_pair_bias(span.field.astype(np.float64), plan.strength, penalty)
            )
        self._np_cache["region_bias"] = arrays
        return arrays

    def _build_main_mask(self, device, dtype) -> torch.Tensor:
        seq = self.sequence_length
        text = self.text_count
        mask = torch.zeros((seq, seq), device=device, dtype=torch.float32)

        biases = self._region_bias_arrays()
        for span, bias in zip(self.bound.spans, biases):
            scale = self.step_scale * self.region_scales.get(span.region_id, 1.0)
            if scale == 0.0:
                continue
            row = torch.as_tensor(
                bias * scale, device=device, dtype=torch.float32
            )
            # Text-Query der Klausel → alle Bildtoken
            mask[span.start : span.end, text:] += row.unsqueeze(0)
            # Bild-Query → Texttoken der Klausel
            mask[text:, span.start : span.end] += row.unsqueeze(1)

        for emphasis in self.bound.emphases:
            scale = self.step_scale * emphasis.strength
            if scale == 0.0:
                continue
            row = torch.as_tensor(
                emphasis.field.astype(np.float64) * scale,
                device=device,
                dtype=torch.float32,
            )
            mask[text:, emphasis.start : emphasis.end] += row.unsqueeze(1)

        if self.strict_isolation:
            neg = self._neg(dtype)
            text_owners = torch.as_tensor(self.text_owners, device=device)
            image_owners = torch.as_tensor(self.image_owners, device=device)
            owners = torch.cat([text_owners, image_owners])

            # Query-Owner: Text erbt seinen Klausel-Owner, Bild seinen Boxbesitzer.
            query_owners = owners
            # Gesperrt sind fremde *Text*-Keys für alle Queries …
            blocked_text = (text_owners.reshape(1, -1) > 0) & (
                query_owners.reshape(-1, 1) != text_owners.reshape(1, -1)
            )
            mask[:, :text].masked_fill_(blocked_text, neg)
            # … und fremde Bild-Keys nur für Text-Queries (Bild↔Bild bleibt offen).
            blocked_images = (image_owners.reshape(1, -1) > 0) & (
                text_owners.reshape(-1, 1) != image_owners.reshape(1, -1)
            )
            mask[:text, text:].masked_fill_(blocked_images, neg)

        return mask.to(dtype)

    def _build_refiner_mask(self, device, dtype) -> torch.Tensor:
        text = self.text_count
        mask = torch.zeros((text, text), device=device, dtype=torch.float32)
        owners = torch.as_tensor(self.text_owners, device=device)
        blocked = (owners.reshape(1, -1) > 0) & (
            owners.reshape(-1, 1) != owners.reshape(1, -1)
        )
        mask.masked_fill_(blocked, self._neg(dtype))
        return mask.to(dtype)

    def _main_mask(self, reference: torch.Tensor, transformer_options) -> torch.Tensor:
        key = (reference.device, reference.dtype, self._scale_version)
        if self._main_cache is not None and self._main_cache[0] == key:
            base = self._main_cache[1]
        else:
            base = self._build_main_mask(reference.device, reference.dtype)
            self._main_cache = (key, base)
        return self._batch_mask(base, reference, transformer_options)

    def _refiner_mask(self, reference: torch.Tensor) -> torch.Tensor:
        key = (reference.device, reference.dtype)
        if self._refiner_cache is not None and self._refiner_cache[0] == key:
            return self._refiner_cache[1]
        mask = self._build_refiner_mask(reference.device, reference.dtype)
        self._refiner_cache = (key, mask)
        return mask

    @staticmethod
    def _batch_mask(base: torch.Tensor, reference: torch.Tensor, transformer_options):
        """Neutralisiert die Maske für Uncond-Slots, wenn CFG > 1 aktiv ist.

        Der negative Prompt hat andere Tokenspannen; die regionale Maske darf ihn
        deshalb nicht treffen. Bei gleicher Sequenzlänge lässt sich das nur über
        ``cond_or_uncond`` unterscheiden.
        """
        if not isinstance(transformer_options, dict):
            return base
        groups = transformer_options.get("cond_or_uncond")
        if not groups or all(int(g) == 0 for g in groups):
            return base
        batch = int(reference.shape[0])
        n_groups = len(groups)
        if n_groups == 0 or batch % n_groups:
            return base
        per_group = batch // n_groups
        stacked = torch.zeros(
            (n_groups, 1, base.shape[0], base.shape[1]),
            device=base.device,
            dtype=base.dtype,
        )
        for index, group in enumerate(groups):
            if int(group) == 0:
                stacked[index, 0] = base
        if per_group == 1:
            return stacked
        return stacked.repeat_interleave(per_group, dim=0)

    # ── Chunk-Pfad (speicherarm) ─────────────────────────────────────────

    def _chunked_attention(self, q, k, v, scale: float, *, main_stream: bool):
        out = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], v.shape[-1]),
            dtype=v.dtype,
            device=v.device,
        )
        mask = (
            self._main_mask(q, None) if main_stream else self._refiner_mask(q)
        ).float()
        key_t = k.transpose(-2, -1)
        for start in range(0, q.shape[-2], self.query_chunk_size):
            end = min(q.shape[-2], start + self.query_chunk_size)
            scores = torch.matmul(q[:, :, start:end], key_t).float() * scale
            scores += mask[start:end].reshape(1, 1, end - start, -1)
            probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
            out[:, :, start:end] = torch.matmul(probabilities, v)
            del scores, probabilities
        return out

    # ── Diagnose ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "backend": self.bound.plan.backend,
            "strict_isolation": self.strict_isolation,
            "main_stream_attention_calls": self.main_calls,
            "text_refiner_attention_calls": self.refiner_calls,
            "path": "dense_additive_mask" if self.dense_path_used else "chunked",
            "chunk_fallback_used": self.chunk_path_used,
            "image_to_image_attention": "unmodified",
            "late_step_scale": self.bound.plan.late_step_scale,
            "final_step_scale": self.step_scale,
            "lora_delta_adaptation": self.lora_delta_adaptation,
            "final_region_scales": dict(self.region_scales),
            "subject_text_lanes": int((self.text_owners > 0).sum()),
            "subject_image_lanes": int((self.image_owners > 0).sum()),
        }


__all__ = ["K2SpatialAttention", "image_region_owners", "text_region_owners"]
