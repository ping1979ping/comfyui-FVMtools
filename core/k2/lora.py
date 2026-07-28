"""K2 Lab — regionales LoRA-Routing für Krea 2.

Ein normaler LoRA verschiebt die Modellgewichte global: er wirkt auf jedes Token
im Bild. Für „Person A links trägt LoRA X, Person B rechts LoRA Y" braucht es
etwas anderes — **ungefusioneter Delta-Gate**:

1. Der LoRA wird *nicht* in die Basisgewichte eingerechnet, sondern als
   Forward-Adapter installiert (ComfyUIs Bypass-Injection, funktioniert dadurch
   auch auf FP8-/INT8-Gewichten).
2. Sein Delta ``h(x)`` wird pro Token mit einer Maske multipliziert: Texttoken
   der eigenen Klausel und Bildtoken der eigenen Box → 1, alles andere → 0.
3. Zusätzlich lässt der Attention-Router fremde Bildtoken gar nicht erst auf die
   LoRA-modifizierten Textspalten schauen (strict isolation).

Beides zusammen hält den LoRA in einem Denoising-Pass in seiner Region, ohne
Crop-Compositing und ohne zweiten Sampler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .binding import BoundPlan

logger = logging.getLogger("FVM.K2.lora")

BACKEND = "fvm-k2-regional-lora-delta-gating-v1"

STANDARD_ROUTING = "standard"
CHARACTER_ROUTING = "character_identity"
ROUTING_MODES = (STANDARD_ROUTING, CHARACTER_ROUTING)

_LORA_PAIRS = (
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.weight", ".lora_up.weight"),
)
_LOKR_SUFFIXES = (
    ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b",
    ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2",
)
_ADAPTER_SUFFIXES = tuple(s for pair in _LORA_PAIRS for s in pair) + _LOKR_SUFFIXES
_AUX_SUFFIXES = (".alpha", ".dora_scale", ".diff", ".diff_b")
_KREA_INTERNAL = ("blocks.", "txtfusion.", "txtmlp.", "tmlp.", "tproj.")

# Main-Stream-Key/Value-Projektionen werden von *jeder* Bild-Query gelesen.
# Ein regionaler Delta darauf ist damit nicht mehr ortsgebunden.
_NON_LOCAL_MODULES = {"wk", "wv", "k_proj", "v_proj"}


# ── Key-Normalisierung ───────────────────────────────────────────────────


def normalize_key(key: str) -> str:
    """``diffusion_model.blocks.…`` → ``blocks.…`` falls der Worker das erwartet."""
    prefix = "diffusion_model."
    if key.startswith(prefix) and key[len(prefix) :].startswith(_KREA_INTERNAL):
        return key[len(prefix) :]
    return key


def _adapter_base(key: str) -> str | None:
    for suffix in (*_ADAPTER_SUFFIXES, *_AUX_SUFFIXES):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def align_state_dict(state: dict, supported_prefixes) -> dict:
    """Wählt je Key den Namensraum, den das geladene Modell tatsächlich kennt."""
    supported = set(supported_prefixes)
    aligned: dict = {}
    for key, value in state.items():
        original_base = _adapter_base(key)
        normalized = normalize_key(key)
        normalized_base = _adapter_base(normalized)
        if original_base in supported:
            target = key
        elif normalized_base in supported:
            target = normalized
        else:
            target = key
        if target in aligned:
            raise ValueError(f"Kollision bei der LoRA-Key-Angleichung: {target}")
        aligned[target] = value
    return aligned


def adapter_prefixes(keys) -> tuple[str, ...]:
    prefixes = set()
    for key in keys:
        for suffix in _ADAPTER_SUFFIXES:
            if key.endswith(suffix):
                prefixes.add(key[: -len(suffix)])
                break
    return tuple(sorted(prefixes))


def inspect_lora(path: str | Path) -> dict[str, Any]:
    """Header-Analyse ohne Tensor-Load: Rang, Typ, Namensraum, Trainingsquelle."""
    import json
    import struct
    from collections import Counter

    path = Path(path)
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    metadata = header.get("__metadata__") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    tensors = {
        k: v for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)
    }
    prefixes = adapter_prefixes(tensors)
    ranks: Counter = Counter()
    types: Counter = Counter()
    complete = 0
    for prefix in prefixes:
        is_lokr = any(f"{prefix}{s}" in tensors for s in _LOKR_SUFFIXES)
        types["lokr" if is_lokr else "lora"] += 1
        if is_lokr:
            has_w1 = any(
                f"{prefix}.lokr_w1{tail}" in tensors for tail in ("", "_a")
            )
            has_w2 = any(
                f"{prefix}.lokr_w2{tail}" in tensors for tail in ("", "_a")
            )
            complete += int(has_w1 and has_w2)
        else:
            for down, up in _LORA_PAIRS:
                if f"{prefix}{down}" in tensors and f"{prefix}{up}" in tensors:
                    complete += 1
                    shape = tensors[f"{prefix}{down}"].get("shape") or []
                    if shape:
                        ranks[int(shape[0])] += 1
                    break
    namespaces = Counter(
        ".".join(normalize_key(prefix).split(".")[:1]) for prefix in prefixes
    )
    return {
        "path": str(path),
        "name": metadata.get("modelspec.title")
        or metadata.get("ss_output_name")
        or path.stem,
        "tensor_count": len(tensors),
        "adapter_count": len(prefixes),
        "complete_adapter_pairs": complete,
        "adapter_types": dict(sorted(types.items())),
        "ranks": dict(sorted(ranks.items())),
        "namespaces": dict(sorted(namespaces.items())),
        "base_model": metadata.get("ss_base_model_version")
        or metadata.get("ss_sd_model_name"),
        "software": metadata.get("software"),
    }


# ── Routen ───────────────────────────────────────────────────────────────


@dataclass
class LoraSpec:
    """Benutzerangabe vor der Kompilierung."""

    lora_id: str
    lora_name: str
    strength: float = 1.0
    global_scope: bool = True
    region_ids: tuple[str, ...] = ()
    routing_mode: str = STANDARD_ROUTING
    trigger_phrase: str = ""
    display_name: str = ""

    def __post_init__(self) -> None:
        if self.routing_mode not in ROUTING_MODES:
            raise ValueError(f"Unbekannter Routing-Modus: {self.routing_mode!r}")
        if not -4.0 <= float(self.strength) <= 4.0:
            raise ValueError("LoRA-Stärke muss zwischen -4 und 4 liegen")
        if self.routing_mode == CHARACTER_ROUTING:
            if self.global_scope:
                raise ValueError(
                    "character_identity braucht regionalen Scope (Global aus)"
                )
            if not self.trigger_phrase.strip():
                raise ValueError("character_identity braucht eine Trigger-Phrase")
        if not self.display_name:
            self.display_name = Path(self.lora_name).stem


@dataclass
class LoraRoute:
    lora_id: str
    display_name: str
    lora_name: str
    strength: float
    global_scope: bool
    region_ids: tuple[str, ...]
    region_names: tuple[str, ...]
    text_mask: np.ndarray
    image_mask: np.ndarray
    routing_mode: str = STANDARD_ROUTING
    trigger_phrase: str = ""
    backend: str = BACKEND

    @property
    def text_count(self) -> int:
        return int(self.text_mask.shape[0])

    @property
    def image_count(self) -> int:
        return int(self.image_mask.shape[0])

    def summary(self) -> dict:
        return {
            "backend": self.backend,
            "lora": self.lora_name,
            "strength": self.strength,
            "global": self.global_scope,
            "region_ids": list(self.region_ids),
            "region_names": list(self.region_names),
            "routing_mode": self.routing_mode,
            "trigger_phrase": self.trigger_phrase,
            "text_tokens_enabled": int((self.text_mask > 0).sum()),
            "image_tokens_enabled": int((self.image_mask > 0).sum()),
            "image_coverage": float(self.image_mask.mean()) if self.image_count else 0.0,
        }


def identity_triggers_from_specs(specs) -> dict[str, tuple[str, ...]]:
    """Sammelt Trigger-Phrasen je Region für die Prompt-Kompilierung."""
    collected: dict[str, list[str]] = {}
    for spec in specs:
        if spec.routing_mode != CHARACTER_ROUTING:
            continue
        trigger = spec.trigger_phrase.strip()
        for region_id in spec.region_ids:
            triggers = collected.setdefault(region_id, [])
            if trigger not in triggers:
                triggers.append(trigger)
    return {rid: tuple(t) for rid, t in collected.items()}


def compile_routes(specs, bound: BoundPlan) -> tuple[LoraRoute, ...]:
    """Erzeugt exakte Text-/Bildtoken-Gates für jede aktive LoRA-Zuweisung."""
    text_count = bound.text_token_count
    image_count = bound.image_token_count
    all_text = np.ones(text_count, dtype=np.float32)
    all_image = np.ones(image_count, dtype=np.float32)
    spans = {span.region_id: span for span in bound.spans}
    names = {span.region_id: span.name for span in bound.spans}

    routes: list[LoraRoute] = []
    for spec in specs:
        if float(spec.strength) == 0.0:
            continue
        if spec.global_scope:
            routes.append(
                LoraRoute(
                    lora_id=spec.lora_id,
                    display_name=spec.display_name,
                    lora_name=spec.lora_name,
                    strength=float(spec.strength),
                    global_scope=True,
                    region_ids=(),
                    region_names=(),
                    text_mask=all_text,
                    image_mask=all_image,
                    routing_mode=STANDARD_ROUTING,
                    trigger_phrase=spec.trigger_phrase,
                )
            )
            continue

        if not spec.region_ids:
            raise ValueError(
                f"Regionaler LoRA {spec.display_name!r} hat keine zugewiesene Region"
            )
        missing = [rid for rid in spec.region_ids if rid not in spans]
        if missing:
            raise ValueError(
                f"LoRA {spec.display_name!r} zeigt auf Regionen ohne aktiven Prompt: "
                + ", ".join(missing)
            )

        text_mask = np.zeros(text_count, dtype=np.float32)
        image_mask = np.zeros(image_count, dtype=np.float32)
        for region_id in spec.region_ids:
            span = spans[region_id]
            text_mask[span.start : span.end] = 1.0
            # Bildseitig immer die *harte* Box — ein weiches Feld würde den Delta
            # außerhalb der Region weiterleben lassen.
            image_mask = np.maximum(image_mask, (span.mask > 0.0).astype(np.float32))

        routes.append(
            LoraRoute(
                lora_id=spec.lora_id,
                display_name=spec.display_name,
                lora_name=spec.lora_name,
                strength=float(spec.strength),
                global_scope=False,
                region_ids=tuple(spec.region_ids),
                region_names=tuple(names[r] for r in spec.region_ids),
                text_mask=text_mask,
                image_mask=image_mask,
                routing_mode=spec.routing_mode,
                trigger_phrase=spec.trigger_phrase,
            )
        )
    return tuple(routes)


def route_allows_target(route: LoraRoute, key: str, strict: bool) -> bool:
    """Hält regionale Deltas auf tokenlokalen Pfaden.

    Textfusions-Deltas sind durch die Textpartition geschützt. Main-Stream-K/V
    werden dagegen von jeder Bild-Query gelesen — ein regionaler Delta darauf
    verlässt seine Box, also wird das Ziel im strikten Modus ausgelassen.
    """
    if not strict or route.global_scope or route.routing_mode == CHARACTER_ROUTING:
        return True
    lowered = key.casefold()
    if ".txtfusion." in lowered or ".txtmlp." in lowered:
        return True
    parts = lowered.split(".")
    module = parts[-2] if parts and parts[-1] == "weight" else parts[-1]
    return module not in _NON_LOCAL_MODULES


def route_kind(key: str) -> str:
    """Welches Tensor-Layout hat der Eingang dieses Ziels?"""
    lowered = str(key)
    if ".txtfusion.layerwise_blocks." in lowered:
        return "text_layerwise"
    if ".txtfusion.projector" in lowered:
        return "text_projector"
    if ".txtfusion." in lowered or ".txtmlp." in lowered:
        return "text_refiner"
    if ".blocks." in lowered:
        return "combined"
    return "unmasked"


# ── Delta-Statistik (für LoRA-Delta-Adaption) ────────────────────────────


@dataclass
class LoraDeltaStatistics:
    routes: tuple[LoraRoute, ...]
    values: dict[str, dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.values = {
            route.lora_id: {
                "calls": 0,
                "energy": None,
                "count": 0,
                "step_energy": None,
                "step_count": 0,
                "reference": None,
            }
            for route in self.routes
        }

    @staticmethod
    def _rms(energy, count: int) -> float:
        if energy is None or count == 0:
            return 0.0
        return float((energy / count) ** 0.5)

    def observe(self, route: LoraRoute, applied) -> None:
        state = self.values.get(route.lora_id)
        if state is None:
            return
        energy = float(applied.detach().float().pow(2).sum().item())
        count = int(applied.numel())
        state["calls"] += 1
        state["energy"] = energy if state["energy"] is None else state["energy"] + energy
        state["count"] += count
        state["step_energy"] = (
            energy if state["step_energy"] is None else state["step_energy"] + energy
        )
        state["step_count"] += count

    def region_scales(self, gain: float) -> dict[str, float]:
        """Bounded Korrekturfaktor je Region aus der beobachteten Delta-Energie."""
        per_region: dict[str, list[float]] = {}
        for route in self.routes:
            if route.global_scope or not route.region_ids:
                continue
            state = self.values[route.lora_id]
            observed = self._rms(state["step_energy"], state["step_count"])
            if observed <= 0.0:
                continue
            reference = state["reference"] or observed
            ratio = observed / max(reference, 1e-12)
            scale = min(1.5, max(0.5, 1.0 + gain * (ratio - 1.0)))
            state["reference"] = 0.85 * reference + 0.15 * observed
            for region_id in route.region_ids:
                per_region.setdefault(region_id, []).append(scale)
        return {r: sum(v) / len(v) for r, v in per_region.items()}

    def reset_step(self) -> None:
        for state in self.values.values():
            state["step_energy"] = None
            state["step_count"] = 0

    def release(self) -> None:
        for state in self.values.values():
            state.update(
                {
                    "energy": None,
                    "count": 0,
                    "step_energy": None,
                    "step_count": 0,
                    "reference": None,
                    "calls": 0,
                }
            )

    def summary(self) -> dict:
        return {
            route.lora_id: {
                "display_name": route.display_name,
                "calls": self.values[route.lora_id]["calls"],
                "delta_rms": self._rms(
                    self.values[route.lora_id]["energy"],
                    self.values[route.lora_id]["count"],
                ),
            }
            for route in self.routes
        }


# ── Anwendung ────────────────────────────────────────────────────────────


def load_lora_patches(model, lora_path: str, *, report_name: str = "") -> tuple[dict, dict]:
    """Lädt eine LoRA und mappt sie auf die Krea-Zielschlüssel des Modells."""
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils

    state = comfy.utils.load_torch_file(lora_path, safe_load=True)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    aligned = align_state_dict(state, key_map)
    converted = comfy.lora_convert.convert_lora(aligned)
    patches = comfy.lora.load_lora(converted, key_map, log_missing=False)

    header = inspect_lora(lora_path)
    prefixes = adapter_prefixes(converted)
    unmatched = [p for p in prefixes if p not in key_map]
    report = {
        **header,
        "display_name": report_name or header["name"],
        "matched_model_targets": len(patches),
        "unmatched_adapter_targets": len(unmatched),
        "unmatched_examples": unmatched[:6],
        "compatible": bool(patches),
    }
    return patches, report


def build_routed_adapter_class():
    """Erzeugt die Composite-Adapter-Klasse (lazy, damit torch nur bei Bedarf lädt)."""
    import torch
    import comfy.weight_adapter

    base_type = comfy.weight_adapter.WeightAdapterBase

    class RoutedCompositeAdapter(base_type):
        """Summiert mehrere LoRA-Deltas auf einem Ziel, jeweils tokenmaskiert."""

        name = "fvm_k2_routed_composite"

        def __init__(self, entries, kind: str, statistics: LoraDeltaStatistics) -> None:
            self.entries = entries  # [(adapter, route)]
            self.kind = kind
            self.statistics = statistics
            self.weights = []
            self.loaded_keys = set()
            self._prepared: set[int] = set()
            self._mask_cache: dict = {}
            self.skipped_shapes: set = set()

        # Der Bypass-Hook setzt diese Felder auf *uns*; die Sub-Adapter brauchen
        # sie ebenfalls, sonst rechnen sie mit falschem Layer-Typ.
        def _prepare(self, adapter, route, x) -> None:
            adapter.multiplier = route.strength
            for attr in (
                "is_conv", "conv_dim", "kernel_size",
                "in_channels", "out_channels", "kw_dict",
            ):
                setattr(adapter, attr, getattr(self, attr, None))
            identity = id(adapter)
            if identity in self._prepared:
                return
            weights = getattr(adapter, "weights", None)
            if isinstance(weights, (tuple, list)):
                moved = []
                for weight in weights:
                    if isinstance(weight, torch.Tensor):
                        dtype = x.dtype if weight.is_floating_point() else weight.dtype
                        moved.append(weight.to(device=x.device, dtype=dtype))
                    else:
                        moved.append(weight)
                adapter.weights = type(weights)(moved)
            self._prepared.add(identity)

        def _mask(self, route: LoraRoute, x):
            """Tokenmaske passend zum Eingangslayout dieses Ziels; None = ungemaskt."""
            if route.global_scope:
                return None
            key = (route.lora_id, self.kind, tuple(x.shape))
            if key in self._mask_cache:
                return self._mask_cache[key]

            text = route.text_count
            mask = None
            try:
                if self.kind == "text_layerwise":
                    batch = int(x.shape[0])
                    if batch % text == 0:
                        values = np.tile(route.text_mask, batch // text)
                        mask = torch.as_tensor(
                            values, device=x.device, dtype=x.dtype
                        ).view(-1, 1, 1)
                elif self.kind == "text_projector":
                    if x.ndim == 4 and int(x.shape[1]) == text:
                        mask = torch.as_tensor(
                            route.text_mask, device=x.device, dtype=x.dtype
                        ).view(1, -1, 1, 1)
                elif self.kind == "text_refiner":
                    if int(x.shape[-2]) == text:
                        mask = torch.as_tensor(
                            route.text_mask, device=x.device, dtype=x.dtype
                        ).view(1, -1, 1)
                elif self.kind == "combined":
                    sequence = int(x.shape[-2])
                    if sequence == text + route.image_count:
                        values = np.concatenate([route.text_mask, route.image_mask])
                        mask = torch.as_tensor(
                            values, device=x.device, dtype=x.dtype
                        ).view(1, -1, 1)
            except Exception as error:  # pragma: no cover — defensiv
                logger.warning("K2: Maskenbau fehlgeschlagen (%s): %s", self.kind, error)
                mask = None

            self._mask_cache[key] = mask
            return mask

        def h(self, x, base_out):
            total = None
            for adapter, route in self.entries:
                self._prepare(adapter, route, x)
                mask = self._mask(route, x)
                if mask is None and not route.global_scope:
                    # Unbekanntes Layout — regionaler Delta wäre nicht ortsgebunden.
                    self.skipped_shapes.add((self.kind, tuple(x.shape)))
                    continue
                applied = adapter.h(x, base_out)
                if mask is not None:
                    applied = applied * mask
                if not route.global_scope:
                    self.statistics.observe(route, applied)
                total = applied if total is None else total + applied
            if total is None:
                return torch.zeros_like(base_out)
            return total

        def release_device_state(self) -> None:
            """Adapter-Gewichte zurück in den RAM — sonst wächst der VRAM je Edit."""
            for adapter, _route in self.entries:
                weights = getattr(adapter, "weights", None)
                if not isinstance(weights, (tuple, list)):
                    continue
                adapter.weights = type(weights)(
                    [
                        w.detach().to(device="cpu") if isinstance(w, torch.Tensor) else w
                        for w in weights
                    ]
                )
            self._prepared.clear()
            self._mask_cache.clear()

    return RoutedCompositeAdapter


def apply_routes(
    model,
    specs,
    routes: tuple[LoraRoute, ...],
    *,
    strict_isolation: bool = True,
    fuse_global: bool = True,
):
    """Installiert alle LoRA-Routen auf einem geklonten MODEL.

    Globale Routen werden als normale (gefusionierte) Patches angewandt, weil sie
    keine Maske brauchen. Regionale Routen laufen über Bypass-Adapter.
    """
    import comfy.weight_adapter

    route_map = {route.lora_id: route for route in routes}
    statistics = LoraDeltaStatistics(routes)
    reports: list[dict] = []
    target_entries: dict[str, list] = {}
    skipped: dict[str, list[str]] = {}
    patched = model

    base_adapter_type = comfy.weight_adapter.WeightAdapterBase

    for spec in specs:
        route = route_map.get(spec.lora_id)
        if route is None:
            reports.append(
                {"display_name": spec.display_name, "status": "disabled_zero_strength"}
            )
            continue

        import folder_paths

        lora_path = folder_paths.get_full_path_or_raise("loras", spec.lora_name)
        patches, report = load_lora_patches(
            patched, lora_path, report_name=spec.display_name
        )
        report["strength"] = route.strength
        report["route"] = route.summary()
        if not patches:
            raise ValueError(
                f"LoRA {spec.display_name!r} passt auf 0 Krea-Ziele — falsche "
                "Architektur oder unbekanntes Key-Schema "
                f"(Namensräume: {report.get('namespaces')})"
            )

        if route.global_scope and fuse_global:
            patched = patched.clone()
            applied = patched.add_patches(patches, strength_patch=route.strength)
            report["status"] = "applied_global_fused"
            report["applied_model_targets"] = len(applied)
            reports.append(report)
            continue

        local: list[str] = []
        for key, adapter in patches.items():
            if not isinstance(adapter, base_adapter_type):
                # z.B. reine "diff"-Patches — regional nicht maskierbar.
                skipped.setdefault(route.lora_id, []).append(str(key))
                continue
            if route_allows_target(route, str(key), strict_isolation):
                target_entries.setdefault(key, []).append((adapter, route))
                local.append(str(key))
            else:
                skipped.setdefault(route.lora_id, []).append(str(key))

        report["status"] = (
            "applied_global_unfused" if route.global_scope else "applied_regional"
        )
        report["application_mode"] = "unfused_token_delta_gate"
        report["applied_model_targets"] = len(local)
        report["locality_skipped_targets"] = len(skipped.get(route.lora_id, []))
        report["locality_skipped_examples"] = skipped.get(route.lora_id, [])[:6]
        if not local:
            raise ValueError(
                f"Regionaler LoRA {spec.display_name!r} hat kein ortsgebundenes Ziel — "
                "nichts angewandt"
            )
        reports.append(report)

    # Die Bypass-Hooks binden an konkrete nn.Module-Objekte. ModelPatcher.clone()
    # kann dem Klon ein eigenes Modell geben, wodurch Hooks auf toten Modulen
    # landen. Deshalb werden die Adapter hier nur *vorbereitet* und erst vom
    # Aufrufer auf dem finalen Patcher installiert.
    pending: dict[str, Any] = {}
    if target_entries:
        adapter_class = build_routed_adapter_class()
        for key, entries in target_entries.items():
            pending[key] = adapter_class(entries, route_kind(str(key)), statistics)

    return patched, reports, statistics, pending


def install_routed_adapters(model, pending: dict):
    """Installiert die vorbereiteten Composite-Adapter auf dem FINALEN Patcher.

    Muss der letzte Patch-Schritt sein — jeder weitere ``clone()`` danach kann
    die Hooks von den tatsächlich ausgeführten Modulen trennen.
    """
    import comfy.weight_adapter

    if not pending:
        return model, ()

    manager = comfy.weight_adapter.BypassInjectionManager()
    for key, composite in pending.items():
        manager.add_adapter(key, composite, strength=1.0)

    injections = manager.create_injections(model.model)
    model.set_injections("fvm_k2_regional_loras", injections)
    hooks = manager.get_hook_count()
    if hooks != len(pending):
        logger.warning(
            "K2: %s/%s LoRA-Hooks installiert — einige Ziele fehlen im Modell",
            hooks,
            len(pending),
        )
    return model, tuple(c.release_device_state for c in pending.values())


__all__ = [
    "BACKEND",
    "CHARACTER_ROUTING",
    "ROUTING_MODES",
    "STANDARD_ROUTING",
    "LoraDeltaStatistics",
    "LoraRoute",
    "LoraSpec",
    "align_state_dict",
    "apply_routes",
    "install_routed_adapters",
    "compile_routes",
    "identity_triggers_from_specs",
    "inspect_lora",
    "load_lora_patches",
    "normalize_key",
    "route_allows_target",
    "route_kind",
]
