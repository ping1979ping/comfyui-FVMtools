"""FVM K2 Lab — Kernknoten: kompiliert Regionen, LoRAs und Projector zu einem
lauffähigen Krea-2-Graphen.

Der Ausgang ist bewusst gewöhnliches ComfyUI-Material: ein gepatchtes MODEL,
normale CONDITIONING, ein normales LATENT und eine normale MASK. Damit bleiben
ControlNet, Guider, FreeU, Upscaler oder Save-Nodes uneingeschränkt nutzbar.
"""

import json

import torch

from ...core.k2.attention import K2SpatialAttention
from ...core.k2.binding import bind_plan
from ...core.k2.lora import (
    apply_routes,
    compile_routes,
    identity_triggers_from_specs,
    install_routed_adapters,
)
from ...core.k2.projector import PROJECTOR_TARGET, token_delta_mask
from ...core.k2.prompt import GLOBAL_SCOPE, compile_plan
from ...core.k2.runtime import K2_ATTACHMENT, K2Runtime, union_mask_tensor

CATEGORY = "FVM Tools/K2"

DEFAULT_TUNING = {
    "spatial_enabled": True,
    "strict_isolation": True,
    "inside_strength": 1.0,
    "outside_penalty": 1.0,
    "falloff_pixels": 128.0,
    "late_step_scale": 0.35,
    "subject_competition": True,
    "subject_fill": True,
    "spatial_instructions": True,
    "lora_delta_adaptation": False,
    "lora_adaptation_gain": 0.35,
    "fuse_global_loras": True,
}


class FVM_K2_Tuning:
    """Feineinstellung des räumlichen Routers."""

    DESCRIPTION = (
        "Tuning for the K2 spatial attention router. Optional — K2 Compose uses these "
        "same defaults when nothing is connected.\n\n"
        "inside_strength binds regional text harder to its box. outside_penalty raises "
        "center-to-edge contrast and hard-blocks subject text outside its box. "
        "late_step_scale relaxes the binding in the last denoising steps so edges do "
        "not look pasted on."
    )
    CATEGORY = CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("K2_TUNING",)
    RETURN_NAMES = ("tuning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spatial_enabled": ("BOOLEAN", {"default": True,
                                    "tooltip": "Off disables the router completely. Not "
                                    "allowed while a regional LoRA is active — its text "
                                    "delta would become shared scene conditioning."}),
                "strict_isolation": ("BOOLEAN", {"default": True,
                                     "tooltip": "Hard partition: subject-owned text tokens "
                                     "are private to their subject and unreachable for "
                                     "image tokens outside its box. Off = soft bias only "
                                     "(attributes can bleed between subjects)."}),
                "inside_strength": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0,
                                    "step": 0.05,
                                    "tooltip": "Positive attention bias inside each region. "
                                    "Larger values bind regional text harder to its box; "
                                    "very large values can flatten the image."}),
                "outside_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0,
                                              "step": 0.05,
                                              "tooltip": "Background regions use one quarter "
                                              "of this value."}),
                "falloff_pixels": ("FLOAT", {"default": 128.0, "min": 0.0, "max": 2048.0,
                                             "step": 8.0,
                                             "tooltip": "Soft edge width beyond a background "
                                             "box. Subject text stays hard-confined."}),
                "late_step_scale": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0,
                                              "step": 0.05,
                                              "tooltip": "Fraction of spatial strength kept "
                                              "at the last step. Relaxation starts after "
                                              "55% progress. Needs K2 Sampler."}),
                "subject_competition": ("BOOLEAN", {"default": True,
                                        "tooltip": "Overlapping subjects share tokens by "
                                        "squared field strength instead of both claiming "
                                        "them fully (prevents merged people)."}),
                "subject_fill": ("BOOLEAN", {"default": True,
                                 "tooltip": "Keeps the field strong towards box edges so "
                                 "subjects fill their area."}),
                "spatial_instructions": ("BOOLEAN", {"default": True,
                                         "tooltip": "Adds the generated location sentences "
                                         "to the prompt. Off = attention-only routing "
                                         "(shorter prompt, weaker placement)."}),
                "lora_delta_adaptation": ("BOOLEAN", {"default": False,
                                          "tooltip": "Rebalances each region's spatial scale "
                                          "from the observed regional LoRA delta energy. "
                                          "Needs K2 Sampler."}),
                "lora_adaptation_gain": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0,
                                         "step": 0.05,
                                         "tooltip": "Maximum correction gain for LoRA-delta "
                                         "adaptation. 0 measures without correcting."}),
                "fuse_global_loras": ("BOOLEAN", {"default": True,
                                      "tooltip": "Global LoRAs as normal fused patches "
                                      "(faster). Off routes them through the same unfused "
                                      "path as regional LoRAs."}),
            },
        }

    def build(self, **kwargs):
        tuning = dict(DEFAULT_TUNING)
        tuning.update(kwargs)
        return (tuning,)


class FVM_K2_Compose:
    """Kompiliert das komplette K2-Projekt zu Standard-ComfyUI-Objekten."""

    DESCRIPTION = (
        "Compiles regions, regional LoRAs, emphases and the projector into a patched "
        "Krea 2 MODEL plus ordinary CONDITIONING, LATENT and MASK.\n\n"
        "What happens here:\n"
        "1. every enabled region prompt is compiled into one unified Krea prompt with "
        "explicit location clauses and inter-subject relationships;\n"
        "2. the compiled prompt is tokenized so each clause knows its token span;\n"
        "3. the spatial attention router is installed on the model branch;\n"
        "4. regional LoRAs are installed as unfused, token-gated forward adapters;\n"
        "5. the projector delta is applied (token-selective if identity protection is on).\n\n"
        "Outputs stay standard types, so the patched model can feed ControlNet, guiders "
        "or any sampler. Use K2 Sampler when late-step relaxation or LoRA-delta "
        "adaptation is enabled."
    )
    CATEGORY = CATEGORY
    FUNCTION = "compose"
    RETURN_TYPES = (
        "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "MASK", "K2_PLAN",
        "STRING", "STRING",
    )
    RETURN_NAMES = (
        "model", "positive", "negative", "latent", "mask", "plan",
        "compiled_prompt", "report",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Krea 2 MODEL. It is cloned and patched; "
                                    "the incoming branch stays untouched."}),
                "clip": ("CLIP", {"tooltip": "CLIPLoader with type 'krea2' (Qwen3-VL)."}),
                "global_prompt": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "Scene-wide description. Regional clauses are "
                                  "appended after it."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Global negative. Krea 2 Turbo runs CFG-free, "
                                    "so this only matters with CFG > 1."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                          "tooltip": "Output width AND the coordinate system for every "
                          "region box. Changing it moves all boxes."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                           "tooltip": "Output height and vertical box coordinate system."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                               "tooltip": "Region and LoRA routing is broadcast across the "
                               "batch."}),
            },
            "optional": {
                "regions": ("K2_REGION", {"tooltip": "Region chain. Without it this behaves "
                            "like a plain prompt encode."}),
                "loras": ("K2_LORA", {"tooltip": "LoRA assignments from K2 Regional LoRA."}),
                "emphasis": ("K2_EMPHASIS", {"tooltip": "Phrase emphases."}),
                "projector": ("K2_PROJECTOR", {"tooltip": "Projector delta."}),
                "tuning": ("K2_TUNING", {"tooltip": "Spatial tuning. Defaults are used when "
                           "nothing is connected."}),
            },
        }

    # ── Hilfsfunktionen ──────────────────────────────────────────────────

    @staticmethod
    def _region_lookup(regions):
        by_name = {r.name.strip().casefold(): r.region_id for r in regions}
        by_id = {r.region_id: r.region_id for r in regions}
        return by_id, by_name

    @classmethod
    def _resolve_emphasis_scopes(cls, emphases, regions):
        """Emphasis-Scopes dürfen den Regionsnamen tragen — intern zählt die ID."""
        by_id, by_name = cls._region_lookup(regions)
        for emphasis in emphases:
            if emphasis.scope_id == GLOBAL_SCOPE:
                continue
            key = emphasis.scope_id.strip()
            target = by_id.get(key) or by_name.get(key.casefold())
            if target is None:
                available = ", ".join(sorted(r.name for r in regions)) or "<none>"
                raise ValueError(
                    f"Emphasis {emphasis.phrase!r} references unknown scope {key!r}. "
                    f"Use '{GLOBAL_SCOPE}' or one of: {available}"
                )
            emphasis.scope_id = target
        return emphases

    @staticmethod
    def _resolve_region_refs(specs, regions):
        """Übersetzt Regionsnamen in stabile IDs und meldet Tippfehler früh."""
        by_name = {r.name.strip().casefold(): r.region_id for r in regions}
        by_id = {r.region_id: r.region_id for r in regions}
        resolved = []
        for spec in specs:
            if spec.global_scope:
                resolved.append(spec)
                continue
            ids = []
            for reference in spec.region_ids:
                key = reference.strip()
                target = by_id.get(key) or by_name.get(key.casefold())
                if target is None:
                    available = ", ".join(sorted(r.name for r in regions)) or "<none>"
                    raise ValueError(
                        f"LoRA {spec.display_name!r} references unknown region "
                        f"{reference!r}. Available regions: {available}"
                    )
                if target not in ids:
                    ids.append(target)
            spec.region_ids = tuple(ids)
            resolved.append(spec)
        return resolved

    @staticmethod
    def _apply_projector(model, projector, bound):
        """Projector-Delta als Patch oder tokenselektiver Bypass-Adapter."""
        report = {"status": "not_configured"}
        if not projector:
            return model, report

        effective = tuple(float(v) for v in projector.get("effective", ()))
        report = {
            "enabled": bool(projector.get("enabled", False)),
            "preset": projector.get("preset"),
            "source": projector.get("source"),
            "values": list(projector.get("values", ())),
            "effective": list(effective),
            "identity_protection": float(projector.get("identity_protection", 1.0)),
            "target": PROJECTOR_TARGET,
        }
        if not report["enabled"] or not any(effective):
            report["status"] = "disabled" if not report["enabled"] else "zero_effect"
            return model, report

        state = model.model.state_dict()
        if PROJECTOR_TARGET not in state:
            raise RuntimeError(
                f"Projector target missing on this model: {PROJECTOR_TARGET}. "
                "Is this really a Krea 2 checkpoint?"
            )
        shape = tuple(state[PROJECTOR_TARGET].shape)
        if shape != (1, len(effective)):
            raise RuntimeError(f"Unexpected projector shape {shape}, expected (1, 12)")

        delta = torch.tensor((effective,), dtype=torch.float32)
        protection = float(projector.get("identity_protection", 1.0))
        protected = tuple(
            (span.start, span.end)
            for span in (*bound.identity_spans, *bound.trigger_spans)
        )

        if protected and protection > 0.0:
            import comfy.weight_adapter
            import torch.nn.functional as functional

            mask_values = token_delta_mask(
                bound.text_token_count, protected, protection
            )
            base_type = comfy.weight_adapter.WeightAdapterBase

            class TokenSelectiveProjector(base_type):
                name = "fvm_k2_projector"

                def __init__(self):
                    self.weights = (delta,)
                    self.loaded_keys = set()
                    self._cache = {}

                def h(self, x, base_out):
                    del base_out
                    if x.ndim != 4 or int(x.shape[1]) != len(mask_values):
                        # Andere Sequenzlänge (z.B. Negativ-Prompt) → kein Delta,
                        # sonst würde die Identitätsmaske daneben greifen.
                        return functional.linear(
                            x, self.weights[0].to(device=x.device, dtype=x.dtype)
                        )
                    key = (x.device, x.dtype)
                    cached = self._cache.get(key)
                    if cached is None:
                        weight = self.weights[0].to(device=x.device, dtype=x.dtype)
                        mask = torch.as_tensor(
                            mask_values, device=x.device, dtype=x.dtype
                        ).view(1, -1, 1, 1)
                        cached = (weight, mask)
                        self._cache[key] = cached
                    weight, mask = cached
                    return functional.linear(x, weight) * mask

            manager = comfy.weight_adapter.BypassInjectionManager()
            manager.add_adapter(PROJECTOR_TARGET, TokenSelectiveProjector(), strength=1.0)
            patched = model.clone()
            patched.set_injections(
                "fvm_k2_projector", manager.create_injections(patched.model)
            )
            report["status"] = "applied_token_selective"
            report["protected_token_spans"] = [list(s) for s in protected]
            return patched, report

        patched = model.clone()
        applied = patched.add_patches({PROJECTOR_TARGET: ("diff", (delta,))})
        if PROJECTOR_TARGET not in applied:
            raise RuntimeError("Could not apply the Krea projector patch")
        report["status"] = "applied_global_diff"
        return patched, report

    # ── Hauptlauf ────────────────────────────────────────────────────────

    def compose(self, model, clip, global_prompt, negative_prompt, width, height,
                batch_size, regions=None, loras=None, emphasis=None,
                projector=None, tuning=None):
        import comfy.sample

        settings = dict(DEFAULT_TUNING)
        settings.update(tuning or {})

        region_defs = list(regions or [])
        lora_specs = list(loras or [])
        lora_specs = self._resolve_region_refs(lora_specs, region_defs)
        emphasis_reqs = self._resolve_emphasis_scopes(list(emphasis or []), region_defs)

        active_regional_lora = any(
            not spec.global_scope and float(spec.strength) != 0.0
            for spec in lora_specs
        )
        if active_regional_lora and not settings["spatial_enabled"]:
            raise ValueError(
                "A regional LoRA is active but spatial attention is disabled. Without "
                "the router its text delta becomes shared scene conditioning and leaks "
                "into every region — enable spatial_enabled or make the LoRA global."
            )

        # 1) Prompt kompilieren
        plan = compile_plan(
            width,
            height,
            global_prompt,
            region_defs,
            strength=float(settings["inside_strength"]),
            outside_penalty=float(settings["outside_penalty"]),
            falloff_pixels=float(settings["falloff_pixels"]),
            subject_competition=bool(settings["subject_competition"]),
            subject_fill=bool(settings["subject_fill"]),
            late_step_scale=float(settings["late_step_scale"]),
            emphases=emphasis_reqs,
            identity_triggers=identity_triggers_from_specs(lora_specs),
            spatial_instructions=bool(settings["spatial_instructions"]),
        )
        prompt_text = plan.prompt if plan.prompt.strip() else global_prompt

        # 2) Konditionierung
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt_text))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        if not positive:
            raise RuntimeError("The Krea text encoder returned no conditioning")
        lengths = {int(entry[0].shape[1]) for entry in positive}
        if len(lengths) != 1:
            raise RuntimeError(
                "Krea conditioning must have exactly one text sequence length; got "
                f"{sorted(lengths)}"
            )
        text_token_count = lengths.pop()

        # 3) Tokenbindung
        bound = bind_plan(
            plan, clip.tokenize, conditioning_text_token_count=text_token_count
        )

        # 4) Projector
        patched, projector_report = self._apply_projector(model, projector, bound)

        # 5) LoRA-Routen (Adapter werden vorbereitet, aber noch nicht installiert)
        routes = compile_routes(lora_specs, bound)
        patched, lora_reports, statistics, pending_adapters = apply_routes(
            patched,
            lora_specs,
            routes,
            strict_isolation=bool(settings["strict_isolation"]),
            fuse_global=bool(settings["fuse_global_loras"]),
        )

        # 6) Attention-Router
        attention = None
        patched = patched.clone()
        if settings["spatial_enabled"] and (bound.spans or bound.emphases):
            attention = K2SpatialAttention(
                bound,
                strict_isolation=bool(settings["strict_isolation"]),
                lora_delta_adaptation=bool(settings["lora_delta_adaptation"]),
                lora_delta_adaptation_gain=float(settings["lora_adaptation_gain"]),
            )
            transformer_options = patched.model_options.setdefault(
                "transformer_options", {}
            )
            if "optimized_attention_override" in transformer_options:
                raise RuntimeError(
                    "Another node already owns 'optimized_attention_override' on this "
                    "MODEL branch. Branch the model before applying either override."
                )
            transformer_options["optimized_attention_override"] = attention

        # 6b) Erst jetzt die Bypass-Adapter installieren — nach dem letzten clone(),
        # sonst hängen die Forward-Hooks an Modulen, die nie ausgeführt werden.
        patched, release_callbacks = install_routed_adapters(patched, pending_adapters)

        runtime = K2Runtime(
            bound=bound,
            attention=attention,
            statistics=statistics,
            routes=routes,
            lora_reports=lora_reports,
            projector_report=projector_report,
            settings=settings,
            release_callbacks=release_callbacks,
        )
        patched.set_attachments(K2_ATTACHMENT, runtime)

        # 7) Latent + Maske
        latent = torch.zeros(
            [int(batch_size), 4, plan.geometry.aligned_height // 8,
             plan.geometry.aligned_width // 8],
            device=torch.device("cpu"),
        )
        latent = comfy.sample.fix_empty_latent_channels(
            patched, latent, downscale_ratio_spacial=8
        )
        mask = (
            union_mask_tensor(bound, plan.geometry.aligned_width,
                              plan.geometry.aligned_height)
            if bound.spans
            else torch.ones(
                1, plan.geometry.aligned_height, plan.geometry.aligned_width
            )
        )

        report = json.dumps(runtime.report(), indent=2, default=str)
        return (
            patched, positive, negative, {"samples": latent}, mask, runtime,
            prompt_text, report,
        )


NODE_CLASS_MAPPINGS = {
    "FVM_K2_Tuning": FVM_K2_Tuning,
    "FVM_K2_Compose": FVM_K2_Compose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_K2_Tuning": "K2 Spatial Tuning",
    "FVM_K2_Compose": "K2 Compose",
}
