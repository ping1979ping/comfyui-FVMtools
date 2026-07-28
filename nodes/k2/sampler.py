"""FVM K2 Lab — Sampler mit Rückmeldung an den regionalen Router."""

import json

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import torch

from ...core.k2.runtime import K2_ATTACHMENT, K2Runtime

CATEGORY = "FVM Tools/K2"


def _runtime_from(model, plan):
    if isinstance(plan, K2Runtime):
        return plan
    getter = getattr(model, "get_attachment", None)
    if callable(getter):
        attached = getter(K2_ATTACHMENT)
        if isinstance(attached, K2Runtime):
            return attached
    return None


class FVM_K2_Sampler:
    """KSampler, der dem K2-Plan den Denoising-Fortschritt meldet."""

    DESCRIPTION = (
        "Standard ComfyUI sampling plus per-step feedback to the K2 region plan.\n\n"
        "That feedback drives two things: late-step relaxation (spatial binding is "
        "gradually released after 55% progress so region edges do not look pasted on) "
        "and LoRA-delta adaptation. Without a connected plan this behaves like a "
        "normal KSampler.\n\n"
        "Krea 2 Turbo defaults: 8 steps, CFG 1.0, euler / simple."
    )
    CATEGORY = CATEGORY
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Patched model from K2 Compose."}),
                "positive": ("CONDITIONING", {"tooltip": "Compiled positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Global negative conditioning."}),
                "latent": ("LATENT", {"tooltip": "Starting latent — K2 Compose supplies an "
                           "empty one, but any compatible latent works."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                         "control_after_generate": True,
                         "tooltip": "Noise seed. Same seed + settings reproduces the run."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 1000,
                                  "tooltip": "Krea 2 Turbo is distilled for ~8 steps."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                  "tooltip": "Turbo is CFG-free; raising this is usually "
                                  "not an improvement. The RAW model uses CFG > 1."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler",
                                 "tooltip": "euler is the Krea 2 Turbo default."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple",
                              "tooltip": "simple is the Krea 2 Turbo default."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                            "tooltip": "1.0 for text-to-image; lower values keep more of a "
                            "supplied latent (img2img / regional edit)."}),
            },
            "optional": {
                "plan": ("K2_PLAN", {"tooltip": "K2 Compose plan. Enables late-step "
                                     "relaxation, LoRA-delta adaptation and the report."}),
            },
        }

    def sample(self, model, positive, negative, latent, seed, steps, cfg,
               sampler_name, scheduler, denoise, plan=None):
        runtime = _runtime_from(model, plan)

        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(
            model,
            latent_image,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        noise = comfy.sample.prepare_noise(
            latent_image, seed, latent.get("batch_index")
        )
        noise_mask = latent.get("noise_mask")

        preview = latent_preview.prepare_callback(model, steps)

        def callback(step, x0, x, total_steps):
            if runtime is not None:
                runtime.update_step(step + 1, total_steps)
            if preview is not None:
                preview(step, x0, x, total_steps)

        if runtime is not None:
            runtime.update_step(0, max(steps, 1))

        try:
            samples = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image,
                denoise=denoise, noise_mask=noise_mask, callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
            )
        finally:
            if runtime is not None:
                # Auch bei Abbruch/OOM die geroutete LoRA-Kopie vom Gerät nehmen,
                # sonst wächst der VRAM mit jedem Regionsedit.
                runtime.release()

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples

        if runtime is None:
            return (out, json.dumps({"status": "no_plan_connected"}, indent=2))

        data = runtime.report()
        warnings = runtime.sanity_warnings()
        data["warnings"] = warnings
        for message in warnings:
            print(f"[FVM K2] WARNUNG: {message}")
        return (out, json.dumps(data, indent=2, default=str))


class FVM_K2_PlanReport:
    """Zeigt den Planbericht ohne zu samplen."""

    DESCRIPTION = (
        "Prints the compiled K2 plan: token spans per region, LoRA routing decisions, "
        "skipped non-local targets, projector state and the spatial attention summary.\n\n"
        "Use it to check that a regional LoRA really matched Krea targets before "
        "spending a generation on it."
    )
    CATEGORY = CATEGORY
    FUNCTION = "show"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("K2_PLAN", {"tooltip": "Plan output of K2 Compose."}),
            }
        }

    def show(self, plan):
        text = json.dumps(plan.report(), indent=2, default=str)
        print(f"[FVM K2] plan report:\n{text}")
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "FVM_K2_Sampler": FVM_K2_Sampler,
    "FVM_K2_PlanReport": FVM_K2_PlanReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_K2_Sampler": "K2 Regional Sampler",
    "FVM_K2_PlanReport": "K2 Plan Report",
}
