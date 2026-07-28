"""FVM K2 Lab — Gesichtsverfeinerung pro Region."""

import json
import os

import comfy.sample
import comfy.samplers
import folder_paths
import numpy as np
import torch

from ...core.k2.face import (
    DETECTOR_BACKENDS,
    DETECTOR_NANODET,
    DETECTOR_YOLO,
    FaceDetailSettings,
    assign_faces,
    composite_crop,
    detect_faces,
    discover_nanodet,
    expanded_square_crop,
)
from ...core.k2.runtime import K2_ATTACHMENT, K2Runtime, as_image_batch, as_image_latent

CATEGORY = "FVM Tools/K2"


def _yolo_models():
    try:
        from ..utils.yolo_detector import get_available_yolo_models

        models = get_available_yolo_models()
    except Exception:
        models = []
    return models or ["face_yolov8m.pt"]


class FVM_K2_FaceDetail:
    """Verfeinert erkannte Gesichter mit den LoRAs ihrer Region."""

    DESCRIPTION = (
        "Detects faces, assigns each one to the subject region it sits in, and "
        "re-samples a padded crop with ONLY that region's LoRAs before compositing it "
        "back with a feathered mask.\n\n"
        "This is what makes multi-character LoRA routing hold up: in a full-body "
        "composition each face covers few latent tokens, so identity gets lost even "
        "with perfect regional routing. Refining the crop at full resolution restores "
        "it.\n\n"
        "Detector: any Ultralytics face model (recommended) or the NanoDet "
        "face_det.onnx shipped by FantasyPortrait. Both are already present in typical "
        "installs — no extra dependency."
    )
    CATEGORY = CATEGORY
    FUNCTION = "refine"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "face_mask", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Decoded image to inspect and refine."}),
                "model": ("MODEL", {"tooltip": "Base Krea model. Connect the ORIGINAL "
                          "model, not the K2 Compose output — regional gating makes no "
                          "sense on an isolated crop."}),
                "clip": ("CLIP", {"tooltip": "Krea/Qwen CLIP for the crop prompt."}),
                "vae": ("VAE", {"tooltip": "VAE used to encode and decode each crop."}),
                "plan": ("K2_PLAN", {"tooltip": "Plan from K2 Compose — supplies regions, "
                         "priorities, prompts and LoRA assignments."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                         "control_after_generate": True,
                         "tooltip": "Base seed; each face gets a deterministic offset so "
                         "different faces do not share identical noise."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 200,
                          "tooltip": "Denoising steps per face crop."}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                            "tooltip": "Low values keep facial structure, higher values "
                            "let the LoRA reshape the face."}),
                "crop_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16,
                              "tooltip": "Working resolution per face crop."}),
                "padding": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                            "tooltip": "Expands the detected face box before cropping; "
                            "more padding includes hair and context."}),
                "feather": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01,
                            "tooltip": "Soft border of the crop mask as a fraction of the "
                            "crop size."}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                          "tooltip": "Opacity of the refined crop. 0 keeps the original."}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                               "tooltip": "Extra multiplier on the region LoRAs during the "
                               "crop pass."}),
                "detector": (list(DETECTOR_BACKENDS), {"default": DETECTOR_YOLO,
                             "tooltip": "yolo: any Ultralytics face model (recommended).\n"
                             "nanodet_onnx: the face_det.onnx used by K2Lab."}),
                "detector_model": (_yolo_models(), {"tooltip": "Ultralytics face model. "
                                   "Ignored when detector is nanodet_onnx."}),
                "threshold": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95, "step": 0.05,
                              "tooltip": "Minimum detection confidence. Raise it to reject "
                              "background faces."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                        "tooltip": "Guidance for the crop pass; 1.0 for Turbo."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler",
                                 "tooltip": "Sampler for every crop pass."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple",
                              "tooltip": "Sigma schedule for every crop pass."}),
            },
            "optional": {
                "detector_path": ("STRING", {"default": "",
                                  "tooltip": "Explicit path to face_det.onnx. Empty uses "
                                  "the FantasyPortrait auto-discovery path."}),
                "require_region_lora": ("BOOLEAN", {"default": False,
                                        "tooltip": "On: only refine regions that actually "
                                        "have a regional LoRA assigned."}),
            },
        }

    @staticmethod
    def _face_model(base_model, routes, scale):
        """Klont das Modell und fusioniert nur die LoRAs dieser Region."""
        from ...core.k2.lora import load_lora_patches

        model = base_model
        reports = []
        for route in routes:
            path = folder_paths.get_full_path_or_raise("loras", route.lora_name)
            patches, _report = load_lora_patches(model, path)
            model = model.clone()
            applied = model.add_patches(
                patches, strength_patch=float(route.strength) * float(scale)
            )
            reports.append(
                {
                    "lora": route.lora_name,
                    "strength": round(float(route.strength) * float(scale), 3),
                    "targets": len(applied),
                }
            )
        return model, reports

    def refine(self, image, model, clip, vae, plan, seed, steps, denoise, crop_size,
               padding, feather, blend, lora_scale, detector, detector_model,
               threshold, cfg, sampler_name, scheduler, detector_path="",
               require_region_lora=False):
        if not isinstance(plan, K2Runtime):
            raise ValueError("K2 Face Detail needs the plan output of K2 Compose")

        settings = FaceDetailSettings(
            enabled=True, steps=steps, denoise=denoise, crop_size=crop_size,
            padding=padding, feather=feather, blend=blend, lora_scale=lora_scale,
            threshold=threshold,
        )
        settings.validate()

        if detector == DETECTOR_NANODET:
            path = detector_path.strip()
            if not path:
                discovered = discover_nanodet(folder_paths.base_path)
                if discovered is None:
                    raise FileNotFoundError(
                        "No face_det.onnx found. Install ComfyUI-WanVideoWrapper "
                        "(FantasyPortrait) or set detector_path."
                    )
                path = str(discovered)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Detector not found: {path}")
            detector_target = path
        else:
            detector_target = detector_model

        outputs = []
        masks = []
        batch_report = []

        for batch_index in range(int(image.shape[0])):
            canvas = image[batch_index].detach().cpu().float().clamp(0, 1).numpy()
            height, width = canvas.shape[:2]
            face_mask = np.zeros((height, width), dtype=np.float32)

            detections = detect_faces(
                canvas, backend=detector, model=detector_target,
                threshold=settings.threshold,
            )
            targets = assign_faces(
                detections, plan.bound, plan.routes,
                require_lora=bool(require_region_lora),
            )

            target_reports = []
            for target_index, target in enumerate(targets):
                crop_box = expanded_square_crop(
                    target.detection.box, width, height, settings.padding
                )
                x0, y0, x1, y1 = crop_box
                crop = canvas[y0:y1, x0:x1, :]
                if crop.size == 0:
                    continue

                from PIL import Image

                pil = Image.fromarray(
                    (np.clip(crop, 0, 1) * 255.0).round().astype(np.uint8)
                ).resize(
                    (settings.crop_size, settings.crop_size), Image.Resampling.LANCZOS
                )
                pixels = torch.from_numpy(
                    np.asarray(pil, dtype=np.float32) / 255.0
                ).unsqueeze(0)

                with torch.no_grad():
                    latent = as_image_latent(vae.encode(pixels))
                latent = comfy.sample.fix_empty_latent_channels(
                    model, latent, downscale_ratio_spacial=8
                )

                positive = clip.encode_from_tokens_scheduled(
                    clip.tokenize(target.prompt)
                )
                negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

                crop_model, lora_reports = self._face_model(
                    model, target.lora_specs, settings.lora_scale
                )
                face_seed = int(seed) + batch_index * 10_000 + target_index
                noise = comfy.sample.prepare_noise(latent, face_seed)
                samples = comfy.sample.sample(
                    crop_model, noise, settings.steps, cfg, sampler_name, scheduler,
                    positive, negative, latent, denoise=settings.denoise,
                    disable_pbar=True, seed=face_seed,
                )
                with torch.no_grad():
                    decoded = as_image_batch(vae.decode(samples))
                refined = decoded[0].detach().cpu().float().clamp(0, 1).numpy()

                canvas = composite_crop(
                    canvas, refined, crop_box, settings.feather, settings.blend
                )
                face_mask[y0:y1, x0:x1] = 1.0
                target_reports.append(
                    {
                        "region": target.region_name,
                        "prompt": target.prompt[:120],
                        "crop_box": list(crop_box),
                        "detector_score": round(target.detection.score, 3),
                        "seed": face_seed,
                        "loras": lora_reports,
                    }
                )

            outputs.append(torch.from_numpy(canvas))
            masks.append(torch.from_numpy(face_mask))
            batch_report.append(
                {
                    "batch_index": batch_index,
                    "detections": len(detections),
                    "refined": len(target_reports),
                    "targets": target_reports,
                }
            )

        report = json.dumps(
            {
                "status": "complete",
                "detector": detector,
                "detector_target": str(detector_target),
                "batches": batch_report,
            },
            indent=2,
            default=str,
        )
        total = sum(entry["refined"] for entry in batch_report)
        print(f"[FVM K2] Face Detail: {total} Gesicht(er) verfeinert")
        return (torch.stack(outputs), torch.stack(masks), report)


NODE_CLASS_MAPPINGS = {"FVM_K2_FaceDetail": FVM_K2_FaceDetail}
NODE_DISPLAY_NAME_MAPPINGS = {"FVM_K2_FaceDetail": "K2 Regional Face Detail"}
