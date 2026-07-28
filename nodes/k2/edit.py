"""FVM K2 Lab — regionales Bild-Editieren (Image-Edit-Workspace).

Bewusst in zwei Knoten geteilt: Vorbereitung (Encode + Latentmaske) und
Rückblendung (Pixel-Composite). Dazwischen passt jeder Sampler, jeder
ControlNet-Zweig und jede Detailstufe.
"""

import json

import comfy.sample
import numpy as np
import torch

from ...core.k2.geometry import TOKEN_PIXELS
from ...core.k2.runtime import K2Runtime, as_image_latent

CATEGORY = "FVM Tools/K2"


def _blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Weicher Rand über wiederholtes Average-Pooling (kein cv2/scipy nötig)."""
    if radius <= 0:
        return mask
    kernel = max(3, int(radius) | 1)
    padding = kernel // 2
    working = mask.unsqueeze(1)
    for _ in range(2):
        working = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(working, (padding,) * 4, mode="replicate"),
            kernel_size=kernel,
            stride=1,
        )
    return working.squeeze(1)


class FVM_K2_EditLatent:
    """Bereitet ein Quellbild für regionales Editieren vor."""

    DESCRIPTION = (
        "Encodes a source image and builds the latent denoise mask for regional "
        "editing: only the selected region boxes are re-denoised, everything else "
        "keeps its original latent.\n\n"
        "latent_feather is the transition collar used DURING denoising — it lets the "
        "model blend structure around the edit instead of meeting a hard latent "
        "boundary. The final pixel-level protection is done by K2 Edit Composite.\n\n"
        "Start with low denoise on the sampler plus the default feather; if a seam is "
        "visible, widen the feather or enlarge the box rather than raising denoise."
    )
    CATEGORY = CATEGORY
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "edit_mask", "source", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image to edit."}),
                "vae": ("VAE", {"tooltip": "VAE used to encode the source."}),
                "latent_feather": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8,
                                   "tooltip": "Soft collar around each edit box in pixels."}),
                "grow_px": ("INT", {"default": 0, "min": -256, "max": 512, "step": 8,
                            "tooltip": "Expand every edit box before masking."}),
                "edit_entire_image": ("BOOLEAN", {"default": False,
                                      "tooltip": "Ignores the boxes and edits everything. "
                                      "Only for intentional whole-scene changes."}),
            },
            "optional": {
                "regions": ("K2_REGION", {"tooltip": "Edit boxes. Alternatively connect a "
                            "plan or an explicit mask."}),
                "plan": ("K2_PLAN", {"tooltip": "Uses the region layout of a compiled plan."}),
                "mask": ("MASK", {"tooltip": "Explicit mask; overrides regions/plan."}),
            },
        }

    def prepare(self, image, vae, latent_feather, grow_px, edit_entire_image,
                regions=None, plan=None, mask=None):
        batch, height, width = image.shape[0], image.shape[1], image.shape[2]

        if edit_entire_image:
            pixel_mask = torch.ones(1, height, width, dtype=torch.float32)
            source = "entire_image"
        elif mask is not None:
            pixel_mask = mask.detach().cpu().float()
            if pixel_mask.ndim == 2:
                pixel_mask = pixel_mask.unsqueeze(0)
            if pixel_mask.shape[-2:] != (height, width):
                pixel_mask = torch.nn.functional.interpolate(
                    pixel_mask.unsqueeze(1), size=(height, width),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
            source = "mask_input"
        else:
            boxes = []
            if plan is not None and isinstance(plan, K2Runtime):
                boxes = [region.box for region in plan.bound.plan.regions]
            elif regions:
                boxes = [r.box for r in regions if r.enabled]
            if not boxes:
                raise ValueError(
                    "No edit region given. Connect regions, a plan or a mask — or turn "
                    "on edit_entire_image if that is really intended."
                )
            pixel_mask = torch.zeros(1, height, width, dtype=torch.float32)
            for box in boxes:
                grown = box.grown(float(grow_px)) if grow_px else box
                x0 = max(0, int(grown.x0))
                y0 = max(0, int(grown.y0))
                x1 = min(width, int(round(grown.x1)))
                y1 = min(height, int(round(grown.y1)))
                if x1 > x0 and y1 > y0:
                    pixel_mask[0, y0:y1, x0:x1] = 1.0
            source = f"{len(boxes)} region box(es)"

        if latent_feather > 0:
            pixel_mask = _blur_mask(pixel_mask, int(latent_feather))
        pixel_mask = pixel_mask.clamp(0.0, 1.0)

        with torch.no_grad():
            latent = as_image_latent(vae.encode(image[:, :, :, :3]))

        latent_height = latent.shape[-2]
        latent_width = latent.shape[-1]
        noise_mask = torch.nn.functional.interpolate(
            pixel_mask.unsqueeze(1), size=(latent_height, latent_width),
            mode="bilinear", align_corners=False,
        )
        if noise_mask.shape[0] != latent.shape[0]:
            noise_mask = noise_mask.expand(latent.shape[0], -1, -1, -1)

        info = json.dumps(
            {
                "mask_source": source,
                "image": [int(width), int(height)],
                "latent": [int(latent_width), int(latent_height)],
                "latent_shape": [int(v) for v in latent.shape],
                "token_pixels": TOKEN_PIXELS,
                "masked_fraction": round(float(pixel_mask.mean()), 4),
                "latent_feather": int(latent_feather),
            },
            indent=2,
        )
        return (
            {"samples": latent, "noise_mask": noise_mask},
            pixel_mask,
            image,
            info,
        )


class FVM_K2_EditComposite:
    """Blendet ein editiertes Bild geschützt über das Original."""

    DESCRIPTION = (
        "Composites an edited image back over its source using a feathered mask.\n\n"
        "Pixels outside the composite support are copied byte-exact from the source, "
        "so a regional edit can never degrade the rest of the image through VAE "
        "round-trip loss."
    )
    CATEGORY = CATEGORY
    FUNCTION = "composite"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE", {"tooltip": "Original image — everything outside the "
                           "mask is copied from here unchanged."}),
                "edited": ("IMAGE", {"tooltip": "Result of the edit pass."}),
                "mask": ("MASK", {"tooltip": "Where the edit is allowed (K2 Edit Latent "
                         "output)."}),
                "composite_feather": ("INT", {"default": 48, "min": 0, "max": 512, "step": 4,
                                      "tooltip": "Pixel-space blend width. Narrower than "
                                      "the latent feather on purpose."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                             "tooltip": "Opacity of the edit. 0 returns the source."}),
            },
        }

    def composite(self, source, edited, mask, composite_feather, strength):
        height, width = source.shape[1], source.shape[2]
        if edited.shape[1:3] != source.shape[1:3]:
            edited = torch.nn.functional.interpolate(
                edited.permute(0, 3, 1, 2), size=(height, width),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        blend = mask.detach().cpu().float()
        if blend.ndim == 2:
            blend = blend.unsqueeze(0)
        if blend.shape[-2:] != (height, width):
            blend = torch.nn.functional.interpolate(
                blend.unsqueeze(1), size=(height, width),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        if composite_feather > 0:
            blend = _blur_mask(blend, int(composite_feather))
        blend = (blend.clamp(0.0, 1.0) * float(strength)).to(source.device)

        count = max(source.shape[0], edited.shape[0])
        source_batch = source.expand(count, -1, -1, -1) if source.shape[0] == 1 else source
        edited_batch = edited.expand(count, -1, -1, -1) if edited.shape[0] == 1 else edited
        blend_batch = blend.expand(count, -1, -1) if blend.shape[0] == 1 else blend

        alpha = blend_batch.unsqueeze(-1)
        result = source_batch * (1.0 - alpha) + edited_batch * alpha
        return (result.clamp(0.0, 1.0), blend_batch)


class FVM_K2_LatentPin:
    """Bindet das Latent außerhalb der Regionen an eine Referenz-Trajektorie."""

    DESCRIPTION = (
        "Replaces everything outside the region union with a reference latent — hard "
        "spatial locality.\n\n"
        "Why this exists: token-gated regional LoRAs stop the DELTA at the box edge, "
        "but image-to-image attention stays continuous (on purpose — otherwise you get "
        "tile seams). A regional LoRA therefore still nudges the trajectory outside its "
        "box a little. Sample the same seed twice — once with and once without the "
        "regional LoRAs — and pin the outside to the clean run to remove that residual "
        "drift.\n\n"
        "feather_tokens softens the boundary in latent space; 0 is a hard cut."
    )
    CATEGORY = CATEGORY
    FUNCTION = "pin"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regional": ("LATENT", {"tooltip": "Result of the run WITH regional LoRAs."}),
                "base": ("LATENT", {"tooltip": "Result of the same seed WITHOUT them."}),
                "mask": ("MASK", {"tooltip": "Region union mask from K2 Compose."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                             "tooltip": "1.0 pins the outside completely to the base run, "
                             "0 returns the regional result unchanged."}),
                "feather_tokens": ("INT", {"default": 2, "min": 0, "max": 32,
                                   "tooltip": "Soft boundary width in latent cells."}),
            },
        }

    def pin(self, regional, base, mask, strength, feather_tokens):
        samples = regional["samples"]
        reference = base["samples"]
        if reference.shape != samples.shape:
            raise ValueError(
                f"Latent shapes differ: regional {tuple(samples.shape)} vs base "
                f"{tuple(reference.shape)} — both runs need the same size and batch."
            )

        blend = mask.detach().to(samples.device).float()
        if blend.ndim == 2:
            blend = blend.unsqueeze(0)
        blend = torch.nn.functional.interpolate(
            blend.unsqueeze(1), size=samples.shape[-2:], mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        if feather_tokens > 0:
            blend = _blur_mask(blend, int(feather_tokens) * 2 + 1)
        blend = blend.clamp(0.0, 1.0)

        keep = blend + (1.0 - blend) * (1.0 - float(strength))
        keep = keep.unsqueeze(1)
        if keep.shape[0] != samples.shape[0]:
            keep = keep.expand(samples.shape[0], -1, -1, -1)

        out = dict(regional)
        out["samples"] = samples * keep + reference.to(samples.device) * (1.0 - keep)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "FVM_K2_EditLatent": FVM_K2_EditLatent,
    "FVM_K2_EditComposite": FVM_K2_EditComposite,
    "FVM_K2_LatentPin": FVM_K2_LatentPin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_K2_EditLatent": "K2 Edit Latent",
    "FVM_K2_EditComposite": "K2 Edit Composite",
    "FVM_K2_LatentPin": "K2 Latent Pin (outside regions)",
}
