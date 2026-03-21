import torch
import numpy as np
import cv2

import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.model_management
import latent_preview
from comfy_extras.nodes_custom_sampler import Guider_Basic

from .mask_utils import expand_mask, feather_mask, fill_mask_holes_2d
from .detail_daemon import apply_detail_daemon_to_sigmas, DD_DEFAULTS


def compute_crop_region(mask_2d, context_expand_factor=1.20, padding=32):
    """Compute expanded crop region around non-zero mask pixels.

    Args:
        mask_2d: [H, W] float32 tensor
        context_expand_factor: how much to expand the bounding box
        padding: additional padding in pixels (ensures divisible by 8)

    Returns:
        dict with y_min, y_max, x_min, x_max, orig_crop_h, orig_crop_w
        or None if mask is empty.
    """
    coords = torch.nonzero(mask_2d > 0.5)
    if coords.numel() == 0:
        return None

    H, W = mask_2d.shape
    y_min, y_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    x_min, x_max = coords[:, 1].min().item(), coords[:, 1].max().item()

    # Expand by context factor
    crop_h = y_max - y_min
    crop_w = x_max - x_min
    expand_h = int(crop_h * (context_expand_factor - 1.0) / 2)
    expand_w = int(crop_w * (context_expand_factor - 1.0) / 2)

    y_min = max(0, y_min - expand_h - padding)
    y_max = min(H - 1, y_max + expand_h + padding)
    x_min = max(0, x_min - expand_w - padding)
    x_max = min(W - 1, x_max + expand_w + padding)

    return {
        "y_min": y_min, "y_max": y_max,
        "x_min": x_min, "x_max": x_max,
        "orig_crop_h": y_max - y_min + 1,
        "orig_crop_w": x_max - x_min + 1,
    }


def crop_and_resize(image, mask_2d, crop_region, target_w, target_h):
    """Crop image and mask from region, resize to target dimensions.

    Args:
        image: [H, W, C] float32 tensor
        mask_2d: [H, W] float32 tensor
        crop_region: dict from compute_crop_region
        target_w, target_h: target dimensions

    Returns:
        (cropped_image [1, tH, tW, C], cropped_mask [tH, tW], crop_region)
    """
    y1, y2 = crop_region["y_min"], crop_region["y_max"] + 1
    x1, x2 = crop_region["x_min"], crop_region["x_max"] + 1

    cropped_image = image[y1:y2, x1:x2, :]  # [cH, cW, C]
    cropped_mask = mask_2d[y1:y2, x1:x2]    # [cH, cW]

    # Resize image using comfy.utils.common_upscale (expects [B, C, H, W])
    img_bchw = cropped_image.unsqueeze(0).permute(0, 3, 1, 2)  # [1, C, cH, cW]
    img_bchw = comfy.utils.common_upscale(img_bchw, target_w, target_h, "lanczos", "disabled")
    resized_image = img_bchw.permute(0, 2, 3, 1)  # [1, tH, tW, C]

    # Resize mask
    mask_4d = cropped_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, cH, cW]
    mask_4d = torch.nn.functional.interpolate(mask_4d, size=(target_h, target_w), mode="bilinear", align_corners=False)
    resized_mask = mask_4d.squeeze(0).squeeze(0)  # [tH, tW]

    return resized_image, resized_mask


def stitch_back(original_image, decoded_crop, blend_mask, crop_region):
    """Stitch decoded crop back into the original image with blending.

    Args:
        original_image: [H, W, C] float32 tensor
        decoded_crop: [1, tH, tW, C] float32 tensor (decoded output)
        blend_mask: [cH, cW] float32 tensor (mask at original crop size, for blending)
        crop_region: dict from compute_crop_region

    Returns:
        [H, W, C] float32 tensor — stitched result
    """
    y1, y2 = crop_region["y_min"], crop_region["y_max"] + 1
    x1, x2 = crop_region["x_min"], crop_region["x_max"] + 1
    orig_crop_h = y2 - y1
    orig_crop_w = x2 - x1

    # Resize decoded back to original crop size
    dec_bchw = decoded_crop.permute(0, 3, 1, 2)  # [1, C, tH, tW]
    dec_bchw = comfy.utils.common_upscale(dec_bchw, orig_crop_w, orig_crop_h, "lanczos", "disabled")
    decoded_at_orig = dec_bchw.permute(0, 2, 3, 1).squeeze(0)  # [cH, cW, C]

    # Blend using mask
    blend_3c = blend_mask.unsqueeze(-1)  # [cH, cW, 1]

    result = original_image.clone()
    orig_crop = result[y1:y2, x1:x2, :]
    result[y1:y2, x1:x2, :] = decoded_at_orig * blend_3c + orig_crop * (1 - blend_3c)

    return result


def inpaint_slot(
    image,
    mask_2d,
    model,
    positive_cond,
    negative_cond,
    vae,
    seed,
    steps,
    denoise,
    sampler_name,
    scheduler,
    target_width,
    target_height,
    mask_expand_pixels=0,
    mask_blend_pixels=32,
    mask_fill_holes=True,
    context_expand_factor=1.20,
    output_padding=32,
    dd_enabled=False,
    dd_amount=0.0,
    dd_smooth=True,
    dd_options=None,
):
    """Run the full inpaint pipeline for a single masked region.

    Args:
        image: [H, W, C] single image float32
        mask_2d: [H, W] single mask float32
        model: ModelPatcher (possibly LoRA-patched clone)
        positive_cond: CONDITIONING (already encoded)
        negative_cond: CONDITIONING
        vae: VAE
        seed: int
        steps: int
        denoise: float
        sampler_name: str
        scheduler: str
        target_width, target_height: crop resize target
        mask_expand_pixels: int
        mask_blend_pixels: int
        mask_fill_holes: bool
        context_expand_factor: float
        output_padding: int
        dd_enabled: bool — whether to apply Detail Daemon
        dd_amount: float — Detail Daemon strength
        dd_smooth: bool
        dd_options: dict or None — advanced DD options from DetailDaemonOptions node

    Returns:
        (stitched_image [H,W,C], refined_crop [1, tH, tW, C])
        or (image, None) if mask is empty
    """
    # Step 1: Mask preprocessing
    processed_mask = mask_2d.clone()
    if mask_fill_holes:
        processed_mask = fill_mask_holes_2d(processed_mask)
    if mask_expand_pixels > 0:
        processed_mask = expand_mask(processed_mask, mask_expand_pixels)

    # Step 2: Compute crop region
    crop_region = compute_crop_region(processed_mask, context_expand_factor, output_padding)
    if crop_region is None:
        return image, None

    # Keep a copy of the mask at original crop size for stitching
    y1, y2 = crop_region["y_min"], crop_region["y_max"] + 1
    x1, x2 = crop_region["x_min"], crop_region["x_max"] + 1
    blend_mask_orig = processed_mask[y1:y2, x1:x2].clone()
    if mask_blend_pixels > 0:
        blend_mask_orig = feather_mask(blend_mask_orig, mask_blend_pixels)

    # Step 3: Crop and resize to target
    cropped_image, cropped_mask = crop_and_resize(
        image, processed_mask, crop_region, target_width, target_height
    )

    # Step 4: Feather the cropped mask for noise masking
    if mask_blend_pixels > 0:
        cropped_mask = feather_mask(cropped_mask, mask_blend_pixels)

    # Step 5: VAE encode
    device = comfy.model_management.get_torch_device()
    cropped_image_device = cropped_image.to(device)
    latent_samples = vae.encode(cropped_image_device[:, :, :, :3])
    latent = {"samples": latent_samples}

    # Step 6: Set noise mask
    noise_mask = cropped_mask.reshape(1, 1, target_height, target_width).to(device)
    latent["noise_mask"] = noise_mask

    # Step 7: Compute sigmas
    model_sampling = model.get_model_object("model_sampling")
    total_steps = steps
    if denoise < 1.0:
        total_steps = int(steps / denoise)

    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps).cpu()
    sigmas = sigmas[-(steps + 1):]

    # Apply Detail Daemon if enabled
    if dd_enabled and dd_amount != 0:
        dd_opts = dd_options or DD_DEFAULTS
        sigmas = apply_detail_daemon_to_sigmas(
            sigmas,
            detail_amount=dd_amount,
            start=dd_opts.get("dd_start", DD_DEFAULTS["dd_start"]),
            end=dd_opts.get("dd_end", DD_DEFAULTS["dd_end"]),
            bias=dd_opts.get("dd_bias", DD_DEFAULTS["dd_bias"]),
            exponent=dd_opts.get("dd_exponent", DD_DEFAULTS["dd_exponent"]),
            start_offset=dd_opts.get("dd_start_offset", DD_DEFAULTS["dd_start_offset"]),
            end_offset=dd_opts.get("dd_end_offset", DD_DEFAULTS["dd_end_offset"]),
            fade=dd_opts.get("dd_fade", DD_DEFAULTS["dd_fade"]),
            smooth=dd_smooth,
        )

    # Step 8: Prepare noise and sample
    noise = comfy.sample.prepare_noise(latent_samples, seed)
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_samples)

    sampler_obj = comfy.samplers.sampler_object(sampler_name)

    guider = Guider_Basic(model)
    guider.set_conds(positive_cond)

    noise_mask_for_sample = latent.get("noise_mask", None)

    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(
        noise, latent_image, sampler_obj, sigmas,
        denoise_mask=noise_mask_for_sample,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    # Step 9: VAE decode
    decoded = vae.decode(samples)  # [1, tH, tW, C]

    # Step 10: Stitch back
    stitched = stitch_back(image, decoded, blend_mask_orig, crop_region)

    return stitched, decoded
