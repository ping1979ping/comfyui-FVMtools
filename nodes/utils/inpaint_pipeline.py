"""Inpaint pipeline: crop → resize → encode → sample → decode → stitch.

Aspect-ratio-preserving crop approach inspired by ComfyUI-Inpaint-CropAndStitch.
The crop region is expanded to match the target aspect ratio BEFORE resizing,
so the resize step never distorts the image.
"""

import torch
import numpy as np
import math

import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.model_management
import latent_preview
from comfy_extras.nodes_custom_sampler import Guider_Basic

from .mask_utils import expand_mask, feather_mask, fill_mask_holes_2d
from .detail_daemon import apply_detail_daemon_to_sigmas, DD_DEFAULTS


def compute_crop_region(mask_2d, context_expand_factor=1.20, padding=32):
    """Find bounding box of mask and expand by context factor + padding."""
    coords = torch.nonzero(mask_2d > 0.5)
    if coords.numel() == 0:
        return None

    H, W = mask_2d.shape
    y_min, y_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    x_min, x_max = coords[:, 1].min().item(), coords[:, 1].max().item()

    crop_h = y_max - y_min
    crop_w = x_max - x_min
    expand_h = int(crop_h * (context_expand_factor - 1.0) / 2)
    expand_w = int(crop_w * (context_expand_factor - 1.0) / 2)

    y_min = max(0, y_min - expand_h - padding)
    y_max = min(H - 1, y_max + expand_h + padding)
    x_min = max(0, x_min - expand_w - padding)
    x_max = min(W - 1, x_max + expand_w + padding)

    return {
        "x": x_min, "y": y_min,
        "w": x_max - x_min + 1,
        "h": y_max - y_min + 1,
        "img_w": W, "img_h": H,
    }


def adjust_crop_to_aspect_ratio(crop, target_w, target_h):
    """Expand crop region to match target aspect ratio without distortion.

    Only expands — never shrinks. One axis stays the same, the other grows
    to match the target AR. Handles boundary clamping.
    """
    x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
    img_w, img_h = crop["img_w"], crop["img_h"]

    target_ar = target_w / target_h
    crop_ar = w / h

    if crop_ar < target_ar:
        # Need wider — expand horizontally
        new_w = int(h * target_ar)
        new_h = h
        new_x = x - (new_w - w) // 2
        new_y = y
    else:
        # Need taller — expand vertically
        new_w = w
        new_h = int(w / target_ar)
        new_x = x
        new_y = y - (new_h - h) // 2

    # Clamp to image bounds, shift if possible
    if new_x < 0:
        new_x = 0
    if new_x + new_w > img_w:
        new_x = max(0, img_w - new_w)
        new_w = min(new_w, img_w)
    if new_y < 0:
        new_y = 0
    if new_y + new_h > img_h:
        new_y = max(0, img_h - new_h)
        new_h = min(new_h, img_h)

    # Ensure divisible by 8 (VAE requirement)
    new_w = max(8, math.ceil(new_w / 8) * 8)
    new_h = max(8, math.ceil(new_h / 8) * 8)

    # Re-clamp after rounding
    if new_x + new_w > img_w:
        new_x = max(0, img_w - new_w)
    if new_y + new_h > img_h:
        new_y = max(0, img_h - new_h)

    return {
        "x": new_x, "y": new_y,
        "w": min(new_w, img_w), "h": min(new_h, img_h),
        "img_w": img_w, "img_h": img_h,
    }


def create_canvas_with_edge_replication(image, crop):
    """Create a canvas that handles out-of-bounds crop regions via edge replication.

    If the crop is fully within the image, just returns the crop directly.
    If it extends beyond, creates a padded canvas with replicated edge pixels.

    Args:
        image: [H, W, C] float32 tensor
        crop: dict with x, y, w, h, img_w, img_h

    Returns:
        (cropped_image [cH, cW, C], actual_crop dict)
    """
    x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
    H, W = image.shape[0], image.shape[1]

    # Simple case: crop fits within image
    x_end = min(x + w, W)
    y_end = min(y + h, H)
    x_start = max(0, x)
    y_start = max(0, y)

    cropped = image[y_start:y_end, x_start:x_end, :]

    # If crop fits perfectly, return directly
    actual_w = x_end - x_start
    actual_h = y_end - y_start
    if actual_w == w and actual_h == h and x >= 0 and y >= 0:
        return cropped, {"x": x_start, "y": y_start, "w": actual_w, "h": actual_h,
                         "img_w": W, "img_h": H, "pad_l": 0, "pad_t": 0}

    # Need edge replication for out-of-bounds areas
    canvas = torch.zeros(h, w, image.shape[2], dtype=image.dtype, device=image.device)

    # Calculate padding amounts
    pad_l = max(0, -x)
    pad_t = max(0, -y)
    pad_r = max(0, (x + w) - W)
    pad_b = max(0, (y + h) - H)

    # Place original crop
    canvas[pad_t:pad_t + actual_h, pad_l:pad_l + actual_w, :] = cropped

    # Replicate edges
    if pad_t > 0:
        canvas[:pad_t, pad_l:pad_l + actual_w, :] = cropped[0:1, :, :].expand(pad_t, -1, -1)
    if pad_b > 0:
        canvas[pad_t + actual_h:, pad_l:pad_l + actual_w, :] = cropped[-1:, :, :].expand(pad_b, -1, -1)
    if pad_l > 0:
        canvas[:, :pad_l, :] = canvas[:, pad_l:pad_l + 1, :].expand(-1, pad_l, -1)
    if pad_r > 0:
        canvas[:, pad_l + actual_w:, :] = canvas[:, pad_l + actual_w - 1:pad_l + actual_w, :].expand(-1, pad_r, -1)

    return canvas, {"x": x_start, "y": y_start, "w": actual_w, "h": actual_h,
                    "img_w": W, "img_h": H, "pad_l": pad_l, "pad_t": pad_t}


def crop_and_resize(image, mask_2d, crop, target_w, target_h):
    """Crop image/mask from AR-adjusted region and resize proportionally.

    Since the crop region already matches the target AR, the resize is
    proportional — no distortion.
    """
    canvas, stitch_info = create_canvas_with_edge_replication(image, crop)

    # Crop mask (clamp to image bounds, no edge replication needed for mask)
    x, y = max(0, crop["x"]), max(0, crop["y"])
    x_end = min(x + crop["w"], mask_2d.shape[1])
    y_end = min(y + crop["h"], mask_2d.shape[0])
    cropped_mask = torch.zeros(crop["h"], crop["w"], dtype=mask_2d.dtype)
    actual_h = y_end - y
    actual_w = x_end - x
    pad_l = stitch_info["pad_l"]
    pad_t = stitch_info["pad_t"]
    cropped_mask[pad_t:pad_t + actual_h, pad_l:pad_l + actual_w] = mask_2d[y:y_end, x:x_end]

    # Resize to target (now proportional since AR matches)
    img_bchw = canvas.unsqueeze(0).permute(0, 3, 1, 2)
    img_bchw = comfy.utils.common_upscale(img_bchw, target_w, target_h, "lanczos", "disabled")
    resized_image = img_bchw.permute(0, 2, 3, 1)

    mask_4d = cropped_mask.unsqueeze(0).unsqueeze(0)
    mask_4d = torch.nn.functional.interpolate(mask_4d, size=(target_h, target_w), mode="bilinear", align_corners=False)
    resized_mask = mask_4d.squeeze(0).squeeze(0)

    return resized_image, resized_mask, stitch_info


def stitch_back(original_image, decoded_crop, blend_mask_orig, crop, stitch_info):
    """Stitch decoded crop back into original image with alpha blending.

    Resizes decoded back to crop dimensions, blends using feathered mask,
    and pastes into the original image.
    """
    x, y = stitch_info["x"], stitch_info["y"]
    actual_w, actual_h = stitch_info["w"], stitch_info["h"]
    pad_l, pad_t = stitch_info["pad_l"], stitch_info["pad_t"]

    # Resize decoded back to crop dimensions
    crop_w, crop_h = crop["w"], crop["h"]
    dec_bchw = decoded_crop.permute(0, 3, 1, 2)
    dec_bchw = comfy.utils.common_upscale(dec_bchw, crop_w, crop_h, "lanczos", "disabled")
    decoded_full = dec_bchw.permute(0, 2, 3, 1).squeeze(0)  # [crop_h, crop_w, C]

    # Extract only the non-padded region
    decoded_actual = decoded_full[pad_t:pad_t + actual_h, pad_l:pad_l + actual_w, :]

    # Blend mask at original size
    blend_3c = blend_mask_orig.unsqueeze(-1)  # [actual_h, actual_w, 1]

    result = original_image.clone()
    orig_region = result[y:y + actual_h, x:x + actual_w, :]
    result[y:y + actual_h, x:x + actual_w, :] = decoded_actual * blend_3c + orig_region * (1 - blend_3c)

    return result


def _parse_progression(value_str, count, default, cast_fn=float):
    """Parse a pipe-separated progression string into a list of values.

    Args:
        value_str: e.g. "0.5|0.3" or "" or None.
        count: number of rounds (desired list length).
        default: fallback value when string is empty.
        cast_fn: float or int.

    Returns:
        list of length `count`. If fewer values given, last value repeats.
    """
    if not value_str or not value_str.strip():
        return [default] * count

    try:
        parts = [cast_fn(v.strip()) for v in value_str.strip().split("|") if v.strip()]
    except (ValueError, TypeError):
        return [default] * count

    if not parts:
        return [default] * count

    # Pad with last value if fewer than count
    while len(parts) < count:
        parts.append(parts[-1])
    return parts[:count]


def inpaint_slot(
    image, mask_2d, model, positive_cond, negative_cond, vae,
    seed, steps, denoise, sampler_name, scheduler,
    target_width, target_height,
    mask_expand_pixels=0, mask_blend_pixels=32, mask_fill_holes=True,
    context_expand_factor=1.20, output_padding=32,
    dd_enabled=False, dd_amount=0.0, dd_smooth=True, dd_options=None,
    repeat=1,
    denoise_progression="", steps_progression="",
    controlnet_apply_fn=None,
):
    """Run the full inpaint pipeline for a single masked region.

    Args:
        repeat: number of inpaint rounds (latent cycling).
        denoise_progression: pipe-separated denoise per round (e.g. "0.5|0.3").
        steps_progression: pipe-separated steps per round (e.g. "6|4").
        When empty, `denoise` and `steps` are used for all rounds.
        controlnet_apply_fn: optional callable(model, positive, negative, crop_image) -> (model, pos, neg).
            When provided, applies control guidance per round using the current crop.
            Can patch the model (Z-Image ControlNet Union) and/or modify conditioning.
            Used by PersonDetailerControlNet to inject pose/depth guidance.

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
    crop = compute_crop_region(processed_mask, context_expand_factor, output_padding)
    if crop is None:
        return image, None

    # Step 3: Adjust crop to target aspect ratio (no distortion)
    crop = adjust_crop_to_aspect_ratio(crop, target_width, target_height)

    # Step 4: Prepare blend mask at original crop size
    x, y = max(0, crop["x"]), max(0, crop["y"])
    x_end = min(x + crop["w"], processed_mask.shape[1])
    y_end = min(y + crop["h"], processed_mask.shape[0])
    blend_mask_orig = processed_mask[y:y_end, x:x_end].clone()
    if mask_blend_pixels > 0:
        blend_mask_orig = feather_mask(blend_mask_orig, mask_blend_pixels)

    # Step 5: Crop and resize (proportional — AR already matches)
    cropped_image, cropped_mask, stitch_info = crop_and_resize(
        image, processed_mask, crop, target_width, target_height
    )

    # Step 6: Feather mask for noise masking
    if mask_blend_pixels > 0:
        cropped_mask = feather_mask(cropped_mask, mask_blend_pixels)

    # Step 7: VAE encode
    device = comfy.model_management.get_torch_device()
    cropped_image_device = cropped_image.to(device)
    latent_samples = vae.encode(cropped_image_device[:, :, :, :3])

    # Step 8: Set noise mask
    noise_mask = cropped_mask.reshape(1, 1, target_height, target_width).to(device)

    # Step 9: Prepare per-round parameters
    repeat = max(1, int(repeat))
    denoise_per_round = _parse_progression(denoise_progression, repeat, denoise, float)
    steps_per_round = _parse_progression(steps_progression, repeat, steps, int)

    # Step 10: Sample (with optional multi-round cycling)
    sampler_obj = comfy.samplers.sampler_object(sampler_name)
    model_sampling = model.get_model_object("model_sampling")

    for iteration in range(repeat):
        if iteration > 0:
            # Re-encode previous result as new latent input (latent cycling)
            latent_samples = vae.encode(decoded[:, :, :, :3].to(device))

        # Apply control guidance per round (if provided)
        round_model = model
        round_positive = positive_cond
        round_negative = negative_cond
        if controlnet_apply_fn is not None:
            current_crop = cropped_image if iteration == 0 else decoded
            round_model, round_positive, round_negative = controlnet_apply_fn(
                model, round_positive, round_negative, current_crop)

        # Create guider with (possibly patched) model and conditioning
        guider = Guider_Basic(round_model)
        guider.set_conds(round_positive)

        # Per-round denoise and steps
        round_denoise = denoise_per_round[iteration]
        round_steps = max(1, int(steps_per_round[iteration]))

        # Compute sigmas for this round
        total_steps = round_steps
        if round_denoise < 1.0:
            total_steps = int(round_steps / round_denoise)

        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps).cpu()
        sigmas = sigmas[-(round_steps + 1):]

        if dd_enabled and dd_amount != 0:
            dd_opts = dd_options or DD_DEFAULTS
            sigmas = apply_detail_daemon_to_sigmas(
                sigmas, detail_amount=dd_amount,
                start=dd_opts.get("dd_start", DD_DEFAULTS["dd_start"]),
                end=dd_opts.get("dd_end", DD_DEFAULTS["dd_end"]),
                bias=dd_opts.get("dd_bias", DD_DEFAULTS["dd_bias"]),
                exponent=dd_opts.get("dd_exponent", DD_DEFAULTS["dd_exponent"]),
                start_offset=dd_opts.get("dd_start_offset", DD_DEFAULTS["dd_start_offset"]),
                end_offset=dd_opts.get("dd_end_offset", DD_DEFAULTS["dd_end_offset"]),
                fade=dd_opts.get("dd_fade", DD_DEFAULTS["dd_fade"]),
                smooth=dd_smooth,
            )

        if repeat > 1:
            print(f"      round {iteration+1}/{repeat}: denoise={round_denoise}, steps={round_steps}")

        noise = comfy.sample.prepare_noise(latent_samples, seed + iteration)
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_samples)

        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = guider.sample(
            noise, latent_image, sampler_obj, sigmas,
            denoise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=seed + iteration,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        # VAE decode
        decoded = vae.decode(samples)  # [1, tH, tW, C]

    # Cleanup after all rounds — break references to patched model clones
    del guider, round_model, round_positive, round_negative

    # Step 12: Stitch back
    stitched = stitch_back(image, decoded, blend_mask_orig, crop, stitch_info)

    return stitched, decoded
