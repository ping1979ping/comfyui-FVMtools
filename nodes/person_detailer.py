import torch
import numpy as np

import comfy.sd
import comfy.utils
import comfy.samplers
import folder_paths

from .utils.mask_utils import is_mask_empty, split_mask_to_components
from .utils.inpaint_pipeline import inpaint_slot
from .utils.detail_daemon import DD_DEFAULTS
from .utils.lora_utils import is_z_image_turbo, needs_qkv_conversion, convert_qkv_lora


# Default inpaint options (used when InpaintOptions node is not connected)
INPAINT_DEFAULTS = {
    "mask_fill_holes": True,
    "context_expand_factor": 1.20,
    "output_padding": 32,
    "slots": {
        prefix: {"mask_type": "head", "lora_strength": 1.0, "detail_daemon": True}
        for prefix in ["reference_1", "reference_2", "reference_3", "reference_4", "reference_5", "generic"]
    },
}


class PersonDetailer:
    """All-in-one face detailing node. Iterates over reference slots per batch image,
    applies per-slot LoRA and prompt, inpaints the masked region, and stitches back.
    Unmatched faces can be handled by the generic slot via connected components."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "refined")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Per-person face detailing with individual LoRA and prompt per reference slot.\n\n"
        "Connect PERSON_DATA from Person Selector Multi to assign faces to slots.\n"
        "Each enabled slot inpaints its matched face region using the slot's LoRA and prompt.\n"
        "The generic slot handles unmatched faces via connected components with a size threshold.\n\n"
        "Optional inputs:\n"
        "- positive_base: fallback conditioning when a slot's prompt is empty\n"
        "- negative: negative conditioning (empty if not connected)\n"
        "- dd_options: Detail Daemon fine-tuning from Detail Daemon Options node\n"
        "- inpaint_options: advanced inpaint settings from Inpaint Options node"
    )

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")

        slot_widgets = {}
        for i in range(1, 6):
            prefix = f"reference_{i}_"
            slot_widgets[f"{prefix}enabled"] = ("BOOLEAN", {"default": i == 1,
                                                             "tooltip": f"Enable reference slot {i} for detailing"})
            slot_widgets[f"{prefix}lora"] = (lora_list, {"tooltip": f"LoRA to apply when detailing reference {i}"})
            slot_widgets[f"{prefix}lora_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                                                  "tooltip": f"LoRA strength for reference {i}"})
            slot_widgets[f"{prefix}prompt"] = ("STRING", {"multiline": True, "default": "",
                                                           "tooltip": f"Positive prompt for reference {i}. If empty, uses base conditioning."})

        # Generic slot
        slot_widgets["generic_enabled"] = ("BOOLEAN", {"default": False,
                                                        "tooltip": "Enable detailing for unmatched faces (not assigned to any reference)"})
        slot_widgets["generic_lora"] = (lora_list, {"tooltip": "LoRA to apply for unmatched faces"})
        slot_widgets["generic_lora_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                                            "tooltip": "LoRA strength for unmatched faces"})
        slot_widgets["generic_prompt"] = ("STRING", {"multiline": True, "default": "",
                                                      "tooltip": "Positive prompt for unmatched faces. If empty, uses base conditioning."})

        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image batch [B, H, W, C]"}),
                "person_data": ("PERSON_DATA", {"tooltip": "Person data from Person Selector Multi node"}),
                "model": ("MODEL", {"tooltip": "Base model (LoRAs are applied as temporary clones per slot)"}),
                "clip": ("CLIP", {"tooltip": "CLIP model for prompt encoding"}),
                "vae": ("VAE", {"tooltip": "VAE for encode/decode in the inpaint pipeline"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                  "tooltip": "Global seed, fixed across all slots and batch items"}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100,
                                   "tooltip": "Number of sampling steps per inpaint"}),
                "denoise": ("FLOAT", {"default": 0.52, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Denoise strength for inpainting (lower = more original detail preserved)"}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"tooltip": "Sampler algorithm"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"tooltip": "Noise schedule"}),
                "detail_daemon_enabled": ("BOOLEAN", {"default": True,
                                                       "tooltip": "Enable Detail Daemon sigma manipulation for enhanced detail preservation"}),
                "detail_amount": ("FLOAT", {"default": 0.20, "min": -5.0, "max": 5.0, "step": 0.01,
                                             "tooltip": "Detail Daemon strength. Positive = more detail, negative = smoother. 0 = off."}),
                "dd_smooth": ("BOOLEAN", {"default": True,
                                           "tooltip": "Smooth the Detail Daemon sigma curve to avoid artifacts"}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 128, "step": 1,
                                               "tooltip": "Gaussian feather radius at mask edges for seamless blending"}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                                "tooltip": "Dilate/expand mask by this many pixels before inpainting"}),
                "target_width": ("INT", {"default": 800, "min": 64, "max": 4096, "step": 8,
                                          "tooltip": "Width to resize face crops to before sampling"}),
                "target_height": ("INT", {"default": 1200, "min": 64, "max": 4096, "step": 8,
                                           "tooltip": "Height to resize face crops to before sampling"}),
                **slot_widgets,
            },
            "optional": {
                "positive_base": ("CONDITIONING", {"tooltip": "Base positive conditioning, used as fallback when a slot's prompt is empty"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning. If not connected, an empty negative is used."}),
                "dd_options": ("DD_OPTIONS", {"tooltip": "Advanced Detail Daemon parameters from Detail Daemon Options node"}),
                "inpaint_options": ("INPAINT_OPTIONS", {"tooltip": "Advanced inpaint settings and per-slot overrides from Inpaint Options node"}),
            },
        }

    def _get_slot_config(self, slot_key, inpaint_options):
        """Get per-slot config from InpaintOptions or defaults."""
        opts = inpaint_options or INPAINT_DEFAULTS
        slots = opts.get("slots", INPAINT_DEFAULTS["slots"])
        return slots.get(slot_key, INPAINT_DEFAULTS["slots"]["reference_1"])

    def _encode_prompt(self, clip, prompt_text):
        """Encode a text prompt using CLIP."""
        tokens = clip.tokenize(prompt_text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return [[cond, output]]

    def _apply_lora(self, model, clip, lora_name, strength):
        """Apply LoRA to model and clip, returning patched clones.
        Auto-converts LoRA for Z-Image Turbo if needed."""
        if lora_name == "None" or strength == 0:
            return model, clip
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # Auto-convert for Z-Image Turbo (Lumina2) if LoRA has separate Q/K/V
        if is_z_image_turbo(model) and needs_qkv_conversion(lora):
            print(f"[FVMTools] Auto-converting LoRA '{lora_name}' for Z-Image Turbo QKV format")
            lora = convert_qkv_lora(lora)

        patched_model, patched_clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
        return patched_model, patched_clip

    def execute(self, images, person_data, model, clip, vae,
                seed, steps, denoise, sampler_name, scheduler,
                detail_daemon_enabled, detail_amount, dd_smooth,
                mask_blend_pixels, mask_expand_pixels, target_width, target_height,
                reference_1_enabled, reference_1_lora, reference_1_lora_strength, reference_1_prompt,
                reference_2_enabled, reference_2_lora, reference_2_lora_strength, reference_2_prompt,
                reference_3_enabled, reference_3_lora, reference_3_lora_strength, reference_3_prompt,
                reference_4_enabled, reference_4_lora, reference_4_lora_strength, reference_4_prompt,
                reference_5_enabled, reference_5_lora, reference_5_lora_strength, reference_5_prompt,
                generic_enabled, generic_lora, generic_lora_strength, generic_prompt,
                positive_base=None, negative=None, dd_options=None, inpaint_options=None):

        batch_size = images.shape[0]
        inpaint_opts = inpaint_options or INPAINT_DEFAULTS

        # If no negative conditioning, create empty one
        if negative is None:
            negative = self._encode_prompt(clip, "")

        # Collect slot configs
        slots = []
        for i, (enabled, lora, lora_str, prompt) in enumerate([
            (reference_1_enabled, reference_1_lora, reference_1_lora_strength, reference_1_prompt),
            (reference_2_enabled, reference_2_lora, reference_2_lora_strength, reference_2_prompt),
            (reference_3_enabled, reference_3_lora, reference_3_lora_strength, reference_3_prompt),
            (reference_4_enabled, reference_4_lora, reference_4_lora_strength, reference_4_prompt),
            (reference_5_enabled, reference_5_lora, reference_5_lora_strength, reference_5_prompt),
        ], start=1):
            if not enabled:
                continue
            slot_key = f"reference_{i}"
            slot_cfg = self._get_slot_config(slot_key, inpaint_options)
            slots.append({
                "index": i - 1,  # 0-based index into person_data masks
                "label": f"Reference {i}",
                "lora": lora,
                "prompt": prompt,
                "mask_type": slot_cfg.get("mask_type", "head"),
                "lora_strength": lora_str,
                "use_dd": slot_cfg.get("detail_daemon", True),
            })

        generic_cfg = self._get_slot_config("generic", inpaint_options)

        active_slots = len(slots) + (1 if generic_enabled else 0)
        total_steps = batch_size * active_slots
        current_step = 0

        print(f"\n{'='*50}")
        print(f"  PersonDetailer v1.0")
        print(f"  Batch: {batch_size} images | Active slots: {active_slots}")
        print(f"{'='*50}")

        results = []
        refined_parts = []

        for b in range(batch_size):
            current_image = images[b]  # [H, W, C]

            print(f"\n  [Batch {b+1}/{batch_size}]")

            # Process reference slots
            for slot in slots:
                ri = slot["index"]
                mask_type = slot["mask_type"]

                # Get mask from PERSON_DATA
                mask_key = f"{mask_type}_masks"
                mask = person_data[mask_key][ri][b]  # [H, W]

                if is_mask_empty(mask):
                    print(f"    {slot['label']} — no match, skip")
                    continue

                current_step += 1
                print(f"    [{current_step}/{total_steps}] {slot['label']} — detailing...")

                # Apply LoRA
                patched_model, patched_clip = self._apply_lora(
                    model, clip, slot["lora"], slot["lora_strength"]
                )

                # Encode prompt (slot prompt or fallback to base)
                if slot["prompt"].strip():
                    positive_cond = self._encode_prompt(patched_clip, slot["prompt"])
                elif positive_base is not None:
                    positive_cond = positive_base
                else:
                    positive_cond = self._encode_prompt(patched_clip, "")

                # Determine DD settings for this slot
                slot_dd_enabled = detail_daemon_enabled and slot["use_dd"]

                # Run inpaint pipeline
                stitched, refined = inpaint_slot(
                    image=current_image,
                    mask_2d=mask,
                    model=patched_model,
                    positive_cond=positive_cond,
                    negative_cond=negative,
                    vae=vae,
                    seed=seed,
                    steps=steps,
                    denoise=denoise,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    target_width=target_width,
                    target_height=target_height,
                    mask_expand_pixels=mask_expand_pixels,
                    mask_blend_pixels=mask_blend_pixels,
                    mask_fill_holes=inpaint_opts.get("mask_fill_holes", True),
                    context_expand_factor=inpaint_opts.get("context_expand_factor", 1.20),
                    output_padding=inpaint_opts.get("output_padding", 32),
                    dd_enabled=slot_dd_enabled,
                    dd_amount=detail_amount,
                    dd_smooth=dd_smooth,
                    dd_options=dd_options,
                )

                current_image = stitched
                if refined is not None:
                    refined_parts.append(refined)

            # Generic slot: unmatched faces
            if generic_enabled:
                unmatched_mask = person_data["all_faces_mask"][b] - person_data["matched_faces_mask"][b]
                unmatched_mask = unmatched_mask.clamp(0, 1)

                # Split into individual face components
                components = split_mask_to_components(unmatched_mask, min_area_fraction=0.001)

                if components:
                    current_step += 1
                    print(f"    [{current_step}/{total_steps}] Generic — {len(components)} unmatched face(s)")

                    generic_use_dd = generic_cfg.get("detail_daemon", True)

                    patched_model, patched_clip = self._apply_lora(
                        model, clip, generic_lora, generic_lora_strength
                    )

                    if generic_prompt.strip():
                        positive_cond = self._encode_prompt(patched_clip, generic_prompt)
                    elif positive_base is not None:
                        positive_cond = positive_base
                    else:
                        positive_cond = self._encode_prompt(patched_clip, "")

                    slot_dd_enabled = detail_daemon_enabled and generic_use_dd

                    for comp_mask in components:
                        stitched, refined = inpaint_slot(
                            image=current_image,
                            mask_2d=comp_mask,
                            model=patched_model,
                            positive_cond=positive_cond,
                            negative_cond=negative,
                            vae=vae,
                            seed=seed,
                            steps=steps,
                            denoise=denoise,
                            sampler_name=sampler_name,
                            scheduler=scheduler,
                            target_width=target_width,
                            target_height=target_height,
                            mask_expand_pixels=mask_expand_pixels,
                            mask_blend_pixels=mask_blend_pixels,
                            mask_fill_holes=inpaint_opts.get("mask_fill_holes", True),
                            context_expand_factor=inpaint_opts.get("context_expand_factor", 1.20),
                            output_padding=inpaint_opts.get("output_padding", 32),
                            dd_enabled=slot_dd_enabled,
                            dd_amount=detail_amount,
                            dd_smooth=dd_smooth,
                            dd_options=dd_options,
                        )
                        current_image = stitched
                        if refined is not None:
                            refined_parts.append(refined)
                else:
                    print(f"    Generic — no unmatched faces")

            results.append(current_image)

        print(f"\n{'='*50}")
        print(f"  Done! Processed {batch_size} images.")
        print(f"{'='*50}\n")

        # Stack results
        output_images = torch.stack(results)  # [B, H, W, C]

        if refined_parts:
            output_refined = torch.cat(refined_parts, dim=0)  # [N, tH, tW, C]
        else:
            output_refined = output_images

        return {
            "ui": {"text": [f"Processed {batch_size} images, {len(refined_parts)} refinements"]},
            "result": (output_images, output_refined),
        }
