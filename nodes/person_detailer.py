import torch
import numpy as np

import comfy.sd
import comfy.utils
import comfy.samplers
import folder_paths

from .utils.mask_utils import is_mask_empty, split_mask_to_components
from .utils.inpaint_pipeline import inpaint_slot
from .utils.detail_daemon import DD_DEFAULTS


# Default inpaint options (used when InpaintOptions node is not connected)
INPAINT_DEFAULTS = {
    "mask_fill_holes": True,
    "context_expand_factor": 1.20,
    "output_padding": 32,
    "slots": {
        prefix: {"mask_type": "face", "lora_strength": 1.0, "detail_daemon": True}
        for prefix in ["ref_1", "ref_2", "ref_3", "ref_4", "ref_5", "generic"]
    },
}


class PersonDetailer:
    """All-in-one face detailing node with per-slot LoRA and inpaint pipeline."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "refined", "preview")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")

        slot_widgets = {}
        for i in range(1, 6):
            prefix = f"ref_{i}_"
            slot_widgets[f"{prefix}enabled"] = ("BOOLEAN", {"default": i == 1,
                                                             "tooltip": f"Enable reference slot {i}"})
            slot_widgets[f"{prefix}lora"] = (lora_list, {"tooltip": f"LoRA for reference {i}"})
            slot_widgets[f"{prefix}prompt"] = ("STRING", {"multiline": True, "default": "",
                                                           "tooltip": f"Positive prompt for reference {i}"})

        # Generic slot
        slot_widgets["generic_enabled"] = ("BOOLEAN", {"default": False,
                                                        "tooltip": "Enable generic slot for unmatched faces"})
        slot_widgets["generic_lora"] = (lora_list, {"tooltip": "LoRA for unmatched faces"})
        slot_widgets["generic_prompt"] = ("STRING", {"multiline": True, "default": "",
                                                      "tooltip": "Positive prompt for unmatched faces"})

        return {
            "required": {
                "images": ("IMAGE",),
                "person_data": ("PERSON_DATA",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "denoise": ("FLOAT", {"default": 0.52, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "detail_daemon_enabled": ("BOOLEAN", {"default": True,
                                                       "tooltip": "Enable Detail Daemon sigma manipulation"}),
                "detail_amount": ("FLOAT", {"default": 0.20, "min": -5.0, "max": 5.0, "step": 0.01,
                                             "tooltip": "Detail Daemon strength (0 = off)"}),
                "dd_smooth": ("BOOLEAN", {"default": True,
                                           "tooltip": "Smooth the Detail Daemon sigma curve"}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 128, "step": 1,
                                               "tooltip": "Feather/blend pixels at mask edges"}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                                "tooltip": "Expand mask by this many pixels"}),
                "target_width": ("INT", {"default": 800, "min": 64, "max": 4096, "step": 8,
                                          "tooltip": "Width to resize face crops to before sampling"}),
                "target_height": ("INT", {"default": 1200, "min": 64, "max": 4096, "step": 8,
                                           "tooltip": "Height to resize face crops to before sampling"}),
                **slot_widgets,
            },
            "optional": {
                "positive_base": ("CONDITIONING",),
                "dd_options": ("DD_OPTIONS",),
                "inpaint_options": ("INPAINT_OPTIONS",),
            },
        }

    def _get_slot_config(self, slot_key, inpaint_options):
        """Get per-slot config from InpaintOptions or defaults."""
        opts = inpaint_options or INPAINT_DEFAULTS
        slots = opts.get("slots", INPAINT_DEFAULTS["slots"])
        return slots.get(slot_key, INPAINT_DEFAULTS["slots"]["ref_1"])

    def _encode_prompt(self, clip, prompt_text):
        """Encode a text prompt using CLIP."""
        tokens = clip.tokenize(prompt_text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return [[cond, output]]

    def _apply_lora(self, model, clip, lora_name, strength):
        """Apply LoRA to model and clip, returning patched clones."""
        if lora_name == "None" or strength == 0:
            return model, clip
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        patched_model, patched_clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
        return patched_model, patched_clip

    def execute(self, images, person_data, model, clip, vae, negative,
                seed, steps, denoise, sampler_name, scheduler,
                detail_daemon_enabled, detail_amount, dd_smooth,
                mask_blend_pixels, mask_expand_pixels, target_width, target_height,
                ref_1_enabled, ref_1_lora, ref_1_prompt,
                ref_2_enabled, ref_2_lora, ref_2_prompt,
                ref_3_enabled, ref_3_lora, ref_3_prompt,
                ref_4_enabled, ref_4_lora, ref_4_prompt,
                ref_5_enabled, ref_5_lora, ref_5_prompt,
                generic_enabled, generic_lora, generic_prompt,
                positive_base=None, dd_options=None, inpaint_options=None):

        batch_size = images.shape[0]
        inpaint_opts = inpaint_options or INPAINT_DEFAULTS

        # Collect slot configs
        slots = []
        for i, (enabled, lora, prompt) in enumerate([
            (ref_1_enabled, ref_1_lora, ref_1_prompt),
            (ref_2_enabled, ref_2_lora, ref_2_prompt),
            (ref_3_enabled, ref_3_lora, ref_3_prompt),
            (ref_4_enabled, ref_4_lora, ref_4_prompt),
            (ref_5_enabled, ref_5_lora, ref_5_prompt),
        ], start=1):
            if not enabled:
                continue
            slot_key = f"ref_{i}"
            slot_cfg = self._get_slot_config(slot_key, inpaint_options)
            slots.append({
                "index": i - 1,  # 0-based index into person_data masks
                "lora": lora,
                "prompt": prompt,
                "mask_type": slot_cfg.get("mask_type", "face"),
                "lora_strength": slot_cfg.get("lora_strength", 1.0),
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
                    print(f"    Reference {ri+1} — no match, skip")
                    continue

                current_step += 1
                print(f"    [{current_step}/{total_steps}] Reference {ri+1} — detailing...")

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

                    generic_lora_strength = generic_cfg.get("lora_strength", 1.0)
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
            preview = refined_parts[-1]  # last refined part
        else:
            # No refinement happened — return input images
            output_refined = output_images
            preview = output_images[0:1]

        return {
            "ui": {"text": [f"Processed {batch_size} images, {len(refined_parts)} refinements"]},
            "result": (output_images, output_refined, preview),
        }
