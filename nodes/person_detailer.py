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
    "denoise_progression": "",
    "steps_progression": "",
    "slots": {
        prefix: {"mask_type": "head", "detail_daemon": True, "rounds": 1}
        for prefix in ["reference_1", "reference_2", "reference_3", "reference_4", "reference_5", "generic"]
    },
}


class PersonDetailer:
    """All-in-one face/body detailing node. Iterates over reference slots per batch image,
    applies per-slot LoRA and prompt, inpaints the masked region, and stitches back.
    Unmatched faces can be handled by the generic slot via connected components.

    When mask_type is set to 'aux', uses SEGS from connected detector or pre-computed
    SEGS input for body-part detailing instead of face/head masks."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "refined", "refined_references", "refined_generic")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Per-person face/body detailing with individual LoRA and prompt per reference slot.\n\n"
        "Connect PERSON_DATA from Person Selector Multi to assign faces to slots.\n"
        "Each enabled slot inpaints its matched region using the slot's LoRA and prompt.\n"
        "The generic slot handles unmatched faces via connected components.\n\n"
        "Mask type per slot (via Inpaint Options):\n"
        "- face/head/body/hair/etc: use pre-computed masks from Person Selector Multi\n"
        "- aux: use body-part SEGS from detector (connect SEGS or detector input)\n\n"
        "For both head + body-part detailing, chain two PersonDetailer nodes.\n\n"
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
        slot_widgets["generic_catch_unprocessed"] = ("BOOLEAN", {"default": True,
                                                                   "tooltip": "ON: detail all faces not processed by active slots (including matched but disabled slots). OFF: only truly unmatched faces from Person Selector Multi."})
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
        if not isinstance(lora_name, str):
            lora_list = ["None"] + folder_paths.get_filename_list("loras")
            if isinstance(lora_name, int) and 0 <= lora_name < len(lora_list):
                lora_name = lora_list[lora_name]
            else:
                print(f"[FVMTools] Warning: invalid lora value {lora_name!r}, skipping")
                return model, clip
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

    def _prepare_conditioning(self, clip, patched_clip, prompt, positive_base):
        """Get conditioning from prompt, falling back to positive_base or empty."""
        if prompt.strip():
            return self._encode_prompt(patched_clip, prompt)
        elif positive_base is not None:
            return positive_base
        else:
            return self._encode_prompt(patched_clip, "")

    def _inpaint_mask(self, current_image, mask, slot, model, clip, positive_base, negative, vae,
                      seed, steps, denoise, sampler_name, scheduler,
                      detail_daemon_enabled, detail_amount, dd_smooth, dd_options,
                      mask_blend_pixels, mask_expand_pixels, target_width, target_height,
                      inpaint_opts):
        """Inpaint a single mask region. Returns (stitched_image, refined_crop_or_None)."""
        patched_model, patched_clip = self._apply_lora(
            model, clip, slot["lora"], slot["lora_strength"]
        )
        positive_cond = self._prepare_conditioning(clip, patched_clip, slot["prompt"], positive_base)
        slot_dd_enabled = detail_daemon_enabled and slot["use_dd"]

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
            repeat=slot.get("rounds", 1),
            denoise_progression=inpaint_opts.get("denoise_progression", ""),
            steps_progression=inpaint_opts.get("steps_progression", ""),
        )
        return stitched, refined

    def execute(self, images, person_data, model, clip, vae,
                seed, steps, denoise, sampler_name, scheduler,
                detail_daemon_enabled, detail_amount, dd_smooth,
                mask_blend_pixels, mask_expand_pixels, target_width, target_height,
                reference_1_enabled, reference_1_lora, reference_1_lora_strength, reference_1_prompt,
                reference_2_enabled, reference_2_lora, reference_2_lora_strength, reference_2_prompt,
                reference_3_enabled, reference_3_lora, reference_3_lora_strength, reference_3_prompt,
                reference_4_enabled, reference_4_lora, reference_4_lora_strength, reference_4_prompt,
                reference_5_enabled, reference_5_lora, reference_5_lora_strength, reference_5_prompt,
                generic_enabled, generic_catch_unprocessed, generic_lora, generic_lora_strength, generic_prompt,
                positive_base=None, negative=None, dd_options=None, inpaint_options=None):

        batch_size = images.shape[0]
        inpaint_opts = inpaint_options or INPAINT_DEFAULTS
        has_aux = "aux_masks" in person_data

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
                "label": f"Ref{i}",
                "lora": lora,
                "prompt": prompt,
                "mask_type": slot_cfg.get("mask_type", "head"),
                "lora_strength": lora_str,
                "use_dd": slot_cfg.get("detail_daemon", True),
                "rounds": slot_cfg.get("rounds", 1),
            })

        generic_cfg = self._get_slot_config("generic", inpaint_options)
        generic_mask_type = generic_cfg.get("mask_type", "head")

        print(f"\n{'='*50}")
        print(f"  PersonDetailer v2.0")
        print(f"  Batch: {batch_size} images | Slots: {len(slots)} refs" +
              (f" + generic" if generic_enabled else "") +
              (f" | aux data: yes" if has_aux else ""))
        print(f"{'='*50}")

        results = []
        refined_parts = []
        refined_ref_parts = []
        refined_gen_parts = []
        all_summaries = []  # Per-batch-image summaries for preview text

        # Common inpaint kwargs
        inpaint_kwargs = dict(
            model=model, clip=clip, positive_base=positive_base, negative=negative, vae=vae,
            seed=seed, steps=steps, denoise=denoise, sampler_name=sampler_name, scheduler=scheduler,
            detail_daemon_enabled=detail_daemon_enabled, detail_amount=detail_amount,
            dd_smooth=dd_smooth, dd_options=dd_options,
            mask_blend_pixels=mask_blend_pixels, mask_expand_pixels=mask_expand_pixels,
            target_width=target_width, target_height=target_height,
            inpaint_opts=inpaint_opts,
        )

        for b in range(batch_size):
            current_image = images[b]  # [H, W, C]
            img_summary = {}

            print(f"\n  [Batch {b+1}/{batch_size}]")

            # Process reference slots
            num_refs = person_data["num_references"]
            for slot in slots:
                ri = slot["index"]
                mask_type = slot["mask_type"]

                # Skip if this slot index exceeds available references
                if ri >= num_refs:
                    print(f"    {slot['label']} — no reference connected, skip")
                    img_summary[slot["label"]] = {"status": "no ref", "mask_type": mask_type}
                    continue

                if mask_type == "aux":
                    # Body-part detailing from pre-computed aux_masks in PERSON_DATA
                    if not has_aux:
                        print(f"    {slot['label']} — aux: no detector connected to Person Selector Multi, skip")
                        img_summary[slot["label"]] = {"status": "no aux data", "mask_type": "aux"}
                        continue

                    aux_mask = person_data["aux_masks"][ri][b]  # [H, W]
                    if is_mask_empty(aux_mask):
                        print(f"    {slot['label']} — aux: no body parts assigned, skip")
                        img_summary[slot["label"]] = {"status": "no parts", "mask_type": "aux", "parts": 0}
                        continue

                    # Get part count from PERSON_DATA
                    part_count = 0
                    if "aux_part_counts" in person_data and b < len(person_data["aux_part_counts"]):
                        part_count = person_data["aux_part_counts"][b].get(ri, 0)
                    print(f"    {slot['label']} — aux: {part_count} body part(s), detailing merged mask...")

                    stitched, refined = self._inpaint_mask(
                        current_image, aux_mask, slot, **inpaint_kwargs)
                    current_image = stitched
                    if refined is not None:
                        refined_parts.append(refined)
                        refined_ref_parts.append(refined)

                    img_summary[slot["label"]] = {"status": "ok", "mask_type": "aux", "parts": part_count}

                else:
                    # Standard mask-based detailing (face/head/body/hair/etc.)
                    mask_key = f"{mask_type}_masks"
                    mask = person_data[mask_key][ri][b]  # [H, W]

                    if is_mask_empty(mask):
                        print(f"    {slot['label']} — no match, skip")
                        img_summary[slot["label"]] = {"status": "no match", "mask_type": mask_type}
                        continue

                    print(f"    {slot['label']} — {mask_type} detailing...")

                    stitched, refined = self._inpaint_mask(
                        current_image, mask, slot, **inpaint_kwargs)
                    current_image = stitched
                    if refined is not None:
                        refined_parts.append(refined)
                        refined_ref_parts.append(refined)

                    img_summary[slot["label"]] = {"status": "ok", "mask_type": mask_type}

            # Generic slot: unmatched/unprocessed faces or body parts
            if generic_enabled:
                if generic_mask_type == "aux":
                    # Generic aux: detail unassigned body parts from PERSON_DATA
                    if not has_aux:
                        print(f"    Generic — aux: no detector connected to Person Selector Multi, skip")
                        img_summary["Generic"] = {"status": "no aux data", "mask_type": "aux"}
                    else:
                        unassigned_mask = person_data.get("aux_unassigned_masks")
                        if unassigned_mask is not None and not is_mask_empty(unassigned_mask[b]):
                            print(f"    Generic — aux: detailing unassigned body parts...")
                            generic_slot = {
                                "lora": generic_lora, "lora_strength": generic_lora_strength,
                                "prompt": generic_prompt, "use_dd": generic_cfg.get("detail_daemon", True), "rounds": generic_cfg.get("rounds", 1),
                            }
                            stitched, refined = self._inpaint_mask(
                                current_image, unassigned_mask[b], generic_slot, **inpaint_kwargs)
                            current_image = stitched
                            if refined is not None:
                                refined_parts.append(refined)
                                refined_gen_parts.append(refined)
                            img_summary["Generic"] = {"status": "ok", "mask_type": "aux", "parts": 1}
                        else:
                            print(f"    Generic — aux: no unassigned body parts")
                            img_summary["Generic"] = {"status": "no parts", "mask_type": "aux", "parts": 0}
                else:
                    # Standard generic: unmatched faces
                    if generic_catch_unprocessed:
                        h, w = person_data["image_height"], person_data["image_width"]
                        processed_mask = torch.zeros(h, w, dtype=torch.float32)
                        for slot in slots:
                            ri = slot["index"]
                            if ri >= num_refs or slot["mask_type"] == "aux":
                                continue
                            mask_key = f"{slot['mask_type']}_masks"
                            slot_mask = person_data[mask_key][ri][b]
                            processed_mask = torch.max(processed_mask, slot_mask)
                        unmatched_mask = (person_data["all_faces_mask"][b] - processed_mask).clamp(0, 1)
                    else:
                        unmatched_mask = (person_data["all_faces_mask"][b] - person_data["matched_faces_mask"][b]).clamp(0, 1)

                    components = split_mask_to_components(unmatched_mask, min_area_fraction=0.001)

                    if components:
                        print(f"    Generic — {len(components)} unmatched face(s) ({generic_mask_type})")
                        generic_slot = {
                            "lora": generic_lora, "lora_strength": generic_lora_strength,
                            "prompt": generic_prompt, "use_dd": generic_cfg.get("detail_daemon", True), "rounds": generic_cfg.get("rounds", 1),
                        }
                        for comp_mask in components:
                            stitched, refined = self._inpaint_mask(
                                current_image, comp_mask, generic_slot, **inpaint_kwargs)
                            current_image = stitched
                            if refined is not None:
                                refined_parts.append(refined)
                                refined_gen_parts.append(refined)
                        img_summary["Generic"] = {"status": "ok", "mask_type": generic_mask_type, "faces": len(components)}
                    else:
                        print(f"    Generic — no unmatched faces")
                        img_summary["Generic"] = {"status": "empty", "mask_type": generic_mask_type, "faces": 0}

            results.append(current_image)
            all_summaries.append(img_summary)

        print(f"\n{'='*50}")
        print(f"  Done! {batch_size} images, {len(refined_parts)} refinements.")
        print(f"{'='*50}\n")

        # Build preview text from first batch image summary
        preview_text = self._build_preview_text(all_summaries[0] if all_summaries else {}, batch_size, len(refined_parts))

        # Stack results
        output_images = torch.stack(results)  # [B, H, W, C]

        _empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        if refined_parts:
            output_refined = torch.cat(refined_parts, dim=0)
        else:
            output_refined = _empty

        if refined_ref_parts:
            output_refined_refs = torch.cat(refined_ref_parts, dim=0)
        else:
            output_refined_refs = _empty

        if refined_gen_parts:
            output_refined_gen = torch.cat(refined_gen_parts, dim=0)
        else:
            output_refined_gen = _empty

        return {
            "ui": {"text": [preview_text]},
            "result": (output_images, output_refined, output_refined_refs, output_refined_gen),
        }

    @staticmethod
    def _build_preview_text(summary, batch_size, num_refinements):
        """Build a concise preview text from the per-image summary dict."""
        if not summary:
            return f"{batch_size} img, {num_refinements} refined"

        parts = []
        for label, info in summary.items():
            status = info.get("status", "?")
            mask_type = info.get("mask_type", "?")

            if status == "ok":
                if mask_type == "aux":
                    n_parts = info.get("parts", 0)
                    parts.append(f"{label}: aux({n_parts})")
                elif "faces" in info:
                    parts.append(f"{label}: {info['faces']}x {mask_type}")
                else:
                    parts.append(f"{label}: {mask_type}")
            elif status == "no SEGS":
                parts.append(f"{label}: aux(no input)")
            elif status == "no match":
                parts.append(f"{label}: skip")
            elif status == "no parts":
                parts.append(f"{label}: aux(0)")
            elif status == "no ref":
                parts.append(f"{label}: no ref")
            elif status == "empty":
                parts.append(f"{label}: 0 faces")

        return " | ".join(parts) if parts else f"{batch_size} img, {num_refinements} refined"
