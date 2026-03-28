"""Person Detailer with integrated Z-Image ControlNet Union (Pose + Depth).

Loads the Z-Image ControlNet Union model patch internally,
generates pose and/or depth maps from each person crop via DWPose/DepthAnythingV2,
and applies them as model patches during sampling.
"""

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
from .utils.controlnet_preprocess import (
    build_zimage_control_model,
    HAS_CONTROLNET_AUX, HAS_ZIMAGE_CONTROLNET,
)
from .person_detailer import INPAINT_DEFAULTS


def _get_model_patch_list():
    """Get list of Z-Image ControlNet files from model_patches folder."""
    try:
        all_patches = folder_paths.get_filename_list("model_patches")
        # Filter to Z-Image ControlNet files
        zimage_patches = [f for f in all_patches
                          if "controlnet" in f.lower() and ("z-image" in f.lower() or "zimage" in f.lower())]
        if not zimage_patches:
            return ["None"] + all_patches
        return ["None"] + zimage_patches
    except Exception:
        return ["None"]


class PersonDetailerControlNet:
    """Person detailing with integrated Z-Image ControlNet Union.

    Loads the ControlNet Union model patch internally and generates
    pose/depth control maps from each person crop automatically.
    When model_patch is 'None', behaves identically to PersonDetailer."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "refined", "refined_references", "refined_generic")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Per-person face/body detailing with integrated Z-Image ControlNet Union.\n\n"
        "Automatically loads the ControlNet Union model patch and generates\n"
        "DWPose and/or DepthAnythingV2 maps from each person crop.\n\n"
        "Select a model patch from models/model_patches/ — the Union model\n"
        "handles depth, pose, canny, and inpainting through one model.\n\n"
        "When model_patch is 'None', behaves identically to Person Detailer.\n\n"
        "All features from Person Detailer are available: per-slot LoRA, prompt,\n"
        "Detail Daemon, multi-round latent cycling, Z-Image Turbo auto-conversion."
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

        slot_widgets["generic_enabled"] = ("BOOLEAN", {"default": False,
                                                        "tooltip": "Enable detailing for unmatched faces"})
        slot_widgets["generic_catch_unprocessed"] = ("BOOLEAN", {"default": True,
                                                                   "tooltip": "ON: detail all faces not processed by active slots. OFF: only truly unmatched faces."})
        slot_widgets["generic_lora"] = (lora_list, {"tooltip": "LoRA for unmatched faces"})
        slot_widgets["generic_lora_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                                            "tooltip": "LoRA strength for unmatched faces"})
        slot_widgets["generic_prompt"] = ("STRING", {"multiline": True, "default": "",
                                                      "tooltip": "Positive prompt for unmatched faces."})

        return {
            "required": {
                # ── Inputs ──
                "images": ("IMAGE", {"tooltip": "Input image batch [B, H, W, C]"}),
                "person_data": ("PERSON_DATA", {"tooltip": "Person data from Person Selector Multi node"}),
                "model": ("MODEL", {"tooltip": "Base model (Z-Image Turbo or other)"}),
                "clip": ("CLIP", {"tooltip": "CLIP model for prompt encoding"}),
                "vae": ("VAE", {"tooltip": "VAE for encode/decode (also used to encode control images for the model patch)"}),
                # ── Sampler ──
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "denoise": ("FLOAT", {"default": 0.52, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                # ── Detail Daemon ──
                "detail_daemon_enabled": ("BOOLEAN", {"default": True}),
                "detail_amount": ("FLOAT", {"default": 0.20, "min": -5.0, "max": 5.0, "step": 0.01}),
                "dd_smooth": ("BOOLEAN", {"default": True}),
                # ── Inpaint ──
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 128, "step": 1}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "target_width": ("INT", {"default": 800, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 1200, "min": 64, "max": 4096, "step": 8}),
                # ── ControlNet Union ──
                "controlnet_enabled": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Enable/disable ControlNet guidance. When off, behaves like Person Detailer."}),
                "model_patch": (_get_model_patch_list(),
                                {"tooltip": "Z-Image ControlNet Union model patch from models/model_patches/."}),
                "control_type": (["depth", "pose", "depth+pose"],
                                 {"default": "depth",
                                  "tooltip": "Which control signal to generate from each crop:\n"
                                             "  depth — structural depth guidance (DepthAnythingV2)\n"
                                             "  pose — skeletal pose guidance (DWPose)\n"
                                             "  depth+pose — both applied as separate patches"}),
                "control_strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 2.0, "step": 0.05,
                                                "tooltip": "ControlNet Union strength. 0 = disabled."}),
                "cn_resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64,
                                           "tooltip": "Resolution for DWPose/DepthAnything preprocessors"}),
                "depth_model": (["depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth",
                                 "depth_anything_v2_vitg.pth", "depth_anything_v2_vits.pth"],
                                {"default": "depth_anything_v2_vitl.pth",
                                 "tooltip": "DepthAnythingV2 checkpoint (vitl=balanced, vitg=best, vits=fastest)"}),
                # ── References ──
                **slot_widgets,
            },
            "optional": {
                "positive_base": ("CONDITIONING", {"tooltip": "Base positive conditioning"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning"}),
                "dd_options": ("DD_OPTIONS", {"tooltip": "Advanced Detail Daemon parameters"}),
                "inpaint_options": ("INPAINT_OPTIONS", {"tooltip": "Advanced inpaint settings"}),
            },
        }

    # ── Helper methods (shared with PersonDetailer) ──────────────────────────

    def _get_slot_config(self, slot_key, inpaint_options):
        opts = inpaint_options or INPAINT_DEFAULTS
        slots = opts.get("slots", INPAINT_DEFAULTS["slots"])
        return slots.get(slot_key, INPAINT_DEFAULTS["slots"]["reference_1"])

    def _encode_prompt(self, clip, prompt_text):
        tokens = clip.tokenize(prompt_text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return [[cond, output]]

    def _apply_lora(self, model, clip, lora_name, strength):
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

        if is_z_image_turbo(model) and needs_qkv_conversion(lora):
            print(f"[FVMTools] Auto-converting LoRA '{lora_name}' for Z-Image Turbo QKV format")
            lora = convert_qkv_lora(lora)

        patched_model, patched_clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
        return patched_model, patched_clip

    def _prepare_conditioning(self, clip, patched_clip, prompt, positive_base):
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
                      inpaint_opts, controlnet_apply_fn=None):
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
            controlnet_apply_fn=controlnet_apply_fn,
        )
        return stitched, refined

    @staticmethod
    def _build_preview_text(summary, batch_size, num_refinements, elapsed_s=0, cn_info=""):
        time_suffix = f" | {elapsed_s}s" if elapsed_s > 0 else ""
        cn_tag = f" [{cn_info}]" if cn_info else ""
        if not summary:
            return f"{batch_size} img, {num_refinements} refined{cn_tag}{time_suffix}"

        parts = []
        for label, info in summary.items():
            status = info.get("status", "?")
            mask_type = info.get("mask_type", "?")
            if status == "ok":
                if mask_type == "aux":
                    parts.append(f"{label}: aux({info.get('parts', 0)})")
                elif "faces" in info:
                    parts.append(f"{label}: {info['faces']}x {mask_type}")
                else:
                    parts.append(f"{label}: {mask_type}")
            elif status == "no match":
                parts.append(f"{label}: skip")
            elif status == "no parts":
                parts.append(f"{label}: aux(0)")
            elif status == "no ref":
                parts.append(f"{label}: no ref")
            elif status == "empty":
                parts.append(f"{label}: 0 faces")

        text = " | ".join(parts) if parts else f"{batch_size} img, {num_refinements} refined"
        return f"{text}{cn_tag}{time_suffix}"

    # ── Main execution ───────────────────────────────────────────────────────

    def execute(self, images, person_data, model, clip, vae,
                seed, steps, denoise, sampler_name, scheduler,
                detail_daemon_enabled, detail_amount, dd_smooth,
                mask_blend_pixels, mask_expand_pixels, target_width, target_height,
                controlnet_enabled, model_patch, control_type, control_strength, cn_resolution, depth_model,
                reference_1_enabled, reference_1_lora, reference_1_lora_strength, reference_1_prompt,
                reference_2_enabled, reference_2_lora, reference_2_lora_strength, reference_2_prompt,
                reference_3_enabled, reference_3_lora, reference_3_lora_strength, reference_3_prompt,
                reference_4_enabled, reference_4_lora, reference_4_lora_strength, reference_4_prompt,
                reference_5_enabled, reference_5_lora, reference_5_lora_strength, reference_5_prompt,
                generic_enabled, generic_catch_unprocessed, generic_lora, generic_lora_strength, generic_prompt,
                positive_base=None, negative=None, dd_options=None, inpaint_options=None):

        import time as _time
        _t0 = _time.monotonic()

        batch_size = images.shape[0]
        inpaint_opts = inpaint_options or INPAINT_DEFAULTS
        has_aux = "aux_masks" in person_data

        if negative is None:
            negative = self._encode_prompt(clip, "")

        # ── Build ControlNet apply function ──────────────────────────────────
        has_cn = controlnet_enabled and model_patch != "None" and control_strength > 0

        if has_cn and not HAS_CONTROLNET_AUX:
            print("[FVMTools] WARNING: comfyui_controlnet_aux not installed — ControlNet disabled")
            has_cn = False
        if has_cn and not HAS_ZIMAGE_CONTROLNET:
            print("[FVMTools] WARNING: Z-Image ControlNet not available in this ComfyUI version — disabled")
            has_cn = False

        if has_cn:
            def controlnet_apply_fn(base_model, pos, neg, crop_image):
                patched = build_zimage_control_model(
                    base_model, vae, model_patch, crop_image,
                    control_type=control_type,
                    strength=control_strength,
                    cn_resolution=cn_resolution,
                    depth_ckpt=depth_model,
                )
                return patched, pos, neg
        else:
            controlnet_apply_fn = None

        # ── Collect slot configs ─────────────────────────────────────────────
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
                "index": i - 1,
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

        cn_info = ""
        if has_cn:
            cn_info = f"{control_type} s={control_strength:.2f}"

        print(f"\n{'='*60}")
        print(f"  PersonDetailer ControlNet")
        print(f"  Batch: {batch_size} images | Slots: {len(slots)} refs" +
              (f" + generic" if generic_enabled else "") +
              (f" | aux: yes" if has_aux else "") +
              (f" | CN: {cn_info}" if cn_info else " | CN: off"))
        if has_cn:
            print(f"  Model patch: {model_patch}")
        print(f"{'='*60}")

        results = []
        refined_parts = []
        refined_ref_parts = []
        refined_gen_parts = []
        all_summaries = []

        inpaint_kwargs = dict(
            model=model, clip=clip, positive_base=positive_base, negative=negative, vae=vae,
            seed=seed, steps=steps, denoise=denoise, sampler_name=sampler_name, scheduler=scheduler,
            detail_daemon_enabled=detail_daemon_enabled, detail_amount=detail_amount,
            dd_smooth=dd_smooth, dd_options=dd_options,
            mask_blend_pixels=mask_blend_pixels, mask_expand_pixels=mask_expand_pixels,
            target_width=target_width, target_height=target_height,
            inpaint_opts=inpaint_opts,
            controlnet_apply_fn=controlnet_apply_fn,
        )

        for b in range(batch_size):
            current_image = images[b]
            img_summary = {}

            print(f"\n  [Batch {b+1}/{batch_size}]")

            # Process reference slots
            num_refs = person_data["num_references"]
            for slot in slots:
                ri = slot["index"]
                mask_type = slot["mask_type"]

                if ri >= num_refs:
                    print(f"    {slot['label']} — no reference connected, skip")
                    img_summary[slot["label"]] = {"status": "no ref", "mask_type": mask_type}
                    continue

                if mask_type == "aux":
                    if not has_aux:
                        print(f"    {slot['label']} — aux: no detector connected, skip")
                        img_summary[slot["label"]] = {"status": "no aux data", "mask_type": "aux"}
                        continue

                    aux_mask = person_data["aux_masks"][ri][b]
                    if is_mask_empty(aux_mask):
                        print(f"    {slot['label']} — aux: no body parts assigned, skip")
                        img_summary[slot["label"]] = {"status": "no parts", "mask_type": "aux", "parts": 0}
                        continue

                    part_count = 0
                    if "aux_part_counts" in person_data and b < len(person_data["aux_part_counts"]):
                        part_count = person_data["aux_part_counts"][b].get(ri, 0)
                    print(f"    {slot['label']} — aux: {part_count} body part(s), detailing...")

                    stitched, refined = self._inpaint_mask(
                        current_image, aux_mask, slot, **inpaint_kwargs)
                    current_image = stitched
                    if refined is not None:
                        refined_parts.append(refined)
                        refined_ref_parts.append(refined)
                    img_summary[slot["label"]] = {"status": "ok", "mask_type": "aux", "parts": part_count}

                else:
                    mask_key = f"{mask_type}_masks"
                    mask = person_data[mask_key][ri][b]

                    if is_mask_empty(mask):
                        print(f"    {slot['label']} — no match, skip")
                        img_summary[slot["label"]] = {"status": "no match", "mask_type": mask_type}
                        continue

                    cn_tag = f" +{control_type}" if has_cn else ""
                    print(f"    {slot['label']} — {mask_type} detailing{cn_tag}...")

                    stitched, refined = self._inpaint_mask(
                        current_image, mask, slot, **inpaint_kwargs)
                    current_image = stitched
                    if refined is not None:
                        refined_parts.append(refined)
                        refined_ref_parts.append(refined)
                    img_summary[slot["label"]] = {"status": "ok", "mask_type": mask_type}

            # Generic slot
            if generic_enabled:
                if generic_mask_type == "aux":
                    if not has_aux:
                        print(f"    Generic — aux: no detector connected, skip")
                        img_summary["Generic"] = {"status": "no aux data", "mask_type": "aux"}
                    else:
                        unassigned_mask = person_data.get("aux_unassigned_masks")
                        if unassigned_mask is not None and not is_mask_empty(unassigned_mask[b]):
                            print(f"    Generic — aux: detailing unassigned body parts...")
                            generic_slot = {
                                "lora": generic_lora, "lora_strength": generic_lora_strength,
                                "prompt": generic_prompt, "use_dd": generic_cfg.get("detail_daemon", True),
                                "rounds": generic_cfg.get("rounds", 1),
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
                    per_face = person_data.get("per_face_masks", [])
                    face_to_ref = person_data.get("face_to_ref", [])

                    if per_face and b < len(per_face):
                        active_ref_indices = {slot["index"] for slot in slots}
                        generic_slot = {
                            "lora": generic_lora, "lora_strength": generic_lora_strength,
                            "prompt": generic_prompt, "use_dd": generic_cfg.get("detail_daemon", True),
                            "rounds": generic_cfg.get("rounds", 1),
                        }
                        face_count = 0
                        for fi, face_masks in enumerate(per_face[b]):
                            ref_idx = face_to_ref[b][fi] if b < len(face_to_ref) else None
                            if generic_catch_unprocessed:
                                if ref_idx is not None and ref_idx in active_ref_indices:
                                    continue
                            else:
                                if ref_idx is not None:
                                    continue
                            mask = face_masks.get(generic_mask_type, face_masks.get("head"))
                            if mask.dim() == 3:
                                mask = mask[0]
                            if is_mask_empty(mask):
                                continue
                            face_count += 1
                            print(f"    Generic — face #{fi+1} ({generic_mask_type}) detailing...")
                            stitched, refined = self._inpaint_mask(
                                current_image, mask, generic_slot, **inpaint_kwargs)
                            current_image = stitched
                            if refined is not None:
                                refined_parts.append(refined)
                                refined_gen_parts.append(refined)
                        if face_count > 0:
                            img_summary["Generic"] = {"status": "ok", "mask_type": generic_mask_type, "faces": face_count}
                        else:
                            print(f"    Generic — no unmatched faces")
                            img_summary["Generic"] = {"status": "empty", "mask_type": generic_mask_type, "faces": 0}
                    else:
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
                                "prompt": generic_prompt, "use_dd": generic_cfg.get("detail_daemon", True),
                                "rounds": generic_cfg.get("rounds", 1),
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

        _elapsed = int(_time.monotonic() - _t0)
        print(f"\n{'='*60}")
        print(f"  Done! {batch_size} images, {len(refined_parts)} refinements in {_elapsed}s." +
              (f" (CN: {cn_info})" if cn_info else ""))
        print(f"{'='*60}\n")

        preview_text = self._build_preview_text(
            all_summaries[0] if all_summaries else {},
            batch_size, len(refined_parts), _elapsed, cn_info=cn_info)

        output_images = torch.stack(results)
        _empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        output_refined = torch.cat(refined_parts, dim=0) if refined_parts else _empty
        output_refined_refs = torch.cat(refined_ref_parts, dim=0) if refined_ref_parts else _empty
        output_refined_gen = torch.cat(refined_gen_parts, dim=0) if refined_gen_parts else _empty

        return {
            "ui": {"text": [preview_text]},
            "result": (output_images, output_refined, output_refined_refs, output_refined_gen),
        }
