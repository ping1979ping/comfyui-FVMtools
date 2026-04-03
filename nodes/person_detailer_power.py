"""PersonDetailerPower — rgthree-style LoRA UI for PersonDetailer.

Same backend logic as PersonDetailer but with compact custom LoRA widgets
(toggle + dropdown + strength arrows) instead of standard combo/float widgets.
"""

import comfy.samplers
import folder_paths

from .person_detailer import PersonDetailer, INPAINT_DEFAULTS


# ── Flexible input type for dynamic LoRA widget kwargs ──────────────────

class _AnyType(str):
    """A string type that matches any other type in ComfyUI's validation."""
    def __ne__(self, __value: object) -> bool:
        return False

_any_type = _AnyType("*")


class _FlexibleOptionalInputType(dict):
    """Dict that claims to contain any key, returning a flexible type for unknowns.
    Allows custom JS widgets to pass dict values as kwargs to the Python backend."""

    def __init__(self, known=None):
        super().__init__()
        self._known = known or {}
        self.update(self._known)

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        if key in self._known:
            return self._known[key]
        return (_any_type,)


class PersonDetailerPower(PersonDetailer):
    """Person Detailer with rgthree-style compact LoRA widgets.

    Same functionality as PersonDetailer — per-person face/body detailing with
    individual LoRA and prompt per reference slot. The difference is purely UI:
    compact toggle+dropdown+strength widgets instead of checkbox+combo+slider."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "refined", "refined_references", "refined_generic")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Per-person face/body detailing with rgthree-style LoRA widgets.\n\n"
        "Same as Person Detailer but with compact toggle/dropdown/strength controls.\n"
        "Right-click a LoRA widget for Show Info, Toggle, Move, Remove."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image batch [B, H, W, C]"}),
                "person_data": ("PERSON_DATA", {"tooltip": "Person data from Person Selector Multi node"}),
                "model": ("MODEL", {"tooltip": "Base model (LoRAs are applied as temporary clones per slot)"}),
                "clip": ("CLIP", {"tooltip": "CLIP model for prompt encoding"}),
                "vae": ("VAE", {"tooltip": "VAE for encode/decode in the inpaint pipeline"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "denoise": ("FLOAT", {"default": 0.52, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "detail_daemon_enabled": ("BOOLEAN", {"default": True}),
                "detail_amount": ("FLOAT", {"default": 0.20, "min": -5.0, "max": 5.0, "step": 0.01}),
                "dd_smooth": ("BOOLEAN", {"default": True}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 128, "step": 1}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "target_width": ("INT", {"default": 800, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 1200, "min": 64, "max": 4096, "step": 8}),
                # Prompts as standard multiline STRING widgets
                "ref_prompt_1": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Positive prompt for reference 1"}),
                "ref_prompt_2": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Positive prompt for reference 2"}),
                "ref_prompt_3": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Positive prompt for reference 3"}),
                "ref_prompt_4": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Positive prompt for reference 4"}),
                "ref_prompt_5": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Positive prompt for reference 5"}),
                "generic_catch_unprocessed": ("BOOLEAN", {"default": True,
                                                           "tooltip": "ON: detail all unprocessed faces. OFF: only truly unmatched."}),
                "gen_prompt": ("STRING", {"multiline": True, "default": "",
                                          "tooltip": "Positive prompt for generic/unmatched faces"}),
            },
            # LoRA widget values (ref_lora_1..5, gen_lora) come through here as dicts
            "optional": _FlexibleOptionalInputType(known={
                "positive_base": ("CONDITIONING", {"tooltip": "Base positive conditioning fallback"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning"}),
                "dd_options": ("DD_OPTIONS", {"tooltip": "Detail Daemon parameters"}),
                "inpaint_options": ("INPAINT_OPTIONS", {"tooltip": "Advanced inpaint settings"}),
            }),
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def _parse_lora_dict(self, data):
        """Parse a LoRA widget dict value into (enabled, lora_name, strength)."""
        if not isinstance(data, dict):
            return False, "None", 1.0
        return (
            data.get("on", False),
            data.get("lora", "None") or "None",
            data.get("strength", 1.0),
        )

    def execute(self, images, person_data, model, clip, vae,
                seed, steps, denoise, sampler_name, scheduler,
                detail_daemon_enabled, detail_amount, dd_smooth,
                mask_blend_pixels, mask_expand_pixels, target_width, target_height,
                ref_prompt_1="", ref_prompt_2="", ref_prompt_3="",
                ref_prompt_4="", ref_prompt_5="",
                generic_catch_unprocessed=True, gen_prompt="",
                positive_base=None, negative=None, dd_options=None,
                inpaint_options=None, unique_id=None, **kwargs):

        # Extract LoRA configs from custom widget kwargs
        ref_prompts = [ref_prompt_1, ref_prompt_2, ref_prompt_3, ref_prompt_4, ref_prompt_5]
        ref_args = []
        for i in range(1, 6):
            lora_data = kwargs.get(f"ref_lora_{i}", {"on": False, "lora": "None", "strength": 1.0})
            enabled, lora_name, strength = self._parse_lora_dict(lora_data)
            ref_args.extend([enabled, lora_name, strength, ref_prompts[i - 1]])

        gen_lora_data = kwargs.get("gen_lora", {"on": False, "lora": "None", "strength": 1.0})
        gen_enabled, gen_lora_name, gen_strength = self._parse_lora_dict(gen_lora_data)

        # Delegate to parent PersonDetailer.execute with reconstructed positional args
        return super().execute(
            images, person_data, model, clip, vae,
            seed, steps, denoise, sampler_name, scheduler,
            detail_daemon_enabled, detail_amount, dd_smooth,
            mask_blend_pixels, mask_expand_pixels, target_width, target_height,
            # 5 reference slots: enabled, lora, strength, prompt
            ref_args[0], ref_args[1], ref_args[2], ref_args[3],
            ref_args[4], ref_args[5], ref_args[6], ref_args[7],
            ref_args[8], ref_args[9], ref_args[10], ref_args[11],
            ref_args[12], ref_args[13], ref_args[14], ref_args[15],
            ref_args[16], ref_args[17], ref_args[18], ref_args[19],
            # Generic slot
            gen_enabled, generic_catch_unprocessed, gen_lora_name, gen_strength, gen_prompt,
            positive_base=positive_base, negative=negative,
            dd_options=dd_options, inpaint_options=inpaint_options,
            unique_id=unique_id,
        )
