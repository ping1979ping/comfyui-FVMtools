class InpaintOptions:
    """Optional configuration node for advanced inpaint parameters.
    Connect to PersonDetailer's inpaint_options input to override defaults."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("INPAINT_OPTIONS",)
    RETURN_NAMES = ("inpaint_options",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_fill_holes": ("BOOLEAN", {"default": True,
                                                 "tooltip": "Fill holes in masks before inpainting"}),
                "context_expand_factor": ("FLOAT", {"default": 1.20, "min": 1.0, "max": 3.0, "step": 0.05,
                                                     "tooltip": "How much to expand crop region beyond the mask bbox"}),
                "output_padding": ("INT", {"default": 32, "min": 0, "max": 128, "step": 8,
                                            "tooltip": "Additional padding around crop region (pixels)"}),
                "ref_1_mask_type": (["face", "head", "body"], {"default": "face", "tooltip": "Mask type for reference slot 1"}),
                "ref_2_mask_type": (["face", "head", "body"], {"default": "face", "tooltip": "Mask type for reference slot 2"}),
                "ref_3_mask_type": (["face", "head", "body"], {"default": "face", "tooltip": "Mask type for reference slot 3"}),
                "ref_4_mask_type": (["face", "head", "body"], {"default": "face", "tooltip": "Mask type for reference slot 4"}),
                "ref_5_mask_type": (["face", "head", "body"], {"default": "face", "tooltip": "Mask type for reference slot 5"}),
                "generic_mask_type": (["face", "head", "body"], {"default": "face", "tooltip": "Mask type for generic slot"}),
                "ref_1_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "ref_2_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "ref_3_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "ref_4_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "ref_5_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "generic_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "ref_1_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for slot 1"}),
                "ref_2_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for slot 2"}),
                "ref_3_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for slot 3"}),
                "ref_4_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for slot 4"}),
                "ref_5_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for slot 5"}),
                "generic_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for generic slot"}),
            }
        }

    def execute(self, **kwargs):
        # Build per-slot config
        slots = {}
        for prefix in ["ref_1", "ref_2", "ref_3", "ref_4", "ref_5", "generic"]:
            slots[prefix] = {
                "mask_type": kwargs.get(f"{prefix}_mask_type", "face"),
                "lora_strength": kwargs.get(f"{prefix}_lora_strength", 1.0),
                "detail_daemon": kwargs.get(f"{prefix}_detail_daemon", True),
            }

        return ({
            "mask_fill_holes": kwargs["mask_fill_holes"],
            "context_expand_factor": kwargs["context_expand_factor"],
            "output_padding": kwargs["output_padding"],
            "slots": slots,
        },)
