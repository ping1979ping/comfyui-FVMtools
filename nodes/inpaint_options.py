class InpaintOptions:
    """Optional configuration node for advanced inpaint parameters.
    Connect to PersonDetailer's inpaint_options input to override defaults."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("INPAINT_OPTIONS",)
    RETURN_NAMES = ("inpaint_options",)
    DESCRIPTION = (
        "Advanced inpaint settings and per-slot overrides for Person Detailer.\n\n"
        "Controls mask preprocessing, crop region expansion, and per-slot options:\n"
        "- mask_type: which mask to use (face/head/body) per reference slot\n"
        "- lora_strength: LoRA strength per slot\n"
        "- detail_daemon: enable/disable Detail Daemon per slot"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_fill_holes": ("BOOLEAN", {"default": True,
                                                 "tooltip": "Fill holes in masks before inpainting (closes gaps in segmentation)"}),
                "context_expand_factor": ("FLOAT", {"default": 1.20, "min": 1.0, "max": 3.0, "step": 0.05,
                                                     "tooltip": "Expand crop region beyond mask bbox by this factor (1.0 = tight crop)"}),
                "output_padding": ("INT", {"default": 32, "min": 0, "max": 128, "step": 8,
                                            "tooltip": "Additional padding around crop region in pixels"}),
                "reference_1_mask_type": (["face", "head", "body"], {"default": "head", "tooltip": "Mask type for reference slot 1"}),
                "reference_2_mask_type": (["face", "head", "body"], {"default": "head", "tooltip": "Mask type for reference slot 2"}),
                "reference_3_mask_type": (["face", "head", "body"], {"default": "head", "tooltip": "Mask type for reference slot 3"}),
                "reference_4_mask_type": (["face", "head", "body"], {"default": "head", "tooltip": "Mask type for reference slot 4"}),
                "reference_5_mask_type": (["face", "head", "body"], {"default": "head", "tooltip": "Mask type for reference slot 5"}),
                "generic_mask_type": (["face", "head", "body"], {"default": "head", "tooltip": "Mask type for generic (unmatched) slot"}),
                "reference_1_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for reference 1"}),
                "reference_2_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for reference 2"}),
                "reference_3_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for reference 3"}),
                "reference_4_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for reference 4"}),
                "reference_5_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for reference 5"}),
                "generic_detail_daemon": ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for generic slot"}),
            }
        }

    def execute(self, **kwargs):
        slots = {}
        for prefix in ["reference_1", "reference_2", "reference_3", "reference_4", "reference_5", "generic"]:
            slots[prefix] = {
                "mask_type": kwargs.get(f"{prefix}_mask_type", "head"),
                "detail_daemon": kwargs.get(f"{prefix}_detail_daemon", True),
            }

        return ({
            "mask_fill_holes": kwargs["mask_fill_holes"],
            "context_expand_factor": kwargs["context_expand_factor"],
            "output_padding": kwargs["output_padding"],
            "slots": slots,
        },)
