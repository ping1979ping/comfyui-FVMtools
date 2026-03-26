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
        "- mask_type: which mask to use per reference slot (head/face/body/aux)\n"
        "  'aux' uses body-part masks from detector connected to Person Selector Multi\n"
        "- repeat: number of sampling iterations per crop (latent cycling)\n"
        "- detail_daemon: enable/disable Detail Daemon per slot"
    )

    @classmethod
    def INPUT_TYPES(cls):
        mask_types = ["face", "head", "body", "hair", "facial_skin", "eyes", "mouth", "neck", "accessories", "aux"]

        widgets = {
            "mask_fill_holes": ("BOOLEAN", {"default": True,
                                             "tooltip": "Fill holes in masks before inpainting (closes gaps in segmentation)"}),
            "context_expand_factor": ("FLOAT", {"default": 1.20, "min": 1.0, "max": 3.0, "step": 0.05,
                                                 "tooltip": "Expand crop region beyond mask bbox by this factor (1.0 = tight crop)"}),
            "output_padding": ("INT", {"default": 32, "min": 0, "max": 128, "step": 8,
                                        "tooltip": "Additional padding around crop region in pixels"}),
        }

        for i in range(1, 6):
            prefix = f"reference_{i}"
            widgets[f"{prefix}_mask_type"] = (mask_types, {"default": "head", "tooltip": f"Mask type for reference slot {i}"})
            widgets[f"{prefix}_repeat"] = ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                                                    "tooltip": f"Sampling iterations on crop for reference {i} (latent cycling)"})
            widgets[f"{prefix}_detail_daemon"] = ("BOOLEAN", {"default": True, "tooltip": f"Use Detail Daemon for reference {i}"})

        widgets["generic_mask_type"] = (mask_types, {"default": "head", "tooltip": "Mask type for generic (unmatched) slot"})
        widgets["generic_repeat"] = ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                                              "tooltip": "Sampling iterations on crop for generic slot (latent cycling)"})
        widgets["generic_detail_daemon"] = ("BOOLEAN", {"default": True, "tooltip": "Use Detail Daemon for generic slot"})

        return {"required": widgets}

    def execute(self, **kwargs):
        slots = {}
        for prefix in ["reference_1", "reference_2", "reference_3", "reference_4", "reference_5", "generic"]:
            slots[prefix] = {
                "mask_type": kwargs.get(f"{prefix}_mask_type", "head"),
                "repeat": kwargs.get(f"{prefix}_repeat", 1),
                "detail_daemon": kwargs.get(f"{prefix}_detail_daemon", True),
            }

        return ({
            "mask_fill_holes": kwargs["mask_fill_holes"],
            "context_expand_factor": kwargs["context_expand_factor"],
            "output_padding": kwargs["output_padding"],
            "slots": slots,
        },)
