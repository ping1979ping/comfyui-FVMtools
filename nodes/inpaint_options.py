class InpaintOptions:
    """Optional configuration node for advanced inpaint parameters.
    Connect to PersonDetailer's inpaint_options input to override defaults."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("INPAINT_OPTIONS",)
    RETURN_NAMES = ("inpaint_options",)
    DESCRIPTION = (
        "Advanced inpaint settings and per-slot overrides for Person Detailer.\n\n"
        "Controls mask preprocessing, crop region expansion, CFG, and per-slot options:\n"
        "- cfg: Classifier-Free Guidance scale for sampling contrast\n"
        "- negative_prompt: additional negative prompt combined with the negative conditioning input\n"
        "- mask_type: which mask to use per reference slot (head/face/body/aux)\n"
        "  'aux' uses body-part masks from detector connected to Person Selector Multi\n"
        "- rounds: number of inpaint passes per crop (latent cycling)\n"
        "- denoise_progression / steps_progression: per-round overrides\n"
        "  Format: values separated by | (e.g. '0.5|0.3' for 2 rounds)\n"
        "  When rounds=1 or empty, PersonDetailer's global denoise/steps are used.\n"
        "- detail_daemon: enable/disable Detail Daemon per slot"
    )

    @classmethod
    def INPUT_TYPES(cls):
        mask_types = ["face", "head", "body", "hair", "facial_skin", "eyes", "mouth", "neck", "accessories", "aux"]

        widgets = {
            "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.5,
                               "tooltip": "Classifier-Free Guidance scale.\n\n"
                                          "Controls how strongly the sampler follows the prompt vs. generating freely.\n"
                                          "Higher values = more contrast and prompt adherence, but can oversaturate.\n\n"
                                          "Recommended values:\n"
                                          "  1.0 — no guidance (default, best for distilled/turbo models)\n"
                                          "  1.5-3.0 — light guidance, reduces paleness\n"
                                          "  4.0-7.0 — standard guidance for non-distilled models\n"
                                          "  8.0+ — strong guidance, risk of artifacts"}),
            "negative_prompt": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": "Additional negative prompt text for inpainting.\n\n"
                                                       "This is encoded and combined with the negative conditioning\n"
                                                       "input on the Person Detailer node. If no negative conditioning\n"
                                                       "is connected, this text alone is used as negative.\n\n"
                                                       "Useful for suppressing common inpaint artifacts:\n"
                                                       "  'blurry, washed out, pale, low contrast'\n"
                                                       "  'extra fingers, deformed hands, bad anatomy'\n\n"
                                                       "Leave empty to use only the connected negative conditioning."}),
            "blend_mode": (["auto", "gaussian", "poisson", "poisson_mixed"], {
                "default": "auto",
                "tooltip": "How the inpainted crop blends back into the original image.\n\n"
                           "auto/poisson — seamless cloning via Poisson solver, matches boundary colors\n"
                           "  automatically. Best for faces and skin. (recommended)\n\n"
                           "gaussian — feathered alpha blend with delta clamping. Faster but may\n"
                           "  show color discontinuities at high denoise values.\n\n"
                           "poisson_mixed — Poisson with mixed gradients, preserves more original\n"
                           "  texture at the boundary. Good for detailed backgrounds.",
            }),
            "denoise_gradient": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                "tooltip": "Center-to-edge denoise falloff during sampling.\n\n"
                           "Creates a gradient denoise mask: full denoise at the center of the\n"
                           "masked region, fading toward the edges.\n\n"
                           "0.0 = uniform denoise everywhere (current behavior)\n"
                           "0.3 = gentle gradient (edges stay closer to original)\n"
                           "0.5 = moderate gradient (recommended starting point)\n"
                           "1.0 = full gradient (edges almost untouched by sampling)\n\n"
                           "This uses ComfyUI's native denoise_mask which blends the sampled\n"
                           "result with the original at every step. Very effective at reducing\n"
                           "visible seams without post-processing.",
            }),
            "edge_refine": ("BOOLEAN", {"default": False,
                "tooltip": "Run a quick low-denoise refinement pass on the blend zone.\n\n"
                           "After the main inpaint is stitched back, this identifies the\n"
                           "transition area (where the feathered mask is partially transparent)\n"
                           "and runs a short 2-3 step harmonization pass at very low denoise.\n\n"
                           "Adds ~0.5s per crop but can significantly reduce visible seams.\n"
                           "Most effective when combined with gaussian blend_mode.",
            }),
            "edge_refine_denoise": ("FLOAT", {"default": 0.12, "min": 0.05, "max": 0.30, "step": 0.01,
                "tooltip": "Denoise strength for the edge refinement pass.\n\n"
                           "Keep low (0.10-0.15) to harmonize without changing content.\n"
                           "Higher values may alter the inpainted result.",
            }),
            "mask_fill_holes": ("BOOLEAN", {"default": True,
                                             "tooltip": "Fill holes in masks before inpainting (closes gaps in segmentation)"}),
            "context_expand_factor": ("FLOAT", {"default": 1.20, "min": 1.0, "max": 3.0, "step": 0.05,
                                                 "tooltip": "Expand crop region beyond mask bbox by this factor (1.0 = tight crop)"}),
            "output_padding": ("INT", {"default": 32, "min": 0, "max": 128, "step": 8,
                                        "tooltip": "Additional padding around crop region in pixels"}),
            "denoise_progression": ("STRING", {"default": "",
                                                "tooltip": "Global per-round denoise override for ALL slots (applied when rounds > 1).\n"
                                                           "Pipe-separated values, one per round.\n\n"
                                                           "Examples:\n"
                                                           "  0.5|0.3     — round 1: denoise 0.5, round 2: denoise 0.3\n"
                                                           "  0.6|0.4|0.2 — 3 rounds with decreasing denoise\n"
                                                           "  0.5|0.2     — strong first pass, gentle refinement\n\n"
                                                           "If empty: PersonDetailer's global denoise is used for all rounds.\n"
                                                           "If fewer values than rounds: last value repeats.\n"
                                                           "If more values than rounds: extra values are ignored."}),
            "steps_progression": ("STRING", {"default": "",
                                              "tooltip": "Global per-round steps override for ALL slots (applied when rounds > 1).\n"
                                                         "Pipe-separated values, one per round.\n\n"
                                                         "Examples:\n"
                                                         "  6|4       — round 1: 6 steps, round 2: 4 steps\n"
                                                         "  8|6|4     — 3 rounds with decreasing steps\n"
                                                         "  10|4      — thorough first pass, quick refinement\n\n"
                                                         "If empty: PersonDetailer's global steps are used for all rounds.\n"
                                                         "If fewer values than rounds: last value repeats.\n"
                                                         "If more values than rounds: extra values are ignored."}),
        }

        for i in range(1, 6):
            prefix = f"reference_{i}"
            widgets[f"{prefix}_mask_type"] = (mask_types, {"default": "head",
                                                            "tooltip": f"Mask type for reference slot {i}.\n"
                                                                       f"face=skin only, head=face+hair, body=full body,\n"
                                                                       f"aux=body-part SEGS from detector"})
            widgets[f"{prefix}_rounds"] = ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                                                    "tooltip": f"Number of inpaint passes for reference {i}.\n"
                                                               f"Each round re-encodes the previous result (latent cycling).\n"
                                                               f"Use denoise_progression/steps_progression to control per-round values.\n\n"
                                                               f"1 = single pass (default), 2-3 = progressive refinement"})
            widgets[f"{prefix}_detail_daemon"] = ("BOOLEAN", {"default": True,
                                                               "tooltip": f"Enable Detail Daemon sigma manipulation for reference {i}.\n"
                                                                          f"Enhances detail preservation during inpainting."})

        widgets["generic_mask_type"] = (mask_types, {"default": "head",
                                                      "tooltip": "Mask type for generic (unmatched) slot.\n"
                                                                 "face=skin only, head=face+hair, body=full body,\n"
                                                                 "aux=body-part SEGS from detector"})
        widgets["generic_rounds"] = ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                                              "tooltip": "Number of inpaint passes for generic slot.\n"
                                                         "Each round re-encodes the previous result (latent cycling).\n"
                                                         "Use denoise_progression/steps_progression to control per-round values.\n\n"
                                                         "1 = single pass (default), 2-3 = progressive refinement"})
        widgets["generic_detail_daemon"] = ("BOOLEAN", {"default": True,
                                                         "tooltip": "Enable Detail Daemon sigma manipulation for generic slot.\n"
                                                                    "Enhances detail preservation during inpainting."})

        return {"required": widgets}

    def execute(self, **kwargs):
        slots = {}
        for prefix in ["reference_1", "reference_2", "reference_3", "reference_4", "reference_5", "generic"]:
            slots[prefix] = {
                "mask_type": kwargs.get(f"{prefix}_mask_type", "head"),
                "rounds": kwargs.get(f"{prefix}_rounds", 1),
                "detail_daemon": kwargs.get(f"{prefix}_detail_daemon", True),
            }

        return ({
            "cfg": kwargs.get("cfg", 1.0),
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "blend_mode": kwargs.get("blend_mode", "auto"),
            "denoise_gradient": kwargs.get("denoise_gradient", 0.0),
            "edge_refine": kwargs.get("edge_refine", False),
            "edge_refine_denoise": kwargs.get("edge_refine_denoise", 0.12),
            "mask_fill_holes": kwargs["mask_fill_holes"],
            "context_expand_factor": kwargs["context_expand_factor"],
            "output_padding": kwargs["output_padding"],
            "denoise_progression": kwargs.get("denoise_progression", ""),
            "steps_progression": kwargs.get("steps_progression", ""),
            "slots": slots,
        },)
