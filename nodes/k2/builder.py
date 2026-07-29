"""FVM K2 Lab — Region Builder: Boxen, Prompts und LoRAs in einer Node.

Der visuelle Editor lebt im Frontend (``web/js/fvm_k2_builder.js``), die
Layout-Logik in ``core/k2/layout.py``. Diese Datei ist nur die Node-Hülle.
"""

import json

from ...core.k2.layout import default_layout_json, parse_layout

CATEGORY = "FVM Tools/K2"


class FVM_K2_RegionBuilder:
    """Visueller Box-Editor: Regionen, Prompts und LoRAs in einer Node."""

    DESCRIPTION = (
        "Visual region editor for Krea 2 — boxes, per-box prompts and per-box LoRAs "
        "in a single node.\n\n"
        "Press 'Edit layout' to open a detached editor window: drag and resize boxes "
        "on the canvas, give each one a prompt and an identity prompt, and attach any "
        "number of LoRAs per box with their own strength and routing mode. "
        "'Backdrop' loads a recent render behind the canvas so boxes can be placed "
        "against a real composition.\n\n"
        "Boxes are stored NORMALIZED (0..1) and rescaled shape-preserving when the "
        "canvas changes: a box keeps its width-to-height ratio and its relative centre "
        "instead of being squashed. That is the part plain fraction-based editors get "
        "wrong — there a square box silently turns into a wide rectangle when you go "
        "from 1:1 to 16:9.\n\n"
        "Wire regions + loras + global_prompt + width + height straight into K2 Compose."
    )
    CATEGORY = CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("K2_REGION", "K2_LORA", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "regions", "loras", "global_prompt", "width", "height", "report",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                          "tooltip": "Output width. Changing it rescales the layout "
                          "shape-preserving instead of distorting the boxes."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                           "tooltip": "Output height. Same shape-preserving rescale "
                           "applies."}),
                "global_prompt": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "Scene-wide description. Keep the subjects "
                                  "out of it — describe them in their boxes, otherwise "
                                  "the model adds extra people next to the regions."}),
                "layout": ("STRING", {"multiline": True, "default": default_layout_json(),
                           "tooltip": "Layout JSON written by the editor window. The "
                           "frontend hides this widget; you can still edit it by hand "
                           "or paste a saved layout."}),
            },
        }

    def build(self, width, height, global_prompt, layout):
        regions, loras, notes = parse_layout(layout, int(width), int(height))
        report = json.dumps(
            {
                "regions": [
                    {
                        "id": region.region_id,
                        "name": region.name,
                        "role": region.role,
                        "priority": region.priority,
                        "box_pixels": [round(v, 1) for v in region.box.as_tuple()],
                        "has_prompt": bool(region.prompt.strip()),
                        "has_identity": bool(region.identity_prompt.strip()),
                        "loras": [
                            spec.lora_name for spec in loras
                            if region.region_id in spec.region_ids
                        ],
                    }
                    for region in regions
                ],
                "global_loras": [spec.lora_name for spec in loras if spec.global_scope],
                "canvas": [int(width), int(height)],
                "notes": notes,
            },
            indent=2,
        )
        return (regions, loras, global_prompt, int(width), int(height), report)


NODE_CLASS_MAPPINGS = {"FVM_K2_RegionBuilder": FVM_K2_RegionBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"FVM_K2_RegionBuilder": "K2 Region Builder"}
