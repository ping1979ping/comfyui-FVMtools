"""FVM K2 Lab — Kontrolle über den ``txtfusion.projector``."""

import json

import folder_paths

from ...core.k2.projector import (
    CUSTOM_PRESET,
    PROJECTOR_PRESET_NAMES,
    parse_values,
    preset_values,
    projector_delta_from_lora,
    scaled_values,
)

CATEGORY = "FVM Tools/K2"


class FVM_K2_Projector:
    """Delta auf Krea 2s 12-Werte-Layer-Mischung."""

    DESCRIPTION = (
        "Applies a delta to Krea 2's 'txtfusion.projector' — the Linear(12 -> 1) that "
        "mixes the twelve Qwen3-VL hidden-state taps into the text vector.\n\n"
        "This is the single strongest semantic lever in Krea 2: shifting the layer mix "
        "changes which language level dominates the image. It is the mechanism behind "
        "the well known projector LoRAs.\n\n"
        "Base weights are never overwritten. With identity_protection > 0 the delta is "
        "skipped on identity prompt tokens, so a strong style shift does not deform "
        "faces.\n\n"
        "The built-in presets mirror the published K2Lab reference table. For exact "
        "values use 'K2 Projector from LoRA' instead."
    )
    CATEGORY = CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("K2_PROJECTOR", "STRING")
    RETURN_NAMES = ("projector", "values")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(PROJECTOR_PRESET_NAMES), {"default": "filter_bypass2",
                           "tooltip": "Built-in vector from the published K2Lab reference "
                           "table. 'none' is all zeros, 'custom' reads custom_values."}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0, "step": 0.05,
                               "tooltip": "Signed scale on the whole vector. 0 = no effect, "
                               "negative reverses it."}),
                "identity_protection": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                        "step": 0.05,
                                        "tooltip": "1.0 keeps identity prompt tokens on the "
                                        "unmodified baseline mix; 0 applies the delta "
                                        "everywhere."}),
                "enabled": ("BOOLEAN", {"default": True,
                            "tooltip": "Off passes the model through untouched."}),
            },
            "optional": {
                "custom_values": ("STRING", {"default": "",
                                  "multiline": True,
                                  "tooltip": "Twelve numbers (comma or space separated). "
                                  "Only used with preset 'custom'."}),
            },
        }

    def build(self, preset, multiplier, identity_protection, enabled, custom_values=""):
        if preset == CUSTOM_PRESET:
            if not custom_values.strip():
                raise ValueError("Preset 'custom' needs twelve values in custom_values")
            base = parse_values(custom_values)
        else:
            base = preset_values(preset)
        effective = scaled_values(base, multiplier)
        payload = {
            "enabled": bool(enabled),
            "preset": preset,
            "values": list(base),
            "effective": list(effective),
            "identity_protection": float(identity_protection),
            "source": "preset",
        }
        return (payload, json.dumps([round(v, 4) for v in effective]))


class FVM_K2_ProjectorFromLoRA:
    """Liest das exakte Projector-Delta aus einer LoRA-Datei."""

    DESCRIPTION = (
        "Extracts the exact twelve projector values from a projector LoRA instead of "
        "using an approximated preset.\n\n"
        "Supports both published formats: a direct "
        "'diffusion_model.txtfusion.projector.diff' tensor and rank-1 "
        "'…projector.lora_A/lora_B' adapters.\n\n"
        "Files like fedor_bypass, skc3vo or z0jglf are exactly this kind of LoRA — "
        "loading them here is more accurate than the preset table and keeps the rest "
        "of the graph free of an extra LoRA loader."
    )
    CATEGORY = CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("K2_PROJECTOR", "STRING")
    RETURN_NAMES = ("projector", "values")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),
                              {"tooltip": "A projector LoRA (usually only 1-2 tensors)."}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0, "step": 0.05,
                               "tooltip": "Signed scale on the extracted vector — the "
                               "equivalent of LoRA strength for this delta."}),
                "identity_protection": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                        "step": 0.05,
                                        "tooltip": "Keeps identity prompt tokens on the "
                                        "unmodified baseline layer mix."}),
                "enabled": ("BOOLEAN", {"default": True,
                            "tooltip": "Off passes the model through untouched."}),
            },
        }

    def build(self, lora_name, multiplier, identity_protection, enabled):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        base = projector_delta_from_lora(path)
        effective = scaled_values(base, multiplier)
        payload = {
            "enabled": bool(enabled),
            "preset": f"lora:{lora_name}",
            "values": list(base),
            "effective": list(effective),
            "identity_protection": float(identity_protection),
            "source": "lora",
        }
        print(
            f"[FVM K2] Projector delta from {lora_name}: "
            + ", ".join(f"{v:+.4f}" for v in base)
        )
        return (payload, json.dumps([round(v, 4) for v in effective]))


NODE_CLASS_MAPPINGS = {
    "FVM_K2_Projector": FVM_K2_Projector,
    "FVM_K2_ProjectorFromLoRA": FVM_K2_ProjectorFromLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_K2_Projector": "K2 Projector",
    "FVM_K2_ProjectorFromLoRA": "K2 Projector from LoRA",
}
