"""FVM K2 Lab — regionale LoRA-Zuweisung."""

import json
from pathlib import Path

import folder_paths

from ...core.k2.lora import (
    CHARACTER_ROUTING,
    ROUTING_MODES,
    STANDARD_ROUTING,
    LoraSpec,
    inspect_lora,
)

CATEGORY = "FVM Tools/K2"


class FVM_K2_LoRA:
    """Bindet eine LoRA global oder an eine Region."""

    DESCRIPTION = (
        "Assigns one Krea 2 LoRA either globally or to a union of named regions.\n\n"
        "Regional LoRAs are applied UNFUSED: the delta is computed in the forward "
        "pass and gated per token, so it only touches the text tokens of the assigned "
        "clause and the image tokens inside the assigned boxes. Base weights are never "
        "rewritten, which also makes this work on FP8/INT8 checkpoints.\n\n"
        "routing 'character_identity' additionally injects an identity anchor sentence "
        "using the trigger phrase — use it for face/person LoRAs.\n\n"
        "Chain multiple LoRA nodes through the 'loras' input."
    )
    CATEGORY = CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("K2_LORA",)
    RETURN_NAMES = ("loras",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),
                              {"tooltip": "Krea 2 LoRA. Non-Krea architectures are "
                               "rejected at compose time instead of silently doing "
                               "nothing."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                                       "tooltip": "Delta multiplier. 0 disables the "
                                       "assignment, negative inverts the learned delta."}),
                "global_scope": ("BOOLEAN", {"default": False,
                                 "tooltip": "On: applies to the whole image (fused, fast).\n"
                                 "Off: strict regional routing using the regions below."}),
                "regions": ("STRING", {"default": "",
                            "tooltip": "Comma separated region names (or ids) that receive "
                            "this LoRA. Only used when global_scope is off."}),
                "routing": (list(ROUTING_MODES), {"default": STANDARD_ROUTING,
                            "tooltip": "standard: gate text-fusion and local main-stream "
                            "deltas to the assigned boxes.\n"
                            "character_identity: same isolation plus an explicit identity "
                            "anchor built from the trigger phrase."}),
                "trigger_phrase": ("STRING", {"default": "",
                                   "tooltip": "Activation phrase learned during LoRA "
                                   "training. Required for character_identity."}),
            },
            "optional": {
                "loras": ("K2_LORA", {"tooltip": "Chain input: previously assigned LoRAs."}),
                "display_name": ("STRING", {"default": "",
                                 "tooltip": "Optional label used in reports."}),
            },
        }

    def build(self, lora_name, strength, global_scope, regions, routing,
              trigger_phrase, loras=None, display_name=""):
        collected = list(loras or [])
        region_ids = tuple(
            part.strip() for part in regions.split(",") if part.strip()
        )
        spec = LoraSpec(
            lora_id=f"lora-{len(collected) + 1}",
            lora_name=lora_name,
            strength=float(strength),
            global_scope=bool(global_scope),
            region_ids=region_ids,
            routing_mode=routing,
            trigger_phrase=trigger_phrase.strip(),
            display_name=display_name.strip() or Path(lora_name).stem,
        )
        collected.append(spec)
        return (collected,)


class FVM_K2_LoRAInfo:
    """Diagnose: passt diese LoRA überhaupt auf Krea 2?"""

    DESCRIPTION = (
        "Reads a LoRA header and reports adapter count, rank, adapter type "
        "(lora/lokr), key namespace and training metadata — without loading tensors.\n\n"
        "Use it when a LoRA seems to do nothing: a namespace other than "
        "'diffusion_model'/'blocks'/'txtfusion' means it was trained for a different "
        "architecture."
    )
    CATEGORY = CATEGORY
    FUNCTION = "inspect"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("report", "looks_like_krea2")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),
                              {"tooltip": "LoRA file to analyse. Nothing is loaded onto "
                               "the GPU — only the safetensors header is read."}),
            }
        }

    def inspect(self, lora_name):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        report = inspect_lora(path)
        namespaces = set(report.get("namespaces") or {})
        krea_like = bool(
            namespaces & {"diffusion_model", "blocks", "txtfusion", "transformer"}
        )
        text = json.dumps(report, indent=2, default=str)
        print(f"[FVM K2] LoRA info {lora_name}:\n{text}")
        return (text, krea_like)


NODE_CLASS_MAPPINGS = {
    "FVM_K2_LoRA": FVM_K2_LoRA,
    "FVM_K2_LoRAInfo": FVM_K2_LoRAInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_K2_LoRA": "K2 Regional LoRA",
    "FVM_K2_LoRAInfo": "K2 LoRA Info",
}
