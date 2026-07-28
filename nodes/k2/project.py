"""FVM K2 Lab — Import/Export des K2Lab-Projektformats.

Damit lassen sich bestehende K2Lab-/K2-Region-Studio-Projekte (die JSON-Blobs aus
der Desktop-App bzw. aus dem ``region_config``-Widget) direkt in den Graphen
laden — und umgekehrt ein hier gebauter Aufbau als portables JSON sichern.
"""

import json

from ...core.k2.geometry import PixelBox
from ...core.k2.lora import ROUTING_MODES, STANDARD_ROUTING, LoraSpec
from ...core.k2.prompt import GLOBAL_SCOPE, ROLES, EmphasisRequest, RegionDefinition
from ...core.k2.projector import PROJECTOR_LENGTH, preset_values, scaled_values
from ...core.k2.runtime import K2Runtime
from .compose import DEFAULT_TUNING

CATEGORY = "FVM Tools/K2"

CONFIG_VERSION = 1

_DEFAULT_SPATIAL = {
    "enabled": True,
    "strength": 1.0,
    "outside_penalty": 1.0,
    "falloff_pixels": 128.0,
    "subject_competition": True,
    "subject_fill": True,
    "late_step_scale": 0.35,
    "lora_delta_adaptation": False,
    "lora_delta_adaptation_gain": 0.35,
    "strict_lora_isolation": True,
}
_DEFAULT_PROJECTOR = {
    "enabled": False,
    "preset": "filter_bypass2",
    "values": [0.0] * PROJECTOR_LENGTH,
    "multiplier": 1.0,
    "identity_protection": 1.0,
}


class FVM_K2_ProjectImport:
    """Lädt ein K2Lab-Projekt-JSON in Graph-Objekte."""

    DESCRIPTION = (
        "Imports a K2Lab / K2 Region Studio project JSON and turns it into normal K2 "
        "graph objects: regions, regional LoRAs, emphases, tuning and projector.\n\n"
        "Accepts the published schema (version 1) with 'regions', 'loras', 'emphases', "
        "'spatial', 'projector'. Region boxes may be given as {x0,y0,x1,y1} or "
        "{x,y,width,height}.\n\n"
        "LoRA entries are matched against the local models/loras inventory by file "
        "name; a missing file is reported instead of silently dropped."
    )
    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = (
        "K2_REGION", "K2_LORA", "K2_EMPHASIS", "K2_TUNING", "K2_PROJECTOR",
        "STRING", "STRING", "INT", "INT", "STRING",
    )
    RETURN_NAMES = (
        "regions", "loras", "emphasis", "tuning", "projector",
        "global_prompt", "negative_prompt", "width", "height", "info",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_json": ("STRING", {"multiline": True, "default": "{}",
                                 "tooltip": "K2Lab / K2 Region Studio project JSON."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                          "tooltip": "Canvas the stored pixel boxes refer to."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                           "tooltip": "Canvas height the stored boxes refer to."}),
            },
            "optional": {
                "strict_lora_names": ("BOOLEAN", {"default": True,
                                      "tooltip": "On: a LoRA file that is not installed "
                                      "raises an error. Off: it is skipped and listed in "
                                      "the info output."}),
            },
        }

    @staticmethod
    def _box(payload) -> PixelBox:
        if not isinstance(payload, dict):
            raise ValueError("region box must be a JSON object")
        if "x0" in payload:
            return PixelBox(
                float(payload["x0"]), float(payload["y0"]),
                float(payload["x1"]), float(payload["y1"]),
            )
        return PixelBox.from_xywh(
            float(payload.get("x", 0)), float(payload.get("y", 0)),
            float(payload.get("width", payload.get("w", 0))),
            float(payload.get("height", payload.get("h", 0))),
        )

    def load(self, project_json, width, height, strict_lora_names=True):
        import folder_paths

        try:
            data = json.loads(project_json or "{}")
        except json.JSONDecodeError as error:
            raise ValueError(f"Project JSON is invalid: {error}") from error
        if not isinstance(data, dict):
            raise ValueError("Project JSON must be an object")
        version = int(data.get("version", CONFIG_VERSION))
        if version > CONFIG_VERSION:
            raise ValueError(
                f"Project version {version} is newer than supported version "
                f"{CONFIG_VERSION}"
            )

        notes = []
        regions = []
        for index, item in enumerate(data.get("regions") or []):
            role = str(item.get("spatial_role", item.get("role", "auto")))
            if role not in ROLES:
                role = "auto"
            regions.append(
                RegionDefinition(
                    region_id=str(item.get("id", f"region-{index + 1}")),
                    name=str(item.get("name", f"Region {index + 1}")),
                    box=self._box(item.get("box", {})),
                    prompt=str(item.get("prompt", "")),
                    identity_prompt=str(item.get("face_identity_prompt",
                                                 item.get("identity_prompt", ""))),
                    negative_prompt=str(item.get("negative_prompt", "")),
                    enabled=bool(item.get("enabled", True)),
                    priority=int(item.get("priority", 100 - index)),
                    role=role,
                )
            )

        available = set(folder_paths.get_filename_list("loras"))
        by_basename = {name.replace("\\", "/").split("/")[-1]: name for name in available}
        loras = []
        for index, item in enumerate(data.get("loras") or []):
            raw_name = str(item.get("name") or item.get("lora_name") or "").strip()
            if not raw_name or raw_name == "None":
                continue
            resolved = (
                raw_name
                if raw_name in available
                else by_basename.get(raw_name.replace("\\", "/").split("/")[-1])
            )
            if resolved is None:
                message = f"LoRA not installed: {raw_name}"
                if strict_lora_names:
                    raise ValueError(message)
                notes.append(message)
                continue
            routing = str(item.get("routing_mode", STANDARD_ROUTING))
            if routing not in ROUTING_MODES:
                routing = STANDARD_ROUTING
            loras.append(
                LoraSpec(
                    lora_id=str(item.get("id", f"lora-{index + 1}")),
                    lora_name=resolved,
                    strength=float(item.get("strength", 1.0)),
                    global_scope=bool(item.get("global", True)),
                    region_ids=tuple(str(r) for r in item.get("region_ids", ())),
                    routing_mode=routing,
                    trigger_phrase=str(item.get("trigger_phrase", "")),
                    display_name=str(item.get("display_name", "")),
                )
            )

        emphases = [
            EmphasisRequest(
                scope_id=str(item.get("scope_id", GLOBAL_SCOPE)),
                phrase=str(item.get("phrase", "")),
                strength=float(item.get("strength", 0.5)),
                occurrence=int(item.get("occurrence", 0)),
            )
            for item in (data.get("emphases") or [])
            if str(item.get("phrase", "")).strip()
        ]

        spatial = {**_DEFAULT_SPATIAL, **(data.get("spatial") or {})}
        tuning = dict(DEFAULT_TUNING)
        tuning.update(
            {
                "spatial_enabled": bool(spatial["enabled"]),
                "strict_isolation": bool(spatial["strict_lora_isolation"]),
                "inside_strength": float(spatial["strength"]),
                "outside_penalty": float(spatial["outside_penalty"]),
                "falloff_pixels": float(spatial["falloff_pixels"]),
                "late_step_scale": float(spatial["late_step_scale"]),
                "subject_competition": bool(spatial["subject_competition"]),
                "subject_fill": bool(spatial["subject_fill"]),
                "lora_delta_adaptation": bool(spatial["lora_delta_adaptation"]),
                "lora_adaptation_gain": float(spatial["lora_delta_adaptation_gain"]),
            }
        )

        projector_payload = {**_DEFAULT_PROJECTOR, **(data.get("projector") or {})}
        preset = str(projector_payload.get("preset", "filter_bypass2"))
        if preset == "custom":
            base = tuple(float(v) for v in projector_payload.get("values", ()))
            if len(base) != PROJECTOR_LENGTH:
                raise ValueError("custom projector needs twelve values")
        else:
            base = preset_values(preset)
        projector = {
            "enabled": bool(projector_payload.get("enabled", False)),
            "preset": preset,
            "values": list(base),
            "effective": list(
                scaled_values(base, float(projector_payload.get("multiplier", 1.0)))
            ),
            "identity_protection": float(
                projector_payload.get("identity_protection", 1.0)
            ),
            "source": "project_json",
        }

        info = json.dumps(
            {
                "version": version,
                "regions": len(regions),
                "loras": len(loras),
                "emphases": len(emphases),
                "notes": notes,
            },
            indent=2,
        )
        return (
            regions, loras, emphases, tuning, projector,
            str(data.get("global_prompt", "")),
            str(data.get("global_negative", data.get("negative_prompt", ""))),
            int(width), int(height), info,
        )


class FVM_K2_ProjectExport:
    """Schreibt den aktuellen Aufbau als K2Lab-kompatibles JSON."""

    DESCRIPTION = (
        "Serializes the current setup into the K2Lab project schema.\n\n"
        "The result can be pasted back into K2Lab / K2 Region Studio, stored next to "
        "the output, or fed to K2 Project Import to reproduce the layout in another "
        "workflow."
    )
    CATEGORY = CATEGORY
    FUNCTION = "export"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("project_json",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                          "tooltip": "Canvas width stored with the project."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                           "tooltip": "Canvas height stored with the project."}),
            },
            "optional": {
                "regions": ("K2_REGION", {"tooltip": "Region chain to serialize."}),
                "loras": ("K2_LORA", {"tooltip": "LoRA assignments to serialize."}),
                "emphasis": ("K2_EMPHASIS", {"tooltip": "Emphases to serialize."}),
                "tuning": ("K2_TUNING", {"tooltip": "Spatial tuning to serialize."}),
                "projector": ("K2_PROJECTOR", {"tooltip": "Projector settings to serialize."}),
                "plan": ("K2_PLAN", {"tooltip": "Alternative source for the regions when no "
                         "region chain is connected."}),
                "global_prompt": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "Global prompt stored with the project."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Global negative stored with the project."}),
            },
        }

    def export(self, width, height, regions=None, loras=None, emphasis=None,
               tuning=None, projector=None, plan=None, global_prompt="",
               negative_prompt=""):
        if isinstance(plan, K2Runtime) and not regions:
            regions = [
                RegionDefinition(
                    region_id=r.region_id, name=r.name, box=r.box,
                    prompt=r.prompt, identity_prompt=r.identity_prompt,
                    negative_prompt=r.negative_prompt, role=r.role,
                )
                for r in plan.bound.plan.regions
            ]

        settings = dict(DEFAULT_TUNING)
        settings.update(tuning or {})
        payload = {
            "version": CONFIG_VERSION,
            "global_prompt": global_prompt,
            "global_negative": negative_prompt,
            "regions": [
                {
                    "id": r.region_id,
                    "name": r.name,
                    "box": {
                        "x0": r.box.x0, "y0": r.box.y0, "x1": r.box.x1, "y1": r.box.y1
                    },
                    "prompt": r.prompt,
                    "negative_prompt": r.negative_prompt,
                    "face_identity_prompt": r.identity_prompt,
                    "enabled": r.enabled,
                    "priority": r.priority,
                    "spatial_role": r.role,
                }
                for r in (regions or [])
            ],
            "loras": [
                {
                    "id": spec.lora_id,
                    "name": spec.lora_name,
                    "display_name": spec.display_name,
                    "strength": spec.strength,
                    "global": spec.global_scope,
                    "region_ids": list(spec.region_ids),
                    "routing_mode": spec.routing_mode,
                    "trigger_phrase": spec.trigger_phrase,
                }
                for spec in (loras or [])
            ],
            "emphases": [
                {
                    "scope_id": item.scope_id,
                    "phrase": item.phrase,
                    "strength": item.strength,
                    "occurrence": item.occurrence,
                }
                for item in (emphasis or [])
            ],
            "spatial": {
                "enabled": bool(settings["spatial_enabled"]),
                "strength": float(settings["inside_strength"]),
                "outside_penalty": float(settings["outside_penalty"]),
                "falloff_pixels": float(settings["falloff_pixels"]),
                "subject_competition": bool(settings["subject_competition"]),
                "subject_fill": bool(settings["subject_fill"]),
                "late_step_scale": float(settings["late_step_scale"]),
                "lora_delta_adaptation": bool(settings["lora_delta_adaptation"]),
                "lora_delta_adaptation_gain": float(settings["lora_adaptation_gain"]),
                "strict_lora_isolation": bool(settings["strict_isolation"]),
            },
            "projector": {
                "enabled": bool((projector or {}).get("enabled", False)),
                "preset": (projector or {}).get("preset", "filter_bypass2"),
                "values": list((projector or {}).get("values", [0.0] * PROJECTOR_LENGTH)),
                "multiplier": 1.0,
                "identity_protection": float(
                    (projector or {}).get("identity_protection", 1.0)
                ),
            },
            "canvas": {"width": int(width), "height": int(height)},
        }
        text = json.dumps(payload, indent=2)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "FVM_K2_ProjectImport": FVM_K2_ProjectImport,
    "FVM_K2_ProjectExport": FVM_K2_ProjectExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_K2_ProjectImport": "K2 Project Import",
    "FVM_K2_ProjectExport": "K2 Project Export",
}
