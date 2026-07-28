"""FVM K2 Lab — Node-Registrierung.

Krea-2-Regionalsteuerung als graph-native ComfyUI-Bausteine: regionales
Prompting, strikte regionale LoRA-Führung, räumliche Attention, Token-Emphase,
Projector-Kontrolle, Gesichtsverfeinerung, regionales Editieren, Upscaling.
"""

from . import (
    compose,
    detail,
    edit,
    loader,
    loras,
    project,
    projector_nodes,
    regions,
    sampler,
    upscale,
)

_MODULES = (
    loader,
    regions,
    loras,
    projector_nodes,
    compose,
    sampler,
    detail,
    edit,
    upscale,
    project,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _module in _MODULES:
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
