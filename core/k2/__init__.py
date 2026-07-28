"""FVM K2 Lab — Krea-2-Regionalsteuerung (Kern-Engine).

Reimplementiert die Fähigkeiten der K2Lab-Desktopanwendung als graph-native
ComfyUI-Bausteine: regionales Prompting, strikte regionale LoRA-Führung,
räumliche Attention, Token-Emphase, Projector-Kontrolle, Gesichtsverfeinerung.

Die Module hier sind frei von ComfyUI-Importen auf Modulebene (torch/comfy
werden erst in den Funktionen geladen), damit sie ohne laufenden Server
getestet werden können.
"""

from .geometry import (
    TOKEN_PIXELS,
    CanvasGeometry,
    PixelBox,
    apply_subject_competition,
    spatial_pair_bias,
)
from .prompt import (
    GLOBAL_SCOPE,
    ROLE_AUTO,
    ROLE_BACKGROUND,
    ROLE_SUBJECT,
    ROLES,
    CompiledPlan,
    CompiledRegion,
    EmphasisRequest,
    RegionDefinition,
    compile_plan,
)
from .binding import BoundPlan, bind_plan, krea_prompt_token_count
from .attention import K2SpatialAttention
from .lora import (
    CHARACTER_ROUTING,
    ROUTING_MODES,
    STANDARD_ROUTING,
    LoraRoute,
    LoraSpec,
    apply_routes,
    compile_routes,
    identity_triggers_from_specs,
    inspect_lora,
)
from .runtime import K2_ATTACHMENT, K2Runtime, region_mask_tensor, union_mask_tensor

__all__ = [
    "CHARACTER_ROUTING",
    "GLOBAL_SCOPE",
    "K2_ATTACHMENT",
    "ROLES",
    "ROLE_AUTO",
    "ROLE_BACKGROUND",
    "ROLE_SUBJECT",
    "ROUTING_MODES",
    "STANDARD_ROUTING",
    "TOKEN_PIXELS",
    "BoundPlan",
    "CanvasGeometry",
    "CompiledPlan",
    "CompiledRegion",
    "EmphasisRequest",
    "K2Runtime",
    "K2SpatialAttention",
    "LoraRoute",
    "LoraSpec",
    "PixelBox",
    "RegionDefinition",
    "apply_routes",
    "apply_subject_competition",
    "bind_plan",
    "compile_plan",
    "compile_routes",
    "identity_triggers_from_specs",
    "inspect_lora",
    "krea_prompt_token_count",
    "region_mask_tensor",
    "spatial_pair_bias",
    "union_mask_tensor",
]
