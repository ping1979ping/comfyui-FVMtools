"""Sign Tools — find, judge and re-render text regions in generated images.

Self-registering submodule, mirroring the K2 Lab pattern: a failure in here must
not take the rest of FVMtools down with it.
"""

from .selector import SignSelectorSAM3
from .proposer import SignTextProposer
from .detailer import SignDetailer
from .options import SignOptions
from .ideogram_bridge import FVM_SignsToIdeogram
from .split_lines import FVM_SignSplitLines
from .scene_describe import FVM_SceneDescribe


NODE_CLASS_MAPPINGS = {
    "FVM_SignSelectorSAM3": SignSelectorSAM3,
    "FVM_SignTextProposer": SignTextProposer,
    "FVM_SignDetailer": SignDetailer,
    "FVM_SignOptions": SignOptions,
    "FVM_SignsToIdeogram": FVM_SignsToIdeogram,
    "FVM_SignSplitLines": FVM_SignSplitLines,
    "FVM_SceneDescribe": FVM_SceneDescribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_SignSelectorSAM3": "Sign Selector SAM3",
    "FVM_SignTextProposer": "Sign Text Proposer (LM Studio)",
    "FVM_SignDetailer": "Sign Detailer",
    "FVM_SignOptions": "Sign Options",
    "FVM_SignsToIdeogram": "Signs to Ideogram Caption",
    "FVM_SignSplitLines": "Sign Split Lines",
    "FVM_SceneDescribe": "Scene Describe (LM Studio)",
}

__all__ = [
    "SignSelectorSAM3",
    "SignTextProposer",
    "SignDetailer",
    "SignOptions",
    "FVM_SignsToIdeogram",
    "FVM_SignSplitLines",
    "FVM_SceneDescribe",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
