"""Sign Tools — find, judge and re-render text regions in generated images.

Self-registering submodule, mirroring the K2 Lab pattern: a failure in here must
not take the rest of FVMtools down with it.
"""

from .selector import SignSelectorSAM3
from .proposer import SignTextProposer
from .detailer import SignDetailer
from .options import SignOptions


NODE_CLASS_MAPPINGS = {
    "FVM_SignSelectorSAM3": SignSelectorSAM3,
    "FVM_SignTextProposer": SignTextProposer,
    "FVM_SignDetailer":     SignDetailer,
    "FVM_SignOptions":      SignOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_SignSelectorSAM3": "Sign Selector SAM3",
    "FVM_SignTextProposer": "Sign Text Proposer (LM Studio)",
    "FVM_SignDetailer":     "Sign Detailer",
    "FVM_SignOptions":      "Sign Options",
}

__all__ = [
    "SignSelectorSAM3", "SignTextProposer", "SignDetailer", "SignOptions",
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
]
