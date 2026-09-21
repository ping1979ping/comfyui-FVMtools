"""Batch Tools — walk a folder of renders, judge each picture, sort it away.

Self-registering submodule, mirroring the K2 Lab and Sign Tools pattern: a
failure in here must not take the rest of FVMtools down with it.

The intended chain::

    Batch Load ──image──> Reality Check ──passed──┐
               ──image──> Person Selector ────────┤
               │            (match, face_count)   ├─> Batch Router ─> Batch Save
               ├──pass_dir ───────────────────────┘        target_dir  ↑
               ├──fail_dir ────────────────────────────────────────────┘
               └──source_path ─────────────────────────────────────────┘

Person Selector is the existing identity node — it answers "is this the right
person" and "how many faces are there", which is exactly the second half of the
gate.
"""

from .loader import FVM_BatchLoadImage
from .router import FVM_BatchRouter
from .saver import FVM_BatchSaveImage
from .save_multi import FVM_BatchSaveMulti
from .reality import FVM_RealityCheck, FVM_RealityCheckProbe


NODE_CLASS_MAPPINGS = {
    "FVM_BatchLoadImage": FVM_BatchLoadImage,
    "FVM_RealityCheck": FVM_RealityCheck,
    "FVM_RealityCheckProbe": FVM_RealityCheckProbe,
    "FVM_BatchRouter": FVM_BatchRouter,
    "FVM_BatchSaveImage": FVM_BatchSaveImage,
    "FVM_BatchSaveMulti": FVM_BatchSaveMulti,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_BatchLoadImage": "Batch Load Image",
    "FVM_RealityCheck": "Reality Check (LM Studio)",
    "FVM_RealityCheckProbe": "Reality Check Probe",
    "FVM_BatchRouter": "Batch Router",
    "FVM_BatchSaveImage": "Batch Save Image",
    "FVM_BatchSaveMulti": "Batch Save Multi",
}

__all__ = [
    "FVM_BatchLoadImage",
    "FVM_RealityCheck",
    "FVM_RealityCheckProbe",
    "FVM_BatchRouter",
    "FVM_BatchSaveImage",
    "FVM_BatchSaveMulti",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
