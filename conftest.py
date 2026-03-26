import os
import sys
from unittest.mock import MagicMock

# Add project root to sys.path for direct imports in tests
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Prevent pytest from collecting the root __init__.py
collect_ignore = ["__init__.py"]


def mock_comfy_modules():
    """Mock ComfyUI imports so nodes can be imported without a running ComfyUI instance."""
    modules = [
        "comfy", "comfy.model_management", "comfy.utils",
        "comfy.sd", "comfy.samplers", "comfy.sample",
        "comfy.model_patcher", "comfy.controlnet",
        "folder_paths", "server", "execution",
        "impact", "impact.core", "impact.subcore",
        "segment_anything",
    ]
    for mod in modules:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


mock_comfy_modules()
