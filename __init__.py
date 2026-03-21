# comfyui-FVMtools — Unified Face Detailing Toolkit
from .nodes.person_selector import PersonSelector
from .nodes.person_selector_multi import PersonSelectorMulti
from .nodes.person_detailer import PersonDetailer
from .nodes.detail_daemon_options import DetailDaemonOptions
from .nodes.inpaint_options import InpaintOptions

NODE_CLASS_MAPPINGS = {
    "PersonSelector": PersonSelector,
    "PersonSelectorMulti": PersonSelectorMulti,
    "PersonDetailer": PersonDetailer,
    "DetailDaemonOptions": DetailDaemonOptions,
    "InpaintOptions": InpaintOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonSelector": "Person Selector (Match)",
    "PersonSelectorMulti": "Person Selector Multi",
    "PersonDetailer": "Person Detailer",
    "DetailDaemonOptions": "Detail Daemon Options",
    "InpaintOptions": "Inpaint Options",
}

WEB_DIRECTORY = "./web/js"
