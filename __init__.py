# comfyui-FVMtools — Unified Face Detailing Toolkit + Color Tools
try:
    from .nodes.person_selector import PersonSelector
    from .nodes.person_selector_multi import PersonSelectorMulti
    from .nodes.person_detailer import PersonDetailer
    from .nodes.detail_daemon_options import DetailDaemonOptions
    from .nodes.inpaint_options import InpaintOptions
    from .nodes.prompt_color_replace import FVM_PromptColorReplace
    from .nodes.color_palette_generator import FVM_ColorPaletteGenerator
    from .nodes.palette_from_image import FVM_PaletteFromImage
    from .nodes.outfit_generator import FVM_OutfitGenerator

    NODE_CLASS_MAPPINGS = {
        "PersonSelector": PersonSelector,
        "PersonSelectorMulti": PersonSelectorMulti,
        "PersonDetailer": PersonDetailer,
        "DetailDaemonOptions": DetailDaemonOptions,
        "InpaintOptions": InpaintOptions,
        "FVM_PromptColorReplace": FVM_PromptColorReplace,
        "FVM_ColorPaletteGenerator": FVM_ColorPaletteGenerator,
        "FVM_PaletteFromImage": FVM_PaletteFromImage,
        "FVM_OutfitGenerator": FVM_OutfitGenerator,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "PersonSelector": "Person Selector (Match)",
        "PersonSelectorMulti": "Person Selector Multi",
        "PersonDetailer": "Person Detailer",
        "DetailDaemonOptions": "Detail Daemon Options",
        "InpaintOptions": "Inpaint Options",
        "FVM_PromptColorReplace": "Prompt Color Replace",
        "FVM_ColorPaletteGenerator": "Color Palette Generator",
        "FVM_PaletteFromImage": "Palette From Image",
        "FVM_OutfitGenerator": "Outfit Generator",
    }

    WEB_DIRECTORY = "./web/js"
except ImportError:
    # Running outside ComfyUI context (e.g. pytest) — skip node registration
    pass
