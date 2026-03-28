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
    from .nodes.person_data_refiner import PersonDataRefiner

    # ── API routes for outfit list editing ──
    import os
    from aiohttp import web
    from server import PromptServer
    from .core.outfit_lists import _get_lists_path

    @PromptServer.instance.routes.get("/fvmtools/outfit-files")
    async def _get_outfit_files(request):
        """List .txt files in an outfit set directory."""
        outfit_set = request.rel_url.query.get("set", "")
        if not outfit_set or "/" in outfit_set or "\\" in outfit_set or ".." in outfit_set:
            return web.json_response({"error": "invalid set"}, status=400)
        set_dir = os.path.join(_get_lists_path(), outfit_set)
        if not os.path.isdir(set_dir):
            return web.json_response({"error": "set not found"}, status=404)
        files = sorted(f[:-4] for f in os.listdir(set_dir) if f.endswith(".txt"))
        return web.json_response({"files": files})

    @PromptServer.instance.routes.get("/fvmtools/outfit-list")
    async def _get_outfit_list(request):
        """Read a .txt list file."""
        outfit_set = request.rel_url.query.get("set", "")
        filename = request.rel_url.query.get("file", "")
        if not outfit_set or not filename:
            return web.json_response({"error": "missing params"}, status=400)
        for val in (outfit_set, filename):
            if "/" in val or "\\" in val or ".." in val:
                return web.json_response({"error": "invalid path"}, status=400)
        file_path = os.path.join(_get_lists_path(), outfit_set, f"{filename}.txt")
        if not os.path.isfile(file_path):
            return web.json_response({"error": "file not found"}, status=404)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return web.json_response({"content": content, "path": file_path})

    @PromptServer.instance.routes.post("/fvmtools/outfit-list")
    async def _save_outfit_list(request):
        """Save a .txt list file."""
        data = await request.json()
        outfit_set = data.get("set", "")
        filename = data.get("file", "")
        content = data.get("content", "")
        if not outfit_set or not filename:
            return web.json_response({"error": "missing params"}, status=400)
        for val in (outfit_set, filename):
            if "/" in val or "\\" in val or ".." in val:
                return web.json_response({"error": "invalid path"}, status=400)
        file_path = os.path.join(_get_lists_path(), outfit_set, f"{filename}.txt")
        set_dir = os.path.dirname(file_path)
        if not os.path.isdir(set_dir):
            return web.json_response({"error": "set not found"}, status=404)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return web.json_response({"success": True})

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
        "PersonDataRefiner": PersonDataRefiner,
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
        "PersonDataRefiner": "Person Data Refiner",
    }

    WEB_DIRECTORY = "./web/js"
except ImportError:
    # Running outside ComfyUI context (e.g. pytest) — skip node registration
    pass
