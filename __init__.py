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
    from .nodes.person_detailer_controlnet import PersonDetailerControlNet

    # ── API routes for outfit list editing ──
    import os
    from aiohttp import web
    from server import PromptServer
    from .core.outfit_lists import _get_lists_path

    # ── API routes for LoRA info (CivitAI lookup) ──
    import hashlib
    import json as _json
    import aiohttp as _aiohttp

    _lora_info_cache = {}  # sha256 → civitai data

    def _sha256_file(path, chunk_size=1 << 20):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @PromptServer.instance.routes.get("/fvmtools/lora-info")
    async def _get_lora_info(request):
        """Get LoRA info from CivitAI by SHA256 hash lookup."""
        lora_name = request.rel_url.query.get("file", "")
        if not lora_name:
            return web.json_response({"error": "missing file param"}, status=400)

        try:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        except Exception:
            return web.json_response({"error": "lora not found"}, status=404)

        # Compute SHA256 (cached by path)
        file_hash = _lora_info_cache.get(f"hash:{lora_path}")
        if not file_hash:
            file_hash = _sha256_file(lora_path)
            _lora_info_cache[f"hash:{lora_path}"] = file_hash

        # Check civitai cache
        cached = _lora_info_cache.get(f"civitai:{file_hash}")
        if cached is not None:
            return web.json_response(cached)

        # Query CivitAI API
        try:
            async with _aiohttp.ClientSession() as session:
                url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
                async with session.get(url, timeout=_aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        result = {"error": f"CivitAI returned {resp.status}", "sha256": file_hash}
                        _lora_info_cache[f"civitai:{file_hash}"] = result
                        return web.json_response(result)
                    data = await resp.json()
        except Exception as e:
            return web.json_response({"error": f"CivitAI request failed: {e}", "sha256": file_hash})

        # Extract useful info
        model_info = data.get("model", {})
        trained_words = data.get("trainedWords", [])
        model_id = data.get("modelId", "")
        version_id = data.get("id", "")
        civitai_url = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}" if model_id else ""

        result = {
            "name": model_info.get("name", lora_name),
            "version": data.get("name", ""),
            "type": model_info.get("type", ""),
            "baseModel": data.get("baseModel", ""),
            "triggerWords": trained_words,
            "civitaiUrl": civitai_url,
            "sha256": file_hash,
        }
        _lora_info_cache[f"civitai:{file_hash}"] = result
        return web.json_response(result)

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
        "PersonDetailerControlNet": PersonDetailerControlNet,
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
        "PersonDetailerControlNet": "Person Detailer ControlNet",
    }

    WEB_DIRECTORY = "./web/js"
except ImportError:
    # Running outside ComfyUI context (e.g. pytest) — skip node registration
    pass
