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
    from .nodes.person_detailer_power import PersonDetailerPower

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

    import folder_paths as _folder_paths

    def _read_sidecar_metadata(lora_path):
        """Read metadata from sidecar files next to the LoRA.
        Checks: .metadata.json (model manager) and .safetensors.rgthree-info.json (rgthree)."""
        result = {}
        base = lora_path
        # Strip .safetensors extension for .metadata.json pattern
        stem = lora_path.rsplit(".safetensors", 1)[0] if lora_path.endswith(".safetensors") else lora_path

        # Try .metadata.json (written by model manager tools)
        meta_path = stem + ".metadata.json"
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = _json.load(f)
                result["name"] = meta.get("model_name") or meta.get("file_name", "")
                result["baseModel"] = meta.get("base_model", "")
                result["sha256"] = meta.get("sha256", "")
                civitai = meta.get("civitai", {})
                if civitai:
                    result["version"] = civitai.get("name", "")
                    result["triggerWords"] = civitai.get("trainedWords", [])
                    model_id = civitai.get("modelId", "")
                    version_id = civitai.get("id", "")
                    if model_id:
                        result["civitaiUrl"] = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
                    result["type"] = "LORA"
                # Preview image
                for ext in (".jpeg", ".jpg", ".png", ".webp"):
                    img_path = stem + ext
                    if os.path.isfile(img_path):
                        result["previewImage"] = img_path
                        break
            except Exception:
                pass

        # Try .safetensors.rgthree-info.json (written by rgthree)
        rgthree_path = lora_path + ".rgthree-info.json"
        if os.path.isfile(rgthree_path):
            try:
                with open(rgthree_path, "r", encoding="utf-8") as f:
                    info = _json.load(f)
                if not result.get("name"):
                    result["name"] = info.get("name", "")
                if not result.get("baseModel"):
                    result["baseModel"] = info.get("baseModel", "")
                if not result.get("sha256"):
                    result["sha256"] = info.get("sha256", "")
                if not result.get("type"):
                    result["type"] = info.get("type", "LORA")
                # rgthree stores trainedWords differently
                trained = info.get("trainedWords", [])
                if trained and not result.get("triggerWords"):
                    result["triggerWords"] = [w.get("word", w) if isinstance(w, dict) else w for w in trained]
                # Links
                links = info.get("links", [])
                if links and not result.get("civitaiUrl"):
                    for link in links:
                        if "civitai.com" in str(link):
                            result["civitaiUrl"] = link
                            break
            except Exception:
                pass

        return result

    @PromptServer.instance.routes.get("/fvmtools/lora-info")
    async def _get_lora_info(request):
        """Get LoRA info — reads sidecar metadata first, falls back to CivitAI API."""
        lora_name = request.rel_url.query.get("file", "")
        if not lora_name:
            return web.json_response({"error": "missing file param"}, status=400)

        try:
            lora_path = _folder_paths.get_full_path_or_raise("loras", lora_name)
        except Exception:
            return web.json_response({"error": "lora not found", "file": lora_name}, status=404)

        # Check in-memory cache first
        cached = _lora_info_cache.get(f"info:{lora_path}")
        if cached is not None:
            return web.json_response(cached)

        # Read sidecar metadata files (.metadata.json, .rgthree-info.json)
        sidecar = _read_sidecar_metadata(lora_path)

        # If sidecar has good data (name + triggerWords or civitaiUrl), use it directly
        if sidecar.get("name") and (sidecar.get("triggerWords") or sidecar.get("civitaiUrl")):
            result = {
                "name": sidecar.get("name", lora_name),
                "version": sidecar.get("version", ""),
                "type": sidecar.get("type", "LORA"),
                "baseModel": sidecar.get("baseModel", ""),
                "triggerWords": sidecar.get("triggerWords", []),
                "civitaiUrl": sidecar.get("civitaiUrl", ""),
                "sha256": sidecar.get("sha256", ""),
                "source": "sidecar",
            }
            _lora_info_cache[f"info:{lora_path}"] = result
            return web.json_response(result)

        # Fall back to CivitAI API lookup by SHA256
        file_hash = sidecar.get("sha256", "")
        if not file_hash:
            file_hash = _lora_info_cache.get(f"hash:{lora_path}")
        if not file_hash:
            file_hash = _sha256_file(lora_path)
            _lora_info_cache[f"hash:{lora_path}"] = file_hash

        try:
            async with _aiohttp.ClientSession() as session:
                url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
                async with session.get(url, timeout=_aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        # No CivitAI data — return what we have from sidecar
                        result = {
                            "name": sidecar.get("name", lora_name),
                            "version": sidecar.get("version", ""),
                            "type": sidecar.get("type", ""),
                            "baseModel": sidecar.get("baseModel", ""),
                            "triggerWords": sidecar.get("triggerWords", []),
                            "civitaiUrl": sidecar.get("civitaiUrl", ""),
                            "sha256": file_hash,
                            "error": f"CivitAI returned {resp.status}",
                        }
                        _lora_info_cache[f"info:{lora_path}"] = result
                        return web.json_response(result)
                    data = await resp.json()
        except Exception as e:
            result = {
                "name": sidecar.get("name", lora_name),
                "sha256": file_hash,
                "error": f"CivitAI request failed: {e}",
                **{k: sidecar.get(k, "") for k in ("version", "type", "baseModel", "triggerWords", "civitaiUrl")},
            }
            return web.json_response(result)

        # Extract CivitAI info
        model_info = data.get("model", {})
        trained_words = data.get("trainedWords", [])
        model_id = data.get("modelId", "")
        version_id = data.get("id", "")
        civitai_url = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}" if model_id else ""

        result = {
            "name": model_info.get("name", sidecar.get("name", lora_name)),
            "version": data.get("name", sidecar.get("version", "")),
            "type": model_info.get("type", sidecar.get("type", "")),
            "baseModel": data.get("baseModel", sidecar.get("baseModel", "")),
            "triggerWords": trained_words or sidecar.get("triggerWords", []),
            "civitaiUrl": civitai_url or sidecar.get("civitaiUrl", ""),
            "sha256": file_hash,
            "source": "civitai",
        }
        _lora_info_cache[f"info:{lora_path}"] = result

        # Save as sidecar .metadata.json so next lookup is instant
        stem = lora_path.rsplit(".safetensors", 1)[0] if lora_path.endswith(".safetensors") else lora_path
        meta_path = stem + ".metadata.json"
        if not os.path.isfile(meta_path):
            try:
                sidecar_data = {
                    "file_name": os.path.basename(stem),
                    "model_name": result["name"],
                    "base_model": result["baseModel"],
                    "sha256": file_hash,
                    "from_civitai": True,
                    "civitai": {
                        "id": version_id,
                        "modelId": model_id,
                        "name": result["version"],
                        "baseModel": result["baseModel"],
                        "trainedWords": trained_words,
                    },
                    "metadata_source": "fvmtools_auto",
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    _json.dump(sidecar_data, f, indent=2)
                print(f"[FVMTools] Saved metadata sidecar: {meta_path}")
            except Exception as e:
                print(f"[FVMTools] Could not save sidecar: {e}")

        return web.json_response(result)

    @PromptServer.instance.routes.get("/fvmtools/loras")
    async def _get_loras(request):
        """List available LoRA files for the Power LoRA widget."""
        import folder_paths as _fp
        loras = _fp.get_filename_list("loras")
        return web.json_response({"loras": loras})

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
        "PersonDetailerPower": PersonDetailerPower,
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
        "PersonDetailerPower": "Person Detailer Power",
    }

    WEB_DIRECTORY = "./web/js"
except ImportError:
    # Running outside ComfyUI context (e.g. pytest) — skip node registration
    pass
