"""SMP SidecarSaver — saves an IMAGE batch and writes a PROMPT_DICT JSON next to each.

Mirrors the path-naming convention of ComfyUI's stock SaveImage so the
JSON sits exactly beside the rendered image. OUTPUT_NODE so it triggers
on Queue Prompt without needing a downstream consumer.
"""

from __future__ import annotations

import json
import os

try:
    from ...core.smp.merge import deep_merge  # noqa: F401  (kept for future use)
except ImportError:  # pragma: no cover
    from core.smp.merge import deep_merge  # noqa: F401


def _to_pil(image_tensor):
    """Convert a single ComfyUI IMAGE [H, W, C] (or [B, H, W, C]) tensor to PIL.Image."""
    import numpy as np
    from PIL import Image

    arr = image_tensor
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


class FVM_SMP_SidecarSaver:
    """Save an IMAGE batch + write a PROMPT_DICT sidecar next to each frame.

    Naming follows ComfyUI's SaveImage: ``<output>/<filename_prefix>_<NNNNN>_.png``
    plus ``<output>/<filename_prefix>_<NNNNN>_.prompt.json`` for the dict.
    """

    CATEGORY = "FVM Tools/SMP/Output"
    FUNCTION = "save"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Saves the IMAGE batch and writes a JSON sidecar containing the\n"
        "full PROMPT_DICT next to each image. Use in place of SaveImage\n"
        "when you want reproducible structural-prompt metadata on disk."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "SMP/img"}),
            },
            "optional": {
                "prompt_dict": ("PROMPT_DICT",),
                "structured":  ("STRUCTURED_PROMPTS",),
                "extra_metadata_json": ("STRING", {"default": "{}", "multiline": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save(self, images, filename_prefix, prompt_dict=None, structured=None,
             extra_metadata_json="{}", prompt=None, extra_pnginfo=None):
        # Lazy imports — keep node importable under pytest where folder_paths is mocked.
        import folder_paths
        from PIL.PngImagePlugin import PngInfo

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(filename_prefix, output_dir,
                                             images[0].shape[1], images[0].shape[0])
            if hasattr(folder_paths, "get_save_image_path") else
            (os.path.join(output_dir, filename_prefix.replace("/", os.sep).rsplit(os.sep, 1)[0]),
             os.path.basename(filename_prefix), 1, "", filename_prefix)
        )
        os.makedirs(full_output_folder, exist_ok=True)

        # Compose the sidecar payload.
        sidecar: dict = {}
        if prompt_dict:
            sidecar["prompt_dict"] = prompt_dict
        if structured:
            sidecar["structured_prompts"] = structured
        try:
            extras = json.loads(extra_metadata_json) if extra_metadata_json else {}
            if isinstance(extras, dict) and extras:
                sidecar["extras"] = extras
        except (json.JSONDecodeError, TypeError):
            sidecar["extras_parse_error"] = extra_metadata_json

        results: list[dict] = []
        for idx, image in enumerate(images):
            pil = _to_pil(image)

            metadata = PngInfo()
            if prompt is not None:
                try:
                    metadata.add_text("prompt", json.dumps(prompt))
                except (TypeError, ValueError):
                    pass
            if extra_pnginfo:
                for k, v in extra_pnginfo.items():
                    try:
                        metadata.add_text(k, json.dumps(v))
                    except (TypeError, ValueError):
                        pass
            if sidecar:
                try:
                    metadata.add_text("smp_sidecar",
                                      json.dumps(sidecar, default=str))
                except (TypeError, ValueError):
                    pass

            file_id = f"{filename}_{counter + idx:05}_"
            png_path = os.path.join(full_output_folder, f"{file_id}.png")
            json_path = os.path.join(full_output_folder, f"{file_id}.prompt.json")

            pil.save(png_path, pnginfo=metadata, compress_level=4)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2, ensure_ascii=False, default=str)

            results.append({
                "filename": f"{file_id}.png",
                "subfolder": subfolder,
                "type": "output",
            })

        return {"ui": {"images": results}}
