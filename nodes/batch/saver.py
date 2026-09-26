"""FVM_BatchSaveImage — put one picture where the pipeline decided it belongs.

Takes the target folder as a string, so the routing decision is made upstream
(by :class:`FVM_BatchRouter`) and this node only carries it out.

Three modes, because "sort a folder" and "render a folder" are different jobs:

``save``  encode the tensor and write a new file — use when the graph changed
          the picture.
``move``  move the original file, leaving the source folder emptier — the
          tidy-up mode: afterwards the folder holds only what nothing was
          decided about.
``copy``  copy the original, leaving the source folder intact.

``move`` and ``copy`` need ``source_path`` and hand the original bytes over
untouched, which keeps the JPEG the generator wrote rather than re-encoding it.
"""

import os
import shutil

import numpy as np
from PIL import Image

try:
    from ...core.batch_state import IMAGE_EXTENSIONS
except ImportError:
    from core.batch_state import IMAGE_EXTENSIONS


def unique_path(path, overwrite=False):
    """Return a free path — ``name.jpg`` → ``name_001.jpg`` — unless overwriting."""
    if overwrite or not os.path.exists(path):
        return path
    stem, extension = os.path.splitext(path)
    for counter in range(1, 10000):
        candidate = f"{stem}_{counter:03d}{extension}"
        if not os.path.exists(candidate):
            return candidate
    raise OSError(f"no free filename next to {path}")


def tensor_to_pil(image):
    """First frame of a ComfyUI IMAGE batch as a PIL image."""
    array = (
        image[0].detach().cpu().numpy()
        if image.ndim == 4
        else image.detach().cpu().numpy()
    )
    return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))


class FVM_BatchSaveImage:
    """Save, move or copy one picture into a directory chosen upstream."""

    CATEGORY = "FVM Tools/Batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Target folder — wire this to the router's "
                        "target_dir or the loader's pass_dir/fail_dir.",
                    },
                ),
                "mode": (
                    ["save", "move", "copy"],
                    {
                        "default": "move",
                        "tooltip": "save: write the image tensor as a new file. "
                        "move: move the original file (tidies the source "
                        "folder). copy: copy the original, source stays.",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Prepended to the filename. Empty keeps the "
                        "original name.",
                    },
                ),
                "format": (
                    ["keep", "png", "jpg", "webp"],
                    {
                        "default": "keep",
                        "tooltip": "Output format in save mode. keep: reuse the "
                        "source file's extension, PNG if unknown.",
                    },
                ),
                "quality": (
                    "INT",
                    {
                        "default": 95,
                        "min": 1,
                        "max": 100,
                        "tooltip": "JPEG/WebP quality in save mode.",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Off: a colliding name gets _001, _002, … "
                        "On: the existing file is replaced.",
                    },
                ),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Off: do nothing and pass the image through. Lets "
                        "one branch of a fork stay idle.",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Required for save mode."}),
                "source_path": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "The loader's source_path. Required for move/copy.",
                    },
                ),
                "report": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": "Written next to the picture as a .txt sidecar, so "
                        "the folder records why each file landed there.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("image", "saved_path", "written")
    FUNCTION = "execute"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Writes, moves or copies one picture into a directory decided "
        "upstream. Move mode tidies the source folder as it goes."
    )

    def execute(
        self,
        directory,
        mode,
        filename_prefix,
        format,
        quality,
        overwrite,
        enabled,
        image=None,
        source_path="",
        report="",
    ):
        passthrough = image if image is not None else None

        if not enabled:
            return {"ui": {"text": ["disabled"]}, "result": (passthrough, "", False)}

        directory = (directory or "").strip().strip('"')
        if not directory:
            message = "no target directory"
            print(f"[FVM Batch Save] {message}")
            return {"ui": {"text": [message]}, "result": (passthrough, "", False)}

        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as error:
            message = f"cannot create {directory}: {error}"
            print(f"[FVM Batch Save] {message}")
            return {"ui": {"text": [message]}, "result": (passthrough, "", False)}

        source_path = (source_path or "").strip().strip('"')
        base = os.path.basename(source_path) if source_path else "image.png"
        stem, source_ext = os.path.splitext(base)
        prefix = (filename_prefix or "").strip()

        try:
            if mode in ("move", "copy"):
                target = self._transfer(
                    mode, source_path, directory, prefix, stem, source_ext, overwrite
                )
            else:
                target = self._save(
                    passthrough,
                    directory,
                    prefix,
                    stem,
                    source_ext,
                    format,
                    quality,
                    overwrite,
                )
        except (OSError, ValueError) as error:
            message = f"{mode} failed: {error}"
            print(f"[FVM Batch Save] {message}")
            return {"ui": {"text": [message]}, "result": (passthrough, "", False)}

        if report and report.strip():
            sidecar = os.path.splitext(target)[0] + ".txt"
            try:
                with open(sidecar, "w", encoding="utf-8") as handle:
                    handle.write(report)
            except OSError as error:
                print(f"[FVM Batch Save] Could not write sidecar: {error}")

        status = f"{mode} → {target}"
        print(f"[FVM Batch Save] {status}")
        return {"ui": {"text": [status]}, "result": (passthrough, target, True)}

    def _transfer(
        self, mode, source_path, directory, prefix, stem, extension, overwrite
    ):
        """Move or copy the original file, bytes untouched."""
        if not source_path:
            raise ValueError(f"{mode} mode needs source_path")
        if not os.path.isfile(source_path):
            raise ValueError(f"source file is gone: {source_path}")

        target = unique_path(
            os.path.join(directory, f"{prefix}{stem}{extension}"), overwrite
        )
        if os.path.abspath(source_path) == os.path.abspath(target):
            return target  # already where it belongs
        if mode == "move":
            shutil.move(source_path, target)
        else:
            shutil.copy2(source_path, target)
        return target

    def _save(
        self, image, directory, prefix, stem, source_ext, format, quality, overwrite
    ):
        """Encode the tensor into a new file."""
        if image is None:
            raise ValueError("save mode needs an image input")

        if format == "keep":
            extension = source_ext if source_ext.lower() in IMAGE_EXTENSIONS else ".png"
        else:
            extension = "." + format
        target = unique_path(
            os.path.join(directory, f"{prefix}{stem}{extension}"), overwrite
        )

        pil = tensor_to_pil(image)
        lowered = extension.lower()
        if lowered in (".jpg", ".jpeg"):
            # JPEG has no alpha; GLSL Shader and some VAE decodes hand over
            # RGBA, which PIL refuses to encode.
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            pil.save(target, quality=int(quality), subsampling=0)
        elif lowered == ".webp":
            pil.save(target, quality=int(quality))
        else:
            pil.save(target)
        return target
