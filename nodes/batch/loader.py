"""FVM_BatchLoadImage — hand out one image per queue run and remember where you were.

Point it at a folder, press Queue with a run count, and each run picks up the
next picture. Two subfolder names are turned into absolute paths and passed
downstream, so the rest of the graph can sort each picture into "keep" or
"reject" without anyone typing a path twice.

The progress marker is a set of done filenames rather than a counter — see
:mod:`core.batch_state` for why that matters as soon as files are moved instead
of copied.
"""

import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

try:  # relative inside ComfyUI's loader, absolute under pytest
    from ...core.batch_state import (
        clear_done, get_done, list_images, next_file, progress, set_done,
    )
except ImportError:
    from core.batch_state import (
        clear_done, get_done, list_images, next_file, progress, set_done,
    )

try:
    from comfy.model_management import InterruptProcessingException as _Interrupt
    # Under pytest ``comfy`` is a MagicMock, so the import succeeds but hands
    # back something that cannot be raised. Only inherit from the real thing.
    if not (isinstance(_Interrupt, type) and issubclass(_Interrupt, BaseException)):
        raise ImportError("not an exception type")
except ImportError:  # pragma: no cover — outside ComfyUI
    class _Interrupt(Exception):
        pass


class BatchFinished(_Interrupt):
    """Raised when the last picture has been handed out and looping is off.

    Derives from ComfyUI's interrupt so the run ends the way a Cancel does —
    quietly, without painting the graph red — because reaching the end of a
    batch is success, not an error. The message still reaches the log and the
    node's status line.
    """


def load_image_file(path):
    """Read one file into ComfyUI's (IMAGE, MASK) pair.

    Follows the stock LoadImage behaviour: EXIF rotation applied, frames beyond
    the first ignored, alpha channel inverted into a mask, and a fully opaque
    mask when the file has no alpha.
    """
    image = Image.open(path)

    frame = next(iter(ImageSequence.Iterator(image)))
    frame = ImageOps.exif_transpose(frame)
    if frame.mode == "I":                      # 32-bit integer TIFFs
        frame = frame.point(lambda value: value * (1 / 255))
    rgb = frame.convert("RGB")

    pixels = np.array(rgb).astype(np.float32) / 255.0
    tensor = torch.from_numpy(pixels)[None, ]

    if "A" in frame.getbands():
        alpha = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(alpha)
    else:
        mask = torch.zeros((rgb.height, rgb.width), dtype=torch.float32)
    return tensor, mask.unsqueeze(0)


def make_bar(fraction, width=24):
    """A text progress bar, so the count is readable even without the canvas."""
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "█" * filled + "░" * (width - filled)


class FVM_BatchLoadImage:
    """Load the next image from a directory, one per queue run."""

    CATEGORY = "FVM Tools/Batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "tooltip": "Folder to walk through. One image per queue run.",
                }),
                "pass_subdir": ("STRING", {
                    "default": "keep",
                    "tooltip": "Subfolder of the source directory for pictures that "
                               "pass. Handed downstream as an absolute path.",
                }),
                "fail_subdir": ("STRING", {
                    "default": "reject",
                    "tooltip": "Subfolder for pictures that fail.",
                }),
                "on_finish": (["stop", "loop"], {
                    "default": "stop",
                    "tooltip": "stop: end the run once every image has been handed "
                               "out. loop: forget the progress and start over.",
                }),
                "sort_by": (["name", "modified"], {"default": "name"}),
                "include_subdirs": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Walk subfolders too. The pass/reject folders and "
                               "any other dot-folders are always skipped.",
                }),
                "create_dirs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Create the pass/reject subfolders if missing.",
                }),
                "tracker": ("STRING", {
                    "default": "default",
                    "tooltip": "Name of this run's progress marker. Use different "
                               "names to walk one folder several times "
                               "independently.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING",
                    "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "source_path", "pass_dir", "fail_dir",
                    "filename", "index", "total", "progress")
    FUNCTION = "execute"
    DESCRIPTION = ("Hands out one image per queue run and remembers how far it "
                   "got, so a folder can be worked through across many runs.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Every queue run must actually execute this node, otherwise ComfyUI
        # serves the cached first image forever and the batch never advances.
        return float("nan")

    def execute(self, directory, pass_subdir, fail_subdir, on_finish, sort_by,
                include_subdirs, create_dirs, tracker, unique_id=None):
        directory = os.path.abspath(os.path.expanduser((directory or "").strip().strip('"')))
        if not os.path.isdir(directory):
            raise ValueError(f"[FVM Batch Load] Not a directory: {directory}")

        tracker = (tracker or "default").strip() or "default"
        done = get_done(directory, tracker)

        filename, done, wrapped = next_file(
            directory, done, include_subdirs=include_subdirs, sort_by=sort_by,
            loop=on_finish == "loop",
        )

        if filename is None:
            total = len(list_images(directory, include_subdirs, sort_by)) or len(done)
            message = (f"[FVM Batch Load] Finished: all {total} images in "
                       f"{directory} have been handed out (tracker '{tracker}'). "
                       f"Reset the progress in the node to run it again.")
            print(message)
            raise BatchFinished(message)

        source_path = os.path.join(directory, filename)
        image, mask = load_image_file(source_path)

        done = set(done) | {filename}
        position, total, remaining = progress(directory, done, include_subdirs, sort_by)
        set_done(directory, done, tracker, extra={"last": filename})

        pass_dir = os.path.join(directory, (pass_subdir or "").strip()) if pass_subdir.strip() else directory
        fail_dir = os.path.join(directory, (fail_subdir or "").strip()) if fail_subdir.strip() else directory
        if create_dirs:
            for target in (pass_dir, fail_dir):
                try:
                    os.makedirs(target, exist_ok=True)
                except OSError as error:
                    print(f"[FVM Batch Load] Could not create {target}: {error}")

        fraction = position / total if total else 0.0
        status = (f"{position}/{total}  {fraction * 100:.0f}%  "
                  f"{make_bar(fraction)}  {filename}")
        if wrapped:
            status = "↻ restarted  " + status
        print(f"[FVM Batch Load] {position}/{total} ({fraction * 100:.0f}%) {filename}")

        return {
            "ui": {"text": [status], "fvm_batch": [{
                "position": position, "total": total, "remaining": remaining,
                "fraction": fraction, "filename": filename, "wrapped": wrapped,
            }]},
            "result": (image, mask, source_path, pass_dir, fail_dir, filename,
                       position, total, status),
        }


def reset_tracker(directory, tracker="default"):
    """Forget a tracker's progress — used by the node's Reset button."""
    directory = os.path.abspath(os.path.expanduser((directory or "").strip().strip('"')))
    if not os.path.isdir(directory):
        return False
    return clear_done(directory, (tracker or "default").strip() or "default")


def tracker_status(directory, tracker="default", include_subdirs=False, sort_by="name"):
    """Counts for the node's status line without running the graph."""
    directory = os.path.abspath(os.path.expanduser((directory or "").strip().strip('"')))
    if not os.path.isdir(directory):
        return {"ok": False, "error": "not a directory", "total": 0,
                "done": 0, "remaining": 0}
    done = get_done(directory, (tracker or "default").strip() or "default")
    available = list_images(directory, include_subdirs, sort_by)
    moved_away = [name for name in done if name not in set(available)]
    total = len(available) + len(moved_away)
    remaining = len([name for name in available if name not in done])
    return {"ok": True, "error": None, "total": total, "done": len(done),
            "remaining": remaining,
            "fraction": (total - remaining) / total if total else 0.0}
