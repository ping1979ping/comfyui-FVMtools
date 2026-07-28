"""FVM K2 Lab — Nachvergrößerung."""

import comfy.utils
import torch

CATEGORY = "FVM Tools/K2"


class FVM_K2_PostUpscale:
    """Exakte Endgröße per Lanczos oder neuronalem Upscaler."""

    DESCRIPTION = (
        "Scales the finished image to an exact final size.\n\n"
        "'lanczos' is a deterministic high-quality resize. 'upscale_model' first runs "
        "the connected ComfyUI UPSCALE_MODEL (typically 2x or 4x) and then resizes to "
        "the exact requested factor, so a 4x model can serve a 2.5x target without a "
        "second node."
    )
    CATEGORY = CATEGORY
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image batch to enlarge."}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05,
                          "tooltip": "Final width and height multiplier."}),
                "method": (["lanczos", "bicubic", "bilinear", "nearest-exact",
                            "upscale_model"], {"default": "lanczos",
                           "tooltip": "lanczos: deterministic high-quality resize.\n"
                           "upscale_model: run the neural model first, then resize to the "
                           "exact target."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Required only for the "
                                  "'upscale_model' method."}),
                "tile": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64,
                         "tooltip": "Tile size for the neural upscaler (lower = less "
                         "VRAM)."}),
            },
        }

    def upscale(self, image, scale, method, upscale_model=None, tile=512):
        height, width = int(image.shape[1]), int(image.shape[2])
        target_w = max(1, int(round(width * float(scale))))
        target_h = max(1, int(round(height * float(scale))))

        working = image
        if method == "upscale_model":
            if upscale_model is None:
                raise ValueError("Method 'upscale_model' needs a connected UPSCALE_MODEL")
            # Kein `import comfy.…` hier — das würde `comfy` lokal binden und
            # comfy.utils weiter unten unerreichbar machen.
            from comfy.model_management import get_torch_device

            device = get_torch_device()
            upscale_model.to(device)
            samples = image.movedim(-1, -3).to(device)
            overlap = 32
            steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(
                samples.shape[3], samples.shape[2],
                tile_x=tile, tile_y=tile, overlap=overlap,
            )
            pbar = comfy.utils.ProgressBar(steps)
            upscaled = comfy.utils.tiled_scale(
                samples, lambda a: upscale_model(a), tile_x=tile, tile_y=tile,
                overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar,
            )
            upscale_model.to("cpu")
            working = upscaled.movedim(-3, -1).clamp(0.0, 1.0).cpu()
            if int(working.shape[1]) == target_h and int(working.shape[2]) == target_w:
                return (working,)
            method = "lanczos"

        samples = working.movedim(-1, 1)
        result = comfy.utils.common_upscale(
            samples, target_w, target_h, method, "disabled"
        )
        return (result.movedim(1, -1).clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {"FVM_K2_PostUpscale": FVM_K2_PostUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"FVM_K2_PostUpscale": "K2 Post Upscale"}
