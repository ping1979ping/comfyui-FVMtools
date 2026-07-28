"""FVM K2 Lab — Sammel-Loader für die drei Krea-2-Komponenten."""

import comfy.sd
import comfy.utils
import folder_paths
import torch

CATEGORY = "FVM Tools/K2"

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


class FVM_K2_Loader:
    """Lädt Krea-2-Transformer, Qwen-Textencoder und VAE in einer Node."""

    DESCRIPTION = (
        "Convenience loader for the three Krea 2 components. Equivalent to "
        "UNETLoader + CLIPLoader(type=krea2) + VAELoader, just tidier.\n\n"
        "Use the native loaders instead when the graph needs quantization, caching or "
        "device nodes in between — everything downstream works with either.\n\n"
        "The text encoder must be a Krea-compatible Qwen3-VL-4B; a generic encoder "
        "produces the wrong 12-layer conditioning layout and Krea will refuse it."
    )
    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_model": (folder_paths.get_filename_list("diffusion_models"),
                                    {"tooltip": "Krea 2 Turbo or RAW transformer."}),
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),
                                 {"tooltip": "Qwen3-VL-4B encoder for Krea 2."}),
                "vae": (folder_paths.get_filename_list("vae"),
                        {"tooltip": "Qwen-Image VAE (Krea 2 shares its autoencoder)."}),
                "weight_dtype": (WEIGHT_DTYPES, {"default": "default",
                                 "tooltip": "'default' follows the file. FP8 modes lower "
                                 "model memory where the device supports them; "
                                 "fp8_e4m3fn_fast also enables ComfyUI FP8 matmul."}),
                "text_encoder_device": (["default", "cpu"], {"default": "default",
                                        "tooltip": "'cpu' keeps Qwen off the GPU — slower "
                                        "prompt encoding, noticeably less VRAM."}),
            },
        }

    def load(self, diffusion_model, text_encoder, vae, weight_dtype,
             text_encoder_device):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", diffusion_model
        )
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        clip_options = {}
        if text_encoder_device == "cpu":
            clip_options["load_device"] = clip_options["offload_device"] = torch.device(
                "cpu"
            )
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.KREA2,
            model_options=clip_options,
        )

        vae_path = folder_paths.get_full_path_or_raise("vae", vae)
        vae_object = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        vae_object.throw_exception_if_invalid()
        return (model, clip, vae_object)


NODE_CLASS_MAPPINGS = {"FVM_K2_Loader": FVM_K2_Loader}
NODE_DISPLAY_NAME_MAPPINGS = {"FVM_K2_Loader": "K2 Load Krea 2"}
