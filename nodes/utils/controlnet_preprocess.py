"""ControlNet preprocessing for Z-Image Turbo model patches.

Generates pose/depth maps via comfyui_controlnet_aux (DWPose, DepthAnythingV2),
then applies them as Z-Image ControlNet Union model patches.
Loads the model patch internally — no external ControlNet loader needed.

Also supports standard ControlNet for non-Z-Image models.
"""

import torch
import numpy as np

import comfy.utils
import comfy.model_management as model_management
import comfy.model_patcher
import folder_paths

# Import Z-Image model patch classes from ComfyUI core
try:
    from comfy_extras.nodes_model_patch import ZImageControlPatch, z_image_convert
    import comfy.ldm.lumina.controlnet
    HAS_ZIMAGE_CONTROLNET = True
except ImportError:
    HAS_ZIMAGE_CONTROLNET = False

# Try importing controlnet_aux preprocessors
try:
    from custom_controlnet_aux.dwpose import DwposeDetector
    from custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
    from comfyui_controlnet_aux.utils import common_annotator_call
    HAS_CONTROLNET_AUX = True
except ImportError:
    HAS_CONTROLNET_AUX = False


# ── Lazy-cached models ───────────────────────────────────────────────────────

class _PreprocessorCache:
    """Class-level cache for expensive preprocessor models."""
    _dwpose_model = None
    _dwpose_key = None
    _depth_model = None
    _depth_key = None

    @classmethod
    def get_dwpose(cls, bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384.onnx"):
        key = (bbox_detector, pose_estimator)
        if cls._dwpose_model is not None and cls._dwpose_key == key:
            return cls._dwpose_model

        if not HAS_CONTROLNET_AUX:
            raise RuntimeError("comfyui_controlnet_aux not installed — cannot generate pose maps")

        DWPOSE_MODEL_NAME = "yzd-v/DWPose"
        if bbox_detector in ("yolox_l.onnx", "None"):
            yolo_repo = DWPOSE_MODEL_NAME
        elif "yolox" in bbox_detector:
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        else:
            yolo_repo = DWPOSE_MODEL_NAME

        if pose_estimator == "dw-ll_ucoco_384.onnx":
            pose_repo = DWPOSE_MODEL_NAME
        elif pose_estimator.endswith(".onnx"):
            pose_repo = "hr16/UnJIT-DWPose"
        elif pose_estimator.endswith(".torchscript.pt"):
            pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
        else:
            pose_repo = DWPOSE_MODEL_NAME

        det_filename = None if bbox_detector == "None" else bbox_detector
        model = DwposeDetector.from_pretrained(
            pose_repo, yolo_repo,
            det_filename=det_filename, pose_filename=pose_estimator,
            torchscript_device=model_management.get_torch_device(),
        )
        cls._dwpose_model = model
        cls._dwpose_key = key
        print(f"[FVMTools] DWPose loaded: {bbox_detector} + {pose_estimator}")
        return model

    @classmethod
    def get_depth(cls, ckpt_name="depth_anything_v2_vitl.pth"):
        if cls._depth_model is not None and cls._depth_key == ckpt_name:
            return cls._depth_model

        if not HAS_CONTROLNET_AUX:
            raise RuntimeError("comfyui_controlnet_aux not installed — cannot generate depth maps")

        model = DepthAnythingV2Detector.from_pretrained(filename=ckpt_name)
        model = model.to(model_management.get_torch_device())
        cls._depth_model = model
        cls._depth_key = ckpt_name
        print(f"[FVMTools] DepthAnythingV2 loaded: {ckpt_name}")
        return model

    @classmethod
    def cleanup(cls):
        cls._dwpose_model = None
        cls._dwpose_key = None
        cls._depth_model = None
        cls._depth_key = None


class _ModelPatchCache:
    """Class-level cache for the Z-Image ControlNet Union model patch."""
    _model_patch = None
    _model_patch_key = None

    @classmethod
    def get(cls, name):
        if cls._model_patch is not None and cls._model_patch_key == name:
            return cls._model_patch

        if not HAS_ZIMAGE_CONTROLNET:
            raise RuntimeError("Z-Image ControlNet support not available in this ComfyUI version")

        model_patch_path = folder_paths.get_full_path_or_raise("model_patches", name)
        sd = comfy.utils.load_torch_file(model_patch_path, safe_load=True)
        dtype = comfy.utils.weight_dtype(sd)

        # Z-Image Fun ControlNet detection (from ComfyUI ModelPatchLoader)
        if 'control_all_x_embedder.2-1.weight' not in sd:
            raise ValueError(f"'{name}' is not a Z-Image ControlNet Union model patch")

        sd = z_image_convert(sd)
        config = {}
        if 'control_layers.4.adaLN_modulation.0.weight' not in sd:
            config['n_control_layers'] = 3
            config['additional_in_dim'] = 17
            config['refiner_control'] = True
        if 'control_layers.14.adaLN_modulation.0.weight' in sd:
            config['n_control_layers'] = 15
            config['additional_in_dim'] = 17
            config['refiner_control'] = True
            ref_weight = sd.get("control_noise_refiner.0.after_proj.weight", None)
            if ref_weight is not None:
                if torch.count_nonzero(ref_weight) == 0:
                    config['broken'] = True

        model = comfy.ldm.lumina.controlnet.ZImage_Control(
            device=model_management.unet_offload_device(),
            dtype=dtype,
            operations=comfy.ops.manual_cast,
            **config,
        )

        model_patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=model_management.get_torch_device(),
            offload_device=model_management.unet_offload_device(),
        )
        model.load_state_dict(sd, assign=model_patcher.is_dynamic())

        cls._model_patch = model_patcher
        cls._model_patch_key = name
        print(f"[FVMTools] Z-Image ControlNet Union loaded: {name} "
              f"(layers={config.get('n_control_layers', 6)}, "
              f"inpaint={'yes' if config.get('additional_in_dim', 0) > 0 else 'no'})")
        return model_patcher

    @classmethod
    def cleanup(cls):
        cls._model_patch = None
        cls._model_patch_key = None


# ── Preprocessor functions ───────────────────────────────────────────────────

def generate_pose_map(image_tensor, resolution=512,
                      bbox_detector="yolox_l.onnx",
                      pose_estimator="dw-ll_ucoco_384.onnx"):
    """Generate a DWPose map from an image tensor.

    Args:
        image_tensor: ComfyUI IMAGE [1, H, W, C] float32 0-1.
        resolution: Detection resolution.

    Returns:
        torch.Tensor [1, H, W, C] float32 0-1 — pose visualization image.
    """
    model = _PreprocessorCache.get_dwpose(bbox_detector, pose_estimator)

    def pose_fn(image, **kwargs):
        pose_img, _ = model(image, **kwargs)
        return pose_img

    out = common_annotator_call(
        pose_fn, image_tensor,
        include_hand=True, include_face=True, include_body=True,
        image_and_json=True, resolution=resolution,
        show_pbar=False,
    )

    # Soften sharp skeleton lines to prevent them bleeding into the output
    # as visible artifacts. A light Gaussian blur turns hard edges into
    # smooth gradients that guide structure without imprinting lines.
    import cv2
    pose_np = (out[0].cpu().numpy() * 255).astype(np.uint8)
    ksize = max(3, (resolution // 64) * 2 + 1)  # ~7 at 512, scales with resolution
    pose_np = cv2.GaussianBlur(pose_np, (ksize, ksize), 0)
    out = torch.from_numpy(pose_np.astype(np.float32) / 255.0).unsqueeze(0)

    return out


def generate_depth_map(image_tensor, resolution=512,
                       ckpt_name="depth_anything_v2_vitl.pth"):
    """Generate a depth map from an image tensor.

    Args:
        image_tensor: ComfyUI IMAGE [1, H, W, C] float32 0-1.
        resolution: Detection resolution.

    Returns:
        torch.Tensor [1, H, W, C] float32 0-1 — depth map image.
    """
    model = _PreprocessorCache.get_depth(ckpt_name)
    out = common_annotator_call(model, image_tensor, resolution=resolution, max_depth=1, show_pbar=False)
    return out


# ── Z-Image Model Patch application ─────────────────────────────────────────

def apply_zimage_control(model, model_patch, vae, control_image, strength=1.0, mask=None):
    """Apply Z-Image ControlNet Union as model patch.

    Replicates QwenImageDiffsynthControlnet.diffsynth_controlnet() from ComfyUI.

    Args:
        model: MODEL (will be cloned).
        model_patch: MODEL_PATCH (from _ModelPatchCache).
        vae: VAE for encoding control images to latent space.
        control_image: [1, H, W, C] float32 — control signal (pose/depth/canny).
        strength: Control strength multiplier.
        mask: Optional [1, H, W] float32 mask for inpainting guidance.

    Returns:
        Patched MODEL with control signal applied.
    """
    model_patched = model.clone()

    if control_image is not None:
        control_image = control_image[:, :, :, :3]

    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim == 4:
            mask = mask.unsqueeze(2)
        mask = 1.0 - mask

    if isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control):
        patch = ZImageControlPatch(model_patch, vae, control_image, strength, mask=mask)
        model_patched.set_model_noise_refiner_patch(patch)
        model_patched.set_model_double_block_patch(patch)
    else:
        raise TypeError(f"Unsupported model patch type: {type(model_patch.model)}")

    return model_patched


def build_zimage_control_model(model, vae, model_patch_name, cropped_image,
                                control_type="depth", strength=1.0,
                                cn_resolution=512,
                                depth_ckpt="depth_anything_v2_vitl.pth",
                                mask=None):
    """Generate control map from crop and apply as Z-Image model patch.

    Args:
        model: Base MODEL.
        vae: VAE for latent encoding.
        model_patch_name: Filename of the Z-Image ControlNet Union in model_patches/.
        cropped_image: [1, H, W, C] float32 — the inpaint crop.
        control_type: "depth", "pose", or "depth+pose".
        strength: Control strength.
        cn_resolution: Resolution for preprocessors.
        depth_ckpt: DepthAnythingV2 checkpoint.
        mask: Optional inpaint mask.

    Returns:
        Patched MODEL ready for sampling.
    """
    model_patch = _ModelPatchCache.get(model_patch_name)

    # Generate control image based on type
    if control_type == "depth":
        control_image = generate_depth_map(cropped_image, resolution=cn_resolution, ckpt_name=depth_ckpt)
    elif control_type == "pose":
        control_image = generate_pose_map(cropped_image, resolution=cn_resolution)
    elif control_type == "depth+pose":
        # For union model: apply both as separate patches (additive)
        depth_map = generate_depth_map(cropped_image, resolution=cn_resolution, ckpt_name=depth_ckpt)
        pose_map = generate_pose_map(cropped_image, resolution=cn_resolution)
        # Apply depth first, then pose on top
        patched = apply_zimage_control(model, model_patch, vae, depth_map, strength, mask)
        patched = apply_zimage_control(patched, model_patch, vae, pose_map, strength, mask)
        return patched
    else:
        raise ValueError(f"Unknown control_type: {control_type}")

    return apply_zimage_control(model, model_patch, vae, control_image, strength, mask)
