"""LoRA utilities with auto-conversion for Z-Image Turbo (Lumina2) models.

Z-Image Turbo uses fused QKV attention layers instead of separate to_q/to_k/to_v.
LoRAs trained with diffusers-style trainers produce separate Q/K/V weights that
must be concatenated into the fused format for Z-Image Turbo compatibility.

QKV conversion logic based on Comfyui-ZiT-Lora-loader by capitan01R (MIT License).
https://github.com/capitan01R/Comfyui-ZiT-Lora-loader
"""

import torch
import comfy.model_base


def is_z_image_turbo(model):
    """Check if model uses Z-Image Turbo (Lumina2) architecture."""
    return isinstance(model.model, comfy.model_base.Lumina2)


def needs_qkv_conversion(lora_sd):
    """Check if LoRA has separate to_q/to_k/to_v that need fusing for Z-Image Turbo."""
    return any(
        ".to_q.lora_A" in k or ".to_k.lora_A" in k or ".to_v.lora_A" in k
        for k in lora_sd
    )


def convert_qkv_lora(lora_sd):
    """Convert separate to_q/to_k/to_v LoRA weights to fused qkv format.

    Z-Image Turbo uses fused attention.qkv layers with shape [dim*3, dim].
    Standard diffusers LoRAs have separate to_q, to_k, to_v projections.
    This function concatenates them into the fused format.

    Returns:
        New state dict with converted keys.
    """
    converted = {}
    processed_bases = set()
    qkv_keys = set()  # track all keys that belong to Q/K/V groups

    # First pass: identify all attention bases that need conversion
    for key in lora_sd:
        for component in ("to_q", "to_k", "to_v"):
            marker = f".{component}."
            if marker in key:
                base = key.split(marker)[0]
                processed_bases.add(base)
                qkv_keys.add(key)

    # Second pass: convert each base
    for base in processed_bases:
        # Gather Q/K/V tensors
        q_down = lora_sd.get(f"{base}.to_q.lora_A.weight")
        k_down = lora_sd.get(f"{base}.to_k.lora_A.weight")
        v_down = lora_sd.get(f"{base}.to_v.lora_A.weight")
        q_up = lora_sd.get(f"{base}.to_q.lora_B.weight")
        k_up = lora_sd.get(f"{base}.to_k.lora_B.weight")
        v_up = lora_sd.get(f"{base}.to_v.lora_B.weight")

        if all(t is not None for t in [q_down, k_down, v_down, q_up, k_up, v_up]):
            # Fuse into QKV format with block-diagonal lora_B
            #
            # lora_A (down projection): simple concat along dim=0
            #   q_down: [rank, dim], k_down: [rank, dim], v_down: [rank, dim]
            #   → [rank*3, dim]
            #
            # lora_B (up projection): block-diagonal so inner dims match
            #   q_up: [dim, rank], k_up: [dim, rank], v_up: [dim, rank]
            #   → [dim*3, rank*3] with Q/K/V on the diagonal
            #
            # Result: lora_B @ lora_A = [dim*3, rank*3] @ [rank*3, dim] = [dim*3, dim] ✓
            rank = q_down.shape[0]
            dim = q_down.shape[1]

            converted[f"{base}.qkv.lora_A.weight"] = torch.cat([q_down, k_down, v_down], dim=0)

            # Block-diagonal lora_B: each Q/K/V occupies its own block
            lora_B = torch.zeros(dim * 3, rank * 3, dtype=q_up.dtype, device=q_up.device)
            lora_B[0:dim, 0:rank] = q_up
            lora_B[dim:dim*2, rank:rank*2] = k_up
            lora_B[dim*2:dim*3, rank*2:rank*3] = v_up
            converted[f"{base}.qkv.lora_B.weight"] = lora_B

            print(f"[FVMTools] QKV fused: {base} rank={rank} dim={dim} "
                  f"lora_A=[{rank*3},{dim}] lora_B=[{dim*3},{rank*3}]")

            # Average alpha values if present
            alphas = []
            for comp in ("to_q", "to_k", "to_v"):
                alpha_key = f"{base}.{comp}.alpha"
                if alpha_key in lora_sd:
                    alphas.append(lora_sd[alpha_key])
                    qkv_keys.add(alpha_key)
            if alphas:
                converted[f"{base}.qkv.alpha"] = sum(alphas) / len(alphas)

    # Third pass: handle remaining keys
    for key in lora_sd:
        if key in qkv_keys:
            continue  # already processed

        # Remap to_out.0 -> out (diffusers -> Z-Image Turbo naming)
        if ".to_out.0." in key:
            new_key = key.replace(".to_out.0.", ".out.")
            converted[new_key] = lora_sd[key]
        else:
            converted[key] = lora_sd[key]

    return converted
