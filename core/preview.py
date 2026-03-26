"""Generate palette swatch preview images as ComfyUI IMAGE tensors."""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .color_utils import hsl_to_rgb


def render_palette_preview(colors, width=512, height=128):
    """
    Render a horizontal swatch image from a list of palette color dicts.

    Args:
        colors: list of dicts, each with at least {"name": str, "hsl": (h, s, l)} or {"rgb": (r, g, b)}
        width: image width in pixels
        height: image height in pixels

    Returns:
        torch.Tensor of shape [1, H, W, 3], dtype float32, range [0, 1]
    """
    if not colors:
        # Return blank white image
        blank = torch.ones(1, height, width, 3, dtype=torch.float32)
        return blank

    img = Image.new("RGB", (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(img)

    n = len(colors)
    block_w = width // n
    remainder = width - (block_w * n)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, c in enumerate(colors):
        # Get RGB
        if "rgb" in c:
            r, g, b = c["rgb"]
        elif "hsl" in c:
            r, g, b = hsl_to_rgb(*c["hsl"])
        else:
            r, g, b = 0, 0, 0

        # Last block gets remainder pixels
        x0 = i * block_w
        x1 = x0 + block_w + (remainder if i == n - 1 else 0)

        # Draw color swatch (top 75%)
        swatch_h = int(height * 0.75)
        draw.rectangle([x0, 0, x1, swatch_h], fill=(r, g, b))

        # Draw name label (bottom 25%)
        name = c.get("name", "")
        if font and name:
            # Center text
            text_y = swatch_h + 4
            text_x = x0 + 4
            # Choose text color for readability
            draw.text((text_x, text_y), name, fill=(200, 200, 200), font=font)

    # Convert to ComfyUI IMAGE tensor [1, H, W, 3] float32
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def render_source_annotated(original_tensor, colors, swatch_size=30, margin=10):
    """
    Overlay small palette swatches on the bottom-left of the original image.

    Args:
        original_tensor: ComfyUI IMAGE [B, H, W, C] float32 0-1
        colors: list of palette color dicts
        swatch_size: size of each swatch square
        margin: margin from image edge

    Returns:
        torch.Tensor [1, H, W, 3] float32 0-1
    """
    if not colors:
        return original_tensor[:1].clone()

    img_np = (original_tensor[0].cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)

    h = pil_img.height
    y = h - swatch_size - margin
    x = margin

    for c in colors:
        if "rgb" in c:
            r, g, b = c["rgb"]
        elif "hsl" in c:
            r, g, b = hsl_to_rgb(*c["hsl"])
        else:
            r, g, b = 0, 0, 0

        # White border
        draw.rectangle([x - 1, y - 1, x + swatch_size + 1, y + swatch_size + 1],
                        outline=(255, 255, 255))
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=(r, g, b))
        x += swatch_size + 4

    result = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(result).unsqueeze(0)
