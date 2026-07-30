"""Render glyph-guidance examples to PNG for eyeballing.

Not a test — a visual artifact generator. Run it directly:

    "D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe" tests/artifacts/glyph_demo.py

Writes three PNGs next to this file, one per geometry case:

    glyph_demo_1_axis_aligned.png     a straight-on shopfront sign
    glyph_demo_2_rotated_30.png       the same sign rotated 30 degrees
    glyph_demo_3_perspective.png      a trapezoid quad seen at an angle

Each one runs the full chain: fake garbled sign -> estimate_text_colors ->
render_glyph_layer / warp_to_quad -> composite_glyph, so the output shows what the
diffusion model would receive as an init image.
"""

import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from nodes.utils.glyph import (  # noqa: E402
    composite_glyph,
    estimate_text_colors,
    mask_quad,
    quad_angle,
    quad_size,
    render_glyph_layer,
    render_text_block,
    resolve_font,
    warp_to_quad,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CANVAS_H = 420
CANVAS_W = 680
MIN_FILE_BYTES = 2048
MIN_PIXEL_STD = 10.0


def build_scene(seed: int) -> np.ndarray:
    """A synthetic wall: vertical gradient, brick-ish banding and film grain."""
    rng = np.random.RandomState(seed)
    gradient = np.linspace(0.42, 0.20, CANVAS_H, dtype=np.float32)[:, None]
    scene = np.repeat(gradient, CANVAS_W, axis=1)[..., None].repeat(3, axis=2)
    scene *= np.array([1.06, 0.94, 0.86], dtype=np.float32)  # warm brick tint

    for y in range(0, CANVAS_H, 24):
        scene[y:y + 2, :, :] *= 0.86
    for x in range(0, CANVAS_W, 52):
        scene[:, x:x + 2, :] *= 0.90

    scene += rng.normal(0.0, 0.02, scene.shape).astype(np.float32)
    return np.clip(scene, 0.0, 1.0).astype(np.float32)


def paint_garbled_sign(scene, quad, plate_rgb, ink_rgb, seed):
    """Paint a plate with unreadable pseudo-lettering — the "before" state."""
    rng = np.random.RandomState(seed)
    poly = np.asarray(quad).round().astype(np.int32)

    plate_layer = np.zeros_like(scene)
    plate_layer[:] = np.array(plate_rgb, dtype=np.float32) / 255.0
    coverage = np.zeros(scene.shape[:2], dtype=np.uint8)
    cv2.fillPoly(coverage, [poly], 255)
    alpha = (coverage.astype(np.float32) / 255.0) * 0.97
    scene = composite_glyph(scene, plate_layer, alpha)

    scratch = (scene * 255.0).astype(np.uint8)
    ink = tuple(int(c) for c in ink_rgb)
    centre = poly.mean(axis=0)
    span_x = float(np.abs(poly[:, 0] - centre[0]).max()) * 0.72
    span_y = float(np.abs(poly[:, 1] - centre[1]).max()) * 0.42
    for _ in range(26):
        x0 = int(centre[0] + rng.uniform(-span_x, span_x))
        y0 = int(centre[1] + rng.uniform(-span_y, span_y))
        x1 = x0 + int(rng.uniform(-16, 16))
        y1 = y0 + int(rng.uniform(-14, 14))
        cv2.line(scratch, (x0, y0), (x1, y1), ink, thickness=int(rng.randint(2, 5)))

    blurred = cv2.GaussianBlur(scratch, (5, 5), 0).astype(np.float32) / 255.0
    return composite_glyph(scene, blurred, alpha)


def save(name: str, rgb_float: np.ndarray) -> str:
    """Write a float32 0..1 RGB image as PNG, return its absolute path."""
    path = os.path.join(OUT_DIR, name)
    bgr = cv2.cvtColor((np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    return path


def label(image, text):
    """Burn a small caption into the top-left corner."""
    out = (image * 255.0).astype(np.uint8)
    cv2.rectangle(out, (0, 0), (CANVAS_W, 26), (18, 18, 18), -1)
    cv2.putText(out, text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return out.astype(np.float32) / 255.0


def demo_axis_aligned(font_path):
    """Straight-on fascia: mask -> quad -> glyph layer -> composite."""
    mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    mask[120:250, 90:590] = 1.0

    quad = mask_quad(mask)
    scene = paint_garbled_sign(build_scene(1), quad, (232, 226, 210), (40, 38, 44), seed=11)
    ink, plate = estimate_text_colors(scene, mask)

    glyph, alpha = render_glyph_layer(
        "HOTEL CENTRAL", mask, font_path=font_path, fill=ink, bg=plate, uppercase=True
    )
    out = composite_glyph(scene, glyph, alpha)
    caption = "axis-aligned  |  angle {:+.1f}deg  box {}x{}  ink {}  plate {}".format(
        quad_angle(quad), *quad_size(quad), ink, plate
    )
    return save("glyph_demo_1_axis_aligned.png", label(out, caption))


def demo_rotated(font_path):
    """Sign rotated 30 degrees counter-clockwise, built with cv2.warpAffine."""
    mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    mask[150:270, 100:580] = 1.0
    matrix = cv2.getRotationMatrix2D((CANVAS_W / 2.0, CANVAS_H / 2.0), 30.0, 0.82)
    mask = cv2.warpAffine(mask, matrix, (CANVAS_W, CANVAS_H), flags=cv2.INTER_NEAREST)

    quad = mask_quad(mask)
    scene = paint_garbled_sign(build_scene(2), quad, (46, 92, 138), (245, 240, 225), seed=22)
    ink, plate = estimate_text_colors(scene, mask)

    glyph, alpha = render_glyph_layer(
        "PLATFORM 9", mask, font_path=font_path, fill=ink, bg=plate, uppercase=True
    )
    out = composite_glyph(scene, glyph, alpha)
    caption = "rotated 30deg CCW  |  angle {:+.1f}deg  box {}x{}  ink {}  plate {}".format(
        quad_angle(quad), *quad_size(quad), ink, plate
    )
    return save("glyph_demo_2_rotated_30.png", label(out, caption))


def demo_perspective(font_path):
    """Hand-built trapezoid: warp_to_quad used directly, since minAreaRect would
    square the perspective away."""
    quad = np.array(
        [[95.0, 130.0], [595.0, 92.0], [560.0, 292.0], [130.0, 246.0]], dtype=np.float32
    )
    mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    cv2.fillPoly(mask, [quad.round().astype(np.int32)], 1.0)

    scene = paint_garbled_sign(build_scene(3), quad, (222, 84, 52), (250, 246, 238), seed=33)
    ink, plate = estimate_text_colors(scene, mask)

    block_w, block_h = quad_size(quad)
    block = render_text_block(
        "OPEN 24 HOURS", block_w, block_h, font_path=font_path,
        fill=ink, bg=plate, uppercase=True,
    )
    glyph, alpha = warp_to_quad(block, quad, (CANVAS_H, CANVAS_W))
    out = composite_glyph(scene, glyph, alpha)
    caption = "perspective quad  |  angle {:+.1f}deg  box {}x{}  ink {}  plate {}".format(
        quad_angle(quad), block_w, block_h, ink, plate
    )
    return save("glyph_demo_3_perspective.png", label(out, caption))


def main():
    font_path = resolve_font("bold condensed sans")
    print("font: {}".format(font_path or "<PIL default>"))

    paths = [
        demo_axis_aligned(font_path),
        demo_rotated(font_path),
        demo_perspective(font_path),
    ]

    failures = []
    print()
    for path in paths:
        size = os.path.getsize(path)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        std = float(image.std()) if image is not None else -1.0
        ok = image is not None and size > MIN_FILE_BYTES and std > MIN_PIXEL_STD
        print("{}  {:>8} bytes  std {:6.2f}  {}".format(path, size, std, "OK" if ok else "FAIL"))
        if not ok:
            failures.append(path)

    if failures:
        raise SystemExit("non-trivial PNG check failed for: {}".format(failures))
    print("\nall {} artifacts written to {}".format(len(paths), OUT_DIR))


if __name__ == "__main__":
    main()
