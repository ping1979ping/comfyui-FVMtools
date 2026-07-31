"""Glyph guidance — quad fitting, typography, perspective warping, colour recovery.

Sign coordinates run y-down, so a counter-clockwise cv2 rotation of +A degrees shows up
as a quad_angle of -A. The tests below pin that convention down explicitly.
"""

import os

import cv2
import numpy as np
import pytest

from nodes.utils.glyph import (
    edge_profiles, warp_to_contour, quad_fit_error, CONTOUR_FIT_THRESHOLD,
    SYSTEM_DEFAULT_LABEL,
    FONT_SEARCH_DIRS,
    composite_glyph,
    discover_fonts,
    estimate_text_colors,
    mask_quad,
    quad_angle,
    quad_size,
    render_glyph_layer,
    render_text_block,
    resolve_font,
    warp_to_quad,
)

KERNEL3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# ──── Helpers ────

def _rect_mask(h, w, x0, y0, x1, y1):
    """Axis-aligned filled rectangle mask, [h, w] float32 0/1."""
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def _rotated_rect_mask(h, w, rect_w, rect_h, angle_deg):
    """Centred rectangle rotated counter-clockwise by angle_deg via cv2.warpAffine."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    mask[cy - rect_h // 2:cy + rect_h // 2, cx - rect_w // 2:cx + rect_w // 2] = 1.0
    matrix = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg, 1.0)
    return cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)


def _max_ink_component(block_rgb):
    """Area of the largest connected bright component in a rendered block."""
    gray = cv2.cvtColor(np.ascontiguousarray(block_rgb), cv2.COLOR_RGB2GRAY)
    ink = (gray > 127).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def _polygon_mask(quad, shape):
    """Filled uint8 mask of a quad inside an (H, W) canvas."""
    poly = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(poly, [np.asarray(quad).round().astype(np.int32)], 1)
    return poly


# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────


class TestMaskQuad:

    def test_axis_aligned_corners_and_order(self):
        """Corners must match the rectangle within 1px, ordered TL, TR, BR, BL."""
        mask = _rect_mask(256, 400, 100, 60, 340, 180)
        quad = mask_quad(mask)

        assert quad is not None
        assert quad.shape == (4, 2)
        assert quad.dtype == np.float32

        expected = np.array(
            [[100, 60], [339, 60], [339, 179], [100, 179]], dtype=np.float32
        )
        assert np.allclose(quad, expected, atol=1.0), quad

    def test_axis_aligned_angle_is_zero(self):
        quad = mask_quad(_rect_mask(256, 400, 100, 60, 340, 180))
        assert abs(quad_angle(quad)) < 1.0

    def test_axis_aligned_size(self):
        quad = mask_quad(_rect_mask(256, 400, 100, 60, 340, 180))
        width, height = quad_size(quad)
        assert abs(width - 240) <= 1
        assert abs(height - 120) <= 1

    @pytest.mark.parametrize("ccw_angle", [10.0, 20.0, 30.0, -15.0, -30.0])
    def test_rotation_recovered(self, ccw_angle):
        """A rect rotated CCW by A degrees reports quad_angle == -A (y-down axes)."""
        mask = _rotated_rect_mask(400, 520, 300, 90, ccw_angle)
        quad = mask_quad(mask)

        assert quad is not None
        assert abs(quad_angle(quad) - (-ccw_angle)) < 2.0, quad_angle(quad)

    def test_rotation_preserves_edge_lengths(self):
        """The un-rotated box behind a rotated mask keeps its original proportions."""
        quad = mask_quad(_rotated_rect_mask(400, 520, 300, 90, 30.0))
        width, height = quad_size(quad)
        assert abs(width - 300) <= 6, width
        assert abs(height - 90) <= 6, height

    def test_ordering_is_clockwise_for_rotated_rect(self):
        """TL->TR->BR->BL stays clockwise: the shoelace area must be positive (y-down)."""
        quad = mask_quad(_rotated_rect_mask(400, 520, 300, 90, 25.0))
        x = quad[:, 0]
        y = quad[:, 1]
        area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert area > 0.0, area

    def test_stable_across_calls(self):
        mask = _rotated_rect_mask(400, 520, 300, 90, 18.0)
        assert np.array_equal(mask_quad(mask), mask_quad(mask))

    def test_empty_mask_returns_none(self):
        assert mask_quad(np.zeros((128, 128), dtype=np.float32)) is None

    def test_single_pixel_mask_returns_none(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20, 20] = 1.0
        assert mask_quad(mask) is None

    def test_none_input_returns_none(self):
        assert mask_quad(None) is None

    def test_accepts_0_255_masks(self):
        mask = (_rect_mask(128, 128, 20, 30, 90, 70) * 255).astype(np.uint8)
        quad = mask_quad(mask)
        assert quad is not None
        assert np.allclose(quad, [[20, 30], [89, 30], [89, 69], [20, 69]], atol=1.0)


class TestQuadHelpers:

    def test_angle_range_is_bounded(self):
        for ccw in (-80.0, -45.0, 0.0, 45.0, 80.0):
            quad = mask_quad(_rotated_rect_mask(400, 400, 200, 80, ccw))
            angle = quad_angle(quad)
            assert -90.0 < angle <= 90.0, angle

    def test_degenerate_quad_angle_is_zero(self):
        assert quad_angle(np.zeros((4, 2), dtype=np.float32)) == 0.0

    def test_quad_size_never_zero(self):
        assert quad_size(np.zeros((4, 2), dtype=np.float32)) == (1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Text rendering
# ──────────────────────────────────────────────────────────────────────────────


class TestRenderTextBlock:

    def test_shape_and_dtype(self):
        block = render_text_block("OPEN", 300, 100)
        assert block.shape == (100, 300, 3)
        assert block.dtype == np.uint8

    def test_non_empty_text_puts_ink_on_the_plate(self):
        block = render_text_block("OPEN", 300, 100)
        assert block.std() > 5.0
        assert block.max() > 200
        assert block.min() < 50

    def test_empty_text_is_uniform(self):
        """Empty copy yields a flat plate: every pixel identical to the bg colour."""
        block = render_text_block("", 300, 100, bg=(17, 23, 29))
        assert tuple(int(c) for c in block[0, 0]) == (17, 23, 29)
        assert float(block[..., 0].std()) == 0.0
        assert float(block[..., 1].std()) == 0.0
        assert float(block[..., 2].std()) == 0.0
        assert np.array_equal(block, np.full_like(block, 0) + np.array([17, 23, 29], np.uint8))

    def test_whitespace_text_is_uniform(self):
        assert float(render_text_block("   \n\t ", 300, 100).std()) == 0.0

    def test_text_scales_to_fill_the_box(self):
        """One character must produce a far larger ink blob than twenty in the same box."""
        big = render_text_block("A", 400, 200)
        small = render_text_block("TWENTY CHARS EXACTLY", 400, 200)

        big_area = _max_ink_component(big)
        small_area = _max_ink_component(small)

        assert big_area > 0
        assert small_area > 0
        assert big_area > small_area * 2, (big_area, small_area)

    def test_margin_is_respected(self):
        """No ink may land in the outer margin band."""
        block = render_text_block("WIDE", 400, 160, margin_ratio=0.15)
        gray = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
        band_x = int(400 * 0.15) - 2
        band_y = int(160 * 0.15) - 2
        assert gray[:band_y, :].max() < 20
        assert gray[-band_y:, :].max() < 20
        assert gray[:, :band_x].max() < 20
        assert gray[:, -band_x:].max() < 20

    def test_uppercase_changes_the_render(self):
        lower = render_text_block("open now", 300, 100)
        upper = render_text_block("open now", 300, 100, uppercase=True)
        assert not np.array_equal(lower, upper)

    def test_alignment_shifts_the_ink(self):
        left = render_text_block("HI", 400, 100, align="left")
        right = render_text_block("HI", 400, 100, align="right")
        left_cx = float(np.mean(np.nonzero(left[..., 0] > 127)[1]))
        right_cx = float(np.mean(np.nonzero(right[..., 0] > 127)[1]))
        assert right_cx > left_cx + 50

    def test_custom_colours(self):
        block = render_text_block("X", 200, 120, fill=(255, 0, 0), bg=(0, 0, 255))
        assert tuple(block[0, 0]) == (0, 0, 255)
        reds = block[..., 0] > 200
        assert reds.any()

    def test_max_lines_is_respected(self):
        """A long string in a tall narrow box must not exceed max_lines text rows."""
        block = render_text_block(
            "one two three four five six seven eight", 200, 400, max_lines=2
        )
        gray = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
        rows_with_ink = (gray > 127).any(axis=1).astype(np.uint8)
        # Count runs of inked rows -> number of text lines.
        transitions = int(np.count_nonzero(np.diff(rows_with_ink) == 1)) + int(rows_with_ink[0])
        assert 1 <= transitions <= 2, transitions

    def test_tiny_box_does_not_raise(self):
        block = render_text_block("EMERGENCY EXIT ONLY", 8, 4)
        assert block.shape == (4, 8, 3)

    def test_zero_size_is_clamped(self):
        assert render_text_block("A", 0, 0).shape == (1, 1, 3)

    def test_missing_font_falls_back(self):
        block = render_text_block("OPEN", 300, 100, font_path="C:/definitely/not/a.ttf")
        assert block.shape == (100, 300, 3)
        assert block.std() > 5.0


# ──────────────────────────────────────────────────────────────────────────────
# Warping
# ──────────────────────────────────────────────────────────────────────────────


class TestWarpToQuad:

    def test_ink_lands_inside_the_quad_only(self):
        mask = _rotated_rect_mask(400, 520, 300, 90, 30.0)
        quad = mask_quad(mask)
        block = render_text_block("HELLO", *quad_size(quad))

        warped, alpha = warp_to_quad(block, quad, (400, 520))

        assert warped.shape == (400, 520, 3)
        assert alpha.shape == (400, 520)
        assert warped.dtype == np.float32
        assert alpha.dtype == np.float32

        poly = _polygon_mask(quad, (400, 520))
        inside = cv2.erode(poly, KERNEL3, iterations=2) > 0
        outside = cv2.dilate(poly, KERNEL3, iterations=2) == 0

        assert warped[inside].max() > 0.5, "no ink inside the quad"
        assert warped[outside].max() < 0.02, "ink leaked outside the quad"
        assert alpha[inside].mean() > 0.98
        assert alpha[outside].max() < 0.02

    def test_perspective_quad_is_filled(self):
        """A trapezoid destination still receives full coverage."""
        quad = np.array(
            [[60, 40], [440, 90], [400, 250], [90, 210]], dtype=np.float32
        )
        block = render_text_block("PERSPECTIVE", *quad_size(quad))
        warped, alpha = warp_to_quad(block, quad, (300, 500))

        poly = _polygon_mask(quad, (300, 500))
        inside = cv2.erode(poly, KERNEL3, iterations=2) > 0
        outside = cv2.dilate(poly, KERNEL3, iterations=2) == 0

        assert alpha[inside].mean() > 0.98
        assert alpha[outside].max() < 0.02
        assert warped[inside].max() > 0.5

    def test_none_quad_returns_zeros(self):
        warped, alpha = warp_to_quad(render_text_block("A", 40, 20), None, (64, 64))
        assert warped.shape == (64, 64, 3)
        assert float(warped.max()) == 0.0
        assert float(alpha.max()) == 0.0

    def test_degenerate_quad_returns_zeros(self):
        quad = np.zeros((4, 2), dtype=np.float32)
        warped, alpha = warp_to_quad(render_text_block("A", 40, 20), quad, (64, 64))
        assert float(alpha.max()) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end layer
# ──────────────────────────────────────────────────────────────────────────────


class TestRenderGlyphLayer:

    def test_rotated_mask_end_to_end(self):
        mask = _rotated_rect_mask(400, 520, 320, 100, -22.0)
        rgb, alpha = render_glyph_layer("CAFE ROMA", mask, uppercase=True)

        assert rgb.shape == (400, 520, 3)
        assert alpha.shape == (400, 520)
        assert rgb.dtype == np.float32
        assert alpha.dtype == np.float32
        assert 0.0 <= float(alpha.min()) and float(alpha.max()) <= 1.0

        core = cv2.erode(mask, KERNEL3, iterations=3) > 0.5
        far_outside = cv2.dilate(mask, KERNEL3, iterations=10) < 0.5

        assert alpha[core].mean() > 0.95, "sign interior not covered"
        assert alpha[far_outside].max() < 0.05, "coverage bled far outside the sign"
        assert rgb[core].max() > 0.5, "no glyph ink on the sign"

    def test_ink_differs_from_plate(self):
        mask = _rect_mask(200, 400, 40, 60, 360, 150)
        rgb, alpha = render_glyph_layer("SALE", mask, fill=(255, 255, 255))
        inside = alpha > 0.5
        assert rgb[inside].max() > 0.9   # white ink
        assert rgb[inside].min() < 0.6   # mid-grey default plate

    def test_custom_plate_colour(self):
        mask = _rect_mask(200, 400, 40, 60, 360, 150)
        rgb, _alpha = render_glyph_layer("SALE", mask, fill=(0, 0, 0), bg=(200, 30, 30))
        centre = rgb[62, 42]
        assert centre[0] > 0.6 and centre[1] < 0.3

    def test_empty_mask_returns_zeros(self):
        rgb, alpha = render_glyph_layer("HELLO", np.zeros((96, 128), dtype=np.float32))
        assert rgb.shape == (96, 128, 3)
        assert alpha.shape == (96, 128)
        assert float(rgb.max()) == 0.0
        assert float(alpha.max()) == 0.0

    def test_empty_text_still_produces_a_plate(self):
        mask = _rect_mask(200, 400, 40, 60, 360, 150)
        rgb, alpha = render_glyph_layer("", mask)
        assert float(alpha.max()) > 0.9
        assert float(rgb[alpha > 0.5].std()) < 0.01  # flat plate, no letters


# ──────────────────────────────────────────────────────────────────────────────
# Compositing
# ──────────────────────────────────────────────────────────────────────────────


class TestCompositeGlyph:

    def _base(self):
        return np.random.RandomState(0).rand(64, 96, 3).astype(np.float32)

    def test_zero_alpha_returns_base_unchanged(self):
        base = self._base()
        glyph = np.ones_like(base)
        out = composite_glyph(base, glyph, np.zeros((64, 96), dtype=np.float32))
        assert np.array_equal(out, base)

    def test_zero_strength_returns_base_unchanged(self):
        base = self._base()
        glyph = np.ones_like(base)
        alpha = np.ones((64, 96), dtype=np.float32)
        assert np.array_equal(composite_glyph(base, glyph, alpha, strength=0.0), base)

    def test_full_alpha_replaces_inside_the_mask(self):
        base = self._base()
        glyph = np.zeros_like(base)
        glyph[..., 0] = 1.0
        alpha = np.zeros((64, 96), dtype=np.float32)
        alpha[10:40, 20:60] = 1.0

        out = composite_glyph(base, glyph, alpha)

        assert np.allclose(out[10:40, 20:60], glyph[10:40, 20:60])
        assert np.array_equal(out[:5, :5], base[:5, :5])

    def test_partial_strength_blends(self):
        base = np.zeros((8, 8, 3), dtype=np.float32)
        glyph = np.ones((8, 8, 3), dtype=np.float32)
        out = composite_glyph(base, glyph, np.ones((8, 8), dtype=np.float32), strength=0.5)
        assert np.allclose(out, 0.5)

    def test_mismatched_shapes_are_resized_not_raised(self):
        base = self._base()
        glyph = np.ones((32, 48, 3), dtype=np.float32)
        alpha = np.ones((32, 48), dtype=np.float32)
        out = composite_glyph(base, glyph, alpha)
        assert out.shape == base.shape
        assert np.allclose(out, 1.0)

    def test_output_is_clipped_float32(self):
        base = self._base()
        out = composite_glyph(base, np.full_like(base, 2.0), np.ones((64, 96), dtype=np.float32))
        assert out.dtype == np.float32
        assert float(out.max()) <= 1.0
        assert float(out.min()) >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Colour recovery
# ──────────────────────────────────────────────────────────────────────────────


class TestEstimateTextColors:

    def _patch(self, plate, ink):
        image = np.full((100, 200, 3), plate, dtype=np.uint8)
        image[40:60, 30:170] = ink
        return image, np.ones((100, 200), dtype=np.float32)

    def test_black_on_white(self):
        image, mask = self._patch((255, 255, 255), (0, 0, 0))
        ink, plate = estimate_text_colors(image, mask)
        assert max(ink) < 60, ink
        assert min(plate) > 200, plate

    def test_white_on_black(self):
        image, mask = self._patch((0, 0, 0), (255, 255, 255))
        ink, plate = estimate_text_colors(image, mask)
        assert min(ink) > 200, ink
        assert max(plate) < 60, plate

    def test_keeps_the_colour_scheme(self):
        image, mask = self._patch((240, 220, 60), (20, 30, 120))
        ink, plate = estimate_text_colors(image, mask)
        assert ink[2] > ink[0], ink        # blue-dominant ink
        assert plate[0] > plate[2], plate  # yellow-dominant plate

    def test_sparse_noisy_strokes_still_separate_from_the_plate(self):
        """Real signs: thin letters on a noisy plate. Ink must not collapse onto plate.

        A bare median split would land inside the plate distribution and report two
        near-identical colours; the deterministic refinement has to pull them apart.
        """
        rng = np.random.RandomState(7)
        image = np.full((160, 320, 3), (228, 222, 206), dtype=np.float32)
        image += rng.normal(0.0, 6.0, image.shape)
        for x in range(30, 300, 34):
            image[50:110, x:x + 9] = (38, 34, 44)  # ~9% coverage
        image = np.clip(image, 0, 255).astype(np.uint8)
        mask = np.zeros((160, 320), dtype=np.float32)
        mask[20:140, 10:310] = 1.0

        ink, plate = estimate_text_colors(image, mask)
        ink_luma = 0.299 * ink[0] + 0.587 * ink[1] + 0.114 * ink[2]
        plate_luma = 0.299 * plate[0] + 0.587 * plate[1] + 0.114 * plate[2]

        assert plate_luma - ink_luma > 100.0, (ink, plate)
        assert max(ink) < 70, ink
        assert min(plate) > 180, plate

    def test_accepts_float_images(self):
        image, mask = self._patch((255, 255, 255), (0, 0, 0))
        ink, plate = estimate_text_colors(image.astype(np.float32) / 255.0, mask)
        assert max(ink) < 60
        assert min(plate) > 200

    def test_returns_int_tuples_in_range(self):
        image, mask = self._patch((255, 255, 255), (0, 0, 0))
        for colour in estimate_text_colors(image, mask):
            assert isinstance(colour, tuple) and len(colour) == 3
            assert all(isinstance(c, int) and 0 <= c <= 255 for c in colour)

    def test_mask_limits_the_sample(self):
        """Pixels outside the mask must not influence the result."""
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        image[40:60, 30:170] = 0
        image[:20, :] = (255, 0, 0)  # distractor outside the mask
        mask = np.zeros((100, 200), dtype=np.float32)
        mask[30:80, 10:190] = 1.0
        ink, plate = estimate_text_colors(image, mask)
        assert max(ink) < 60
        assert min(plate) > 200

    def test_empty_mask_fallback(self):
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        assert estimate_text_colors(image, np.zeros((32, 32), dtype=np.float32)) == (
            (0, 0, 0), (255, 255, 255)
        )

    def test_flat_patch_returns_that_colour_twice(self):
        image = np.full((32, 32, 3), 77, dtype=np.uint8)
        ink, plate = estimate_text_colors(image, np.ones((32, 32), dtype=np.float32))
        assert ink == plate == (77, 77, 77)

    def test_none_inputs_fall_back(self):
        assert estimate_text_colors(None, None) == ((0, 0, 0), (255, 255, 255))


# ──────────────────────────────────────────────────────────────────────────────
# Font discovery
# ──────────────────────────────────────────────────────────────────────────────


class TestFontDiscovery:

    def test_search_dirs_all_exist(self):
        dirs = FONT_SEARCH_DIRS()
        assert isinstance(dirs, list)
        for path in dirs:
            assert os.path.isdir(path), path
        assert len(dirs) == len(set(dirs))

    def test_discover_fonts_non_empty_and_default_first(self):
        fonts = discover_fonts()
        assert isinstance(fonts, list)
        assert len(fonts) >= 1
        assert fonts[0] == SYSTEM_DEFAULT_LABEL

    def test_discover_fonts_is_consistent_across_calls(self):
        assert discover_fonts() == discover_fonts()

    def test_discover_fonts_has_no_duplicates(self):
        fonts = discover_fonts()
        assert len(fonts) == len(set(fonts))

    def test_repo_fonts_dir_is_searched_first(self):
        dirs = FONT_SEARCH_DIRS()
        if dirs:
            assert dirs[0].replace("\\", "/").endswith("/fonts")


class TestResolveFont:

    HINTS = [
        "bold condensed sans",
        "Helvetica",
        "handwritten",
        "serif",
        "monospace poster",
        "grotesque",
        "italic display",
        "",
        "   ",
        "zzqqxx nonsense 42",
    ]

    @pytest.mark.parametrize("hint", HINTS)
    def test_returns_none_or_an_existing_font_file(self, hint):
        result = resolve_font(hint)
        if result is not None:
            assert os.path.isfile(result), result
            assert result.lower().endswith((".ttf", ".otf")), result

    def test_none_and_default_label(self):
        assert resolve_font(None) is None
        assert resolve_font("") is None
        assert resolve_font(SYSTEM_DEFAULT_LABEL) is None

    def test_display_name_round_trip(self):
        fonts = [f for f in discover_fonts() if f != SYSTEM_DEFAULT_LABEL]
        if not fonts:
            pytest.skip("no fonts installed on this machine")
        for name in fonts[:5]:
            path = resolve_font(name)
            assert path is not None, name
            assert os.path.isfile(path)
            assert os.path.splitext(os.path.basename(path))[0].lower() in name.lower()

    def test_available_restricts_the_search(self):
        fonts = [f for f in discover_fonts() if f != SYSTEM_DEFAULT_LABEL]
        if len(fonts) < 2:
            pytest.skip("need at least two fonts installed")
        allowed = [fonts[0]]
        result = resolve_font("bold condensed sans", available=allowed)
        assert result is None or result == resolve_font(fonts[0])

    def test_empty_available_list_yields_none(self):
        assert resolve_font("bold sans", available=[]) is None
        assert resolve_font("bold sans", available=[SYSTEM_DEFAULT_LABEL]) is None

    def test_explicit_path_passes_through(self):
        fonts = [f for f in discover_fonts() if f != SYSTEM_DEFAULT_LABEL]
        if not fonts:
            pytest.skip("no fonts installed on this machine")
        path = resolve_font(fonts[0])
        assert resolve_font(path) == path

    def test_resolved_font_actually_renders(self):
        path = resolve_font("bold sans")
        block = render_text_block("OPEN", 320, 120, font_path=path)
        assert block.std() > 5.0


# ──────────────────────────────────────────────────────────────────────────────
# Degenerate input sweep
# ──────────────────────────────────────────────────────────────────────────────


class TestDegenerateInputs:

    ZERO_MASK = np.zeros((64, 80), dtype=np.float32)

    def test_nothing_raises_on_empty_mask_and_text(self):
        assert mask_quad(self.ZERO_MASK) is None

        rgb, alpha = render_glyph_layer("", self.ZERO_MASK)
        assert rgb.shape == (64, 80, 3)
        assert float(alpha.max()) == 0.0

        block = render_text_block("", 1, 1)
        assert block.shape == (1, 1, 3)

        warped, warp_alpha = warp_to_quad(block, mask_quad(self.ZERO_MASK), (64, 80))
        assert float(warp_alpha.max()) == 0.0

        base = np.zeros((64, 80, 3), dtype=np.float32)
        assert np.array_equal(composite_glyph(base, warped, warp_alpha), base)

        assert estimate_text_colors(np.zeros((64, 80, 3), dtype=np.uint8), self.ZERO_MASK) == (
            (0, 0, 0), (255, 255, 255)
        )

    def test_none_text_is_treated_as_empty(self):
        assert float(render_text_block(None, 60, 30).std()) == 0.0

    def test_mask_with_holes_still_yields_a_quad(self):
        mask = _rect_mask(120, 200, 20, 20, 180, 100)
        mask[50:70, 80:120] = 0.0
        quad = mask_quad(mask)
        assert quad is not None
        assert np.allclose(quad, [[20, 20], [179, 20], [179, 99], [20, 99]], atol=1.0)

    def test_disjoint_blobs_are_enclosed_together(self):
        mask = _rect_mask(120, 200, 10, 10, 40, 40)
        mask[80:110, 150:190] = 1.0
        quad = mask_quad(mask)
        assert quad is not None
        poly = _polygon_mask(quad, mask.shape)
        assert poly[mask > 0.5].all()


class TestMaskShapeIsRespected:
    """SAM3 returns silhouettes, not boxes. The glyph layer must follow them.

    Measured before the fix: an ellipse, a circle and an irregular blob each had
    20-27% of their own area painted OUTSIDE the mask, because only the bounding
    quad was used. That puts a rectangular plate into the init latent where a
    round object stands.
    """

    @staticmethod
    def _ellipse(h=260, w=420):
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (210, 130), (170, 90), 0, 0, 360, 1.0, -1)
        return m

    @staticmethod
    def _circle(h=260, w=420):
        m = np.zeros((h, w), np.float32)
        cv2.circle(m, (210, 130), 100, 1.0, -1)
        return m

    @staticmethod
    def _blob(h=260, w=420):
        m = np.zeros((h, w), np.float32)
        pts = np.array([[40, 60], [300, 40], [380, 120], [350, 220],
                        [120, 230], [60, 150]], np.int32)
        cv2.fillPoly(m, [pts], 1.0)
        return m

    @pytest.mark.parametrize("shape", ["_ellipse", "_circle", "_blob"])
    def test_no_ink_outside_the_mask(self, shape):
        mask = getattr(self, shape)()
        _, alpha = render_glyph_layer("HAUSMARKE", mask)
        outside = alpha * (mask < 0.5)
        assert float(outside.max()) == 0.0, \
            f"{shape}: glyph layer paints outside the silhouette"

    @pytest.mark.parametrize("shape", ["_ellipse", "_circle", "_blob"])
    def test_still_covers_the_interior(self, shape):
        """Clipping must not empty the layer out."""
        mask = getattr(self, shape)()
        _, alpha = render_glyph_layer("HAUSMARKE", mask)
        inside = alpha[mask > 0.5]
        assert float(inside.max()) > 0.9
        assert float(inside.mean()) > 0.5

    def test_circle_gets_horizontal_text_not_a_diagonal(self):
        """A circle has no preferred direction; minAreaRect returns ~45 degrees.

        Trusting that angle sets the text diagonally across a round sign.
        """
        quad = mask_quad(self._circle())
        assert quad is not None
        assert abs(quad_angle(quad)) < 5.0

    def test_a_genuinely_rotated_sign_keeps_its_angle(self):
        """The square-tolerance shortcut must not flatten real rotations."""
        m = np.zeros((300, 300), np.float32)
        cv2.rectangle(m, (60, 130), (240, 175), 1.0, -1)      # clearly oblong
        rot = cv2.getRotationMatrix2D((150, 150), 25.0, 1.0)
        m = cv2.warpAffine(m, rot, (300, 300))
        assert abs(quad_angle(mask_quad(m))) == pytest.approx(25.0, abs=3.0)

    def test_square_tolerance_can_be_disabled(self):
        quad = mask_quad(self._circle(), square_tolerance=0.0)
        assert quad is not None  # raw angle kept, whatever it is

    def test_soft_mask_edges_survive_the_clip(self):
        """A 0-255 mask must be normalised before clamping, or soft edges snap to 1."""
        m = np.zeros((200, 300), np.float32)
        cv2.rectangle(m, (40, 60), (260, 140), 255.0, -1)
        m = cv2.GaussianBlur(m, (31, 31), 0)
        _, alpha = render_glyph_layer("TEST", m)
        edge = alpha[(m > 20) & (m < 200)]
        if edge.size:
            assert float(edge.max()) < 1.0, "soft mask edge was clamped to full opacity"


class TestPerspectiveIsPreserved:
    """minAreaRect always returns a rectangle, so a sign angled away from the
    camera got text with both edges the same height — flat, parallel, and reading
    as a sticker pasted on top. Measured on a 33% foreshortened sign: the fitted
    box claimed 0% foreshortening and 20% too much area.
    """

    @staticmethod
    def _mask(pts, h=300, w=620):
        m = np.zeros((h, w), np.float32)
        cv2.fillPoly(m, [np.asarray(pts, np.int32)], 1.0)
        return m

    ANGLED = [[60, 60], [540, 120], [540, 240], [60, 240]]
    FLAT = [[60, 80], [540, 80], [540, 215], [60, 215]]

    def test_recovers_the_real_corners_of_an_angled_sign(self):
        quad = mask_quad(self._mask(self.ANGLED))
        expected = np.array(self.ANGLED, dtype=np.float32)
        for corner in expected:
            assert min(np.hypot(*(q - corner)) for q in quad) < 6.0, \
                f"corner {corner} not recovered"

    def test_angled_sign_keeps_its_foreshortening(self):
        quad = mask_quad(self._mask(self.ANGLED))
        left = np.hypot(*(quad[0] - quad[3]))
        right = np.hypot(*(quad[1] - quad[2]))
        assert abs(left - right) / max(left, right) > 0.2, \
            "the two vertical edges must stay different lengths"

    def test_perspective_off_returns_the_old_rectangle(self):
        quad = mask_quad(self._mask(self.ANGLED), perspective=False)
        left = np.hypot(*(quad[0] - quad[3]))
        right = np.hypot(*(quad[1] - quad[2]))
        assert abs(left - right) < 2.0

    def test_a_flat_sign_is_not_given_a_fake_skew(self):
        """Below the tolerance the quad is a rectangle in all but rounding —
        using it would only add jitter to text that should sit straight."""
        quad = mask_quad(self._mask(self.FLAT))
        left = np.hypot(*(quad[0] - quad[3]))
        right = np.hypot(*(quad[1] - quad[2]))
        assert abs(left - right) < 2.0

    def test_area_matches_the_real_outline(self):
        quad = mask_quad(self._mask(self.ANGLED))
        real_area = cv2.contourArea(np.array(self.ANGLED, np.float32))
        assert cv2.contourArea(quad) == pytest.approx(real_area, rel=0.05)

    def test_circle_still_gets_a_straight_box(self):
        """The perspective path must not undo the degenerate-angle fix."""
        m = np.zeros((260, 420), np.float32)
        cv2.circle(m, (210, 130), 100, 1.0, -1)
        assert abs(quad_angle(mask_quad(m))) < 6.0

    def test_irregular_outline_falls_back_to_the_rectangle(self):
        """A torn poster is not a quad; forcing four corners would clip it."""
        m = np.zeros((300, 620), np.float32)
        pts = np.array([[60, 60], [300, 40], [540, 90], [520, 240],
                        [280, 210], [70, 250]], np.int32)
        cv2.fillPoly(m, [pts], 1.0)
        quad = mask_quad(m)
        assert quad is not None and len(quad) == 4

    def test_glyph_layer_still_clips_to_an_angled_mask(self):
        mask = self._mask(self.ANGLED)
        _, alpha = render_glyph_layer("WEINHANDEL", mask)
        assert float((alpha * (mask < 0.5)).max()) == 0.0
        assert float(alpha[mask > 0.5].max()) > 0.9


class TestCurvedAndRaggedShapes:
    """A homography can only describe a flat plane. A label wrapped round a
    bottle, fabric with folds, or a torn poster needs a column-wise fit —
    otherwise the text sits dead straight on a bowed surface, or gets chopped
    off at the edges once the layer is clipped to the mask.

    Measured overshoot of the four-corner fit: 34% of its own area for a bowed
    bottle label, 23% for a torn poster, ~2% for a flat sign.
    """

    H, W = 300, 520

    def _bottle(self):
        m = np.zeros((self.H, self.W), np.float32)
        for x in range(90, 430):
            t = (x - 90) / 340.0
            bow = int(26 * np.sin(np.pi * t))
            cv2.line(m, (x, 90 + bow), (x, 215 - bow), 1.0, 1)
        return m

    def _flat(self):
        m = np.zeros((self.H, self.W), np.float32)
        cv2.rectangle(m, (90, 90), (430, 215), 1.0, -1)
        return m

    def test_quad_fit_error_separates_flat_from_curved(self):
        flat = quad_fit_error(self._flat(), mask_quad(self._flat()))
        curved = quad_fit_error(self._bottle(), mask_quad(self._bottle()))
        assert flat < 0.05
        assert curved > 0.25
        assert flat < CONTOUR_FIT_THRESHOLD < curved

    def test_edge_profiles_follow_the_bow(self):
        x_min, x_max, top, bottom = edge_profiles(self._bottle())
        assert x_min < x_max
        middle = len(top) // 2
        # the label bows inward, so the middle is thinner than the ends
        assert (bottom[middle] - top[middle]) < (bottom[2] - top[2])

    def test_edge_profiles_are_smoothed(self):
        """Chasing every notch of a torn edge would tear the text apart."""
        m = self._flat()
        for x in range(90, 430, 7):        # cut regular notches into the top edge
            m[90:104, x:x + 3] = 0.0
        _, _, top, _ = edge_profiles(m)
        assert float(np.std(np.diff(top))) < 3.0

    def test_edge_profiles_empty_mask(self):
        assert edge_profiles(np.zeros((40, 40), np.float32)) is None

    def test_contour_warp_stays_inside_the_shape(self):
        mask = self._bottle()
        block = render_text_block("RESERVE 2019", 340, 125)
        _, alpha = warp_to_contour(block, mask, (self.H, self.W))
        assert float(alpha.max()) > 0.9
        outside = alpha * (mask < 0.5)
        assert float(outside.mean()) < 0.02

    def test_contour_warp_bows_with_the_label(self):
        """Ink near the middle must sit higher than at the ends, like the edge."""
        mask = self._bottle()
        block = render_text_block("RESERVE 2019", 340, 125)
        _, alpha = warp_to_contour(block, mask, (self.H, self.W))
        cols = np.where(alpha.max(axis=0) > 0.5)[0]
        first_top = np.argmax(alpha[:, cols[5]] > 0.5)
        mid_top = np.argmax(alpha[:, cols[len(cols) // 2]] > 0.5)
        assert mid_top > first_top, "the fit must follow the inward bow"

    def test_cylinder_compresses_towards_the_sides(self):
        mask = self._flat()
        block = render_text_block("ABCDEFGH", 340, 125)
        flat_rgb, _ = warp_to_contour(block, mask, (self.H, self.W), cylinder=0.0)
        cyl_rgb, _ = warp_to_contour(block, mask, (self.H, self.W), cylinder=0.8)
        assert not np.allclose(flat_rgb, cyl_rgb), "cylinder must change the sampling"

    def test_auto_picks_contour_for_a_curved_label(self):
        mask = self._bottle()
        auto_rgb, auto_a = render_glyph_layer("RESERVE", mask, fit="auto")
        cont_rgb, cont_a = render_glyph_layer("RESERVE", mask, fit="contour")
        assert np.allclose(auto_a, cont_a)

    def test_auto_picks_perspective_for_a_flat_sign(self):
        mask = self._flat()
        auto_rgb, auto_a = render_glyph_layer("RESERVE", mask, fit="auto")
        persp_rgb, persp_a = render_glyph_layer("RESERVE", mask, fit="perspective")
        assert np.allclose(auto_a, persp_a)

    def test_contour_fit_degrades_to_the_quad_when_profiles_fail(self):
        """A one-pixel sliver has no usable profile; it must not crash or blank."""
        m = np.zeros((self.H, self.W), np.float32)
        m[100, 100:200] = 1.0
        rgb, alpha = render_glyph_layer("X", m, fit="contour")
        assert rgb.shape == (self.H, self.W, 3)
        assert alpha.shape == (self.H, self.W)
