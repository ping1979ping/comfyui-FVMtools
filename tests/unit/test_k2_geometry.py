"""K2 Lab — Geometrie und Bildtokenraster."""

import numpy as np
import pytest

from core.k2.geometry import (
    TOKEN_PIXELS,
    CanvasGeometry,
    PixelBox,
    align_up,
    apply_subject_competition,
    spatial_pair_bias,
)


class TestPixelBox:
    def test_basic_dimensions(self):
        box = PixelBox(10, 20, 110, 220)
        assert box.width == 100
        assert box.height == 200
        assert box.center == (60, 120)

    def test_rejects_degenerate(self):
        with pytest.raises(ValueError):
            PixelBox(10, 10, 10, 20)
        with pytest.raises(ValueError):
            PixelBox(10, 10, 20, 10)

    def test_from_xywh(self):
        assert PixelBox.from_xywh(5, 6, 10, 20).as_tuple() == (5.0, 6.0, 15.0, 26.0)

    def test_clip_to_canvas(self):
        clipped = PixelBox(-50, -50, 600, 600).clipped(512, 512)
        assert clipped.as_tuple() == (0.0, 0.0, 512.0, 512.0)

    def test_clip_outside_canvas_raises(self):
        with pytest.raises(ValueError):
            PixelBox(600, 600, 700, 700).clipped(512, 512)

    def test_grow(self):
        assert PixelBox(10, 10, 20, 20).grown(5).as_tuple() == (5.0, 5.0, 25.0, 25.0)


class TestCanvasGeometry:
    def test_token_grid(self):
        geometry = CanvasGeometry.resolve(1024, 1024)
        assert TOKEN_PIXELS == 16
        assert geometry.token_width == 64
        assert geometry.token_height == 64
        assert geometry.token_count == 4096

    def test_alignment(self):
        geometry = CanvasGeometry.resolve(1000, 700)
        assert geometry.aligned_width == 1008
        assert geometry.aligned_height == 704
        assert align_up(1000, 16) == 1008

    def test_rasterize_full_canvas(self):
        geometry = CanvasGeometry.resolve(256, 256)
        mask = geometry.rasterize_box(PixelBox(0, 0, 256, 256))
        assert mask.shape == (geometry.token_count,)
        assert np.allclose(mask, 1.0)

    def test_rasterize_left_half(self):
        geometry = CanvasGeometry.resolve(1024, 1024)
        mask = geometry.rasterize_box(PixelBox(0, 0, 512, 1024)).reshape(64, 64)
        assert np.allclose(mask[:, :32], 1.0)
        assert np.allclose(mask[:, 32:], 0.0)

    def test_partial_token_coverage(self):
        """Eine halb überdeckte 16px-Zelle bekommt den Flächenanteil, nicht 1.0."""
        geometry = CanvasGeometry.resolve(64, 64)
        mask = geometry.rasterize_box(PixelBox(0, 0, 8, 64)).reshape(4, 4)
        assert np.allclose(mask[:, 0], 0.5)
        assert np.allclose(mask[:, 1:], 0.0)

    def test_soft_field_decays_outside(self):
        geometry = CanvasGeometry.resolve(512, 512)
        field = geometry.soft_box_field(PixelBox(0, 0, 128, 512), 64.0).reshape(32, 32)
        assert np.allclose(field[:, :8], 1.0)          # innerhalb
        assert field[0, 9] > 0.0                        # Abklingzone
        assert np.allclose(field[:, 16:], 0.0)          # jenseits des Falloffs

    def test_subject_field_peaks_in_centre(self):
        geometry = CanvasGeometry.resolve(512, 512)
        field = geometry.subject_target_field(
            PixelBox(128, 128, 384, 384), 0.0, edge_weight=0.5
        ).reshape(32, 32)
        assert field[16, 16] > field[9, 9]
        assert field.max() <= 1.0

    def test_zero_falloff_has_hard_edge(self):
        geometry = CanvasGeometry.resolve(256, 256)
        field = geometry.soft_box_field(PixelBox(0, 0, 128, 256), 0.0).reshape(16, 16)
        assert np.allclose(field[:, :8], 1.0)
        assert np.allclose(field[:, 8:], 0.0)


class TestSubjectCompetition:
    def test_disjoint_fields_unchanged(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        result = apply_subject_competition([a, b], ["subject", "subject"])
        assert np.allclose(result[0], a)
        assert np.allclose(result[1], b)

    def test_overlap_is_shared(self):
        """Gleich starke Subjekte teilen ein Token hälftig statt es beide voll zu nehmen."""
        a = np.array([1.0], dtype=np.float32)
        b = np.array([1.0], dtype=np.float32)
        result = apply_subject_competition([a, b], ["subject", "subject"])
        assert np.isclose(result[0][0], 0.5)
        assert np.isclose(result[1][0], 0.5)

    def test_stronger_subject_wins_more(self):
        a = np.array([1.0], dtype=np.float32)
        b = np.array([0.5], dtype=np.float32)
        result = apply_subject_competition([a, b], ["subject", "subject"])
        assert result[0][0] > result[1][0]

    def test_single_subject_untouched(self):
        a = np.array([1.0, 0.4], dtype=np.float32)
        result = apply_subject_competition([a], ["subject"])
        assert np.allclose(result[0], a)

    def test_background_not_affected(self):
        a = np.array([1.0], dtype=np.float32)
        b = np.array([1.0], dtype=np.float32)
        result = apply_subject_competition([a, b], ["subject", "background"])
        assert np.allclose(result[1], b)


class TestSpatialPairBias:
    def test_endpoints(self):
        bias = spatial_pair_bias(np.array([1.0, 0.0]), strength=2.0, outside_penalty=1.0)
        assert np.isclose(bias[0], 2.0)
        assert np.isclose(bias[1], -1.0)

    def test_monotonic(self):
        field = np.linspace(0.0, 1.0, 11)
        bias = spatial_pair_bias(field, 1.0, 1.0)
        assert np.all(np.diff(bias) > 0)
