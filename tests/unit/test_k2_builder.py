"""K2 Lab — Layout-Parsing der Region-Builder-Node."""

import json
import sys
from unittest.mock import MagicMock

import pytest

INSTALLED_LORAS = [
    "krea2\\helper\\slider\\age_krea2_loraholic.safetensors",
    "krea2\\character\\female\\My Stars - Joanna.safetensors",
]

sys.modules["folder_paths"].get_filename_list = MagicMock(return_value=INSTALLED_LORAS)

from core.k2.layout import (  # noqa: E402
    default_layout_json,
    parse_layout,
    rescale_layout,
    rescale_rect,
)


def layout(boxes, canvas=(1024, 1024), global_loras=None):
    payload = {
        "version": 1,
        "canvas": {"width": canvas[0], "height": canvas[1]},
        "boxes": boxes,
    }
    if global_loras is not None:
        payload["global_loras"] = global_loras
    return json.dumps(payload)


def box(name="Anna", x=0.05, y=0.05, w=0.4, h=0.9, **extra):
    payload = {
        "id": extra.pop("id", None) or f"box-{name.lower()}",
        "name": name,
        "rect": {"x": x, "y": y, "w": w, "h": h},
        "prompt": extra.pop("prompt", "a woman"),
        "enabled": extra.pop("enabled", True),
    }
    payload.update(extra)
    return payload


class TestLayoutBasics:
    def test_default_layout_is_valid(self):
        regions, loras, notes = parse_layout(default_layout_json(), 1024, 1024)
        assert regions == [] and loras == [] and notes == []

    def test_empty_string_is_accepted(self):
        assert parse_layout("", 1024, 1024) == ([], [], [])

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_layout("{nope", 1024, 1024)

    def test_future_version_rejected(self):
        with pytest.raises(ValueError, match="newer than supported"):
            parse_layout(json.dumps({"version": 99, "boxes": []}), 1024, 1024)

    def test_boxes_must_be_array(self):
        with pytest.raises(ValueError, match="must be a JSON array"):
            parse_layout(json.dumps({"boxes": {}}), 1024, 1024)


class TestNormalizedGeometry:
    def test_normalized_rect_maps_to_pixels(self):
        regions, _, _ = parse_layout(
            layout([box(x=0.25, y=0.5, w=0.25, h=0.5)]), 1024, 1024
        )
        assert regions[0].box.as_tuple() == (256.0, 512.0, 512.0, 1024.0)

    def test_aspect_change_keeps_proportions(self):
        """Kern des Ganzen: dasselbe Layout auf einer anderen Leinwand."""
        data = layout([box(x=0.1, y=0.2, w=0.3, h=0.6)])
        square, _, _ = parse_layout(data, 1024, 1024)
        wide, _, _ = parse_layout(data, 1536, 640)

        assert square[0].box.as_tuple() == (102.4, 204.8, 409.6, 819.2)
        assert wide[0].box.as_tuple() == pytest.approx((153.6, 128.0, 614.4, 512.0))
        # relative Lage identisch
        assert wide[0].box.x0 / 1536 == pytest.approx(square[0].box.x0 / 1024)
        assert wide[0].box.height / 640 == pytest.approx(square[0].box.height / 1024)

    def test_portrait_to_landscape_stays_inside(self):
        data = layout([box(x=0.6, y=0.05, w=0.35, h=0.9)])
        regions, _, _ = parse_layout(data, 1920, 512)
        assert regions[0].box.x1 <= 1920
        assert regions[0].box.y1 <= 512

    def test_rect_is_clamped_to_canvas(self):
        regions, _, _ = parse_layout(layout([box(x=0.8, y=0.8, w=0.5, h=0.5)]), 1024, 1024)
        assert regions[0].box.x1 == 1024.0
        assert regions[0].box.y1 == 1024.0

    def test_legacy_pixel_rect_is_converted(self):
        data = json.dumps({
            "version": 1,
            "canvas": {"width": 512, "height": 512},
            "boxes": [{"id": "b1", "name": "Anna", "prompt": "a woman",
                       "rect": {"x": 128, "y": 0, "w": 256, "h": 512}}],
        })
        regions, _, notes = parse_layout(data, 1024, 1024)
        assert regions[0].box.as_tuple() == (256.0, 0.0, 768.0, 1024.0)
        assert any("pixel rect" in note for note in notes)

    def test_zero_size_rejected(self):
        with pytest.raises(ValueError, match="zero width or height"):
            parse_layout(layout([box(w=0.0)]), 1024, 1024)

    def test_flat_rect_fields_supported(self):
        data = json.dumps({"version": 1, "canvas": {"width": 1024, "height": 1024},
                           "boxes": [{"id": "b", "name": "A", "prompt": "x",
                                      "x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}]})
        regions, _, _ = parse_layout(data, 1024, 1024)
        assert regions[0].box.as_tuple() == (0.0, 0.0, 512.0, 1024.0)


class TestBoxFields:
    def test_disabled_box_skipped(self):
        regions, _, _ = parse_layout(
            layout([box("Anna"), box("Bea", x=0.55, enabled=False)]), 1024, 1024
        )
        assert [r.name for r in regions] == ["Anna"]

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="used twice"):
            parse_layout(layout([box("Anna"), box("Anna", x=0.55)]), 1024, 1024)

    def test_role_and_priority_pass_through(self):
        regions, _, _ = parse_layout(
            layout([box(role="background", priority=7)]), 1024, 1024
        )
        assert regions[0].role == "background"
        assert regions[0].priority == 7

    def test_unknown_role_falls_back(self):
        regions, _, notes = parse_layout(layout([box(role="weird")]), 1024, 1024)
        assert regions[0].role == "auto"
        assert any("unknown role" in note for note in notes)

    def test_identity_and_negative_kept(self):
        regions, _, _ = parse_layout(
            layout([box(identity_prompt="green eyes", negative_prompt="blurry")]),
            1024, 1024,
        )
        assert regions[0].identity_prompt == "green eyes"
        assert regions[0].negative_prompt == "blurry"

    def test_default_priority_descends(self):
        regions, _, _ = parse_layout(
            layout([box("A"), box("B", x=0.55), box("C", x=0.3, y=0.3, w=0.2, h=0.2)]),
            1024, 1024,
        )
        assert [r.priority for r in regions] == [100, 99, 98]


class TestLoraRouting:
    def test_lora_bound_to_its_box(self):
        regions, loras, _ = parse_layout(
            layout([box("Anna", loras=[{"name": INSTALLED_LORAS[0], "strength": 1.5}])]),
            1024, 1024,
        )
        assert len(loras) == 1
        assert loras[0].global_scope is False
        assert loras[0].region_ids == (regions[0].region_id,)
        assert loras[0].strength == 1.5

    def test_multiple_loras_per_box(self):
        entries = [
            {"name": INSTALLED_LORAS[0], "strength": 1.0},
            {"name": INSTALLED_LORAS[1], "strength": 0.6},
        ]
        _, loras, _ = parse_layout(layout([box(loras=entries)]), 1024, 1024)
        assert len(loras) == 2
        assert {spec.lora_name for spec in loras} == set(INSTALLED_LORAS)

    def test_loras_across_four_boxes_stay_separate(self):
        boxes = [
            box("A", x=0.00, w=0.24, loras=[{"name": INSTALLED_LORAS[0]}]),
            box("B", x=0.25, w=0.24, loras=[{"name": INSTALLED_LORAS[1]}]),
            box("C", x=0.50, w=0.24, loras=[{"name": INSTALLED_LORAS[0], "strength": 2.0}]),
            box("D", x=0.75, w=0.24),
        ]
        regions, loras, _ = parse_layout(layout(boxes), 1024, 1024)
        assert len(regions) == 4
        assert len(loras) == 3
        by_region = {spec.region_ids[0] for spec in loras}
        assert by_region == {"box-a", "box-b", "box-c"}

    def test_disabled_lora_skipped(self):
        _, loras, _ = parse_layout(
            layout([box(loras=[{"name": INSTALLED_LORAS[0], "enabled": False}])]),
            1024, 1024,
        )
        assert loras == []

    def test_none_entry_skipped(self):
        _, loras, _ = parse_layout(
            layout([box(loras=[{"name": "None"}, {"name": ""}])]), 1024, 1024
        )
        assert loras == []

    def test_missing_lora_file_raises(self):
        with pytest.raises(ValueError, match="not installed"):
            parse_layout(layout([box(loras=[{"name": "ghost.safetensors"}])]), 1024, 1024)

    def test_character_routing_carries_trigger(self):
        _, loras, _ = parse_layout(
            layout([box(loras=[{"name": INSTALLED_LORAS[1],
                                "routing": "character_identity",
                                "trigger": "Joanna"}])]),
            1024, 1024,
        )
        assert loras[0].routing_mode == "character_identity"
        assert loras[0].trigger_phrase == "Joanna"

    def test_unknown_routing_falls_back(self):
        _, loras, notes = parse_layout(
            layout([box(loras=[{"name": INSTALLED_LORAS[0], "routing": "weird"}])]),
            1024, 1024,
        )
        assert loras[0].routing_mode == "standard"
        assert any("unknown routing" in note for note in notes)

    def test_global_lora_supported(self):
        _, loras, _ = parse_layout(
            layout([box()], global_loras=[{"name": INSTALLED_LORAS[0], "strength": 0.8}]),
            1024, 1024,
        )
        globals_ = [spec for spec in loras if spec.global_scope]
        assert len(globals_) == 1
        assert globals_[0].region_ids == ()


class TestShapePreservingRescale:
    """Der Punkt, an dem fraction-basierte Editoren wie der KJ-Builder scheitern."""

    def test_square_box_keeps_its_shape_on_wide_canvas(self):
        # 0.3 x 0.3 auf 1024x1024 ist ein Quadrat (307x307 px).
        rect = rescale_rect((0.35, 0.35, 0.3, 0.3), (1024, 1024), (1920, 1080))
        pixel_w = rect[2] * 1920
        pixel_h = rect[3] * 1080
        assert pixel_w == pytest.approx(pixel_h, rel=0.02)

    def test_naive_normalization_would_distort(self):
        # Gegenprobe: ohne Umrechnung wären 0.3/0.3 auf 1920x1080 → 576x324.
        assert 0.3 * 1920 != pytest.approx(0.3 * 1080, rel=0.02)

    def test_centre_stays_relative(self):
        rect = rescale_rect((0.4, 0.4, 0.2, 0.2), (1024, 1024), (1536, 640))
        assert rect[0] + rect[2] / 2 == pytest.approx(0.5, abs=0.01)
        assert rect[1] + rect[3] / 2 == pytest.approx(0.5, abs=0.01)

    def test_oversized_box_shrinks_uniformly(self):
        rect = rescale_rect((0.0, 0.0, 1.0, 1.0), (1024, 1024), (512, 2048))
        assert rect[2] * 512 == pytest.approx(rect[3] * 2048, rel=0.02)
        assert rect[2] <= 1.0 and rect[3] <= 1.0

    def test_stays_inside_canvas(self):
        rect = rescale_rect((0.75, 0.75, 0.25, 0.25), (1024, 1024), (2048, 512))
        assert rect[0] >= 0.0 and rect[1] >= 0.0
        assert rect[0] + rect[2] <= 1.0 + 1e-9
        assert rect[1] + rect[3] <= 1.0 + 1e-9

    def test_identical_canvas_is_a_noop(self):
        original = (0.1, 0.2, 0.3, 0.4)
        assert rescale_rect(original, (1024, 1024), (1024, 1024)) == original

    def test_uniform_scale_is_a_noop(self):
        rect = rescale_rect((0.1, 0.2, 0.3, 0.4), (512, 512), (1024, 1024))
        assert rect == pytest.approx((0.1, 0.2, 0.3, 0.4), abs=1e-9)

    def test_layout_rescale_updates_canvas(self):
        data = layout([box("A", x=0.1, y=0.1, w=0.3, h=0.3)])
        rescaled = json.loads(rescale_layout(data, 1920, 1080))
        assert rescaled["canvas"] == {"width": 1920, "height": 1080}
        rect = rescaled["boxes"][0]["rect"]
        assert rect["w"] * 1920 == pytest.approx(rect["h"] * 1080, rel=0.02)

    def test_round_trip_back_to_square(self):
        wide = json.loads(rescale_layout(layout([box("A", x=0.3, y=0.3, w=0.2, h=0.2)]),
                                         1920, 1080))
        back = json.loads(rescale_layout(json.dumps(wide), 1024, 1024))
        rect = back["boxes"][0]["rect"]
        assert rect["w"] == pytest.approx(0.2, abs=0.01)
        assert rect["h"] == pytest.approx(0.2, abs=0.01)
