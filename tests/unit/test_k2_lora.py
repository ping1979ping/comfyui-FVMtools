"""K2 Lab — LoRA-Routing, Projector und Gesichtszuordnung."""

import numpy as np
import pytest

from core.k2.binding import bind_plan
from core.k2.face import (
    FaceDetection,
    assign_faces,
    composite_crop,
    expanded_square_crop,
    feather_mask,
)
from core.k2.geometry import PixelBox
from core.k2.lora import (
    CHARACTER_ROUTING,
    STANDARD_ROUTING,
    LoraSpec,
    align_state_dict,
    compile_routes,
    identity_triggers_from_specs,
    normalize_key,
    route_allows_target,
    route_kind,
)
from core.k2.projector import (
    PROJECTOR_LENGTH,
    parse_values,
    preset_values,
    scaled_values,
    token_delta_mask,
    validate_values,
)
from core.k2.prompt import RegionDefinition, compile_plan

from .test_k2_prompt import fake_tokenize, two_subjects


def bound_plan():
    plan = compile_plan(1024, 1024, "two women in a park", two_subjects())
    return bind_plan(plan, fake_tokenize)


class TestKeyNormalisation:
    def test_strips_diffusion_model_prefix(self):
        assert normalize_key("diffusion_model.blocks.0.attn.wq") == "blocks.0.attn.wq"
        assert normalize_key("diffusion_model.txtfusion.projector") == "txtfusion.projector"

    def test_leaves_unknown_namespaces(self):
        assert normalize_key("transformer.text_fusion.x") == "transformer.text_fusion.x"

    def test_align_picks_supported_namespace(self):
        state = {"diffusion_model.blocks.0.attn.wq.lora_A.weight": 1}
        aligned = align_state_dict(state, {"blocks.0.attn.wq"})
        assert "blocks.0.attn.wq.lora_A.weight" in aligned

    def test_align_keeps_original_when_supported(self):
        key = "diffusion_model.blocks.0.attn.wq.lora_A.weight"
        aligned = align_state_dict({key: 1}, {"diffusion_model.blocks.0.attn.wq"})
        assert key in aligned


class TestRouteCompilation:
    def test_global_route_covers_everything(self):
        bound = bound_plan()
        routes = compile_routes(
            [LoraSpec("l1", "x.safetensors", 1.0, global_scope=True)], bound
        )
        assert routes[0].global_scope
        assert np.allclose(routes[0].text_mask, 1.0)
        assert np.allclose(routes[0].image_mask, 1.0)

    def test_regional_route_masks_are_partial(self):
        bound = bound_plan()
        routes = compile_routes(
            [LoraSpec("l1", "x.safetensors", 1.0, global_scope=False,
                      region_ids=("r1",))],
            bound,
        )
        route = routes[0]
        assert 0 < route.text_mask.sum() < route.text_count
        assert 0 < route.image_mask.sum() < route.image_count

    def test_regional_masks_are_disjoint_for_disjoint_boxes(self):
        bound = bound_plan()
        routes = compile_routes(
            [
                LoraSpec("l1", "a.safetensors", 1.0, global_scope=False, region_ids=("r1",)),
                LoraSpec("l2", "b.safetensors", 1.0, global_scope=False, region_ids=("r2",)),
            ],
            bound,
        )
        overlap = routes[0].image_mask * routes[1].image_mask
        assert overlap.sum() == 0
        assert (routes[0].text_mask * routes[1].text_mask).sum() == 0

    def test_zero_strength_dropped(self):
        bound = bound_plan()
        routes = compile_routes([LoraSpec("l1", "x.safetensors", 0.0)], bound)
        assert routes == ()

    def test_unknown_region_raises(self):
        bound = bound_plan()
        with pytest.raises(ValueError):
            compile_routes(
                [LoraSpec("l1", "x.safetensors", 1.0, global_scope=False,
                          region_ids=("nope",))],
                bound,
            )

    def test_regional_without_region_raises(self):
        bound = bound_plan()
        with pytest.raises(ValueError):
            compile_routes([LoraSpec("l1", "x.safetensors", 1.0, global_scope=False)], bound)

    def test_out_of_range_strength_rejected(self):
        with pytest.raises(ValueError):
            LoraSpec("l1", "x.safetensors", 9.0)

    def test_character_routing_requires_trigger(self):
        with pytest.raises(ValueError):
            LoraSpec("l1", "x.safetensors", 1.0, global_scope=False,
                     region_ids=("r1",), routing_mode=CHARACTER_ROUTING)

    def test_character_routing_requires_regional_scope(self):
        with pytest.raises(ValueError):
            LoraSpec("l1", "x.safetensors", 1.0, global_scope=True,
                     routing_mode=CHARACTER_ROUTING, trigger_phrase="ohwx")

    def test_identity_triggers_collected(self):
        specs = [LoraSpec("l1", "x.safetensors", 1.0, global_scope=False,
                          region_ids=("r1",), routing_mode=CHARACTER_ROUTING,
                          trigger_phrase="ohwx woman")]
        assert identity_triggers_from_specs(specs) == {"r1": ("ohwx woman",)}


class TestTargetPolicy:
    @staticmethod
    def regional_route():
        bound = bound_plan()
        return compile_routes(
            [LoraSpec("l1", "x.safetensors", 1.0, global_scope=False, region_ids=("r1",))],
            bound,
        )[0]

    def test_key_value_targets_skipped_when_strict(self):
        route = self.regional_route()
        assert not route_allows_target(route, "diffusion_model.blocks.0.attn.wk.weight", True)
        assert not route_allows_target(route, "diffusion_model.blocks.0.attn.wv.weight", True)

    def test_query_and_output_allowed(self):
        route = self.regional_route()
        assert route_allows_target(route, "diffusion_model.blocks.0.attn.wq.weight", True)
        assert route_allows_target(route, "diffusion_model.blocks.0.attn.wo.weight", True)
        assert route_allows_target(route, "diffusion_model.blocks.0.mlp.down.weight", True)

    def test_text_fusion_always_allowed(self):
        route = self.regional_route()
        assert route_allows_target(
            route, "diffusion_model.txtfusion.refiner_blocks.0.attn.wk.weight", True
        )

    def test_loose_mode_allows_everything(self):
        route = self.regional_route()
        assert route_allows_target(route, "diffusion_model.blocks.0.attn.wk.weight", False)

    def test_global_route_unrestricted(self):
        bound = bound_plan()
        route = compile_routes([LoraSpec("l1", "x.safetensors", 1.0)], bound)[0]
        assert route_allows_target(route, "diffusion_model.blocks.0.attn.wk.weight", True)

    def test_route_kind_detection(self):
        assert route_kind("diffusion_model.blocks.0.attn.wq.weight") == "combined"
        assert route_kind(
            "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight"
        ) == "text_layerwise"
        assert route_kind("diffusion_model.txtfusion.projector.weight") == "text_projector"
        assert route_kind(
            "diffusion_model.txtfusion.refiner_blocks.0.mlp.up.weight"
        ) == "text_refiner"
        assert route_kind("diffusion_model.first.weight") == "unmasked"


class TestProjector:
    def test_presets_have_twelve_values(self):
        for name in ("filter_bypass2", "filter_bypass3", "skc3vo", "z0jglf"):
            assert len(preset_values(name)) == PROJECTOR_LENGTH

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            preset_values("nope")

    def test_multiplier_scales_linearly(self):
        base = preset_values("filter_bypass2")
        assert np.allclose(scaled_values(base, 2.0), np.array(base) * 2.0)

    def test_zero_multiplier_disables(self):
        assert not any(scaled_values(preset_values("skc3vo"), 0.0))

    def test_parse_various_formats(self):
        expected = tuple(float(v) for v in range(12))
        assert parse_values("0,1,2,3,4,5,6,7,8,9,10,11") == expected
        assert parse_values("[0 1 2 3 4 5 6 7 8 9 10 11]") == expected

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            validate_values([1.0, 2.0])

    def test_identity_protection_masks_span(self):
        mask = token_delta_mask(10, ((2, 5),), 1.0)
        assert np.allclose(mask[2:5], 0.0)
        assert np.allclose(mask[:2], 1.0)
        assert np.allclose(mask[5:], 1.0)

    def test_partial_protection(self):
        mask = token_delta_mask(10, ((0, 2),), 0.25)
        assert np.allclose(mask[:2], 0.75)

    def test_span_outside_sequence_raises(self):
        with pytest.raises(ValueError):
            token_delta_mask(5, ((3, 9),), 1.0)


class TestFaceHelpers:
    def test_crop_is_square_and_inside_canvas(self):
        box = expanded_square_crop(PixelBox(10, 10, 60, 90), 512, 512, 2.0)
        x0, y0, x1, y1 = box
        assert x1 - x0 == y1 - y0
        assert x0 >= 0 and y0 >= 0 and x1 <= 512 and y1 <= 512

    def test_crop_clamped_at_border(self):
        box = expanded_square_crop(PixelBox(0, 0, 40, 40), 256, 256, 4.0)
        assert box[0] == 0 and box[1] == 0

    def test_feather_mask_edges_soft(self):
        mask = feather_mask((64, 64), 0.25)
        assert mask[32, 32] == pytest.approx(1.0)
        assert mask[0, 32] < 0.2

    def test_no_feather_is_flat(self):
        assert np.allclose(feather_mask((32, 32), 0.0), 1.0)

    def test_composite_keeps_outside_pixels(self):
        canvas = np.zeros((64, 64, 3), dtype=np.float32)
        refined = np.ones((16, 16, 3), dtype=np.float32)
        result = composite_crop(canvas, refined, (16, 16, 32, 32), 0.0, 1.0)
        assert np.allclose(result[0, 0], 0.0)
        assert result[24, 24].mean() > 0.9

    def test_faces_assigned_to_containing_region(self):
        bound = bound_plan()
        detections = [
            FaceDetection(PixelBox(200, 150, 280, 250), 0.9),   # in Annas Box
            FaceDetection(PixelBox(700, 150, 780, 250), 0.8),   # in Beas Box
        ]
        targets = assign_faces(detections, bound)
        assert [t.region_name for t in targets] == ["Anna", "Bea"]

    def test_detection_outside_all_boxes_ignored(self):
        bound = bound_plan()
        targets = assign_faces([FaceDetection(PixelBox(500, 20, 530, 50), 0.9)], bound)
        assert targets == []

    def test_require_lora_filters(self):
        bound = bound_plan()
        detections = [FaceDetection(PixelBox(200, 150, 280, 250), 0.9)]
        assert assign_faces(detections, bound, (), require_lora=True) == []

    def test_crop_prompt_puts_identity_first(self):
        plan = compile_plan(
            1024, 1024, "scene",
            [RegionDefinition("r1", "Anna", PixelBox(48, 64, 488, 1004),
                              prompt="a woman in a red dress",
                              identity_prompt="a freckled face", role="subject")],
        )
        bound = bind_plan(plan, fake_tokenize)
        targets = assign_faces([FaceDetection(PixelBox(200, 150, 280, 250), 0.9)], bound)
        assert targets[0].prompt.startswith("a close-up portrait of a freckled face")
