"""K2 Lab — Prompt-Kompilierung, Tokenbindung und Attention-Maske."""

import numpy as np
import pytest
import torch

from core.k2.attention import (
    K2SpatialAttention,
    image_region_owners,
    text_region_owners,
)
from core.k2.binding import bind_plan, krea_prompt_token_count
from core.k2.geometry import PixelBox
from core.k2.prompt import (
    GLOBAL_SCOPE,
    EmphasisRequest,
    RegionDefinition,
    compile_plan,
)


def fake_tokenize(text):
    """Ein Token je vier Zeichen — deterministisch und ausreichend für Spannen."""
    count = max(1, len(text) // 4)
    return {"qwen3vl_4b": [[(index, 1.0) for index in range(count)]]}


def two_subjects():
    return [
        RegionDefinition("r1", "Anna", PixelBox(48, 64, 488, 1004),
                         prompt="a blonde woman in a red dress",
                         role="subject", priority=100),
        RegionDefinition("r2", "Bea", PixelBox(536, 64, 976, 1004),
                         prompt="a dark-haired woman in a blue dress",
                         role="subject", priority=99),
    ]


class TestCompilePlan:
    def test_regions_appear_in_prompt(self):
        plan = compile_plan(1024, 1024, "two women in a park", two_subjects())
        assert "two women in a park." in plan.prompt
        assert "a blonde woman in a red dress" in plan.prompt
        assert "a dark-haired woman in a blue dress" in plan.prompt

    def test_location_clauses_reflect_boxes(self):
        plan = compile_plan(1024, 1024, "", two_subjects())
        assert "left side" in plan.regions[0].clause
        assert "right side" in plan.regions[1].clause

    def test_relationship_clause_orders_subjects(self):
        plan = compile_plan(1024, 1024, "", two_subjects())
        assert "Anna is to the left of Bea" in plan.prompt

    def test_char_spans_match_prompt(self):
        plan = compile_plan(1024, 1024, "scene", two_subjects())
        for region in plan.regions:
            start, end = region.char_span
            assert plan.prompt[start:end] == region.clause

    def test_disabled_region_excluded(self):
        regions = two_subjects()
        regions[1].enabled = False
        plan = compile_plan(1024, 1024, "scene", regions)
        assert len(plan.regions) == 1

    def test_empty_prompt_region_excluded(self):
        regions = two_subjects()
        regions[1].prompt = ""
        plan = compile_plan(1024, 1024, "scene", regions)
        assert len(plan.regions) == 1

    def test_priority_orders_compilation(self):
        regions = two_subjects()
        regions[0].priority = 1
        regions[1].priority = 99
        plan = compile_plan(1024, 1024, "", regions)
        assert plan.regions[0].name == "Bea"

    def test_auto_role_wide_box_becomes_background(self):
        regions = [RegionDefinition("bg", "Sky", PixelBox(0, 0, 1024, 300),
                                    prompt="a stormy sky", role="auto")]
        plan = compile_plan(1024, 1024, "", regions)
        assert plan.regions[0].role == "background"

    def test_auto_role_narrow_box_becomes_subject(self):
        regions = [RegionDefinition("s", "Anna", PixelBox(0, 0, 400, 1000),
                                    prompt="a woman", role="auto")]
        plan = compile_plan(1024, 1024, "", regions)
        assert plan.regions[0].role == "subject"

    def test_identity_is_attached_not_prepended(self):
        """Ein vorangestellter Identitätssatz wird als eigenes Objekt gemalt."""
        regions = [RegionDefinition("s", "Anna", PixelBox(0, 0, 400, 1000),
                                    prompt="a woman in a red dress",
                                    identity_prompt="a freckled face",
                                    role="subject")]
        plan = compile_plan(1024, 1024, "", regions)
        clause = plan.regions[0].clause
        assert "a woman in a red dress, with a freckled face" in clause
        assert not clause.startswith("a freckled face")

    def test_identity_span_locatable(self):
        regions = [RegionDefinition("s", "Anna", PixelBox(0, 0, 400, 1000),
                                    prompt="a woman", identity_prompt="green eyes",
                                    role="subject")]
        plan = compile_plan(1024, 1024, "", regions)
        span = plan.regions[0].identity_char_span
        assert span is not None
        assert plan.prompt[span[0]:span[1]] == "green eyes"

    def test_spatial_instructions_can_be_disabled(self):
        plan = compile_plan(1024, 1024, "", two_subjects(), spatial_instructions=False)
        assert "left side" not in plan.prompt
        assert "a blonde woman in a red dress" in plan.prompt

    def test_invalid_parameters_rejected(self):
        with pytest.raises(ValueError):
            compile_plan(1024, 1024, "", two_subjects(), strength=0.0)
        with pytest.raises(ValueError):
            compile_plan(1024, 1024, "", two_subjects(), late_step_scale=2.0)


class TestEmphasis:
    def test_global_phrase_located(self):
        plan = compile_plan(
            1024, 1024, "a red car in a field", two_subjects(),
            emphases=[EmphasisRequest(GLOBAL_SCOPE, "red car", 0.5, 0)],
        )
        start, end = plan.emphases[0].char_span
        assert plan.prompt[start:end] == "red car"

    def test_regional_phrase_located(self):
        plan = compile_plan(
            1024, 1024, "scene", two_subjects(),
            emphases=[EmphasisRequest("r1", "red dress", 0.5, 0)],
        )
        start, end = plan.emphases[0].char_span
        assert plan.prompt[start:end] == "red dress"

    def test_missing_phrase_raises(self):
        with pytest.raises(ValueError):
            compile_plan(1024, 1024, "scene", two_subjects(),
                         emphases=[EmphasisRequest(GLOBAL_SCOPE, "unicorn", 0.5, 0)])

    def test_unknown_scope_raises(self):
        with pytest.raises(ValueError):
            compile_plan(1024, 1024, "scene", two_subjects(),
                         emphases=[EmphasisRequest("nope", "scene", 0.5, 0)])


class TestTokenCounting:
    def test_plain_token_list(self):
        assert krea_prompt_token_count(fake_tokenize("abcdefgh")) == 2

    def test_chat_template_stripped(self):
        pairs = [(151644, 1.0), (100, 1.0), (151645, 1.0),
                 (151644, 1.0), (872, 1.0), (198, 1.0),
                 (1, 1.0), (2, 1.0), (3, 1.0), (151645, 1.0)]
        assert krea_prompt_token_count({"qwen3vl_4b": [pairs]}) == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            krea_prompt_token_count({})


class TestBinding:
    def test_spans_are_ordered_and_disjoint(self):
        plan = compile_plan(1024, 1024, "scene", two_subjects())
        bound = bind_plan(plan, fake_tokenize)
        first, second = bound.spans
        assert first.start < first.end <= second.start < second.end

    def test_span_within_conditioning(self):
        plan = compile_plan(1024, 1024, "scene", two_subjects())
        bound = bind_plan(plan, fake_tokenize)
        assert max(s.end for s in bound.spans) <= bound.text_token_count

    def test_too_short_conditioning_raises(self):
        plan = compile_plan(1024, 1024, "scene", two_subjects())
        with pytest.raises(ValueError):
            bind_plan(plan, fake_tokenize, conditioning_text_token_count=5)

    def test_image_token_count_matches_grid(self):
        plan = compile_plan(1024, 1024, "scene", two_subjects())
        bound = bind_plan(plan, fake_tokenize)
        assert bound.image_token_count == 4096
        assert bound.sequence_length == bound.text_token_count + 4096


class TestAttentionMask:
    @staticmethod
    def build():
        plan = compile_plan(1024, 1024, "two women in a park", two_subjects())
        bound = bind_plan(plan, fake_tokenize)
        return bound, K2SpatialAttention(bound, strict_isolation=True)

    def test_owners_are_exclusive(self):
        bound, _ = self.build()
        image_owners = image_region_owners(bound)
        text_owners = text_region_owners(bound)
        assert set(np.unique(image_owners)) <= {0, 1, 2}
        assert (image_owners == 1).sum() > 0
        assert (image_owners == 2).sum() > 0
        assert (text_owners > 0).sum() > 0

    def test_mask_shape(self):
        bound, attention = self.build()
        mask = attention._build_main_mask(torch.device("cpu"), torch.float32)
        assert mask.shape == (bound.sequence_length, bound.sequence_length)

    def test_image_to_image_untouched(self):
        """Bild↔Bild darf nie maskiert werden, sonst entstehen Kachelkanten."""
        bound, attention = self.build()
        mask = attention._build_main_mask(torch.device("cpu"), torch.float32)
        text = bound.text_token_count
        assert torch.count_nonzero(mask[text:, text:]) == 0

    def test_no_fully_blocked_query(self):
        """Eine komplett gesperrte Zeile ergäbe NaN im Softmax."""
        bound, attention = self.build()
        mask = attention._build_main_mask(torch.device("cpu"), torch.float32)
        limit = torch.finfo(torch.float32).min / 4
        assert int((mask > limit).sum(dim=1).min()) > 0

    def test_subject_text_is_private(self):
        bound, attention = self.build()
        mask = attention._build_main_mask(torch.device("cpu"), torch.float32)
        limit = torch.finfo(torch.float32).min / 4
        a, b = bound.spans
        assert bool((mask[a.start:a.end, b.start:b.end] <= limit).all())
        assert bool((mask[b.start:b.end, a.start:a.end] <= limit).all())

    def test_image_token_prefers_its_own_region(self):
        bound, attention = self.build()
        mask = attention._build_main_mask(torch.device("cpu"), torch.float32)
        text = bound.text_token_count
        limit = torch.finfo(torch.float32).min / 4
        left = text + 32 * 64 + 5      # Zeile 32, Spalte 5 → Annas Box
        right = text + 32 * 64 + 58    # Zeile 32, Spalte 58 → Beas Box
        anna, bea = bound.spans
        assert float(mask[left, anna.start]) > 0
        assert float(mask[left, bea.start]) <= limit
        assert float(mask[right, bea.start]) > 0
        assert float(mask[right, anna.start]) <= limit

    def test_soft_mode_has_no_hard_blocks(self):
        plan = compile_plan(1024, 1024, "scene", two_subjects())
        bound = bind_plan(plan, fake_tokenize)
        attention = K2SpatialAttention(bound, strict_isolation=False)
        mask = attention._build_main_mask(torch.device("cpu"), torch.float32)
        limit = torch.finfo(torch.float32).min / 4
        assert torch.count_nonzero(mask <= limit) == 0

    def test_late_step_relaxation(self):
        _, attention = self.build()
        attention.set_denoising_progress(1, 8)
        assert attention.step_scale == 1.0
        attention.set_denoising_progress(8, 8)
        assert np.isclose(attention.step_scale, 0.35)

    def test_relaxation_scales_the_bias(self):
        _, attention = self.build()
        full = attention._build_main_mask(torch.device("cpu"), torch.float32)
        attention.set_denoising_progress(8, 8)
        relaxed = attention._build_main_mask(torch.device("cpu"), torch.float32)
        finite = full > torch.finfo(torch.float32).min / 4
        assert float(relaxed[finite].abs().sum()) < float(full[finite].abs().sum())

    def test_invalid_total_steps(self):
        _, attention = self.build()
        with pytest.raises(ValueError):
            attention.set_denoising_progress(1, 0)
