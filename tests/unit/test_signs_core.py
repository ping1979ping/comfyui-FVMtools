"""FVM Signs core — class registry, slop heuristics and crop clustering."""

import cv2
import numpy as np
import pytest

from core.signs import (
    BIGRAM_FREQ,
    CLASS_KEYS,
    DEFAULT_CLUSTER_DISTANCE,
    DEFAULT_SLOP_WEIGHTS,
    DEFAULT_THRESHOLD,
    EMPTY_DETECTED_FLOOR,
    FALLBACK_CLASS,
    MAX_BIGRAM_WEIGHT,
    MAX_DENOISE_BIAS,
    MIN_DENOISE_BIAS,
    SIGN_CLASSES,
    SLOP_THRESHOLD,
    WORD_LIST,
    all_class_names,
    bigram_plausibility,
    build_prompt,
    clamp_denoise_bias,
    cluster_crops,
    collapse_separators,
    color_signature,
    crop_distance,
    dictionary_ratio,
    extract_features,
    get_class,
    parse_custom_prompts,
    phash,
    pick_cluster_representative,
    repeated_glyph_ratio,
    score_slop,
)

EXPECTED_CLASSES = [
    "sign", "label", "garment_print", "poster", "screen",
    "book", "plate", "paper", "graffiti",
]


# ──── Synthetic crop helpers ────

def _paint(img, y0, y1, x0, x1, color):
    """Fill a box given in relative 0..1 coordinates."""
    size = img.shape[0]
    img[int(y0 * size):int(y1 * size), int(x0 * size):int(x1 * size)] = color


def make_label_crop(size=64):
    """A structured, label-like crop: gradient background plus text bars."""
    yy, xx = np.mgrid[0:size, 0:size]
    span = float(max(1, size - 1))
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = np.clip(60 + (xx / span) * 160, 0, 255)
    img[..., 1] = np.clip(40 + (yy / span) * 160, 0, 255)
    img[..., 2] = 90
    _paint(img, 0.19, 0.31, 0.125, 0.875, (230, 230, 220))
    _paint(img, 0.44, 0.56, 0.125, 0.625, (230, 230, 220))
    _paint(img, 0.69, 0.81, 0.125, 0.750, (20, 20, 30))
    return img


def make_poster_crop(size=64):
    """A clearly different crop: other layout, other colours."""
    yy, xx = np.mgrid[0:size, 0:size]
    span = float(max(1, size - 1))
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = 25
    img[..., 1] = np.clip(200 - (yy / span) * 130, 0, 255)
    img[..., 2] = np.clip(60 + (xx / span) * 65, 0, 255)
    _paint(img, 0.06, 0.44, 0.47, 0.94, (250, 250, 250))
    _paint(img, 0.63, 0.94, 0.06, 0.38, (10, 10, 10))
    return img


def upscale(image, factor=2):
    """Same content at higher resolution (nearest neighbour)."""
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def enlarge(image, factor=2):
    """Bigger crop with the same detail density, so only the area grows."""
    return np.tile(image, (factor, factor, 1))


def blur(image, ksize=9):
    """Softened variant used to test the sharpness ranking."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def jitter(image, amount=4, seed=7):
    """Near-identical variant: light sensor-style noise."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-amount, amount + 1, image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def brighten(image, amount=6):
    """Near-identical variant: slight exposure change."""
    return np.clip(image.astype(np.int16) + amount, 0, 255).astype(np.uint8)


# ──── classes.py ────

class TestClassRegistry:

    def test_nine_classes_in_stable_order(self):
        assert all_class_names() == EXPECTED_CLASSES
        assert len(SIGN_CLASSES) == 9

    def test_every_entry_has_exactly_the_required_keys(self):
        for name in EXPECTED_CLASSES:
            assert set(SIGN_CLASSES[name].keys()) == set(CLASS_KEYS), name

    @pytest.mark.parametrize("name", EXPECTED_CLASSES)
    def test_prompt_template_contains_text_placeholder(self, name):
        template = SIGN_CLASSES[name]["prompt_template"]
        assert "{text}" in template

    @pytest.mark.parametrize("name", EXPECTED_CLASSES)
    def test_template_renders_without_leftover_placeholders(self, name):
        rendered = build_prompt(name, "OPEN", "neon style")
        assert "{" not in rendered and "}" not in rendered
        assert "OPEN" in rendered

    @pytest.mark.parametrize("name", EXPECTED_CLASSES)
    def test_numeric_fields_are_sane(self, name):
        entry = SIGN_CLASSES[name]
        assert 0.0 < entry["threshold"] <= 1.0
        assert isinstance(entry["min_height_px"], int)
        assert entry["min_height_px"] > 0
        assert MIN_DENOISE_BIAS <= entry["denoise_bias"] <= MAX_DENOISE_BIAS

    @pytest.mark.parametrize("name", EXPECTED_CLASSES)
    def test_prompts_and_instruction_are_usable(self, name):
        entry = SIGN_CLASSES[name]
        assert entry["sam3_prompts"], name
        assert all(isinstance(p, str) and p.strip() for p in entry["sam3_prompts"])
        assert isinstance(entry["vlm_instruction"], str)
        assert len(entry["vlm_instruction"]) > 20

    def test_specific_registry_values(self):
        assert SIGN_CLASSES["plate"]["threshold"] == 0.35
        assert SIGN_CLASSES["plate"]["min_height_px"] == 20
        assert SIGN_CLASSES["graffiti"]["denoise_bias"] == -0.10
        assert SIGN_CLASSES["label"]["denoise_bias"] == 0.05
        assert SIGN_CLASSES["sign"]["sam3_prompts"] == ["sign", "street sign", "shop sign"]


class TestGetClass:

    def test_known_name(self):
        assert get_class("sign") == SIGN_CLASSES["sign"]

    def test_name_is_normalised(self):
        assert get_class("  SIGN  ") == SIGN_CLASSES["sign"]

    def test_unknown_name_falls_back(self):
        assert get_class("does_not_exist") == FALLBACK_CLASS

    @pytest.mark.parametrize("bad", ["", None, "   "])
    def test_empty_input_falls_back(self, bad):
        entry = get_class(bad)
        assert set(entry.keys()) == set(CLASS_KEYS)
        assert entry == FALLBACK_CLASS

    def test_returns_a_copy(self):
        entry = get_class("sign")
        entry["threshold"] = 0.99
        entry["sam3_prompts"].append("mutated")
        assert SIGN_CLASSES["sign"]["threshold"] == 0.30
        assert "mutated" not in SIGN_CLASSES["sign"]["sam3_prompts"]


class TestParseCustomPrompts:

    def test_typical_spec(self):
        assert parse_custom_prompts("bottle label:0.3, neon sign:0.25") == [
            ("bottle label", 0.3), ("neon sign", 0.25),
        ]

    def test_bare_prompt_gets_default_threshold(self):
        assert parse_custom_prompts("neon sign") == [("neon sign", DEFAULT_THRESHOLD)]

    def test_mixed_bare_and_explicit(self):
        assert parse_custom_prompts("neon sign, bottle label:0.4") == [
            ("neon sign", DEFAULT_THRESHOLD), ("bottle label", 0.4),
        ]

    @pytest.mark.parametrize("spec", ["", "   ", None, ",", " , , "])
    def test_empty_specs(self, spec):
        assert parse_custom_prompts(spec) == []

    def test_trailing_comma_is_ignored(self):
        assert parse_custom_prompts("shop sign:0.3,") == [("shop sign", 0.3)]

    def test_bad_float_falls_back_to_default(self):
        assert parse_custom_prompts("neon sign:abc") == [("neon sign", DEFAULT_THRESHOLD)]

    def test_entry_without_prompt_is_dropped(self):
        assert parse_custom_prompts(":0.5, real one:0.2") == [("real one", 0.2)]

    def test_whitespace_is_collapsed(self):
        assert parse_custom_prompts("  neon    sign  :  0.42 ") == [("neon sign", 0.42)]

    def test_threshold_is_clamped(self):
        assert parse_custom_prompts("a:5.0")[0][1] == 1.0
        assert parse_custom_prompts("b:-3")[0][1] == 0.01


class TestBuildPrompt:

    def test_text_is_inserted(self):
        assert "BAHNHOF" in build_prompt("sign", "BAHNHOF")

    def test_no_dangling_separators_without_style(self):
        out = build_prompt("poster", "SALE")
        assert ", ," not in out
        assert "  " not in out
        assert not out.startswith(",")
        assert not out.endswith(",")

    def test_style_is_included(self):
        out = build_prompt("poster", "SALE", "vintage print")
        assert "vintage print" in out
        assert "  " not in out

    def test_unknown_class_uses_fallback_template(self):
        out = build_prompt("nonsense_class", "HELLO")
        assert "HELLO" in out
        assert out == build_prompt("", "HELLO")

    @pytest.mark.parametrize("text", ["", None, "   "])
    def test_empty_text_is_safe(self, text):
        out = build_prompt("sign", text)
        assert isinstance(out, str)
        assert out

    def test_text_is_stripped(self):
        assert build_prompt("sign", "  OPEN  ") == build_prompt("sign", "OPEN")


class TestHelpers:

    def test_collapse_separators(self):
        assert collapse_separators("a  ,  , b ,,, c") == "a, b, c"
        assert collapse_separators(" , leading and trailing , ") == "leading and trailing"
        assert collapse_separators("") == ""
        assert collapse_separators(None) == ""

    @pytest.mark.parametrize("value,expected", [
        (0.0, 0.0), (0.1, 0.1), (0.9, MAX_DENOISE_BIAS), (-9.0, MIN_DENOISE_BIAS),
        ("nope", 0.0), (None, 0.0),
    ])
    def test_clamp_denoise_bias(self, value, expected):
        assert clamp_denoise_bias(value) == pytest.approx(expected)


# ──── slop.py ────

class TestLexiconTables:

    def test_word_list_size_and_normalisation(self):
        assert 400 <= len(WORD_LIST) <= 900
        assert all(word == word.upper() for word in WORD_LIST)
        assert all(word.strip() == word and word for word in WORD_LIST)

    def test_word_list_covers_both_languages(self):
        for word in ("OPEN", "COFFEE", "SHOP", "BAHNHOF", "STRASSE", "ACHTUNG"):
            assert word in WORD_LIST

    def test_bigram_table_size_and_range(self):
        assert 150 <= len(BIGRAM_FREQ) <= 320
        assert all(len(pair) == 2 and pair.isalpha() for pair in BIGRAM_FREQ)
        assert all(0 < weight <= MAX_BIGRAM_WEIGHT for weight in BIGRAM_FREQ.values())

    def test_implausible_pairs_are_absent(self):
        for pair in ("NQ", "QZ", "VX", "XZ", "JQ"):
            assert pair not in BIGRAM_FREQ

    def test_default_weights_cover_every_signal(self):
        assert set(DEFAULT_SLOP_WEIGHTS) == {
            "ocr_conf", "dictionary", "bigram", "repeat",
            "empty_but_detected", "vlm",
        }
        assert all(w > 0 for w in DEFAULT_SLOP_WEIGHTS.values())


class TestDictionaryRatio:

    def test_all_known(self):
        assert dictionary_ratio("COFFEE SHOP") == 1.0
        assert dictionary_ratio("bahnhof") == 1.0

    def test_none_known(self):
        assert dictionary_ratio("SHOPPINQ WRRLD") == 0.0

    def test_half_known(self):
        assert dictionary_ratio("COFFEE SHOPPINQ") == pytest.approx(0.5)

    def test_short_tokens_are_ignored(self):
        # "A" is a single character, so only "OPEN" counts.
        assert dictionary_ratio("A OPEN") == 1.0

    def test_pure_number_tokens_are_ignored(self):
        assert dictionary_ratio("OPEN 24/7") == 1.0

    def test_punctuation_is_stripped(self):
        assert dictionary_ratio("OPEN!") == 1.0

    def test_umlauts_are_folded(self):
        assert dictionary_ratio("BÄCKEREI") == 1.0

    @pytest.mark.parametrize("text", ["", "   ", None, "!!", "7"])
    def test_empty_input(self, text):
        assert dictionary_ratio(text) == 0.0


class TestBigramPlausibility:

    def test_real_words_score_high(self):
        assert bigram_plausibility("RESTAURANT") > 0.6
        assert bigram_plausibility("THE STATION") > 0.6

    def test_gibberish_scores_low(self):
        assert bigram_plausibility("XQZJ") < 0.2
        assert bigram_plausibility("WRRLD") < bigram_plausibility("WORLD")

    def test_result_is_bounded(self):
        for text in ("OPEN", "XQZJ", "BAHNHOF", "AENVX"):
            assert 0.0 <= bigram_plausibility(text) <= 1.0

    def test_unknown_pairs_count_as_zero(self):
        # "NQ" is not in the table, so SHOPPINQ must be below SHOPPING.
        assert bigram_plausibility("SHOPPINQ") < bigram_plausibility("SHOPPING")

    @pytest.mark.parametrize("text", ["", "   ", None, "A", "5"])
    def test_empty_input(self, text):
        assert bigram_plausibility(text) == 0.0


class TestRepeatedGlyphRatio:

    def test_full_run(self):
        assert repeated_glyph_ratio("AAAA") == 1.0

    def test_partial_run(self):
        assert repeated_glyph_ratio("OPENNNN") == pytest.approx(4 / 7)

    def test_double_letter_is_not_a_run(self):
        assert repeated_glyph_ratio("COFFEE") == 0.0

    def test_duplicated_pair_is_detected(self):
        assert repeated_glyph_ratio("ABABAB") == 1.0
        assert repeated_glyph_ratio("XYXY") == 1.0

    def test_clean_text_is_zero(self):
        assert repeated_glyph_ratio("OPEN") == 0.0
        assert repeated_glyph_ratio("COFFEE SHOP") == 0.0

    def test_result_is_bounded(self):
        for text in ("AAAAAA", "ABABABAB", "OPEN", "MMMOPENAAA"):
            assert 0.0 <= repeated_glyph_ratio(text) <= 1.0

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_input(self, text):
        assert repeated_glyph_ratio(text) == 0.0


class TestScoreSlopRules:
    """Every rule the pipeline relies on gets its own test."""

    def test_rule_empty_but_detected(self):
        result = score_slop(ocr_text="", ocr_conf=0.0, text_region_detected=True)
        assert result["signals"]["empty_but_detected"] == 1.0
        assert result["score"] >= 0.7
        assert result["verdict"] == "slop"

    def test_rule_empty_but_detected_whitespace_counts_as_empty(self):
        result = score_slop(ocr_text="   \n\t ", text_region_detected=True)
        assert result["signals"]["empty_but_detected"] == 1.0
        assert result["score"] >= 0.7

    def test_rule_empty_but_detected_floor_holds_with_high_confidence(self):
        # Even a nonsensically high confidence cannot push the score below the floor.
        result = score_slop(ocr_text="", ocr_conf=1.0, text_region_detected=True)
        assert result["score"] >= EMPTY_DETECTED_FLOOR

    @pytest.mark.parametrize("text", ["OPEN", "BAHNHOF", "COFFEE SHOP"])
    def test_rule_clean_real_text(self, text):
        result = score_slop(ocr_text=text, ocr_conf=0.9, text_region_detected=True)
        assert result["score"] <= 0.3
        assert result["verdict"] == "clean"

    @pytest.mark.parametrize("text", ["SHOPPINQ", "RESTAURENT", "WRRLD"])
    def test_rule_gibberish_with_mediocre_confidence(self, text):
        result = score_slop(ocr_text=text, ocr_conf=0.55, text_region_detected=True)
        assert result["score"] >= 0.5
        assert result["verdict"] == "slop"

    def test_rule_vlm_none_does_not_contribute(self):
        result = score_slop(ocr_text="SHOPPINQ", ocr_conf=0.55, text_region_detected=True)
        assert result["signals"]["vlm"] is None

        active = {k: v for k, v in result["signals"].items() if v is not None}
        total = sum(DEFAULT_SLOP_WEIGHTS[k] for k in active)
        expected = sum(DEFAULT_SLOP_WEIGHTS[k] * v for k, v in active.items()) / total
        assert result["score"] == pytest.approx(expected)

    def test_rule_vlm_shifts_the_score_when_given(self):
        without = score_slop(ocr_text="SHOPPINQ", ocr_conf=0.55)["score"]
        illegible = score_slop(ocr_text="SHOPPINQ", ocr_conf=0.55, vlm_legible=0.0)["score"]
        legible = score_slop(ocr_text="SHOPPINQ", ocr_conf=0.55, vlm_legible=1.0)["score"]
        assert legible < without < illegible

    def test_rule_vlm_accepts_booleans(self):
        assert score_slop(ocr_text="OPEN", ocr_conf=0.9, vlm_legible=True)["signals"]["vlm"] == 0.0
        assert score_slop(ocr_text="OPEN", ocr_conf=0.9, vlm_legible=False)["signals"]["vlm"] == 1.0

    def test_rule_all_inputs_absent_is_unknown(self):
        result = score_slop(
            ocr_text="", ocr_conf=0.0, char_confs=None,
            text_region_detected=False, vlm_legible=None,
        )
        assert result["verdict"] == "unknown"
        assert 0.0 <= result["score"] <= 1.0
        assert all(value is None for value in result["signals"].values())


class TestScoreSlopBehaviour:

    def test_result_shape(self):
        result = score_slop(ocr_text="OPEN", ocr_conf=0.9)
        assert set(result) == {"score", "verdict", "signals"}
        assert set(result["signals"]) == set(DEFAULT_SLOP_WEIGHTS)
        assert result["verdict"] in ("slop", "clean", "unknown")

    def test_score_always_bounded(self):
        cases = [
            dict(ocr_text="OPEN", ocr_conf=0.99),
            dict(ocr_text="XQZJ", ocr_conf=0.0),
            dict(ocr_text="AAAAAA", ocr_conf=0.5),
            dict(ocr_text="", text_region_detected=True),
            dict(ocr_text="OPEN", ocr_conf=42.0),
            dict(ocr_text="OPEN", ocr_conf=-5.0),
            dict(ocr_text="OPEN", ocr_conf=float("nan")),
        ]
        for kwargs in cases:
            assert 0.0 <= score_slop(**kwargs)["score"] <= 1.0, kwargs

    def test_non_string_text_does_not_crash(self):
        assert score_slop(ocr_text=None, text_region_detected=True)["verdict"] == "slop"
        assert 0.0 <= score_slop(ocr_text=1234, ocr_conf=0.8)["score"] <= 1.0

    def test_char_confs_are_used_when_region_conf_is_missing(self):
        weak = score_slop(ocr_text="OPEN", ocr_conf=0.0, char_confs=[0.1, 0.1, 0.1, 0.1])
        strong = score_slop(ocr_text="OPEN", ocr_conf=0.0, char_confs=[0.95, 0.95, 0.95, 0.95])
        assert weak["score"] > strong["score"]

    def test_one_bad_glyph_lowers_confidence(self):
        even = score_slop(ocr_text="OPEN", ocr_conf=0.9, char_confs=[0.9, 0.9, 0.9, 0.9])
        spiky = score_slop(ocr_text="OPEN", ocr_conf=0.9, char_confs=[0.9, 0.9, 0.9, 0.05])
        assert spiky["score"] > even["score"]

    def test_char_confs_tolerate_junk_entries(self):
        result = score_slop(ocr_text="OPEN", ocr_conf=0.9, char_confs=[None, "x", 0.9])
        assert 0.0 <= result["score"] <= 1.0

    def test_repeated_glyphs_push_towards_slop(self):
        assert score_slop(ocr_text="ABABABAB", ocr_conf=0.6)["score"] >= SLOP_THRESHOLD

    def test_weight_override_changes_the_score(self):
        base = score_slop(ocr_text="RESTAURENT", ocr_conf=0.55)["score"]
        lenient = score_slop(
            ocr_text="RESTAURENT", ocr_conf=0.55, weights={"dictionary": 0.0}
        )["score"]
        assert lenient < base

    def test_unknown_weight_keys_are_ignored(self):
        plain = score_slop(ocr_text="OPEN", ocr_conf=0.9)
        overridden = score_slop(ocr_text="OPEN", ocr_conf=0.9, weights={"nonsense": 5.0})
        assert plain["score"] == pytest.approx(overridden["score"])

    @pytest.mark.parametrize("weights", [None, {}, {"dictionary": "x"}, {"bigram": -1}, "junk"])
    def test_bad_weight_payloads_are_tolerated(self, weights):
        result = score_slop(ocr_text="OPEN", ocr_conf=0.9, weights=weights)
        assert result["verdict"] == "clean"

    def test_zero_weights_do_not_divide_by_zero(self):
        zeroed = {key: 0.0 for key in DEFAULT_SLOP_WEIGHTS}
        result = score_slop(ocr_text="OPEN", ocr_conf=0.9, weights=zeroed)
        assert 0.0 <= result["score"] <= 1.0


# ──── cluster.py ────

class TestPhash:

    def test_shape_and_dtype(self):
        bits = phash(make_label_crop())
        assert bits.shape == (64,)
        assert bits.dtype == np.bool_

    def test_custom_hash_size(self):
        assert phash(make_label_crop(), hash_size=4).shape == (16,)

    def test_stable_under_small_brightness_change(self):
        base = make_label_crop()
        changed = brighten(base, 12)
        assert int(np.sum(phash(base) != phash(changed))) <= 2

    def test_stable_under_light_noise(self):
        base = make_label_crop()
        assert float(np.mean(phash(base) != phash(jitter(base)))) < 0.15

    def test_different_images_differ(self):
        distance = float(np.mean(phash(make_label_crop()) != phash(make_poster_crop())))
        assert distance > 0.2

    def test_size_invariant(self):
        base = make_label_crop()
        upscaled = upscale(base, 3)
        assert np.array_equal(phash(base), phash(upscaled))

    def test_grayscale_input_is_accepted(self):
        gray = make_label_crop()[..., 0]
        assert phash(gray).shape == (64,)

    def test_float_tensor_style_input(self, small_image):
        arr = small_image[0].numpy()  # [H, W, 3] float32 0-1
        assert phash(arr).shape == (64,)

    @pytest.mark.parametrize("bad", [None, np.zeros((0, 0, 3), np.uint8), np.array([])])
    def test_empty_input_is_safe(self, bad):
        bits = phash(bad)
        assert bits.shape == (64,)
        assert not bits.any()


class TestColorSignature:

    def test_length_and_normalisation(self):
        hist = color_signature(make_label_crop())
        assert hist.ndim == 1
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)
        assert (hist >= 0).all()

    def test_size_invariant(self):
        base = make_label_crop()
        upscaled = upscale(base, 3)
        assert np.abs(color_signature(base) - color_signature(upscaled)).sum() < 0.1

    def test_different_colours_differ(self):
        delta = np.abs(
            color_signature(make_label_crop()) - color_signature(make_poster_crop())
        ).sum()
        assert delta > 0.5

    def test_float_input_is_accepted(self, small_image):
        hist = color_signature(small_image[0].numpy())
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("bad", [None, np.zeros((0, 0, 3), np.uint8)])
    def test_empty_input_is_safe(self, bad):
        hist = color_signature(bad)
        assert hist.sum() == 0.0
        assert hist.size > 0


class TestCropDistance:

    def test_identical_crops_have_zero_distance(self):
        feat = extract_features(make_label_crop())
        assert crop_distance(feat, feat) == 0.0

    def test_near_identical_crops_are_close(self):
        base = extract_features(make_label_crop())
        noisy = extract_features(jitter(make_label_crop()))
        assert crop_distance(base, noisy) < DEFAULT_CLUSTER_DISTANCE

    def test_different_crops_are_far(self):
        base = extract_features(make_label_crop())
        other = extract_features(make_poster_crop())
        assert crop_distance(base, other) > 0.3

    def test_symmetric(self):
        a = extract_features(make_label_crop())
        b = extract_features(make_poster_crop())
        assert crop_distance(a, b) == pytest.approx(crop_distance(b, a))

    def test_bounded(self):
        a = extract_features(make_label_crop())
        b = extract_features(make_poster_crop())
        assert 0.0 <= crop_distance(a, b) <= 1.0

    @pytest.mark.parametrize("a,b", [
        ({}, {}), (None, None), ({"phash": []}, {"hist": []}),
        ({"phash": np.zeros(64, bool)}, {"phash": np.zeros(16, bool)}),
    ])
    def test_missing_or_mismatched_features(self, a, b):
        assert crop_distance(a, b) == pytest.approx(1.0)


class TestExtractFeatures:

    def test_keys(self):
        feat = extract_features(make_label_crop())
        assert set(feat) == {"phash", "hist"}

    def test_independent_of_input_size(self):
        small = extract_features(make_label_crop(32))
        large = extract_features(make_label_crop(128))
        assert small["phash"].shape == large["phash"].shape
        assert small["hist"].shape == large["hist"].shape

    def test_empty_input_is_safe(self):
        feat = extract_features(None)
        assert feat["phash"].shape == (64,)
        assert feat["hist"].sum() == 0.0


class TestClusterCrops:

    def test_three_near_identical_plus_one_different(self):
        base = make_label_crop()
        crops = [base, jitter(base), brighten(base), make_poster_crop()]
        labels = cluster_crops(crops)
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] != labels[0]
        assert len(set(labels)) == 2

    def test_ids_start_at_zero_in_order_of_first_appearance(self):
        base = make_label_crop()
        crops = [make_poster_crop(), base, jitter(base)]
        labels = cluster_crops(crops)
        assert labels[0] == 0
        assert labels[1] == labels[2] == 1

    def test_singletons_get_their_own_id(self):
        crops = [make_label_crop(), make_poster_crop()]
        labels = cluster_crops(crops)
        assert sorted(labels) == [0, 1]
        assert all(label >= 0 for label in labels)

    def test_single_crop(self):
        assert cluster_crops([make_label_crop()]) == [0]

    def test_empty_input(self):
        assert cluster_crops([]) == []

    def test_generous_threshold_merges_everything(self):
        crops = [make_label_crop(), make_poster_crop()]
        assert cluster_crops(crops, distance=1.0) == [0, 0]

    def test_strict_threshold_keeps_only_exact_matches(self):
        base = make_label_crop()
        crops = [base, base.copy(), make_poster_crop()]
        labels = cluster_crops(crops, distance=0.0)
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]

    def test_label_count_matches_crop_count(self):
        crops = [make_label_crop(), jitter(make_label_crop()), make_poster_crop()]
        assert len(cluster_crops(crops)) == len(crops)

    def test_single_linkage_chains_through_a_bridge(self):
        base = make_label_crop()
        crops = [base, jitter(base, amount=3, seed=1), jitter(base, amount=3, seed=2)]
        assert len(set(cluster_crops(crops))) == 1

    def test_unusable_crops_do_not_crash(self):
        crops = [make_label_crop(), None, np.zeros((0, 0, 3), np.uint8)]
        labels = cluster_crops(crops)
        assert len(labels) == 3
        assert all(label >= 0 for label in labels)


class TestPickClusterRepresentative:

    def test_prefers_the_sharper_member(self):
        sharp = make_label_crop()
        crops = [blur(sharp), sharp]
        assert pick_cluster_representative(crops, [0, 0], 0) == 1

    def test_prefers_the_larger_member_at_equal_sharpness(self):
        small = make_label_crop()
        large = enlarge(small, 2)
        assert pick_cluster_representative([small, large], [0, 0], 0) == 1

    def test_only_looks_at_the_requested_cluster(self):
        base = make_label_crop()
        crops = [make_poster_crop(), base, enlarge(base, 2)]
        assert pick_cluster_representative(crops, [0, 1, 1], 0) == 0
        assert pick_cluster_representative(crops, [0, 1, 1], 1) == 2

    def test_unknown_cluster_returns_minus_one(self):
        assert pick_cluster_representative([make_label_crop()], [0], 5) == -1

    @pytest.mark.parametrize("crops,labels", [([], []), (None, None), ([], [0, 1])])
    def test_empty_input_returns_minus_one(self, crops, labels):
        assert pick_cluster_representative(crops, labels, 0) == -1

    def test_unusable_members_are_survivable(self):
        crops = [None, make_label_crop()]
        assert pick_cluster_representative(crops, [0, 0], 0) == 1


# ──── package surface ────

class TestPackageSurface:

    def test_public_api_is_importable(self):
        import core.signs as signs

        for name in (
            "SIGN_CLASSES", "all_class_names", "get_class", "parse_custom_prompts",
            "build_prompt", "WORD_LIST", "BIGRAM_FREQ", "DEFAULT_SLOP_WEIGHTS",
            "dictionary_ratio", "bigram_plausibility", "repeated_glyph_ratio",
            "score_slop", "phash", "color_signature", "crop_distance",
            "extract_features", "cluster_crops", "pick_cluster_representative",
        ):
            assert hasattr(signs, name), name
            assert name in signs.__all__, name
