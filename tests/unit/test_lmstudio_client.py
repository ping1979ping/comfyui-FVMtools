"""Unit tests for nodes.utils.lmstudio_client.

No real network traffic: every test that reaches HTTP monkeypatches
``urllib.request.urlopen``. A leaked real call would hang on the timeout, so
the fakes below also assert on the payload that would have gone out.
"""

import base64
import email.message
import io
import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from nodes.utils import lmstudio_client as lc


# ──── Fakes ────

class FakeResponse:
    """Minimal stand-in for the object urlopen returns (context manager + read)."""

    def __init__(self, body, status=200):
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self.status = status

    def read(self):
        return self._body

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(url, code, body):
    return urllib.error.HTTPError(
        url, code, "Not Found", email.message.Message(), io.BytesIO(body.encode())
    )


def install_urlopen(monkeypatch, handler, captured=None):
    """Patch urlopen with ``handler(request) -> FakeResponse`` and record requests."""
    def fake_urlopen(req, timeout=None, **kwargs):
        if captured is not None:
            captured.append({
                "url": req.full_url,
                "method": req.get_method(),
                "timeout": timeout,
                "body": json.loads(req.data.decode("utf-8")) if req.data else None,
            })
        return handler(req)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def chat_response(content, model="test-vlm"):
    return json.dumps({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant",
                                             "content": content},
                     "finish_reason": "stop"}],
        "usage": {"total_tokens": 42},
    })


VALID_PROPOSAL = json.dumps({
    "text": "BACKEREI",
    "style": "gold on dark green board",
    "font_hint": "high-contrast serif, all caps",
    "legible_original": 0.1,
    "confidence": 0.8,
})


@pytest.fixture
def rgb_image():
    """Deterministic 300x200 RGB image (H, W, C) uint8."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)


@pytest.fixture
def big_image():
    """2000x1200 image — long edge well above the default 1024 cap."""
    return np.full((1200, 2000, 3), 128, dtype=np.uint8)


# ──── encode_image_data_uri ────

class TestEncodeImageDataUri:

    def test_prefix_and_valid_png(self, rgb_image):
        uri = lc.encode_image_data_uri(rgb_image)
        assert uri.startswith("data:image/png;base64,")
        raw = base64.b64decode(uri.split(",", 1)[1])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", "payload must be a real PNG"

    def test_base64_roundtrips_through_cv2(self, rgb_image):
        import cv2
        uri = lc.encode_image_data_uri(rgb_image)
        raw = base64.b64decode(uri.split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (200, 300, 3)

    def test_honours_max_size_downscale(self, big_image):
        import cv2
        uri = lc.encode_image_data_uri(big_image, max_size=512)
        raw = base64.b64decode(uri.split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert max(decoded.shape[:2]) == 512
        # aspect ratio preserved: 2000x1200 -> 512x307
        assert decoded.shape[0] == 307 and decoded.shape[1] == 512

    def test_never_upscales(self, rgb_image):
        import cv2
        uri = lc.encode_image_data_uri(rgb_image, max_size=4096)
        raw = base64.b64decode(uri.split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == (200, 300)

    def test_one_by_one_image(self):
        tiny = np.array([[[255, 0, 0]]], dtype=np.uint8)
        uri = lc.encode_image_data_uri(tiny, max_size=64)
        assert uri.startswith("data:image/png;base64,")
        assert base64.b64decode(uri.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"

    def test_float_image_in_zero_one_range(self):
        img = np.ones((8, 8, 3), dtype=np.float32)
        uri = lc.encode_image_data_uri(img)
        assert uri.startswith("data:image/png;base64,")

    def test_grayscale_is_promoted_to_rgb(self):
        import cv2
        gray = np.full((16, 32), 200, dtype=np.uint8)
        uri = lc.encode_image_data_uri(gray)
        raw = base64.b64decode(uri.split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (16, 32, 3)

    def test_rgba_alpha_dropped(self):
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        assert lc.encode_image_data_uri(rgba).startswith("data:image/png;base64,")

    def test_jpeg_format(self, rgb_image):
        uri = lc.encode_image_data_uri(rgb_image, fmt="jpeg")
        assert uri.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(uri.split(",", 1)[1])[:2] == b"\xff\xd8"

    def test_invalid_input_raises_value_error(self):
        with pytest.raises(ValueError):
            lc.encode_image_data_uri(np.zeros((0,), dtype=np.uint8))


# ──── strip_thinking ────

class TestStripThinking:

    def test_paired_tags(self):
        out = lc.strip_thinking("<think>hmm, a bakery</think>{\"text\": \"BROT\"}")
        assert out == '{"text": "BROT"}'

    def test_thinking_variant_tag(self):
        assert lc.strip_thinking("<thinking>x</thinking> answer") == "answer"

    def test_nested_blocks(self):
        text = "<think>outer <think>inner</think> still outer</think>ANSWER"
        assert lc.strip_thinking(text) == "ANSWER"

    def test_unterminated_trailing_block(self):
        text = '{"text": "OK"}\n<think>I should reconsider because'
        assert lc.strip_thinking(text) == '{"text": "OK"}'

    def test_no_tags_at_all(self):
        assert lc.strip_thinking('{"text": "OK"}') == '{"text": "OK"}'

    def test_orphan_closing_tag_removed(self):
        assert lc.strip_thinking("lead</think>tail") == "leadtail"

    def test_non_string_and_empty(self):
        assert lc.strip_thinking(None) == ""
        assert lc.strip_thinking("") == ""

    def test_multiple_separate_blocks(self):
        text = "<think>a</think>KEEP1<think>b</think>KEEP2"
        assert lc.strip_thinking(text) == "KEEP1KEEP2"


# ──── parse_json_response ────

class TestParseJsonResponse:

    def test_bare_json(self):
        assert lc.parse_json_response('{"text": "OPEN"}') == {"text": "OPEN"}

    def test_fenced_json(self):
        raw = '```json\n{"text": "OPEN", "confidence": 0.9}\n```'
        assert lc.parse_json_response(raw)["text"] == "OPEN"

    def test_fenced_without_language(self):
        assert lc.parse_json_response('```\n{"text": "A"}\n```') == {"text": "A"}

    def test_prose_before_and_after(self):
        raw = ('Sure! Here is my answer:\n{"text": "CAFE", "confidence": 0.7}\n'
               "Let me know if you want another option.")
        assert lc.parse_json_response(raw) == {"text": "CAFE", "confidence": 0.7}

    def test_preceded_by_thinking_block(self):
        raw = ('<think>The scene is German, so a German word fits.</think>\n'
               '{"text": "BROT"}')
        assert lc.parse_json_response(raw) == {"text": "BROT"}

    def test_thinking_block_containing_braces(self):
        raw = ('<think>maybe {"text": "WRONG"} would work</think>'
               '{"text": "RIGHT"}')
        assert lc.parse_json_response(raw) == {"text": "RIGHT"}

    def test_trailing_comma(self):
        assert lc.parse_json_response('{"text": "A", "style": "b",}') == {
            "text": "A", "style": "b"}

    def test_single_quotes(self):
        assert lc.parse_json_response("{'text': 'OPEN', 'confidence': 0.5}") == {
            "text": "OPEN", "confidence": 0.5}

    def test_single_quotes_with_trailing_comma(self):
        assert lc.parse_json_response("{'text': 'OPEN',}") == {"text": "OPEN"}

    def test_nested_braces_inside_string_value(self):
        raw = '{"text": "MENU {daily}", "style": "chalk on a {board}"}'
        parsed = lc.parse_json_response(raw)
        assert parsed["text"] == "MENU {daily}"
        assert parsed["style"] == "chalk on a {board}"

    def test_nested_object_value(self):
        raw = '{"text": "A", "meta": {"lines": 2}}'
        assert lc.parse_json_response(raw)["meta"] == {"lines": 2}

    def test_garbage_returns_none(self):
        assert lc.parse_json_response("I am afraid I cannot help with that.") is None

    def test_broken_object_returns_none(self):
        assert lc.parse_json_response('{"text": ') is None

    def test_json_array_returns_none(self):
        assert lc.parse_json_response('["a", "b"]') is None

    def test_empty_and_non_string(self):
        assert lc.parse_json_response("") is None
        assert lc.parse_json_response(None) is None


# ──── normalize_proposal ────

class TestNormalizeProposal:

    def test_exact_key_set(self):
        result = lc.normalize_proposal({"text": "OPEN"})
        assert set(result) == set(lc.PROPOSAL_KEYS)

    def test_missing_keys_get_defaults(self):
        result = lc.normalize_proposal({"text": "OPEN"})
        assert result["style"] == ""
        assert result["font_hint"] == ""
        assert result["legible_original"] == 0.0
        assert result["confidence"] == 0.5

    @pytest.mark.parametrize("raw", ["85%", "0.85", 85, 0.85, "0,85"])
    def test_confidence_coercions(self, raw):
        result = lc.normalize_proposal({"text": "A", "confidence": raw})
        assert result["confidence"] == pytest.approx(0.85, abs=0.01)

    def test_unit_clamping(self):
        assert lc.normalize_proposal({"confidence": -3})["confidence"] == 0.0
        assert lc.normalize_proposal({"confidence": 1.0})["confidence"] == 1.0
        assert lc.normalize_proposal({"confidence": 999})["confidence"] == 1.0

    def test_boolean_and_word_scores(self):
        assert lc.normalize_proposal({"legible_original": True})[
            "legible_original"] == 1.0
        assert lc.normalize_proposal({"confidence": "high"})[
            "confidence"] == pytest.approx(0.9)

    def test_unparseable_number_falls_back_to_default(self):
        assert lc.normalize_proposal({"confidence": "banana"})["confidence"] == 0.5

    def test_quoted_text_unwrapped(self):
        assert lc.normalize_proposal({"text": '"OPEN 24H"'})["text"] == "OPEN 24H"
        assert lc.normalize_proposal({"text": "'CAFE'"})["text"] == "CAFE"
        assert lc.normalize_proposal({"text": "“BROT”"})["text"] == "BROT"

    def test_whitespace_collapsed(self):
        assert lc.normalize_proposal({"text": "  OPEN\n  24 H  "})[
            "text"] == "OPEN 24 H"

    def test_long_text_truncated_at_word_boundary(self):
        long_text = ("Fresh artisan sourdough bread baked every single morning "
                     "in our little neighbourhood bakery since nineteen twenty "
                     "eight and still going strong today with the very same "
                     "stone oven and the very same starter culture")
        assert len(long_text) >= 200
        result = lc.normalize_proposal({"text": long_text})["text"]
        assert len(result) <= lc.MAX_TEXT_CHARS
        assert not result.endswith(" ")
        # cut on a space, not mid-word
        assert long_text.startswith(result)
        assert long_text[len(result)] in " " or len(result) == lc.MAX_TEXT_CHARS

    def test_non_dict_input(self):
        for bad in (None, "just a string", 42, ["a"]):
            result = lc.normalize_proposal(bad)
            assert set(result) == set(lc.PROPOSAL_KEYS)
            assert result["text"] == ""

    def test_fallback_text_used_when_text_missing(self):
        assert lc.normalize_proposal({}, fallback_text="OPEN")["text"] == "OPEN"
        # a real text wins over the fallback
        assert lc.normalize_proposal({"text": "CAFE"},
                                     fallback_text="OPEN")["text"] == "CAFE"

    def test_style_is_capped(self):
        result = lc.normalize_proposal({"text": "A", "style": "x " * 400})
        assert len(result["style"]) <= lc.MAX_STYLE_CHARS


# ──── chat_vision ────

class TestChatVision:

    def test_happy_path(self, monkeypatch, rgb_image):
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(VALID_PROPOSAL)))
        result = lc.chat_vision(lc.DEFAULT_BASE_URL, "test-vlm",
                                lc.DEFAULT_SYSTEM_PROMPT, "what text?",
                                [rgb_image])
        assert result["ok"] is True
        assert result["error"] is None
        assert result["content"] == VALID_PROPOSAL
        assert result["raw"]["model"] == "test-vlm"

    def test_url_error_is_structured(self, monkeypatch, rgb_image):
        def boom(req):
            raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

        install_urlopen(monkeypatch, boom)
        result = lc.chat_vision(lc.DEFAULT_BASE_URL, "m", "sys", "usr",
                                [rgb_image], timeout=1)
        assert result["ok"] is False
        assert result["raw"] is None
        assert isinstance(result["error"], str) and result["error"]
        assert "localhost:1234" in result["error"]

    def test_http_404_is_structured(self, monkeypatch, rgb_image):
        def boom(req):
            raise _http_error(req.full_url, 404, '{"error": "model not found"}')

        install_urlopen(monkeypatch, boom)
        result = lc.chat_vision(lc.DEFAULT_BASE_URL, "nope", "sys", "usr",
                                [rgb_image], timeout=1)
        assert result["ok"] is False
        assert "404" in result["error"]
        assert "model not found" in result["error"]

    def test_timeout_is_structured(self, monkeypatch, rgb_image):
        def boom(req):
            raise TimeoutError("timed out")

        install_urlopen(monkeypatch, boom)
        result = lc.chat_vision(lc.DEFAULT_BASE_URL, "m", "sys", "usr",
                                [rgb_image], timeout=1)
        assert result["ok"] is False
        assert result["error"]

    def test_malformed_body_is_structured(self, monkeypatch, rgb_image):
        install_urlopen(monkeypatch, lambda req: FakeResponse("<html>nope</html>"))
        result = lc.chat_vision(lc.DEFAULT_BASE_URL, "m", "sys", "usr", [rgb_image])
        assert result["ok"] is False
        assert result["error"]

    def test_empty_message_is_not_ok(self, monkeypatch, rgb_image):
        install_urlopen(monkeypatch, lambda req: FakeResponse(chat_response("  ")))
        result = lc.chat_vision(lc.DEFAULT_BASE_URL, "m", "sys", "usr", [rgb_image])
        assert result["ok"] is False
        assert result["raw"] is not None

    def test_content_block_list_response(self, monkeypatch, rgb_image):
        body = json.dumps({"choices": [{"message": {
            "content": [{"type": "text", "text": '{"text": "A"}'}]}}]})
        install_urlopen(monkeypatch, lambda req: FakeResponse(body))
        result = lc.chat_vision(lc.DEFAULT_BASE_URL, "m", "sys", "usr", [rgb_image])
        assert result["ok"] is True
        assert result["content"] == '{"text": "A"}'

    def test_payload_shape_and_image_order(self, monkeypatch, rgb_image):
        captured = []
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(VALID_PROPOSAL)),
                        captured)

        scene = np.zeros((40, 40, 3), dtype=np.uint8)
        neighbor = np.full((20, 20, 3), 255, dtype=np.uint8)
        lc.chat_vision(lc.DEFAULT_BASE_URL, "vlm-1", "SYSTEM", "USER",
                       [rgb_image, scene, neighbor], temperature=0.2,
                       max_tokens=128, seed=7)

        assert len(captured) == 1
        call = captured[0]
        assert call["url"] == "http://localhost:1234/v1/chat/completions"
        assert call["method"] == "POST"

        body = call["body"]
        assert body["model"] == "vlm-1"
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 128
        assert body["seed"] == 7
        assert body["stream"] is False
        assert body["messages"][0] == {"role": "system", "content": "SYSTEM"}

        blocks = body["messages"][1]["content"]
        assert blocks[0] == {"type": "text", "text": "USER"}
        assert [b["type"] for b in blocks] == [
            "text", "image_url", "image_url", "image_url"]
        for block in blocks[1:]:
            assert block["image_url"]["url"].startswith("data:image/png;base64,")

        # documented order: crop, scene, neighbour — distinct payloads
        uris = [b["image_url"]["url"] for b in blocks[1:]]
        assert len(set(uris)) == 3

    def test_extra_options_passthrough(self, monkeypatch, rgb_image):
        captured = []
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(VALID_PROPOSAL)),
                        captured)
        lc.chat_vision(lc.DEFAULT_BASE_URL, "m", "sys", "usr", [rgb_image],
                       extra_options={"top_p": 0.9, "top_k": 40,
                                      "repeat_penalty": 1.1,
                                      "stop": ["</s>"], "ignored_key": "x"})
        body = captured[0]["body"]
        assert body["top_p"] == 0.9
        assert body["top_k"] == 40
        assert body["repeat_penalty"] == 1.1
        assert body["stop"] == ["</s>"]
        assert "ignored_key" not in body

    def test_seed_omitted_when_none(self, monkeypatch, rgb_image):
        captured = []
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(VALID_PROPOSAL)),
                        captured)
        lc.chat_vision(lc.DEFAULT_BASE_URL, "m", "sys", "usr", [rgb_image],
                       seed=None)
        assert "seed" not in captured[0]["body"]

    def test_timeout_is_passed_to_urlopen(self, monkeypatch, rgb_image):
        captured = []
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(VALID_PROPOSAL)),
                        captured)
        lc.chat_vision(lc.DEFAULT_BASE_URL, "m", "sys", "usr", [rgb_image],
                       timeout=33)
        assert captured[0]["timeout"] == 33

    def test_unencodable_image_is_skipped(self, monkeypatch, rgb_image):
        captured = []
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(VALID_PROPOSAL)),
                        captured)
        lc.chat_vision(lc.DEFAULT_BASE_URL, "m", "sys", "usr",
                       [rgb_image, np.zeros((0,), dtype=np.uint8), None])
        blocks = captured[0]["body"]["messages"][1]["content"]
        assert [b["type"] for b in blocks] == ["text", "image_url"]


# ──── probe / list_models ────

class TestProbe:

    def test_reachable(self, monkeypatch):
        body = json.dumps({"data": [{"id": "qwen2-vl-7b"}, {"id": "llava-1.6"}]})
        install_urlopen(monkeypatch, lambda req: FakeResponse(body))
        result = lc.probe()
        assert result == {"reachable": True,
                          "models": ["qwen2-vl-7b", "llava-1.6"],
                          "error": None}

    def test_unreachable_does_not_raise(self, monkeypatch):
        def boom(req):
            raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

        install_urlopen(monkeypatch, boom)
        result = lc.probe(timeout=1)
        assert result["reachable"] is False
        assert result["models"] == []
        assert isinstance(result["error"], str) and result["error"]

    def test_http_error_reported(self, monkeypatch):
        install_urlopen(monkeypatch,
                        lambda req: (_ for _ in ()).throw(
                            _http_error(req.full_url, 404, "nope")))
        result = lc.probe(timeout=1)
        assert result["reachable"] is False
        assert "404" in result["error"]

    def test_uses_models_endpoint(self, monkeypatch):
        captured = []
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(json.dumps({"data": []})),
                        captured)
        lc.probe(base_url="http://127.0.0.1:9999/v1/", timeout=3)
        assert captured[0]["url"] == "http://127.0.0.1:9999/v1/models"
        assert captured[0]["method"] == "GET"
        assert captured[0]["timeout"] == 3


class TestListModels:

    def test_ids_extracted(self, monkeypatch):
        body = json.dumps({"data": [{"id": "a"}, {"id": "b"}]})
        install_urlopen(monkeypatch, lambda req: FakeResponse(body))
        assert lc.list_models() == ["a", "b"]

    def test_failure_returns_empty_list(self, monkeypatch):
        def boom(req):
            raise urllib.error.URLError("refused")

        install_urlopen(monkeypatch, boom)
        assert lc.list_models(timeout=1) == []

    def test_unexpected_payload_returns_empty_list(self, monkeypatch):
        install_urlopen(monkeypatch, lambda req: FakeResponse('{"oops": 1}'))
        assert lc.list_models() == []


# ──── propose_text ────

class TestProposeText:

    def test_happy_path(self, monkeypatch, rgb_image):
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(VALID_PROPOSAL)))
        scene = np.zeros((64, 64, 3), dtype=np.uint8)
        result = lc.propose_text(rgb_image, scene_rgb=scene,
                                 class_name="shop sign",
                                 scene_hint="a german street at dusk",
                                 model_id="vlm-1")
        assert set(result) == set(lc.PROPOSAL_KEYS) | {"ok", "error", "source"}
        assert result["ok"] is True
        assert result["error"] is None
        assert result["source"] == "vlm"
        assert result["text"] == "BACKEREI"
        assert result["confidence"] == pytest.approx(0.8)
        assert result["legible_original"] == pytest.approx(0.1)

    def test_chatty_model_answer_is_parsed(self, monkeypatch, rgb_image):
        chatty = ("<think>German scene, so German wording.</think>\n"
                  "Here you go:\n```json\n" + VALID_PROPOSAL + "\n```\nHope that helps!")
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(chatty)))
        result = lc.propose_text(rgb_image)
        assert result["ok"] is True and result["source"] == "vlm"
        assert result["text"] == "BACKEREI"

    def test_connection_failure_falls_back(self, monkeypatch, rgb_image):
        def boom(req):
            raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

        install_urlopen(monkeypatch, boom)
        result = lc.propose_text(rgb_image, timeout=1, fallback_text="OPEN")
        assert result["ok"] is False
        assert result["source"] == "fallback"
        assert result["text"] == "OPEN"
        assert isinstance(result["error"], str) and result["error"]
        assert set(result) == set(lc.PROPOSAL_KEYS) | {"ok", "error", "source"}

    def test_unparsable_answer_falls_back(self, monkeypatch, rgb_image):
        install_urlopen(
            monkeypatch,
            lambda req: FakeResponse(chat_response("I cannot read that sign.")))
        result = lc.propose_text(rgb_image, fallback_text="EXIT")
        assert result["ok"] is False
        assert result["source"] == "fallback"
        assert result["text"] == "EXIT"

    def test_missing_crop_falls_back_without_network(self, monkeypatch):
        def boom(req):
            raise AssertionError("no HTTP call expected without a crop")

        install_urlopen(monkeypatch, boom)
        result = lc.propose_text(None, fallback_text="SIGN")
        assert result["ok"] is False and result["source"] == "fallback"
        assert result["text"] == "SIGN"

    def test_image_order_crop_scene_neighbors(self, monkeypatch, rgb_image):
        captured = []
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(VALID_PROPOSAL)),
                        captured)

        scene = np.zeros((50, 50, 3), dtype=np.uint8)
        neighbors = [np.full((10, 10, 3), v, dtype=np.uint8)
                     for v in (10, 20, 30, 40, 50)]
        lc.propose_text(rgb_image, scene_rgb=scene, neighbor_crops=neighbors)

        blocks = captured[0]["body"]["messages"][1]["content"]
        # 1 text + crop + scene + max 3 neighbours
        assert [b["type"] for b in blocks] == ["text"] + ["image_url"] * 5

        expected = [lc.encode_image_data_uri(img)
                    for img in [rgb_image, scene] + neighbors[:3]]
        assert [b["image_url"]["url"] for b in blocks[1:]] == expected

    def test_prompt_mentions_class_hint_and_language(self, monkeypatch, rgb_image):
        captured = []
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(VALID_PROPOSAL)),
                        captured)
        lc.propose_text(rgb_image, class_name="book spine",
                        class_instruction="Give a plausible book title.",
                        scene_hint="a cluttered study",
                        language="German")
        prompt = captured[0]["body"]["messages"][1]["content"][0]["text"]
        assert "book spine" in prompt
        assert "Give a plausible book title." in prompt
        assert "a cluttered study" in prompt
        assert "German" in prompt

    def test_default_system_prompt_is_sent(self, monkeypatch, rgb_image):
        captured = []
        install_urlopen(monkeypatch,
                        lambda req: FakeResponse(chat_response(VALID_PROPOSAL)),
                        captured)
        lc.propose_text(rgb_image)
        system = captured[0]["body"]["messages"][0]["content"]
        assert system == lc.DEFAULT_SYSTEM_PROMPT
        for key in lc.PROPOSAL_KEYS:
            assert key in system


# ──── Module hygiene ────

class TestModuleContract:

    def test_defaults(self):
        assert lc.DEFAULT_BASE_URL == "http://localhost:1234/v1"
        assert lc.DEFAULT_TIMEOUT == 120

    def test_default_temperature_stays_below_the_measured_cliff(self):
        """Measured against qwen3-8b-vl on a garbled-signage crop: the model
        echoed the gibberish in 0/6 runs at temperature 0.2 but 3/6 at 0.25.
        The cliff is sharp, so the default must stay at or below 0.2."""
        assert lc.DEFAULT_TEMPERATURE <= 0.2
        import inspect
        for fn in (lc.chat_vision, lc.propose_text):
            default = inspect.signature(fn).parameters["temperature"].default
            assert default == lc.DEFAULT_TEMPERATURE

    def test_system_prompt_demands_single_json_object(self):
        prompt = lc.DEFAULT_SYSTEM_PROMPT.lower()
        assert "one json object" in prompt
        assert "trademark" in prompt
        assert "language" in prompt

    @staticmethod
    def _flat(text):
        """Collapse whitespace so assertions survive re-wrapping of the prompt."""
        return " ".join(text.split()).lower()

    def test_system_prompt_forbids_transcribing_the_gibberish(self):
        """Regression guard for the live failure where a scene_hint made the
        model transcribe the garbled lettering instead of replacing it
        ("CAFFEE RSTRNQ" -> "CAFFEE RSTRNC").

        Without these instructions the node re-renders the same nonsense, so a
        later prompt edit must not silently drop them.
        """
        prompt = self._flat(lc.DEFAULT_SYSTEM_PROMPT)
        assert "do not transcribe them" in prompt
        assert "do not spell-correct them" in prompt
        assert "do not use them as a starting point" in prompt
        assert "correctly spelled dictionary word" in prompt
        assert "replace every garbled token" in prompt
        # the scene hint must be scoped to setting/language/style, not letters
        assert "it never tells you the letters" in prompt

    def test_system_prompt_guards_the_near_miss_spelling(self):
        """The stubborn live failure was a near-miss the model believed was a
        real foreign word, so the doubled-letter guard must stay in place."""
        prompt = self._flat(lc.DEFAULT_SYSTEM_PROMPT)
        assert "beware the near-miss" in prompt
        assert "never licenses a misspelling" in prompt

    def test_user_prompt_repeats_the_anti_transcription_rule(self):
        """The hint line itself is the trigger — it must carry the caveat, and
        the reminder must survive in the final (highest-recency) line."""
        prompt = lc.build_user_prompt(class_name="sign",
                                      scene_hint="Vienna, old town, evening")
        flat = self._flat(prompt)
        assert "do not transcribe" in flat
        assert "does not tell you the letters" in flat
        assert "Vienna, old town, evening" in prompt
        assert "gibberish" in prompt.strip().splitlines()[-1].lower()

    def test_legible_original_semantics_unchanged(self):
        prompt = self._flat(lc.DEFAULT_SYSTEM_PROMPT)
        assert "1.0 means it is clean, real, readable writing that needs no repair" in prompt
        assert "0.0 means it is ai gibberish" in prompt

    def test_no_requests_dependency(self):
        import sys
        source = open(lc.__file__, encoding="utf-8").read()
        assert "import requests" not in source
        assert "custom_nodes" not in source
        del sys  # keep flake-ish linters quiet


class TestAvoidTexts:
    """Each region is its own request, so the model has no memory of its earlier
    answers. Measured on four near-identical shopfronts: 1/4 distinct texts
    without this constraint, 4/4 with it.
    """

    def test_listed_texts_appear_in_the_prompt(self):
        p = lc.build_user_prompt(avoid_texts=["BAECKEREI", "APOTHEKE"])
        assert "BAECKEREI" in p and "APOTHEKE" in p
        assert "must be clearly different" in p

    def test_no_line_added_without_any(self):
        for empty in (None, [], ["", "   "]):
            assert "already reads" not in lc.build_user_prompt(avoid_texts=empty)

    def test_duplicates_collapse(self):
        p = lc.build_user_prompt(avoid_texts=["OPEN", "OPEN", "OPEN"])
        assert p.count('"OPEN"') == 1

    def test_long_lists_are_capped(self):
        many = [f"TEXT{i}" for i in range(40)]
        p = lc.build_user_prompt(avoid_texts=many)
        listed = sum(1 for t in many if f'"{t}"' in p)
        assert listed <= 12, "an unbounded list would crowd out the actual instructions"
        assert "TEXT39" in p, "the most recent entries are the ones worth keeping"

    def test_the_anti_transcription_rule_still_comes_last(self):
        """The recency slot must not be taken over by the avoid list."""
        p = lc.build_user_prompt(avoid_texts=["OPEN"])
        assert "gibberish" in p.splitlines()[-1].lower() or \
               "transcribe" in p.splitlines()[-1].lower()

    def test_propose_text_passes_it_through(self, monkeypatch):
        captured = {}

        def fake_chat(*args, **kwargs):
            captured["prompt"] = kwargs.get("user_prompt") or args[3]
            return {"ok": True, "content": '{"text":"NEU"}', "error": None, "raw": None}

        monkeypatch.setattr("nodes.utils.lmstudio_client.chat_vision", fake_chat)
        lc.propose_text(crop_rgb=np.zeros((8, 8, 3), np.uint8),
                     avoid_texts=["SCHON VERGEBEN"])
        assert "SCHON VERGEBEN" in captured["prompt"]
