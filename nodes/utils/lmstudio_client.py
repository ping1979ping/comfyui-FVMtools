"""LM Studio vision client — "what text SHOULD be here?" proposals over plain HTTP.

FVMtools stays standalone: no ``requests`` dependency and no imports from other
ComfyUI extensions. Everything here speaks to an OpenAI-compatible endpoint
(LM Studio's default ``http://localhost:1234/v1``) through stdlib
``urllib.request``.

Contract for the whole module:

- **Nothing blocks or touches the network at import time.** Import is pure
  stdlib + numpy + cv2 setup.
- **Every network call carries an explicit timeout** and degrades gracefully:
  connection refused, DNS failure, timeout, HTTP error and malformed JSON are
  all turned into a structured failure value. No exception escapes a public
  function.

The high-level entry point is :func:`propose_text`, which sends the garbled
crop plus the surrounding scene to a vision model and returns a normalised
proposal dict::

    {"text": str, "style": str, "font_hint": str,
     "legible_original": float, "confidence": float,
     "ok": bool, "error": str | None, "source": "vlm" | "fallback"}
"""

from __future__ import annotations

import base64
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

# ──── Constants ────

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TIMEOUT = 120
DEFAULT_PROBE_TIMEOUT = 5
DEFAULT_MODELS_TIMEOUT = 10

DEFAULT_MAX_IMAGE_SIZE = 1024
DEFAULT_JPEG_QUALITY = 92

#: Sampling temperature. Deliberately low: choosing the right word for a sign is
#: a low-entropy task, not a creative one. Above a sharp threshold the model
#: stops inventing and instead returns a near-miss of the very gibberish it was
#: asked to replace (e.g. keeping a doubled-letter pseudo-word because a foreign
#: setting made it feel authentic).
#:
#: Measured against qwen3-8b-vl-instruct on a garbled-signage crop, counting how
#: often the answer echoed the garbled lettering:
#:     0.0 → 0/4   0.1 → 0/6   0.2 → 0/6   0.25 → 3/6   0.3 → 3/6
#:     0.4 → 3/6   0.6 → 2/4   0.8 → 2/4
#: The cliff sits between 0.2 and 0.25 and is not a gentle slope. Do NOT raise
#: this without re-running the sweep — 0.25 already fails half the time.
DEFAULT_TEMPERATURE = 0.2

#: Hard cap on a proposal's ``text`` — signage lettering has to fit in a crop.
MAX_TEXT_CHARS = 64
#: Cap on the free-form descriptive fields, so a rambling model can't blow up
#: a downstream prompt.
MAX_STYLE_CHARS = 160

#: Neighbour crops are context, not the subject — three is plenty.
MAX_NEIGHBOR_IMAGES = 3

#: Sampling knobs forwarded verbatim to the server when present in
#: ``extra_options``. Anything else is ignored so a stray UI widget can't
#: poison the payload.
PASSTHROUGH_OPTIONS = (
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "repeat_penalty",
    "stop",
)

#: The exact key set a proposal must expose.
PROPOSAL_KEYS = ("text", "style", "font_hint", "legible_original", "confidence")

DEFAULT_SYSTEM_PROMPT = """\
You are a plausible-signage-text engine inside an image-repair pipeline.

An AI image generator drew a sign, label, poster, badge, book spine, screen or
garment print whose lettering came out as nonsense. You are shown the cropped
region first, then the full scene, then (optionally) a few neighbouring regions.
Invent the text that SHOULD be there, so it can be re-rendered as real writing.

THE RULE THAT MATTERS MOST: the letters in the crop are not words. Do not
transcribe them, do not spell-correct them, do not change one letter and keep
the rest, do not use them as a starting point. Every word you output must be a
correctly spelled dictionary word in a real language, and none of them may be a
near-spelling of anything in the crop. Replace every garbled token, not just the
first one.

Rules:
1. Answer with ONE JSON object and nothing else. No markdown fence, no ```json,
   no preamble, no explanation, no trailing commentary.
2. The object has exactly these five keys:
   {"text", "style", "font_hint", "legible_original", "confidence"}
   Use no other keys and omit none of them.
3. "text" is short — a real sign is a few words, not a sentence — and fits the
   object class and the scene: a bakery window gets a bakery word, a fire door
   gets an exit word, a street sign gets a street name. Spell every word out in
   full, vowels included; never emit an abbreviation, an acronym or a vowel-less
   consonant skeleton (write "PHARMACY", never "PHRMCY"). If a garbled token
   cannot become a real word, drop it — a shorter correct sign beats a longer
   one with nonsense in it. Use "\\n" if the region holds more than one line.
4. Never invent or reproduce a real trademark, brand, logotype or company name.
   Invent neutral, generic wording instead.
5. A scene description or hint constrains only the SETTING, the LANGUAGE and the
   STYLE. It never tells you the letters, it is never a reason to stay faithful
   to what is painted there, and it never makes a misspelling acceptable.
6. Match the language of the scene. If the surroundings are German, answer in
   German; if Japanese, answer in Japanese; and so on. Do not translate to
   English unless the scene itself is English.
7. "style" describes the sign's visual character in a few words, e.g.
   "weathered enamel plate, white on dark blue". "font_hint" names a plausible
   lettering style, e.g. "bold condensed grotesque, all caps" — describe the
   shapes, do not name a licensed typeface.
8. "legible_original" is a number from 0.0 to 1.0 judging the text ALREADY in
   the crop: 1.0 means it is clean, real, readable writing that needs no repair;
   0.0 means it is AI gibberish — fake letterforms, melted glyphs, nonsense.
   It scores what you were given; it never licenses copying it.
9. "confidence" is a number from 0.0 to 1.0 stating how sure you are that your
   proposed "text" fits this scene.

Beware the near-miss — it is the artefact you will most often be tempted to keep.
A doubled letter or a dropped accent makes a nonsense string look like a real
foreign word. It is still nonsense. A foreign, historical or exotic setting never
licenses a misspelling:
  crop "HOTELL BARR"       WRONG "HOTELL BAR"        RIGHT "HOTEL BAR"
  crop "PIZZARIA MRKT"     WRONG "PIZZARIA MRKT"     RIGHT "PIZZERIA"
  crop "BOKSTOER LIBRARE"  WRONG "BOOKSTOER LIBRARE" RIGHT "LIVRARIA"

And never blend two languages into a hybrid spelling. Neighbouring languages
spell the same everyday shop word differently, and an image generator loves to
land halfway between them — that halfway form is a real word in neither language
and is exactly the kind of string you must not keep. Choose ONE language that
suits the scene and use that language's exact dictionary spelling, accents and
all. If the setting makes a hybrid feel authentic, that is the trap, not a
licence.

Valid answer (structure only — invent your own content; note the correctly
spelled, fully accented word):
{"text": "BÄCKEREI", "style": "gold leaf on dark green shopfront board",
 "font_hint": "high-contrast serif, all caps", "legible_original": 0.1,
 "confidence": 0.8}
"""

# ──── Regex helpers ────

_THINK_OPEN_RE = re.compile(r"<\s*(?:think|thinking)\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*(?:think|thinking)\s*>", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*|\s*```\s*$")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_WHITESPACE_RE = re.compile(r"\s+")
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("`", "`"), ("“", "”"),
                ("‘", "’"), ("«", "»"))

_WORD_SCORES = {
    "none": 0.0, "no": 0.0, "false": 0.0, "never": 0.0,
    "very low": 0.1, "low": 0.2, "unlikely": 0.25,
    "medium": 0.5, "moderate": 0.5, "maybe": 0.5, "unsure": 0.5,
    "high": 0.9, "likely": 0.8, "very high": 0.95,
    "yes": 1.0, "true": 1.0, "certain": 1.0, "sure": 1.0,
}


# ──── Image encoding ────

def _to_rgb_uint8(image_rgb) -> np.ndarray:
    """Coerce arbitrary array-ish input into an (H, W, 3) uint8 RGB image."""
    arr = np.asarray(image_rgb)
    if arr.size == 0 or arr.ndim not in (2, 3):
        raise ValueError(f"unsupported image shape {getattr(arr, 'shape', None)}")

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:          # drop alpha
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise ValueError(f"expected 3 channels, got {arr.shape[2]}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        peak = float(arr.max()) if arr.size else 0.0
        if peak <= 1.0001:         # float image in [0, 1]
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(arr)


def _downscale(arr: np.ndarray, max_size: int) -> np.ndarray:
    """Shrink so the long edge is at most ``max_size``. Never upscales."""
    h, w = arr.shape[:2]
    long_edge = max(h, w)
    if max_size <= 0 or long_edge <= max_size:
        return arr
    scale = float(max_size) / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_image_data_uri(image_rgb, max_size: int = DEFAULT_MAX_IMAGE_SIZE,
                          fmt: str = "png") -> str:
    """Encode an RGB numpy image as an OpenAI-style ``data:`` URI.

    The image is downscaled so its long edge is at most ``max_size`` (never
    upscaled), then encoded as PNG (default) or JPEG.

    Raises ``ValueError`` if the array cannot be interpreted as an image — the
    callers in this module catch that and turn it into a structured failure.
    """
    arr = _downscale(_to_rgb_uint8(image_rgb), int(max_size))

    fmt = (fmt or "png").lower().lstrip(".")
    if fmt in ("jpg", "jpeg"):
        ext, mime = ".jpg", "jpeg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), DEFAULT_JPEG_QUALITY]
    else:
        ext, mime = ".png", "png"
        params = []

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise ValueError(f"cv2.imencode failed for format {ext}")

    payload = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/{mime};base64,{payload}"


# ──── HTTP plumbing ────

def _join_url(base_url: str, path: str) -> str:
    return f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/{path.lstrip('/')}"


def _describe_error(exc: Exception, url: str) -> str:
    """Turn a urllib/socket exception into a short, actionable message."""
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace").strip()
        except Exception:  # pragma: no cover - body already consumed
            body = ""
        if len(body) > 300:
            body = body[:300] + "…"
        return f"HTTP {exc.code} from {url}" + (f": {body}" if body else "")
    if isinstance(exc, urllib.error.URLError):
        return f"Cannot reach LM Studio at {url}: {exc.reason}"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return f"Timed out talking to {url}"
    if isinstance(exc, json.JSONDecodeError):
        return f"Malformed JSON from {url}: {exc}"
    return f"{type(exc).__name__} talking to {url}: {exc}"


def _request_json(url: str, timeout: float, payload: dict | None = None) -> tuple:
    """GET (payload=None) or POST JSON. Returns ``(data, error)``; never raises."""
    try:
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            url, data=data, headers=headers,
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        if isinstance(body, bytes):
            body = body.decode("utf-8", "replace")
        return json.loads(body), None
    except Exception as exc:  # noqa: BLE001 — deliberate: never raise outwards
        return None, _describe_error(exc, url)


def list_models(base_url: str = DEFAULT_BASE_URL,
                timeout: float = DEFAULT_MODELS_TIMEOUT) -> list:
    """Return the model ids the server advertises. Empty list on any failure."""
    data, error = _request_json(_join_url(base_url, "models"), timeout)
    if error or not isinstance(data, dict):
        return []
    entries = data.get("data")
    if not isinstance(entries, list):
        return []
    ids = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id"):
            ids.append(str(entry["id"]))
        elif isinstance(entry, str):
            ids.append(entry)
    return ids


def probe(base_url: str = DEFAULT_BASE_URL,
          timeout: float = DEFAULT_PROBE_TIMEOUT) -> dict:
    """Cheap reachability check for UI status lines.

    Returns ``{"reachable": bool, "models": [...], "error": str | None}``.
    Never raises — an unreachable server is a normal, reportable state.
    """
    data, error = _request_json(_join_url(base_url, "models"), timeout)
    if error:
        return {"reachable": False, "models": [], "error": error}
    if not isinstance(data, dict):
        return {"reachable": False, "models": [],
                "error": f"Unexpected /models payload: {type(data).__name__}"}

    entries = data.get("data") if isinstance(data.get("data"), list) else []
    models = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id"):
            models.append(str(entry["id"]))
        elif isinstance(entry, str):
            models.append(entry)
    return {"reachable": True, "models": models, "error": None}


# ──── Chat completion ────

def _content_blocks(user_prompt: str, images, max_size: int) -> list:
    """Text block first, then one ``image_url`` block per encodable image."""
    blocks = [{"type": "text", "text": user_prompt or ""}]
    for image in images or []:
        if image is None:
            continue
        try:
            uri = encode_image_data_uri(image, max_size=max_size)
        except Exception:  # noqa: BLE001 — a bad crop must not kill the call
            continue
        blocks.append({"type": "image_url", "image_url": {"url": uri}})
    return blocks


def _extract_content(raw: dict) -> str:
    """Pull the assistant text out of an OpenAI-shaped response."""
    if not isinstance(raw, dict):
        return ""
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        # Some servers answer /completions-style with a bare "text" field.
        return str(choices[0].get("text", "")) if isinstance(choices[0], dict) else ""

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal servers may answer with content blocks.
        parts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(parts)
    return "" if content is None else str(content)


def chat_vision(base_url: str, model_id: str, system_prompt: str, user_prompt: str,
                images, temperature: float = DEFAULT_TEMPERATURE,
                max_tokens: int = 256,
                seed=None, timeout: float = DEFAULT_TIMEOUT,
                extra_options: dict | None = None,
                max_image_size: int = DEFAULT_MAX_IMAGE_SIZE) -> dict:
    """POST one vision chat completion.

    ``images`` is a list of RGB numpy arrays; they become ``image_url`` content
    blocks in the given order, after the text block.

    Returns ``{"ok": bool, "content": str, "error": str | None, "raw": dict | None}``.
    Never raises.
    """
    try:
        blocks = _content_blocks(user_prompt, images, max_image_size)
    except Exception as exc:  # noqa: BLE001 - defensive; _content_blocks swallows
        return {"ok": False, "content": "",
                "error": f"Image encoding failed: {exc}", "raw": None}

    payload = {
        "model": model_id or "",
        "messages": [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": blocks},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if seed is not None:
        try:
            payload["seed"] = int(seed)
        except (TypeError, ValueError):
            pass

    for key in PASSTHROUGH_OPTIONS:
        if isinstance(extra_options, dict) and extra_options.get(key) is not None:
            payload[key] = extra_options[key]

    url = _join_url(base_url, "chat/completions")
    raw, error = _request_json(url, timeout, payload=payload)
    if error:
        return {"ok": False, "content": "", "error": error, "raw": None}

    content = _extract_content(raw)
    if not content.strip():
        return {"ok": False, "content": "",
                "error": "Model returned an empty message", "raw": raw}
    return {"ok": True, "content": content, "error": None, "raw": raw}


# ──── Response parsing ────

def strip_thinking(text) -> str:
    """Remove ``<think>``/``<thinking>`` blocks, including an unterminated one.

    Nesting is handled by depth counting, and an opening tag with no closing tag
    swallows the rest of the string (reasoning models truncated by ``max_tokens``
    look exactly like that).
    """
    if not isinstance(text, str) or not text:
        return ""

    out = []
    pos = 0
    while True:
        opening = _THINK_OPEN_RE.search(text, pos)
        if not opening:
            out.append(text[pos:])
            break
        out.append(text[pos:opening.start()])

        depth = 1
        cursor = opening.end()
        while depth > 0:
            nxt_open = _THINK_OPEN_RE.search(text, cursor)
            nxt_close = _THINK_CLOSE_RE.search(text, cursor)
            if nxt_close is None:
                cursor = len(text)      # unterminated → drop the remainder
                break
            if nxt_open is not None and nxt_open.start() < nxt_close.start():
                depth += 1
                cursor = nxt_open.end()
            else:
                depth -= 1
                cursor = nxt_close.end()
        pos = cursor

    # Orphaned closing tags can survive malformed nesting.
    return _THINK_CLOSE_RE.sub("", "".join(out)).strip()


def _strip_fences(text: str) -> str:
    """Drop markdown code fences, keeping the fenced body."""
    if "```" not in text:
        return text
    lines = [ln for ln in text.splitlines()
             if not ln.strip().startswith("```")]
    stripped = "\n".join(lines)
    return stripped if stripped.strip() else _FENCE_RE.sub("", text)


def _extract_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` span, ignoring braces inside strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        quote = ""
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    in_string = False
                continue
            if ch in ('"', "'"):
                in_string = True
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _repair_single_quotes(candidate: str) -> str:
    """Last-resort ``'`` → ``"`` swap for Python-dict-ish output."""
    if '"' not in candidate:
        return candidate.replace("'", '"')
    # Mixed quoting: only swap single-quoted spans that sit in key/value slots.
    return re.sub(r"(?<=[{\[,:\s])'([^'\"]*)'(?=\s*[,:}\]])", r'"\1"', candidate)


def parse_json_response(text) -> dict | None:
    """Extract the JSON object a chatty model buried in its answer.

    Handles thinking blocks, ``` fences, prose before/after, trailing commas and
    single-quoted Python-dict output. Returns ``None`` if nothing parses into a
    dict.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = _strip_fences(strip_thinking(text))
    candidate = _extract_object(cleaned)
    if candidate is None:
        return None

    attempts = [
        candidate,
        _TRAILING_COMMA_RE.sub(r"\1", candidate),
        _repair_single_quotes(candidate),
        _repair_single_quotes(_TRAILING_COMMA_RE.sub(r"\1", candidate)),
    ]
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# ──── Proposal normalisation ────

def _unwrap_quotes(value: str) -> str:
    """Strip matched surrounding quote characters, however many layers deep."""
    changed = True
    while changed and len(value) >= 2:
        changed = False
        for left, right in _QUOTE_PAIRS:
            if value.startswith(left) and value.endswith(right):
                value = value[1:-1].strip()
                changed = True
                break
    return value


def _clean_text(value, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value if v is not None)
    elif not isinstance(value, str):
        value = str(value)

    value = _unwrap_quotes(_WHITESPACE_RE.sub(" ", value).strip())
    value = _WHITESPACE_RE.sub(" ", value).strip()
    if len(value) <= limit:
        return value

    cut = value[:limit]
    boundary = cut.rfind(" ")
    if boundary >= max(1, limit // 3):
        cut = cut[:boundary]
    return cut.rstrip(" ,;:.-–—")


def _coerce_unit(value, default: float) -> float:
    """Coerce 0.85 / "0.85" / "85%" / 85 / "high" into a 0..1 float."""
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0

    number = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    elif isinstance(value, str):
        raw = value.strip().lower()
        if not raw:
            return default
        if raw in _WORD_SCORES:
            return _WORD_SCORES[raw]
        percent = raw.endswith("%")
        raw = raw.rstrip("%").strip().replace(",", ".")
        try:
            number = float(raw)
        except ValueError:
            return default
        if percent:
            number /= 100.0
    else:
        return default

    if number != number:            # NaN
        return default
    if number > 1.0:                # 85 and 85.0 mean "85 percent"
        number /= 100.0
    return max(0.0, min(1.0, number))


def normalize_proposal(obj, fallback_text: str = "") -> dict:
    """Coerce any parsed object into the exact five-key proposal shape.

    Missing keys get sane defaults, numeric-ish strings are coerced into 0..1
    floats, ``text`` is unquoted, whitespace-collapsed and truncated to
    :data:`MAX_TEXT_CHARS` at a word boundary.
    """
    source = obj if isinstance(obj, dict) else {}

    text = _clean_text(source.get("text"), MAX_TEXT_CHARS)
    if not text:
        text = _clean_text(fallback_text, MAX_TEXT_CHARS)

    return {
        "text": text,
        "style": _clean_text(source.get("style"), MAX_STYLE_CHARS),
        "font_hint": _clean_text(source.get("font_hint"), MAX_STYLE_CHARS),
        "legible_original": _coerce_unit(source.get("legible_original"), 0.0),
        "confidence": _coerce_unit(source.get("confidence"), 0.5),
    }


# ──── High-level region call ────

def build_user_prompt(class_name: str = "sign", class_instruction: str = "",
                      scene_hint: str = "", language: str = "auto",
                      neighbor_count: int = 0, has_scene: bool = False) -> str:
    """Compose the per-region user message that accompanies the images."""
    lines = [
        f"Region class: {class_name or 'sign'}.",
        "The lettering in this region was produced by an image generator and is "
        "meaningless gibberish. Do not transcribe or spell-correct it — invent "
        "new, correctly spelled text that should be written there instead.",
    ]

    if class_instruction and class_instruction.strip():
        lines.append(class_instruction.strip())
    if scene_hint and scene_hint.strip():
        # Spelled out because a bare hint reads as "stay faithful to the image"
        # to smaller vision models, which then transcribe the gibberish.
        lines.append(
            f"Scene context (setting, language and style only — this does NOT "
            f"tell you the letters): {scene_hint.strip()}")

    lang = (language or "auto").strip()
    if not lang or lang.lower() == "auto":
        lines.append("Write in the language that fits the scene; if in doubt, "
                     "match the language of any readable text around it.")
    else:
        lines.append(f"Write the text in {lang}.")

    order = ["Image 1 is the cropped region itself."]
    if has_scene:
        order.append(f"Image {len(order) + 1} is the full scene for context.")
    if neighbor_count > 0:
        first = len(order) + 1
        last = first + neighbor_count - 1
        span = f"Image {first}" if first == last else f"Images {first}-{last}"
        order.append(f"{span} show neighbouring regions from the same picture — "
                     "match their language and styling.")
    lines.append(" ".join(order))

    # Last line = the recency slot the model weighs most heavily. Small vision
    # models drift back into transcribing the crop if the anti-copy rule only
    # appears near the top, so it is repeated here, immediately before decoding.
    lines.append('Answer with one JSON object only: {"text","style","font_hint",'
                 '"legible_original","confidence"} — and remember, the letters '
                 'in image 1 are gibberish, not words. Do not transcribe them, '
                 'do not spell-correct them, do not keep any of them. Invent '
                 'new, correctly spelled words instead.')
    return "\n".join(lines)


def propose_text(crop_rgb, scene_rgb=None, neighbor_crops=None,
                 class_name: str = "sign", scene_hint: str = "",
                 language: str = "auto", base_url: str = DEFAULT_BASE_URL,
                 model_id: str = "", system_prompt: str | None = None,
                 class_instruction: str = "",
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = 256, seed=None,
                 timeout: float = DEFAULT_TIMEOUT,
                 extra_options: dict | None = None,
                 max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
                 fallback_text: str = "") -> dict:
    """Ask the vision model what text belongs in one region.

    Images are sent in a fixed order: the crop first, then the whole scene, then
    up to :data:`MAX_NEIGHBOR_IMAGES` neighbouring crops.

    Keep ``temperature`` low. Above ~0.3 the model starts returning a near-miss
    of the garbled lettering it was asked to replace — see
    :data:`DEFAULT_TEMPERATURE` for the measurements.

    Returns the five :func:`normalize_proposal` keys plus
    ``{"ok": bool, "error": str | None, "source": "vlm" | "fallback"}``.
    On any failure — unreachable server, HTTP error, unparsable answer — the
    fallback shape is returned with ``ok=False`` and ``source="fallback"``.
    Never raises.
    """
    def _fallback(error: str) -> dict:
        result = normalize_proposal({}, fallback_text=fallback_text)
        result.update({"ok": False, "error": error, "source": "fallback"})
        return result

    if crop_rgb is None:
        return _fallback("No crop image supplied")

    neighbors = [n for n in (neighbor_crops or []) if n is not None]
    neighbors = neighbors[:MAX_NEIGHBOR_IMAGES]

    images = [crop_rgb]
    if scene_rgb is not None:
        images.append(scene_rgb)
    images.extend(neighbors)

    user_prompt = build_user_prompt(
        class_name=class_name,
        class_instruction=class_instruction,
        scene_hint=scene_hint,
        language=language,
        neighbor_count=len(neighbors),
        has_scene=scene_rgb is not None,
    )

    try:
        response = chat_vision(
            base_url=base_url,
            model_id=model_id,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            timeout=timeout,
            extra_options=extra_options,
            max_image_size=max_image_size,
        )
    except Exception as exc:  # noqa: BLE001 — belt and braces; chat_vision traps
        return _fallback(f"{type(exc).__name__}: {exc}")

    if not response.get("ok"):
        return _fallback(response.get("error") or "Vision request failed")

    parsed = parse_json_response(response.get("content", ""))
    if parsed is None:
        snippet = strip_thinking(response.get("content", ""))[:200]
        return _fallback(f"Could not parse JSON from model answer: {snippet!r}")

    result = normalize_proposal(parsed, fallback_text=fallback_text)
    result.update({"ok": True, "error": None, "source": "vlm"})
    return result
