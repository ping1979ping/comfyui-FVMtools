"""Registry of text-region classes for the FVM sign / text repair pipeline.

Every class bundles everything the pipeline needs for one kind of readable
surface: how to ground it with SAM3, how small a hit may be before it is not
worth repairing, what a vision LLM should transcribe, how to phrase the
diffusion prompt and how much the base denoise should be nudged.
"""
import re

DEFAULT_THRESHOLD = 0.30
DEFAULT_MIN_HEIGHT_PX = 24
MIN_DENOISE_BIAS = -0.2
MAX_DENOISE_BIAS = 0.2

FALLBACK_CLASS_NAME = "generic"

SIGN_CLASSES = {
    "sign": {
        "sam3_prompts": ["sign", "street sign", "shop sign"],
        "threshold": 0.30,
        "min_height_px": 32,
        "vlm_instruction": (
            "Transcribe the wording on this sign exactly as a passer-by would "
            "read it, including street names, shop names and directions."
        ),
        "prompt_template": (
            'a sign with the clear legible text "{text}", {style} '
            "crisp high-contrast lettering, sharp typography"
        ),
        "denoise_bias": 0.0,
    },
    "label": {
        "sam3_prompts": ["bottle label", "product label", "packaging label"],
        "threshold": 0.28,
        "min_height_px": 24,
        "vlm_instruction": (
            "Transcribe the brand name and product wording printed on this "
            "label, keeping the original line breaks in reading order."
        ),
        "prompt_template": (
            'a product label printed with the clear text "{text}", {style} '
            "clean printed typography, sharp product photography"
        ),
        "denoise_bias": 0.05,
    },
    "garment_print": {
        "sam3_prompts": ["printed text on clothing", "t-shirt print", "logo on shirt"],
        "threshold": 0.30,
        "min_height_px": 40,
        "vlm_instruction": (
            "Transcribe the slogan or logo text printed on the garment, "
            "ignoring folds, seams and stitching patterns."
        ),
        "prompt_template": (
            'clothing with the printed text "{text}", {style} '
            "screen-printed lettering following the fabric folds"
        ),
        "denoise_bias": -0.05,
    },
    "poster": {
        "sam3_prompts": ["poster", "banner", "billboard"],
        "threshold": 0.28,
        "min_height_px": 40,
        "vlm_instruction": (
            "Transcribe the headline and any secondary lines of this poster, "
            "starting with the largest and most prominent text."
        ),
        "prompt_template": (
            'a poster with the bold headline text "{text}", {style} '
            "graphic design layout, clean print typography"
        ),
        "denoise_bias": 0.0,
    },
    "screen": {
        "sam3_prompts": ["phone screen", "computer monitor", "display screen"],
        "threshold": 0.30,
        "min_height_px": 32,
        "vlm_instruction": (
            "Transcribe the text shown on this screen, including interface "
            "labels, buttons and status lines."
        ),
        "prompt_template": (
            'a screen displaying the crisp text "{text}", {style} '
            "rendered user interface typography, backlit display"
        ),
        "denoise_bias": -0.05,
    },
    "book": {
        "sam3_prompts": ["book cover", "magazine cover"],
        "threshold": 0.30,
        "min_height_px": 32,
        "vlm_instruction": (
            "Transcribe the title, subtitle and author wording on this cover "
            "in the order they are printed."
        ),
        "prompt_template": (
            'a book cover with the title text "{text}", {style} '
            "printed cover typography, sharp focus"
        ),
        "denoise_bias": 0.0,
    },
    "plate": {
        "sam3_prompts": ["license plate"],
        "threshold": 0.35,
        "min_height_px": 20,
        "vlm_instruction": (
            "Transcribe the character sequence on this license plate exactly, "
            "keeping separators, spacing and letter case."
        ),
        "prompt_template": (
            'a vehicle license plate showing the characters "{text}", {style} '
            "embossed plate lettering, high contrast"
        ),
        "denoise_bias": 0.10,
    },
    "paper": {
        "sam3_prompts": ["document", "menu", "price tag", "receipt"],
        "threshold": 0.28,
        "min_height_px": 24,
        "vlm_instruction": (
            "Transcribe the printed or handwritten wording on this piece of "
            "paper, keeping the reading order of the lines."
        ),
        "prompt_template": (
            'a printed document showing the text "{text}", {style} '
            "clean printed typography on paper"
        ),
        "denoise_bias": 0.0,
    },
    "graffiti": {
        "sam3_prompts": ["graffiti", "handwritten text"],
        "threshold": 0.30,
        "min_height_px": 40,
        "vlm_instruction": (
            "Transcribe the sprayed or handwritten wording, reading stylised "
            "and overlapping letterforms as best as possible."
        ),
        "prompt_template": (
            'spray-painted graffiti lettering reading "{text}", {style} '
            "hand-painted wall art, expressive strokes"
        ),
        "denoise_bias": -0.10,
    },
}

CLASS_ORDER = tuple(SIGN_CLASSES)

FALLBACK_CLASS = {
    "sam3_prompts": ["text", "written text"],
    "threshold": DEFAULT_THRESHOLD,
    "min_height_px": DEFAULT_MIN_HEIGHT_PX,
    "vlm_instruction": (
        "Transcribe any legible written text visible in this region exactly "
        "as it appears."
    ),
    "prompt_template": (
        'a surface with the clear legible text "{text}", {style} '
        "sharp readable typography"
    ),
    "denoise_bias": 0.0,
}

CLASS_KEYS = (
    "sam3_prompts",
    "threshold",
    "min_height_px",
    "vlm_instruction",
    "prompt_template",
    "denoise_bias",
)


def all_class_names():
    """Return the class names in their stable registry order."""
    return list(CLASS_ORDER)


def get_class(name):
    """
    Return a copy of the registry entry for ``name``.

    Unknown or empty names fall back to a generic entry instead of raising,
    so a node can pass user input straight through.
    """
    key = (name or "").strip().lower()
    entry = SIGN_CLASSES.get(key, FALLBACK_CLASS)
    return {k: (list(v) if isinstance(v, list) else v) for k, v in entry.items()}


def parse_custom_prompts(spec):
    """
    Parse a comma separated ``prompt:threshold`` specification.

    ``"bottle label:0.3, neon sign"`` becomes
    ``[("bottle label", 0.3), ("neon sign", 0.30)]``. Entries without a usable
    float fall back to DEFAULT_THRESHOLD, empty entries are dropped.
    """
    if not spec:
        return []

    parsed = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        threshold = DEFAULT_THRESHOLD
        if ":" in chunk:
            prompt, _, raw = chunk.rpartition(":")
            prompt = prompt.strip()
            try:
                threshold = float(raw.strip())
            except ValueError:
                threshold = DEFAULT_THRESHOLD
        else:
            prompt = chunk

        prompt = re.sub(r"\s+", " ", prompt).strip()
        if not prompt:
            continue

        parsed.append((prompt, max(0.01, min(1.0, threshold))))

    return parsed


def build_prompt(class_name, text, style=""):
    """
    Fill the class prompt template with ``text`` and an optional ``style``.

    Empty style fragments leave no dangling commas or double spaces behind.
    """
    entry = get_class(class_name)
    template = entry["prompt_template"]
    fields = {"text": (text or "").strip(), "style": (style or "").strip()}
    try:
        filled = template.format(**fields)
    except (KeyError, IndexError, ValueError):
        filled = template
    return collapse_separators(filled)


def collapse_separators(text):
    """Collapse repeated whitespace and comma runs into single separators."""
    out = re.sub(r"\s+", " ", (text or "").strip())
    out = re.sub(r"\s*,\s*", ", ", out)
    while ", ," in out:
        out = out.replace(", ,", ",")
    out = re.sub(r"\s*,\s*", ", ", out)
    return out.strip(" ,")


def clamp_denoise_bias(value):
    """Clamp a denoise bias into the documented -0.2 .. +0.2 range."""
    try:
        bias = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(MIN_DENOISE_BIAS, min(MAX_DENOISE_BIAS, bias))
