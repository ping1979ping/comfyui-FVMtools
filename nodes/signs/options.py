"""SignOptions — per-class overrides and advanced knobs for the Sign Detailer.

Keeps the Detailer's own widget list at a workable size. Everything here is
optional; the Detailer falls back to SIGN_DEFAULTS when nothing is connected.
"""

try:  # relative inside ComfyUI's loader, absolute under pytest
    from ...core.signs.classes import all_class_names
except ImportError:
    from core.signs.classes import all_class_names


SIGN_DEFAULTS = {
    "cfg": 1.0,
    "context_expand_factor": 1.30,
    "output_padding": 32,
    "mask_fill_holes": True,
    "denoise_progression": "",
    "steps_progression": "",
    "negative_prompt": "gibberish, garbled letters, misspelled words, random symbols, "
                       "warped typography, double exposure text, watermark",
    "class_denoise": {},
    "class_skip": set(),
    "uppercase": False,
    "margin_ratio": 0.08,
}


def _parse_class_map(spec, cast=float):
    """Parse 'class: value' lines into a dict, ignoring unknown class names."""
    out = {}
    known = set(all_class_names())
    for line in (spec or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key not in known:
            continue
        try:
            out[key] = cast(value.strip())
        except (ValueError, TypeError):
            continue
    return out


class SignOptions:
    """Advanced settings and per-class overrides for the Sign Detailer."""

    CATEGORY = "FVM Tools/Text"
    FUNCTION = "execute"
    RETURN_TYPES = ("SIGN_OPTIONS",)
    RETURN_NAMES = ("sign_options",)

    DESCRIPTION = (
        "Optional advanced settings for the Sign Detailer.\n\n"
        "Per-class denoise overrides let a license plate render harder than a t-shirt print\n"
        "without touching the global setting. skip_classes drops whole categories late,\n"
        "so you can re-run without re-detecting."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance. Turbo/distilled models want 1.0."}),
                "negative_prompt": ("STRING", {"default": SIGN_DEFAULTS["negative_prompt"], "multiline": True,
                    "tooltip": "Applied to every region. Aimed at suppressing fake lettering."}),
                "context_expand_factor": ("FLOAT", {"default": 1.30, "min": 1.0, "max": 3.0, "step": 0.05,
                    "tooltip": "How much surrounding context enters the crop. Text needs more than faces\n"
                               "so the model can match the sign's material and perspective."}),
                "output_padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "mask_fill_holes": ("BOOLEAN", {"default": True,
                    "tooltip": "Close holes in the mask — letter counters otherwise punch through."}),
                "denoise_progression": ("STRING", {"default": "",
                    "tooltip": "Pipe-separated denoise per round, e.g. '0.85|0.45'."}),
                "steps_progression": ("STRING", {"default": "",
                    "tooltip": "Pipe-separated steps per round, e.g. '8|4'."}),
                "class_denoise": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Per-class denoise override, one 'class: value' per line.\n"
                               "Example:\nplate: 0.95\ngarment_print: 0.70"}),
                "skip_classes": ("STRING", {"default": "",
                    "tooltip": "Comma-separated class names to leave untouched, e.g. 'screen, graffiti'."}),
                "uppercase": ("BOOLEAN", {"default": False,
                    "tooltip": "Force glyph rendering to uppercase — helps on street signs and plates."}),
                "margin_ratio": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.4, "step": 0.01,
                    "tooltip": "Padding between the rendered text and the edge of the sign."}),
            },
        }

    def execute(self, cfg, negative_prompt, context_expand_factor, output_padding,
                mask_fill_holes, denoise_progression, steps_progression,
                class_denoise, skip_classes, uppercase, margin_ratio):
        known = set(all_class_names())
        skip = {c.strip().lower() for c in (skip_classes or "").split(",") if c.strip()}
        unknown = skip - known
        if unknown:
            print(f"[SignOptions] ignoring unknown class name(s): {', '.join(sorted(unknown))}")

        return ({
            "cfg": cfg,
            "negative_prompt": negative_prompt,
            "context_expand_factor": context_expand_factor,
            "output_padding": output_padding,
            "mask_fill_holes": mask_fill_holes,
            "denoise_progression": denoise_progression,
            "steps_progression": steps_progression,
            "class_denoise": _parse_class_map(class_denoise, float),
            "class_skip": skip & known,
            "uppercase": uppercase,
            "margin_ratio": margin_ratio,
        },)
