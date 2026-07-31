"""SignDetailer — re-renders detected text regions with believable lettering.

Reuses the shared inpaint pipeline. The interesting part is glyph guidance: the
target text is typeset with PIL, warped onto the sign's actual quadrilateral and
composited into the crop before encoding. The sampler then only has to restyle
material, perspective and lighting instead of inventing letterforms — the
AnyText/GlyphControl idea without an extra model, so it also works with
checkpoints that are weak at text.

Regions too small to ever read are not forced: the soften policy renders them as
plausibly out-of-focus lettering, which is what a real photograph looks like.
"""

import difflib

import cv2
import numpy as np
import torch

import comfy.samplers

from ..utils.inpaint_pipeline import inpaint_slot
from ..utils.detail_daemon import DD_DEFAULTS
from ..utils.glyph import (
    discover_fonts, resolve_font, render_glyph_layer, composite_glyph,
    estimate_text_colors, surface_preserving_alpha, measure_ink_height,
    existing_ink_mask, glyph_ink_coverage, text_band, reconstruct_surface,
)
from ..utils.mask_utils import fill_mask_holes_2d
from ..utils.ocr_backend import ocr_region
from ..utils.tensor_utils import tensor2np
try:  # relative inside ComfyUI's loader, absolute under pytest
    from ...core.signs.classes import build_prompt, get_class
except ImportError:
    from core.signs.classes import build_prompt, get_class
from .options import SIGN_DEFAULTS, parse_hex_rgb


# Prompt used when a region is below the legibility floor and soften is chosen.
SOFTEN_PROMPT = ("out-of-focus printed text, too distant to read, soft blurred lettering, "
                 "natural photographic depth of field")

# Text-weak model families: warned about once per run.
_TEXT_WEAK_HINTS = ("z-image", "zimage", "lumina", "sd15", "sd_15", "sdxl")

# Defaults verified by live renders on Krea 2 Turbo (krea2_turbo_fp8 +
# qwen3vl_4b_fp8_scaled + qwen_image_vae): 8 steps, cfg 1, er_sde / simple.
# Resolved against the running ComfyUI so a build without er_sde still loads.
def _pick(name, options, fallback=0):
    """Prefer `name`, fall back to the first option, never raise.

    Runs at import time, so it has to survive a ComfyUI build without this
    sampler — and a stubbed comfy.samplers under pytest.
    """
    try:
        if name in options:
            return name
        chosen = options[fallback]
        return chosen if isinstance(chosen, str) else name
    except (TypeError, IndexError, KeyError):
        return name


SAMPLER_DEFAULT = _pick("er_sde", comfy.samplers.SAMPLER_NAMES)
SCHEDULER_DEFAULT = _pick("simple", comfy.samplers.SCHEDULER_NAMES)

# Above this the sampler stops treating the typeset layer as a template.
# Measured on Krea 2 Turbo (8 steps, cfg 1, er_sde/simple): 0.55 gives sharp text
# with real material, 0.65 adds a ghost copy of the word behind the real one,
# 0.70+ reinvents the sign entirely (warped shape, invented hardware, faded text).
GLYPH_DENOISE_SAFE_MAX = 0.60

# Aspect cap for the auto resolution. VRAM follows the pixel COUNT, not the edge
# length — 2456x424 costs the same as 1024x1024 — so capping an edge would only
# throw budget away. What does need bounding is the ratio itself: past roughly
# 4:1 the short side reaches the latent floor and attention over one very long
# axis stops paying off.
AUTO_RESOLUTION_MAX_ASPECT = 4.0

# Room left around the new strokes for the sampler to work in, as a multiple of
# the lettering height. Enough that letters get their own shading and edges,
# little enough that swept-clean surface stays out of reach — handed an empty
# sheet at full denoise the model writes its own notes on it.
HOT_ZONE_MARGIN = 0.5


def _fuzzy_match(a, b):
    """Similarity of two strings, case- and whitespace-insensitive."""
    na = "".join(a.upper().split())
    nb = "".join(b.upper().split())
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


class SignDetailer:
    """Re-renders sign/label/print regions with the proposed text."""

    CATEGORY = "FVM Tools/Text"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "refined_crops", "glyph_preview", "report")
    OUTPUT_NODE = True

    DESCRIPTION = (
        "Re-renders each detected text region with the text proposed upstream.\n\n"
        "glyph_guidance typesets the target text, warps it onto the sign's real quad and\n"
        "composites it into the crop before sampling. This is what makes the lettering come\n"
        "out readable — leave it on unless you are deliberately testing the raw model.\n\n"
        "Use a text-capable checkpoint (Qwen-Image, Ideogram, Krea 2). Turbo and SDXL-class\n"
        "models cannot render legible words and will just produce different gibberish.\n\n"
        "denoise defaults high (0.85): at low denoise the original garbled strokes bleed\n"
        "through. With glyph_guidance on, glyph_denoise takes over and can be much lower."
    )

    @classmethod
    def INPUT_TYPES(cls):
        fonts = ["<auto>"] + discover_fonts()
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The image batch the selector scanned"}),
                "sign_data": ("SIGN_DATA", {"tooltip": "Regions with proposals, from Sign Text Proposer"}),
                "model": ("MODEL", {"tooltip": "Use a text-capable model — Qwen-Image, Ideogram, Krea 2"}),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100,
                    "tooltip": "8 is the verified value for Krea 2 Turbo, which is distilled —\n"
                               "more steps over-cook it. Non-distilled models (Krea 2 Raw,\n"
                               "Qwen-Image) want 15-25."}),
                "denoise": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Used when glyph guidance is off. High on purpose: the old garbled\n"
                               "strokes survive anything below ~0.7."}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": SAMPLER_DEFAULT,
                    "tooltip": "er_sde is the verified pairing for Krea 2. Without an explicit\n"
                               "default ComfyUI would pick euler, which is not what these\n"
                               "settings were measured against."}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": SCHEDULER_DEFAULT}),
                "auto_resolution": ("BOOLEAN", {"default": True,
                    "tooltip": "Shape the sampling resolution like the region instead of using\n"
                               "target_width x target_height literally. Those two then only set\n"
                               "the pixel BUDGET, so VRAM use stays the same.\n\n"
                               "A fixed square wastes most of that budget on a wide sign: the crop\n"
                               "is expanded to 1:1 first, leaving the lettering in a thin band.\n"
                               "Measured on a 550x95 sign — square 1024 gives the text 1.9x\n"
                               "upscale, the same budget at the region's own aspect gives 4.5x."}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "With auto_resolution on this is half of the pixel budget, not a literal width."}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "With auto_resolution on this is half of the pixel budget, not a literal height."}),
                "max_upscale": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 32.0, "step": 0.5,
                    "tooltip": "Caps how far a small region is blown up before sampling.\n"
                               "Beyond this the model hallucinates detail that breaks on stitch-back."}),
                "glyph_guidance": (["init", "init_strong", "off"], {"default": "init",
                    "tooltip": "init: typeset text composited into the crop, sampled at glyph_denoise\n"
                               "init_strong: same, but 0.15 lower denoise — maximum letter fidelity,\n"
                               "  less freedom to match the sign's material and lighting\n"
                               "off: prompt-only, needs a genuinely text-capable model"}),
                "glyph_font": (fonts, {"default": "<auto>",
                    "tooltip": "Typeface for the rendered text. <auto> follows the model's font hint."}),
                "glyph_denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise when glyph guidance is on. Lower keeps the letterforms exact,\n"
                               "higher blends better into the surface.\n\n"
                               "Measured on Krea 2 Turbo (8 steps, cfg 1, er_sde/simple):\n"
                               "  0.35  clean text, but the surface stays flat and characterless\n"
                               "  0.55  best - sharp text AND real material (enamel, screws, wear)\n"
                               "  0.65  a second ghost copy of the word appears behind the first\n"
                               "  0.70+ the sign is reinvented: warped, extra hardware, text faded\n"
                               "Stay at or below 0.60."}),
                "glyph_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Opacity of the typeset layer before sampling.\n"
                               "Keep at 1.0: measured, even 0.95 leaves the ORIGINAL garbled\n"
                               "lettering clearly readable underneath, and the sampler then has\n"
                               "it in the init latent. Lower this only to deliberately carry over\n"
                               "the old surface shading."}),
                "glyph_fit": (["auto", "perspective", "contour"], {"default": "auto",
                    "tooltip": "How the typeset text is fitted onto the region.\n"
                               "auto: measures how badly a four-corner fit misses the outline\n"
                               "  and switches to the column-wise fit when it does\n"
                               "perspective: four corners — correct for flat signs, gives the\n"
                               "  text a vanishing line when the sign is angled away\n"
                               "contour: follows the mask's top and bottom edge column by\n"
                               "  column — for curved labels, folded fabric, torn paper.\n"
                               "  A flat plane cannot describe those, so the text would\n"
                               "  otherwise sit dead straight on a bowed surface."}),
                "glyph_cylinder": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Extra horizontal compression towards the sides, the way lettering\n"
                               "wrapped around a bottle or can foreshortens away from the viewer.\n"
                               "Only applies to the contour fit. Try 0.4-0.6 for bottle labels,\n"
                               "0 for anything flat."}),
                "glyph_match_source_size": ("BOOLEAN", {"default": True,
                    "tooltip": "Set the replacement text at the size the surface already used,\n"
                               "instead of scaling it to fill the region.\n\n"
                               "Filling the box is right for a headline sign and wrong for\n"
                               "everything else. A wine label carries one large number over three\n"
                               "lines of small print — scaled to the box it becomes a poster, and\n"
                               "long words no longer fit and get cut off at the edge.\n\n"
                               "The size is measured from the existing strokes, so it adapts per\n"
                               "region. It is a cap, not a fixed size: longer replacement text\n"
                               "still shrinks to fit."}),
                "glyph_preserve_surface": ("BOOLEAN", {"default": True,
                    "tooltip": "Replace only the lettering, not the whole region.\n\n"
                               "Painting the full masked area with the plate colour erases\n"
                               "everything the surface had, and the sampler must then invent it\n"
                               "back from a flat fill. On real photographs that turned a bordered\n"
                               "enamel oval into a plain white disc, and a curved shop-window\n"
                               "inscription into a pasted-on banner covering the glass.\n\n"
                               "On: only the old strokes and the new ones are covered. Parts that\n"
                               "run into the region's edge — frames, rims, painted borders — are\n"
                               "recognised as structure and left alone.\n"
                               "Off: the old full-area behaviour."}),
                "glyph_surface_restyle": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much of the denoise the surface AROUND the lettering gets,\n"
                               "as a fraction. The lettering itself always gets all of it.\n\n"
                               "This is what makes a second pass over the same sign safe. At 1.0\n"
                               "the whole plate is re-rendered every time, so three passes means\n"
                               "three generations of drift — measured on a navy enamel sign, the\n"
                               "plate walked to dark grey and the border broke up.\n\n"
                               "0.0  the surface is untouched — most faithful for repeat edits\n"
                               "0.35 default — enough for a flat synthetic plate to gain material\n"
                               "1.0  the old behaviour\n"
                               "Only applies with glyph_preserve_surface on."}),
                "glyph_autocolor": ("BOOLEAN", {"default": True,
                    "tooltip": "Sample ink and plate colour from the original sign so the replacement keeps its scheme."}),
                "glyph_plate_color": ("STRING", {"default": "",
                    "tooltip": "Force the SURFACE colour of the typeset layer. Empty = use the\n"
                               "colour sampled from the original.\n"
                               "Accepts '#ffe680', 'ffe680', '#fe8' or '255,230,128'.\n"
                               "Use this when you want a different surface than the one in the\n"
                               "picture — a yellow post-it, a white plate, a green road sign —\n"
                               "and describe that surface in the Sign Options prompt_suffix too."}),
                "glyph_ink_color": ("STRING", {"default": "",
                    "tooltip": "Force the LETTER colour of the typeset layer. Empty = sampled.\n"
                               "Same formats as the plate colour."}),
                "too_small_policy": (["soften", "skip", "render"], {"default": "soften",
                    "tooltip": "Regions below the legibility floor.\n"
                               "soften: render as believable out-of-focus text (recommended)\n"
                               "skip: leave untouched\n"
                               "render: try anyway"}),
                "cluster_mode": (["shared_seed", "independent"], {"default": "shared_seed",
                    "tooltip": "shared_seed renders every member of a cluster with the same seed and prompt,\n"
                               "so a shelf of identical bottles stays identical."}),
                "verify_after": (["off", "ocr"], {"default": "off",
                    "tooltip": "ocr: read the result back and retry with a new seed if the text did not land.\n"
                               "Needs an OCR backend; without one it silently stays off."}),
                "verify_similarity": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum similarity between the read-back text and the target."}),
                "max_attempts": ("INT", {"default": 2, "min": 1, "max": 5,
                    "tooltip": "Render attempts per region when verification fails."}),
                "mask_expand_pixels": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Small expansion helps cover anti-aliased edges of the old lettering."}),
                "mask_blend_pixels": ("INT", {"default": 16, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Lower than for faces — a hard-edged sign should not fade into its surroundings."}),
                "detail_daemon_enabled": ("BOOLEAN", {"default": False,
                    "tooltip": "Usually off for text: extra detail turns clean strokes noisy."}),
                "detail_amount": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "positive_base": ("CONDITIONING", {"tooltip": "Fallback conditioning when a region has no text"}),
                "negative": ("CONDITIONING", {"tooltip": "Overrides the negative prompt from Sign Options"}),
                "sign_options": ("SIGN_OPTIONS", {"tooltip": "Per-class overrides and advanced settings"}),
                "dd_options": ("DD_OPTIONS",),
            },
        }

    # ── Helpers ──

    def _encode(self, clip, text):
        tokens = clip.tokenize(text)
        out = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = out.pop("cond")
        return [[cond, out]]

    def _warn_if_text_weak(self, model):
        """Best-effort check that the chosen checkpoint can render words at all."""
        try:
            name = type(model.model).__name__.lower()
            cfg = str(getattr(getattr(model, "model", None), "model_config", "")).lower()
            blob = f"{name} {cfg}"
        except Exception:
            return None
        for hint in _TEXT_WEAK_HINTS:
            if hint in blob:
                return (f"WARNING: '{hint}' detected in the model config. Turbo/SDXL-class models "
                        f"cannot render legible text — expect different gibberish. Use Qwen-Image, "
                        f"Ideogram or Krea 2, or rely on glyph_guidance=init_strong with a low glyph_denoise.")
        return None

    def _clamp_target(self, region, target_w, target_h, max_upscale):
        """Shrink the sampling resolution when a tiny region would be blown up too far."""
        x1, y1, x2, y2 = region["bbox"]
        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)
        scale = min(target_w / crop_w, target_h / crop_h)
        if scale <= max_upscale:
            return target_w, target_h
        factor = max_upscale / scale
        new_w = max(64, int(target_w * factor) // 8 * 8)
        new_h = max(64, int(target_h * factor) // 8 * 8)
        return new_w, new_h

    def _resolve_target(self, region, target_w, target_h, max_upscale, auto):
        """Sampling resolution for this region, shaped like the region itself.

        A fixed square wastes most of the budget on a wide sign: the crop is
        expanded to a 1:1 aspect before resizing, so the lettering ends up
        occupying a thin band. Measured on a 550x95 sign — a square 1024 gives the
        text 1.9x upscale, the same pixel budget spent at the region's own aspect
        gives 4.5x. Keeping the budget constant means VRAM use does not change.
        """
        if not auto:
            return self._clamp_target(region, target_w, target_h, max_upscale)

        x1, y1, x2, y2 = region["bbox"]
        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)
        aspect = crop_w / crop_h
        # Bound the ratio, then spend the whole budget at that ratio — clipping an
        # edge instead would quietly hand back the very resolution we are after.
        aspect = min(max(aspect, 1.0 / AUTO_RESOLUTION_MAX_ASPECT), AUTO_RESOLUTION_MAX_ASPECT)
        budget = float(target_w) * float(target_h)

        width = max(64, int(round((budget * aspect) ** 0.5)) // 8 * 8)
        height = max(64, int(round((budget / aspect) ** 0.5)) // 8 * 8)
        return self._clamp_target(region, width, height, max_upscale)

    def _apply_glyph(self, image_hwc, region, text, font_choice, strength,
                     autocolor, uppercase, margin_ratio,
                     ink_override=None, plate_override=None,
                     fit="auto", cylinder=0.0, preserve_surface=True,
                     match_source_size=True):
        """Composite typeset text onto the image inside the region mask.

        Returns ``(new_image, glyph_rgb_preview, hot)``. ``hot`` is where the NEW
        lettering stands, and it is what the caller samples at full strength — the
        rest of the region gets a fraction of it. Emptied surface must not be
        included: given free rein over a cleared sheet the model fills it with
        invented handwriting, which reads exactly like leftover text.
        ``None`` when the surface is not being kept.
        """
        mask_np = region["mask"]
        img_np = image_hwc.cpu().numpy() if isinstance(image_hwc, torch.Tensor) else image_hwc

        font_hint = (region.get("proposal") or {}).get("font_hint", "")
        if font_choice and font_choice != "<auto>":
            font_path = font_choice
        else:
            font_path = resolve_font(font_hint or get_class(region["class"])["vlm_instruction"])

        ink, plate = ((255, 255, 255), (0, 0, 0))
        if autocolor:
            rgb_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            ink, plate = estimate_text_colors(rgb_u8, mask_np)
        # Explicit colours win over the estimate — this is how you ask for a
        # surface the picture does not contain yet (a yellow post-it, say).
        if ink_override is not None:
            ink = ink_override
        if plate_override is not None:
            plate = plate_override

        # Match the size the surface already used. Filling the box turns a wine
        # label into a poster and pushes long words past the edge.
        line_height = None
        if match_source_size:
            rgb_probe = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            line_height = measure_ink_height(rgb_probe, mask_np, plate)

        try:
            glyph_rgb, alpha = render_glyph_layer(
                text=text, mask_2d=mask_np, font_path=font_path,
                fill=ink, bg=plate, uppercase=uppercase, margin_ratio=margin_ratio,
                fit=fit, cylinder=cylinder, target_line_height=line_height,
            )
        except Exception as exc:
            print(f"[SignDetailer] glyph rendering failed for #{region['index'] + 1}: {exc}")
            return image_hwc, None, None

        device = image_hwc.device if isinstance(image_hwc, torch.Tensor) else "cpu"
        band = None
        if preserve_surface:
            rgb_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            coverage = glyph_ink_coverage(glyph_rgb, alpha, plate, ink)
            band = text_band(mask_np,
                             old_ink=existing_ink_mask(rgb_u8, mask_np, plate),
                             new_ink=None if coverage is None else (coverage > 0.15),
                             line_height=line_height)
            if band is not None and coverage is not None and float(coverage.max()) > 0.0:
                # Wipe the whole band first. Covering stroke by stroke is what
                # left ghosts: every stroke the detector missed stayed in the
                # picture and the next pass added to it.
                surface = reconstruct_surface(img_np, band)
                base = img_np.astype(np.float32)
                if surface is not None:
                    keep = band[..., None].astype(np.float32)
                    base = base * (1.0 - keep) + surface * keep
                blended = composite_glyph(base, glyph_rgb, coverage, strength=strength)
                hot = self._hot_zone(coverage, mask_np, line_height)
                return torch.from_numpy(blended).to(device), glyph_rgb, hot
            # No usable band — fall back to covering the strokes themselves.
            alpha = surface_preserving_alpha(glyph_rgb, alpha, rgb_u8, mask_np, plate)
            band = None

        if alpha is None or float(alpha.max()) <= 0.0:
            return image_hwc, None, None

        blended = composite_glyph(img_np.astype(np.float32), glyph_rgb, alpha, strength=strength)
        return torch.from_numpy(blended).to(device), glyph_rgb, band

    @staticmethod
    def _hot_zone(coverage, mask_2d, line_height=None):
        """Where the new lettering stands, with room around it to render into."""
        if coverage is None:
            return None
        strokes = (np.asarray(coverage, np.float32) > 0.15).astype(np.uint8)
        if strokes.sum() == 0:
            return None
        height = float(line_height) if line_height else 20.0
        pad = max(2, int(round(height * HOT_ZONE_MARGIN)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1,) * 2)
        grown = cv2.dilate(strokes, k, iterations=1).astype(np.float32)
        region = np.asarray(mask_2d, np.float32)
        if region.shape[:2] == grown.shape[:2]:
            grown = grown * (region > 0.5)
        return grown

    def _split_noise_mask(self, region_mask, hot, restyle, fill_holes=True):
        """Full denoise where the new text goes, a fraction of it everywhere else.

        Returns ``None`` when there is nothing to split, which leaves the region
        sampled evenly the way it always was.
        """
        if hot is None or restyle >= 1.0:
            return None
        base = region_mask
        if fill_holes:
            base = fill_mask_holes_2d(base)
        inside = (base > 0.5).float()
        hot_t = torch.from_numpy(np.asarray(hot, np.float32))
        if hot_t.shape != inside.shape:
            return None
        return torch.clamp(hot_t + (1.0 - hot_t) * float(restyle), 0.0, 1.0) * inside

    # ── Main ──

    def execute(self, images, sign_data, model, clip, vae, seed, steps, denoise,
                sampler_name, scheduler, target_width, target_height, max_upscale,
                auto_resolution=True,
                glyph_guidance="init", glyph_font="<auto>", glyph_denoise=0.55,
                glyph_strength=1.0, glyph_fit="auto", glyph_cylinder=0.0,
                glyph_preserve_surface=True, glyph_surface_restyle=0.35,
                glyph_match_source_size=True,
                glyph_autocolor=True,
                glyph_plate_color="", glyph_ink_color="", too_small_policy="soften",
                cluster_mode="shared_seed", verify_after="off", verify_similarity=0.60,
                max_attempts=2, mask_expand_pixels=4, mask_blend_pixels=16,
                detail_daemon_enabled=False, detail_amount=0.0,
                positive_base=None, negative=None, sign_options=None, dd_options=None):

        opts = {**SIGN_DEFAULTS, **(sign_options or {})}
        regions = sign_data.get("regions", []) if isinstance(sign_data, dict) else []
        report = [f"Sign Detailer — {len(regions)} region(s), glyph_guidance={glyph_guidance}"]

        warning = self._warn_if_text_weak(model)
        if warning:
            report.append(warning)
            print(f"[SignDetailer] {warning}")

        negative_cond = negative if negative is not None else self._encode(clip, opts["negative_prompt"])

        if glyph_guidance == "init_strong":
            eff_strength = min(1.0, glyph_strength + 0.15)
            eff_glyph_denoise = max(0.0, glyph_denoise - 0.15)
        else:
            eff_strength, eff_glyph_denoise = glyph_strength, glyph_denoise

        if glyph_guidance != "off" and eff_glyph_denoise > GLYPH_DENOISE_SAFE_MAX:
            msg = (f"WARNING: glyph_denoise {eff_glyph_denoise:.2f} exceeds the measured safe "
                   f"ceiling of {GLYPH_DENOISE_SAFE_MAX}. Above it the sampler stops treating the "
                   f"typeset layer as a template: a ghost copy of the word appears behind the "
                   f"real one, and past ~0.70 the whole sign gets reinvented.")
            report.append(msg)
            print(f"[SignDetailer] {msg}")

        ink_override = parse_hex_rgb(glyph_ink_color)
        plate_override = parse_hex_rgb(glyph_plate_color)
        for label, raw, parsed in (("glyph_ink_color", glyph_ink_color, ink_override),
                                   ("glyph_plate_color", glyph_plate_color, plate_override)):
            if raw and raw.strip() and parsed is None:
                msg = f"WARNING: could not parse {label}={raw!r} — falling back to the sampled colour"
                report.append(msg)
                print(f"[SignDetailer] {msg}")
        # Forcing a plate colour means repainting the surface, which is exactly
        # what preserve_surface refuses to do. Honour the explicit instruction.
        if plate_override is not None and glyph_preserve_surface:
            glyph_preserve_surface = False
            report.append("NOTE: glyph_plate_color is set, so glyph_preserve_surface was turned "
                          "off for this run — forcing a new surface colour and preserving the old "
                          "surface are opposite instructions.")

        if plate_override is not None and not opts["prompt_suffix"]:
            report.append("NOTE: a plate colour is forced but prompt_suffix is empty — the typeset "
                          "layer will carry the new surface colour while the prompt still describes "
                          "the old surface. Describe the surface in Sign Options prompt_suffix.")

        result = images.clone()
        refined_crops = []
        glyph_previews = []
        rendered = skipped = softened = retried = 0

        for b in range(images.shape[0]):
            batch_regions = [r for r in regions if r.get("batch_index", 0) == b]
            if not batch_regions:
                continue
            current = result[b]

            for region in batch_regions:
                idx = region.get("index", 0)
                cls_name = region.get("class", "sign")
                proposal = region.get("proposal") or {}
                text = (proposal.get("text") or "").strip()

                if cls_name in opts["class_skip"]:
                    report.append(f"  #{idx + 1} {cls_name}: skipped by sign_options")
                    skipped += 1
                    continue

                soften = False
                if region.get("too_small"):
                    if too_small_policy == "skip":
                        report.append(f"  #{idx + 1} {cls_name}: {region['height_px']}px — skipped (too small)")
                        skipped += 1
                        continue
                    if too_small_policy == "soften":
                        soften = True

                if not text and not soften:
                    report.append(f"  #{idx + 1} {cls_name}: no text proposed — skipped")
                    skipped += 1
                    continue

                # Prompt and denoise for this region
                if soften:
                    prompt = SOFTEN_PROMPT
                    region_denoise = 0.35
                    use_glyph = False
                else:
                    prompt = build_prompt(cls_name, text, proposal.get("style", ""))
                    if opts["prompt_suffix"]:
                        prompt = f"{prompt}, {opts['prompt_suffix']}"
                    use_glyph = glyph_guidance != "off"
                    base_denoise = eff_glyph_denoise if use_glyph else denoise
                    region_denoise = base_denoise + get_class(cls_name)["denoise_bias"]
                    region_denoise = float(np.clip(
                        opts["class_denoise"].get(cls_name, region_denoise), 0.05, 1.0))

                positive_cond = self._encode(clip, prompt) if prompt else (
                    positive_base if positive_base is not None else self._encode(clip, ""))

                tw, th = self._resolve_target(region, target_width, target_height,
                                              max_upscale, auto_resolution)
                if (tw, th) != (target_width, target_height):
                    report.append(f"  #{idx + 1} sampling at {tw}x{th}")

                # Cluster members share a seed so identical objects stay identical
                if cluster_mode == "shared_seed" and region.get("cluster_id", -1) >= 0:
                    region_seed = seed + region["cluster_id"] * 1000
                else:
                    region_seed = seed + idx

                attempts = max_attempts if (verify_after == "ocr" and not soften and text) else 1
                stitched, refined, verdict = current, None, ""

                for attempt in range(attempts):
                    work_image = current
                    glyph_rgb, hot = None, None
                    if use_glyph and text:
                        work_image, glyph_rgb, hot = self._apply_glyph(
                            current, region, text, glyph_font, eff_strength,
                            glyph_autocolor, opts["uppercase"], opts["margin_ratio"],
                            ink_override=ink_override, plate_override=plate_override,
                            fit=glyph_fit, cylinder=glyph_cylinder,
                            preserve_surface=glyph_preserve_surface,
                            match_source_size=glyph_match_source_size)
                        if glyph_rgb is not None and attempt == 0:
                            glyph_previews.append(torch.from_numpy(
                                np.clip(glyph_rgb, 0, 1).astype(np.float32)))

                    region_mask = torch.from_numpy(region["mask"])
                    noise_mask = self._split_noise_mask(
                        region_mask, hot, glyph_surface_restyle,
                        fill_holes=opts["mask_fill_holes"])
                    if noise_mask is not None and attempt == 0:
                        share = float((hot > 0.5).sum()) / max(1.0, float((region["mask"] > 0.5).sum()))
                        report.append(f"  #{idx + 1} new lettering covers {share * 100:.0f}% of the "
                                      f"region; the rest is sampled at {glyph_surface_restyle:.2f}x")

                    stitched, refined = inpaint_slot(
                        image=work_image,
                        mask_2d=region_mask,
                        noise_mask_2d=noise_mask,
                        model=model,
                        positive_cond=positive_cond,
                        negative_cond=negative_cond,
                        vae=vae,
                        seed=region_seed + attempt * 7919,
                        steps=steps,
                        denoise=region_denoise,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        target_width=tw,
                        target_height=th,
                        mask_expand_pixels=mask_expand_pixels,
                        mask_blend_pixels=mask_blend_pixels,
                        mask_fill_holes=opts["mask_fill_holes"],
                        context_expand_factor=opts["context_expand_factor"],
                        output_padding=opts["output_padding"],
                        dd_enabled=detail_daemon_enabled,
                        dd_amount=detail_amount,
                        dd_options=dd_options or DD_DEFAULTS,
                        denoise_progression=opts["denoise_progression"],
                        steps_progression=opts["steps_progression"],
                        cfg=opts["cfg"],
                    )

                    if refined is None:
                        verdict = "empty mask"
                        break

                    if attempts == 1:
                        break

                    read_back = ocr_region(
                        (np.clip(stitched.cpu().numpy(), 0, 1) * 255).astype(np.uint8),
                        region["mask"])
                    similarity = _fuzzy_match(read_back.get("text", ""), text)
                    if similarity >= verify_similarity:
                        verdict = f"verified {similarity:.2f}"
                        break
                    verdict = f"read back {read_back.get('text', '')!r} ({similarity:.2f})"
                    if attempt < attempts - 1:
                        retried += 1
                        report.append(f"  #{idx + 1} attempt {attempt + 1} failed: {verdict} — retrying")

                current = stitched
                if refined is not None:
                    refined_crops.append(refined)
                    if soften:
                        softened += 1
                        report.append(f"  #{idx + 1} {cls_name}: softened ({region['height_px']}px)")
                    else:
                        rendered += 1
                        report.append(
                            f"  #{idx + 1} {cls_name}: {text!r} denoise={region_denoise:.2f} "
                            f"seed={region_seed}{' — ' + verdict if verdict else ''}")

            result[b] = current

        report.append(f"Summary: {rendered} rendered, {softened} softened, "
                      f"{skipped} skipped, {retried} retries")

        if refined_crops:
            size = refined_crops[0].shape[1:3]
            normalized = []
            for c in refined_crops:
                if c.shape[1:3] != size:
                    c = torch.nn.functional.interpolate(
                        c.permute(0, 3, 1, 2), size=size, mode="bilinear", align_corners=False
                    ).permute(0, 2, 3, 1)
                normalized.append(c)
            crops_out = torch.cat(normalized, dim=0)
        else:
            crops_out = torch.zeros(1, 64, 64, 3, dtype=torch.float32)

        if glyph_previews:
            gp_size = glyph_previews[0].shape[:2]
            gp = [g if g.shape[:2] == gp_size else torch.from_numpy(
                np.array(torch.nn.functional.interpolate(
                    g.permute(2, 0, 1).unsqueeze(0), size=gp_size,
                    mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)))
                  for g in glyph_previews]
            glyph_out = torch.stack(gp)
        else:
            glyph_out = torch.zeros(1, 64, 64, 3, dtype=torch.float32)

        print(f"[SignDetailer] {rendered} rendered, {softened} softened, {skipped} skipped")
        return (result, crops_out, glyph_out, "\n".join(report))
