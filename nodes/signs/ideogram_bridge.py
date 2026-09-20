"""SIGN_DATA -> an Ideogram 4 caption, so the whole chain fits on the canvas.

The selector finds the lettered surfaces and the proposer asks a local model what
should stand on each of them. Both already run as nodes. What was missing was the
piece that turns their result into the one thing Ideogram 4 actually consumes: a
structured caption with a bounding box per text element.

It is wired into `Ideogram4PromptBuilderKJ`'s `import_json` input, with
`import_mode` set to `always`. That is not a preference. The builder's
`elements_data` widget is re-serialised from a live array inside the browser
every time the graph is queued, so anything written into it from outside is
overwritten the instant you press run — silently. `import_json` with
`import_mode=always` is the only path the node itself treats as authoritative.

Everything this node decides was measured elsewhere in this project and is
written up in `tests/live/ideogram/JOURNAL.md`:

* **The mask is generous, the text box is tight.** They pull in opposite
  directions and one rectangle serves neither: a mask that misses a stroke leaves
  the old scribble standing bit-identically (`AHMENWERK`, `ANTIQUARIA`), and a
  text box that reaches the edge of a surface puts letters past that edge
  (`WEINHANDEL` running off its panel).
* **A surface that cannot carry a readable word is emptied, not written on.**
  Anything that renders letter-like structure gets transcribed and counted as
  invented text, however soft. The selector already decides this per region
  (`too_small`), so the judgement is not made twice with two different rules.
* **The blur radius follows the smallest shape.** A constant radius is wrong
  once regions differ in size by a lot, and wrong in the dangerous direction:
  nine pixels of blur on an eight-pixel line spreads the mask further than the
  line is tall, into whatever sits above it.
* **The caption states the camera, not the design.** Ideogram's untended pull is
  poster art; without an explicit photographic lock it answers a sign request
  with flat vector lettering on a clean plate.
"""

import json

import numpy as np
import torch

# Ideogram's own guidance, confirmed here: describing a surface as photographic
# is the counterweight to its pull towards poster art.
PHOTO_STYLE = {
    "aesthetics": "unretouched documentary photograph, real place, real weather",
    "medium": "35mm photograph, shallow depth of field, natural film grain",
    "art_style": (
        "strictly photographic: NOT an illustration, NOT vector art, "
        "NOT graphic design, NOT a poster, NOT a 3D render"
    ),
    "lighting": "existing available light only, matching the rest of the frame",
}

FLAT_STYLE = {
    "aesthetics": "a plain flat sign, evenly lit, no scene around it",
    "medium": "flat matte paint on a flat panel, no grain, no texture",
    "art_style": (
        "clean and flat, exactly as the surrounding image already is; "
        "do NOT add photographic texture, depth of field or lighting"
    ),
    "lighting": "flat and even, no highlights, no shadows",
}

SURFACE_REALISM = (
    "The lettering is physically part of the surface it sits on: it shares the "
    "same focus, the same grain, the same light and the same wear as everything "
    "around it. It is painted, enamelled, printed or carved onto that object, "
    "never overlaid on top of it. No drop shadow, no glow, no outline, no "
    "sticker edge, no second copy of the text showing through."
)

BLANK_TAIL = (
    "There is no text, no lettering, no writing, no numbers and no "
    "printed characters anywhere in this area — the surface is blank."
)

# What a class of surface looks like, so the caption says something true about
# the material instead of describing every region as "a lettered surface".
KLASSEN = {
    "sign": "a painted or enamelled shop sign mounted on the building",
    "label": "a paper label stuck to a bottle or package",
    "poster": "a printed poster or banner on a wall",
    "screen": "a lit display screen",
    "book": "a printed book or magazine cover",
    "plate": "a stamped metal licence plate",
    "garment_print": "print applied to woven fabric, following its folds",
    "paper": "a sheet of paper",
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _auto_feather(boxes, share=0.25, lo=2, hi=9):
    """Blur radius that fits the SMALLEST region in the set."""
    if not boxes:
        return lo
    kleinste = min(min(b[2] - b[0], b[3] - b[1]) for b in boxes)
    return int(_clamp(round(kleinste * share), lo, hi))


def _soft_union(boxes, h, w, feather, pad):
    """Union of the boxes, blurred, then hardened so the far field is exact."""
    import cv2

    m = np.zeros((h, w), np.float32)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(
            m,
            (max(0, int(x0 - pad)), max(0, int(y0 - pad))),
            (min(w - 1, int(x1 + pad)), min(h - 1, int(y1 + pad))),
            1.0,
            -1,
        )
    if feather:
        k = int(feather) * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    # Below this the rim is invisible but still rewrites pixels; clipping it is
    # what makes "unchanged" mean unchanged.
    m = np.clip((m - 0.02) / 0.98, 0.0, 1.0)
    return m


SZENE_SYSTEM = (
    "You describe a photograph so that a text-to-image model can repaint a small "
    "part of it without the repair looking pasted on. Answer with ONE JSON "
    'object, keys exactly {"summary", "background"}. '
    '"summary" is one sentence naming what the picture shows. '
    '"background" is two or three sentences on the light, the weather, the '
    "materials and the depth of field — everything a painter would need to match "
    "the surroundings. Name no text and no lettering; the writing in this picture "
    "is nonsense and is about to be replaced."
)


def _describe_scene(image, base_url, model_id):
    """Ask a local vision model what the picture is. Returns (dict|None, reason)."""
    try:
        from ..utils.lmstudio_client import chat_vision, parse_json_response
    except Exception:
        # Also reachable when the module is loaded standalone (a probe script, a
        # unit test), where the relative import has no package to resolve
        # against. Worth the four lines: the alternative is a helper that only
        # works inside ComfyUI and therefore cannot be checked before a restart.
        try:
            import importlib.util
            import os

            pfad = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "utils",
                "lmstudio_client.py",
            )
            spec = importlib.util.spec_from_file_location("_lmc", pfad)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            chat_vision, parse_json_response = mod.chat_vision, mod.parse_json_response
        except Exception as e:  # noqa: BLE001
            return None, f"Client nicht ladbar: {e}"
    try:
        rgb = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        res = chat_vision(
            base_url=base_url,
            model_id=model_id,
            system_prompt=SZENE_SYSTEM,
            user_prompt="Describe this photograph. Report the JSON.",
            images=[rgb],
            temperature=0.0,
            max_tokens=300,
            timeout=180,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    if not res.get("ok"):
        return None, str(res.get("error"))[:120]
    d = parse_json_response(res.get("content", "")) or {}
    return (
        (d, "") if d.get("summary") or d.get("background") else (None, "leere Antwort")
    )


class FVM_SignsToIdeogram:
    """Turn selected sign regions into an Ideogram 4 caption plus its mask."""

    CATEGORY = "FVMtools/signs"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "STRING", "MASK", "BOUNDING_BOX", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "caption_json",
        "box_prompts",
        "mask",
        "bboxes",
        "width",
        "height",
        "report",
    )
    DESCRIPTION = (
        "SIGN_DATA -> Ideogram 4 caption JSON. Wire `caption_json` into "
        "Ideogram4PromptBuilderKJ's `import_json` and set `import_mode` to "
        "'always' — the builder's own elements_data is re-serialised from the "
        "browser at queue time, so any other route is silently overwritten."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sign_data": (
                    "SIGN_DATA",
                    {
                        "tooltip": "From Sign Text Proposer (or the selector, if you "
                        "supply the words yourself)."
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The image the selector scanned. Only its size is "
                        "used, to convert boxes into Ideogram's 0-1000 grid."
                    },
                ),
                "high_level_description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "One line about the whole picture. Blank is allowed.",
                    },
                ),
                "background": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "The scene around the lettering: light, weather, "
                        "materials. This is what keeps the replacement in "
                        "the same photograph as the rest of the frame.",
                    },
                ),
                "look": (
                    ["photographic", "flat graphic"],
                    {
                        "default": "photographic",
                        "tooltip": "Ideogram drifts towards poster art unless told "
                        "otherwise. Use 'flat graphic' only when the source "
                        "genuinely is one.",
                    },
                ),
                "text_inset": (
                    "FLOAT",
                    {
                        "default": 0.07,
                        "min": 0.0,
                        "max": 0.45,
                        "step": 0.01,
                        "tooltip": "How far the TEXT box sits inside the region. The "
                        "mask stays full size. A box that reaches the edge "
                        "of a surface puts letters past that edge.",
                    },
                ),
                "mask_pad_px": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 64,
                        "tooltip": "Extra mask around each region. Err high: what the "
                        "mask misses is restored unchanged and stands there "
                        "as leftover scribble.",
                    },
                ),
                "empty_too_small": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Regions the selector judged too small for a "
                        "readable word are described as blank instead of "
                        "being written on. Off: every region gets a word, "
                        "which produces letter-like shapes that count as "
                        "invented text.",
                    },
                ),
                "min_word_px": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Box heights below this get no word — the "
                        "surface is described as blank instead. 0 turns the "
                        "test off.\n\n"
                        "The selector has its own height floor, but it measures "
                        "the region BEFORE the line splitter cuts it into bands, "
                        "so a 90px sign split into four passes the selector and "
                        "arrives here as four 20px strips that the splitter only "
                        "checks against its own 8px 'is this a line at all' bar. "
                        "This is the last station before the caption and the only "
                        "one that sees the geometry that will actually be "
                        "rendered.",
                    },
                ),
                "fallback_text": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Used for a region the proposer left without a "
                        "word. Blank means that region is emptied.",
                    },
                ),
            },
            "optional": {
                "auto_scene": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Look at the picture and write the scene "
                        "description itself when the two text fields above are "
                        "empty. That is what lets the graph run from nothing but "
                        "a chosen image.",
                    },
                ),
                "lm_base_url": (
                    "STRING",
                    {
                        "default": "http://localhost:1234/v1",
                        "tooltip": "LM Studio endpoint for the scene description.",
                    },
                ),
                "lm_model_id": (
                    "STRING",
                    {
                        "default": "qwen3-8b-vl-instruct-abliterated",
                        "tooltip": "Name it. 'Whatever is loaded' is not a "
                        "default, it is a silent failure: LM Studio lists models "
                        "it has not loaded, the call returns nothing, and the "
                        "run continues with an empty description. Must be a "
                        "VISION model.",
                    },
                ),
                "surface_hint": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Per class, one per line, 'class: description'. "
                        "Overrides the built-in wording for that class.",
                    },
                ),
            },
        }

    def build(
        self,
        sign_data,
        image,
        high_level_description,
        background,
        look,
        text_inset,
        mask_pad_px,
        empty_too_small,
        min_word_px,
        fallback_text,
        auto_scene=True,
        lm_base_url="http://localhost:1234/v1",
        lm_model_id="qwen3-8b-vl-instruct-abliterated",
        surface_hint="",
    ):
        h, w = int(image.shape[1]), int(image.shape[2])
        regionen = list((sign_data or {}).get("regions") or [])

        eigene = {}
        for zeile in (surface_hint or "").splitlines():
            if ":" in zeile:
                k, _, v = zeile.partition(":")
                eigene[k.strip().lower()] = v.strip()

        # A surface that is about to be wiped cannot also carry signs. SAM3 finds
        # small sign-like things INSIDE a shop window — plates propped behind the
        # glass — and they arrive as their own regions sitting on top of a region
        # already marked "clear this". The caption then says, of the same pixels,
        # both "there is no writing anywhere in this area" and "write WATCHMAKER
        # here". Measured result: the model resolved the contradiction by
        # inventing an entire reflected street where the shop interior had been.
        # The wipe wins — it is the thing that removes the original nonsense, and
        # clean glass is what looked right before these children appeared.
        flaechen = [
            tuple(float(v) for v in r["bbox"])
            for r in regionen
            if r.get("too_big") and r.get("bbox")
        ]

        def steckt_drin(b):
            x0, y0, x1, y1 = (float(v) for v in b)
            fl = max(1.0, (x1 - x0) * (y1 - y0))
            for a0, b0, a1, b1 in flaechen:
                if (x0, y0, x1, y1) == (a0, b0, a1, b1):
                    continue
                ux = max(0.0, min(x1, a1) - max(x0, a0))
                uy = max(0.0, min(y1, b1) - max(y0, b0))
                if ux * uy / fl >= 0.80:
                    return True
            return False

        elements, box_prompts, boxes, bericht = [], {}, [], []
        beschriftet = geleert = verdeckt = 0
        for i, r in enumerate(regionen):
            if flaechen and r.get("bbox") and steckt_drin(r["bbox"]):
                verdeckt += 1
                bericht.append(
                    f"  #{i + 1} liegt in einer zu leerenden Flaeche — uebergangen"
                )
                continue
            bbox = r.get("bbox")
            if not bbox:
                continue
            x0, y0, x1, y1 = (float(v) for v in bbox)
            boxes.append((x0, y0, x1, y1))

            klasse = str(r.get("class") or "sign").lower()
            flaeche = eigene.get(klasse) or KLASSEN.get(klasse, "a lettered surface")
            wort = ((r.get("proposal") or {}).get("text") or "").strip()
            zu_klein = bool(r.get("too_small")) and empty_too_small
            # Two different reasons to write nothing, one outcome. Too small: a
            # word would be six pixels a glyph. Too big: it is a shop window, and
            # a word there becomes a plank. Both get the surface described and
            # left blank — which is what actually clears the original nonsense,
            # rather than covering it.
            zu_gross = bool(r.get("too_big"))
            zu_flach = bool(min_word_px) and (y1 - y0) < float(min_word_px)
            leeren = zu_klein or zu_gross or zu_flach
            if not wort and not leeren:
                wort = fallback_text.strip()

            # Ideogram's grid is [top, left, bottom, right] on 0-1000 — not
            # pixels and not x/y/w/h. Read the other way round the word lands
            # somewhere else entirely and it looks like a model failure.
            def to1000(a, b, c, d, inset):
                dx, dy = (c - a) * inset, (d - b) * inset
                return [
                    round((b + dy) / h * 1000),
                    round((a + dx) / w * 1000),
                    round((d - dy) / h * 1000),
                    round((c - dx) / w * 1000),
                ]

            if wort and not leeren:
                # The text box is the region. A previous version capped it to the
                # height one line of the word actually needs, to stop a word
                # being stretched over a three-line block — and it made things
                # worse in the way this project has now measured twice: the MASK
                # stays the size of the region, so shrinking the text box
                # ENLARGES the masked area that no element describes, and the
                # model fills it. On the shelf that produced small print under
                # every name; here it produced a large white enamel plaque
                # invented to carry the word, laid across a shop window.
                #
                # Mask and text box must agree about how much surface is being
                # claimed. If a region genuinely holds several lines, the answer
                # is to split the REGION, not to shrink the box inside it.
                ty0, ty1 = y0, y1
                # Use what the proposer SAW, not a stock sentence about the
                # class. It looked at this crop and came back with "vintage
                # enamel sign, blue on white circular plate" and "gold leaf on
                # dark red shopfront board"; the generic line said "a painted or
                # enamelled shop sign mounted on the building" for all of them,
                # and the model duly mounted boards over a shop window instead of
                # painting on its glass. The observation was already there and
                # was being thrown away.
                p = r.get("proposal") or {}
                gesehen = (p.get("style") or "").strip()
                schrift = (p.get("font_hint") or "").strip()
                desc = "; ".join(x for x in (gesehen or flaeche, schrift) if x)
                desc += (
                    ". The word is set on ONE single line, sized to fill "
                    "the width of this surface, painted or printed directly "
                    "onto it — not on a new sign added in front of it."
                )
                elements.append(
                    {
                        "type": "text",
                        "bbox": to1000(x0, ty0, x1, ty1, text_inset),
                        "text": wort,
                        "desc": desc,
                    }
                )
                box_prompts[f"box{i + 1}"] = wort
                beschriftet += 1
                bericht.append(f"  #{i + 1} {klasse}: {wort!r} [{y1 - y0:.0f}px]")
            else:
                if zu_gross:
                    # Deliberately does NOT say "window". The size test says this
                    # is a surface, not which surface — asserting glass over what
                    # turns out to be a wall would trade one invention for
                    # another. What is true either way is that it should carry on
                    # looking like whatever surrounds it, and carry no writing.
                    grund = "Flaeche statt Schild"
                    was = (
                        "the plain shopfront surface here, continuing the same "
                        "material, colour, reflections and light as the area "
                        "immediately around it, with nothing mounted on it and "
                        "nothing hanging in front of it"
                    )
                elif zu_flach:
                    grund = f"nur {y1 - y0:.0f} px hoch, unter {min_word_px} px"
                    was = f"{flaeche}, bare and unmarked"
                else:
                    grund = (
                        "zu klein fuer ein lesbares Wort" if zu_klein else "kein Wort"
                    )
                    was = f"{flaeche}, bare and unmarked"
                elements.append(
                    {
                        "type": "obj",
                        "bbox": to1000(x0, y0, x1, y1, 0.0),
                        "desc": f"{was}. {BLANK_TAIL}",
                    }
                )
                geleert += 1
                bericht.append(f"  #{i + 1} {klasse}: geleert ({grund})")

        if auto_scene and not (high_level_description.strip() or background.strip()):
            gesehen, warum = _describe_scene(image, lm_base_url, lm_model_id)
            if gesehen:
                high_level_description = high_level_description.strip() or gesehen.get(
                    "summary", ""
                )
                background = background.strip() or gesehen.get("background", "")
                bericht.append(f"  Szene selbst gelesen: {high_level_description!r}")
            else:
                # Said out loud. A blank description does not fail the run, it
                # just quietly removes the one thing that keeps the replacement
                # inside the same photograph as the rest of the frame.
                bericht.append(
                    f"  ACHTUNG Szene NICHT gelesen ({warum}) — "
                    f"die Beschreibung bleibt leer"
                )

        stil = FLAT_STYLE if look == "flat graphic" else PHOTO_STYLE
        caption = {
            "high_level_description": high_level_description.strip()
            or "A photograph of a real place.",
            "style_description": dict(stil, color_palette=[]),
            "compositional_deconstruction": {
                "background": f"{background.strip()} {SURFACE_REALISM}".strip(),
                "elements": elements,
            },
            "_canvas": {"width": w, "height": h},
        }

        feather = _auto_feather(boxes)
        m = _soft_union(boxes, h, w, feather, mask_pad_px)
        mask = torch.from_numpy(m).unsqueeze(0)

        kopf = (
            f"[SignsToIdeogram] {len(elements)} Elemente "
            f"({beschriftet} beschriftet, {geleert} geleert, {verdeckt} verdeckt), "
            f"Leinwand {w}x{h}, Feder {feather} px, Rand {mask_pad_px} px"
        )
        print(kopf)
        # Pixel boxes for the builder's own `bboxes` socket. The caption is what
        # actually drives the render; these only seed the editor so the regions
        # are VISIBLE on the canvas instead of being an invisible string. Width
        # and height come from the image, so the builder's canvas can be wired
        # instead of typed — a mismatch there (1312x2320 against a 1280x768
        # picture) puts every box in the wrong place and looks like bad detection.
        pixel = [
            {
                "x": int(b[0]),
                "y": int(b[1]),
                "width": int(b[2] - b[0]),
                "height": int(b[3] - b[1]),
            }
            for b in boxes
        ]
        return (
            json.dumps(caption, ensure_ascii=False, indent=1),
            json.dumps(box_prompts, ensure_ascii=False, indent=1),
            mask,
            pixel,
            w,
            h,
            "\n".join([kopf] + bericht),
        )


NODE_CLASS_MAPPINGS = {"FVM_SignsToIdeogram": FVM_SignsToIdeogram}
NODE_DISPLAY_NAME_MAPPINGS = {"FVM_SignsToIdeogram": "Signs to Ideogram Caption (FVM)"}
