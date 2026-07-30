"""SignTextProposer — asks a vision LLM what each detected sign should actually say.

Talks to LM Studio's OpenAI-compatible endpoint directly over HTTP, so FVMtools
stays standalone. For every region (or once per cluster) the model sees the crop,
the whole scene and optionally the neighbouring regions, and answers with JSON:
the replacement text, a style hint, a font hint, and its own judgement of whether
the original lettering was already legible.

Manual overrides always win over the model. With LM Studio unreachable the node
still produces usable output from the fallback list instead of failing the run.
"""

import difflib
import re

import numpy as np
import torch

from ..utils.lmstudio_client import (
    DEFAULT_BASE_URL, DEFAULT_SYSTEM_PROMPT, DEFAULT_TIMEOUT, DEFAULT_TEMPERATURE,
    propose_text, probe,
)
from ..utils.tensor_utils import tensor2np
try:  # relative inside ComfyUI's loader, absolute under pytest
    from ...core.signs.classes import get_class
except ImportError:
    from core.signs.classes import get_class


def _content_words(text):
    """Subject-carrying words of a proposal, upper-cased. Numbers dropped."""
    words = set()
    for raw in re.split(r"[^0-9A-Za-zÄÖÜäöüß\-]+", str(text or "")):
        w = raw.strip("-").upper()
        if len(w) >= 3 and not any(c.isdigit() for c in w):
            words.add(w)
    return words


def is_too_similar(text, existing, word_overlap=0.6, string_ratio=0.75):
    """Is this proposal a re-run of one already handed out?

    Two checks, because the model repeats itself in two different ways: it
    reuses the same subject with a new number ("Coffee Break $3.50" after
    "Coffee Break $5"), which the word overlap catches, and it reuses the whole
    phrase with different punctuation, which the string ratio catches.
    """
    candidate = _content_words(text)
    flat = "".join(str(text).upper().split())
    if not flat:
        return False
    for other in existing:
        other_flat = "".join(str(other).upper().split())
        if not other_flat:
            continue
        if difflib.SequenceMatcher(None, flat, other_flat).ratio() >= string_ratio:
            return True
        other_words = _content_words(other)
        if candidate and other_words:
            shared = len(candidate & other_words)
            if shared / min(len(candidate), len(other_words)) >= word_overlap:
                return True
    return False


def _parse_overrides(spec):
    """Parse 'index: text' lines into {zero_based_index: text}.

    Accepts '3: ACHTUNG', '3 = ACHTUNG' and '3 ACHTUNG'. Indices are 1-based in
    the UI (matching the preview labels) and stored zero-based.
    """
    out = {}
    if not spec:
        return out
    for line in spec.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        idx_str, sep, text = "", "", ""
        for delim in (":", "="):
            if delim in line:
                idx_str, sep, text = line.partition(delim)
                break
        if not sep:
            parts = line.split(None, 1)
            if len(parts) == 2:
                idx_str, text = parts
        try:
            idx = int(idx_str.strip()) - 1
        except (ValueError, AttributeError):
            continue
        if idx >= 0:
            out[idx] = text.strip()
    return out


def _parse_fallbacks(spec):
    """Fallback texts, either 'class: text' keyed or a plain cyclic list."""
    keyed, plain = {}, []
    if not spec:
        return keyed, plain
    for line in spec.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, text = line.partition(":")
            key, text = key.strip().lower(), text.strip()
            if key and text:
                keyed[key] = text
                continue
        plain.append(line)
    return keyed, plain


class SignTextProposer:
    """Turns detected text regions into concrete replacement texts."""

    CATEGORY = "FVM Tools/Text"
    FUNCTION = "execute"
    RETURN_TYPES = ("SIGN_DATA", "STRING", "STRING")
    RETURN_NAMES = ("sign_data", "proposed_texts", "report")
    OUTPUT_NODE = True

    DESCRIPTION = (
        "Asks a vision model in LM Studio what each detected sign should read.\n\n"
        "Sends the crop plus the whole scene, so the proposal fits the setting instead of\n"
        "guessing from a floating rectangle. One call per cluster keeps a shelf of twelve\n"
        "identical bottles at one request.\n\n"
        "Precedence: manual_override > model proposal > fallback list > existing OCR text.\n"
        "LM Studio unreachable is not an error — the node falls back and says so in the report.\n\n"
        "Talks plain HTTP to the OpenAI-compatible endpoint; no other extension required."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sign_data": ("SIGN_DATA", {"tooltip": "Regions from Sign Selector SAM3"}),
                "image": ("IMAGE", {"tooltip": "The same image the selector scanned — used as scene context"}),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL,
                    "tooltip": "LM Studio OpenAI-compatible endpoint"}),
                "model_id": ("STRING", {"default": "",
                    "tooltip": "Model id as listed by LM Studio. Empty = use whatever is loaded."}),
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "OFF: skip the model entirely and use overrides plus fallbacks only."}),
                "context_mode": (["crop+scene", "crop_only", "crop+scene+neighbors"], {"default": "crop+scene",
                    "tooltip": "How much the model sees.\n"
                               "crop_only is cheapest but invents text that ignores the setting.\n"
                               "Neighbours help a row of shopfronts stay coherent."}),
                "scene_hint": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Overrides the model's read of the setting, e.g. 'Berlin, 1985' or 'rural Japan'."}),
                "language": (["auto", "en", "de", "fr", "es", "it", "ja", "zh"], {"default": "auto",
                    "tooltip": "Language for the proposed text. 'auto' lets the model follow the scene."}),
                "temperature": ("FLOAT", {"default": DEFAULT_TEMPERATURE, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Keep at or below 0.2. Measured cliff, not a slope: at 0.2 the model\n"
                               "never transcribes the garbled original, at 0.25 it does so in half of\n"
                               "all runs — it lands in the near-miss token neighbourhood and returns\n"
                               "e.g. 'CAFFEE' because the setting makes that spelling feel authentic.\n"
                               "Picking the right word for a sign is a low-entropy task; it does not\n"
                               "benefit from sampling variety."}),
                "max_tokens": ("INT", {"default": 256, "min": 32, "max": 4096, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Passed through to LM Studio for reproducible proposals"}),
                "one_call_per_cluster": ("BOOLEAN", {"default": True,
                    "tooltip": "ON: only the cluster representative is sent; siblings inherit its text."}),
                "variety_retries": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1,
                    "tooltip": "How often to ask again when the answer repeats text already used\n"
                               "elsewhere in this picture. 0 = accept the first answer.\n\n"
                               "The ban list alone does not always land — the model will return\n"
                               "the same subject with a different price. Each retry says so\n"
                               "explicitly and uses a different seed."}),
                "avoid_repeats": ("BOOLEAN", {"default": True,
                    "tooltip": "Tell the model which wording it already used elsewhere in this\n"
                               "picture, so similar-looking motifs get different text.\n\n"
                               "Each region is a separate request — without this the model has no\n"
                               "memory of its own answers and returns the same name for every\n"
                               "bottle on a shelf. Raising temperature would also break the tie,\n"
                               "but brings back transcription of the original gibberish, so the\n"
                               "variety comes from a constraint instead.\n\n"
                               "Cluster siblings still share their text — this only separates\n"
                               "regions that were NOT grouped together."}),
                "skip_legible": ("BOOLEAN", {"default": False,
                    "tooltip": "ON: regions the selector judged already legible are left untouched."}),
                "timeout": ("INT", {"default": DEFAULT_TIMEOUT, "min": 5, "max": 600, "step": 5}),
                "manual_override": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "One per line, 'index: text' using the numbers from the preview.\n"
                               "Example:\n3: ACHTUNG\n7: Café Mozart\nAlways wins over the model."}),
                "fallback_texts": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Used when the model is unreachable or returns nothing.\n"
                               "Either 'class: text' lines (sign: OPEN) or a plain list cycled per region."}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": DEFAULT_SYSTEM_PROMPT, "multiline": True,
                    "tooltip": "System prompt. Must keep demanding a single JSON object."}),
                "class_instructions": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Per-class extra instruction, 'class: instruction' per line.\n"
                               "Example: plate: use a German plate format like B-XY 1234"}),
            },
        }

    def _neighbors(self, regions, current, limit=3):
        """Closest other regions by bbox centre distance."""
        cx = (current["bbox"][0] + current["bbox"][2]) / 2
        cy = (current["bbox"][1] + current["bbox"][3]) / 2
        scored = []
        for r in regions:
            if r is current:
                continue
            ox = (r["bbox"][0] + r["bbox"][2]) / 2
            oy = (r["bbox"][1] + r["bbox"][3]) / 2
            scored.append((((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5, r))
        scored.sort(key=lambda t: t[0])
        return [r for _, r in scored[:limit]]

    def execute(self, sign_data, image, base_url=DEFAULT_BASE_URL, model_id="", enabled=True,
                context_mode="crop+scene", scene_hint="", language="auto",
                temperature=DEFAULT_TEMPERATURE, max_tokens=256, seed=0, one_call_per_cluster=True,
                avoid_repeats=True, variety_retries=2,
                skip_legible=False, timeout=DEFAULT_TIMEOUT, manual_override="",
                fallback_texts="", system_prompt=None, class_instructions=""):

        # Shallow-copy each region before writing proposals into it. ComfyUI hands
        # every downstream node the SAME cached object, so mutating in place would
        # let two Proposers on one Selector overwrite each other's texts. The heavy
        # values (mask, crop) stay shared by reference — nothing mutates them.
        source_regions = sign_data.get("regions", []) if isinstance(sign_data, dict) else []
        regions = [dict(r) for r in source_regions]
        sign_data = {**sign_data, "regions": regions} if isinstance(sign_data, dict) else sign_data
        overrides = _parse_overrides(manual_override)
        fb_keyed, fb_plain = _parse_fallbacks(fallback_texts)
        class_instr = {}
        for line in (class_instructions or "").splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                class_instr[k.strip().lower()] = v.strip()

        report = [f"Sign Text Proposer — {len(regions)} region(s)"]

        if enabled and temperature > DEFAULT_TEMPERATURE:
            warning = (f"WARNING: temperature {temperature:.2f} is above the measured cliff at "
                       f"{DEFAULT_TEMPERATURE}. Above it the model starts transcribing the garbled "
                       f"original instead of replacing it (half of all runs at 0.25). Lower it "
                       f"unless you are deliberately trading correctness for variety.")
            report.append(warning)
            print(f"[SignTextProposer] {warning}")

        status = probe(base_url, timeout=5) if enabled else {"reachable": False, "models": [], "error": "disabled"}
        if enabled:
            if status.get("reachable"):
                found = status.get("models", [])
                report.append(f"LM Studio reachable, {len(found)} model(s): {', '.join(found[:4])}")
                if model_id and found and model_id not in found:
                    report.append(f"WARNING: '{model_id}' is not in the list — LM Studio may reject the call")
            else:
                report.append(f"LM Studio NOT reachable ({status.get('error')}) — using overrides and fallbacks")
        else:
            report.append("Model disabled by the enabled toggle — using overrides and fallbacks")

        use_model = enabled and status.get("reachable", False)
        scene_rgb = tensor2np(image[0:1]) if image is not None and image.shape[0] > 0 else None

        cluster_cache = {}
        fb_cursor = 0
        made, inherited, failed = 0, 0, 0
        retried, duplicates = 0, 0
        # Wording already used in this picture. Keyed by cluster so siblings can
        # still share a text — only regions that were NOT grouped are pushed apart.
        used_texts = {}          # cluster_id (or -1-index) -> text

        for i, region in enumerate(regions):
            cls_name = region.get("class", "sign")

            if i in overrides:
                region["proposal"] = {
                    "text": overrides[i], "style": "", "font_hint": "",
                    "legible_original": 0.0, "confidence": 1.0,
                    "ok": True, "error": None, "source": "manual",
                }
                # A hand-set text is still text in this picture — the model must
                # not propose the same thing for the region next to it.
                if overrides[i].strip():
                    used_texts.setdefault(f"manual{i}", overrides[i].strip())
                report.append(f"  #{i + 1} manual override: {overrides[i]!r}")
                continue

            if skip_legible and region.get("slop", {}).get("verdict") == "clean":
                region["proposal"] = {
                    "text": region.get("slop", {}).get("ocr_text", ""), "style": "", "font_hint": "",
                    "legible_original": 1.0, "confidence": 1.0,
                    "ok": True, "error": None, "source": "kept",
                }
                kept_text = region.get("slop", {}).get("ocr_text", "").strip()
                if kept_text:
                    used_texts.setdefault(f"kept{i}", kept_text)
                report.append(f"  #{i + 1} already legible — kept")
                continue

            cid = region.get("cluster_id", -1)
            if one_call_per_cluster and cid >= 0 and cid in cluster_cache:
                inheritedProposal = dict(cluster_cache[cid])
                inheritedProposal["source"] = "cluster"
                region["proposal"] = inheritedProposal
                inherited += 1
                report.append(f"  #{i + 1} inherits cluster {cid}: {inheritedProposal['text']!r}"
                              + (f" [{inheritedProposal['style']}]"
                                 if inheritedProposal.get("style", "").strip() else ""))
                continue

            proposal = None
            own_key = cid if cid >= 0 else f"solo{i}"
            if use_model:
                scene = scene_rgb if context_mode != "crop_only" else None
                neighbors = None
                if context_mode == "crop+scene+neighbors":
                    neighbors = [n["crop"] for n in self._neighbors(regions, region)]

                # Everything already used in this picture, except this region's
                # own cluster — siblings are meant to match, strangers are not.
                avoid = ([t for k, t in used_texts.items() if k != own_key]
                         if avoid_repeats else None)

                base_instruction = class_instr.get(
                    cls_name, get_class(cls_name)["vlm_instruction"])
                others = [t for k, t in used_texts.items() if k != own_key]

                # Ask again when the answer is a re-run of one already handed
                # out. The ban list alone does not always land: the model happily
                # returns the same subject with a different price.
                attempts = 1 + (max(0, variety_retries) if avoid_repeats and others else 0)
                for attempt in range(attempts):
                    extra = ""
                    if attempt:
                        extra = (" You already suggested something too close to text elsewhere "
                                 "in this picture. Change the SUBJECT completely - a different "
                                 "kind of notice about a different thing, not the same message "
                                 "reworded or repriced.")
                    proposal = propose_text(
                        avoid_texts=avoid,
                        crop_rgb=region.get("crop"),
                        scene_rgb=scene,
                        neighbor_crops=neighbors,
                        class_name=cls_name,
                        scene_hint=scene_hint,
                        language=language,
                        base_url=base_url,
                        model_id=model_id,
                        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
                        class_instruction=base_instruction + extra,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed + i + attempt * 7919,
                        timeout=timeout,
                    )
                    candidate = (proposal.get("text") or "").strip()
                    if not candidate or not others:
                        break
                    if not is_too_similar(candidate, others):
                        break
                    if attempt < attempts - 1:
                        retried += 1
                        report.append(f"  #{i + 1} {candidate!r} repeats earlier text — asking again")
                    else:
                        duplicates += 1
                        report.append(f"  #{i + 1} still repetitive after {attempts} tries: {candidate!r}")
                if proposal.get("ok") and proposal.get("text", "").strip():
                    made += 1
                    report.append(f"  #{i + 1} {cls_name}: {proposal['text']!r} "
                                  f"(legible_original={proposal.get('legible_original', 0):.2f})")
                    # style carries the surface description into the diffusion
                    # prompt, so it needs to be visible when a result looks off.
                    if proposal.get("style", "").strip():
                        report.append(f"        style: {proposal['style']}")
                    if proposal.get("font_hint", "").strip():
                        report.append(f"        font:  {proposal['font_hint']}")
                else:
                    failed += 1
                    report.append(f"  #{i + 1} {cls_name}: model gave nothing ({proposal.get('error')})")
                    proposal = None

            if proposal is None:
                text = fb_keyed.get(cls_name, "")
                if not text and fb_plain:
                    text = fb_plain[fb_cursor % len(fb_plain)]
                    fb_cursor += 1
                if not text:
                    text = region.get("slop", {}).get("ocr_text", "").strip()
                proposal = {
                    "text": text, "style": "", "font_hint": "",
                    "legible_original": 0.0, "confidence": 0.0,
                    "ok": bool(text), "error": None,
                    "source": "fallback" if text else "empty",
                }
                if text:
                    report.append(f"  #{i + 1} fallback: {text!r}")

            region["proposal"] = proposal
            # Only seed the cluster cache with a usable text. Caching an empty
            # proposal would make every sibling inherit the failure instead of
            # getting its own attempt.
            if proposal.get("text", "").strip():
                if cid >= 0:
                    cluster_cache.setdefault(cid, proposal)
                used_texts.setdefault(own_key, proposal["text"].strip())

        summary = f"Summary: {made} from the model, {inherited} inherited, {failed} failed"
        if retried:
            summary += f", {retried} re-asked for variety"
        if duplicates:
            summary += f", {duplicates} still repetitive"
        report.append(summary)

        # Tab-separated so it stays readable in a text preview and still splits
        # cleanly if anyone parses it. style is included because it is what
        # carries the surface ("black ink on yellow sticky note") into the
        # diffusion prompt — without it an odd render is hard to explain.
        lines = ["#\tclass\tsource\ttext\tstyle\tfont_hint"] if regions else []
        for i, r in enumerate(regions):
            p = r.get("proposal") or {}
            lines.append("\t".join([
                str(i + 1),
                str(r.get("class", "?")),
                str(p.get("source", "-")),
                str(p.get("text", "")),
                str(p.get("style", "")),
                str(p.get("font_hint", "")),
            ]))

        return (sign_data, "\n".join(lines), "\n".join(report))
