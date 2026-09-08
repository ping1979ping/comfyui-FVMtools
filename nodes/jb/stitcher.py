"""FVM_JB_Stitcher — wrap N JSON fragments under a common title.

Behavior per the user's brief:

  - One ``title`` string field (the top-level key).
  - Dynamic optional STRING inputs (``input_1``, ``input_2``, ...). The JS
    side auto-spawns the next slot when the previous one is connected.
  - Each connected input is parsed as JSON; objects merge into the
    title's child object via deep-merge (last-wins on scalar collision,
    but new sub-fields are added recursively); arrays append under
    ``inputs``; bare strings land under their slot name (``input_3``).
  - Reuses ``core/smp/merge.deep_merge`` — same semantics the SMP
    Aggregator already uses.

The slot keys are deliberately *not* underscore-prefixed: ``natural`` and
``sentences`` drop every ``_``-key as generation metadata, and a prose
fragment parked under ``__input3`` vanished from the output entirely.

Outputs ``raw_json`` (strict) and ``string`` (chosen format).
"""

from __future__ import annotations

import copy

try:
    from ...core.jb.serialize import (
        ALL_FORMATS,
        SENTENCES,
        emit,
        emit_strict_json,
        parse_input,
    )
    from ...core.jb.sentences import emit_sentences
    from ...core.smp.merge import deep_merge
except ImportError:  # pragma: no cover
    from core.jb.serialize import (
        ALL_FORMATS,
        SENTENCES,
        emit,
        emit_strict_json,
        parse_input,
    )
    from core.jb.sentences import emit_sentences
    from core.smp.merge import deep_merge


# Maximum number of dynamic optional input slots we declare on the Python
# side. The JS layer only shows / connects as many as the user needs.
MAX_INPUTS = 24

# Bucket for array inputs. Public (no underscore) so the natural / sentences
# formats keep the content; see module docstring.
ARRAY_KEY = "inputs"


class FVM_JB_Stitcher:
    """Wraps N stringy JSON fragments under one top-level title.

    Two outputs:
      raw_json — strict JSON, e.g. {"character_1": {"hosiery": {...}}}
      string   — same payload in the chosen output_format
    """

    CATEGORY = "FVM Tools/JB"
    FUNCTION = "stitch"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("raw_json", "string")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Wraps multiple JSON fragments under a single top-level title.\n\n"
        "Each input is parsed as JSON; objects deep-merge into the title's\n"
        "child object (same-level scalar leaves: last input wins; new\n"
        "sub-fields underneath are added recursively). Arrays append\n"
        "under 'inputs'. Bare strings keep their slot name (input_3).\n\n"
        "output_format 'sentences' writes prose and puts the title in\n"
        "front: 'character_1: The scene takes place ...'.\n\n"
        "Connect input_1 → the next input slot auto-spawns."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "title":         ("STRING", {"default": "character_1"}),
            "output_format": (list(ALL_FORMATS), {"default": "loose_keys",
                              "tooltip": "natural / sentences: prose for Krea 2 "
                              "and Qwen encoders (sentences = full sentences "
                              "with lead-ins, title prepended). loose_keys / "
                              "pretty_json / compact_json: structured, for "
                              "Ideogram 4 style JSON prompting."}),
        }
        optional = {f"input_{i}": ("STRING", {"forceInput": True})
                    for i in range(1, MAX_INPUTS + 1)}
        return {"required": required, "optional": optional}

    def stitch(self, title, output_format, **kwargs):
        merged: dict = {}

        # Collect inputs in declaration order (input_1, input_2, ...).
        inputs = []
        for i in range(1, MAX_INPUTS + 1):
            v = kwargs.get(f"input_{i}")
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            inputs.append((i, v))

        for idx, value in inputs:
            parsed = parse_input(value) if isinstance(value, str) else value
            if isinstance(parsed, dict):
                merged = deep_merge(merged, parsed)
            elif isinstance(parsed, list):
                # Append into the shared array bucket — keeps non-merging
                # array data accessible without clobbering a dict-merge case.
                # If a merged fragment already used that key for something
                # else, the array goes to its own slot key instead.
                bucket = merged.get(ARRAY_KEY)
                if bucket is None:
                    merged[ARRAY_KEY] = copy.deepcopy(parsed)
                elif isinstance(bucket, list):
                    bucket.extend(copy.deepcopy(parsed))
                else:
                    merged[f"input_{idx}"] = copy.deepcopy(parsed)
            else:
                # Scalar / non-JSON string — store under its slot name.
                merged[f"input_{idx}"] = parsed

        title_str = (title or "").strip() or "untitled"
        wrapped = {title_str: merged}

        if output_format == SENTENCES:
            # The title is a label (often a LoRA trigger), not a fact about
            # the image — it goes in front instead of becoming a sentence.
            string_out = emit_sentences(merged, prefix=f"{title_str}: ")
        else:
            string_out = emit(wrapped, output_format)
        return (emit_strict_json(wrapped, indent=2), string_out)
