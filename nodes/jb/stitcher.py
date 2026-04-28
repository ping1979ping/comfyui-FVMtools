"""FVM_JB_Stitcher — wrap N JSON fragments under a common title.

Behavior per the user's brief:

  - One ``title`` string field (the top-level key).
  - Dynamic optional STRING inputs (``input_1``, ``input_2``, ...). The JS
    side auto-spawns the next slot when the previous one is connected.
  - Each connected input is parsed as JSON; objects merge into the
    title's child object via deep-merge (last-wins on scalar collision,
    but new sub-fields are added recursively); arrays append; bare
    strings get a synthetic key ``__inputN``.
  - Reuses ``core/smp/merge.deep_merge`` — same semantics the SMP
    Aggregator already uses.

Outputs ``raw_json`` (strict) and ``string`` (chosen format).
"""

from __future__ import annotations

import copy

try:
    from ...core.jb.serialize import (
        ALL_FORMATS,
        emit,
        emit_strict_json,
        parse_input,
    )
    from ...core.smp.merge import deep_merge
except ImportError:  # pragma: no cover
    from core.jb.serialize import (
        ALL_FORMATS,
        emit,
        emit_strict_json,
        parse_input,
    )
    from core.smp.merge import deep_merge


# Maximum number of dynamic optional input slots we declare on the Python
# side. The JS layer only shows / connects as many as the user needs.
MAX_INPUTS = 24


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
        "sub-fields underneath are added recursively). Arrays append.\n"
        "Bare strings get a synthetic '__inputN' key.\n\n"
        "Connect input_1 → the next input slot auto-spawns."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "title":         ("STRING", {"default": "character_1"}),
            "output_format": (list(ALL_FORMATS), {"default": "loose_keys"}),
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
                # Append into a synthetic '__inputs' array — keeps non-merging
                # array data accessible without clobbering a dict-merge case.
                merged.setdefault("__inputs", [])
                merged["__inputs"].extend(copy.deepcopy(parsed))
            else:
                # Scalar / non-JSON string — store under a synthetic key.
                merged[f"__input{idx}"] = parsed

        title_str = (title or "").strip() or "untitled"
        wrapped = {title_str: merged}

        return (emit_strict_json(wrapped, indent=2),
                emit(wrapped, output_format))
