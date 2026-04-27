"""SMP LocationCombiner — resolves atmosphere tokens in LOCATION_DICT_RAW.

Substitutes ``#ambient_light#`` and ``#shadow_tone#`` placeholders in every
element's ``prompt_fragment`` using a ``COLOR_PALETTE_DICT``.
"""

import copy

try:
    from ...core.smp.defaults import ATMOSPHERE_TOKEN_MAP, GARMENT_TOKEN_MAP
except ImportError:  # pragma: no cover
    from core.smp.defaults import ATMOSPHERE_TOKEN_MAP, GARMENT_TOKEN_MAP


class FVM_SMP_LocationCombiner:
    """Resolve atmosphere tokens in a LOCATION_DICT_RAW using a palette."""

    CATEGORY = "FVM Tools/SMP/Combiners"
    FUNCTION = "combine"
    RETURN_TYPES = ("LOCATION_DICT", "STRING")
    RETURN_NAMES = ("location_dict", "summary")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Resolves #ambient_light# and #shadow_tone# tokens in every\n"
        "element's prompt_fragment using the supplied palette."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "location_raw":  ("LOCATION_DICT_RAW",),
                "color_palette": ("COLOR_PALETTE_DICT",),
            },
        }

    def combine(self, location_raw, color_palette):
        resolved = copy.deepcopy(location_raw or {})
        atmosphere_colors = (color_palette or {}).get("atmosphere_colors", {}) or {}
        garment_colors = (color_palette or {}).get("garment_colors", {}) or {}
        raw_tokens = (color_palette or {}).get("raw_tokens", {}) or {}

        # Build full substitution table — atmosphere tokens take priority but
        # also resolve any leftover garment tokens that snuck in (rare but
        # possible if a user customised an entry).
        subs: dict[str, str] = {}
        for token, key in ATMOSPHERE_TOKEN_MAP.items():
            if token in raw_tokens:
                subs[token] = raw_tokens[token]
            elif key in atmosphere_colors:
                subs[token] = atmosphere_colors[key]
        for token, role in GARMENT_TOKEN_MAP.items():
            if token in raw_tokens:
                subs[token] = raw_tokens[token]
            elif role in garment_colors:
                subs[token] = garment_colors[role]

        elements = resolved.get("elements", {}) or {}
        for elem_id, e in elements.items():
            fragment = e.get("prompt_fragment", "") or ""
            for token, value in subs.items():
                if token in fragment:
                    fragment = fragment.replace(token, value)
            e["prompt_fragment"] = fragment

        if not resolved.get("color_tone") and color_palette:
            resolved["color_tone"] = color_palette.get("color_tone")

        lines = [
            f"Combined location: set={resolved.get('set_name')}, "
            f"seed={resolved.get('seed')}, tone={resolved.get('color_tone')}"
        ]
        for elem_id, e in elements.items():
            lines.append(f"  {elem_id}: {e.get('prompt_fragment')}")
        return (resolved, "\n".join(lines))
