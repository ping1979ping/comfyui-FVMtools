"""SMP OutfitCombiner — resolves color tokens in OUTFIT_DICT_RAW → OUTFIT_DICT.

Substitutes ``#primary#`` / ``#secondary#`` / ``#accent#`` / ``#neutral#`` /
``#metallic#`` / ``#tertiary#`` placeholders inside every garment's
``prompt_fragment`` using the provided ``COLOR_PALETTE_DICT``.

Sets ``color_resolved`` on each garment based on its ``color_role``.
"""

import copy

try:
    from ...core.smp.defaults import GARMENT_TOKEN_MAP
except ImportError:  # pragma: no cover
    from core.smp.defaults import GARMENT_TOKEN_MAP


class FVM_SMP_OutfitCombiner:
    """Resolve color tokens in an OUTFIT_DICT_RAW using a COLOR_PALETTE_DICT."""

    CATEGORY = "FVM Tools/SMP/Combiners"
    FUNCTION = "combine"
    RETURN_TYPES = ("OUTFIT_DICT", "STRING")
    RETURN_NAMES = ("outfit_dict", "summary")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Resolves #primary#/#secondary#/#accent#/#neutral#/#metallic# tokens\n"
        "in every garment's prompt_fragment using the supplied palette.\n"
        "Sets color_resolved per garment based on its color_role."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outfit_raw":    ("OUTFIT_DICT_RAW",),
                "color_palette": ("COLOR_PALETTE_DICT",),
            },
        }

    def combine(self, outfit_raw, color_palette):
        resolved = copy.deepcopy(outfit_raw or {})
        garment_colors = (color_palette or {}).get("garment_colors", {}) or {}
        raw_tokens = (color_palette or {}).get("raw_tokens", {}) or {}

        # Build a substitution table: token → color name. Fall back to
        # role-derived tokens if the palette didn't pre-build raw_tokens.
        subs: dict[str, str] = {}
        for token, role in GARMENT_TOKEN_MAP.items():
            if token in raw_tokens:
                subs[token] = raw_tokens[token]
            elif role in garment_colors:
                subs[token] = garment_colors[role]

        garments = resolved.get("garments", {}) or {}
        for region_id, garment in garments.items():
            fragment = garment.get("prompt_fragment", "") or ""
            for token, value in subs.items():
                if token in fragment:
                    fragment = fragment.replace(token, value)
            garment["prompt_fragment"] = fragment

            role = garment.get("color_role")
            if role and role in garment_colors:
                garment["color_resolved"] = garment_colors[role]

        # Inherit tone label from palette if outfit didn't carry one.
        if not resolved.get("color_tone") and color_palette:
            resolved["color_tone"] = color_palette.get("color_tone")

        # Build a human-readable summary.
        lines = [
            f"Combined outfit: set={resolved.get('set_name')}, "
            f"seed={resolved.get('seed')}, tone={resolved.get('color_tone')}"
        ]
        for rid, g in garments.items():
            lines.append(f"  {rid} [{g.get('color_role')}={g.get('color_resolved')}]: "
                         f"{g.get('prompt_fragment')}")
        return (resolved, "\n".join(lines))
