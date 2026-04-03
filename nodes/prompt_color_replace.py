import re


class FVM_PromptColorReplace:
    """Replaces color tags (#color1#, #primary#, etc.) in prompts with palette colors.
    Supports numbered tags (#color1#-#color8#, #c1#-#c8#), semantic tags
    (#primary#, #secondary#, #accent#, #neutral#, #metallic#) and short aliases
    (#pri#, #sec#, #acc#, #neu#, #met#)."""

    CATEGORY = "FVM Tools/Color"
    FUNCTION = "replace"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "replacements_log")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Replaces color placeholder tags in a prompt with actual color names from a palette.\n\n"
        "Supported tags:\n"
        "  Numbered: #color1# - #color8# (or #c1# - #c8#)\n"
        "  Semantic: #primary# #secondary# #accent# #neutral# #metallic#\n"
        "  Short:    #pri# #sec# #acc# #neu# #met#\n\n"
        "Connect palette_string from Color Palette Generator or type manually.\n"
        "Semantic override inputs take precedence over palette positions."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "wearing #primary# miniskirt with #neutral# top, #metallic# earrings",
                    "tooltip": "Text with color placeholder tags to replace.\n\n"
                               "Supported tags:\n"
                               "  #color1# to #color8# (or #c1# to #c8#) — by palette position\n"
                               "  #primary# #secondary# #accent# #neutral# #metallic# — by role\n"
                               "  #pri# #sec# #acc# #neu# #met# — short aliases\n\n"
                               "Connect the outfit_prompt from Outfit Generator, or write manually.\n"
                               "Tags are case-insensitive.",
                }),
                "palette_string": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "navy-blue, soft-pink, charcoal-gray, gold, cream",
                    "forceInput": True,
                    "tooltip": "Comma-separated color names from Color Palette Generator.\n\n"
                               "Position mapping:\n"
                               "  1st color = #color1# = #primary#\n"
                               "  2nd color = #color2# = #secondary#\n"
                               "  3rd color = #color3# = #accent#\n"
                               "  4th color = #color4# = #neutral#\n"
                               "  5th color = #color5# = #metallic#\n"
                               "  6th-8th   = #color6# to #color8#\n\n"
                               "Connect palette_string output from Color Palette Generator.",
                }),
            },
            "optional": {
                "primary": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Override the primary role color.\nTakes precedence over the palette_string position.\nConnect from Color Palette Generator's 'primary' output."}),
                "secondary": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Override the secondary role color.\nTakes precedence over the palette_string position.\nConnect from Color Palette Generator's 'secondary' output."}),
                "accent": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Override the accent role color.\nTakes precedence over the palette_string position.\nConnect from Color Palette Generator's 'accent' output."}),
                "neutral": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Override the neutral role color.\nTakes precedence over the palette_string position.\nConnect from Color Palette Generator's 'neutral' output."}),
                "metallic": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Override the metallic role color.\nTakes precedence over the palette_string position.\nConnect from Color Palette Generator's 'metallic' output."}),
                "fallback_color": ("STRING", {"default": "black",
                    "tooltip": "Color used when a tag has no matching palette color.\nDefault: 'black'. Used if palette has fewer colors than tags referenced."}),
                "strip_hyphens": ("BOOLEAN", {"default": True,
                    "tooltip": "Convert hyphenated colors to spaces.\n\n"
                               "ON: navy-blue → navy blue (better for SD/SDXL prompts)\n"
                               "OFF: keep navy-blue as-is (useful if your model expects hyphens)"}),
            },
        }

    def replace(self, prompt, palette_string, primary="", secondary="",
                accent="", neutral="", metallic="", fallback_color="black",
                strip_hyphens=True):
        replacements = _build_replacement_map(
            palette_string, primary, secondary, accent, neutral, metallic,
            fallback_color,
        )
        result, log = _replace_tags(prompt, replacements, strip_hyphens, fallback_color)
        return (result, log)


# ── Internal helpers ──

_TAG_PATTERN = re.compile(
    r'#(color[1-8]|c[1-8]|primary|secondary|accent|neutral|metallic|pri|sec|acc|neu|met)#',
    re.IGNORECASE,
)

_SEMANTIC_ALIASES = {
    "#pri#": "#primary#",
    "#sec#": "#secondary#",
    "#acc#": "#accent#",
    "#neu#": "#neutral#",
    "#met#": "#metallic#",
}

_SEMANTIC_TO_INDEX = {
    "#primary#": 0,
    "#secondary#": 1,
    "#accent#": 2,
    "#neutral#": 3,
    "#metallic#": 4,
}


def _build_replacement_map(palette_string, primary_override, secondary_override,
                           accent_override, neutral_override, metallic_override,
                           fallback):
    """Build a dict mapping every recognized tag to its replacement color name."""
    colors = [c.strip() for c in palette_string.split(",") if c.strip()]

    replacements = {}

    # Numbered tags → palette positions
    for i in range(8):
        color = colors[i] if i < len(colors) else fallback
        replacements[f"#color{i+1}#"] = color
        replacements[f"#c{i+1}#"] = color

    # Semantic tags → palette positions (default mapping)
    for tag, idx in _SEMANTIC_TO_INDEX.items():
        replacements[tag] = colors[idx] if idx < len(colors) else fallback

    # Short aliases
    for alias, canonical in _SEMANTIC_ALIASES.items():
        replacements[alias] = replacements[canonical]

    # Apply overrides
    overrides = {
        "#primary#": primary_override,
        "#secondary#": secondary_override,
        "#accent#": accent_override,
        "#neutral#": neutral_override,
        "#metallic#": metallic_override,
    }
    for tag, override in overrides.items():
        if override:
            replacements[tag] = override
            # Also update short alias
            for alias, canonical in _SEMANTIC_ALIASES.items():
                if canonical == tag:
                    replacements[alias] = override

    return replacements


def _replace_tags(prompt, replacements, strip_hyphens, fallback):
    """Replace all color tags in prompt. Returns (result_string, log_string)."""
    if not prompt:
        return ("", "No tags found in prompt")

    log_entries = []

    def _replace_match(match):
        tag = match.group(0).lower()
        # Resolve short aliases to canonical form for lookup
        canonical = _SEMANTIC_ALIASES.get(tag, tag)
        replacement = replacements.get(canonical, replacements.get(tag, fallback))
        if strip_hyphens:
            replacement = replacement.replace("-", " ")
        log_entries.append(f"{match.group(0)} -> {replacement}")
        return replacement

    result = _TAG_PATTERN.sub(_replace_match, prompt)
    log = ", ".join(log_entries) if log_entries else "No tags found in prompt"
    return (result, log)
