"""ComfyUI node for generating named color palettes."""

try:
    from ..core.palette_engine import generate_palette
    from ..core.style_presets import STYLE_PRESETS
    from ..core.preview import render_palette_preview
except ImportError:
    from core.palette_engine import generate_palette
    from core.style_presets import STYLE_PRESETS
    from core.preview import render_palette_preview


class FVM_ColorPaletteGenerator:
    """Generates a harmonious named color palette for fashion/style prompts."""

    CATEGORY = "FVM Tools/Color"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING", "STRING",
                    "IMAGE", "STRING")
    RETURN_NAMES = ("palette_string", "color_1", "color_2", "color_3", "color_4",
                    "color_5", "color_6", "color_7", "color_8",
                    "primary", "secondary", "accent", "neutral", "metallic",
                    "palette_preview", "palette_info")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Generates a harmonious named color palette.\n\n"
        "Outputs individual color names, semantic role outputs, a preview image, "
        "and a comma-separated palette_string for Prompt Color Replace."
    )

    _HARMONY_TYPES = ["auto", "analogous", "complementary", "split_complementary",
                      "triadic", "tetradic", "monochromatic"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for deterministic palette generation.\n"
                               "Same seed + same settings = same palette every time.",
                }),
                "num_colors": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 8,
                    "tooltip": "Number of colors to generate (2-8).\n\n"
                               "5 is recommended — maps to the semantic roles:\n"
                               "primary, secondary, accent, neutral, metallic.\n"
                               "Extra colors are available as color_6, color_7, color_8.",
                }),
                "harmony_type": (cls._HARMONY_TYPES, {
                    "tooltip": "Color harmony algorithm for hue selection.\n\n"
                               "auto — picks a harmony type based on seed\n"
                               "analogous — neighboring hues (calm, cohesive)\n"
                               "complementary — opposite hues (bold contrast)\n"
                               "split_complementary — one hue + two adjacent opposites\n"
                               "triadic — three evenly spaced hues (vibrant)\n"
                               "tetradic — four hues in two complementary pairs\n"
                               "monochromatic — single hue, varied lightness/saturation",
                }),
                "style_preset": (sorted(STYLE_PRESETS.keys()), {
                    "tooltip": "Visual style that influences color selection.\n\n"
                               "Each preset defines preferred hue ranges, saturation,\n"
                               "and lightness profiles. E.g. 'pastel' favors light desaturated\n"
                               "colors, 'gothic' favors dark reds/purples/blacks.",
                }),
                "vibrancy": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Color saturation intensity.\n\n"
                               "0.0 = muted, desaturated (grays, pastels)\n"
                               "0.5 = balanced (default)\n"
                               "1.0 = vivid, highly saturated colors",
                }),
                "contrast": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Lightness spread between colors.\n\n"
                               "0.0 = all colors at similar brightness\n"
                               "0.5 = moderate light/dark variation (default)\n"
                               "1.0 = maximum spread (very light + very dark colors)",
                }),
                "warmth": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Color temperature bias.\n\n"
                               "0.0 = cool (blues, greens, purples)\n"
                               "0.5 = neutral (default)\n"
                               "1.0 = warm (reds, oranges, yellows)",
                }),
                "neutral_ratio": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Proportion of neutral/muted colors in the palette.\n\n"
                               "0.0 = all colors are chromatic (vibrant)\n"
                               "0.4 = ~40% neutrals like gray, beige, cream (default)\n"
                               "1.0 = mostly neutrals (subdued palette)",
                }),
                "include_metallics": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include metallic colors (gold, silver, bronze, rose-gold, etc.).\n\n"
                               "When enabled, one palette slot is reserved for a metallic color\n"
                               "and output as the 'metallic' role. Useful for jewelry/accessories.",
                }),
            },
            "optional": {
                "palette_source": (["generate", "from_file"], {
                    "tooltip": "generate — create palette from seed + settings (default)\n"
                               "from_file — pick a palette from the wildcard_file text below",
                }),
                "wildcard_file": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "One palette per line: red, blue, green, white, gold",
                    "tooltip": "Manual palette list (one per line, comma-separated color names).\n\n"
                               "Only used when palette_source = 'from_file'.\n"
                               "Example:\n"
                               "  navy-blue, soft-pink, charcoal, gold, cream\n"
                               "  forest-green, burgundy, tan, silver, ivory\n\n"
                               "Use palette_index to pick a specific line, or -1 for random by seed.",
                }),
                "palette_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 9999,
                    "tooltip": "Which line to pick from wildcard_file.\n\n"
                               "-1 = random line based on seed (default)\n"
                               "0+ = specific line number (wraps around if too large)",
                }),
            },
        }

    def generate(self, seed, num_colors, harmony_type, style_preset,
                 vibrancy, contrast, warmth, neutral_ratio, include_metallics,
                 palette_source="generate", wildcard_file="", palette_index=-1):

        # From-file mode: parse wildcard_file
        if palette_source == "from_file" and wildcard_file.strip():
            return self._from_file(wildcard_file, palette_index, seed, num_colors)

        # Generate mode
        result = generate_palette(
            seed=seed,
            num_colors=num_colors,
            harmony_type=harmony_type,
            style_preset=style_preset,
            vibrancy=vibrancy,
            contrast=contrast,
            warmth=warmth,
            neutral_ratio=neutral_ratio,
            include_metallics=include_metallics,
        )

        return self._pack_output(result)

    def _from_file(self, wildcard_file, palette_index, seed, num_colors):
        """Parse palettes from text, pick one by index or seed."""
        try:
            from ..core.color_database import COLOR_DATABASE
            from ..core.color_utils import hsl_to_rgb
            from ..core.role_assignment import assign_roles
        except ImportError:
            from core.color_database import COLOR_DATABASE
            from core.color_utils import hsl_to_rgb
            from core.role_assignment import assign_roles
        import random

        lines = [l.strip() for l in wildcard_file.strip().splitlines() if l.strip()]
        if not lines:
            # Fallback to generate
            result = generate_palette(seed=seed, num_colors=num_colors)
            return self._pack_output(result)

        if palette_index >= 0:
            idx = palette_index % len(lines)
        else:
            rng = random.Random(seed)
            idx = rng.randint(0, len(lines) - 1)

        names = [n.strip() for n in lines[idx].split(",") if n.strip()]
        colors = []
        for name in names[:8]:
            if name in COLOR_DATABASE:
                hsl = COLOR_DATABASE[name]
                rgb = hsl_to_rgb(*hsl)
            else:
                hsl = (0, 0, 0)
                rgb = (0, 0, 0)
            colors.append({"name": name, "hsl": hsl, "rgb": rgb, "role": None})

        assign_roles(colors)
        palette_string = ", ".join(c["name"] for c in colors)
        info = f"Source: file (line {idx + 1}/{len(lines)})\nColors: {len(colors)}"

        return self._pack_output({
            "colors": colors,
            "palette_string": palette_string,
            "info": info,
        })

    def _pack_output(self, result):
        """Pack result dict into the output tuple."""
        colors = result["colors"]
        palette_string = result["palette_string"]
        info = result["info"]

        # Individual color names (pad to 8)
        color_names = [c["name"] for c in colors]
        while len(color_names) < 8:
            color_names.append("")

        # Semantic role outputs
        role_map = {}
        for c in colors:
            if c.get("role"):
                role_map[c["role"]] = c["name"]

        primary = role_map.get("primary", "")
        secondary = role_map.get("secondary", "")
        accent = role_map.get("accent", "")
        neutral = role_map.get("neutral", "")
        metallic = role_map.get("metallic", "")

        # Preview image
        preview = render_palette_preview(colors)

        return (
            palette_string,
            color_names[0], color_names[1], color_names[2], color_names[3],
            color_names[4], color_names[5], color_names[6], color_names[7],
            primary, secondary, accent, neutral, metallic,
            preview,
            info,
        )
