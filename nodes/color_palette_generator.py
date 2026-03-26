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
                }),
                "num_colors": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 8,
                }),
                "harmony_type": (cls._HARMONY_TYPES,),
                "style_preset": (sorted(STYLE_PRESETS.keys()),),
                "vibrancy": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "contrast": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "warmth": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "neutral_ratio": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "include_metallics": ("BOOLEAN", {
                    "default": True,
                }),
            },
            "optional": {
                "palette_source": (["generate", "from_file"],),
                "wildcard_file": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "One palette per line: red, blue, green, white, gold",
                }),
                "palette_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 9999,
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
