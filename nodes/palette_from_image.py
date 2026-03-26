"""ComfyUI node: extract a color palette from an input image."""

try:
    from ..core.image_extraction import extract_palette_from_image
    from ..core.preview import render_palette_preview, render_source_annotated
except ImportError:
    from core.image_extraction import extract_palette_from_image
    from core.preview import render_palette_preview, render_source_annotated


class FVM_PaletteFromImage:
    """Extracts a color palette from an image using K-Means clustering.

    Supports multiple extraction modes (dominant, vibrant, fashion-aware),
    region selection, and optional skin-tone / background filtering.
    """

    CATEGORY = "FVM Tools/Color"
    FUNCTION = "extract"
    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "IMAGE", "IMAGE", "STRING",
    )
    RETURN_NAMES = (
        "palette_string",
        "color_1", "color_2", "color_3", "color_4",
        "color_5", "color_6", "color_7", "color_8",
        "primary", "secondary", "accent", "neutral", "metallic",
        "palette_preview", "source_annotated", "palette_info",
    )
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Extracts a color palette from an image using K-Means clustering.\n\n"
        "Modes:\n"
        "  dominant — sort by pixel count (most common colors)\n"
        "  vibrant — sort by saturation (most vivid colors)\n"
        "  fashion_aware — greedy max-hue-distance selection\n\n"
        "Outputs individual color names, semantic role outputs, "
        "a palette preview swatch, and an annotated source image."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                }),
                "extraction_mode": (["dominant", "vibrant", "fashion_aware"],),
                "ignore_background": ("BOOLEAN", {"default": True}),
                "ignore_skin": ("BOOLEAN", {"default": True}),
                "sample_region": (["full", "center_crop", "upper_half", "lower_half"],),
                "saturation_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "include_neutrals": ("BOOLEAN", {"default": True}),
                "include_metallics": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                }),
            },
        }

    def extract(self, image, num_colors, extraction_mode, ignore_background,
                ignore_skin, sample_region, saturation_threshold,
                include_neutrals, include_metallics, seed):
        # Run extraction
        result = extract_palette_from_image(
            image_tensor=image,
            num_colors=num_colors,
            mode=extraction_mode,
            region=sample_region,
            filter_skin=ignore_skin,
            filter_background=ignore_background,
            saturation_threshold=saturation_threshold,
            include_neutrals=include_neutrals,
            include_metallics=include_metallics,
            seed=seed,
        )

        colors = result["colors"]
        palette_string = result["palette_string"]
        info = result["info"]

        # Individual color slots (1-8), empty string if not enough colors
        color_slots = []
        for i in range(8):
            if i < len(colors):
                color_slots.append(colors[i]["name"])
            else:
                color_slots.append("")

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

        # Preview images
        palette_preview = render_palette_preview(colors)
        source_annotated = render_source_annotated(image, colors)

        return (
            palette_string,
            color_slots[0], color_slots[1], color_slots[2], color_slots[3],
            color_slots[4], color_slots[5], color_slots[6], color_slots[7],
            primary, secondary, accent, neutral, metallic,
            palette_preview, source_annotated, info,
        )
