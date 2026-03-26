"""ComfyUI node for generating seed-controlled outfit descriptions with color tags."""

try:
    from ..core.outfit_engine import generate_outfit
    from ..core.outfit_parser import parse_overrides
    from ..core.outfit_presets import OUTFIT_PRESETS
    from ..core.outfit_lists import get_available_sets
except ImportError:
    from core.outfit_engine import generate_outfit
    from core.outfit_parser import parse_overrides
    from core.outfit_presets import OUTFIT_PRESETS
    from core.outfit_lists import get_available_sets


class FVM_OutfitGenerator:
    """Generates seed-controlled outfit descriptions with color tags for fashion prompts."""

    CATEGORY = "FVM Tools/Fashion"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("outfit_prompt", "outfit_details", "outfit_info")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Generates seed-controlled outfit descriptions with color tags.\n\n"
        "Outputs a prompt string with #primary#, #secondary#, etc. color tags "
        "ready for Prompt Color Replace, a structured details string, and info."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outfit_set": (get_available_sets(),),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                }),
                "style_preset": (sorted(OUTFIT_PRESETS.keys()),),
                "formality": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "coverage": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "enable_headwear": ("BOOLEAN", {"default": False}),
                "enable_top": ("BOOLEAN", {"default": True}),
                "enable_outerwear": ("BOOLEAN", {"default": False}),
                "enable_bottom": ("BOOLEAN", {"default": True}),
                "enable_footwear": ("BOOLEAN", {"default": True}),
                "enable_accessories": ("BOOLEAN", {"default": False}),
                "enable_bag": ("BOOLEAN", {"default": False}),
                "print_probability": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Probability of adding prints, patterns, logos, or text to garments (0=never, 1=always)",
                }),
                "text_mode": (["auto", "quoted", "descriptive", "off"], {
                    "tooltip": "auto/quoted: exact text in quotes (ZImage/Flux2). descriptive: generic description (safe for SD/SDXL). off: no text, prints only.",
                }),
            },
            "optional": {
                "override_string": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "top: silk blouse | #primary#\nbottom: exclude",
                }),
                "prefix": ("STRING", {
                    "default": "wearing ",
                }),
                "separator": ("STRING", {
                    "default": ", ",
                }),
            },
        }

    def generate(self, outfit_set, seed, style_preset, formality, coverage,
                 enable_headwear, enable_top, enable_outerwear,
                 enable_bottom, enable_footwear, enable_accessories,
                 enable_bag, print_probability=0.3, text_mode="auto",
                 override_string="", prefix="wearing ", separator=", "):

        slot_enables = {
            "headwear": enable_headwear,
            "top": enable_top,
            "outerwear": enable_outerwear,
            "bottom": enable_bottom,
            "footwear": enable_footwear,
            "accessories": enable_accessories,
            "bag": enable_bag,
        }

        overrides = parse_overrides(override_string) if override_string.strip() else None

        result = generate_outfit(
            seed=seed,
            outfit_set=outfit_set,
            style_preset=style_preset,
            formality=formality,
            coverage=coverage,
            slot_enables=slot_enables,
            overrides=overrides,
            prefix=prefix,
            separator=separator,
            print_probability=print_probability,
            text_mode=text_mode,
        )

        return (result["outfit_prompt"], result["outfit_details"], result["outfit_info"])
