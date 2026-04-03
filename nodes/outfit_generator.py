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
                "outfit_set": (get_available_sets(), {
                    "tooltip": "Outfit theme/collection to generate from.\n\n"
                               "Each set is a folder in outfit_lists/ with .txt files\n"
                               "defining garments per slot (top, bottom, footwear, etc.).\n"
                               "Use the Edit List button below to customize items.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for deterministic outfit generation.\n"
                               "Same seed + same settings = same outfit every time.\n"
                               "Change seed to get a different outfit combination.",
                }),
                "style_preset": (sorted(OUTFIT_PRESETS.keys()), {
                    "tooltip": "Style profile that influences garment selection.\n\n"
                               "Each preset defines a formality range, preferred fabric families,\n"
                               "and slot probabilities. E.g. 'formal' favors silk/wool and\n"
                               "always includes outerwear, 'casual' prefers cotton/jersey.",
                }),
                "formality": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Casual-to-formal spectrum for garment filtering.\n\n"
                               "Each garment in the list files has a formality range (e.g. 0.0-0.3).\n"
                               "Only garments whose range contains this value are eligible.\n\n"
                               "0.0 = very casual (tank tops, shorts, sandals)\n"
                               "0.3 = relaxed (jeans, sneakers)\n"
                               "0.5 = smart casual (default)\n"
                               "0.7 = semi-formal (blazers, dress shoes)\n"
                               "1.0 = very formal (gowns, tuxedos)\n\n"
                               "Tip: Match to your outfit set. Yoga sets use 0.0-0.2,\n"
                               "business sets work best at 0.5-0.8.",
                }),
                "coverage": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Controls how many optional slots get activated.\n\n"
                               "Affects headwear, outerwear, accessories, bags — slots that\n"
                               "aren't always worn. Top/bottom/footwear are always active.\n\n"
                               "0.0 = minimal (just top + bottom + shoes)\n"
                               "0.5 = moderate (some accessories, maybe outerwear)\n"
                               "1.0 = maximal (all enabled slots very likely to appear)",
                }),
                "enable_headwear": ("BOOLEAN", {"default": False,
                    "tooltip": "Allow hats, caps, headbands in the outfit.\nSubject to probability roll based on coverage."}),
                "enable_top": ("BOOLEAN", {"default": True,
                    "tooltip": "Include a top garment (shirt, blouse, etc.).\nAlways active when enabled — not subject to probability."}),
                "enable_outerwear": ("BOOLEAN", {"default": False,
                    "tooltip": "Allow jackets, coats, vests over the top.\nSubject to probability roll based on coverage."}),
                "enable_bottom": ("BOOLEAN", {"default": True,
                    "tooltip": "Include a bottom garment (pants, skirt, etc.).\nAlways active when enabled — not subject to probability."}),
                "enable_footwear": ("BOOLEAN", {"default": True,
                    "tooltip": "Include shoes/boots.\nAlways active when enabled — not subject to probability."}),
                "enable_accessories": ("BOOLEAN", {"default": False,
                    "tooltip": "Allow jewelry, watches, scarves, belts.\nSubject to probability roll based on coverage."}),
                "enable_bag": ("BOOLEAN", {"default": False,
                    "tooltip": "Allow bags, purses, backpacks.\nSubject to probability roll based on coverage."}),
                "print_probability": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Chance of adding a decorative pattern or text to each garment.\n\n"
                               "0.0 = never add prints/text (solid colors only)\n"
                               "0.3 = occasional prints (default)\n"
                               "1.0 = every garment gets a print or text decoration\n\n"
                               "Prints come from prints.txt, text from texts.txt in the outfit set.",
                }),
                "text_mode": (["auto", "quoted", "descriptive", "off"], {
                    "tooltip": "How text decorations appear in prompts.\n\n"
                               "auto/quoted — exact text in quotes, e.g. '\"REBEL\" text in gothic font'\n"
                               "  Best for ZImage Turbo, Flux2, and other text-aware models.\n\n"
                               "descriptive — generic description, e.g. 'bold text graphic'\n"
                               "  Safe fallback for SD 1.5 / SDXL that can't render text.\n\n"
                               "off — no text decorations, only visual prints/patterns.",
                }),
            },
            "optional": {
                "override_string": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "top: silk blouse | #primary#\nbottom: exclude",
                    "tooltip": "Per-slot overrides. One line per slot.\n\n"
                               "Format: slot_name: garment [| fabric] [| #color_tag#] [| decoration]\n\n"
                               "Examples:\n"
                               "  top: silk blouse | #primary#\n"
                               "  bottom: exclude              (skip this slot)\n"
                               "  footwear: red stiletto heels | leather | #accent#\n"
                               "  accessories: auto            (use normal generation)\n\n"
                               "Overrides bypass formality filtering and probability rolls.",
                }),
                "prefix": ("STRING", {
                    "default": "wearing ",
                    "tooltip": "Text prepended to the outfit prompt.\n"
                               "Default: 'wearing ' — produces 'wearing red silk blouse, ...'",
                }),
                "separator": ("STRING", {
                    "default": ", ",
                    "tooltip": "Text between garment descriptions.\n"
                               "Default: ', ' — produces 'blouse, pants, boots'",
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
