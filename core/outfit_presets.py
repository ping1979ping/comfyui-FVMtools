"""Style presets for outfit generation. Maps style preset names to outfit-specific configs.

Uses the SAME preset keys as STYLE_PRESETS in core/style_presets.py.
"""

OUTFIT_PRESETS = {
    "general": {
        "slot_probabilities": {
            "headwear": 0.15,
            "outerwear": 0.25,
            "accessories": 0.50,
            "bag": 0.20,
        },
        "formality_range": (0.0, 1.0),
        "preferred_fabric_families": None,  # all allowed
        "garment_weights": {},
    },
    "beach": {
        "slot_probabilities": {
            "headwear": 0.60,
            "outerwear": 0.05,
            "accessories": 0.70,
            "bag": 0.30,
        },
        "formality_range": (0.0, 0.25),
        "preferred_fabric_families": ["natural", "casual", "sporty"],
        "garment_weights": {},
    },
    "urban_streetwear": {
        "slot_probabilities": {
            "headwear": 0.40,
            "outerwear": 0.60,
            "accessories": 0.65,
            "bag": 0.35,
        },
        "formality_range": (0.0, 0.4),
        "preferred_fabric_families": ["tough", "casual", "sporty"],
        "garment_weights": {},
    },
    "evening_gala": {
        "slot_probabilities": {
            "headwear": 0.05,
            "outerwear": 0.15,
            "accessories": 0.90,
            "bag": 0.60,
        },
        "formality_range": (0.7, 1.0),
        "preferred_fabric_families": ["luxury"],
        "garment_weights": {},
    },
    "casual_daywear": {
        "slot_probabilities": {
            "headwear": 0.20,
            "outerwear": 0.20,
            "accessories": 0.40,
            "bag": 0.25,
        },
        "formality_range": (0.0, 0.4),
        "preferred_fabric_families": ["natural", "casual"],
        "garment_weights": {},
    },
    "vintage_retro": {
        "slot_probabilities": {
            "headwear": 0.30,
            "outerwear": 0.30,
            "accessories": 0.60,
            "bag": 0.35,
        },
        "formality_range": (0.2, 0.7),
        "preferred_fabric_families": ["natural", "luxury"],
        "garment_weights": {},
    },
    "cyberpunk_neon": {
        "slot_probabilities": {
            "headwear": 0.20,
            "outerwear": 0.55,
            "accessories": 0.80,
            "bag": 0.15,
        },
        "formality_range": (0.1, 0.5),
        "preferred_fabric_families": ["sporty", "tough"],
        "garment_weights": {},
    },
    "pastel_dream": {
        "slot_probabilities": {
            "headwear": 0.25,
            "outerwear": 0.20,
            "accessories": 0.55,
            "bag": 0.30,
        },
        "formality_range": (0.1, 0.6),
        "preferred_fabric_families": ["natural", "luxury", "casual"],
        "garment_weights": {},
    },
    "earthy_natural": {
        "slot_probabilities": {
            "headwear": 0.25,
            "outerwear": 0.30,
            "accessories": 0.45,
            "bag": 0.30,
        },
        "formality_range": (0.1, 0.5),
        "preferred_fabric_families": ["natural", "tough"],
        "garment_weights": {},
    },
    "monochrome_chic": {
        "slot_probabilities": {
            "headwear": 0.10,
            "outerwear": 0.35,
            "accessories": 0.60,
            "bag": 0.30,
        },
        "formality_range": (0.3, 0.8),
        "preferred_fabric_families": None,  # all allowed
        "garment_weights": {},
    },
    "tropical": {
        "slot_probabilities": {
            "headwear": 0.55,
            "outerwear": 0.05,
            "accessories": 0.65,
            "bag": 0.25,
        },
        "formality_range": (0.0, 0.3),
        "preferred_fabric_families": ["natural", "casual", "sporty"],
        "garment_weights": {},
    },
    "winter_cozy": {
        "slot_probabilities": {
            "headwear": 0.55,
            "outerwear": 0.80,
            "accessories": 0.50,
            "bag": 0.20,
        },
        "formality_range": (0.1, 0.6),
        "preferred_fabric_families": ["natural", "casual"],
        "garment_weights": {},
    },
    "festival": {
        "slot_probabilities": {
            "headwear": 0.45,
            "outerwear": 0.15,
            "accessories": 0.85,
            "bag": 0.40,
        },
        "formality_range": (0.0, 0.4),
        "preferred_fabric_families": ["casual", "natural", "sporty"],
        "garment_weights": {},
    },
    "office_professional": {
        "slot_probabilities": {
            "headwear": 0.02,
            "outerwear": 0.40,
            "accessories": 0.65,
            "bag": 0.55,
        },
        "formality_range": (0.5, 1.0),
        "preferred_fabric_families": ["natural", "luxury"],
        "garment_weights": {},
    },
}
