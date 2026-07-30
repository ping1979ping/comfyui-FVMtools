"""FVM Signs — core engine for detecting, judging and repairing rendered text.

Three independent layers:

* ``classes``  — registry of the nine text-region classes (SAM3 prompts,
  thresholds, VLM instructions, diffusion templates, denoise bias).
* ``slop``     — offline heuristics that tell gibberish glyphs from real text.
* ``cluster``  — grouping of near-identical crops so one repair serves many.

No ComfyUI imports at module level, so everything here is testable without a
running server.
"""

from .classes import (
    CLASS_KEYS,
    CLASS_ORDER,
    DEFAULT_MIN_HEIGHT_PX,
    DEFAULT_THRESHOLD,
    FALLBACK_CLASS,
    FALLBACK_CLASS_NAME,
    MAX_DENOISE_BIAS,
    MIN_DENOISE_BIAS,
    SIGN_CLASSES,
    all_class_names,
    build_prompt,
    clamp_denoise_bias,
    collapse_separators,
    get_class,
    parse_custom_prompts,
)
from .slop import (
    BIGRAM_FREQ,
    DEFAULT_SLOP_WEIGHTS,
    EMPTY_DETECTED_FLOOR,
    MAX_BIGRAM_WEIGHT,
    SIGNAL_KEYS,
    SLOP_THRESHOLD,
    WORD_LIST,
    bigram_plausibility,
    dictionary_ratio,
    repeated_glyph_ratio,
    score_slop,
)
from .cluster import (
    DEFAULT_CLUSTER_DISTANCE,
    HIST_BINS,
    HIST_WEIGHT,
    PHASH_SIZE,
    PHASH_WEIGHT,
    cluster_crops,
    color_signature,
    crop_distance,
    extract_features,
    phash,
    pick_cluster_representative,
)

__all__ = [
    "BIGRAM_FREQ",
    "CLASS_KEYS",
    "CLASS_ORDER",
    "DEFAULT_CLUSTER_DISTANCE",
    "DEFAULT_MIN_HEIGHT_PX",
    "DEFAULT_SLOP_WEIGHTS",
    "DEFAULT_THRESHOLD",
    "EMPTY_DETECTED_FLOOR",
    "FALLBACK_CLASS",
    "FALLBACK_CLASS_NAME",
    "HIST_BINS",
    "HIST_WEIGHT",
    "MAX_BIGRAM_WEIGHT",
    "MAX_DENOISE_BIAS",
    "MIN_DENOISE_BIAS",
    "PHASH_SIZE",
    "PHASH_WEIGHT",
    "SIGNAL_KEYS",
    "SIGN_CLASSES",
    "SLOP_THRESHOLD",
    "WORD_LIST",
    "all_class_names",
    "bigram_plausibility",
    "build_prompt",
    "clamp_denoise_bias",
    "cluster_crops",
    "collapse_separators",
    "color_signature",
    "crop_distance",
    "dictionary_ratio",
    "extract_features",
    "get_class",
    "parse_custom_prompts",
    "phash",
    "pick_cluster_representative",
    "repeated_glyph_ratio",
    "score_slop",
]
