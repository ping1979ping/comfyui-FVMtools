"""ComfyUI custom-type identifiers used by SMP nodes.

These are plain string identifiers — ComfyUI custom types are nominal, not
structural. Nodes consume / produce dicts on the wire; pydantic is for
validation only (see schema.py).
"""

PROMPT_DICT          = "PROMPT_DICT"
OUTFIT_DICT_RAW      = "OUTFIT_DICT_RAW"
OUTFIT_DICT          = "OUTFIT_DICT"
LOCATION_DICT_RAW    = "LOCATION_DICT_RAW"
LOCATION_DICT        = "LOCATION_DICT"
COLOR_PALETTE_DICT   = "COLOR_PALETTE_DICT"
SUBJECT_DICT         = "SUBJECT_DICT"
STRUCTURED_PROMPTS   = "STRUCTURED_PROMPTS"
REGION_MAP           = "REGION_MAP"

ALL_SMP_TYPES = (
    PROMPT_DICT,
    OUTFIT_DICT_RAW,
    OUTFIT_DICT,
    LOCATION_DICT_RAW,
    LOCATION_DICT,
    COLOR_PALETTE_DICT,
    SUBJECT_DICT,
    STRUCTURED_PROMPTS,
    REGION_MAP,
)
