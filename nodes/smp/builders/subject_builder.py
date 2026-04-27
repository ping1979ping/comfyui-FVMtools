"""SMP SubjectBuilder — adds a Subject record to a PROMPT_DICT.

Inputs are a multiline JSON or scalar widget set. The resulting subject is
appended to ``prompt_dict.subjects`` (list-merge) so two SubjectBuilders
chained produce a multi-person scene.
"""

import json

try:
    from ....core.smp.merge import deep_merge
except ImportError:  # pragma: no cover
    from core.smp.merge import deep_merge


_SUBJECT_JSON_DEFAULT = (
    '{\n  "skin_tags": ["smooth skin"],\n  "expression": "neutral expression"\n}'
)


def _safe_load(s: str) -> dict:
    s = (s or "").strip()
    if not s or s == "{}":
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


class FVM_SMP_SubjectBuilder:
    CATEGORY = "FVM Tools/SMP/Builders"
    FUNCTION = "build"
    RETURN_TYPES = ("PROMPT_DICT",)
    RETURN_NAMES = ("prompt_dict",)
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Builds a Subject and adds it to a PROMPT_DICT.subjects list.\n"
        "Chain two builders to generate multi-person scenes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject_id":  ("STRING", {"default": "subject_1"}),
                "age_desc":    ("STRING", {"default": "young"}),
                "gender":      ("STRING", {"default": "woman"}),
                "expression":  ("STRING", {"default": "neutral expression"}),
                "hair_color_length": ("STRING", {"default": ""}),
                "pose_hint":   ("STRING", {"default": "", "multiline": True}),
                "extra_json":  ("STRING", {"default": _SUBJECT_JSON_DEFAULT, "multiline": True}),
            },
            "optional": {
                "prompt_dict_in": ("PROMPT_DICT",),
                "subject_dict":   ("SUBJECT_DICT",),
            },
        }

    def build(self, subject_id, age_desc, gender, expression, hair_color_length,
              pose_hint, extra_json, prompt_dict_in=None, subject_dict=None):
        # Direct dict input wins, otherwise compose from widgets + extras.
        if subject_dict:
            subject = dict(subject_dict)
            subject.setdefault("id", subject_id or "subject_1")
        else:
            subject = {
                "id":         subject_id or "subject_1",
                "age_desc":   age_desc or None,
                "gender":     gender or None,
                "expression": expression or None,
            }
            if hair_color_length:
                subject["hair_color_length"] = hair_color_length
            if pose_hint:
                subject["pose_hint"] = pose_hint
            extras = _safe_load(extra_json)
            if extras:
                subject.update(extras)
            # Strip None entries
            subject = {k: v for k, v in subject.items() if v is not None}

        partial = {"subjects": [subject]}
        merged = deep_merge(prompt_dict_in or {}, partial)
        return (merged,)
