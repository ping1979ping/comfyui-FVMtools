"""FVM_BatchRouter — turn a handful of checks into one destination folder.

The checks that decide a picture's fate come from different nodes: the reality
check says the anatomy holds up, PersonSelector says the face is the right
person and how many faces it found. This node ANDs them together and answers
with the folder the picture belongs in, so a single Batch Save node does the
writing and the decision lives in one visible place.

Every gate is optional. An unconnected gate is not "false", it is "not asked" —
otherwise adding a check you have not wired yet would silently reject the whole
batch.
"""


def _gate_state(value):
    """None (not connected) stays None; everything else becomes a bool."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return bool(value[0]) if value else None
    return bool(value)


class FVM_BatchRouter:
    """Combine pass/fail gates into one target directory."""

    CATEGORY = "FVM Tools/Batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pass_dir": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Folder for pictures that clear every gate.",
                }),
                "fail_dir": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Folder for pictures that fail any gate.",
                }),
                "combine": (["all", "any"], {
                    "default": "all",
                    "tooltip": "all: every connected gate must pass. "
                               "any: one is enough.",
                }),
                "label_a": ("STRING", {"default": "reality"}),
                "label_b": ("STRING", {"default": "identity"}),
                "label_c": ("STRING", {"default": "extra"}),
                "count_min": ("INT", {
                    "default": 1, "min": 0, "max": 100,
                    "tooltip": "Lowest acceptable value on the count input — "
                               "e.g. 1 face. Ignored when count is unconnected.",
                }),
                "count_max": ("INT", {
                    "default": 1, "min": 0, "max": 100,
                    "tooltip": "Highest acceptable value on the count input. "
                               "1/1 means 'exactly one person'.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Swap the two folders — useful to collect rejects.",
                }),
            },
            "optional": {
                "gate_a": ("BOOLEAN", {"forceInput": True}),
                "gate_b": ("BOOLEAN", {"forceInput": True}),
                "gate_c": ("BOOLEAN", {"forceInput": True}),
                "count": ("INT", {
                    "forceInput": True,
                    "tooltip": "A count to bound, e.g. PersonSelector's "
                               "face_count.",
                }),
                "report_in": ("STRING", {
                    "default": "", "forceInput": True, "multiline": True,
                    "tooltip": "Upstream report, prepended to this node's own.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("target_dir", "passed", "failed", "report")
    FUNCTION = "execute"
    DESCRIPTION = ("Combines reality, identity and count checks into the one "
                   "folder a picture should go to.")

    def execute(self, pass_dir, fail_dir, combine, label_a, label_b, label_c,
                count_min, count_max, invert, gate_a=None, gate_b=None,
                gate_c=None, count=None, report_in=""):
        checks = []
        for label, value in ((label_a or "gate a", gate_a),
                             (label_b or "gate b", gate_b),
                             (label_c or "gate c", gate_c)):
            state = _gate_state(value)
            if state is not None:
                checks.append((label, state, "pass" if state else "fail"))

        if count is not None:
            value = count[0] if isinstance(count, (list, tuple)) and count else count
            try:
                number = int(value)
            except (TypeError, ValueError):
                number = -1
            within = int(count_min) <= number <= int(count_max)
            checks.append((f"count({number})", within,
                           "pass" if within else
                           f"fail — outside {int(count_min)}..{int(count_max)}"))

        if not checks:
            # Nothing to decide on: send everything to pass rather than dumping a
            # whole batch into reject because no gate was wired up yet.
            passed = True
            summary = "no gates connected — passing everything"
        else:
            states = [state for _label, state, _note in checks]
            passed = all(states) if combine == "all" else any(states)
            summary = f"{combine}: " + ", ".join(
                f"{label} {note}" for label, _state, note in checks)

        if invert:
            passed = not passed
            summary += "  (inverted)"

        target = (pass_dir if passed else fail_dir) or ""
        verdict = "PASS" if passed else "FAIL"
        report = f"{verdict}  {summary}\n→ {target}"
        if report_in and report_in.strip():
            report = f"{report_in.rstrip()}\n\n{report}"

        print(f"[FVM Batch Router] {verdict}  {summary}")
        return {"ui": {"text": [f"{verdict}  {summary}"]},
                "result": (target, passed, not passed, report)}
