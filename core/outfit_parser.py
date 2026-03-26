"""Parser for outfit override strings."""

import re


def parse_overrides(override_string):
    """Parse multiline override string into per-slot overrides.

    Format per line: slot_name: garment_spec [fabric] | color_tag
    Special values: 'exclude', 'auto'
    Wildcards: {option1|option2}

    Returns: dict {slot_name: {"garment": str|None, "fabric": str|None,
                                "color_tag": str|None, "mode": "override"|"exclude"|"auto"}}
    """
    if not override_string or not override_string.strip():
        return {}

    result = {}

    for line in override_string.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Split on first colon
        if ":" not in line:
            continue

        slot_name, _, spec = line.partition(":")
        slot_name = slot_name.strip().lower()
        spec = spec.strip()

        if not slot_name or not spec:
            continue

        spec_lower = spec.lower()

        if spec_lower == "exclude":
            result[slot_name] = {
                "garment": None,
                "fabric": None,
                "color_tag": None,
                "mode": "exclude",
                "decoration": None,
            }
            continue

        if spec_lower == "auto":
            result[slot_name] = {
                "garment": None,
                "fabric": None,
                "color_tag": None,
                "mode": "auto",
                "decoration": None,
            }
            continue

        # Parse: garment_spec | color_tag [| decoration]
        color_tag = None
        decoration = None

        if "|" in spec:
            pipe_parts = [p.strip() for p in spec.split("|")]
            spec = pipe_parts[0]
            if len(pipe_parts) >= 2 and pipe_parts[1]:
                tag_part = pipe_parts[1]
                # Ensure it has # delimiters
                if not tag_part.startswith("#"):
                    tag_part = f"#{tag_part}#"
                elif not tag_part.endswith("#"):
                    tag_part = f"{tag_part}#"
                color_tag = tag_part
            if len(pipe_parts) >= 3 and pipe_parts[2]:
                deco_part = pipe_parts[2]
                if deco_part.lower().startswith("text:"):
                    # Text override: text:"STAY WILD" in gothic font
                    decoration = deco_part[5:].strip()
                else:
                    decoration = deco_part

        # Parse garment and fabric from spec
        # Try to find fabric in brackets: "silk blouse" or just "blouse"
        garment = None
        fabric = None

        if spec:
            words = spec.split()
            if len(words) >= 2:
                # First word could be fabric, rest is garment
                fabric = words[0]
                garment = " ".join(words[1:])
            elif len(words) == 1:
                garment = words[0]

        result[slot_name] = {
            "garment": garment,
            "fabric": fabric,
            "color_tag": color_tag,
            "mode": "override",
            "decoration": decoration,
        }

    return result


def resolve_wildcards(text, rng):
    """Replace {a|b|c} patterns with random choice using rng.
    Returns resolved string.
    """
    if not text:
        return text

    pattern = re.compile(r'\{([^}]+)\}')

    def _replace(match):
        options = [o.strip() for o in match.group(1).split("|") if o.strip()]
        if not options:
            return match.group(0)
        return rng.choice(options)

    return pattern.sub(_replace, text)
