"""JB ``sentences`` format — deterministic prose for Krea 2 / Qwen encoders.

``natural`` strips a JB payload down to bare phrases joined by commas.
``sentences`` goes one step further and writes what the keys *mean*: an
intro sentence built from the set path, then one sentence per element or
garment with a lead-in that names the part of the scene or outfit the
phrase describes::

    {"location": {"set_name": "indoor/family_event/graduation_party_at_home",
                  "elements": {"background": {"prompt_fragment": "..."}, ...}}}

    The scene takes place indoors, it is a family event, namely the
    graduation party at home. The background is .... The time of day is ....

Everything is mechanical and deterministic. Set names are read back with
``_`` -> space, articles follow the vowel rule, and nothing is looked up in
a table that could drift away from the directories on disk — the directory
names *are* the wording (see ``scripts/rename_sets_map.txt`` and the naming
rules in ``.claude/commands/build-location-set.md``).

Generation metadata (seed, coverage, colour roles, formality, colour tone,
layer, ...) is dropped exactly as ``natural`` drops it. Where a record
already carries a finished phrase (``prompt_fragment``) only that phrase is
used; ``name``, ``fabric`` and ``texture`` are folded into it already.

Unknown structures (Builder rows, hand-written JSON) fall back to one
sentence per branch: ``The face has age twenties and eyes amber.``
"""

from __future__ import annotations

import re
from typing import Any

from .serialize import NON_PROMPT_KEYS, PHRASE_KEYS, SENTENCES  # noqa: F401  (SENTENCES re-exported)

# ─── Wording ───────────────────────────────────────────────────────────

LOCATION_SCOPES = {"indoor": "indoors", "outdoor": "outdoors"}
OUTFIT_GENDERS = ("female", "male", "unisex")

# One template per known key. ``{}`` receives the resolved phrase verbatim.
LOCATION_LEAD_INS = {
    "background":          "The background is {}.",
    "midground":           "The middle ground shows {}.",
    "architecture_detail": "An architectural detail is {}.",
    "props":               "The props include {}.",
    "foreground_element":  "In the foreground is {}.",
    "time_of_day":         "The time of day is {}.",
    "weather":             "The weather is {}.",
}

# Region ids as the Outfit Block emits them, plus the engine slot names so
# SMP / hand-written payloads read the same way.
GARMENT_LEAD_INS = {
    "headwear":             "The headwear is {}.",
    "upper_body":           "The top is {}.",
    "top":                  "The top is {}.",
    "upper_body_outerwear": "The outerwear is {}.",
    "outerwear":            "The outerwear is {}.",
    "lower_body":           "The bottom is {}.",
    "bottom":               "The bottom is {}.",
    "footwear":             "The footwear is {}.",
    "accessories":          "The accessories are {}.",
    "bag":                  "The bag is {}.",
}

# Dress sets keep the dress in the *bottom* slot (top.txt is a "none" stub),
# so "The bottom is ... sheath dress" would be wrong. A garment whose head
# noun names a one-piece gets its own lead-in — but only in the body slots:
# a kimono in the outerwear slot is still outerwear.
ONE_PIECE_LEAD_IN = "The one-piece garment is {}."
ONE_PIECE_SLOTS = frozenset({"upper_body", "lower_body", "top", "bottom"})

# Head nouns (last word of the garment name) that make the garment cover top
# and bottom at once. Matched on the head noun only, so "dress shirt",
# "dress pants", "bikini top" and "tracksuit bottoms" stay two-piece.
ONE_PIECE_HEAD_NOUNS = frozenset({
    # dresses
    "dress", "sundress", "minidress", "mididress", "maxidress", "shirtdress",
    "housedress", "gown", "ballgown", "nightgown", "nightdress", "nightie",
    "dirndl", "kirtle", "pinafore",
    # all-in-ones
    "jumpsuit", "romper", "playsuit", "catsuit", "bodysuit", "unitard",
    "leotard", "onesie", "overalls", "dungarees", "coveralls", "boilersuit",
    "flightsuit", "one-piece", "onepiece", "all-in-one",
    # swim / sport / weather
    "swimsuit", "monokini", "wetsuit", "snowsuit", "skisuit", "bodystocking",
    # lingerie / sleep
    "teddy", "chemise", "babydoll", "negligee", "slip", "nightshirt",
    # robes and traditional one-pieces worn as the main garment
    "kaftan", "caftan", "abaya", "sari", "saree", "qipao", "cheongsam",
    "djellaba", "thobe", "muumuu", "toga", "cassock", "cover-up", "coverup",
})

# Stitcher slot keys for bare prose / arrays. Their values are emitted as-is
# — "The input 1 is wearing ..." would be nonsense.
_PASSTHROUGH_KEY = re.compile(r"^inputs?(?:_\d+)?$")


# ─── Small helpers ────────────────────────────────────────────────────


def _words(slug: Any) -> str:
    """``graduation_party_at_home`` -> ``graduation party at home``."""
    return " ".join(str(slug).replace("_", " ").replace("-", " ").split())


def _article(phrase: str) -> str:
    return "an" if phrase[:1].lower() in "aeiou" else "a"


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def _sentence(text: str) -> str:
    text = _clean(text).rstrip(" .")
    return text + "." if text else ""


def _segments(set_name: Any) -> list[str]:
    return [p for p in str(set_name or "").replace("\\", "/").split("/") if p]


# ─── Intros from the set path ──────────────────────────────────────────


def location_intro(set_name: str) -> str:
    """``indoor/family_event/graduation_party_at_home`` ->
    ``The scene takes place indoors, it is a family event, namely the
    graduation party at home.``"""
    parts = _segments(set_name)
    if not parts:
        return ""
    scope = LOCATION_SCOPES.get(parts[0])
    if scope is None:
        # Legacy flat slug ("indoor_business_skyscraper_lobby").
        if len(parts) == 1:
            for key, word in LOCATION_SCOPES.items():
                if parts[0].startswith(key + "_"):
                    rest = _words(parts[0][len(key) + 1:])
                    return _sentence(f"The scene takes place {word}, namely the {rest}")
        return _sentence(f"The scene is the {_words(' '.join(parts))}")
    if len(parts) == 1:
        return _sentence(f"The scene takes place {scope}")
    if len(parts) == 2:
        return _sentence(f"The scene takes place {scope}, namely the {_words(parts[1])}")
    category = _words(parts[1])
    leaf = _words(" ".join(parts[2:]))
    return _sentence(f"The scene takes place {scope}, it is {_article(category)} "
                     f"{category}, namely the {leaf}")


def outfit_intro(set_name: str) -> str:
    """``female/business/dress`` -> ``The outfit is a business look, in the
    dress style.`` The gender segment is dropped — the outfit string describes
    the clothes, never a person (Krea 2 paints an extra one otherwise)."""
    parts = _segments(set_name)
    if not parts:
        return ""
    if parts[0] in OUTFIT_GENDERS:
        parts = parts[1:]
    elif len(parts) == 1:
        # Legacy flat slug: tokens before the gender = style, after = variant.
        tokens = parts[0].split("_")
        gender = next((g for g in OUTFIT_GENDERS if g in tokens), None)
        if gender is not None:
            idx = tokens.index(gender)
            parts = ["_".join(tokens[:idx]), "_".join(tokens[idx + 1:])]
            parts = [p for p in parts if p]
    if not parts:
        return ""
    category = _words(parts[0])
    leaf = _words(" ".join(parts[1:])) if len(parts) > 1 else ""
    base = f"The outfit is {_article(category)} {category} look"
    if leaf and leaf != "general":
        return _sentence(f"{base}, in the {leaf} style")
    return _sentence(base)


# ─── Records (elements / garments) ─────────────────────────────────────


def _phrase(node: dict) -> str | None:
    for key in PHRASE_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return _clean(value)
    return None


def _head_noun(name: str) -> str:
    text = re.sub(r"#[^#]*#", " ", str(name or ""))
    text = re.split(r"\s+with\s+", text, maxsplit=1)[0]  # drop "... with floral print"
    words = re.findall(r"[a-z][a-z-]*[a-z]|[a-z]", text.lower())
    return words[-1] if words else ""


def is_one_piece(name: str) -> bool:
    """True when the garment's head noun names a top-and-bottom-in-one garment."""
    return _head_noun(name) in ONE_PIECE_HEAD_NOUNS


def element_sentence(key: str | None, record: dict) -> str:
    """One sentence for a record that carries a finished phrase."""
    phrase = _phrase(record)
    if phrase is None:
        return ""
    k = str(key or "")
    if k in ONE_PIECE_SLOTS and is_one_piece(record.get("name") or phrase):
        return _sentence(ONE_PIECE_LEAD_IN.format(phrase))
    template = LOCATION_LEAD_INS.get(k) or GARMENT_LEAD_INS.get(k)
    if template:
        return _sentence(template.format(phrase))
    if k:
        return _sentence(f"The {_words(k)} is {phrase}")
    return _sentence(phrase)


# ─── Walker ───────────────────────────────────────────────────────────


def _is_block(node: dict, child: str) -> bool:
    return isinstance(node.get(child), dict) or isinstance(node.get("set_name"), str)


def _block(node: dict, intro_fn) -> list[str]:
    out: list[str] = []
    intro = intro_fn(node.get("set_name") or "")
    if intro:
        out.append(intro)
    for key, value in node.items():
        ks = str(key)
        if ks in NON_PROMPT_KEYS or ks.startswith("_"):
            continue  # set_name is consumed by the intro; the rest is metadata
        out.extend(_collect(value, ks))
    return out


def location_sentences(block: dict) -> list[str]:
    return _block(block, location_intro)


def outfit_sentences(block: dict) -> list[str]:
    return _block(block, outfit_intro)


def _collect(node: Any, key: str | None) -> list[str]:
    out: list[str] = []
    passthrough = bool(key) and _PASSTHROUGH_KEY.match(key) is not None

    if isinstance(node, dict):
        if key == "location" and _is_block(node, "elements"):
            return location_sentences(node)
        if key == "outfit" and _is_block(node, "garments"):
            return outfit_sentences(node)
        if _phrase(node) is not None:
            s = element_sentence(key, node)
            return [s] if s else []

        leaves: list[tuple[str, str, bool]] = []   # (key, text, plural)
        branches: list[tuple[str, Any]] = []
        for k, v in node.items():
            ks = str(k)
            if ks in NON_PROMPT_KEYS or ks.startswith("_"):
                continue
            if isinstance(v, dict):
                branches.append((ks, v))
            elif isinstance(v, list):
                if v and all(isinstance(x, str) for x in v):
                    strs = [_clean(x) for x in v if x.strip()]
                    if strs:
                        if _PASSTHROUGH_KEY.match(ks):
                            out.append(_sentence(_join(strs)))
                        else:
                            leaves.append((ks, _join(strs), True))
                else:
                    branches.append((ks, v))
            elif isinstance(v, str):
                if v.strip():
                    if _PASSTHROUGH_KEY.match(ks):
                        out.append(_sentence(v))
                    else:
                        leaves.append((ks, _clean(v), False))
            # numbers, bools and None never describe an image on their own.

        if leaves:
            if key is None or passthrough:
                for k, text, plural in leaves:
                    verb = "are" if plural else "is"
                    out.append(_sentence(f"The {_words(k)} {verb} {text}"))
            else:
                parts = [f"{_words(k)} {text}" for k, text, _ in leaves]
                out.append(_sentence(f"The {_words(key)} has {_join(parts)}"))
        for k, v in branches:
            out.extend(_collect(v, k))
        return out

    if isinstance(node, (list, tuple)):
        strs = [_clean(x) for x in node if isinstance(x, str) and x.strip()]
        if strs:
            if key and not passthrough:
                out.append(_sentence(f"The {_words(key)} are {_join(strs)}"))
            else:
                out.append(_sentence(_join(strs)))
        for item in node:
            if not isinstance(item, str):
                out.extend(_collect(item, key))
        return out

    if isinstance(node, str) and node.strip():
        if key and not passthrough:
            out.append(_sentence(f"The {_words(key)} is {_clean(node)}"))
        else:
            out.append(_sentence(node))
    return out


# ─── Public API ────────────────────────────────────────────────────────


def sentences(obj: Any) -> list[str]:
    """All sentences for a payload, in document order, exact duplicates removed."""
    seen: set[str] = set()
    unique: list[str] = []
    for s in _collect(obj, None):
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        unique.append(s)
    return unique


def emit_sentences(obj: Any, prefix: str = "") -> str:
    """Prose for models that read prompts as language, one sentence per fact."""
    parts = sentences(obj)
    if not parts:
        return ""
    return f"{prefix}{' '.join(parts)}"
