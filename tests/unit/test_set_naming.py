"""Set directory names must read as English prose.

The JB ``sentences`` output format reads set names back mechanically
(``_`` -> space), so the directory names *are* the wording of the intro:

    indoor/family_event/graduation_party_at_home
    -> The scene takes place indoors, it is a family event, namely the
       graduation party at home.

    female/business/dress
    -> The outfit is a business look, in the dress style.

These tests pin the naming rules from ``.claude/commands/build-location-set.md``
and ``build-outfit-set.md`` so new sets keep reading cleanly. A failure here
means: rename the directory (``scripts/rename_sets.py``), not the test.
"""

import re

import pytest

from core.jb.sentences import location_intro, outfit_intro
from core.location_engine import get_available_location_sets
from core.outfit_lists import get_available_sets

SLUG = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

# Country / state codes read as gibberish ("namely the bus stop de").
# Spell them out: german_bus_stop, pennsylvania_suburb, american_office.
# City names people actually say (nyc, la) are fine.
CODE_TOKENS = {"us", "usa", "de", "pa", "uk", "eu"}

# Trailing qualifiers that only make sense as a database column, not in a
# sentence ("namely the sandy beach busy" -> busy_sandy_beach).
TRAILING_QUALIFIERS = {"inside", "interior", "plain", "busy", "private",
                       "public", "us", "de", "pa"}

# Location categories are read as "it is a <category>", so they must be
# singular. Words that end in s but are singular go here.
SINGULAR_S_WORDS = {"sports", "premises", "gymnastics", "athletics", "campus"}

LOCATION_SETS = get_available_location_sets()
OUTFIT_SETS = get_available_sets()


def _tokens(segment: str) -> list[str]:
    return segment.split("_")


# ─── Structure ───────────────────────────────────────────────────────


@pytest.mark.parametrize("set_name", LOCATION_SETS)
def test_location_set_is_scope_category_leaf(set_name):
    parts = set_name.split("/")
    assert len(parts) == 3, f"{set_name}: expected indoor|outdoor/<category>/<leaf>"
    assert parts[0] in ("indoor", "outdoor"), set_name
    for p in parts[1:]:
        assert SLUG.match(p), f"{set_name}: segment {p!r} is not lowercase snake_case"


@pytest.mark.parametrize("set_name", OUTFIT_SETS)
def test_outfit_set_is_gender_category_leaf(set_name):
    parts = set_name.split("/")
    assert len(parts) == 3, (f"{set_name}: expected female|male/<category>/<leaf> "
                             "— legacy flat sets belong in outfit_lists/_archive/")
    assert parts[0] in ("female", "male", "unisex"), set_name
    for p in parts[1:]:
        assert SLUG.match(p), f"{set_name}: segment {p!r} is not lowercase snake_case"


# ─── Wording ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("set_name", LOCATION_SETS + OUTFIT_SETS)
def test_no_country_or_state_codes(set_name):
    for segment in set_name.split("/")[1:]:
        bad = CODE_TOKENS & set(_tokens(segment))
        assert not bad, (f"{set_name}: token(s) {sorted(bad)} read as gibberish — "
                         "spell out (american_, german_, pennsylvania_)")


@pytest.mark.parametrize("set_name", LOCATION_SETS)
def test_location_leaf_has_no_trailing_qualifier(set_name):
    leaf = set_name.split("/")[-1]
    last = _tokens(leaf)[-1]
    assert last not in TRAILING_QUALIFIERS, (
        f"{set_name}: put the qualifier in front (busy_sandy_beach, not sandy_beach_busy)")


@pytest.mark.parametrize("set_name", LOCATION_SETS)
def test_location_leaf_is_a_noun_phrase(set_name):
    leaf = set_name.split("/")[-1]
    assert len(_tokens(leaf)) >= 2, (
        f"{set_name}: one-word leaf reads thin after 'namely the' — "
        "say what it is (hotel_lobby, sandy_beach)")
    assert leaf != "general", f"{set_name}: 'general' is an outfit convention only"


@pytest.mark.parametrize("category", sorted({s.split("/")[1] for s in LOCATION_SETS}))
def test_location_category_is_singular(category):
    last = _tokens(category)[-1]
    assert not (last.endswith("s") and last not in SINGULAR_S_WORDS
                and not last.endswith("ss")), (
        f"{category}: read as 'it is a {category.replace('_', ' ')}' — use the singular")


# ─── The intro must come out clean for every set on disk ─────────────


@pytest.mark.parametrize("set_name", LOCATION_SETS)
def test_location_intro_reads(set_name):
    intro = location_intro(set_name)
    assert re.fullmatch(r"The scene takes place (indoors|outdoors), it is an? [a-z0-9 ]+, "
                        r"namely the [a-z0-9 ]+\.", intro), intro


@pytest.mark.parametrize("set_name", OUTFIT_SETS)
def test_outfit_intro_reads(set_name):
    intro = outfit_intro(set_name)
    assert re.fullmatch(r"The outfit is an? [a-z0-9 ]+ look(, in the [a-z0-9 ]+ style)?\.",
                        intro), intro
