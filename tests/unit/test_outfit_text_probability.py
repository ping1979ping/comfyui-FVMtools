"""text_probability — the text chance decoupled from the print chance.

Two things are pinned here:

1. Backward compatibility. The old engine had ONE probability and split it
   50/50 between print and text. With ``text_probability=0.0`` (the new
   default) the new engine must reproduce the old ``text_mode="off"`` output
   bit for bit — same rng consumption, same garments, same decorations.
   The reference is not a frozen hash (the outfit_lists data files change
   often); it is the pre-change algorithm, re-implemented below and patched
   over the engine so both variants run against the *same* data.

2. The new semantics. With ``p_p + p_t <= 1`` the per-garment chance of a
   print is exactly ``p_p`` and of a text exactly ``p_t``, and the whole
   decision still costs exactly 3 rng calls.
"""

import random

import pytest

import core.outfit_engine as engine
from core.outfit_engine import generate_outfit, generate_outfit_records
from core.outfit_lists import get_available_sets, load_texts


ALL_SLOTS = ["headwear", "top", "outerwear", "bottom", "footwear",
             "accessories", "bag"]
ALL_ENABLED = {s: True for s in ALL_SLOTS}

# Four sets with different prints.txt/texts.txt shapes.
COMPAT_SETS = [
    "female/teen/school_day_skirt",
    "female/casual/coastal_cool",
    "female/business/blazer_combo",
    "female/lingerie/lace",
]
COMPAT_PROBS = [0.0, 0.25, 0.5, 0.75, 1.0]
COMPAT_SEEDS = range(200)


# ─── the pre-change algorithm, verbatim ──────────────────────────────────

def _legacy_pick_decoration(rng, slot, formality, prints_list, texts_list,
                            print_probability, text_mode, text_probability=0.0):
    """``_pick_decoration`` as it stood before text_probability existed.

    ``text_probability`` is accepted and ignored — the old code had no such
    parameter. Kept in the signature so the same call sites can drive it.
    """
    prob_roll = rng.random()
    type_roll = rng.random()
    select_roll = rng.random()

    if prob_roll >= print_probability:
        return None

    use_text = type_roll >= 0.5 and text_mode != "off" and texts_list

    if use_text:
        compatible = [t for t in texts_list if slot in t["slots"]]
        if not compatible:
            return engine._select_print(select_roll, slot, formality, prints_list)
        weights = [t["probability"] for t in compatible]
        total = sum(weights)
        if total <= 0:
            return None
        target = select_roll * total
        cumulative = 0.0
        chosen = compatible[-1]
        for t in compatible:
            cumulative += t["probability"]
            if cumulative >= target:
                chosen = t
                break
        return engine._format_text_decoration(chosen, text_mode)
    return engine._select_print(select_roll, slot, formality, prints_list)


def _sweep(**kwargs):
    """Every (set, print_probability, seed) combination, as a flat dict."""
    out = {}
    for outfit_set in COMPAT_SETS:
        for prob in COMPAT_PROBS:
            for seed in COMPAT_SEEDS:
                r = generate_outfit(seed=seed, outfit_set=outfit_set,
                                    formality=0.5, coverage=0.7,
                                    slot_enables=ALL_ENABLED,
                                    print_probability=prob, **kwargs)
                out[(outfit_set, prob, seed)] = (r["outfit_prompt"],
                                                 r["outfit_details"])
    return out


# ─── 1. backward compatibility ───────────────────────────────────────────

def test_text_off_is_bit_identical_to_the_old_engine(monkeypatch):
    """200 seeds x 4 sets x 5 probabilities, old vs new, no difference."""
    monkeypatch.setattr(engine, "_pick_decoration", _legacy_pick_decoration)
    old = _sweep(text_mode="off")
    monkeypatch.undo()   # back to the real engine for the second sweep
    new = _sweep(text_mode="off", text_probability=0.0)

    assert len(old) == len(COMPAT_SETS) * len(COMPAT_PROBS) * len(COMPAT_SEEDS)
    differing = [k for k in old if old[k] != new[k]]
    assert not differing, (
        f"{len(differing)}/{len(old)} outputs changed, e.g. {differing[:3]}"
    )


def test_default_text_probability_is_zero():
    """Callers that never heard of the parameter keep the print-only path."""
    explicit = generate_outfit(seed=17, outfit_set=COMPAT_SETS[0],
                               slot_enables=ALL_ENABLED, print_probability=0.6,
                               text_mode="quoted", text_probability=0.0)
    implicit = generate_outfit(seed=17, outfit_set=COMPAT_SETS[0],
                               slot_enables=ALL_ENABLED, print_probability=0.6,
                               text_mode="quoted")
    assert explicit == implicit
    assert "text in" not in implicit["outfit_prompt"]


# ─── 2. rng budget ───────────────────────────────────────────────────────

class _CountingRandom(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.calls = 0

    def random(self):
        self.calls += 1
        return super().random()


@pytest.mark.parametrize("p_print,p_text,mode", [
    (0.0, 0.0, "off"),
    (0.3, 0.0, "quoted"),
    (0.0, 0.3, "quoted"),
    (0.5, 0.5, "quoted"),
    (0.9, 0.9, "descriptive"),
    (0.4, 0.4, "off"),
])
def test_pick_decoration_always_costs_three_rolls(p_print, p_text, mode):
    prints = engine.load_prints(COMPAT_SETS[0])
    texts = engine.load_texts(COMPAT_SETS[0])
    for seed in range(50):
        rng = _CountingRandom(seed)
        engine._pick_decoration(rng, "top", 0.5, prints, texts,
                                p_print, mode, p_text)
        assert rng.calls == 3


def test_consume_matches_pick_rng_budget():
    prints = engine.load_prints(COMPAT_SETS[0])
    texts = engine.load_texts(COMPAT_SETS[0])
    a, b = _CountingRandom(5), _CountingRandom(5)
    engine._pick_decoration(a, "top", 0.5, prints, texts, 0.4, "quoted", 0.4)
    engine._consume_decoration_rng(b)
    assert a.calls == b.calls == 3
    # Same seed, same number of draws → both rngs are in the same state.
    assert a.random() == b.random()


# ─── 3. the new probabilities ────────────────────────────────────────────

def _classify(decoration):
    if decoration is None:
        return "none"
    return "text" if "text" in decoration else "print"


def _measure(p_print, p_text, mode="quoted", n=20000, outfit_set=COMPAT_SETS[0]):
    prints = engine.load_prints(outfit_set)
    texts = engine.load_texts(outfit_set)
    # 0.2 sits inside the formality range of the school-set prints, so the
    # print branch always has a candidate and "none" only means "no decoration".
    rng = random.Random(1234)
    counts = {"none": 0, "print": 0, "text": 0}
    for _ in range(n):
        d = engine._pick_decoration(rng, "top", 0.2, prints, texts,
                                    p_print, mode, p_text)
        counts[_classify(d)] += 1
    return {k: v / n for k, v in counts.items()}


@pytest.mark.parametrize("p_print,p_text", [
    (0.3, 0.3),
    (0.5, 0.2),
    (0.1, 0.7),
    (0.0, 0.4),
    (0.6, 0.0),
])
def test_marginal_probabilities_are_independent(p_print, p_text):
    """Below a sum of 1 each slider hits its own number, unaffected by the other."""
    got = _measure(p_print, p_text)
    assert got["print"] == pytest.approx(p_print, abs=0.02)
    assert got["text"] == pytest.approx(p_text, abs=0.02)
    assert got["none"] == pytest.approx(1.0 - p_print - p_text, abs=0.02)


def test_print_chance_survives_a_text_wish():
    """The old engine halved the print chance as soon as text was allowed."""
    alone = _measure(0.4, 0.0)
    with_text = _measure(0.4, 0.4)
    assert alone["print"] == pytest.approx(0.4, abs=0.02)
    assert with_text["print"] == pytest.approx(0.4, abs=0.02)


def test_sum_above_one_normalises_proportionally():
    got = _measure(0.8, 0.4)
    assert got["none"] == pytest.approx(0.0, abs=0.005)
    # 0.8 : 0.4 → two thirds prints, one third texts.
    assert got["print"] == pytest.approx(2 / 3, abs=0.02)
    assert got["text"] == pytest.approx(1 / 3, abs=0.02)


def test_text_mode_off_leaves_the_print_chance_whole():
    """"off" zeroes the text chance instead of stealing half the prints."""
    got = _measure(0.5, 0.9, mode="off")
    assert got["text"] == 0.0
    assert got["print"] == pytest.approx(0.5, abs=0.02)


def test_text_only_never_prints():
    got = _measure(0.0, 1.0)
    assert got["print"] == 0.0
    assert got["none"] == 0.0
    assert got["text"] == pytest.approx(1.0, abs=0.001)


def test_text_probability_reaches_the_prompt():
    """End to end: a set with texts.txt renders slogans when asked to."""
    seen = False
    for seed in range(40):
        r = generate_outfit(seed=seed, outfit_set=COMPAT_SETS[0],
                            slot_enables=ALL_ENABLED, print_probability=0.0,
                            text_mode="quoted", text_probability=1.0)
        if "text in" in r["outfit_prompt"]:
            seen = True
            break
    assert seen, "text_probability=1.0 produced no text decoration at all"


# ─── 4. node wrappers ────────────────────────────────────────────────────

def _widget_order(input_types):
    return list(input_types.get("required", {})) + list(input_types.get("optional", {}))


@pytest.mark.parametrize("import_path,cls_name", [
    ("nodes.outfit_generator", "FVM_OutfitGenerator"),
    ("nodes.smp.outfit_generator", "FVM_SMP_OutfitGenerator"),
    ("nodes.jb.outfit_block", "FVM_JB_OutfitBlock"),
])
def test_text_probability_widget_is_last(import_path, cls_name):
    """Saved workflows map widgets_values by position — the new one goes last."""
    mod = __import__(import_path, fromlist=[cls_name])
    spec = getattr(mod, cls_name).INPUT_TYPES()
    order = _widget_order(spec)
    assert order[-1] == "text_probability", order
    cfg = spec["optional"]["text_probability"]
    assert cfg[0] == "FLOAT"
    assert cfg[1]["default"] == 0.0
    assert cfg[1]["min"] == 0.0 and cfg[1]["max"] == 1.0
    assert cfg[1].get("tooltip")


def test_v1_node_passes_text_probability_through():
    from nodes.outfit_generator import FVM_OutfitGenerator
    node = FVM_OutfitGenerator()
    common = dict(outfit_set=COMPAT_SETS[0], seed=8, style_preset="general",
                  formality=0.5, coverage=0.7, enable_headwear=True,
                  enable_top=True, enable_outerwear=True, enable_bottom=True,
                  enable_footwear=True, enable_accessories=True,
                  enable_bag=True, print_probability=0.25, text_mode="quoted")
    without = node.generate(**common)
    with_text = node.generate(**common, text_probability=0.25)
    assert without != with_text
    assert "text in" in with_text[0]
    assert "text in" not in without[0]


# ─── 5. SMP region collision ─────────────────────────────────────────────

SMP_SET = "female/teen/school_day_skirt"


def test_smp_top_and_outerwear_get_separate_keys():
    """top and outerwear both map to upper_body — the jacket used to eat the shirt."""
    from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator

    rec = generate_outfit_records(
        seed=8, outfit_set=SMP_SET, formality=0.5, coverage=0.7,
        slot_enables=ALL_ENABLED, print_probability=0.25,
        text_mode="quoted", text_probability=0.25,
    )
    assert "top" in rec["garments"] and "outerwear" in rec["garments"], rec["garments"].keys()
    top_decoration = rec["garments"]["top"]["decoration"]
    assert top_decoration, "seed 8 must decorate the top for this regression to bite"

    raw, summary = FVM_SMP_OutfitGenerator().generate(
        outfit_set=SMP_SET, seed=8, style_preset="general", formality=0.5,
        coverage=0.7, enable_headwear=True, enable_top=True,
        enable_bottom=True, enable_footwear=True, enable_outerwear=True,
        enable_accessories=True, enable_bag=True, print_probability=0.25,
        text_mode="quoted", text_probability=0.25,
    )
    garments = raw["garments"]
    assert "upper_body" in garments
    assert "upper_body_outerwear" in garments
    assert garments["upper_body"]["prompt_fragment"] == rec["garments"]["top"]["prompt_fragment"]
    assert garments["upper_body_outerwear"]["prompt_fragment"] == \
        rec["garments"]["outerwear"]["prompt_fragment"]
    # The decoration must survive into the emitted dict and the summary.
    assert top_decoration in garments["upper_body"]["prompt_fragment"]
    assert top_decoration in summary


def test_smp_keeps_every_engine_slot():
    """No garment may be lost to a key collision, whatever the seed."""
    from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator
    node = FVM_SMP_OutfitGenerator()
    for seed in range(60):
        rec = generate_outfit_records(
            seed=seed, outfit_set=SMP_SET, formality=0.5, coverage=0.7,
            slot_enables=ALL_ENABLED, print_probability=0.3, text_mode="quoted",
            text_probability=0.3,
        )
        raw, _ = node.generate(
            outfit_set=SMP_SET, seed=seed, style_preset="general",
            formality=0.5, coverage=0.7, enable_headwear=True, enable_top=True,
            enable_bottom=True, enable_footwear=True, enable_outerwear=True,
            enable_accessories=True, enable_bag=True, print_probability=0.3,
            text_mode="quoted", text_probability=0.3,
        )
        assert len(raw["garments"]) == len(rec["garments"]), (
            f"seed {seed}: {len(rec['garments'])} garments in, "
            f"{len(raw['garments'])} out"
        )


def test_smp_region_hint_stays_upper_body_for_both_layers():
    """The disambiguated key must not leak into the region hint."""
    from nodes.smp.outfit_generator import FVM_SMP_OutfitGenerator
    raw, _ = FVM_SMP_OutfitGenerator().generate(
        outfit_set=SMP_SET, seed=8, style_preset="general", formality=0.5,
        coverage=0.7, enable_headwear=True, enable_top=True, enable_bottom=True,
        enable_footwear=True, enable_outerwear=True, enable_accessories=True,
        enable_bag=True, print_probability=0.25, text_mode="quoted",
        text_probability=0.25,
    )
    for key in ("upper_body", "upper_body_outerwear"):
        hint = raw["garments"][key]["region_hint"]
        assert hint["region_id"] == "upper_body", (key, hint)


def test_none_text_probability_behaves_like_zero():
    """A workflow saved before the slider existed can hand the node a null."""
    zero = generate_outfit(seed=23, outfit_set=COMPAT_SETS[0],
                           slot_enables=ALL_ENABLED, print_probability=0.5,
                           text_mode="quoted", text_probability=0.0)
    none = generate_outfit(seed=23, outfit_set=COMPAT_SETS[0],
                           slot_enables=ALL_ENABLED, print_probability=0.5,
                           text_mode="quoted", text_probability=None)
    assert zero == none


# ─── No cross-category fallback ────────────────────────────────────────
#
# Before the decoupling both decorations shared one dice roll, so the text
# branch was allowed to fall back to a print whenever it found nothing usable.
# With two independent probabilities that fallback breaks the contract: it
# produced prints for callers who had explicitly asked for none. Two paths led
# there — a set whose texts.txt declares no entry for this slot, and a set with
# no texts.txt at all.


def _decorations(outfit_set, *, print_probability, text_probability,
                 text_mode="quoted", seeds=25):
    slots = {s: True for s in ("headwear", "top", "bottom", "footwear",
                               "outerwear", "accessories", "bag")}
    for seed in range(seeds):
        record = generate_outfit_records(
            seed=seed, outfit_set=outfit_set, slot_enables=slots,
            print_probability=print_probability,
            text_probability=text_probability, text_mode=text_mode)
        for garment in record["garments"].values():
            decoration = garment.get("decoration")
            if decoration:
                yield decoration


def _sample_sets(count=12):
    """A stable slice of real sets — data-independent, so filling in a
    missing texts.txt later cannot silently invalidate the assertion."""
    return get_available_sets()[::max(1, len(get_available_sets()) // count)]


def test_zero_print_probability_never_yields_a_print():
    """print_probability=0 means no prints, whatever the text branch does."""
    for outfit_set in _sample_sets():
        prints = [d for d in _decorations(outfit_set, print_probability=0.0,
                                          text_probability=0.9)
                  if "text in" not in d]
        assert prints == [], f"{outfit_set} leaked prints: {prints[:3]}"


def test_set_without_texts_yields_nothing_rather_than_prints():
    """A pure text request against a set that has no texts.txt stays plain."""
    sets_without = [s for s in get_available_sets() if not load_texts(s)]
    if not sets_without:
        pytest.skip("every set has texts.txt entries")
    for outfit_set in sets_without[:8]:
        assert list(_decorations(outfit_set, print_probability=0.0,
                                 text_probability=0.9)) == []


def test_zero_text_probability_never_yields_a_text():
    """The mirror case — text_probability=0 must not produce lettering."""
    for outfit_set in _sample_sets():
        texts = [d for d in _decorations(outfit_set, print_probability=0.9,
                                         text_probability=0.0)
                 if "text in" in d]
        assert texts == [], f"{outfit_set} leaked text: {texts[:3]}"
