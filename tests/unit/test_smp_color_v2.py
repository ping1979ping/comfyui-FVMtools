"""P2 — tests for FVM_SMP_ColorGenerator."""

from nodes.smp.color_generator import FVM_SMP_ColorGenerator


def _gen(**kwargs):
    args = dict(
        seed=42, num_colors=5, harmony_type="auto", style_preset="general",
        vibrancy=0.5, contrast=0.5, warmth=0.5,
    )
    args.update(kwargs)
    return FVM_SMP_ColorGenerator().generate(**args)


def test_returns_dict_and_summary():
    palette, summary = _gen()
    assert isinstance(palette, dict)
    assert isinstance(summary, str)


def test_garment_colors_populated():
    palette, _ = _gen(num_colors=5)
    gc = palette["garment_colors"]
    # At minimum primary + secondary roles are emitted by the engine.
    assert "primary" in gc and gc["primary"]
    assert "secondary" in gc and gc["secondary"]


def test_atmosphere_colors_present():
    palette, _ = _gen()
    atm = palette["atmosphere_colors"]
    assert "ambient_light" in atm and atm["ambient_light"]
    assert "shadow_tone" in atm and atm["shadow_tone"]


def test_raw_tokens_built_from_roles():
    palette, _ = _gen()
    tokens = palette["raw_tokens"]
    assert "#primary#" in tokens
    assert "#secondary#" in tokens
    assert "#ambient_light#" in tokens
    assert "#shadow_tone#" in tokens
    # Token values must be non-empty strings
    for t, v in tokens.items():
        assert isinstance(v, str) and v


def test_warmth_drives_atmosphere_phrasing():
    warm, _ = _gen(seed=1, warmth=0.95)
    cool, _ = _gen(seed=1, warmth=0.05)
    # Warm vs cool seeds pull from disjoint phrase pools — labels must differ
    assert warm["atmosphere_colors"]["ambient_light"] != cool["atmosphere_colors"]["ambient_light"]
    assert warm["color_tone"] == "warm"
    assert cool["color_tone"] == "cool"


def test_seed_determinism():
    a, _ = _gen(seed=7)
    b, _ = _gen(seed=7)
    assert a == b


def test_different_seed_produces_different_palette():
    a, _ = _gen(seed=1)
    b, _ = _gen(seed=2)
    assert a != b


def test_node_metadata():
    assert FVM_SMP_ColorGenerator.CATEGORY.startswith("FVM Tools/SMP")
    assert FVM_SMP_ColorGenerator.RETURN_TYPES == ("COLOR_PALETTE_DICT", "STRING")
