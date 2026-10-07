"""``{a|b|c}`` brackets in location list entries."""

from core.location_engine import _resolve_braces, _split_fields


def test_split_ignores_pipes_inside_braces():
    parts = _split_fields('wall sign "{EXIT|ENTRANCE}" | 0.5 | 0.1-0.2 | red {metal|enamel} plate')
    assert parts == ['wall sign "{EXIT|ENTRANCE}"', "0.5", "0.1-0.2", "red {metal|enamel} plate"]


def test_split_without_braces_unchanged():
    assert _split_fields("a | 0.5 | - | -") == ["a", "0.5", "-", "-"]


def test_resolve_picks_every_option_and_is_deterministic():
    text = 'door sign "{EXIT|ENTRANCE|WAY OUT}"'
    seen = {_resolve_braces(text, k, "props") for k in range(60)}
    assert seen == {f'door sign "{w}"' for w in ("EXIT", "ENTRANCE", "WAY OUT")}
    assert _resolve_braces(text, 7, "props") == _resolve_braces(text, 7, "props")


def test_empty_option_and_spacing_tidy():
    outs = {_resolve_braces('sign{| reading "{EXIT|OPEN}"} , red', k, "props") for k in range(60)}
    assert "sign, red" in outs
    assert all("  " not in o and " ," not in o for o in outs)


def test_plain_text_passthrough():
    assert _resolve_braces("plain name", 1, "props") == "plain name"
    assert _resolve_braces(None, 1, "props") is None
