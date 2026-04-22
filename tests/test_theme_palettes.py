# -*- coding: utf-8 -*-
"""Unit tests for ``ui_widgets.theme``: palettes + runtime theme switching.

Covers the public API added in Sprint 2 of the web migration:
``PALETTES``, ``apply_theme``, ``rebuild_css``, ``get_active_theme``,
``chip``/``badge`` kind ``"accent"`` and graceful degradation of
unknown kinds.

Pure-Python — no ipywidgets dependency, so these run under the stock
``tests`` matrix in CI (3.11 / 3.12 / 3.13).
"""
from __future__ import annotations

import pytest

from ui_widgets import theme


_CANONICAL_PALETTE_KEYS: frozenset[str] = frozenset({
    "bg", "panel", "panel2", "entry_bg",
    "fg", "muted", "muted2",
    "border", "border2",
    "accent", "accent2", "accent3",
    "hover", "success", "warning", "error", "select",
})

_ALL_KINDS: tuple[str, ...] = ("default", "accent", "ok", "warn", "err", "info")


@pytest.fixture(autouse=True)
def _reset_active_theme():
    """Restore the module default after every test so ordering is safe."""
    before = theme.get_active_theme()
    yield
    theme.apply_theme(before)


# ── Palette schema ────────────────────────────────────────────────────


def test_palettes_have_three_themes():
    assert set(theme.PALETTES) == {"dark-teal", "paper", "amber-crt"}


def test_every_palette_has_canonical_schema():
    for name, palette in theme.PALETTES.items():
        assert set(palette) == _CANONICAL_PALETTE_KEYS, (
            f"palette {name!r}: unexpected keys "
            f"{set(palette) ^ _CANONICAL_PALETTE_KEYS}"
        )


def test_palette_values_are_hex_strings():
    for name, palette in theme.PALETTES.items():
        for key, value in palette.items():
            assert isinstance(value, str), f"{name}/{key}: {value!r} not str"
            assert value.startswith("#"), f"{name}/{key}: {value!r} not hex"
            assert len(value) in (4, 7, 9), f"{name}/{key}: {value!r} bad hex len"


# ── apply_theme / rebuild_css / get_active_theme ──────────────────────


def test_default_theme_is_dark_teal():
    assert theme.get_active_theme() == "dark-teal"


def test_apply_theme_switches_active():
    theme.apply_theme("paper")
    assert theme.get_active_theme() == "paper"
    theme.apply_theme("amber-crt")
    assert theme.get_active_theme() == "amber-crt"


def test_apply_theme_unknown_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        theme.apply_theme("neon-vapor")
    assert "neon-vapor" in str(exc_info.value)
    # available palettes are listed in the message
    for known in theme.PALETTES:
        assert known in str(exc_info.value)


def test_apply_theme_with_none_raises():
    # dict-lookup for a non-hashable is a TypeError, a non-string name
    # (e.g. None) must produce a clear error; ValueError is expected from
    # the explicit `name not in PALETTES` guard.
    with pytest.raises(ValueError):
        theme.apply_theme(None)  # type: ignore[arg-type]


def test_rebuild_css_reflects_active_palette():
    theme.apply_theme("paper")
    css_paper = theme.rebuild_css()
    assert theme.PALETTES["paper"]["bg"] in css_paper
    assert theme.PALETTES["paper"]["accent"] in css_paper

    theme.apply_theme("amber-crt")
    css_crt = theme.rebuild_css()
    assert theme.PALETTES["amber-crt"]["bg"] in css_crt
    # paper-specific background must not leak into crt
    assert theme.PALETTES["paper"]["bg"] not in css_crt


def test_rebuild_css_is_self_contained_block():
    css = theme.rebuild_css()
    assert css.lstrip().startswith("<style>")
    assert css.rstrip().endswith("</style>")


# ── Theme-aware CSS classes added in Sprint 2.2–2.3 ───────────────────


@pytest.mark.parametrize("selector", [
    ".brt-chip.accent",
    ".brt-badge-accent",
    ".brt-field-label",
    ".brt-sep",
    ".brt-theme-switcher",
    ".brt-header-title .brand",
    ".brt-header-title .sub",
])
@pytest.mark.parametrize("palette", ["dark-teal", "paper", "amber-crt"])
def test_new_css_selectors_present_in_every_palette(selector, palette):
    theme.apply_theme(palette)
    assert selector in theme.rebuild_css()


# ── chip() / badge() — kind handling ──────────────────────────────────


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_chip_renders_all_canonical_kinds(kind):
    html = theme.chip("x", kind)
    assert html.startswith("<span")
    assert ">x<" in html
    if kind == "default":
        # default maps to class "brt-chip" without a suffix
        assert "class='brt-chip'" in html
    else:
        assert kind in html
        assert "brt-chip" in html


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_badge_renders_all_canonical_kinds(kind):
    html = theme.badge("x", kind)
    assert html.startswith("<span")
    assert ">x<" in html
    if kind == "default":
        assert "class='brt-badge'" in html
    else:
        assert f"brt-badge-{kind}" in html


def test_chip_unknown_kind_degrades_to_default():
    # Stale call-sites passing a retired kind must not crash the UI.
    fallback = theme.chip("x", "default")
    assert theme.chip("x", "retro-wave") == fallback


def test_badge_unknown_kind_degrades_to_default():
    fallback = theme.badge("x", "default")
    assert theme.badge("x", "retro-wave") == fallback


def test_chip_accent_kind_is_distinct_from_info():
    # Sanity: "accent" (new in Sprint 2.2) must not collide with the
    # older teal "info" kind at the class level.
    accent = theme.chip("x", "accent")
    info = theme.chip("x", "info")
    assert "accent" in accent and "info" not in accent
    assert "info" in info and "accent" not in info
