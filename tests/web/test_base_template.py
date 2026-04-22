"""Smoke + contract tests for ``web/templates/base.html`` and ``tokens.css``.

Scope (per NEXT_TASKS.md Week 9 acceptance):

- Template renders without Jinja errors using empty context.
- Template renders with the four supported ``active_tab`` values.
- ``tokens.css`` is loaded from the expected path.
- HTMX core + ws extension + Tailwind + Lucide are all wired in.
- A11y primitives are present: ``<nav aria-label>``, ``<main>``, skip
  link, ``aria-live`` on the toast region.
- All four primary tabs render with the correct ``aria-selected`` and
  active-tab underline for the currently selected tab.
- The CSS token file contains every custom property DESIGN.md promises
  (color, type, spacing, elevation, radius, motion) and a
  ``prefers-reduced-motion`` media block.

These assertions are deliberately structural: we do not snapshot the
rendered HTML (too fragile during the Week-9/10 UI build-out) but we
do pin the pieces downstream templates depend on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from web.rendering import DEFAULT_TEMPLATES_DIR, get_environment, render

_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "web" / "static"
_TOKENS_CSS = _STATIC_DIR / "tokens.css"


@pytest.fixture
def base_html() -> str:
    """Render the bare base template with no context (default path)."""
    return render("base.html")


def test_render_with_no_context_produces_valid_html(base_html: str) -> None:
    """Empty context must still render — defaults fill the slots."""
    assert base_html.startswith("<!DOCTYPE html>")
    assert "</html>" in base_html
    assert "<title>Life OS</title>" in base_html


def test_tokens_css_linked_before_tailwind(base_html: str) -> None:
    """tokens.css must come before Tailwind so the palette wins."""
    tokens_idx = base_html.find('href="/static/tokens.css"')
    tailwind_idx = base_html.find("cdn.tailwindcss.com")
    assert tokens_idx != -1, "tokens.css link missing"
    assert tailwind_idx != -1, "Tailwind CDN script missing"
    assert tokens_idx < tailwind_idx


def test_htmx_and_ws_extension_loaded(base_html: str) -> None:
    assert "htmx.org@" in base_html
    assert "htmx-ext-ws" in base_html
    assert 'hx-ext="ws"' in base_html


def test_lucide_icon_loader_wired(base_html: str) -> None:
    assert "lucide@latest" in base_html
    assert "lucide.createIcons" in base_html
    assert "htmx:afterSwap" in base_html


def test_accessibility_primitives_present(base_html: str) -> None:
    assert 'aria-label="Primary"' in base_html
    assert '<main id="main"' in base_html
    assert 'class="skip-link"' in base_html
    assert 'href="#main"' in base_html
    # Toast region is a polite live region (Undo pattern lands later).
    assert 'aria-live="polite"' in base_html


def test_default_active_tab_is_now(base_html: str) -> None:
    """With no active_tab in context the Now tab must be selected."""
    assert 'data-active-tab="now"' in base_html
    # Selected link carries aria-selected="true" and aria-current="page"
    now_fragment = base_html.split('data-tab="now"')[1].split("</a>")[0]
    assert 'aria-selected="true"' in now_fragment
    assert 'aria-current="page"' in now_fragment


@pytest.mark.parametrize(
    ("active_tab", "expected_label"),
    [
        ("now", "Now"),
        ("you", "You"),
        ("people", "People"),
        ("settings", "Settings"),
    ],
)
def test_active_tab_parametrized(active_tab: str, expected_label: str) -> None:
    """Every valid tab slug selects the correct nav entry."""
    html = render("base.html", {"active_tab": active_tab})
    # The selected tab's link fragment must contain aria-selected="true".
    fragment = html.split(f'data-tab="{active_tab}"')[1].split("</a>")[0]
    assert 'aria-selected="true"' in fragment
    assert expected_label in fragment


def test_non_active_tabs_marked_unselected() -> None:
    html = render("base.html", {"active_tab": "you"})
    # The "now" tab, which is not active, must carry aria-selected="false".
    now_fragment = html.split('data-tab="now"')[1].split("</a>")[0]
    assert 'aria-selected="false"' in now_fragment
    # And the inactive tab must NOT carry the primary-action underline.
    assert "bg-[color:var(--primary-action)]" not in now_fragment


def test_date_and_time_slots_render_when_provided() -> None:
    html = render(
        "base.html",
        {"now_date": "Wed, Apr 22", "now_time": "10:24"},
    )
    assert "Wed, Apr 22" in html
    assert "10:24" in html
    assert 'data-slot="now-date"' in html
    assert 'data-slot="now-time"' in html


def test_date_slot_absent_when_missing() -> None:
    html = render("base.html")
    assert 'data-slot="now-date"' not in html
    assert 'data-slot="now-time"' not in html


def test_blocks_are_extendable(tmp_path: Path) -> None:
    """Child templates must be able to override title and content."""
    # Copy the real base into a temp dir alongside a child that extends it.
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "base.html").write_text((DEFAULT_TEMPLATES_DIR / "base.html").read_text())
    (templates_dir / "child.html").write_text(
        """{% extends "base.html" %}
{% block title %}Custom Title{% endblock %}
{% block content %}<section id="custom">hello</section>{% endblock %}
"""
    )

    # Bypass the cache by passing an explicit templates_dir.
    env = get_environment(str(templates_dir))
    out = env.get_template("child.html").render()
    assert "<title>Custom Title</title>" in out
    assert '<section id="custom">hello</section>' in out


# ---------------------------------------------------------------------------
# tokens.css contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokens_css_text() -> str:
    return _TOKENS_CSS.read_text()


def test_tokens_css_file_exists() -> None:
    assert _TOKENS_CSS.exists(), f"tokens.css not found at {_TOKENS_CSS}"


def test_tokens_css_defines_color_tokens(tokens_css_text: str) -> None:
    """Every color token DESIGN.md promises must be present."""
    required = [
        "--bg-base",
        "--bg-raised",
        "--bg-raised-hover",
        "--bg-sunken",
        "--bg-overlay",
        "--text-primary",
        "--text-secondary",
        "--text-tertiary",
        "--text-disabled",
        "--primary-action",
        "--primary-action-hover",
        "--primary-action-pressed",
        "--success",
        "--warning",
        "--error",
        "--info",
        "--border-subtle",
        "--border-strong",
        "--border-focus",
        "--draft-bg",
        "--draft-border",
    ]
    for token in required:
        assert f"{token}:" in tokens_css_text, f"missing {token}"


def test_tokens_css_defines_type_tokens(tokens_css_text: str) -> None:
    for token in [
        "--font-display",
        "--font-text",
        "--font-mono",
        "--t-11",
        "--t-13",
        "--t-15",
        "--t-17",
        "--t-22",
        "--t-28",
        "--lh-tight",
        "--lh-snug",
        "--lh-default",
        "--lh-loose",
        "--w-regular",
        "--w-medium",
        "--w-semibold",
        "--w-bold",
        "--ls-tight",
        "--ls-normal",
        "--ls-caps",
    ]:
        assert f"{token}:" in tokens_css_text, f"missing {token}"


def test_tokens_css_defines_spacing_tokens(tokens_css_text: str) -> None:
    for token in [
        "--s-0",
        "--s-1",
        "--s-2",
        "--s-3",
        "--s-4",
        "--s-5",
        "--s-6",
        "--s-8",
        "--s-10",
        "--s-12",
        "--s-16",
    ]:
        assert f"{token}:" in tokens_css_text, f"missing {token}"


def test_tokens_css_defines_elevation_and_radius(tokens_css_text: str) -> None:
    for token in [
        "--elev-0",
        "--elev-1",
        "--elev-2",
        "--elev-focus",
        "--elev-modal",
        "--r-sm",
        "--r-md",
        "--r-lg",
        "--r-pill",
    ]:
        assert f"{token}:" in tokens_css_text, f"missing {token}"


def test_tokens_css_defines_motion_tokens(tokens_css_text: str) -> None:
    for token in ["--motion-micro", "--motion-swap", "--motion-modal"]:
        assert f"{token}:" in tokens_css_text, f"missing {token}"


def test_tokens_css_respects_prefers_reduced_motion(tokens_css_text: str) -> None:
    assert "prefers-reduced-motion: reduce" in tokens_css_text
    # Inside that block, motion tokens are collapsed to ~1ms.
    reduced = tokens_css_text.split("prefers-reduced-motion: reduce")[1]
    assert "--motion-micro: 1ms" in reduced
    assert "--motion-swap: 1ms" in reduced
    assert "--motion-modal: 1ms" in reduced


def test_tokens_css_sets_color_scheme_dark(tokens_css_text: str) -> None:
    """Base <html> must signal dark scheme so UA widgets theme correctly."""
    assert "color-scheme: dark" in tokens_css_text


def test_tokens_css_exposes_focus_ring_pattern(tokens_css_text: str) -> None:
    """Keyboard-only focus ring uses --elev-focus (WCAG AA requirement)."""
    assert ":focus-visible" in tokens_css_text
    assert "box-shadow: var(--elev-focus)" in tokens_css_text


def test_tokens_css_skip_link_style_present(tokens_css_text: str) -> None:
    """Skip link is visually hidden until focused."""
    assert ".skip-link" in tokens_css_text
    assert ".skip-link:focus-visible" in tokens_css_text
