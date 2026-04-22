"""Smoke + contract tests for ``web/templates/you.html``.

Scope (NEXT_TASKS.md Week 10 task 1 acceptance):

- ``you.html`` renders all four sections (WHEN YOU'RE AT YOUR BEST,
  HOW YOU WRITE, YOUR ROUTINES, DRIFTING) plus the header.
- Empty-state copy renders when each section is empty.
- Populated sections render the appropriate per-row slots.
- No progress bars / pie charts / mood bars — DESIGN.md § "Principles:
  No horoscope".
- End-to-end rendering via :func:`web.rendering.render` (no mocking).

These are structural assertions — no full-HTML snapshots — so visual
polish tweaks do not churn this file.
"""

from __future__ import annotations

import re

from web.rendering import render

# A minimal self-portrait shape — mirrors api.schemas.YouOut.
_EMPTY_YOU: dict = {
    "observed_months": 0,
    "interactions_count": 0,
    "confidence_pct": 0,
    "when_at_best": [],
    "how_you_write": [],
    "your_routines": [],
    "drifting": [],
}


def _full_you() -> dict:
    """A fixture roughly matching the DESIGN.md wireframe."""
    return {
        "observed_months": 4,
        "interactions_count": 312,
        "confidence_pct": 68,
        "when_at_best": [
            "09:00-11:00 - deep work",
            "14:00-16:00 - afternoon focus",
            "19:00-21:00 - evening write",
        ],
        "how_you_write": [
            {
                "audience": "Mom",
                "tone": "warm, short",
                "formality": 0.2,
                "sample_size": 14,
            },
            {
                "audience": "Work",
                "tone": "direct, bulleted",
                "formality": 0.7,
                "sample_size": 48,
            },
        ],
        "your_routines": [
            {
                "name": "plan_week",
                "detected": True,
                "description": "plan your week",
                "confidence": 0.82,
                "sample_size": 6,
            },
        ],
        "drifting": [
            {
                "contact_id": "mike",
                "name": "Mike",
                "days_since_last": 18,
                "usual_cadence_days": 7,
            },
            {
                "contact_id": "dad",
                "name": "Dad",
                "days_since_last": 9,
                "usual_cadence_days": 5,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Base wiring
# ---------------------------------------------------------------------------


def test_you_page_declares_active_tab() -> None:
    """The page must flag itself as the You tab to base.html."""
    html = render("you.html", {"you": _EMPTY_YOU})
    assert 'data-active-tab="you"' in html


def test_you_page_renders_all_four_sections() -> None:
    """Every named section must be present regardless of data."""
    html = render("you.html", {"you": _EMPTY_YOU})
    assert 'data-section="you-header"' in html
    assert 'data-section="when-at-best"' in html
    assert 'data-section="how-you-write"' in html
    assert 'data-section="your-routines"' in html
    assert 'data-section="drifting"' in html


def test_you_page_has_no_progress_or_pie_chart_markup() -> None:
    """DESIGN.md § 'No horoscope': no bars, no pies."""
    html = render("you.html", {"you": _full_you()})
    assert "progressbar" not in html
    # SVG-based decorative charts are banned for Phase 1.
    assert "<svg" not in html
    assert 'role="progressbar"' not in html


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def test_you_header_renders_observed_interactions_confidence() -> None:
    html = render("you.html", {"you": _full_you()})
    # Each datum lives in its own data-slot so tests and DOM tooling can
    # address them without regex-scraping the header text.
    assert 'data-slot="observed-months">4<' in html
    assert 'data-slot="interactions-count">312<' in html
    assert 'data-slot="confidence-pct">68<' in html
    # The literal "Observed" word anchors the header copy.
    assert "Observed" in html
    assert "interactions" in html
    assert "confidence" in html


def test_you_header_uses_singular_month_at_one() -> None:
    html = render("you.html", {"you": {**_EMPTY_YOU, "observed_months": 1}})
    # "Observed 1 month" not "months" when exactly one.
    assert 'data-slot="observed-months">1<' in html
    # The word "month" (not "months") must follow the observed-months slot.
    match = re.search(r'data-slot="observed-months">1</span>\s*(month)\b', html)
    assert match is not None, "expected singular 'month' label after observed-months slot"


# ---------------------------------------------------------------------------
# WHEN YOU'RE AT YOUR BEST
# ---------------------------------------------------------------------------


def test_when_at_best_renders_each_window() -> None:
    html = render("you.html", {"you": _full_you()})
    assert "09:00-11:00 - deep work" in html
    assert "14:00-16:00 - afternoon focus" in html
    assert "19:00-21:00 - evening write" in html
    # Each row is addressable via data-slot.
    assert html.count('data-slot="best-row"') == 3


def test_when_at_best_empty_state() -> None:
    html = render("you.html", {"you": _EMPTY_YOU})
    assert 'data-empty="when-at-best"' in html
    assert "Not enough to call it a pattern yet." in html


# ---------------------------------------------------------------------------
# HOW YOU WRITE
# ---------------------------------------------------------------------------


def test_how_you_write_renders_audience_and_tone() -> None:
    html = render("you.html", {"you": _full_you()})
    assert 'data-slot="persona-audience">Mom<' in html
    assert "warm, short" in html
    assert 'data-slot="persona-audience">Work<' in html
    assert "direct, bulleted" in html
    # Sample size is rendered in the right-aligned monospace slot.
    assert "14 samples" in html
    assert "48 samples" in html


def test_how_you_write_empty_state() -> None:
    html = render("you.html", {"you": _EMPTY_YOU})
    assert 'data-empty="how-you-write"' in html
    assert "No writing style detected yet." in html


# ---------------------------------------------------------------------------
# YOUR ROUTINES
# ---------------------------------------------------------------------------


def test_your_routines_renders_detected_routine() -> None:
    html = render("you.html", {"you": _full_you()})
    assert 'data-slot="routine-description"' in html
    assert "plan your week" in html
    assert "plan_week" in html
    assert "6 occurrences" in html


def test_your_routines_empty_state_at_section_level() -> None:
    html = render("you.html", {"you": _EMPTY_YOU})
    assert 'data-empty="your-routines"' in html
    assert "No routine detected yet." in html


def test_your_routines_per_row_empty_state_with_detected_false() -> None:
    """A routine row with ``detected=False`` is an in-list empty marker."""
    you = {
        **_EMPTY_YOU,
        "your_routines": [
            {
                "name": "morning_review",
                "detected": False,
                "description": None,
                "confidence": 0.0,
                "sample_size": 0,
            }
        ],
    }
    html = render("you.html", {"you": you})
    # Section-level empty state should NOT render when the list is non-empty.
    assert 'data-empty="your-routines"' not in html
    assert 'data-slot="routine-empty"' in html
    assert "morning_review" in html
    assert "No routine detected yet" in html


# ---------------------------------------------------------------------------
# DRIFTING
# ---------------------------------------------------------------------------


def test_drifting_renders_contact_rows_with_days_and_usual() -> None:
    html = render("you.html", {"you": _full_you()})
    assert 'data-contact-id="mike"' in html
    assert 'data-slot="drifting-name">Mike<' in html
    assert "18 days" in html
    assert "(usual 7)" in html
    # Second contact with singular "day" not "days"... ours is 9 so plural.
    assert 'data-contact-id="dad"' in html
    assert "9 days" in html
    assert "(usual 5)" in html


def test_drifting_uses_singular_day_when_one() -> None:
    you = {
        **_EMPTY_YOU,
        "drifting": [
            {
                "contact_id": "ann",
                "name": "Ann",
                "days_since_last": 1,
                "usual_cadence_days": 1,
            }
        ],
    }
    html = render("you.html", {"you": you})
    assert "1 day" in html
    assert "1 days" not in html


def test_drifting_empty_state() -> None:
    html = render("you.html", {"you": _EMPTY_YOU})
    assert 'data-empty="drifting"' in html
    assert "Nobody's slipping right now." in html


# ---------------------------------------------------------------------------
# End-to-end smoke — nav wiring
# ---------------------------------------------------------------------------


def test_you_page_marks_you_tab_active_in_nav() -> None:
    html = render("you.html", {"you": _EMPTY_YOU})
    # The nav anchor for "You" should be aria-current="page"; others should not.
    you_anchor = html.split('data-tab="you"', 1)[1].split("</a>", 1)[0]
    assert 'aria-current="page"' in you_anchor
    now_anchor = html.split('data-tab="now"', 1)[1].split("</a>", 1)[0]
    assert 'aria-current="page"' not in now_anchor
