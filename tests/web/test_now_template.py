"""Smoke + contract tests for ``web/templates/now.html`` and partials.

Scope (NEXT_TASKS.md Week 9 task 1 acceptance):

- ``now.html`` renders all three sections (NOW · UP NEXT · DONE TODAY)
  including empty-state copy when each bucket is empty.
- The Moment card primitive renders the three documented states:
  default, draft-pending, expanded-evidence.
- The card emits the exact HTMX wiring the next task assumes
  (accept/dismiss hx-post URLs, evidence hx-get URL).
- "Why am I seeing this" microcopy is present per insight type.
- The evidence partial renders both the list and the empty state.

These are structural assertions — no full HTML snapshots — so the
visual polish of Week 9 task 2 (HTMX wiring) doesn't churn this file.
"""

from __future__ import annotations

import uuid

import pytest

from core.moment.types import (
    Action,
    ActionKind,
    InsightType,
    Moment,
    MomentState,
)
from web.rendering import render

REF_NOW = 1_777_204_800  # 2026-04-22T12:00:00Z, mirrors test_routes_now.py


def _make_moment(
    *,
    insight: str = "ping your sister",
    insight_type: InsightType = InsightType.CADENCE,
    action_kind: ActionKind = ActionKind.DRAFT_MESSAGE,
    body: str | None = "Hey — been a minute. How are you?",
    state: MomentState = MomentState.SUGGESTED,
    evidence: list[str] | None = None,
    scheduled_for: int | None = None,
    moment_id: str | None = None,
) -> Moment:
    """Build a Moment with the typical Now-tab shape."""
    params: dict[str, str] = {}
    if body is not None:
        params["body"] = body
    return Moment(
        id=moment_id or str(uuid.uuid4()),
        created_at=REF_NOW,
        expires_at=REF_NOW + 3 * 24 * 3600,
        insight=insight,
        evidence_hash=f"hash-{uuid.uuid4().hex[:8]}",
        proposed_action=Action(kind=action_kind, params=params),
        source_insight_type=insight_type,
        scheduled_for=scheduled_for,
        evidence=list(evidence) if evidence is not None else ["evt-1", "evt-2"],
        state=state,
    )


# ---------------------------------------------------------------------------
# now.html
# ---------------------------------------------------------------------------


def test_now_page_renders_three_sections_with_empty_buckets() -> None:
    """Empty state must still render every named section + warm copy."""
    html = render("now.html", {"pending": [], "scheduled": [], "done": []})
    assert 'data-section="now"' in html
    assert 'data-section="up-next"' in html
    assert 'data-section="done-today"' in html
    assert 'data-empty="now"' in html
    assert "Nothing pressing right now." in html
    assert 'data-empty="up-next"' in html
    assert "Nothing queued." in html
    # DONE TODAY is rendered inside <details>, collapsed by default.
    assert "<details>" in html
    assert "Done Today" in html


def test_now_page_renders_pending_card_in_now_section() -> None:
    """A SUGGESTED moment is rendered as a full Moment card."""
    moment = _make_moment(insight="water the plants")
    html = render("now.html", {"pending": [moment], "scheduled": [], "done": []})
    assert 'class="moment-card' in html
    assert "water the plants" in html
    # Up Next & Done Today are still empty.
    assert 'data-empty="up-next"' in html


def test_now_page_renders_scheduled_compact_list() -> None:
    """Scheduled bucket renders as compact list, NOT a Moment card."""
    moment = _make_moment(insight="call dad", scheduled_for=REF_NOW + 7200)
    html = render(
        "now.html",
        {"pending": [], "scheduled": [moment], "done": []},
    )
    assert "call dad" in html
    # Up-next list does not use the .moment-card primitive (compact).
    upnext_fragment = html.split('data-section="up-next"')[1].split("</section>")[0]
    assert "moment-card" not in upnext_fragment
    assert 'data-slot="scheduled-ts"' in upnext_fragment


def test_now_page_renders_done_today_count_and_collapsed() -> None:
    """DONE TODAY shows a count and remains collapsed (no `open` attr)."""
    done = [
        _make_moment(insight="reply to mike", state=MomentState.DONE, evidence=[]),
        _make_moment(insight="archive newsletter", state=MomentState.DONE, evidence=[]),
    ]
    html = render("now.html", {"pending": [], "scheduled": [], "done": done})
    assert "(2)" in html
    # Collapsed by default — no `<details open` should be present.
    assert "<details open" not in html
    assert "reply to mike" in html


def test_now_page_default_active_tab_is_now() -> None:
    """The page must declare itself as the Now tab to base.html."""
    html = render("now.html", {"pending": [], "scheduled": [], "done": []})
    assert 'data-active-tab="now"' in html


# ---------------------------------------------------------------------------
# partials/moment_card.html — three documented render states
# ---------------------------------------------------------------------------


def test_moment_card_default_state_renders_insight_at_22pt() -> None:
    """Default state: 22pt headline + draft block + action row."""
    moment = _make_moment(insight="check on grandma", body="Hey grandma, just checking in.")
    html = render("partials/moment_card.html", {"moment": moment})
    # Insight uses the 22pt display font.
    assert "text-[length:var(--t-22)]" in html
    assert "font-display" in html
    assert "check on grandma" in html
    # Static draft block is shown (NOT the pending loader).
    assert 'data-slot="draft"' in html
    assert 'data-slot="draft-pending"' not in html
    assert "Hey grandma, just checking in." in html
    # Three action buttons: accept, snooze, dismiss; plus Edit for messages.
    assert 'data-action="accept"' in html
    assert 'data-action="snooze"' in html
    assert 'data-action="dismiss"' in html
    assert 'data-action="edit"' in html
    # HTMX wiring matches the Week 9 task-2 expectation verbatim.
    assert f'hx-post="/api/moments/{moment.id}/accept"' in html
    assert f'hx-post="/api/moments/{moment.id}/dismiss"' in html
    # Evidence reveal pre-wired.
    assert f'hx-get="/api/moments/{moment.id}/evidence"' in html
    assert 'aria-expanded="false"' in html


def test_moment_card_draft_pending_state_shows_loader() -> None:
    """draft_pending=True replaces the static draft with the loader."""
    moment = _make_moment(body=None)  # No body — pending case.
    html = render(
        "partials/moment_card.html",
        {"moment": moment, "draft_pending": True},
    )
    assert 'data-slot="draft-pending"' in html
    assert "Writing a draft" in html
    # And the static draft slot is NOT rendered.
    assert 'data-slot="draft"' not in html.replace('data-slot="draft-pending"', "")


def test_moment_card_no_draft_when_action_has_no_body() -> None:
    """Non-message actions (e.g. NUDGE, NOTE) render no draft block."""
    moment = _make_moment(action_kind=ActionKind.NUDGE, body=None)
    html = render("partials/moment_card.html", {"moment": moment})
    assert 'data-slot="draft"' not in html
    assert 'data-slot="draft-pending"' not in html
    # Edit button only appears for message kinds.
    assert 'data-action="edit"' not in html
    # Primary label matches the kind.
    assert ">Nudge<" in html


def test_moment_card_expanded_evidence_renders_inline() -> None:
    """expanded_evidence=True renders the evidence partial inline."""
    moment = _make_moment(evidence=["evt-1", "evt-2", "evt-3"])
    html = render(
        "partials/moment_card.html",
        {"moment": moment, "expanded_evidence": True},
    )
    assert 'aria-expanded="true"' in html
    assert 'data-slot="evidence-list"' in html
    # All three evidence refs landed inline.
    assert "evt-1" in html
    assert "evt-2" in html
    assert "evt-3" in html


@pytest.mark.parametrize(
    ("insight_type", "fragment"),
    [
        # NB: Jinja autoescape rewrites apostrophes to &#39; in the rendered
        # HTML — fragments here are chosen to be invariant under autoescape.
        (InsightType.CADENCE, "your usual cadence shifted"),
        (InsightType.RELATIONSHIP, "connected in a while"),
        (InsightType.TEMPORAL, "your typical schedule"),
        (InsightType.SPATIAL, "place you visit"),
        (InsightType.COMM_TEMPLATE, "matches a template"),
        (InsightType.ROUTINE, "fits a routine"),
    ],
)
def test_moment_card_why_microcopy_per_insight_type(insight_type: InsightType, fragment: str) -> None:
    """Each insight type has a dedicated 'Why am I seeing this' microcopy."""
    moment = _make_moment(insight_type=insight_type)
    html = render("partials/moment_card.html", {"moment": moment})
    assert 'data-slot="why-microcopy"' in html
    assert fragment in html


def test_moment_card_aria_label_summarises_insight_and_action() -> None:
    """DESIGN.md § WCAG: aria-label summarises insight + primary action."""
    moment = _make_moment(insight="reply to mike", action_kind=ActionKind.SEND_MESSAGE)
    html = render("partials/moment_card.html", {"moment": moment})
    assert 'aria-label="reply to mike — Send"' in html


def test_moment_card_evidence_count_in_button() -> None:
    """The evidence button shows the count so the user knows what's behind."""
    moment = _make_moment(evidence=["a", "b", "c", "d"])
    html = render("partials/moment_card.html", {"moment": moment})
    assert "Evidence (4)" in html


# ---------------------------------------------------------------------------
# partials/evidence.html
# ---------------------------------------------------------------------------


def test_evidence_partial_lists_each_reference() -> None:
    moment = _make_moment(evidence=["evt-100", "evt-200"])
    html = render("partials/evidence.html", {"moment": moment})
    assert 'data-slot="evidence-list"' in html
    assert "evt-100" in html
    assert "evt-200" in html


def test_evidence_partial_empty_state() -> None:
    moment = _make_moment(evidence=[])
    html = render("partials/evidence.html", {"moment": moment})
    assert 'data-slot="evidence-empty"' in html
    assert "No source events recorded." in html
