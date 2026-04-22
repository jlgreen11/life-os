"""Smoke + contract tests for ``web/templates/people.html``,
``web/templates/contact_dossier.html``, and ``partials/people_results.html``.

Scope (NEXT_TASKS.md Week 10 task 1 acceptance):

- ``people.html`` renders header + search input + results container.
- HTMX search wiring is exact (``hx-get="/api/people"``,
  ``hx-trigger="keyup changed delay:200ms"``, target ``#people-results``).
- YOU pinned row is always first; links to ``/you``.
- NEEDS ATTENTION renders drifting contacts with right-aligned monospace
  deviation; ACTIVE THIS WEEK renders recent contacts with last-seen date.
- Each contact row links to ``/people/{contact_id}``.
- No avatars (``<img`` and ``avatar`` strings absent).
- Empty states for both sub-lists; "no matches" copy when query is set
  and both sub-lists are empty.
- Search input pre-fills the existing query (HTMX pages never lose state).
- ``contact_dossier.html`` renders all four sections (header / cadence /
  topics / template) with empty states.
- Singular vs plural copy for cadence days.

Structural assertions only — no full-HTML snapshots.
"""

from __future__ import annotations

import re

from web.rendering import render

REF_NOW = 1_777_204_800  # 2026-04-26T12:00:00Z, mirrors test_routes_people.py


_EMPTY_YOU: dict = {
    "observed_months": 0,
    "interactions_count": 0,
    "confidence_pct": 0,
    "when_at_best": [],
    "how_you_write": [],
    "your_routines": [],
    "drifting": [],
}


def _empty_people(query: str | None = None) -> dict:
    return {
        "you": _EMPTY_YOU,
        "needs_attention": [],
        "active_this_week": [],
        "total": 0,
        "query": query,
    }


def _populated_people() -> dict:
    return {
        "you": {**_EMPTY_YOU, "observed_months": 4, "interactions_count": 312},
        "needs_attention": [
            {
                "contact_id": "alice",
                "name": "Alice",
                "last_contact_ts": REF_NOW - 30 * 86400,
                "cadence_deviation_days": 23,
                "needs_attention": True,
            },
        ],
        "active_this_week": [
            {
                "contact_id": "bob",
                "name": "Bob",
                "last_contact_ts": REF_NOW - 1 * 86400,
                "cadence_deviation_days": -6,
                "needs_attention": False,
            },
        ],
        "total": 2,
        "query": None,
    }


# ---------------------------------------------------------------------------
# people.html — base wiring
# ---------------------------------------------------------------------------


def test_people_page_declares_active_tab() -> None:
    html = render("people.html", {"people": _empty_people()})
    assert 'data-active-tab="people"' in html


def test_people_page_marks_people_tab_active_in_nav() -> None:
    html = render("people.html", {"people": _empty_people()})
    people_anchor = html.split('data-tab="people"', 1)[1].split("</a>", 1)[0]
    assert 'aria-current="page"' in people_anchor
    now_anchor = html.split('data-tab="now"', 1)[1].split("</a>", 1)[0]
    assert 'aria-current="page"' not in now_anchor


def test_people_page_renders_header_and_total() -> None:
    html = render("people.html", {"people": {**_empty_people(), "total": 7}})
    assert 'data-section="people-header"' in html
    assert 'data-slot="people-total">7<' in html
    assert "contacts" in html


def test_people_page_uses_singular_contact_at_one() -> None:
    html = render("people.html", {"people": {**_empty_people(), "total": 1}})
    assert 'data-slot="people-total">1<' in html
    # The next word after the total slot must be "contact" (singular), not "contacts".
    after = html.split('data-slot="people-total">1</span>', 1)[1]
    assert re.search(r"\s+contact\b", after) and not re.search(r"\s+contacts\b", after[:60])


def test_people_search_input_has_exact_htmx_wiring() -> None:
    html = render("people.html", {"people": _empty_people()})
    assert 'data-slot="search-input"' in html
    assert 'hx-get="/api/people"' in html
    assert 'hx-trigger="keyup changed delay:200ms, search"' in html
    assert 'hx-target="#people-results"' in html
    assert 'hx-swap="innerHTML"' in html
    # Target div must exist for the swap.
    assert 'id="people-results"' in html


def test_people_search_input_prefills_existing_query() -> None:
    html = render("people.html", {"people": _empty_people(query="alice")})
    # The exact value attribute carries the query so refreshes don't lose state.
    assert 'value="alice"' in html


def test_people_page_has_no_avatars() -> None:
    """DESIGN.md § 'What's NOT in Phase 1': no avatars."""
    html = render("people.html", {"people": _populated_people()})
    assert "<img" not in html
    assert "avatar" not in html.lower()


# ---------------------------------------------------------------------------
# partials/people_results.html — YOU pinned row
# ---------------------------------------------------------------------------


def test_results_partial_renders_you_pinned_first() -> None:
    html = render("partials/people_results.html", {"people": _populated_people()})
    assert 'data-section="you-pinned"' in html
    assert 'data-slot="you-link"' in html
    # YOU section appears before any sub-list section.
    you_idx = html.index('data-section="you-pinned"')
    needs_idx = html.index('data-section="needs-attention"')
    active_idx = html.index('data-section="active-this-week"')
    assert you_idx < needs_idx < active_idx


def test_results_partial_you_link_points_to_you_tab() -> None:
    html = render("partials/people_results.html", {"people": _populated_people()})
    you_block = html.split('data-section="you-pinned"', 1)[1].split("</section>", 1)[0]
    assert 'href="/you"' in you_block


def test_results_partial_you_summary_renders_observed_and_interactions() -> None:
    html = render("partials/people_results.html", {"people": _populated_people()})
    assert 'data-slot="you-observed-months">4<' in html
    assert 'data-slot="you-interactions">312<' in html


# ---------------------------------------------------------------------------
# NEEDS ATTENTION
# ---------------------------------------------------------------------------


def test_results_needs_attention_renders_contacts() -> None:
    html = render("partials/people_results.html", {"people": _populated_people()})
    assert 'data-section="needs-attention"' in html
    assert 'data-contact-id="alice"' in html
    assert 'data-slot="needs-name">Alice<' in html
    # Positive deviation shows with leading "+".
    assert "+23d" in html


def test_results_needs_attention_row_links_to_dossier() -> None:
    html = render("partials/people_results.html", {"people": _populated_people()})
    needs_block = html.split('data-section="needs-attention"', 1)[1].split("</section>", 1)[0]
    assert 'href="/people/alice"' in needs_block


def test_results_needs_attention_empty_state() -> None:
    html = render("partials/people_results.html", {"people": _empty_people()})
    assert 'data-empty="needs-attention"' in html
    assert "Nobody's slipping right now." in html


def test_results_needs_attention_uses_monospace_for_meta() -> None:
    html = render("partials/people_results.html", {"people": _populated_people()})
    needs_block = html.split('data-section="needs-attention"', 1)[1].split("</section>", 1)[0]
    assert 'data-slot="needs-meta"' in needs_block
    assert "font-mono" in needs_block


# ---------------------------------------------------------------------------
# ACTIVE THIS WEEK
# ---------------------------------------------------------------------------


def test_results_active_this_week_renders_contacts() -> None:
    html = render("partials/people_results.html", {"people": _populated_people()})
    assert 'data-section="active-this-week"' in html
    assert 'data-contact-id="bob"' in html
    assert 'data-slot="active-name">Bob<' in html


def test_results_active_this_week_row_links_to_dossier() -> None:
    html = render("partials/people_results.html", {"people": _populated_people()})
    active_block = html.split('data-section="active-this-week"', 1)[1].split("</section>", 1)[0]
    assert 'href="/people/bob"' in active_block


def test_results_active_this_week_empty_state() -> None:
    html = render("partials/people_results.html", {"people": _empty_people()})
    assert 'data-empty="active-this-week"' in html
    assert "No recent activity." in html


def test_results_active_this_week_renders_last_seen_date() -> None:
    """Active rows show the last_contact_ts as a short formatted date."""
    html = render("partials/people_results.html", {"people": _populated_people()})
    active_block = html.split('data-section="active-this-week"', 1)[1].split("</section>", 1)[0]
    # REF_NOW (2026-04-26T12:00 UTC) - 1d = 2026-04-25 → "Apr 25".
    assert "Apr 25" in active_block


# ---------------------------------------------------------------------------
# No-matches copy when query is set and both sub-lists are empty
# ---------------------------------------------------------------------------


def test_results_no_matches_renders_when_query_yields_empty_sublists() -> None:
    html = render(
        "partials/people_results.html",
        {"people": _empty_people(query="zzz")},
    )
    assert 'data-section="people-no-matches"' in html
    assert "No contacts match" in html
    # Quoted query string is echoed — autoescape rewrites the smart quotes.
    assert "zzz" in html


def test_results_no_matches_absent_when_results_present() -> None:
    html = render("partials/people_results.html", {"people": _populated_people()})
    assert 'data-section="people-no-matches"' not in html


def test_results_no_matches_absent_when_query_empty() -> None:
    """An empty roster with no query is the standard empty state, not no-matches."""
    html = render("partials/people_results.html", {"people": _empty_people()})
    assert 'data-section="people-no-matches"' not in html


# ---------------------------------------------------------------------------
# contact_dossier.html
# ---------------------------------------------------------------------------


def _empty_dossier() -> dict:
    return {
        "contact_id": "newbie",
        "name": "Newbie",
        "last_contact_ts": None,
        "usual_cadence_days": None,
        "comm_template": None,
        "cadence_sparkline": [0] * 14,
        "recent_topics": [],
        "predicted_next": None,
    }


def _full_dossier() -> dict:
    return {
        "contact_id": "alice",
        "name": "Alice",
        "last_contact_ts": REF_NOW - 3 * 86400,
        "usual_cadence_days": 7,
        "comm_template": "Hey {name},",
        "cadence_sparkline": [0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 3, 1, 0, 1],
        "recent_topics": ["birthday", "vacation"],
        "predicted_next": "Next contact expected in ~4 days",
    }


def test_dossier_declares_active_tab_people() -> None:
    html = render("contact_dossier.html", {"contact": _empty_dossier()})
    assert 'data-active-tab="people"' in html


def test_dossier_renders_back_link_to_people() -> None:
    html = render("contact_dossier.html", {"contact": _empty_dossier()})
    assert 'data-slot="dossier-back"' in html
    # The back link must point to /people (the link wraps the slot).
    assert re.search(r'<a[^>]*href="/people"[^>]*data-slot="dossier-back"', html)


def test_dossier_renders_name_in_h1() -> None:
    html = render("contact_dossier.html", {"contact": _full_dossier()})
    assert 'data-slot="dossier-name">Alice<' in html


def test_dossier_renders_last_seen_when_present() -> None:
    html = render("contact_dossier.html", {"contact": _full_dossier()})
    assert "Last seen" in html
    assert 'data-slot="dossier-last-ts"' in html
    # REF_NOW (2026-04-26T12:00 UTC) - 3d = 2026-04-23 → "Apr 23, 2026".
    assert "Apr 23, 2026" in html


def test_dossier_renders_not_yet_observed_when_empty() -> None:
    html = render("contact_dossier.html", {"contact": _empty_dossier()})
    assert "Not yet observed." in html


def test_dossier_renders_cadence_section_with_sparkline_and_count() -> None:
    html = render("contact_dossier.html", {"contact": _full_dossier()})
    assert 'data-section="dossier-cadence"' in html
    assert 'data-slot="dossier-sparkline"' in html
    # Total = 0+0+1+0+2+0+0+0+1+0+3+1+0+1 = 9 events.
    assert re.search(r">\s*9\s+events\b", html)
    # Sparkline character set: at minimum a non-empty glyph string.
    spark_block = html.split('data-slot="dossier-sparkline"', 1)[1].split("</p>", 1)[0]
    assert any(ch in spark_block for ch in "▁▂▃▄▅▆▇█")


def test_dossier_cadence_singular_event_at_one() -> None:
    contact = {**_empty_dossier(), "cadence_sparkline": [1] + [0] * 13}
    html = render("contact_dossier.html", {"contact": contact})
    # Singular: "1 event in window", never the plural "events".
    assert re.search(r">\s*1\s+event\s+in window", html)
    assert not re.search(r">\s*1\s+events\b", html)


def test_dossier_cadence_empty_state_when_no_sparkline_data() -> None:
    """Empty list → empty state copy, not a flat baseline."""
    contact = {**_empty_dossier(), "cadence_sparkline": []}
    html = render("contact_dossier.html", {"contact": contact})
    assert 'data-empty="dossier-cadence"' in html
    assert "No events in the last 14 days." in html


def test_dossier_recent_topics_renders_each() -> None:
    html = render("contact_dossier.html", {"contact": _full_dossier()})
    assert 'data-section="dossier-topics"' in html
    assert html.count('data-slot="dossier-topic"') == 2
    assert "birthday" in html
    assert "vacation" in html


def test_dossier_recent_topics_empty_state() -> None:
    html = render("contact_dossier.html", {"contact": _empty_dossier()})
    assert 'data-empty="dossier-topics"' in html
    assert "No topics extracted yet." in html


def test_dossier_template_renders_when_present() -> None:
    html = render("contact_dossier.html", {"contact": _full_dossier()})
    assert 'data-slot="dossier-template"' in html
    assert "Hey {name}," in html


def test_dossier_template_empty_state() -> None:
    html = render("contact_dossier.html", {"contact": _empty_dossier()})
    assert 'data-empty="dossier-template"' in html
    assert "No outbound style detected yet." in html


def test_dossier_predicted_next_renders_when_present() -> None:
    html = render("contact_dossier.html", {"contact": _full_dossier()})
    assert 'data-slot="dossier-predicted"' in html
    assert "Next contact expected in ~4 days" in html


def test_dossier_predicted_next_absent_when_none() -> None:
    html = render("contact_dossier.html", {"contact": _empty_dossier()})
    assert 'data-slot="dossier-predicted"' not in html


def test_dossier_no_avatars() -> None:
    html = render("contact_dossier.html", {"contact": _full_dossier()})
    assert "<img" not in html
    assert "avatar" not in html.lower()


def test_dossier_singular_day_when_cadence_is_one() -> None:
    contact = {**_empty_dossier(), "usual_cadence_days": 1}
    html = render("contact_dossier.html", {"contact": contact})
    # The cadence slot renders "1 day" (singular), not "1 days".
    assert re.search(r'data-slot="dossier-cadence">1</span>\s*day\b', html)
    assert not re.search(r'data-slot="dossier-cadence">1</span>\s*days\b', html)
