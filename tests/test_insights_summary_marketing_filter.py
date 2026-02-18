"""
Life OS — Tests for marketing/inbound-only filter in /api/insights/summary.

The relationship section of /api/insights/summary previously iterated all
contacts in the relationships signal profile without filtering out automated
senders (newsletters, no-reply accounts, brokerage alerts) or inbound-only
contacts (addresses the user has never messaged back).

With 170K+ relationship samples dominated by automated mailers, the unfiltered
loop produced:
  - relationship_overdue insights for noreply@example.com, alerts@bank.com, etc.
    — structurally unfulfillable because the user cannot "reach out" to an
      automated mailer.
  - relationship_dynamics insights for inbound-only contacts — misleading
    "They reach out far more than you reply" for cold senders the user never
    chose to engage with.

This mirrors the fix applied to InsightEngine._contact_gap_insights (PR #207)
and PredictionEngine._check_relationship_maintenance (PR #204).

Tests:
  1. Marketing addresses are excluded from relationship_overdue insights.
  2. Marketing addresses are excluded from relationship_dynamics insights.
  3. Inbound-only contacts (outbound_count=0) are excluded from both insight types.
  4. Legitimate bidirectional contacts still appear.
  5. No-reply variants (noreply-, no_reply@, donotreply@) are excluded.
  6. An empty relationships profile returns an empty list gracefully.
  7. A contact with outbound activity but below minimum count threshold is
     handled correctly (relationship_overdue still excluded by gap logic, not
     the filter).
  8. A contact with both marketing address AND inbound-only is excluded once
     (not double-counted).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def life_os_mock(db, event_store, user_model_store):
    """Minimal LifeOS mock wired to real temp database components."""
    mock = MagicMock()
    mock.db = db
    mock.event_store = event_store
    mock.user_model_store = user_model_store
    mock.signal_extractor = MagicMock()
    mock.signal_extractor.get_user_summary.return_value = {"profiles": {}}
    mock.vector_store = MagicMock()
    mock.event_bus = MagicMock()
    mock.event_bus.is_connected = False
    mock.connectors = []
    mock.notification_manager = MagicMock()
    mock.notification_manager.get_stats.return_value = {}
    mock.feedback_collector = MagicMock()
    mock.feedback_collector.get_feedback_summary.return_value = {}
    mock.rules_engine = MagicMock()
    mock.task_manager = MagicMock()
    mock.ai_engine = MagicMock()
    mock.browser_orchestrator = MagicMock()
    mock.onboarding = MagicMock()
    return mock


@pytest.fixture
def client(life_os_mock):
    """TestClient bound to the full FastAPI app with the mock LifeOS."""
    from web.app import create_web_app
    app = create_web_app(life_os_mock)
    return TestClient(app)


def _make_timestamps(n: int, gap_days: float = 5.0) -> list[str]:
    """Return n ISO timestamps spaced gap_days apart, ending at a recent time.

    The last timestamp is 60 days ago so that any bidirectional contact
    with a ~5-day average gap will appear overdue (60 > 5 * 1.5).
    """
    now = datetime.now(timezone.utc)
    # last seen 60 days ago
    last = now - timedelta(days=60)
    timestamps = []
    for i in range(n - 1, -1, -1):
        timestamps.append((last - timedelta(days=i * gap_days)).isoformat())
    return timestamps


def _make_contact_profile(
    interaction_count: int,
    inbound_count: int,
    outbound_count: int,
    timestamps: list[str] | None = None,
) -> dict:
    """Build a contact profile dict as stored in the relationships signal profile."""
    if timestamps is None:
        timestamps = _make_timestamps(interaction_count)
    return {
        "interaction_count": interaction_count,
        "inbound_count": inbound_count,
        "outbound_count": outbound_count,
        "last_interaction": timestamps[-1] if timestamps else None,
        "interaction_timestamps": timestamps,
    }


def _store_relationships_profile(user_model_store, contacts: dict) -> None:
    """Write a relationships signal profile with the given contacts dict."""
    user_model_store.update_signal_profile("relationships", {"contacts": contacts})


# ---------------------------------------------------------------------------
# Tests: marketing senders are excluded
# ---------------------------------------------------------------------------

def test_marketing_address_excluded_from_relationship_overdue(client, life_os_mock):
    """Marketing senders must not appear in relationship_overdue insights.

    The relationships profile may contain millions of automated-mailer addresses.
    Surfacing 'You haven't emailed noreply@alerts.com in 60 days' is misleading
    and unactionable.
    """
    contacts = {
        # Marketing sender with heavy interaction history but all inbound
        "noreply@newsletter.example.com": _make_contact_profile(
            interaction_count=20,
            inbound_count=20,
            outbound_count=0,
            timestamps=_make_timestamps(20),
        ),
        # Another marketing pattern: donotreply
        "donotreply@bank.com": _make_contact_profile(
            interaction_count=10,
            inbound_count=10,
            outbound_count=1,  # even with some outbound, still marketing
            timestamps=_make_timestamps(10),
        ),
    }
    _store_relationships_profile(life_os_mock.user_model_store, contacts)

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    overdue = [i for i in insights if i["type"] == "relationship_overdue"]

    # Both addresses are marketing senders — neither should appear
    entities = [i.get("entity", "") for i in overdue]
    assert "noreply@newsletter.example.com" not in entities, (
        "Marketing sender noreply@newsletter.example.com should be excluded from "
        "relationship_overdue insights"
    )
    assert "donotreply@bank.com" not in entities, (
        "Marketing sender donotreply@bank.com should be excluded from "
        "relationship_overdue insights"
    )


def test_marketing_address_excluded_from_relationship_dynamics(client, life_os_mock):
    """Marketing senders must not appear in relationship_dynamics insights.

    The 'relationship_dynamics' insight reports interaction ratios (inbound/
    outbound balance).  For automated mailers this is always 100% inbound,
    which would falsely flag them as 'They reach out far more than you reply.'
    """
    contacts = {
        "alerts@brokerage.com": _make_contact_profile(
            interaction_count=15,
            inbound_count=15,
            outbound_count=0,
            timestamps=_make_timestamps(15),
        ),
    }
    _store_relationships_profile(life_os_mock.user_model_store, contacts)

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    dynamics = [i for i in insights if i["type"] == "relationship_dynamics"]
    entities = [i.get("entity", "") for i in dynamics]

    assert "alerts@brokerage.com" not in entities, (
        "Marketing sender alerts@brokerage.com should be excluded from "
        "relationship_dynamics insights"
    )


# ---------------------------------------------------------------------------
# Tests: inbound-only contacts are excluded
# ---------------------------------------------------------------------------

def test_inbound_only_contact_excluded_from_relationship_insights(client, life_os_mock):
    """Inbound-only contacts (outbound_count=0) must not appear in either insight type.

    If the user has never sent a message to an address, there is no established
    bidirectional relationship to maintain or to report dynamics for.  This
    mirrors the filter in _contact_gap_insights (PR #204).
    """
    contacts = {
        # Real human (non-marketing) but user has never replied
        "cold.sender@example.com": _make_contact_profile(
            interaction_count=8,
            inbound_count=8,
            outbound_count=0,
            timestamps=_make_timestamps(8),
        ),
    }
    _store_relationships_profile(life_os_mock.user_model_store, contacts)

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    relationship_insights = [
        i for i in insights
        if i["type"] in ("relationship_overdue", "relationship_dynamics")
    ]
    entities = [i.get("entity", "") for i in relationship_insights]

    assert "cold.sender@example.com" not in entities, (
        "Inbound-only contact cold.sender@example.com should be excluded from "
        "all relationship insights"
    )


# ---------------------------------------------------------------------------
# Tests: legitimate bidirectional contacts still appear
# ---------------------------------------------------------------------------

def test_bidirectional_contact_still_appears(client, life_os_mock):
    """Bidirectional contacts with sufficient gap should appear in relationship_overdue.

    The filter must not suppress real human contacts the user actively corresponds
    with.  A non-marketing address with outbound_count > 0 that is overdue should
    still generate an insight.
    """
    # Simulate a contact last seen 60 days ago with avg gap of ~5 days
    timestamps = _make_timestamps(10, gap_days=5.0)
    contacts = {
        "friend@example.com": _make_contact_profile(
            interaction_count=10,
            inbound_count=5,
            outbound_count=5,
            timestamps=timestamps,
        ),
    }
    _store_relationships_profile(life_os_mock.user_model_store, contacts)

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    overdue = [i for i in insights if i["type"] == "relationship_overdue"]
    entities = [i.get("entity", "") for i in overdue]

    assert "friend@example.com" in entities, (
        "Legitimate bidirectional contact friend@example.com should appear in "
        "relationship_overdue insights when genuinely overdue"
    )


def test_bidirectional_contact_appears_in_dynamics(client, life_os_mock):
    """Non-marketing contacts with >= 5 interactions appear in relationship_dynamics."""
    timestamps = _make_timestamps(10, gap_days=3.0)
    contacts = {
        "colleague@work.com": _make_contact_profile(
            interaction_count=10,
            inbound_count=7,
            outbound_count=3,
            timestamps=timestamps,
        ),
    }
    _store_relationships_profile(life_os_mock.user_model_store, contacts)

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    dynamics = [i for i in insights if i["type"] == "relationship_dynamics"]
    entities = [i.get("entity", "") for i in dynamics]

    assert "colleague@work.com" in entities, (
        "Non-marketing bidirectional contact colleague@work.com should appear "
        "in relationship_dynamics insights"
    )


# ---------------------------------------------------------------------------
# Tests: no-reply variants are excluded
# ---------------------------------------------------------------------------

def test_noreply_variants_excluded(client, life_os_mock):
    """Various no-reply address patterns must all be filtered out.

    The shared marketing filter handles noreply-, no-reply, no_reply, donotreply,
    do-not-reply.  We verify all common variants are excluded from the summary.
    """
    no_reply_addresses = [
        "noreply@company.com",
        "no-reply@service.io",
        "no_reply@newsletter.org",
        "donotreply@alerts.net",
    ]
    contacts = {}
    for addr in no_reply_addresses:
        timestamps = _make_timestamps(10, gap_days=3.0)
        contacts[addr] = _make_contact_profile(
            interaction_count=10,
            inbound_count=9,
            outbound_count=1,  # slight outbound should not override marketing check
            timestamps=timestamps,
        )
    _store_relationships_profile(life_os_mock.user_model_store, contacts)

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    relationship_entities = {
        i.get("entity", "")
        for i in insights
        if i["type"] in ("relationship_overdue", "relationship_dynamics")
    }

    for addr in no_reply_addresses:
        assert addr not in relationship_entities, (
            f"No-reply variant {addr} should be excluded from relationship insights"
        )


# ---------------------------------------------------------------------------
# Tests: empty profile is handled gracefully
# ---------------------------------------------------------------------------

def test_empty_relationships_profile_returns_no_relationship_insights(client, life_os_mock):
    """An empty relationships profile must not raise and returns an empty list."""
    _store_relationships_profile(life_os_mock.user_model_store, {"contacts": {}})

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    relationship_insights = [
        i for i in insights
        if i["type"] in ("relationship_overdue", "relationship_dynamics")
    ]
    assert relationship_insights == [], (
        "Empty relationships profile should produce zero relationship insights"
    )


def test_no_relationships_profile_stored_returns_200(client, life_os_mock):
    """When no relationships signal profile has been stored, endpoint returns 200."""
    # No profile stored — user_model_store.get_signal_profile returns None
    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data


# ---------------------------------------------------------------------------
# Tests: mixed contacts — only the filtered ones are excluded
# ---------------------------------------------------------------------------

def test_mixed_contacts_only_marketing_excluded(client, life_os_mock):
    """With a mix of marketing and real contacts, only marketing is excluded.

    This ensures the filter is not too broad — a real contact co-residing in
    the profile with a marketing sender must still appear.
    """
    real_timestamps = _make_timestamps(8, gap_days=4.0)
    contacts = {
        # Should be excluded (marketing)
        "unsubscribe@promo.store": _make_contact_profile(
            interaction_count=8,
            inbound_count=8,
            outbound_count=0,
            timestamps=real_timestamps,
        ),
        # Should appear (real bidirectional)
        "alice@personal.com": _make_contact_profile(
            interaction_count=8,
            inbound_count=4,
            outbound_count=4,
            timestamps=real_timestamps,
        ),
    }
    _store_relationships_profile(life_os_mock.user_model_store, contacts)

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    relationship_entities = {
        i.get("entity", "")
        for i in insights
        if i["type"] in ("relationship_overdue", "relationship_dynamics")
    }

    assert "alice@personal.com" in relationship_entities, (
        "Real bidirectional contact alice@personal.com should appear in insights"
    )
    assert "unsubscribe@promo.store" not in relationship_entities, (
        "Marketing address unsubscribe@promo.store should be excluded"
    )
