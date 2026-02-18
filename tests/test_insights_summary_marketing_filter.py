"""
Life OS — Tests for marketing/inbound-only filter in /api/insights/summary.

The /api/insights/summary endpoint now delegates entirely to the InsightEngine,
which runs all 14 correlators including _contact_gap_insights.  The
_contact_gap_insights correlator applies the shared marketing filter and the
inbound-only filter (outbound_count == 0) before generating
``relationship_intelligence`` insights.

Previously the endpoint contained a 220-line hand-rolled implementation that
generated ``relationship_overdue`` and ``relationship_dynamics`` insights
directly from the relationships signal profile.  That implementation was
replaced in iteration 235 because it:
  - Covered only 3 of 14 insight categories (relationships, linguistic, places)
  - Lacked deduplication and source-weight gating
  - Generated structurally unfulfillable insights for automated mailers

The marketing filter and inbound-only filter are now the InsightEngine's
responsibility and are tested here through the endpoint integration path.

InsightEngine insight types used:
  - ``relationship_intelligence`` (category ``contact_gap``) — contact overdue
    relative to their usual interaction interval

Filter requirements (InsightEngine._contact_gap_insights):
  - interaction_count >= 5
  - len(interaction_timestamps) >= 3
  - outbound_count > 0 (bidirectional — not inbound-only)
  - not is_marketing_or_noreply(address)
  - days_since_last > avg_gap * 1.5 AND days_since_last > 7

Tests:
  1. Marketing addresses are excluded from relationship_intelligence insights.
  2. Marketing addresses with inbound-only are excluded.
  3. Inbound-only contacts (outbound_count=0) are excluded.
  4. Legitimate bidirectional overdue contacts appear as relationship_intelligence.
  5. Bidirectional contacts with sufficient gap appear even if not "dynamics" type.
  6. No-reply variants (noreply-, no_reply@, donotreply@) are excluded.
  7. An empty relationships profile returns a 200 with no relationship insights.
  8. Missing relationships profile returns a 200.
  9. Mixed contacts: marketing excluded, real bidirectional contact included.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from services.insight_engine.engine import InsightEngine
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def life_os_mock(db, event_store, user_model_store):
    """Minimal LifeOS mock wired to real temp database components.

    Uses a **real** InsightEngine (wired to the same temp DatabaseManager) so
    the /api/insights/summary endpoint exercises the actual _contact_gap_insights
    correlator and its marketing/inbound-only filters end-to-end.
    """
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
    # Wire a real InsightEngine so the endpoint's delegation path is exercised.
    # No source_weight_manager so all insights pass the weight gate by default.
    mock.insight_engine = InsightEngine(db=db, ums=user_model_store)
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

def test_marketing_address_excluded_from_relationship_intelligence(client, life_os_mock):
    """Marketing senders must not appear in relationship_intelligence insights.

    The relationships profile may contain millions of automated-mailer addresses.
    Surfacing 'You haven't emailed noreply@alerts.com in 60 days' is misleading
    and unactionable.  InsightEngine._contact_gap_insights excludes them via
    the shared is_marketing_or_noreply() filter.
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
    # InsightEngine uses "relationship_intelligence" (not the old "relationship_overdue")
    relationship = [i for i in insights if i["type"] == "relationship_intelligence"]

    # Both addresses are marketing senders — neither should appear
    entities = [i.get("entity", "") for i in relationship]
    assert "noreply@newsletter.example.com" not in entities, (
        "Marketing sender noreply@newsletter.example.com should be excluded from "
        "relationship_intelligence insights"
    )
    assert "donotreply@bank.com" not in entities, (
        "Marketing sender donotreply@bank.com should be excluded from "
        "relationship_intelligence insights"
    )


def test_marketing_address_excluded_from_relationship_intelligence_inbound_only(client, life_os_mock):
    """Marketing senders with inbound-only traffic must not appear in relationship_intelligence.

    alerts@brokerage.com is both a marketing sender (excluded by is_marketing_or_noreply)
    and inbound-only (outbound_count=0, excluded by the bidirectionality filter).
    Either filter alone is sufficient; together they provide defense-in-depth.
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
    relationship = [i for i in insights if i["type"] == "relationship_intelligence"]
    entities = [i.get("entity", "") for i in relationship]

    assert "alerts@brokerage.com" not in entities, (
        "Marketing sender alerts@brokerage.com should be excluded from "
        "relationship_intelligence insights"
    )


# ---------------------------------------------------------------------------
# Tests: inbound-only contacts are excluded
# ---------------------------------------------------------------------------

def test_inbound_only_contact_excluded_from_relationship_intelligence(client, life_os_mock):
    """Inbound-only contacts (outbound_count=0) must not appear in relationship_intelligence.

    If the user has never sent a message to an address, there is no established
    bidirectional relationship to maintain.  InsightEngine._contact_gap_insights
    skips contacts where outbound_count == 0.
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
        if i["type"] == "relationship_intelligence"
    ]
    entities = [i.get("entity", "") for i in relationship_insights]

    assert "cold.sender@example.com" not in entities, (
        "Inbound-only contact cold.sender@example.com should be excluded from "
        "relationship_intelligence insights"
    )


# ---------------------------------------------------------------------------
# Tests: legitimate bidirectional contacts still appear
# ---------------------------------------------------------------------------

def test_bidirectional_contact_appears_in_relationship_intelligence(client, life_os_mock):
    """Bidirectional contacts with sufficient gap appear as relationship_intelligence.

    The filter must not suppress real human contacts the user actively corresponds
    with.  A non-marketing address with outbound_count > 0 that is overdue (last
    seen 60 days ago, avg gap ~5 days) should generate a relationship_intelligence
    insight from InsightEngine._contact_gap_insights.

    InsightEngine threshold: days_since > avg_gap * 1.5 AND days_since > 7
      60 > 5 * 1.5 = 7.5 ✓ and 60 > 7 ✓  →  insight generated
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
    relationship = [i for i in insights if i["type"] == "relationship_intelligence"]
    entities = [i.get("entity", "") for i in relationship]

    assert "friend@example.com" in entities, (
        "Legitimate bidirectional contact friend@example.com should appear in "
        "relationship_intelligence insights when genuinely overdue"
    )


def test_bidirectional_contact_appears_in_contact_gap_intelligence(client, life_os_mock):
    """Non-marketing bidirectional contacts with sufficient gap produce relationship_intelligence.

    InsightEngine._contact_gap_insights generates relationship_intelligence insights
    (not the legacy relationship_dynamics type) for contacts who are overdue relative
    to their usual interaction interval.

    threshold: days_since > avg_gap * 1.5 AND days_since > 7
      60 > 3 * 1.5 = 4.5 ✓ and 60 > 7 ✓  →  insight generated
    """
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
    relationship = [i for i in insights if i["type"] == "relationship_intelligence"]
    entities = [i.get("entity", "") for i in relationship]

    assert "colleague@work.com" in entities, (
        "Non-marketing bidirectional contact colleague@work.com should appear "
        "in relationship_intelligence insights when overdue"
    )


# ---------------------------------------------------------------------------
# Tests: no-reply variants are excluded
# ---------------------------------------------------------------------------

def test_noreply_variants_excluded(client, life_os_mock):
    """Various no-reply address patterns must all be excluded from relationship_intelligence.

    The shared marketing filter handles noreply-, no-reply, no_reply, donotreply,
    do-not-reply.  We verify all common variants are excluded even when they have
    some outbound interaction history (slight outbound should not override the
    marketing address check).
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
        if i["type"] == "relationship_intelligence"
    }

    for addr in no_reply_addresses:
        assert addr not in relationship_entities, (
            f"No-reply variant {addr} should be excluded from relationship_intelligence insights"
        )


# ---------------------------------------------------------------------------
# Tests: empty profile is handled gracefully
# ---------------------------------------------------------------------------

def test_empty_relationships_profile_returns_no_relationship_insights(client, life_os_mock):
    """An empty relationships profile must not raise and returns no relationship insights."""
    _store_relationships_profile(life_os_mock.user_model_store, {"contacts": {}})

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    relationship_insights = [
        i for i in insights
        if i["type"] == "relationship_intelligence"
    ]
    assert relationship_insights == [], (
        "Empty relationships profile should produce zero relationship_intelligence insights"
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
    the profile with a marketing sender must still appear in relationship_intelligence.

    InsightEngine threshold: days_since > avg_gap * 1.5 AND days_since > 7
      alice@personal.com: 60 > 4 * 1.5 = 6 ✓ and 60 > 7 ✓  →  insight generated
      unsubscribe@promo.store: excluded by marketing filter (and inbound-only)
    """
    real_timestamps = _make_timestamps(8, gap_days=4.0)
    contacts = {
        # Should be excluded (marketing address + inbound-only)
        "unsubscribe@promo.store": _make_contact_profile(
            interaction_count=8,
            inbound_count=8,
            outbound_count=0,
            timestamps=real_timestamps,
        ),
        # Should appear (real bidirectional, genuinely overdue)
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
        if i["type"] == "relationship_intelligence"
    }

    assert "alice@personal.com" in relationship_entities, (
        "Real bidirectional contact alice@personal.com should appear in "
        "relationship_intelligence insights when overdue"
    )
    assert "unsubscribe@promo.store" not in relationship_entities, (
        "Marketing address unsubscribe@promo.store should be excluded from "
        "relationship_intelligence insights"
    )
