"""
Tests for _resolve_notification_source_key domain fallback.

Verifies that notifications without a source_event_id correctly fall back to
domain-based classification, fixing the asymmetry between main.py and
web/routes.py that caused 0 source weight dismissals despite 102 notification
dismissals.
"""

import json
from unittest.mock import MagicMock

import pytest

from services.insight_engine.source_weights import SourceWeightManager


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture()
def source_weight_manager(db):
    """A SourceWeightManager with default weights seeded."""
    mgr = SourceWeightManager(db)
    mgr.seed_defaults()
    return mgr


def _insert_event(db, event_id, event_type, source="test", payload=None, metadata=None):
    """Insert a synthetic event into the events DB."""
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, datetime('now'), 'normal', ?, ?)""",
            (event_id, event_type, source, json.dumps(payload or {}), json.dumps(metadata or {})),
        )


def _insert_notification(db, notif_id, source_event_id=None, domain=None):
    """Insert a synthetic notification into the state DB."""
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications (id, title, body, priority, source_event_id, domain, status)
               VALUES (?, 'Test', 'body', 'normal', ?, ?, 'delivered')""",
            (notif_id, source_event_id, domain),
        )


def _make_lifeos_stub(db, source_weight_manager):
    """Create a minimal LifeOS-like object with the real _resolve_notification_source_key."""
    from main import LifeOS

    stub = MagicMock(spec=LifeOS)
    stub.db = db
    stub.source_weight_manager = source_weight_manager
    stub._DOMAIN_TO_SOURCE = LifeOS._DOMAIN_TO_SOURCE
    stub._resolve_notification_source_key = LifeOS._resolve_notification_source_key.__get__(stub)
    return stub


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestDomainFallback:
    """Tests for domain-based fallback in _resolve_notification_source_key."""

    def test_with_source_event_id_classifies_via_event(self, db, source_weight_manager):
        """Notification with source_event_id classifies via event lookup (existing behavior)."""
        _insert_event(db, "evt-df-1", "email.received", payload={"from": "alice@gmail.com"})
        _insert_notification(db, "notif-df-1", source_event_id="evt-df-1", domain="email")

        stub = _make_lifeos_stub(db, source_weight_manager)
        result = stub._resolve_notification_source_key("notif-df-1")
        # Event-based classification takes precedence over domain fallback
        assert result == "email.personal"

    def test_email_domain_fallback(self, db, source_weight_manager):
        """Notification with domain='email' but no source_event_id returns 'email.work'."""
        _insert_notification(db, "notif-df-2", source_event_id=None, domain="email")

        stub = _make_lifeos_stub(db, source_weight_manager)
        result = stub._resolve_notification_source_key("notif-df-2")
        assert result == "email.work"

    def test_messaging_domain_fallback(self, db, source_weight_manager):
        """Notification with domain='messaging' but no source_event_id returns 'messaging.direct'."""
        _insert_notification(db, "notif-df-3", source_event_id=None, domain="messaging")

        stub = _make_lifeos_stub(db, source_weight_manager)
        result = stub._resolve_notification_source_key("notif-df-3")
        assert result == "messaging.direct"

    def test_calendar_domain_fallback(self, db, source_weight_manager):
        """Notification with domain='calendar' returns 'calendar.meetings'."""
        _insert_notification(db, "notif-df-4", source_event_id=None, domain="calendar")

        stub = _make_lifeos_stub(db, source_weight_manager)
        result = stub._resolve_notification_source_key("notif-df-4")
        assert result == "calendar.meetings"

    def test_prediction_domain_returns_none(self, db, source_weight_manager):
        """Notification with domain='prediction' (not in map) returns None."""
        _insert_notification(db, "notif-df-5", source_event_id=None, domain="prediction")

        stub = _make_lifeos_stub(db, source_weight_manager)
        result = stub._resolve_notification_source_key("notif-df-5")
        assert result is None

    def test_no_domain_no_source_returns_none(self, db, source_weight_manager):
        """Notification with no domain and no source_event_id returns None."""
        _insert_notification(db, "notif-df-6", source_event_id=None, domain=None)

        stub = _make_lifeos_stub(db, source_weight_manager)
        result = stub._resolve_notification_source_key("notif-df-6")
        assert result is None

    def test_missing_event_falls_back_to_domain(self, db, source_weight_manager):
        """Notification with source_event_id pointing to missing event falls back to domain."""
        _insert_notification(db, "notif-df-7", source_event_id="nonexistent-evt", domain="finance")

        stub = _make_lifeos_stub(db, source_weight_manager)
        result = stub._resolve_notification_source_key("notif-df-7")
        assert result == "finance.transactions"

    def test_dismissal_records_with_domain_fallback(self, db, source_weight_manager):
        """End-to-end: dismissal of a domain-only notification updates source weights."""
        _insert_notification(db, "notif-df-8", source_event_id=None, domain="messaging")

        stub = _make_lifeos_stub(db, source_weight_manager)
        source_key = stub._resolve_notification_source_key("notif-df-8")
        assert source_key == "messaging.direct"

        # Record the dismissal
        source_weight_manager.record_dismissal(source_key)

        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT dismissals FROM source_weights WHERE source_key = ?",
                (source_key,),
            ).fetchone()
        assert row["dismissals"] == 1
