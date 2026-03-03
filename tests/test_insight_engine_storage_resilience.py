"""
Tests for InsightEngine resilience to corrupted user_model.db.

Validates two fixes:
1. generate_insights() returns computed insights even when _store_insight
   raises sqlite3.DatabaseError (user_model.db corruption).
2. _contact_gap_insights falls back to events.db when the relationships
   signal profile is unavailable, and marketing senders are filtered.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.insight_engine.engine import InsightEngine
from services.insight_engine.models import Insight


# =============================================================================
# Helpers
# =============================================================================


def _insert_email_received(db, from_address: str, timestamp: str) -> str:
    """Insert an email.received event into events.db and return its id."""
    eid = str(uuid.uuid4())
    payload = json.dumps({"from_address": from_address, "subject": "Hello"})
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events
               (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (eid, "email.received", "test", timestamp, 2, payload, "{}"),
        )
    return eid


def _insert_email_sent(db, to_address: str, timestamp: str) -> str:
    """Insert an email.sent event into events.db and return its id."""
    eid = str(uuid.uuid4())
    payload = json.dumps({"to_address": to_address, "subject": "Re: Hello"})
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events
               (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (eid, "email.sent", "test", timestamp, 2, payload, "{}"),
        )
    return eid


def _populate_contact_emails(db, from_address: str, count: int = 6,
                             days_ago_start: int = 60):
    """Insert multiple email.received events for a contact spread over time.

    Creates ``count`` events starting from ``days_ago_start`` days ago,
    evenly spaced. The most recent event is placed at ``days_ago_start``
    divided roughly across the count window — but all are older than 14 days
    so the contact-gap heuristic fires.
    """
    now = datetime.now(timezone.utc)
    for i in range(count):
        # Spread events from days_ago_start..days_ago_start-count (all > 14 days ago)
        ts = (now - timedelta(days=days_ago_start - i)).isoformat()
        _insert_email_received(db, from_address, ts)


# =============================================================================
# Fix 1: generate_insights() resilience to _store_insight failure
# =============================================================================


@pytest.mark.asyncio
async def test_generate_insights_returns_insights_when_store_fails(
    db, user_model_store
):
    """generate_insights() should return insights even when _store_insight raises.

    When user_model.db is corrupted, persisting insights fails.  The fix
    ensures computed insights are still returned to callers (e.g. the
    briefing endpoint), even though they can't be persisted.
    """
    engine = InsightEngine(db, user_model_store, cache_ttl_seconds=0)

    # Insert a task that's overdue so the actionable_alert correlator fires
    now = datetime.now(timezone.utc)
    due = (now - timedelta(days=1)).isoformat()
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO tasks
               (id, title, status, priority, due_date, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), "Overdue task", "pending", 2, due,
             now.isoformat()),
        )

    # Patch _store_insight to raise sqlite3.DatabaseError
    with patch.object(
        engine, "_store_insight",
        side_effect=sqlite3.DatabaseError("database disk image is malformed"),
    ):
        insights = await engine.generate_insights()

    # Insights should still be returned despite storage failure
    assert len(insights) > 0
    types = {i.type for i in insights}
    assert "actionable_alert" in types


@pytest.mark.asyncio
async def test_generate_insights_stores_when_db_healthy(
    db, user_model_store
):
    """generate_insights() should store insights normally when DB is healthy."""
    engine = InsightEngine(db, user_model_store, cache_ttl_seconds=0)

    now = datetime.now(timezone.utc)
    due = (now - timedelta(days=1)).isoformat()
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO tasks
               (id, title, status, priority, due_date, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), "Overdue task", "pending", 2, due,
             now.isoformat()),
        )

    insights = await engine.generate_insights()
    assert len(insights) > 0

    # Verify insights were actually persisted
    with db.get_connection("user_model") as conn:
        rows = conn.execute("SELECT COUNT(*) as cnt FROM insights").fetchone()
        assert rows["cnt"] > 0


# =============================================================================
# Fix 2: _contact_gap_insights events.db fallback
# =============================================================================


def test_contact_gap_fallback_when_profile_is_none(db, user_model_store):
    """_contact_gap_insights returns insights from events.db when profile is None.

    When get_signal_profile('relationships') returns None (corrupted or empty
    user_model.db), the method falls back to querying events.db directly
    for contacts with 5+ inbound emails and no recent reply.
    """
    engine = InsightEngine(db, user_model_store)

    # Insert 6 emails from a real contact, all > 14 days ago
    _populate_contact_emails(db, "alice@example.com", count=6, days_ago_start=60)

    # Ensure get_signal_profile returns None (simulating corrupted DB)
    with patch.object(user_model_store, "get_signal_profile", return_value=None):
        insights = engine._contact_gap_insights()

    contact_gap = [i for i in insights if i.category == "contact_gap"]
    assert len(contact_gap) == 1
    assert "alice@example.com" in contact_gap[0].entity
    assert contact_gap[0].type == "relationship_intelligence"
    assert any("events_db_fallback" in e for e in contact_gap[0].evidence)


def test_contact_gap_fallback_filters_marketing(db, user_model_store):
    """Marketing senders are filtered from the events.db fallback results."""
    engine = InsightEngine(db, user_model_store)

    # Insert emails from a no-reply address
    _populate_contact_emails(
        db, "noreply@newsletters.example.com", count=8, days_ago_start=60
    )

    with patch.object(user_model_store, "get_signal_profile", return_value=None):
        insights = engine._contact_gap_insights()

    # Marketing sender should be filtered out
    contact_gap = [i for i in insights if i.category == "contact_gap"]
    assert len(contact_gap) == 0


def test_contact_gap_fallback_skips_recently_replied(db, user_model_store):
    """Contacts the user has replied to recently are not flagged."""
    engine = InsightEngine(db, user_model_store)

    # Insert 6 old inbound emails
    _populate_contact_emails(db, "bob@company.com", count=6, days_ago_start=60)

    # Insert a recent reply from the user
    now = datetime.now(timezone.utc)
    _insert_email_sent(db, "bob@company.com", (now - timedelta(days=3)).isoformat())

    with patch.object(user_model_store, "get_signal_profile", return_value=None):
        insights = engine._contact_gap_insights()

    contact_gap = [i for i in insights if i.category == "contact_gap"]
    assert len(contact_gap) == 0


def test_contact_gap_fallback_skips_low_interaction_contacts(db, user_model_store):
    """Contacts with fewer than 5 inbound emails are not flagged."""
    engine = InsightEngine(db, user_model_store)

    # Only 3 emails — below the threshold
    _populate_contact_emails(db, "sparse@example.com", count=3, days_ago_start=60)

    with patch.object(user_model_store, "get_signal_profile", return_value=None):
        insights = engine._contact_gap_insights()

    contact_gap = [i for i in insights if i.category == "contact_gap"]
    assert len(contact_gap) == 0


def test_contact_gap_fallback_skips_recent_contacts(db, user_model_store):
    """Contacts with last email < 14 days ago should not be flagged."""
    engine = InsightEngine(db, user_model_store)

    # Insert emails with last one only 5 days ago
    now = datetime.now(timezone.utc)
    for i in range(6):
        ts = (now - timedelta(days=5 + i)).isoformat()
        _insert_email_received(db, "recent@example.com", ts)

    with patch.object(user_model_store, "get_signal_profile", return_value=None):
        insights = engine._contact_gap_insights()

    # The most recent email is 5 days ago (< 14 day threshold), so no insight
    contact_gap = [
        i for i in insights
        if i.category == "contact_gap" and i.entity == "recent@example.com"
    ]
    assert len(contact_gap) == 0


def test_contact_gap_fallback_dedup_key_computed(db, user_model_store):
    """Each insight from the fallback should have a dedup_key set."""
    engine = InsightEngine(db, user_model_store)

    _populate_contact_emails(db, "dedup@example.com", count=7, days_ago_start=60)

    with patch.object(user_model_store, "get_signal_profile", return_value=None):
        insights = engine._contact_gap_insights()

    contact_gap = [i for i in insights if i.category == "contact_gap"]
    assert len(contact_gap) == 1
    assert contact_gap[0].dedup_key != ""
