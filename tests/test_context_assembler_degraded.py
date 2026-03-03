"""
Tests for ContextAssembler degraded-mode fallbacks.

When user_model.db is corrupted or empty, the briefing context normally loses
all personalization (episodes, semantic facts, routines, mood).  These tests
verify that events.db-based fallback helpers provide useful context instead.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from services.ai_engine.context import ContextAssembler
from storage.user_model_store import UserModelStore

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _insert_event(conn, event_type, payload, hours_ago=1):
    """Insert a single event into events.db with a timestamp offset."""
    ts = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
    conn.execute(
        """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
           VALUES (?, ?, 'test', ?, 'normal', ?, '{}')""",
        (str(uuid.uuid4()), event_type, ts, json.dumps(payload)),
    )


def _make_assembler(db, event_bus=None):
    """Create a ContextAssembler with a UserModelStore wired to the given db."""
    if event_bus is None:
        event_bus = AsyncMock()
    ums = UserModelStore(db, event_bus=event_bus)
    return ContextAssembler(db, ums)


# ------------------------------------------------------------------ #
# _get_recent_activity_summary tests
# ------------------------------------------------------------------ #


def test_recent_activity_summary_with_events(db):
    """_get_recent_activity_summary returns a summary when events.db has data."""
    assembler = _make_assembler(db)

    with db.get_connection("events") as conn:
        # Insert various event types
        for i in range(5):
            _insert_event(
                conn,
                "email.received",
                {
                    "from_address": "alice@example.com",
                    "subject": f"Subject {i}",
                },
                hours_ago=i + 1,
            )
        for i in range(3):
            _insert_event(
                conn,
                "message.received",
                {
                    "from_address": "bob@work.com",
                    "subject": f"Chat {i}",
                },
                hours_ago=i + 1,
            )
        _insert_event(
            conn,
            "calendar.event.created",
            {
                "title": "Team Meeting",
                "start_time": "2026-03-03T10:00:00",
            },
            hours_ago=2,
        )
        conn.commit()

    result = assembler._get_recent_activity_summary()

    assert "Recent activity summary (last 24h, from event log):" in result
    assert "email.received" in result
    assert "message.received" in result
    assert "calendar.event.created" in result
    # Should include top senders
    assert "alice@example.com" in result
    # Should include recent subjects
    assert "Subject" in result


def test_recent_activity_summary_empty_events_db(db):
    """_get_recent_activity_summary returns empty string when no recent events exist."""
    assembler = _make_assembler(db)

    result = assembler._get_recent_activity_summary()

    assert result == ""


def test_recent_activity_summary_old_events_excluded(db):
    """Events older than 24h are excluded from the summary."""
    assembler = _make_assembler(db)

    with db.get_connection("events") as conn:
        # Insert events older than 24 hours
        _insert_event(
            conn,
            "email.received",
            {
                "from_address": "old@example.com",
                "subject": "Old email",
            },
            hours_ago=30,
        )
        conn.commit()

    result = assembler._get_recent_activity_summary()

    assert result == ""


# ------------------------------------------------------------------ #
# _get_contact_activity_summary tests
# ------------------------------------------------------------------ #


def test_contact_activity_summary_with_events(db):
    """_get_contact_activity_summary returns top contacts sorted by interaction count."""
    assembler = _make_assembler(db)

    with db.get_connection("events") as conn:
        # Alice: 5 received + 3 sent = 8 total
        for i in range(5):
            _insert_event(
                conn,
                "email.received",
                {
                    "from_address": "alice@example.com",
                },
                hours_ago=i * 12 + 1,
            )
        for i in range(3):
            _insert_event(
                conn,
                "email.sent",
                {
                    "to_address": "alice@example.com",
                },
                hours_ago=i * 12 + 2,
            )

        # Bob: 2 received + 1 sent = 3 total
        for i in range(2):
            _insert_event(
                conn,
                "message.received",
                {
                    "from_address": "bob@work.com",
                },
                hours_ago=i * 12 + 3,
            )
        _insert_event(
            conn,
            "message.sent",
            {
                "to_address": "bob@work.com",
            },
            hours_ago=4,
        )
        conn.commit()

    result = assembler._get_contact_activity_summary()

    assert "Active contacts this week (from event log):" in result
    assert "alice@example.com" in result
    assert "bob@work.com" in result

    # Alice should appear before Bob (more interactions)
    alice_pos = result.index("alice@example.com")
    bob_pos = result.index("bob@work.com")
    assert alice_pos < bob_pos

    # Check interaction counts
    assert "8 interactions" in result
    assert "3 interactions" in result


def test_contact_activity_summary_empty_events_db(db):
    """_get_contact_activity_summary returns empty string when no events exist."""
    assembler = _make_assembler(db)

    result = assembler._get_contact_activity_summary()

    assert result == ""


def test_contact_activity_summary_old_events_excluded(db):
    """Events older than 7 days are excluded from the contact summary."""
    assembler = _make_assembler(db)

    with db.get_connection("events") as conn:
        # Insert events older than 7 days (200 hours ago)
        _insert_event(
            conn,
            "email.received",
            {
                "from_address": "old@example.com",
            },
            hours_ago=200,
        )
        conn.commit()

    result = assembler._get_contact_activity_summary()

    assert result == ""


def test_contact_activity_limits_to_top_5(db):
    """_get_contact_activity_summary returns at most 5 contacts."""
    assembler = _make_assembler(db)

    with db.get_connection("events") as conn:
        # Insert events for 7 different contacts
        for i in range(7):
            for j in range(i + 1):
                _insert_event(
                    conn,
                    "email.received",
                    {
                        "from_address": f"user{i}@example.com",
                    },
                    hours_ago=j + 1,
                )
        conn.commit()

    result = assembler._get_contact_activity_summary()

    # Count the number of contact lines (lines starting with "- ")
    contact_lines = [line for line in result.split("\n") if line.startswith("- ")]
    assert len(contact_lines) <= 5


# ------------------------------------------------------------------ #
# assemble_briefing_context fallback integration tests
# ------------------------------------------------------------------ #


def test_briefing_uses_fallbacks_when_user_model_empty(db):
    """assemble_briefing_context returns non-empty content with events.db fallbacks.

    When user_model.db has no episodes/facts/routines/mood, the fallback
    helpers should produce context from events.db instead.
    """
    assembler = _make_assembler(db)

    # Populate events.db with data so fallbacks have material
    with db.get_connection("events") as conn:
        for i in range(5):
            _insert_event(
                conn,
                "email.received",
                {
                    "from_address": "alice@example.com",
                    "subject": f"Important topic {i}",
                },
                hours_ago=i + 1,
            )
        _insert_event(
            conn,
            "email.sent",
            {
                "to_address": "alice@example.com",
            },
            hours_ago=2,
        )
        conn.commit()

    result = assembler.assemble_briefing_context()

    # Should contain the fallback sections
    assert "Recent activity summary" in result or "Active contacts this week" in result
    # Should still contain the non-user-model sections
    assert "Current time:" in result


def test_briefing_skips_fallbacks_when_episodes_exist(db, event_bus):
    """Fallback methods are NOT called when user_model.db has episode data."""
    ums = UserModelStore(db, event_bus=event_bus)
    assembler = ContextAssembler(db, ums)

    # Insert an episode into user_model.db
    cutoff_recent = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    episode_id = str(uuid.uuid4())
    event_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, event_id, timestamp, interaction_type, content_summary,
                contacts_involved, topics, active_domain)
               VALUES (?, ?, ?, 'email_sent', 'Discussed Q1 budget',
                       '["alice@example.com"]', '["finance"]', 'work')""",
            (episode_id, event_id, cutoff_recent),
        )
        conn.commit()

    # Also insert events.db data that would trigger fallbacks
    with db.get_connection("events") as conn:
        for i in range(3):
            _insert_event(
                conn,
                "email.received",
                {
                    "from_address": "fallback@example.com",
                    "subject": "Should not appear",
                },
                hours_ago=i + 1,
            )
        conn.commit()

    result = assembler.assemble_briefing_context()

    # Should have the real episode data, not the fallback
    assert "Discussed Q1 budget" in result
    # Should NOT have the fallback activity summary
    assert "Recent activity summary" not in result


def test_briefing_skips_contact_fallback_when_semantic_facts_exist(db, event_bus):
    """Contact fallback is NOT used when semantic facts exist in user_model.db."""
    ums = UserModelStore(db, event_bus=event_bus)
    assembler = ContextAssembler(db, ums)

    # Insert semantic facts into user_model.db
    # Values must be JSON-encoded because UserModelStore.get_semantic_facts()
    # calls json.loads() on each value.
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO semantic_facts (key, value, confidence, category)
               VALUES ('work_location', '"home_office"', 0.9, 'preferences')""",
        )
        conn.commit()

    # Also insert events.db data that would trigger contact fallback
    with db.get_connection("events") as conn:
        _insert_event(
            conn,
            "email.received",
            {
                "from_address": "contact-fallback@example.com",
            },
            hours_ago=1,
        )
        conn.commit()

    result = assembler.assemble_briefing_context()

    # Should have the real semantic fact
    assert "work_location" in result or "Work location" in result
    # Should NOT have the fallback contact summary
    assert "Active contacts this week" not in result


def test_briefing_graceful_when_both_dbs_empty(db):
    """Briefing doesn't crash when both user_model.db and events.db are empty."""
    assembler = _make_assembler(db)

    result = assembler.assemble_briefing_context()

    # Should still return a valid string with base sections
    assert "Current time:" in result
    assert isinstance(result, str)
    assert len(result) > 0


# ------------------------------------------------------------------ #
# assemble_search_context fallback integration tests
# ------------------------------------------------------------------ #


def test_search_context_uses_contact_fallback(db):
    """assemble_search_context uses contact fallback when semantic facts unavailable."""
    assembler = _make_assembler(db)

    # Populate events.db with contact data
    with db.get_connection("events") as conn:
        for i in range(3):
            _insert_event(
                conn,
                "email.received",
                {
                    "from_address": "search-contact@example.com",
                },
                hours_ago=i * 24 + 1,
            )
        conn.commit()

    result = assembler.assemble_search_context("find emails from search-contact")

    # Should contain the contact fallback
    assert "Active contacts this week" in result
    assert "search-contact@example.com" in result


def test_search_context_skips_fallback_when_facts_exist(db, event_bus):
    """assemble_search_context skips contact fallback when semantic facts are present."""
    ums = UserModelStore(db, event_bus=event_bus)
    assembler = ContextAssembler(db, ums)

    # Insert semantic facts (value must be JSON-encoded for UserModelStore)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO semantic_facts (key, value, confidence, category)
               VALUES ('employer', '"Acme Corp"', 0.9, 'preferences')""",
        )
        conn.commit()

    result = assembler.assemble_search_context("where do I work")

    # Should have real facts, not fallback
    assert "employer" in result
    assert "Active contacts this week" not in result


# ------------------------------------------------------------------ #
# Edge cases and error handling
# ------------------------------------------------------------------ #


def test_recent_activity_summary_handles_malformed_payload(db):
    """_get_recent_activity_summary handles events with non-JSON or missing payload fields."""
    assembler = _make_assembler(db)

    with db.get_connection("events") as conn:
        # Insert event with missing from_address field
        _insert_event(
            conn,
            "email.received",
            {
                "body": "no sender info",
            },
            hours_ago=1,
        )
        # Insert normal event
        _insert_event(
            conn,
            "email.received",
            {
                "from_address": "valid@example.com",
                "subject": "Valid email",
            },
            hours_ago=2,
        )
        conn.commit()

    result = assembler._get_recent_activity_summary()

    # Should still return a summary without crashing
    assert "Recent activity summary" in result
    assert "email.received" in result


def test_contact_activity_uses_sender_field_fallback(db):
    """_get_contact_activity_summary falls back to 'sender' when 'from_address' is null."""
    assembler = _make_assembler(db)

    with db.get_connection("events") as conn:
        # Use 'sender' instead of 'from_address'
        _insert_event(
            conn,
            "email.received",
            {
                "sender": "sender-field@example.com",
            },
            hours_ago=1,
        )
        conn.commit()

    result = assembler._get_contact_activity_summary()

    assert "sender-field@example.com" in result
