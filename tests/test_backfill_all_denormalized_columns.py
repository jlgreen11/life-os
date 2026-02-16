"""
Tests for the denormalized column backfill script.

Verifies that all email, task, and calendar events get their denormalized columns
populated correctly from JSON payloads, enabling efficient workflow detection queries.
"""

import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from backfill_all_denormalized_columns import backfill_denormalized_columns


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary events.db with schema v2 (denormalized columns)."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Create events table with denormalized columns (schema v2)
    conn.executescript("""
        CREATE TABLE events (
            id              TEXT PRIMARY KEY,
            type            TEXT NOT NULL,
            source          TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            priority        TEXT NOT NULL DEFAULT 'normal',
            payload         TEXT NOT NULL DEFAULT '{}',
            metadata        TEXT NOT NULL DEFAULT '{}',
            embedding_id    TEXT,
            created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from      TEXT,
            email_to        TEXT,
            task_id         TEXT,
            calendar_event_id TEXT
        );

        CREATE INDEX idx_events_type ON events(type);
        CREATE INDEX idx_events_timestamp ON events(timestamp);

        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        INSERT INTO schema_version (version) VALUES (2);
    """)
    conn.commit()
    conn.close()

    return str(db_path)


def test_backfill_email_received(temp_db):
    """Test backfilling email_from for email.received events."""
    conn = sqlite3.connect(temp_db)

    # Insert email.received events without denormalized columns
    events = [
        {
            "id": str(uuid.uuid4()),
            "type": "email.received",
            "source": "protonmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": json.dumps({
                "from_address": "Alice@Example.com",  # Mixed case to test LOWER()
                "to_addresses": "user@example.com",
                "subject": "Test Subject",
                "body": "Test body",
            }),
        }
        for _ in range(5)
    ]

    for event in events:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
            (event["id"], event["type"], event["source"], event["timestamp"], event["payload"]),
        )
    conn.commit()

    # Verify columns are NULL before backfill
    cursor = conn.execute("SELECT COUNT(*) FROM events WHERE email_from IS NULL")
    assert cursor.fetchone()[0] == 5

    conn.close()

    # Run backfill
    stats = backfill_denormalized_columns(temp_db)
    assert stats["email_from_received"] == 5

    # Verify columns are populated and lowercased
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT email_from FROM events WHERE type = 'email.received'")
    for row in cursor.fetchall():
        assert row[0] == "alice@example.com"  # Lowercased

    conn.close()


def test_backfill_email_sent(temp_db):
    """Test backfilling email_from and email_to for email.sent events."""
    conn = sqlite3.connect(temp_db)

    # Insert email.sent events
    events = [
        {
            "id": str(uuid.uuid4()),
            "type": "email.sent",
            "source": "protonmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": json.dumps({
                "from_address": "User@Example.com",
                "to_addresses": "Bob@Example.com",
                "subject": "Reply",
                "body": "Reply body",
            }),
        }
        for _ in range(3)
    ]

    for event in events:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
            (event["id"], event["type"], event["source"], event["timestamp"], event["payload"]),
        )
    conn.commit()
    conn.close()

    # Run backfill
    stats = backfill_denormalized_columns(temp_db)
    assert stats["email_from_sent"] == 3
    assert stats["email_to_sent"] == 3

    # Verify both columns populated
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT email_from, email_to FROM events WHERE type = 'email.sent'")
    for row in cursor.fetchall():
        assert row[0] == "user@example.com"
        assert row[1] == "bob@example.com"

    conn.close()


def test_backfill_task_events(temp_db):
    """Test backfilling task_id for task.* events."""
    conn = sqlite3.connect(temp_db)

    task_types = ["task.created", "task.completed", "task.updated"]
    task_id = str(uuid.uuid4())

    for task_type in task_types:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                task_type,
                "task_manager",
                datetime.now(timezone.utc).isoformat(),
                json.dumps({"task_id": task_id, "title": "Test Task"}),
            ),
        )

    conn.commit()
    conn.close()

    # Run backfill
    stats = backfill_denormalized_columns(temp_db)
    assert stats["task_id"] == 3

    # Verify task_id populated
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT task_id FROM events WHERE type LIKE 'task.%'")
    for row in cursor.fetchall():
        assert row[0] == task_id

    conn.close()


def test_backfill_calendar_events(temp_db):
    """Test backfilling calendar_event_id for calendar.event.* events."""
    conn = sqlite3.connect(temp_db)

    event_id = "cal-event-123"
    # Only calendar.event.* types have event_id in payload (not calendar.conflict.detected)
    calendar_types = ["calendar.event.created", "calendar.event.updated"]

    for cal_type in calendar_types:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                cal_type,
                "caldav",
                datetime.now(timezone.utc).isoformat(),
                json.dumps({"event_id": event_id, "summary": "Meeting"}),
            ),
        )

    conn.commit()
    conn.close()

    # Run backfill
    stats = backfill_denormalized_columns(temp_db)
    assert stats["calendar_event_id"] == 2

    # Verify calendar_event_id populated
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT calendar_event_id FROM events WHERE type LIKE 'calendar.event.%'")
    for row in cursor.fetchall():
        assert row[0] == event_id

    conn.close()


def test_backfill_skips_already_populated(temp_db):
    """Test that backfill skips rows where denormalized column is already populated."""
    conn = sqlite3.connect(temp_db)

    # Insert event with email_from already populated
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, payload, email_from) VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            "email.received",
            "protonmail",
            datetime.now(timezone.utc).isoformat(),
            json.dumps({"from_address": "alice@example.com"}),
            "alice@example.com",  # Already populated
        ),
    )

    conn.commit()
    conn.close()

    # Run backfill
    stats = backfill_denormalized_columns(temp_db)

    # Should not update already-populated row
    assert stats["email_from_received"] == 0


def test_backfill_handles_missing_json_fields(temp_db):
    """Test that backfill handles events where JSON payload lacks expected fields."""
    conn = sqlite3.connect(temp_db)

    # Insert event with incomplete payload (no from_address)
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            "email.received",
            "protonmail",
            datetime.now(timezone.utc).isoformat(),
            json.dumps({"subject": "No sender info"}),  # Missing from_address
        ),
    )

    conn.commit()
    conn.close()

    # Run backfill - should not crash
    stats = backfill_denormalized_columns(temp_db)

    # Should skip event with missing field
    assert stats["email_from_received"] == 0

    # Verify column is still NULL
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT email_from FROM events WHERE type = 'email.received'")
    assert cursor.fetchone()[0] is None
    conn.close()


def test_backfill_idempotent(temp_db):
    """Test that backfill can be run multiple times safely (idempotent)."""
    conn = sqlite3.connect(temp_db)

    # Insert event
    event_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
        (
            event_id,
            "email.received",
            "protonmail",
            datetime.now(timezone.utc).isoformat(),
            json.dumps({"from_address": "alice@example.com"}),
        ),
    )
    conn.commit()
    conn.close()

    # First backfill
    stats1 = backfill_denormalized_columns(temp_db)
    assert stats1["email_from_received"] == 1

    # Second backfill - should be no-op
    stats2 = backfill_denormalized_columns(temp_db)
    assert stats2["email_from_received"] == 0

    # Verify data unchanged
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT email_from FROM events WHERE id = ?", (event_id,))
    assert cursor.fetchone()[0] == "alice@example.com"
    conn.close()


def test_backfill_large_dataset(temp_db):
    """Test backfill performance with a large number of events."""
    conn = sqlite3.connect(temp_db)

    # Insert 1000 email.received events
    events = []
    for i in range(1000):
        events.append(
            (
                str(uuid.uuid4()),
                "email.received",
                "protonmail",
                datetime.now(timezone.utc).isoformat(),
                json.dumps({"from_address": f"sender{i}@example.com"}),
            )
        )

    conn.executemany(
        "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
        events,
    )
    conn.commit()
    conn.close()

    # Run backfill
    stats = backfill_denormalized_columns(temp_db)
    assert stats["email_from_received"] == 1000

    # Verify all populated
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT COUNT(*) FROM events WHERE email_from IS NOT NULL")
    assert cursor.fetchone()[0] == 1000
    conn.close()


def test_backfill_mixed_event_types(temp_db):
    """Test backfill with mixed event types (emails, tasks, calendar)."""
    conn = sqlite3.connect(temp_db)

    # Insert diverse event types
    events = [
        ("email.received", json.dumps({"from_address": "alice@example.com"})),
        ("email.sent", json.dumps({"from_address": "user@example.com", "to_addresses": "bob@example.com"})),
        ("task.created", json.dumps({"task_id": "task-1", "title": "Task"})),
        ("calendar.event.created", json.dumps({"event_id": "cal-1", "summary": "Meeting"})),
        ("system.startup", json.dumps({})),  # Event type with no denormalized columns
    ]

    for event_type, payload in events:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), event_type, "test", datetime.now(timezone.utc).isoformat(), payload),
        )

    conn.commit()
    conn.close()

    # Run backfill
    stats = backfill_denormalized_columns(temp_db)

    # Verify each type processed correctly
    assert stats["email_from_received"] == 1
    assert stats["email_from_sent"] == 1
    # email_to is only backfilled for email.sent, not email.received (different field name)
    assert stats["email_to_received"] == 0  # email.received has no to_addresses in payload
    assert stats["email_to_sent"] == 1
    assert stats["task_id"] == 1
    assert stats["calendar_event_id"] == 1

    # Verify system.startup event unaffected
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute(
        "SELECT email_from, email_to, task_id, calendar_event_id FROM events WHERE type = 'system.startup'"
    )
    row = cursor.fetchone()
    assert all(col is None for col in row)
    conn.close()


def test_backfill_coverage_reporting(temp_db):
    """Test that backfill reports accurate coverage statistics."""
    conn = sqlite3.connect(temp_db)

    # Insert 10 email events, 3 with from_address, 7 without
    for i in range(10):
        payload = {}
        if i < 3:
            payload["from_address"] = f"sender{i}@example.com"

        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                "email.received",
                "protonmail",
                datetime.now(timezone.utc).isoformat(),
                json.dumps(payload),
            ),
        )

    conn.commit()
    conn.close()

    # Run backfill
    stats = backfill_denormalized_columns(temp_db)

    # Should update 3 events (the ones with from_address in payload)
    assert stats["email_from_received"] == 3

    # Verify 3 populated, 7 NULL
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT COUNT(*) FROM events WHERE email_from IS NOT NULL")
    assert cursor.fetchone()[0] == 3

    cursor = conn.execute("SELECT COUNT(*) FROM events WHERE email_from IS NULL")
    assert cursor.fetchone()[0] == 7

    conn.close()
