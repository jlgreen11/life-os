"""
Tests for denormalized column backfill script.

Verifies that the backfill script correctly populates email_from, email_to,
task_id, and calendar_event_id columns from event payloads.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import sys

# Import the backfill script functions
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from backfill_denormalized_columns import (
    backfill_email_received_events,
    backfill_email_sent_events,
    backfill_task_events,
    backfill_calendar_events,
)


def test_backfill_email_received_basic(tmp_path):
    """Test backfilling email_from for email.received events."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))

    # Create events table with denormalized columns
    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            payload TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from TEXT,
            email_to TEXT,
            task_id TEXT,
            calendar_event_id TEXT
        )
    """)

    # Insert email.received events with NULL email_from
    events = [
        {
            "id": "email-1",
            "type": "email.received",
            "source": "protonmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": json.dumps({"from_address": "Boss@Example.Com", "subject": "Test"}),
        },
        {
            "id": "email-2",
            "type": "email.received",
            "source": "protonmail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": json.dumps({"from_address": "client@company.org", "subject": "Question"}),
        },
    ]

    for event in events:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (event["id"], event["type"], event["source"], event["timestamp"], event["priority"], event["payload"])
        )
    conn.commit()

    # Run backfill
    count = backfill_email_received_events(conn)

    # Verify
    assert count == 2

    cursor = conn.cursor()
    cursor.execute("SELECT id, email_from FROM events ORDER BY id")
    results = cursor.fetchall()

    assert results[0][1] == "boss@example.com"  # Lowercased
    assert results[1][1] == "client@company.org"

    conn.close()


def test_backfill_email_received_ignores_already_filled(tmp_path):
    """Test that backfill skips events that already have email_from populated."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            payload TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from TEXT,
            email_to TEXT,
            task_id TEXT,
            calendar_event_id TEXT
        )
    """)

    # Insert event with email_from already populated
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload, email_from) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("email-1", "email.received", "protonmail", datetime.now(timezone.utc).isoformat(), "normal",
         json.dumps({"from_address": "new@example.com"}), "existing@example.com")
    )
    conn.commit()

    # Run backfill
    count = backfill_email_received_events(conn)

    # Should skip the already-filled event
    assert count == 0

    cursor = conn.cursor()
    cursor.execute("SELECT email_from FROM events WHERE id = 'email-1'")
    result = cursor.fetchone()
    assert result[0] == "existing@example.com"  # Unchanged

    conn.close()


def test_backfill_email_received_large_batch(tmp_path):
    """Test backfilling large number of events in batches."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            payload TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from TEXT,
            email_to TEXT,
            task_id TEXT,
            calendar_event_id TEXT
        )
    """)

    # Insert 12,000 events (tests batch processing at 5000 per batch)
    timestamp = datetime.now(timezone.utc).isoformat()
    for i in range(12000):
        payload = json.dumps({"from_address": f"sender{i}@example.com", "subject": f"Email {i}"})
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (f"email-{i}", "email.received", "protonmail", timestamp, "normal", payload)
        )

    conn.commit()

    # Run backfill
    count = backfill_email_received_events(conn)

    # Verify all events backfilled
    assert count == 12000

    # Verify sample events
    cursor = conn.cursor()
    cursor.execute("SELECT email_from FROM events WHERE id = 'email-0'")
    assert cursor.fetchone()[0] == "sender0@example.com"

    cursor.execute("SELECT email_from FROM events WHERE id = 'email-11999'")
    assert cursor.fetchone()[0] == "sender11999@example.com"

    # Verify no NULL values remain
    cursor.execute("SELECT COUNT(*) FROM events WHERE email_from IS NULL")
    assert cursor.fetchone()[0] == 0

    conn.close()


def test_backfill_email_sent_events(tmp_path):
    """Test backfilling email_from and email_to for email.sent events."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            payload TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from TEXT,
            email_to TEXT,
            task_id TEXT,
            calendar_event_id TEXT
        )
    """)

    # Insert email.sent events
    payload = json.dumps({
        "from_address": "Me@Example.Com",
        "to_addresses": "Boss@Company.Org",
        "subject": "Re: Question"
    })
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
        ("sent-1", "email.sent", "protonmail", datetime.now(timezone.utc).isoformat(), "normal", payload)
    )
    conn.commit()

    # Run backfill
    from_count, to_count = backfill_email_sent_events(conn)

    # Verify
    assert from_count == 1
    assert to_count == 1

    cursor = conn.cursor()
    cursor.execute("SELECT email_from, email_to FROM events WHERE id = 'sent-1'")
    result = cursor.fetchone()
    assert result[0] == "me@example.com"  # Lowercased
    assert result[1] == "boss@company.org"  # Lowercased

    conn.close()


def test_backfill_task_events(tmp_path):
    """Test backfilling task_id for task.* events."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            payload TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from TEXT,
            email_to TEXT,
            task_id TEXT,
            calendar_event_id TEXT
        )
    """)

    # Insert task events
    events = [
        ("task-evt-1", "task.created", json.dumps({"task_id": "task-123", "title": "Fix bug"})),
        ("task-evt-2", "task.completed", json.dumps({"task_id": "task-123"})),
        ("task-evt-3", "task.updated", json.dumps({"task_id": "task-456", "status": "in_progress"})),
    ]

    timestamp = datetime.now(timezone.utc).isoformat()
    for event_id, event_type, payload in events:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, event_type, "task_manager", timestamp, "normal", payload)
        )
    conn.commit()

    # Run backfill
    count = backfill_task_events(conn)

    # Verify
    assert count == 3

    cursor = conn.cursor()
    cursor.execute("SELECT id, task_id FROM events WHERE type LIKE 'task.%' ORDER BY id")
    results = cursor.fetchall()

    assert results[0][1] == "task-123"
    assert results[1][1] == "task-123"
    assert results[2][1] == "task-456"

    conn.close()


def test_backfill_calendar_events(tmp_path):
    """Test backfilling calendar_event_id for calendar.event.* events."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            payload TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from TEXT,
            email_to TEXT,
            task_id TEXT,
            calendar_event_id TEXT
        )
    """)

    # Insert calendar events
    events = [
        ("cal-evt-1", "calendar.event.created", json.dumps({"event_id": "cal-123", "title": "Meeting"})),
        ("cal-evt-2", "calendar.event.updated", json.dumps({"event_id": "cal-123", "title": "Updated Meeting"})),
        ("cal-evt-3", "calendar.event.deleted", json.dumps({"event_id": "cal-456"})),
    ]

    timestamp = datetime.now(timezone.utc).isoformat()
    for event_id, event_type, payload in events:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, event_type, "caldav", timestamp, "normal", payload)
        )
    conn.commit()

    # Run backfill
    count = backfill_calendar_events(conn)

    # Verify
    assert count == 3

    cursor = conn.cursor()
    cursor.execute("SELECT id, calendar_event_id FROM events WHERE type LIKE 'calendar.event.%' ORDER BY id")
    results = cursor.fetchall()

    assert results[0][1] == "cal-123"
    assert results[1][1] == "cal-123"
    assert results[2][1] == "cal-456"

    conn.close()


def test_backfill_ignores_events_without_payload_fields(tmp_path):
    """Test that backfill skips events with missing payload fields."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            payload TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from TEXT,
            email_to TEXT,
            task_id TEXT,
            calendar_event_id TEXT
        )
    """)

    # Insert events without the expected payload fields
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
        ("email-1", "email.received", "protonmail", timestamp, "normal", json.dumps({"subject": "No sender"}))
    )
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
        ("task-1", "task.created", "task_manager", timestamp, "normal", json.dumps({"title": "No task_id"}))
    )
    conn.commit()

    # Run backfill
    email_count = backfill_email_received_events(conn)
    task_count = backfill_task_events(conn)

    # Should skip events without required fields
    assert email_count == 0
    assert task_count == 0

    cursor = conn.cursor()
    cursor.execute("SELECT email_from, task_id FROM events")
    results = cursor.fetchall()
    assert results[0][0] is None
    assert results[1][1] is None

    conn.close()


def test_backfill_handles_malformed_json(tmp_path):
    """Test that backfill handles events with malformed JSON gracefully."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            payload TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from TEXT,
            email_to TEXT,
            task_id TEXT,
            calendar_event_id TEXT
        )
    """)

    # Insert event with malformed JSON
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
        ("email-1", "email.received", "protonmail", timestamp, "normal", "{not valid json")
    )

    # Insert valid event
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
        ("email-2", "email.received", "protonmail", timestamp, "normal", json.dumps({"from_address": "valid@example.com"}))
    )
    conn.commit()

    # Run backfill - should handle malformed JSON gracefully
    count = backfill_email_received_events(conn)

    # Should only backfill the valid event
    assert count == 1

    cursor = conn.cursor()
    cursor.execute("SELECT id, email_from FROM events ORDER BY id")
    results = cursor.fetchall()
    assert results[0][1] is None  # Malformed JSON skipped
    assert results[1][1] == "valid@example.com"  # Valid event backfilled

    conn.close()


def test_backfill_preserves_case_sensitivity_in_payload(tmp_path):
    """Test that email addresses are lowercased but payload is preserved."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            payload TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from TEXT,
            email_to TEXT,
            task_id TEXT,
            calendar_event_id TEXT
        )
    """)

    # Insert event with mixed-case email
    original_payload = json.dumps({"from_address": "Boss@Example.Com", "subject": "IMPORTANT"})
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
        ("email-1", "email.received", "protonmail", datetime.now(timezone.utc).isoformat(), "normal", original_payload)
    )
    conn.commit()

    # Run backfill
    backfill_email_received_events(conn)

    # Verify
    cursor = conn.cursor()
    cursor.execute("SELECT email_from, payload FROM events WHERE id = 'email-1'")
    result = cursor.fetchone()

    # Denormalized column should be lowercased
    assert result[0] == "boss@example.com"

    # Original payload should be unchanged
    assert result[1] == original_payload

    conn.close()


def test_backfill_multiple_event_types_together(tmp_path):
    """Integration test: backfill all event types in sequence."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            payload TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            email_from TEXT,
            email_to TEXT,
            task_id TEXT,
            calendar_event_id TEXT
        )
    """)

    # Insert mix of event types
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
        ("email-recv-1", "email.received", "protonmail", timestamp, "normal",
         json.dumps({"from_address": "sender@example.com"}))
    )
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
        ("email-sent-1", "email.sent", "protonmail", timestamp, "normal",
         json.dumps({"from_address": "me@example.com", "to_addresses": "recipient@example.com"}))
    )
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
        ("task-1", "task.created", "task_manager", timestamp, "normal",
         json.dumps({"task_id": "task-123"}))
    )
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
        ("cal-1", "calendar.event.created", "caldav", timestamp, "normal",
         json.dumps({"event_id": "cal-123"}))
    )
    conn.commit()

    # Run all backfills
    email_recv_count = backfill_email_received_events(conn)
    sent_from_count, sent_to_count = backfill_email_sent_events(conn)
    task_count = backfill_task_events(conn)
    cal_count = backfill_calendar_events(conn)

    # Verify counts
    assert email_recv_count == 1
    assert sent_from_count == 1
    assert sent_to_count == 1
    assert task_count == 1
    assert cal_count == 1

    # Verify all columns populated correctly
    cursor = conn.cursor()
    cursor.execute("SELECT id, email_from, email_to, task_id, calendar_event_id FROM events ORDER BY id")
    results = cursor.fetchall()

    # cal-1
    assert results[0] == ("cal-1", None, None, None, "cal-123")
    # email-recv-1
    assert results[1] == ("email-recv-1", "sender@example.com", None, None, None)
    # email-sent-1
    assert results[2] == ("email-sent-1", "me@example.com", "recipient@example.com", None, None)
    # task-1
    assert results[3] == ("task-1", None, None, "task-123", None)

    conn.close()
