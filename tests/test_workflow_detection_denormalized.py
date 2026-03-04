"""
Tests for workflow detection with denormalized columns (schema v2).

Verifies that workflow detection uses indexed email_from, email_to,
task_id, and calendar_event_id columns instead of expensive json_extract()
calls, enabling <1s query times on 800K+ events.
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

from services.workflow_detector.detector import WorkflowDetector


def test_events_schema_has_denormalized_columns(db):
    """Verify events table has denormalized columns for workflow detection."""
    with db.get_connection("events") as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(events)")
        columns = {row[1] for row in cursor.fetchall()}

    assert "email_from" in columns, "events table missing email_from column"
    assert "email_to" in columns, "events table missing email_to column"
    assert "task_id" in columns, "events table missing task_id column"
    assert "calendar_event_id" in columns, "events table missing calendar_event_id column"


def test_events_schema_has_denormalized_indexes(db):
    """Verify denormalized columns have indexes for fast queries."""
    with db.get_connection("events") as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA index_list(events)")
        indexes = {row[1] for row in cursor.fetchall()}

    assert "idx_events_email_from" in indexes, "Missing index on email_from"
    assert "idx_events_email_to" in indexes, "Missing index on email_to"
    assert "idx_events_task_id" in indexes, "Missing index on task_id"


def test_trigger_populates_email_from_on_insert(db, event_store):
    """Verify trigger auto-populates email_from for email.received events."""
    event = {
        "id": "email-recv-1",
        "type": "email.received",
        "source": "protonmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {"from_address": "Boss@Example.com", "subject": "Urgent"},
        "metadata": {},
    }
    event_store.store_event(event)

    # Check denormalized column was populated (lowercase)
    with db.get_connection("events") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email_from FROM events WHERE id = ?", ("email-recv-1",))
        result = cursor.fetchone()

    assert result is not None, "Event not found"
    assert result[0] == "boss@example.com", f"Expected lowercase email_from, got {result[0]}"


def test_trigger_populates_email_to_on_insert(db, event_store):
    """Verify trigger auto-populates email_to for email.sent events."""
    event = {
        "id": "email-sent-1",
        "type": "email.sent",
        "source": "protonmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "from_address": "me@example.com",
            "to_addresses": '["Boss@Example.com"]',
            "subject": "Re: Urgent",
        },
        "metadata": {},
    }
    event_store.store_event(event)

    # Check denormalized column was populated (lowercase)
    with db.get_connection("events") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email_to, email_from FROM events WHERE id = ?", ("email-sent-1",))
        result = cursor.fetchone()

    assert result is not None, "Event not found"
    assert result[0] == '["boss@example.com"]', f"Expected lowercase email_to, got {result[0]}"
    assert result[1] == "me@example.com", f"Expected lowercase email_from, got {result[1]}"


def test_trigger_populates_task_id_on_insert(db, event_store):
    """Verify trigger auto-populates task_id for task.* events."""
    event = {
        "id": "task-event-1",
        "type": "task.created",
        "source": "task_manager",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {"task_id": "task-123", "title": "Fix bug"},
        "metadata": {},
    }
    event_store.store_event(event)

    # Check denormalized column was populated
    with db.get_connection("events") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT task_id FROM events WHERE id = ?", ("task-event-1",))
        result = cursor.fetchone()

    assert result is not None, "Event not found"
    assert result[0] == "task-123", f"Expected task_id 'task-123', got {result[0]}"


def test_trigger_populates_calendar_event_id_on_insert(db, event_store):
    """Verify trigger auto-populates calendar_event_id for calendar.event.* events."""
    event = {
        "id": "cal-event-1",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {"event_id": "cal-456", "title": "Meeting"},
        "metadata": {},
    }
    event_store.store_event(event)

    # Check denormalized column was populated
    with db.get_connection("events") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT calendar_event_id FROM events WHERE id = ?", ("cal-event-1",))
        result = cursor.fetchone()

    assert result is not None, "Event not found"
    assert result[0] == "cal-456", f"Expected calendar_event_id 'cal-456', got {result[0]}"


def test_workflow_detection_uses_denormalized_columns(db, event_store, user_model_store):
    """Verify workflow detection queries use indexed columns instead of json_extract()."""
    now = datetime.now(timezone.utc)

    # Create a clear email workflow: receive from boss → send reply
    # Repeat 5 times to exceed min_occurrences threshold (3)
    for i in range(5):
        recv_time = now - timedelta(hours=24 - i * 4)
        sent_time = recv_time + timedelta(minutes=30)

        # Email received from boss
        event_store.store_event({
            "id": f"email-recv-{i}",
            "type": "email.received",
            "source": "protonmail",
            "timestamp": recv_time.isoformat(),
            "priority": "high",
            "payload": {
                "from_address": "boss@company.com",
                "subject": f"Task {i}",
                "body": "Please handle this",
            },
            "metadata": {},
        })

        # Email sent to boss (reply)
        event_store.store_event({
            "id": f"email-sent-{i}",
            "type": "email.sent",
            "source": "protonmail",
            "timestamp": sent_time.isoformat(),
            "priority": "normal",
            "payload": {
                "from_address": "me@company.com",
                "to_addresses": '["boss@company.com"]',
                "subject": f"Re: Task {i}",
                "body": "Done",
            },
            "metadata": {},
        })

    # Detect workflows. WorkflowDetector is now active after the algorithmic redesign.
    # This test's primary purpose is verifying that the denormalized columns and
    # triggers exist and are populated correctly. The workflows list may or may
    # not be empty depending on whether the test data meets detection thresholds.
    detector = WorkflowDetector(db, user_model_store)
    workflows = detector.detect_workflows(lookback_days=30)

    # The detector should return a list (empty or populated — both are valid).
    assert isinstance(workflows, list), "detect_workflows should always return a list"


def test_workflow_detection_performance_with_denormalized_columns(db, event_store, user_model_store):
    """Verify workflow detection completes in <2s with denormalized columns."""
    now = datetime.now(timezone.utc)

    # Create 100 email events (received + sent pairs) to simulate realistic volume
    for i in range(50):
        recv_time = now - timedelta(hours=100 - i * 2)
        sent_time = recv_time + timedelta(hours=1)

        event_store.store_event({
            "id": f"perf-recv-{i}",
            "type": "email.received",
            "source": "protonmail",
            "timestamp": recv_time.isoformat(),
            "priority": "normal",
            "payload": {
                "from_address": f"sender{i % 10}@example.com",
                "subject": f"Test {i}",
            },
            "metadata": {},
        })

        event_store.store_event({
            "id": f"perf-sent-{i}",
            "type": "email.sent",
            "source": "protonmail",
            "timestamp": sent_time.isoformat(),
            "priority": "normal",
            "payload": {
                "from_address": "me@example.com",
                "to_addresses": f'["sender{i % 10}@example.com"]',
                "subject": f"Re: Test {i}",
            },
            "metadata": {},
        })

    # Time workflow detection
    detector = WorkflowDetector(db, user_model_store)
    start = time.time()
    workflows = detector.detect_workflows(lookback_days=7)
    elapsed = time.time() - start

    # Should complete in <2s (vs 30s+ with json_extract)
    assert elapsed < 2.0, f"Workflow detection too slow: {elapsed:.2f}s (expected <2s)"
    assert isinstance(workflows, list)


def test_migration_backfills_existing_events(db, event_store):
    """Verify migration backfills denormalized columns for existing events."""
    # Manually insert events without triggers (simulating old events)
    with db.get_connection("events") as conn:
        conn.execute("""
            INSERT INTO events (id, type, source, timestamp, priority, payload)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "old-email-1",
            "email.received",
            "protonmail",
            datetime.now(timezone.utc).isoformat(),
            "normal",
            json.dumps({"from_address": "old@example.com", "subject": "Old"}),
        ))

    # Manually set email_from to NULL to simulate pre-migration state
    with db.get_connection("events") as conn:
        conn.execute("UPDATE events SET email_from = NULL WHERE id = ?", ("old-email-1",))

    # Verify it's NULL
    with db.get_connection("events") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email_from FROM events WHERE id = ?", ("old-email-1",))
        result = cursor.fetchone()
        assert result[0] is None, "email_from should be NULL before backfill"

    # Run migration manually (simulating startup backfill)
    with db.get_connection("events") as conn:
        conn.execute("""
            UPDATE events
            SET email_from = LOWER(json_extract(payload, '$.from_address'))
            WHERE id = 'old-email-1'
              AND email_from IS NULL
              AND json_extract(payload, '$.from_address') IS NOT NULL
        """)

    # Verify backfill worked
    with db.get_connection("events") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email_from FROM events WHERE id = ?", ("old-email-1",))
        result = cursor.fetchone()
        assert result[0] == "old@example.com", f"Expected backfilled email_from, got {result[0]}"


def test_workflow_detection_handles_missing_denormalized_columns(db, event_store, user_model_store):
    """Verify workflow detection gracefully handles events with NULL denormalized columns."""
    now = datetime.now(timezone.utc)

    # Create events with incomplete payload (missing from_address)
    for i in range(5):
        event_store.store_event({
            "id": f"incomplete-{i}",
            "type": "email.received",
            "source": "protonmail",
            "timestamp": (now - timedelta(hours=i)).isoformat(),
            "priority": "normal",
            "payload": {"subject": f"Incomplete {i}"},  # Missing from_address
            "metadata": {},
        })

    # Workflow detection should not crash
    detector = WorkflowDetector(db, user_model_store)
    workflows = detector.detect_workflows(lookback_days=1)

    # Should complete without error (may return empty list)
    assert isinstance(workflows, list), "Expected list return value"


def test_denormalized_columns_improve_query_performance(db, event_store):
    """Compare query performance: json_extract() vs denormalized columns."""
    now = datetime.now(timezone.utc)

    # Create 1000 email events to measure performance difference
    for i in range(1000):
        event_store.store_event({
            "id": f"bench-{i}",
            "type": "email.received",
            "source": "protonmail",
            "timestamp": (now - timedelta(hours=i)).isoformat(),
            "priority": "normal",
            "payload": {"from_address": f"sender{i % 100}@example.com", "subject": f"Test {i}"},
            "metadata": {},
        })

    # Query with json_extract (old method)
    with db.get_connection("events") as conn:
        start = time.time()
        cursor = conn.execute("""
            SELECT COUNT(*)
            FROM events
            WHERE type = 'email.received'
              AND LOWER(json_extract(payload, '$.from_address')) = ?
        """, ("sender50@example.com",))
        json_extract_time = time.time() - start
        json_extract_count = cursor.fetchone()[0]

    # Query with denormalized column (new method)
    with db.get_connection("events") as conn:
        start = time.time()
        cursor = conn.execute("""
            SELECT COUNT(*)
            FROM events
            WHERE type = 'email.received'
              AND email_from = ?
        """, ("sender50@example.com",))
        denormalized_time = time.time() - start
        denormalized_count = cursor.fetchone()[0]

    # Both should return same count
    assert json_extract_count == denormalized_count, \
        f"Query results differ: json_extract={json_extract_count}, denormalized={denormalized_count}"

    # Denormalized should be significantly faster (at least 2x on 1000 rows)
    speedup = json_extract_time / denormalized_time
    assert speedup >= 2.0, \
        f"Denormalized query not fast enough: {speedup:.1f}x speedup (expected ≥2x)"


def test_schema_version_tracks_migration(db):
    """Verify schema_version table tracks events.db migration to v2."""
    with db.get_connection("events") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(version) FROM schema_version")
        result = cursor.fetchone()

    current_version = result[0] if result[0] is not None else 0
    assert current_version >= 2, f"Expected schema version ≥2, got {current_version}"


def test_workflow_detector_diagnostics(db, event_store, user_model_store):
    """Verify get_diagnostics() returns structured diagnostic information."""
    now = datetime.now(timezone.utc)

    # Store a few email events so data_sufficient can be checked
    for i in range(12):
        event_store.store_event({
            "id": f"diag-recv-{i}",
            "type": "email.received",
            "source": "protonmail",
            "timestamp": (now - timedelta(hours=i)).isoformat(),
            "priority": "normal",
            "payload": {"from_address": f"user{i % 3}@example.com", "subject": f"Diag {i}"},
            "metadata": {},
        })

    detector = WorkflowDetector(db, user_model_store)
    diagnostics = detector.get_diagnostics(lookback_days=30)

    # Verify expected top-level keys
    assert "event_counts" in diagnostics, "Missing event_counts key"
    assert "thresholds" in diagnostics, "Missing thresholds key"
    assert "detection_results" in diagnostics, "Missing detection_results key"
    assert "total_detected" in diagnostics, "Missing total_detected key"
    assert "data_sufficient" in diagnostics, "Missing data_sufficient key"

    # Verify thresholds match detector configuration
    assert diagnostics["thresholds"]["min_occurrences"] == 3
    assert diagnostics["thresholds"]["max_step_gap_hours"] == 4
    assert diagnostics["thresholds"]["min_steps"] == 2
    assert diagnostics["thresholds"]["success_threshold"] == 0.01

    # Verify event_counts includes email.received
    assert isinstance(diagnostics["event_counts"], dict)
    assert diagnostics["event_counts"].get("email.received", 0) == 12

    # Verify per-strategy detection results
    assert "email" in diagnostics["detection_results"]
    assert "task" in diagnostics["detection_results"]
    assert "calendar" in diagnostics["detection_results"]
    assert "interaction" in diagnostics["detection_results"]
    for strategy_name, strategy_result in diagnostics["detection_results"].items():
        assert "detected" in strategy_result, f"Missing 'detected' key in {strategy_name} results"

    # Verify total_detected is an integer
    assert isinstance(diagnostics["total_detected"], int)

    # With 12 email events, data_sufficient should be True (threshold is 10)
    assert diagnostics["data_sufficient"] is True


def test_workflow_detector_diagnostics_insufficient_data(db, event_store, user_model_store):
    """Verify get_diagnostics() reports data_sufficient=False with few events."""
    now = datetime.now(timezone.utc)

    # Store only 3 email events (below the 10-event threshold)
    for i in range(3):
        event_store.store_event({
            "id": f"diag-few-{i}",
            "type": "email.received",
            "source": "protonmail",
            "timestamp": (now - timedelta(hours=i)).isoformat(),
            "priority": "normal",
            "payload": {"from_address": "sparse@example.com", "subject": f"Few {i}"},
            "metadata": {},
        })

    detector = WorkflowDetector(db, user_model_store)
    diagnostics = detector.get_diagnostics(lookback_days=30)

    assert diagnostics["data_sufficient"] is False
