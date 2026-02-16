"""
Tests for decision profile backfill script.

Verifies that the backfill script correctly processes historical events
through the DecisionExtractor and populates the decision profile with
decision speed, delegation patterns, risk tolerance, and fatigue indicators.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta

from scripts.backfill_decision_profile import backfill_decision_profile
from models.core import EventType


def test_backfill_decision_profile_processes_decision_events(db, user_model_store, tmp_path):
    """Backfill should process all decision-making event types."""
    # Create a variety of decision-making events
    events = [
        # Task completed (decision execution speed)
        {
            "id": "evt-task-created-1",
            "type": EventType.TASK_CREATED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T09:00:00Z",
            "priority": "normal",
            "payload": {"task_id": "task-1", "title": "Review PR", "domain": "work"},
            "metadata": {},
        },
        {
            "id": "evt-task-completed-1",
            "type": EventType.TASK_COMPLETED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T11:00:00Z",  # 2 hours later
            "priority": "normal",
            "payload": {"task_id": "task-1"},
            "metadata": {},
        },
        # Email with delegation pattern
        {
            "id": "evt-email-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:30:00Z",
            "priority": "normal",
            "payload": {
                "to": "partner@example.com",
                "body": "What do you think about this restaurant?",
            },
            "metadata": {},
        },
        # Message with full delegation
        {
            "id": "evt-msg-1",
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": "2026-02-10T20:30:00Z",  # Late evening (fatigue hour)
            "priority": "normal",
            "payload": {"to": "555-0100", "body": "You decide, I don't care"},
            "metadata": {},
        },
        # Calendar event (commitment pattern)
        {
            "id": "evt-cal-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Team meeting",
                "start_time": "2026-02-17T14:00:00Z",  # Scheduled 7 days in advance
            },
            "metadata": {},
        },
    ]

    # Insert events into events.db
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_decision_profile(data_dir=db.data_dir, batch_size=100)

    # Verify events were processed
    assert result["events_processed"] >= 3  # At least task completion, email, message, calendar
    assert result["signals_extracted"] >= 3  # At least one signal per decision event
    assert result["errors"] == 0

    # Verify decision profile was created
    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    assert profile["samples_count"] > 0

    # Verify profile data structure
    data = profile["data"]
    assert "decision_speed_by_domain" in data
    assert "delegation_comfort" in data
    assert "risk_tolerance_by_domain" in data
    assert "fatigue_time_of_day" in data


def test_backfill_decision_profile_tracks_decision_speed(db, user_model_store):
    """Backfill should track decision speed from task creation to completion."""
    # Create task with known decision speed
    events = [
        # Task created
        {
            "id": "evt-task-created-1",
            "type": EventType.TASK_CREATED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T09:00:00Z",
            "priority": "normal",
            "payload": {
                "task_id": "task-immediate",
                "title": "Quick fix",
                "domain": "work",
            },
            "metadata": {},
        },
        # Task completed 30 minutes later (immediate)
        {
            "id": "evt-task-completed-1",
            "type": EventType.TASK_COMPLETED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T09:30:00Z",
            "priority": "normal",
            "payload": {"task_id": "task-immediate"},
            "metadata": {},
        },
        # Another task created
        {
            "id": "evt-task-created-2",
            "type": EventType.TASK_CREATED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {
                "task_id": "task-deliberate",
                "title": "Research",
                "domain": "research",
            },
            "metadata": {},
        },
        # Task completed 3 days later (multi-day)
        {
            "id": "evt-task-completed-2",
            "type": EventType.TASK_COMPLETED.value,
            "source": "web_ui",
            "timestamp": "2026-02-13T10:00:00Z",
            "priority": "normal",
            "payload": {"task_id": "task-deliberate"},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_decision_profile(data_dir=db.data_dir)

    # Verify decision speed signals were extracted
    assert result["decision_speed_samples"] == 2

    # Verify decision speed by domain was populated
    profile = user_model_store.get_signal_profile("decision")
    decision_speed = profile["data"]["decision_speed_by_domain"]

    assert "work" in decision_speed
    assert "research" in decision_speed

    # Work domain should be fast (30 minutes = 1800 seconds)
    assert decision_speed["work"] < 7200  # Less than 2 hours

    # Research domain should be slow (3 days = 259200 seconds)
    assert decision_speed["research"] > 86400  # More than 1 day


def test_backfill_decision_profile_detects_delegation_patterns(db, user_model_store):
    """Backfill should detect delegation patterns in outbound messages."""
    events = [
        # Opinion-seeking (soft delegation)
        {
            "id": "evt-email-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {
                "to": "colleague@example.com",
                "body": "What do you think I should do about this?",
            },
            "metadata": {},
        },
        # Full delegation
        {
            "id": "evt-msg-1",
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": "2026-02-10T15:00:00Z",
            "priority": "normal",
            "payload": {"to": "555-0100", "body": "You decide, whatever you prefer"},
            "metadata": {},
        },
        # Another delegation at night (fatigue indicator)
        {
            "id": "evt-msg-2",
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": "2026-02-10T22:00:00Z",  # 10pm
            "priority": "normal",
            "payload": {"to": "555-0200", "body": "I don't care, your call"},
            "metadata": {},
        },
        # Non-delegation message (control)
        {
            "id": "evt-msg-3",
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": "2026-02-10T16:00:00Z",
            "priority": "normal",
            "payload": {"to": "555-0300", "body": "Let's meet at 3pm tomorrow"},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_decision_profile(data_dir=db.data_dir)

    # Verify delegation signals were extracted
    assert result["delegation_samples"] == 3

    # Verify fatigue time detection (delegation after 8pm)
    profile = user_model_store.get_signal_profile("decision")
    fatigue_hour = profile["data"].get("fatigue_time_of_day")
    assert fatigue_hour is not None
    assert fatigue_hour >= 20  # At or after 8pm


def test_backfill_decision_profile_tracks_commitment_patterns(db, user_model_store):
    """Backfill should track calendar commitment patterns (planning horizon)."""
    now = datetime.now(timezone.utc)
    events = [
        # Immediate scheduling (same day)
        {
            "id": "evt-cal-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Lunch meeting",
                "start_time": (now + timedelta(hours=3)).isoformat(),
            },
            "metadata": {},
        },
        # Week-ahead scheduling (work meeting)
        {
            "id": "evt-cal-2",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Team standup",
                "start_time": (now + timedelta(days=7)).isoformat(),
            },
            "metadata": {},
        },
        # Long-term scheduling (dentist appointment)
        {
            "id": "evt-cal-3",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Dentist checkup",
                "start_time": (now + timedelta(days=30)).isoformat(),
            },
            "metadata": {},
        },
        # Social event (dinner)
        {
            "id": "evt-cal-4",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Dinner with friends",
                "start_time": (now + timedelta(days=2)).isoformat(),
            },
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_decision_profile(data_dir=db.data_dir)

    # Verify commitment signals were extracted
    assert result["commitment_samples"] == 4

    # Verify risk tolerance by domain was populated
    profile = user_model_store.get_signal_profile("decision")
    risk_tolerance = profile["data"]["risk_tolerance_by_domain"]

    # Should have detected multiple domains
    assert len(risk_tolerance) >= 3

    # Social events scheduled sooner should have higher risk tolerance (more spontaneous)
    if "social" in risk_tolerance:
        assert risk_tolerance["social"] > 0

    # Health events scheduled far in advance should have lower risk (more cautious)
    if "health" in risk_tolerance:
        assert risk_tolerance["health"] >= 0


def test_backfill_decision_profile_classifies_event_domains(db, user_model_store):
    """Backfill should classify calendar events into decision domains."""
    now = datetime.now(timezone.utc)
    events = [
        # Work domain
        {
            "id": "evt-work",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Team standup meeting",
                "start_time": (now + timedelta(days=1)).isoformat(),
            },
            "metadata": {},
        },
        # Social domain
        {
            "id": "evt-social",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Coffee with friends",
                "start_time": (now + timedelta(days=2)).isoformat(),
            },
            "metadata": {},
        },
        # Health domain
        {
            "id": "evt-health",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Doctor appointment",
                "start_time": (now + timedelta(days=14)).isoformat(),
            },
            "metadata": {},
        },
        # Finance domain
        {
            "id": "evt-finance",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Bank meeting",
                "start_time": (now + timedelta(days=7)).isoformat(),
            },
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_decision_profile(data_dir=db.data_dir)

    # Verify all domains were detected
    profile = user_model_store.get_signal_profile("decision")
    risk_tolerance = profile["data"]["risk_tolerance_by_domain"]

    assert "work" in risk_tolerance
    assert "social" in risk_tolerance
    assert "health" in risk_tolerance
    assert "finance" in risk_tolerance


def test_backfill_decision_profile_handles_missing_task_creation(db, user_model_store):
    """Backfill should gracefully handle task completions without matching creation events."""
    events = [
        # Task completion without creation event (orphaned)
        {
            "id": "evt-orphan",
            "type": EventType.TASK_COMPLETED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {"task_id": "nonexistent-task"},
            "metadata": {},
        },
        # Valid task pair
        {
            "id": "evt-task-created",
            "type": EventType.TASK_CREATED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T15:00:00Z",
            "priority": "normal",
            "payload": {"task_id": "valid-task", "domain": "work"},
            "metadata": {},
        },
        {
            "id": "evt-task-completed",
            "type": EventType.TASK_COMPLETED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T16:00:00Z",
            "priority": "normal",
            "payload": {"task_id": "valid-task"},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_decision_profile(data_dir=db.data_dir)

    # Should process all decision-making events (both completions + one creation)
    # but only extract decision_speed signal from the valid task completion
    assert result["events_processed"] == 3  # Both completions + one creation
    assert result["decision_speed_samples"] == 1  # Only one valid signal (orphaned completion has no creation event)
    assert result["errors"] == 0  # No errors (graceful handling)


def test_backfill_decision_profile_ignores_past_calendar_events(db, user_model_store):
    """Backfill should ignore calendar events scheduled in the past (imported events)."""
    now = datetime.now(timezone.utc)
    events = [
        # Past event (imported from sync, should ignore)
        {
            "id": "evt-past",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Past meeting",
                "start_time": (now - timedelta(days=7)).isoformat(),  # In the past
            },
            "metadata": {},
        },
        # Future event (should process)
        {
            "id": "evt-future",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Future meeting",
                "start_time": (now + timedelta(days=7)).isoformat(),
            },
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_decision_profile(data_dir=db.data_dir)

    # Should only extract commitment signal from future event
    assert result["commitment_samples"] == 1


def test_backfill_decision_profile_respects_limit(db, user_model_store):
    """Backfill should respect the limit parameter."""
    # Create 10 task completion events
    events = []
    for i in range(10):
        events.extend([
            {
                "id": f"evt-created-{i}",
                "type": EventType.TASK_CREATED.value,
                "source": "web_ui",
                "timestamp": f"2026-02-10T{10+i:02d}:00:00Z",
                "priority": "normal",
                "payload": {"task_id": f"task-{i}", "domain": "work"},
                "metadata": {},
            },
            {
                "id": f"evt-completed-{i}",
                "type": EventType.TASK_COMPLETED.value,
                "source": "web_ui",
                "timestamp": f"2026-02-10T{11+i:02d}:00:00Z",
                "priority": "normal",
                "payload": {"task_id": f"task-{i}"},
                "metadata": {},
            },
        ])

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill with limit=5
    result = backfill_decision_profile(data_dir=db.data_dir, limit=5)

    # Should only process 5 events (not all 20)
    assert result["events_processed"] <= 5


def test_backfill_decision_profile_dry_run_mode(db, user_model_store):
    """Dry run should report what would be done without writing to database."""
    events = [
        {
            "id": "evt-task-created",
            "type": EventType.TASK_CREATED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {"task_id": "task-1", "domain": "work"},
            "metadata": {},
        },
        {
            "id": "evt-task-completed",
            "type": EventType.TASK_COMPLETED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T11:00:00Z",
            "priority": "normal",
            "payload": {"task_id": "task-1"},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run in dry-run mode
    result = backfill_decision_profile(data_dir=db.data_dir, dry_run=True)

    # Should report processing but not create profile
    assert result["dry_run"] is True
    assert result["events_processed"] == 2

    # Profile should NOT be created
    profile = user_model_store.get_signal_profile("decision")
    assert profile is None


def test_backfill_decision_profile_ignores_inbound_events(db, user_model_store):
    """Backfill should ignore inbound/passive events that don't represent decisions."""
    events = [
        # Decision-making (should process)
        {
            "id": "evt-sent",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {"body": "What do you think?"},
            "metadata": {},
        },
        # Inbound/passive (should ignore)
        {
            "id": "evt-received",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:05:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        {
            "id": "evt-notification",
            "type": EventType.NOTIFICATION_CREATED.value,
            "source": "lifeos",
            "timestamp": "2026-02-10T14:10:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_decision_profile(data_dir=db.data_dir)

    # Should only process the email.sent event (1 event)
    assert result["events_processed"] == 1
