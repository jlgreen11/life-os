"""
Tests for scripts/backfill_episode_classification.py

Verifies that the backfill script correctly reclassifies episodes with
granular interaction types, enabling routine and workflow detection.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

# Add scripts directory to path so we can import the backfill script
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from backfill_episode_classification import (
    backfill_episode_classification,
    classify_interaction_type,
)


def test_classify_email_received():
    """Email received events should be classified as email_received."""
    event_type = "email.received"
    payload = {"from_address": "sender@example.com", "subject": "Test"}

    result = classify_interaction_type(event_type, payload)

    assert result == "email_received"


def test_classify_email_sent():
    """Email sent events should be classified as email_sent."""
    event_type = "email.sent"
    payload = {"to_addresses": ["recipient@example.com"], "subject": "Test"}

    result = classify_interaction_type(event_type, payload)

    assert result == "email_sent"


def test_classify_message_received():
    """Message received events should be classified as message_received."""
    event_type = "message.received"
    payload = {"from": "+15555551234", "content": "Hello"}

    result = classify_interaction_type(event_type, payload)

    assert result == "message_received"


def test_classify_message_sent():
    """Message sent events should be classified as message_sent."""
    event_type = "message.sent"
    payload = {"to": "+15555551234", "content": "Hi there"}

    result = classify_interaction_type(event_type, payload)

    assert result == "message_sent"


def test_classify_call_received():
    """Call received events should be classified as call_answered."""
    event_type = "call.received"
    payload = {"from": "+15555551234", "duration": 120}

    result = classify_interaction_type(event_type, payload)

    assert result == "call_answered"


def test_classify_call_missed():
    """Missed call events should be classified as call_missed."""
    event_type = "call.missed"
    payload = {"from": "+15555551234"}

    result = classify_interaction_type(event_type, payload)

    assert result == "call_missed"


def test_classify_meeting_scheduled():
    """Calendar events with participants should be meeting_scheduled."""
    event_type = "calendar.event.created"
    payload = {
        "summary": "Team Sync",
        "participants": ["alice@example.com", "bob@example.com"],
    }

    result = classify_interaction_type(event_type, payload)

    assert result == "meeting_scheduled"


def test_classify_calendar_blocked():
    """Calendar events without participants should be calendar_blocked."""
    event_type = "calendar.event.created"
    payload = {"summary": "Focus time"}

    result = classify_interaction_type(event_type, payload)

    assert result == "calendar_blocked"


def test_classify_calendar_reviewed():
    """Calendar event updates should be calendar_reviewed."""
    event_type = "calendar.event.updated"
    payload = {"summary": "Updated meeting"}

    result = classify_interaction_type(event_type, payload)

    assert result == "calendar_reviewed"


def test_classify_spending():
    """Financial transactions with negative amounts should be spending."""
    event_type = "finance.transaction.new"
    payload = {"amount": -45.67, "merchant": "Coffee Shop"}

    result = classify_interaction_type(event_type, payload)

    assert result == "spending"


def test_classify_income():
    """Financial transactions with positive amounts should be income."""
    event_type = "finance.transaction.new"
    payload = {"amount": 2500.00, "source": "Payroll"}

    result = classify_interaction_type(event_type, payload)

    assert result == "income"


def test_classify_task_created():
    """Task creation events should be task_created."""
    event_type = "task.created"
    payload = {"title": "Review PR", "priority": "high"}

    result = classify_interaction_type(event_type, payload)

    assert result == "task_created"


def test_classify_task_completed():
    """Task completion events should be task_completed."""
    event_type = "task.completed"
    payload = {"title": "Review PR", "completed_at": datetime.now(timezone.utc).isoformat()}

    result = classify_interaction_type(event_type, payload)

    assert result == "task_completed"


def test_classify_location_arrived():
    """Location arrival events should be location_arrived."""
    event_type = "location.arrived"
    payload = {"place": "Home", "latitude": 37.7749, "longitude": -122.4194}

    result = classify_interaction_type(event_type, payload)

    assert result == "location_arrived"


def test_classify_location_departed():
    """Location departure events should be location_departed."""
    event_type = "location.departed"
    payload = {"place": "Office"}

    result = classify_interaction_type(event_type, payload)

    assert result == "location_departed"


def test_classify_location_changed():
    """General location changes should be location_changed."""
    event_type = "location.changed"
    payload = {"from": "Home", "to": "Office"}

    result = classify_interaction_type(event_type, payload)

    assert result == "location_changed"


def test_classify_context_location():
    """Context location events should be context_location."""
    event_type = "context.location"
    payload = {"location": "Office"}

    result = classify_interaction_type(event_type, payload)

    assert result == "context_location"


def test_classify_context_activity():
    """Context activity events should be context_activity."""
    event_type = "context.activity"
    payload = {"activity": "walking"}

    result = classify_interaction_type(event_type, payload)

    assert result == "context_activity"


def test_classify_user_command():
    """User command events should be user_command."""
    event_type = "system.user.command"
    payload = {"command": "show tasks"}

    result = classify_interaction_type(event_type, payload)

    assert result == "user_command"


def test_classify_unknown_event_type_with_dot():
    """Unknown event types with dots should extract the last segment."""
    event_type = "custom.workflow.triggered"
    payload = {}

    result = classify_interaction_type(event_type, payload)

    assert result == "triggered"


def test_classify_unknown_event_type_without_dot():
    """Unknown event types without dots should return 'other'."""
    event_type = "unknowntype"
    payload = {}

    result = classify_interaction_type(event_type, payload)

    assert result == "other"


def test_backfill_empty_database(db):
    """Backfill on empty database should complete without errors."""
    stats = backfill_episode_classification(db, dry_run=False)

    assert stats["total_episodes"] == 0
    assert stats["reclassified"] == 0
    assert stats["unchanged"] == 0
    assert stats["errors"] == 0


def test_backfill_dry_run_does_not_modify_database(db, event_store, user_model_store):
    """Dry run mode should not modify any episodes."""
    # Create an event and episode with old classification
    event_id = str(uuid4())
    event = {
        "id": event_id,
        "type": "email.received",
        "source": "protonmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {"from_address": "sender@example.com", "subject": "Test"},
        "metadata": {},
    }
    event_store.store_event(event)

    episode = {
        "id": str(uuid4()),
        "timestamp": event["timestamp"],
        "event_id": event_id,
        "interaction_type": "communication",  # Old classification
        "content_summary": "Email received",
        "content_full": json.dumps(event["payload"]),
    }
    user_model_store.store_episode(episode)

    # Run backfill in dry-run mode
    stats = backfill_episode_classification(db, dry_run=True)

    # Verify stats
    assert stats["total_episodes"] == 1
    assert stats["reclassified"] == 1  # Would be reclassified
    assert stats["unchanged"] == 0
    assert stats["errors"] == 0
    assert stats["type_distribution"]["email_received"] == 1

    # Verify database was NOT modified
    with db.get_connection("user_model") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT interaction_type FROM episodes WHERE id = ?", (episode["id"],))
        row = cursor.fetchone()
        assert row[0] == "communication"  # Still old value


def test_backfill_reclassifies_episodes_correctly(db, event_store, user_model_store):
    """Backfill should correctly reclassify episodes based on their source events."""
    # Create multiple events and episodes with old classification
    events_and_episodes = [
        ("email.received", {"from_address": "sender@example.com"}, "email_received"),
        ("email.sent", {"to_addresses": ["recipient@example.com"]}, "email_sent"),
        ("message.received", {"from": "+15555551234"}, "message_received"),
        ("calendar.event.created", {"participants": ["alice@example.com"]}, "meeting_scheduled"),
        ("task.created", {"title": "Review PR"}, "task_created"),
    ]

    episode_ids = []
    for event_type, payload, expected_type in events_and_episodes:
        event_id = str(uuid4())
        event = {
            "id": event_id,
            "type": event_type,
            "source": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": payload,
            "metadata": {},
        }
        event_store.store_event(event)

        episode_id = str(uuid4())
        episode = {
            "id": episode_id,
            "timestamp": event["timestamp"],
            "event_id": event_id,
            "interaction_type": "communication",  # Old classification
            "content_summary": f"{event_type} event",
            "content_full": json.dumps(payload),
        }
        user_model_store.store_episode(episode)
        episode_ids.append((episode_id, expected_type))

    # Run backfill
    stats = backfill_episode_classification(db, dry_run=False)

    # Verify stats
    assert stats["total_episodes"] == 5
    assert stats["reclassified"] == 5
    assert stats["unchanged"] == 0
    assert stats["errors"] == 0

    # Verify each episode was reclassified correctly
    with db.get_connection("user_model") as conn:
        cursor = conn.cursor()
        for episode_id, expected_type in episode_ids:
            cursor.execute("SELECT interaction_type FROM episodes WHERE id = ?", (episode_id,))
            row = cursor.fetchone()
            assert row[0] == expected_type


def test_backfill_handles_already_classified_episodes(db, event_store, user_model_store):
    """Episodes that already have correct classification should be counted as unchanged."""
    # Create an event and episode with correct classification
    event_id = str(uuid4())
    event = {
        "id": event_id,
        "type": "email.received",
        "source": "protonmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {"from_address": "sender@example.com", "subject": "Test"},
        "metadata": {},
    }
    event_store.store_event(event)

    episode = {
        "id": str(uuid4()),
        "timestamp": event["timestamp"],
        "event_id": event_id,
        "interaction_type": "email_received",  # Already correct
        "content_summary": "Email received",
        "content_full": json.dumps(event["payload"]),
    }
    user_model_store.store_episode(episode)

    # Run backfill
    stats = backfill_episode_classification(db, dry_run=False)

    # Verify stats
    assert stats["total_episodes"] == 1
    assert stats["reclassified"] == 0
    assert stats["unchanged"] == 1
    assert stats["errors"] == 0


def test_backfill_handles_missing_event(db, user_model_store):
    """Episodes with missing source events should be counted as errors."""
    # Create an episode without a corresponding event
    episode = {
        "id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": "nonexistent-event-id",
        "interaction_type": "communication",
        "content_summary": "Orphaned episode",
        "content_full": "{}",
    }
    user_model_store.store_episode(episode)

    # Run backfill
    stats = backfill_episode_classification(db, dry_run=False)

    # Verify stats
    assert stats["total_episodes"] == 1
    assert stats["reclassified"] == 0
    assert stats["unchanged"] == 0
    assert stats["errors"] == 1


def test_backfill_handles_invalid_payload_json(db, event_store, user_model_store):
    """Episodes with null payloads should still be classified correctly."""
    # Create an event with null payload (edge case)
    event_id = str(uuid4())
    event = {
        "id": event_id,
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": None,  # Null payload
        "metadata": {},
    }
    event_store.store_event(event)

    episode = {
        "id": str(uuid4()),
        "timestamp": event["timestamp"],
        "event_id": event_id,
        "interaction_type": "communication",
        "content_summary": "Episode with null payload",
        "content_full": "{}",
    }
    user_model_store.store_episode(episode)

    # Run backfill
    stats = backfill_episode_classification(db, dry_run=False)

    # Verify stats - null payloads are treated as empty dicts, so classification works
    assert stats["total_episodes"] == 1
    assert stats["reclassified"] == 1
    assert stats["unchanged"] == 0
    assert stats["errors"] == 0
    assert stats["type_distribution"]["email_received"] == 1


def test_backfill_type_distribution(db, event_store, user_model_store):
    """Type distribution should accurately count each interaction type."""
    # Create multiple episodes of different types
    event_types = [
        "email.received",
        "email.received",
        "email.sent",
        "message.received",
        "task.created",
    ]

    for event_type in event_types:
        event_id = str(uuid4())
        event = {
            "id": event_id,
            "type": event_type,
            "source": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {},
            "metadata": {},
        }
        event_store.store_event(event)

        episode = {
            "id": str(uuid4()),
            "timestamp": event["timestamp"],
            "event_id": event_id,
            "interaction_type": "communication",
            "content_summary": f"{event_type} event",
            "content_full": "{}",
        }
        user_model_store.store_episode(episode)

    # Run backfill
    stats = backfill_episode_classification(db, dry_run=False)

    # Verify type distribution
    assert stats["type_distribution"]["email_received"] == 2
    assert stats["type_distribution"]["email_sent"] == 1
    assert stats["type_distribution"]["message_received"] == 1
    assert stats["type_distribution"]["task_created"] == 1
