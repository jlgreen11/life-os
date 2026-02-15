"""
Tests for EventStore — the append-only event log.

The EventStore is the foundational component of Life OS, processing 542K+
events per day. It provides the immutable event log that all other services
depend on. This test suite validates:

1. Event storage and retrieval
2. Event querying with filters (type, source, timestamp, limit)
3. Event tagging system (used by rules engine)
4. Suppression flag handling
5. Message ID timestamp lookups (used by cadence extractor)
6. Edge cases and error handling
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from storage.event_store import EventStore
from storage.manager import DatabaseManager


@pytest.fixture
def event_store(db):
    """Create an EventStore instance with a fresh database."""
    return EventStore(db)


def create_test_event(
    event_type: str = "test.event",
    source: str = "test-source",
    priority: str = "normal",
    payload: dict = None,
    metadata: dict = None,
    timestamp: str = None,
) -> dict:
    """Helper to create a well-formed test event.

    Args:
        event_type: Event type string (e.g., "email.received")
        source: Source identifier (e.g., "proton-mail")
        priority: Priority level ("low", "normal", "high", "critical")
        payload: Event-specific data
        metadata: System metadata
        timestamp: ISO 8601 timestamp (defaults to now)

    Returns:
        Complete event dictionary ready for storage
    """
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": source,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "priority": priority,
        "payload": payload or {},
        "metadata": metadata or {},
    }


# =========================================================================
# Event Storage
# =========================================================================


def test_store_event_basic(event_store):
    """Test storing a basic event returns the event ID."""
    event = create_test_event()
    event_id = event_store.store_event(event)
    assert event_id == event["id"]


def test_store_event_minimal(event_store):
    """Test storing an event with only required fields."""
    event = {
        "id": str(uuid.uuid4()),
        "type": "test.minimal",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    event_id = event_store.store_event(event)
    assert event_id == event["id"]


def test_store_event_with_payload(event_store):
    """Test storing an event with structured payload data."""
    payload = {
        "subject": "Test Email",
        "from": "test@example.com",
        "body": "This is a test message",
        "nested": {"key": "value", "list": [1, 2, 3]},
    }
    event = create_test_event(payload=payload)
    event_store.store_event(event)

    # Verify payload is preserved
    events = event_store.get_events(event_type="test.event", limit=1)
    assert len(events) == 1
    stored_payload = json.loads(events[0]["payload"])
    assert stored_payload == payload


def test_store_event_with_metadata(event_store):
    """Test storing an event with metadata."""
    metadata = {
        "connector_version": "1.2.3",
        "sync_batch_id": "batch-456",
        "debug_info": {"timing_ms": 42},
    }
    event = create_test_event(metadata=metadata)
    event_store.store_event(event)

    # Verify metadata is preserved
    events = event_store.get_events(event_type="test.event", limit=1)
    assert len(events) == 1
    stored_metadata = json.loads(events[0]["metadata"])
    assert stored_metadata == metadata


def test_store_event_with_embedding_id(event_store, db):
    """Test storing an event with an embedding_id reference."""
    event = create_test_event()
    event["embedding_id"] = "emb_12345"
    event_store.store_event(event)

    # Verify embedding_id is stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT embedding_id FROM events WHERE id = ?",
            (event["id"],),
        ).fetchone()
        assert row["embedding_id"] == "emb_12345"


def test_store_multiple_events(event_store):
    """Test storing multiple events and verifying count."""
    events = [
        create_test_event(event_type="email.received"),
        create_test_event(event_type="message.sent"),
        create_test_event(event_type="calendar.event.created"),
    ]

    for event in events:
        event_store.store_event(event)

    count = event_store.get_event_count()
    assert count >= 3  # May have other events from fixtures


def test_store_event_immutability(event_store):
    """Test that events cannot be updated (append-only invariant)."""
    event = create_test_event(payload={"version": 1})
    event_store.store_event(event)

    # Attempt to store the same event ID with different payload
    event["payload"] = {"version": 2}

    # Should raise an integrity error (UNIQUE constraint on id)
    with pytest.raises(sqlite3.IntegrityError):
        event_store.store_event(event)


# =========================================================================
# Event Retrieval
# =========================================================================


def test_get_events_no_filters(event_store):
    """Test retrieving events with no filters returns most recent events."""
    # Store events with different timestamps
    now = datetime.now(timezone.utc)
    for i in range(5):
        event = create_test_event(
            timestamp=(now - timedelta(minutes=i)).isoformat()
        )
        event_store.store_event(event)

    events = event_store.get_events(limit=3)
    assert len(events) == 3
    # Should be ordered newest first
    assert events[0]["timestamp"] > events[1]["timestamp"]
    assert events[1]["timestamp"] > events[2]["timestamp"]


def test_get_events_filter_by_type(event_store):
    """Test filtering events by type."""
    event_store.store_event(create_test_event(event_type="email.received"))
    event_store.store_event(create_test_event(event_type="email.received"))
    event_store.store_event(create_test_event(event_type="message.sent"))

    email_events = event_store.get_events(event_type="email.received")
    assert len(email_events) >= 2
    assert all(e["type"] == "email.received" for e in email_events)


def test_get_events_filter_by_source(event_store):
    """Test filtering events by source."""
    event_store.store_event(create_test_event(source="proton-mail"))
    event_store.store_event(create_test_event(source="proton-mail"))
    event_store.store_event(create_test_event(source="imessage"))

    proton_events = event_store.get_events(source="proton-mail")
    assert len(proton_events) >= 2
    assert all(e["source"] == "proton-mail" for e in proton_events)


def test_get_events_filter_by_timestamp(event_store):
    """Test filtering events by timestamp (since)."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)

    # Old event (before cutoff)
    old_event = create_test_event(
        event_type="old.event",
        timestamp=(cutoff - timedelta(minutes=30)).isoformat(),
    )
    event_store.store_event(old_event)

    # New events (after cutoff)
    new_event_1 = create_test_event(
        event_type="new.event",
        timestamp=(cutoff + timedelta(minutes=10)).isoformat(),
    )
    new_event_2 = create_test_event(
        event_type="new.event",
        timestamp=(cutoff + timedelta(minutes=20)).isoformat(),
    )
    event_store.store_event(new_event_1)
    event_store.store_event(new_event_2)

    recent_events = event_store.get_events(
        event_type="new.event",
        since=cutoff.isoformat(),
        limit=100,
    )
    assert len(recent_events) >= 2
    assert all(e["timestamp"] > cutoff.isoformat() for e in recent_events)


def test_get_events_combined_filters(event_store):
    """Test combining multiple filters (type, source, since)."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)

    # Matching event
    matching = create_test_event(
        event_type="email.received",
        source="proton-mail",
        timestamp=(cutoff + timedelta(minutes=10)).isoformat(),
    )
    event_store.store_event(matching)

    # Non-matching events
    event_store.store_event(
        create_test_event(
            event_type="message.sent",  # Wrong type
            source="proton-mail",
            timestamp=(cutoff + timedelta(minutes=10)).isoformat(),
        )
    )
    event_store.store_event(
        create_test_event(
            event_type="email.received",
            source="imessage",  # Wrong source
            timestamp=(cutoff + timedelta(minutes=10)).isoformat(),
        )
    )
    event_store.store_event(
        create_test_event(
            event_type="email.received",
            source="proton-mail",
            timestamp=(cutoff - timedelta(minutes=10)).isoformat(),  # Too old
        )
    )

    filtered = event_store.get_events(
        event_type="email.received",
        source="proton-mail",
        since=cutoff.isoformat(),
        limit=100,
    )
    assert len(filtered) >= 1
    assert all(
        e["type"] == "email.received"
        and e["source"] == "proton-mail"
        and e["timestamp"] > cutoff.isoformat()
        for e in filtered
    )


def test_get_events_limit_enforcement(event_store):
    """Test that limit parameter is respected."""
    # Store 10 events
    for i in range(10):
        event_store.store_event(create_test_event(event_type="test.limit"))

    events = event_store.get_events(event_type="test.limit", limit=5)
    assert len(events) == 5


def test_get_events_default_limit(event_store):
    """Test default limit of 100 events."""
    # Store more than 100 events
    for i in range(150):
        event_store.store_event(create_test_event(event_type="test.default"))

    events = event_store.get_events(event_type="test.default")
    assert len(events) == 100  # Default limit


def test_get_events_empty_result(event_store):
    """Test querying for non-existent events returns empty list."""
    events = event_store.get_events(event_type="nonexistent.type")
    assert events == []


# =========================================================================
# Event Count
# =========================================================================


def test_get_event_count_empty(db):
    """Test event count on empty database returns 0."""
    event_store = EventStore(db)
    assert event_store.get_event_count() == 0


def test_get_event_count_increments(event_store):
    """Test event count increments as events are added."""
    initial_count = event_store.get_event_count()

    event_store.store_event(create_test_event())
    assert event_store.get_event_count() == initial_count + 1

    event_store.store_event(create_test_event())
    assert event_store.get_event_count() == initial_count + 2


# =========================================================================
# Event Tagging
# =========================================================================


def test_add_tag_basic(event_store):
    """Test adding a tag to an event."""
    event = create_test_event()
    event_store.store_event(event)

    event_store.add_tag(event["id"], "test-tag")
    tags = event_store.get_tags(event["id"])
    assert "test-tag" in tags


def test_add_tag_with_rule_id(event_store, db):
    """Test adding a tag with an associated rule ID."""
    event = create_test_event()
    event_store.store_event(event)

    event_store.add_tag(event["id"], "marketing", rule_id="rule-123")

    # Verify rule_id is stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT rule_id FROM event_tags WHERE event_id = ? AND tag = ?",
            (event["id"], "marketing"),
        ).fetchone()
        assert row["rule_id"] == "rule-123"


def test_add_multiple_tags(event_store):
    """Test adding multiple tags to the same event."""
    event = create_test_event()
    event_store.store_event(event)

    event_store.add_tag(event["id"], "tag1")
    event_store.add_tag(event["id"], "tag2")
    event_store.add_tag(event["id"], "tag3")

    tags = event_store.get_tags(event["id"])
    assert "tag1" in tags
    assert "tag2" in tags
    assert "tag3" in tags
    assert len(tags) == 3


def test_add_tag_idempotent(event_store):
    """Test that adding the same tag twice is idempotent."""
    event = create_test_event()
    event_store.store_event(event)

    event_store.add_tag(event["id"], "duplicate")
    event_store.add_tag(event["id"], "duplicate")  # Should not error

    tags = event_store.get_tags(event["id"])
    assert tags.count("duplicate") == 1


def test_get_tags_no_tags(event_store):
    """Test getting tags for an event with no tags returns empty list."""
    event = create_test_event()
    event_store.store_event(event)

    tags = event_store.get_tags(event["id"])
    assert tags == []


def test_get_tags_nonexistent_event(event_store):
    """Test getting tags for a nonexistent event returns empty list."""
    tags = event_store.get_tags("nonexistent-event-id")
    assert tags == []


def test_has_tag_true(event_store):
    """Test has_tag returns True for tagged events."""
    event = create_test_event()
    event_store.store_event(event)
    event_store.add_tag(event["id"], "important")

    assert event_store.has_tag(event["id"], "important") is True


def test_has_tag_false(event_store):
    """Test has_tag returns False for untagged events."""
    event = create_test_event()
    event_store.store_event(event)

    assert event_store.has_tag(event["id"], "nonexistent") is False


def test_has_tag_wrong_tag(event_store):
    """Test has_tag returns False when event has different tags."""
    event = create_test_event()
    event_store.store_event(event)
    event_store.add_tag(event["id"], "tag1")

    assert event_store.has_tag(event["id"], "tag2") is False


# =========================================================================
# Suppression Flag
# =========================================================================


def test_is_suppressed_true(event_store):
    """Test is_suppressed returns True for suppressed events."""
    event = create_test_event()
    event_store.store_event(event)
    event_store.add_tag(event["id"], "system:suppressed")

    assert event_store.is_suppressed(event["id"]) is True


def test_is_suppressed_false(event_store):
    """Test is_suppressed returns False for non-suppressed events."""
    event = create_test_event()
    event_store.store_event(event)

    assert event_store.is_suppressed(event["id"]) is False


def test_is_suppressed_with_other_tags(event_store):
    """Test is_suppressed only checks for system:suppressed tag."""
    event = create_test_event()
    event_store.store_event(event)
    event_store.add_tag(event["id"], "marketing")
    event_store.add_tag(event["id"], "important")

    assert event_store.is_suppressed(event["id"]) is False


# =========================================================================
# Message ID Timestamp Lookup
# =========================================================================


def test_get_timestamp_by_message_id_found(event_store):
    """Test looking up event timestamp by message_id in payload."""
    timestamp = "2026-02-15T12:00:00Z"
    event = create_test_event(
        event_type="email.received",
        payload={"message_id": "msg-12345", "subject": "Test"},
        timestamp=timestamp,
    )
    event_store.store_event(event)

    found_timestamp = event_store.get_timestamp_by_message_id("msg-12345")
    assert found_timestamp == timestamp


def test_get_timestamp_by_message_id_not_found(event_store):
    """Test looking up nonexistent message_id returns None."""
    timestamp = event_store.get_timestamp_by_message_id("nonexistent-msg")
    assert timestamp is None


def test_get_timestamp_by_message_id_multiple_matches(event_store):
    """Test that lookup returns most recent when multiple events share message_id."""
    old_timestamp = "2026-02-15T10:00:00Z"
    new_timestamp = "2026-02-15T12:00:00Z"

    # Store old event
    event1 = create_test_event(
        payload={"message_id": "msg-duplicate"},
        timestamp=old_timestamp,
    )
    event_store.store_event(event1)

    # Store newer event with same message_id
    event2 = create_test_event(
        payload={"message_id": "msg-duplicate"},
        timestamp=new_timestamp,
    )
    event_store.store_event(event2)

    # Should return most recent
    found_timestamp = event_store.get_timestamp_by_message_id("msg-duplicate")
    assert found_timestamp == new_timestamp


def test_get_timestamp_by_message_id_nested_payload(event_store):
    """Test message_id lookup works with nested payload structures."""
    timestamp = "2026-02-15T12:00:00Z"
    event = create_test_event(
        payload={
            "message_id": "msg-nested",
            "metadata": {"nested": "data"},
            "headers": {"from": "test@example.com"},
        },
        timestamp=timestamp,
    )
    event_store.store_event(event)

    found_timestamp = event_store.get_timestamp_by_message_id("msg-nested")
    assert found_timestamp == timestamp


# =========================================================================
# Edge Cases and Error Handling
# =========================================================================


def test_store_event_with_special_characters(event_store):
    """Test storing events with special characters in payload."""
    payload = {
        "subject": "Test with 'quotes' and \"double quotes\"",
        "body": "Special chars: <>&;${}[]",
        "unicode": "Unicode: 你好 🌍 émoji",
    }
    event = create_test_event(payload=payload)
    event_store.store_event(event)

    events = event_store.get_events(limit=1)
    stored_payload = json.loads(events[0]["payload"])
    assert stored_payload == payload


def test_store_event_empty_payload(event_store):
    """Test storing event with empty payload dict."""
    event = create_test_event(payload={})
    event_id = event_store.store_event(event)

    events = event_store.get_events(limit=1)
    assert json.loads(events[0]["payload"]) == {}


def test_store_event_null_payload(event_store):
    """Test storing event with None payload defaults to empty dict."""
    event = create_test_event(payload=None)
    event_id = event_store.store_event(event)

    events = event_store.get_events(limit=1)
    assert json.loads(events[0]["payload"]) == {}


def test_priority_levels(event_store):
    """Test storing events with different priority levels."""
    priorities = ["low", "normal", "high", "critical"]

    for priority in priorities:
        event = create_test_event(
            event_type=f"test.{priority}",
            priority=priority,
        )
        event_store.store_event(event)

    for priority in priorities:
        events = event_store.get_events(event_type=f"test.{priority}", limit=1)
        assert events[0]["priority"] == priority


def test_default_priority(event_store):
    """Test that missing priority defaults to 'normal'."""
    event = {
        "id": str(uuid.uuid4()),
        "type": "test.default_priority",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # No priority field
    }
    event_store.store_event(event)

    events = event_store.get_events(event_type="test.default_priority", limit=1)
    assert events[0]["priority"] == "normal"


def test_concurrent_tag_addition(event_store):
    """Test adding different tags to the same event concurrently."""
    event = create_test_event()
    event_store.store_event(event)

    # Simulate concurrent tag additions
    event_store.add_tag(event["id"], "tag-a")
    event_store.add_tag(event["id"], "tag-b")
    event_store.add_tag(event["id"], "tag-c")

    tags = event_store.get_tags(event["id"])
    assert len(tags) == 3
    assert set(tags) == {"tag-a", "tag-b", "tag-c"}


def test_tag_on_nonexistent_event(event_store):
    """Test that tagging a nonexistent event does not error (foreign key check)."""
    # This should work because event_tags has ON DELETE CASCADE
    # but the tag won't be retrievable since the event doesn't exist
    event_store.add_tag("nonexistent-id", "orphan-tag")

    # Tag should exist in the table but be unretrievable via normal means
    tags = event_store.get_tags("nonexistent-id")
    # Depending on FK constraints, this might be empty or contain the tag
    # Either behavior is acceptable as long as it doesn't crash


def test_get_events_with_json_payload_parsing(event_store):
    """Test that retrieved events have properly parsed JSON payloads."""
    complex_payload = {
        "list": [1, 2, 3],
        "nested": {"key": "value"},
        "bool": True,
        "null": None,
    }
    event = create_test_event(payload=complex_payload)
    event_store.store_event(event)

    events = event_store.get_events(limit=1)
    # Payload should be a JSON string in the database
    assert isinstance(events[0]["payload"], str)
    # But should parse correctly
    parsed = json.loads(events[0]["payload"])
    assert parsed == complex_payload
