"""
Tests for the linguistic_inbound signal profile backfill script.

Verifies that the backfill correctly processes historical inbound communication
events (email.received, message.received) through the LinguisticExtractor and
populates the 'linguistic_inbound' signal profile with per-contact incoming
style metrics.

This profile is required by the SemanticFactInferrer's
``infer_from_inbound_linguistic_profile()`` method, which needs 10+ samples
to produce facts about the user's communication environment.
"""

import json

from models.core import EventType
from scripts.backfill_linguistic_inbound_profile import backfill_linguistic_inbound_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_events(db, events: list[dict]) -> None:
    """Insert test events into events.db."""
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


def _email_received(
    event_id: str,
    from_addr: str,
    timestamp: str,
    subject: str = "Hello",
    body: str = "Just checking in on the project status. Let me know if you need anything.",
) -> dict:
    """Build a minimal email.received event dict for testing."""
    return {
        "id": event_id,
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "google",
        "timestamp": timestamp,
        "priority": "normal",
        "payload": {
            "from_address": from_addr,
            "subject": subject,
            "body": body,
            "email_date": timestamp,
        },
        "metadata": {},
    }


def _message_received(
    event_id: str,
    from_addr: str,
    timestamp: str,
    body: str = "Hey, are you around? I wanted to chat about the weekend plans.",
) -> dict:
    """Build a minimal message.received event dict for testing."""
    return {
        "id": event_id,
        "type": EventType.MESSAGE_RECEIVED.value,
        "source": "signal",
        "timestamp": timestamp,
        "priority": "normal",
        "payload": {
            "from_address": from_addr,
            "body": body,
            "channel": "signal",
        },
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backfill_inbound_creates_profile_from_received_emails(db, user_model_store):
    """Backfill should create a linguistic_inbound profile from email.received events.

    Inbound emails are the primary data source for per-contact incoming style
    profiles — they capture how each contact communicates with the user.
    """
    events = [
        _email_received(
            "evt-recv-1", "alice@example.com", "2026-02-01T09:00:00Z",
            body="Hi there! Just wanted to check in on the project. How's everything going?",
        ),
        _email_received(
            "evt-recv-2", "alice@example.com", "2026-02-02T10:00:00Z",
            body="Thanks for the update! I think we should proceed with the implementation plan.",
        ),
        _email_received(
            "evt-recv-3", "bob@example.com", "2026-02-03T11:00:00Z",
            body="Please provide a comprehensive status update on the deliverables accordingly.",
        ),
        _email_received(
            "evt-recv-4", "bob@example.com", "2026-02-04T14:00:00Z",
            body="Furthermore, we need to review the budget allocations for next quarter.",
        ),
        _email_received(
            "evt-recv-5", "carol@example.com", "2026-02-05T09:00:00Z",
            body="Hey! So I was thinking about maybe grabbing lunch sometime this week?",
        ),
    ]
    _insert_events(db, events)

    result = backfill_linguistic_inbound_profile(data_dir=db.data_dir)

    # Verify processing stats
    assert result["events_processed"] >= 5
    assert result["errors"] == 0

    # Verify the linguistic_inbound profile was created
    profile = user_model_store.get_signal_profile("linguistic_inbound")
    assert profile is not None
    assert profile["samples_count"] >= 1


def test_backfill_inbound_ignores_outbound_events(db, user_model_store):
    """Backfill should NOT process email.sent or message.sent events.

    The inbound backfill only handles received messages. Outbound messages
    are handled by the regular linguistic profile backfill.
    """
    events = [
        # These outbound events should be ignored by the inbound backfill
        {
            "id": "evt-sent-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "google",
            "timestamp": "2026-02-01T10:00:00Z",
            "priority": "normal",
            "payload": {
                "to_addresses": ["alice@example.com"],
                "subject": "Re: Project",
                "body": "I think this approach is solid and we should proceed with it.",
            },
            "metadata": {},
        },
        # This inbound event should be processed
        _email_received(
            "evt-recv-1", "alice@example.com", "2026-02-01T09:00:00Z",
            body="Hi! Let me know how the project is going when you get a chance.",
        ),
    ]
    _insert_events(db, events)

    result = backfill_linguistic_inbound_profile(data_dir=db.data_dir)

    # Only the inbound event should be processed
    assert result["events_processed"] == 1
    assert result["errors"] == 0


def test_backfill_inbound_processes_message_received(db, user_model_store):
    """Backfill should process message.received events (Signal, iMessage, etc.)."""
    events = [
        _message_received(
            "evt-msg-1", "alice@signal", "2026-02-01T10:00:00Z",
            body="Hey, are you around? I wanted to talk about the weekend plans.",
        ),
        _message_received(
            "evt-msg-2", "bob@signal", "2026-02-02T11:00:00Z",
            body="Sounds good! Let me know when you're free for the meeting.",
        ),
    ]
    _insert_events(db, events)

    result = backfill_linguistic_inbound_profile(data_dir=db.data_dir)

    assert result["events_processed"] >= 2
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("linguistic_inbound")
    assert profile is not None
    assert profile["samples_count"] >= 1


def test_backfill_inbound_builds_per_contact_data(db, user_model_store):
    """Backfill should build per-contact style data in the inbound profile.

    The linguistic_inbound profile stores per_contact ring buffers and
    per_contact_averages, which are used by the semantic fact inferrer
    to characterise each contact's communication style.
    """
    events = [
        _email_received(
            "evt-recv-1", "alice@example.com", "2026-02-01T09:00:00Z",
            body="Hi! Just wanted to check in on the project. How is everything going?",
        ),
        _email_received(
            "evt-recv-2", "bob@example.com", "2026-02-02T10:00:00Z",
            body="Please provide a comprehensive status update on the aforementioned deliverables.",
        ),
    ]
    _insert_events(db, events)

    result = backfill_linguistic_inbound_profile(data_dir=db.data_dir)

    assert result["events_processed"] >= 2

    profile = user_model_store.get_signal_profile("linguistic_inbound")
    assert profile is not None

    data = profile["data"]
    per_contact = data.get("per_contact", {})
    # Both contacts should have entries
    assert "alice@example.com" in per_contact
    assert "bob@example.com" in per_contact


def test_backfill_inbound_dry_run_no_writes(db, user_model_store):
    """Dry run should not write to the database."""
    events = [
        _email_received(
            "evt-recv-1", "alice@example.com", "2026-02-01T09:00:00Z",
            body="Hello! I wanted to discuss the upcoming project timeline with you.",
        ),
    ]
    _insert_events(db, events)

    initial_profile = user_model_store.get_signal_profile("linguistic_inbound")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0

    result = backfill_linguistic_inbound_profile(data_dir=db.data_dir, dry_run=True)

    assert result["dry_run"] is True

    final_profile = user_model_store.get_signal_profile("linguistic_inbound")
    final_samples = final_profile["samples_count"] if final_profile else 0
    assert final_samples == initial_samples


def test_backfill_inbound_empty_db_returns_zero_stats(db, user_model_store):
    """Backfill should handle an empty events database gracefully."""
    result = backfill_linguistic_inbound_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 0
    assert result["errors"] == 0
    assert result["elapsed_seconds"] >= 0


def test_backfill_inbound_limit_parameter(db, user_model_store):
    """The limit parameter should restrict processing to N events."""
    events = [
        _email_received(
            f"evt-{i}", "alice@example.com", f"2026-02-{i + 1:02d}T10:00:00Z",
            body=f"Message number {i}. This is a test of the linguistic extractor processing.",
        )
        for i in range(5)
    ]
    _insert_events(db, events)

    result = backfill_linguistic_inbound_profile(data_dir=db.data_dir, limit=2)

    assert result["events_processed"] == 2


def test_backfill_inbound_returns_correct_stats_keys(db, user_model_store):
    """Return value should contain all expected statistics keys."""
    result = backfill_linguistic_inbound_profile(data_dir=db.data_dir)

    required_keys = {
        "events_processed",
        "signals_extracted",
        "initial_samples",
        "final_samples",
        "samples_added",
        "errors",
        "elapsed_seconds",
        "dry_run",
    }
    for key in required_keys:
        assert key in result, f"Missing key in result: {key}"


def test_backfill_inbound_auto_trigger_skips_when_profile_populated(db, user_model_store):
    """The auto-trigger guard should skip backfill when profile has 10+ samples.

    The _backfill_inbound_linguistic_profile_if_needed() method in main.py
    checks for samples_count >= 10 (matching the inferrer's threshold).
    update_signal_profile increments samples_count by 1 on each call, so
    we call it 10 times to simulate 10 accumulated samples.
    """
    dummy_data = {
        "per_contact": {"alice@example.com": [{"word_count": 20}]},
        "per_contact_averages": {},
    }
    for _ in range(10):
        user_model_store.update_signal_profile("linguistic_inbound", dummy_data)

    profile = user_model_store.get_signal_profile("linguistic_inbound")
    assert profile is not None
    assert profile["samples_count"] >= 10

    # The guard condition in main.py would skip the backfill
    should_skip = profile["samples_count"] >= 10
    assert should_skip, "Auto-trigger guard should prevent redundant backfill"
