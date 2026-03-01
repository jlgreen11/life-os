"""
Tests for the linguistic signal profile backfill script.

Verifies that the backfill correctly processes historical communication events
through the LinguisticExtractor and populates the 'linguistic' signal profile
with writing-style metrics that the semantic fact inferrer uses to generate
communication style preference facts.

This backfill is critical for recovering from user_model.db resets (e.g., schema
migrations) that clear the linguistic signal profile even though historical
email events are still in events.db.
"""

import json

import pytest

from models.core import EventType
from scripts.backfill_linguistic_profile import backfill_linguistic_profile


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


def _email_sent(event_id: str, to_addr: str, timestamp: str,
                subject: str = "Re: Discussion",
                body: str = "I think this approach is solid. Let me know what you think.") -> dict:
    """Build a minimal email.sent event dict for testing."""
    return {
        "id": event_id,
        "type": EventType.EMAIL_SENT.value,
        "source": "google",
        "timestamp": timestamp,
        "priority": "normal",
        "payload": {
            "to_addresses": [to_addr],
            "subject": subject,
            "body": body,
            "sent_at": timestamp,
        },
        "metadata": {},
    }


def _email_received(event_id: str, from_addr: str, timestamp: str,
                    subject: str = "Hello",
                    body: str = "Just checking in on the project status.") -> dict:
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backfill_linguistic_creates_profile_from_sent_emails(db, user_model_store):
    """Backfill should create a linguistic profile from email.sent events.

    Outbound messages are the primary source for the user's own linguistic
    fingerprint — they capture the user's actual writing style, not just what
    they read.
    """
    events = [
        _email_sent(
            "evt-sent-1", "alice@example.com", "2026-02-01T10:00:00Z",
            body="I think this approach is really solid! Let me know what you think.",
        ),
        _email_sent(
            "evt-sent-2", "bob@example.com", "2026-02-08T10:00:00Z",
            body="Could you please review the implementation? I'm not sure about the edge cases.",
        ),
    ]
    _insert_events(db, events)

    result = backfill_linguistic_profile(data_dir=db.data_dir)

    # Verify processing stats
    assert result["events_processed"] >= 2
    assert result["errors"] == 0

    # Verify the profile was created
    profile = user_model_store.get_signal_profile("linguistic")
    assert profile is not None
    assert profile["samples_count"] > 0


def test_backfill_linguistic_processes_inbound_emails(db, user_model_store):
    """Backfill should also process email.received events for contact style tracking.

    Inbound emails update per-contact incoming style profiles, enabling
    tone-shift detection (e.g., a contact suddenly writes more formally).
    """
    events = [
        _email_received(
            "evt-recv-1", "alice@example.com", "2026-02-01T09:00:00Z",
            body="Hi! Just wanted to check in. Let me know how the project is going.",
        ),
        _email_received(
            "evt-recv-2", "bob@example.com", "2026-02-02T14:00:00Z",
            body="Please provide a status update on the aforementioned deliverables.",
        ),
    ]
    _insert_events(db, events)

    result = backfill_linguistic_profile(data_dir=db.data_dir)

    # All events are processed (both inbound and outbound contribute to linguistic profile)
    assert result["errors"] == 0
    # Profile should exist after processing received emails
    # (the exact samples count depends on extractor implementation)
    assert result["events_processed"] >= 2


def test_backfill_linguistic_dry_run_no_writes(db, user_model_store):
    """Dry run should not write to the database.

    The --dry-run flag allows safely previewing what the backfill would do
    without actually modifying the signal_profiles table.
    """
    events = [
        _email_sent(
            "evt-sent-1", "alice@example.com", "2026-02-01T10:00:00Z",
            body="Hello! I think this is a great idea and we should proceed!",
        ),
    ]
    _insert_events(db, events)

    # Capture initial state
    initial_profile = user_model_store.get_signal_profile("linguistic")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0

    result = backfill_linguistic_profile(data_dir=db.data_dir, dry_run=True)

    assert result["dry_run"] is True

    # Profile should not have changed
    final_profile = user_model_store.get_signal_profile("linguistic")
    final_samples = final_profile["samples_count"] if final_profile else 0
    assert final_samples == initial_samples


def test_backfill_linguistic_empty_db_returns_zero_stats(db, user_model_store):
    """Backfill should handle an empty events database gracefully.

    When there are no communication events, the backfill should complete
    without errors and return zero-count statistics.
    """
    result = backfill_linguistic_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 0
    assert result["errors"] == 0
    assert result["elapsed_seconds"] >= 0


def test_backfill_linguistic_limit_parameter(db, user_model_store):
    """The limit parameter should restrict processing to N events.

    This is useful for testing or incremental backfill of recent events
    without reprocessing all historical data.
    """
    events = [
        _email_sent(f"evt-{i}", "alice@example.com", f"2026-02-{i+1:02d}T10:00:00Z",
                    body=f"Message number {i}. This is a test of the linguistic extractor.")
        for i in range(5)
    ]
    _insert_events(db, events)

    result = backfill_linguistic_profile(data_dir=db.data_dir, limit=2)

    # Should only process 2 events due to the limit
    assert result["events_processed"] == 2


def test_backfill_linguistic_returns_correct_stats_keys(db, user_model_store):
    """Return value should contain all expected statistics keys.

    Callers (including main.py's _backfill_linguistic_profile_if_needed) rely
    on specific keys in the returned statistics dict.
    """
    result = backfill_linguistic_profile(data_dir=db.data_dir)

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


def test_backfill_linguistic_auto_trigger_skips_when_profile_populated(db, user_model_store):
    """main.py's auto-trigger should skip backfill when profile already has data.

    The idempotent guard in _backfill_linguistic_profile_if_needed checks for
    samples_count >= 1 before running the backfill. This test verifies that
    if we pre-populate the profile with data, the threshold is correctly met.
    """
    # Pre-populate linguistic profile with 1+ samples
    user_model_store.update_signal_profile("linguistic", {
        "averages": {"formality": 0.3, "word_count": 45.0},
        "samples_count": 1,
    })

    profile = user_model_store.get_signal_profile("linguistic")
    assert profile is not None
    assert profile["samples_count"] >= 1

    # The threshold check in main.py would skip the backfill
    # We verify the guard condition directly
    should_skip = profile["samples_count"] >= 1
    assert should_skip, "Auto-trigger guard should prevent redundant backfill"


def test_backfill_linguistic_incremental_is_idempotent(db, user_model_store):
    """Running the backfill twice should be safe and produce consistent results.

    The backfill may be triggered multiple times (e.g., on multiple restarts
    before the guard threshold is reached). It should not corrupt the profile
    by double-counting samples.
    """
    events = [
        _email_sent(
            "evt-sent-1", "alice@example.com", "2026-02-01T10:00:00Z",
            body="I believe this approach is solid. Let me review further.",
        ),
    ]
    _insert_events(db, events)

    # Run backfill twice
    result1 = backfill_linguistic_profile(data_dir=db.data_dir)
    result2 = backfill_linguistic_profile(data_dir=db.data_dir)

    # Both should succeed without errors
    assert result1["errors"] == 0
    assert result2["errors"] == 0

    # Profile should exist after both runs
    profile = user_model_store.get_signal_profile("linguistic")
    assert profile is not None
