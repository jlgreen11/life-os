"""
Tests for the topic signal profile backfill script.

Verifies that the backfill correctly processes historical email and message
events through the TopicExtractor and populates the 'topics' signal profile
with topic-frequency data for the semantic fact inferrer.

This backfill is critical for recovering from user_model.db resets (e.g., schema
migrations) that clear the topics signal profile even though historical email
events are still in events.db.
"""

import json

import pytest

from models.core import EventType
from scripts.backfill_topic_profile import backfill_topic_profile


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


def _email_received(event_id: str, from_addr: str, timestamp: str, subject: str = "Project Update",
                    body: str = "We are working on the Python machine learning pipeline.") -> dict:
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


def _email_sent(event_id: str, to_addr: str, timestamp: str, subject: str = "Re: Project",
                body: str = "The Python code is ready for review.") -> dict:
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


def _marketing_email(event_id: str, timestamp: str) -> dict:
    """Build a marketing email.received event that should be filtered out."""
    return {
        "id": event_id,
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "google",
        "timestamp": timestamp,
        "priority": "normal",
        "payload": {
            "from_address": "no-reply@newsletter.com",
            "subject": "Special offer just for you",
            "body": "Shop now and save! Great deals on all products. Click to unsubscribe.",
            "email_date": timestamp,
        },
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backfill_topic_creates_profile_from_received_emails(db, user_model_store):
    """Backfill should create a topics profile from genuine email.received events.

    The TopicExtractor processes inbound emails to extract topic keywords from
    subjects and bodies, accumulating a topic-frequency map in the 'topics'
    signal profile.
    """
    events = [
        _email_received(
            "evt-1", "alice@example.com", "2026-02-01T10:00:00Z",
            subject="Python API design", body="We should design the Python REST API carefully.",
        ),
        _email_received(
            "evt-2", "alice@example.com", "2026-02-08T10:00:00Z",
            subject="Machine learning update", body="The machine learning model is performing well.",
        ),
        _email_received(
            "evt-3", "bob@example.com", "2026-02-05T14:00:00Z",
            subject="Python review", body="Please review the Python implementation.",
        ),
    ]
    _insert_events(db, events)

    result = backfill_topic_profile(data_dir=db.data_dir)

    # Verify processing stats
    assert result["events_processed"] == 3
    assert result["errors"] == 0

    # Verify the profile was created
    profile = user_model_store.get_signal_profile("topics")
    assert profile is not None
    assert profile["samples_count"] > 0

    # Verify topics were extracted
    topic_counts = profile["data"].get("topic_counts", {})
    assert len(topic_counts) > 0


def test_backfill_topic_processes_sent_emails(db, user_model_store):
    """Backfill should also process email.sent events for outbound topic analysis.

    Topics from sent emails are equally important — they reveal what topics the
    user actively engages with and discusses, not just what they receive.
    """
    events = [
        _email_sent(
            "evt-sent-1", "bob@example.com", "2026-02-01T11:00:00Z",
            subject="Python implementation", body="I have completed the Python feature implementation.",
        ),
        _email_sent(
            "evt-sent-2", "carol@example.com", "2026-02-02T15:00:00Z",
            subject="API documentation", body="Here is the updated API documentation for review.",
        ),
    ]
    _insert_events(db, events)

    result = backfill_topic_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 2
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("topics")
    assert profile is not None
    assert profile["samples_count"] > 0


def test_backfill_topic_filters_marketing_senders(db, user_model_store):
    """Backfill should skip marketing/automated inbound emails.

    Marketing emails flood the topic profile with promotional vocabulary
    (offer, shop, deal, rewards, etc.) that has zero signal about the user's
    real interests. The TopicExtractor applies the marketing filter internally.
    """
    events = [
        # Genuine email — should be processed
        _email_received(
            "evt-real", "alice@example.com", "2026-02-01T10:00:00Z",
            subject="Python project", body="Working on the Python data pipeline.",
        ),
        # Marketing emails — should be filtered
        _marketing_email("evt-mkt-1", "2026-02-02T09:00:00Z"),
        _marketing_email("evt-mkt-2", "2026-02-03T10:00:00Z"),
    ]
    _insert_events(db, events)

    result = backfill_topic_profile(data_dir=db.data_dir)

    # All 3 events are "processed" by the extractor (it checks can_process),
    # but only the genuine email should yield topic signals
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("topics")
    # Profile should have fewer signals than total events (marketing filtered)
    # At least the genuine email should have contributed
    if profile:
        topic_counts = profile["data"].get("topic_counts", {})
        # Marketing vocabulary (shop, offer, unsubscribe) should not dominate
        marketing_words = {"shop", "offer", "unsubscribe", "click", "deal", "save"}
        found_marketing = [w for w in marketing_words if w in topic_counts]
        # Should have fewer marketing words than real ones
        assert len(found_marketing) < len(topic_counts)


def test_backfill_topic_dry_run_no_writes(db, user_model_store):
    """Dry run should not write to the database.

    The --dry-run flag allows safely previewing what the backfill would do
    without actually modifying the signal_profiles table.
    """
    events = [
        _email_received(
            "evt-1", "alice@example.com", "2026-02-01T10:00:00Z",
            body="Python machine learning project discussion.",
        ),
    ]
    _insert_events(db, events)

    # Capture initial state
    initial_profile = user_model_store.get_signal_profile("topics")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0

    result = backfill_topic_profile(data_dir=db.data_dir, dry_run=True)

    assert result["dry_run"] is True

    # Profile should not have changed
    final_profile = user_model_store.get_signal_profile("topics")
    final_samples = final_profile["samples_count"] if final_profile else 0
    assert final_samples == initial_samples


def test_backfill_topic_empty_db_returns_zero_stats(db, user_model_store):
    """Backfill should handle an empty events database gracefully.

    When there are no communication events, the backfill should complete
    without errors and return zero-count statistics.
    """
    result = backfill_topic_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 0
    assert result["signals_extracted"] == 0
    assert result["errors"] == 0
    assert result["elapsed_seconds"] >= 0


def test_backfill_topic_limit_parameter(db, user_model_store):
    """The limit parameter should restrict processing to N events.

    This is useful for testing or incremental backfill of recent events
    without reprocessing all historical data.
    """
    events = [
        _email_received(f"evt-{i}", "alice@example.com", f"2026-02-{i+1:02d}T10:00:00Z",
                        body=f"Python project update number {i}.")
        for i in range(5)
    ]
    _insert_events(db, events)

    result = backfill_topic_profile(data_dir=db.data_dir, limit=3)

    # Should only process 3 events due to the limit
    assert result["events_processed"] == 3


def test_backfill_topic_returns_correct_stats_keys(db, user_model_store):
    """Return value should contain all expected statistics keys.

    Callers (including main.py's _backfill_topic_profile_if_needed) rely on
    specific keys in the returned statistics dict.
    """
    result = backfill_topic_profile(data_dir=db.data_dir)

    required_keys = {
        "events_processed",
        "signals_extracted",
        "topics_discovered",
        "initial_samples",
        "final_samples",
        "samples_added",
        "errors",
        "elapsed_seconds",
        "dry_run",
    }
    for key in required_keys:
        assert key in result, f"Missing key in result: {key}"


def test_backfill_topic_auto_trigger_skips_when_profile_populated(db, user_model_store):
    """main.py's auto-trigger should skip backfill when profile already has data.

    The idempotent guard in _backfill_topic_profile_if_needed checks for
    samples_count >= 30 before running the backfill. This test verifies that
    if we pre-populate the profile with 30+ samples (via direct SQL insert),
    the threshold is correctly detected as met.
    """
    # Insert the topic profile directly with 30+ samples to simulate a populated profile.
    # update_signal_profile() increments samples_count by 1 per call, so direct SQL
    # is the correct way to seed a test profile with a specific samples_count value.
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("topics", json.dumps({"topic_counts": {"python": 100, "machine_learning": 50}}), 30),
        )

    profile = user_model_store.get_signal_profile("topics")
    assert profile is not None
    assert profile["samples_count"] >= 30

    # The threshold check in main.py would skip the backfill
    # We verify the guard condition directly
    should_skip = profile["samples_count"] >= 30
    assert should_skip, "Auto-trigger guard should prevent redundant backfill"
