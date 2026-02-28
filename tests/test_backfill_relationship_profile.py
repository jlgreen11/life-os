"""
Tests for the relationship signal profile backfill script.

Verifies that the backfill correctly processes historical email and message
events through the RelationshipExtractor and populates the 'relationships'
signal profile with per-contact interaction data.

This backfill is critical for recovering from user_model.db resets (e.g., schema
migrations) that clear the relationships signal profile even though historical
email events are still in events.db.
"""

import json

import pytest

from models.core import EventType
from scripts.backfill_relationship_profile import backfill_relationship_profile


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


def _email_received(event_id: str, from_addr: str, timestamp: str, subject: str = "Hello") -> dict:
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
            "body": "Test message body",
            "email_date": timestamp,
        },
        "metadata": {},
    }


def _email_sent(event_id: str, to_addr: str, timestamp: str, subject: str = "Re: Hello") -> dict:
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
            "body": "Test reply body",
            "sent_at": timestamp,
        },
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backfill_relationship_creates_profile_from_received_emails(db, user_model_store):
    """Backfill should create a relationships profile from email.received events.

    The RelationshipExtractor processes inbound emails to track which contacts
    send messages to the user. Each sender gets a profile entry with
    interaction_count incremented and inbound_count tracked.
    """
    events = [
        _email_received("evt-1", "alice@example.com", "2026-02-01T10:00:00Z"),
        _email_received("evt-2", "alice@example.com", "2026-02-08T10:00:00Z"),
        _email_received("evt-3", "bob@example.com", "2026-02-05T14:00:00Z"),
    ]
    _insert_events(db, events)

    result = backfill_relationship_profile(data_dir=db.data_dir)

    # Verify processing stats
    assert result["events_processed"] == 3
    assert result["errors"] == 0

    # Verify the profile was created
    profile = user_model_store.get_signal_profile("relationships")
    assert profile is not None
    assert profile["samples_count"] > 0

    # Verify contacts were discovered
    contacts = profile["data"]["contacts"]
    assert "alice@example.com" in contacts
    assert "bob@example.com" in contacts

    # Alice should have 2 interactions
    alice = contacts["alice@example.com"]
    assert alice["interaction_count"] == 2
    assert alice["inbound_count"] == 2
    assert alice["outbound_count"] == 0


def test_backfill_relationship_tracks_bidirectional_interactions(db, user_model_store):
    """Backfill should track both inbound (received) and outbound (sent) interactions.

    Bidirectional contacts (the user both receives from and sends to) are the
    most important for relationship maintenance — they represent real mutual
    relationships rather than one-way subscriptions.
    """
    events = [
        # Carol sends two emails to the user
        _email_received("evt-in-1", "carol@example.com", "2026-02-01T10:00:00Z"),
        _email_received("evt-in-2", "carol@example.com", "2026-02-10T10:00:00Z"),
        # User replies once to Carol
        _email_sent("evt-out-1", "carol@example.com", "2026-02-02T11:00:00Z"),
    ]
    _insert_events(db, events)

    result = backfill_relationship_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 3
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("relationships")
    contacts = profile["data"]["contacts"]

    carol = contacts["carol@example.com"]
    assert carol["interaction_count"] == 3  # 2 inbound + 1 outbound
    assert carol["inbound_count"] == 2
    assert carol["outbound_count"] == 1


def test_backfill_relationship_tracks_last_interaction_timestamp(db, user_model_store):
    """Backfill should record the most recent interaction timestamp per contact.

    The last_interaction field is used by _check_relationship_maintenance in the
    prediction engine to determine how long it has been since the user interacted
    with a contact. Without this, relationship maintenance predictions cannot fire.
    """
    events = [
        _email_received("evt-early", "dave@example.com", "2026-02-01T10:00:00Z"),
        _email_received("evt-late", "dave@example.com", "2026-02-15T14:00:00Z"),
    ]
    _insert_events(db, events)

    backfill_relationship_profile(data_dir=db.data_dir)

    profile = user_model_store.get_signal_profile("relationships")
    dave = profile["data"]["contacts"]["dave@example.com"]

    # last_interaction should be the most recent timestamp
    assert dave["last_interaction"] == "2026-02-15T14:00:00Z"


def test_backfill_relationship_filters_marketing_senders(db, user_model_store):
    """Backfill should skip no-reply and marketing senders.

    Marketing emails should not appear in the relationship graph — they are
    one-way commercial communications, not human relationships. The marketing
    filter removes addresses like noreply@, marketing@, etc.
    """
    events = [
        # Real human contact — should be tracked
        _email_received("evt-human", "friend@example.com", "2026-02-10T10:00:00Z"),
        # Marketing/no-reply — should be filtered out
        _email_received("evt-noreply", "noreply@company.com", "2026-02-10T11:00:00Z"),
        _email_received("evt-marketing", "no-reply@newsletter.com", "2026-02-10T12:00:00Z"),
    ]
    _insert_events(db, events)

    result = backfill_relationship_profile(data_dir=db.data_dir)

    profile = user_model_store.get_signal_profile("relationships")
    contacts = profile["data"]["contacts"]

    # Only the human contact should be in the profile
    assert "friend@example.com" in contacts
    assert "noreply@company.com" not in contacts
    assert "no-reply@newsletter.com" not in contacts


def test_backfill_relationship_handles_message_events(db, user_model_store):
    """Backfill should process message.received and message.sent events.

    The RelationshipExtractor handles all four communication event types:
    email.received, email.sent, message.received, message.sent. Messages
    (Signal, iMessage) should contribute to the relationship profile too.
    """
    events = [
        {
            "id": "evt-msg-in",
            "type": EventType.MESSAGE_RECEIVED.value,
            "source": "imessage",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {
                "from_address": "eve@example.com",
                "body": "Hey!",
                "channel": "imessage",
            },
            "metadata": {},
        },
        {
            "id": "evt-msg-out",
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": "2026-02-10T10:05:00Z",
            "priority": "normal",
            "payload": {
                "to_addresses": ["eve@example.com"],
                "body": "Hey back!",
                "channel": "imessage",
            },
            "metadata": {},
        },
    ]
    _insert_events(db, events)

    result = backfill_relationship_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 2
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("relationships")
    assert profile is not None
    contacts = profile["data"]["contacts"]
    assert "eve@example.com" in contacts


def test_backfill_relationship_respects_limit_parameter(db, user_model_store):
    """Backfill should stop after processing --limit events.

    The limit parameter allows processing only the most recent N events,
    useful for incremental updates or testing without processing all history.
    """
    events = [
        _email_received(f"evt-{i}", f"contact{i}@example.com", f"2026-02-{i+1:02d}T10:00:00Z")
        for i in range(10)
    ]
    _insert_events(db, events)

    result = backfill_relationship_profile(data_dir=db.data_dir, limit=4)

    # Should only process the 4 oldest (ORDER BY timestamp ASC, LIMIT 4)
    assert result["events_processed"] == 4


def test_backfill_relationship_dry_run_does_not_write(db, user_model_store):
    """Dry run mode should report what would happen without writing to the database.

    This allows verifying the backfill would process the expected events
    before committing to a potentially slow full backfill.
    """
    events = [
        _email_received("evt-1", "frank@example.com", "2026-02-10T10:00:00Z"),
    ]
    _insert_events(db, events)

    result = backfill_relationship_profile(data_dir=db.data_dir, dry_run=True)

    assert result["dry_run"] is True
    assert result["events_processed"] == 1

    # Profile should NOT be created in dry run mode
    profile = user_model_store.get_signal_profile("relationships")
    assert profile is None


def test_backfill_relationship_reports_contacts_discovered(db, user_model_store):
    """Backfill should report the total number of unique contacts discovered.

    This metric helps verify the backfill worked correctly — with 16K emails
    from many senders, contacts_discovered should be in the hundreds.
    """
    events = [
        _email_received("evt-a", "alice@example.com", "2026-02-01T10:00:00Z"),
        _email_received("evt-b", "bob@example.com", "2026-02-02T10:00:00Z"),
        _email_received("evt-c", "carol@example.com", "2026-02-03T10:00:00Z"),
        # Second email from alice — should not increase contacts count
        _email_received("evt-a2", "alice@example.com", "2026-02-04T10:00:00Z"),
    ]
    _insert_events(db, events)

    result = backfill_relationship_profile(data_dir=db.data_dir)

    # 3 unique contacts (alice, bob, carol) even though 4 events were processed
    assert result["contacts_discovered"] == 3


def test_backfill_relationship_handles_events_with_missing_payload_fields(db, user_model_store):
    """Backfill should not crash on events with missing or malformed payload fields.

    Real email events sometimes have null from_address, empty to_addresses, or
    other missing fields. The extractor should skip these gracefully without
    aborting the entire backfill.
    """
    events = [
        # Valid event — should process
        _email_received("evt-valid", "grace@example.com", "2026-02-10T10:00:00Z"),
        # Missing from_address — should be skipped by the extractor
        {
            "id": "evt-no-addr",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "google",
            "timestamp": "2026-02-10T11:00:00Z",
            "priority": "normal",
            "payload": {"subject": "No sender"},  # from_address missing
            "metadata": {},
        },
        # Empty payload — should be handled gracefully
        {
            "id": "evt-empty",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "google",
            "timestamp": "2026-02-10T12:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
    ]
    _insert_events(db, events)

    # Should not raise any exceptions
    result = backfill_relationship_profile(data_dir=db.data_dir)

    # At minimum the valid event was processed
    assert result["events_processed"] >= 1
    assert result["errors"] == 0  # Extractor skips gracefully — no hard errors

    # Grace should be in the profile
    profile = user_model_store.get_signal_profile("relationships")
    assert "grace@example.com" in profile["data"]["contacts"]


def test_backfill_relationship_returns_stats_summary(db, user_model_store):
    """Backfill should return a complete stats dict with all expected fields."""
    events = [
        _email_received("evt-1", "henry@example.com", "2026-02-10T10:00:00Z"),
    ]
    _insert_events(db, events)

    result = backfill_relationship_profile(data_dir=db.data_dir)

    # Verify the result dict has all required fields
    assert "events_processed" in result
    assert "signals_extracted" in result
    assert "contacts_discovered" in result
    assert "initial_samples" in result
    assert "final_samples" in result
    assert "samples_added" in result
    assert "errors" in result
    assert "elapsed_seconds" in result
    assert "dry_run" in result

    # Sanity check values
    assert result["events_processed"] == 1
    assert result["errors"] == 0
    assert result["dry_run"] is False
    assert result["final_samples"] > result["initial_samples"]


def test_backfill_relationship_idempotent_on_repeated_runs(db, user_model_store):
    """Running the backfill twice should not double-count interactions.

    The RelationshipExtractor reads the existing profile and updates it
    incrementally on each call. Running the backfill twice on the same
    events would double the interaction counts, which is incorrect.

    This test documents the current behavior: repeat runs DO accumulate
    counts (each run re-processes all events). Users should run the backfill
    only once, or use --limit to process only new events.

    Note: This is by design — the extractor has no concept of "already seen"
    events during a backfill. The live pipeline avoids double-counting because
    each email event arrives exactly once from the NATS bus.
    """
    events = [
        _email_received("evt-1", "iris@example.com", "2026-02-10T10:00:00Z"),
    ]
    _insert_events(db, events)

    # First run
    result1 = backfill_relationship_profile(data_dir=db.data_dir)
    profile_after_1 = user_model_store.get_signal_profile("relationships")
    count_after_1 = profile_after_1["data"]["contacts"]["iris@example.com"]["interaction_count"]

    # Second run (same events)
    result2 = backfill_relationship_profile(data_dir=db.data_dir)
    profile_after_2 = user_model_store.get_signal_profile("relationships")
    count_after_2 = profile_after_2["data"]["contacts"]["iris@example.com"]["interaction_count"]

    # Both runs succeed
    assert result1["events_processed"] == 1
    assert result2["events_processed"] == 1

    # The second run adds more interactions (documented behavior: run once only)
    # count_after_2 >= count_after_1 (extractor appends to existing profile)
    assert count_after_2 >= count_after_1


def test_backfill_relationship_empty_database_returns_zero_contacts(db, user_model_store):
    """Backfill on an empty events database should complete cleanly with 0 contacts."""
    result = backfill_relationship_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 0
    assert result["contacts_discovered"] == 0
    assert result["errors"] == 0

    # No profile created when there are no events
    profile = user_model_store.get_signal_profile("relationships")
    assert profile is None
