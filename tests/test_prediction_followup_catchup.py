"""
Tests for prediction follow-up catchup improvements.

Validates two fixes to _check_follow_up_needs():
1. Wider lookback (72h) on first cycle to catch historical emails
2. Follow-up checks run on time-based triggers (not just event-based)
3. Deduplication still prevents duplicate predictions across cycles
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# First-cycle wider lookback (72h vs 24h)
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_cycle_catches_emails_older_than_24h(db, event_store, user_model_store):
    """On first cycle, emails 25-72h old should generate predictions (they wouldn't before)."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert an email from 36 hours ago — outside the old 24h window
    # but inside the new 72h first-run window
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=36)).isoformat(),
        "payload": {
            "from_address": "boss@company.com",
            "subject": "Quarterly review prep",
            "snippet": "Please prepare the deck by Friday",
            "message_id": "msg-36h-old",
        },
        "metadata": {},
    })

    # First run should use 72h lookback and find this email
    predictions = await engine._check_follow_up_needs({})
    assert len(predictions) >= 1, "First cycle should find emails older than 24h but within 72h"
    assert predictions[0].relevant_contacts == ["boss@company.com"]
    assert predictions[0].prediction_type == "reminder"


@pytest.mark.asyncio
async def test_first_cycle_catches_emails_at_boundary(db, event_store, user_model_store):
    """On first cycle, an email from 70h ago should still be caught (within 72h)."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=70)).isoformat(),
        "payload": {
            "from_address": "colleague@work.com",
            "subject": "Can you review my PR?",
            "snippet": "I need your eyes on this",
            "message_id": "msg-70h-old",
        },
        "metadata": {},
    })

    predictions = await engine._check_follow_up_needs({})
    assert len(predictions) >= 1, "First cycle should catch emails up to 72h old"


# -------------------------------------------------------------------------
# Subsequent cycles revert to 24h lookback
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subsequent_cycle_uses_24h_lookback(db, event_store, user_model_store):
    """On subsequent cycles, emails >24h old should NOT generate new predictions."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert an email from 36 hours ago
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=36)).isoformat(),
        "payload": {
            "from_address": "old-email@company.com",
            "subject": "Old thread",
            "message_id": "msg-old-thread",
        },
        "metadata": {},
    })

    # Also insert a recent email (6h ago) for the second cycle to find
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=6)).isoformat(),
        "payload": {
            "from_address": "recent@company.com",
            "subject": "Recent question",
            "message_id": "msg-recent",
        },
        "metadata": {},
    })

    # First cycle: uses 72h lookback — finds both emails
    first_preds = await engine._check_follow_up_needs({})
    first_contacts = {p.relevant_contacts[0] for p in first_preds}
    assert "old-email@company.com" in first_contacts, "First cycle should find 36h-old email"
    assert "recent@company.com" in first_contacts, "First cycle should find recent email"

    # Second cycle: uses 24h lookback — only finds recent email
    # But dedup already caught recent@company.com, so no NEW predictions
    # To test the lookback properly, add a new email in the 24h window
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=5)).isoformat(),
        "payload": {
            "from_address": "another-recent@company.com",
            "subject": "Another question",
            "message_id": "msg-another-recent",
        },
        "metadata": {},
    })

    # Verify the flag was consumed
    assert engine._first_follow_up_run is False, "Flag should be False after first cycle"

    second_preds = await engine._check_follow_up_needs({})
    second_contacts = {p.relevant_contacts[0] for p in second_preds}
    # The 36h-old email should NOT appear in second cycle (outside 24h window)
    assert "old-email@company.com" not in second_contacts, (
        "Second cycle should NOT find emails outside 24h window"
    )
    # The new recent email should be found
    assert "another-recent@company.com" in second_contacts, (
        "Second cycle should find emails within 24h window"
    )


# -------------------------------------------------------------------------
# Follow-up predictions on time-based triggers (no new events)
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_followup_runs_on_time_trigger_without_new_events(db, event_store, user_model_store):
    """Follow-up predictions should generate on time-based triggers even when has_new_events is False."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert an unreplied email old enough to trigger (6 hours)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=6)).isoformat(),
        "payload": {
            "from_address": "boss@company.com",
            "subject": "Need your input",
            "snippet": "Please review the proposal",
            "message_id": "msg-time-trigger-test",
        },
        "metadata": {},
    })

    # First call to generate_predictions sets the cursor and runs
    first_preds = await engine.generate_predictions({})

    # Verify at least one follow-up prediction was generated
    followup_preds = [p for p in first_preds if p.prediction_type == "reminder"]
    assert len(followup_preds) >= 1, "First run should generate follow-up predictions"

    # Now simulate time passing: advance the time-based trigger
    # without adding new events (simulates connector outage)
    engine._last_time_based_run = now - timedelta(minutes=20)

    # has_new_events will be False (no new events added)
    # time_based_due will be True (>15 min since last run)
    # Follow-up should still run (it's now in the HYBRID section)
    assert engine._has_new_events() is False, "No new events should exist"
    assert engine._should_run_time_based_predictions() is True, "Time trigger should be active"

    # Add a NEW unreplied email (within 24h, but not as a new cursor event)
    # We need to insert directly to avoid advancing the cursor
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=4)).isoformat(),
        "payload": {
            "from_address": "teammate@company.com",
            "subject": "Sprint planning",
            "snippet": "What do you think about the timeline?",
            "message_id": "msg-time-trigger-test-2",
        },
        "metadata": {},
    })

    # Reset time trigger again so it fires
    engine._last_time_based_run = now - timedelta(minutes=20)

    # generate_predictions should run follow-up checks even without new events
    # (time_based_due=True but has_new_events may vary due to cursor)
    second_preds = await engine.generate_predictions({})
    # The new email should generate a prediction
    followup_contacts = [
        p.relevant_contacts[0] for p in second_preds if p.prediction_type == "reminder"
    ]
    assert "teammate@company.com" in followup_contacts, (
        "Follow-up should run on time-based trigger and find new unreplied email"
    )


# -------------------------------------------------------------------------
# Deduplication across cycles
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deduplication_prevents_duplicates_across_cycles(db, event_store, user_model_store):
    """Running follow-up checks twice should NOT create duplicate predictions."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert an unreplied email
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=6)).isoformat(),
        "payload": {
            "from_address": "boss@company.com",
            "subject": "Action needed",
            "snippet": "Can you handle this?",
            "message_id": "msg-dedup-test",
        },
        "metadata": {},
    })

    # First cycle: should produce a prediction
    first_preds = await engine.generate_predictions({})
    first_followups = [p for p in first_preds if p.prediction_type == "reminder"]
    assert len(first_followups) >= 1, "First cycle should generate a follow-up prediction"

    # Second cycle: force time trigger (follow-up now runs on time triggers too)
    engine._last_time_based_run = now - timedelta(minutes=20)

    second_preds = await engine.generate_predictions({})
    second_followups = [p for p in second_preds if p.prediction_type == "reminder"]

    # The same message should NOT generate a duplicate prediction
    dedup_contacts = [
        p.relevant_contacts[0]
        for p in second_followups
        if p.supporting_signals and p.supporting_signals.get("message_id") == "msg-dedup-test"
    ]
    assert len(dedup_contacts) == 0, (
        "Deduplication should prevent the same email from generating a second prediction"
    )


@pytest.mark.asyncio
async def test_deduplication_allows_different_messages(db, event_store, user_model_store):
    """Deduplication should only block the same message_id, not different messages."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert two different unreplied emails
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=6)).isoformat(),
        "payload": {
            "from_address": "alice@company.com",
            "subject": "Budget review",
            "message_id": "msg-alice-1",
        },
        "metadata": {},
    })

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=5)).isoformat(),
        "payload": {
            "from_address": "bob@company.com",
            "subject": "Design feedback",
            "message_id": "msg-bob-1",
        },
        "metadata": {},
    })

    predictions = await engine._check_follow_up_needs({})
    contacts = {p.relevant_contacts[0] for p in predictions}
    assert "alice@company.com" in contacts, "Alice's email should generate a prediction"
    assert "bob@company.com" in contacts, "Bob's email should generate a separate prediction"


# -------------------------------------------------------------------------
# First-run flag state management
# -------------------------------------------------------------------------


def test_first_follow_up_run_flag_initializes_true(db, user_model_store):
    """The _first_follow_up_run flag should start as True."""
    engine = PredictionEngine(db, user_model_store)
    assert engine._first_follow_up_run is True


@pytest.mark.asyncio
async def test_first_follow_up_run_flag_flips_after_first_call(db, event_store, user_model_store):
    """The flag should flip to False after the first _check_follow_up_needs call."""
    engine = PredictionEngine(db, user_model_store)
    assert engine._first_follow_up_run is True

    await engine._check_follow_up_needs({})
    assert engine._first_follow_up_run is False

    # Calling again should keep it False
    await engine._check_follow_up_needs({})
    assert engine._first_follow_up_run is False
