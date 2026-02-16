"""
Tests for BehavioralAccuracyTracker email address matching fix.

Regression tests for the bug where email.sent events use 'to_addresses' (list)
instead of 'to' (string), breaking the behavioral accuracy inference loop.

This fix enables the tracker to automatically infer prediction accuracy from
user behavior, closing the feedback loop without explicit user interaction.
"""

import json
import sqlite3
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from models.user_model import Prediction
from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker
from storage.database import DatabaseManager


@pytest.mark.asyncio
async def test_email_sent_with_to_addresses_list(db):
    """Verify tracker matches email.sent events using to_addresses list."""
    # Create a reminder prediction for Alice
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                "Unreplied message from alice@example.com",
                0.75,
                "suggest",
                json.dumps({"contact_email": "alice@example.com"}),
                1,  # was_surfaced
                created_at.isoformat(),
            ),
        )

    # Create email.sent event to alice@example.com (using to_addresses list)
    event_id = str(uuid.uuid4())
    sent_at = created_at + timedelta(hours=1)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event_id,
                "email.sent",
                "proton_mail",
                sent_at.isoformat(),
                json.dumps({
                    "message_id": "<test@mail.example.com>",
                    "to_addresses": ["alice@example.com"],  # LIST format
                    "from_address": "user@example.com",
                    "subject": "Re: Follow up",
                }),
            ),
        )

    # Run tracker
    tracker = BehavioralAccuracyTracker(db)
    stats = await tracker.run_inference_cycle()

    # Should mark prediction as accurate
    assert stats["marked_accurate"] == 1
    assert stats["marked_inaccurate"] == 0

    # Verify database update
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()

        assert row["was_accurate"] == 1
        assert row["user_response"] == "inferred"


@pytest.mark.asyncio
async def test_message_sent_with_to_string(db):
    """Verify tracker still matches message.sent events using to string."""
    # Create a reminder prediction for Bob
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                "Unreplied message from bob@example.com",
                0.75,
                "suggest",
                json.dumps({"contact_email": "bob@example.com"}),
                1,  # was_surfaced
                created_at.isoformat(),
            ),
        )

    # Create message.sent event to bob@example.com (using to string)
    event_id = str(uuid.uuid4())
    sent_at = created_at + timedelta(hours=1)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event_id,
                "message.sent",
                "signal",
                sent_at.isoformat(),
                json.dumps({
                    "to": "bob@example.com",  # STRING format
                    "from": "user",
                    "body": "Hey Bob, following up...",
                }),
            ),
        )

    # Run tracker
    tracker = BehavioralAccuracyTracker(db)
    stats = await tracker.run_inference_cycle()

    # Should mark prediction as accurate
    assert stats["marked_accurate"] == 1
    assert stats["marked_inaccurate"] == 0


@pytest.mark.asyncio
async def test_multiple_recipients_in_to_addresses(db):
    """Verify tracker matches when target is one of multiple recipients."""
    # Create a reminder prediction for Carol
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                "Unreplied message from carol@example.com",
                0.75,
                "suggest",
                json.dumps({"contact_email": "carol@example.com"}),
                1,
                created_at.isoformat(),
            ),
        )

    # Create email.sent event to multiple recipients including Carol
    event_id = str(uuid.uuid4())
    sent_at = created_at + timedelta(hours=1)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event_id,
                "email.sent",
                "google",
                sent_at.isoformat(),
                json.dumps({
                    "message_id": "<test@mail.example.com>",
                    "to_addresses": ["alice@example.com", "bob@example.com", "carol@example.com"],
                    "from_address": "user@example.com",
                    "subject": "Team update",
                }),
            ),
        )

    # Run tracker
    tracker = BehavioralAccuracyTracker(db)
    stats = await tracker.run_inference_cycle()

    # Should match Carol in the recipient list
    assert stats["marked_accurate"] == 1


@pytest.mark.asyncio
async def test_no_match_marks_inaccurate_after_48h(db):
    """Verify tracker marks prediction inaccurate if no action after 48 hours."""
    # Create a reminder prediction from 50 hours ago
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=50)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                "Unreplied message from dave@example.com",
                0.75,
                "suggest",
                json.dumps({"contact_email": "dave@example.com"}),
                1,
                created_at.isoformat(),
            ),
        )

    # No outbound event created

    # Run tracker
    tracker = BehavioralAccuracyTracker(db)
    stats = await tracker.run_inference_cycle()

    # Should mark as inaccurate (48+ hours, no action)
    assert stats["marked_accurate"] == 0
    assert stats["marked_inaccurate"] == 1

    # Verify database
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()

        assert row["was_accurate"] == 0


@pytest.mark.asyncio
async def test_case_insensitive_matching(db):
    """Verify email matching is case-insensitive."""
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                "Unreplied message from Alice.Smith@EXAMPLE.COM",
                0.75,
                "suggest",
                json.dumps({"contact_email": "Alice.Smith@EXAMPLE.COM"}),
                1,
                created_at.isoformat(),
            ),
        )

    # Email sent to lowercase version
    event_id = str(uuid.uuid4())
    sent_at = created_at + timedelta(hours=1)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event_id,
                "email.sent",
                "google",
                sent_at.isoformat(),
                json.dumps({
                    "to_addresses": ["alice.smith@example.com"],  # lowercase
                }),
            ),
        )

    # Run tracker
    tracker = BehavioralAccuracyTracker(db)
    stats = await tracker.run_inference_cycle()

    # Should match despite case difference
    assert stats["marked_accurate"] == 1


@pytest.mark.asyncio
async def test_ignores_unsurfaced_predictions(db):
    """Verify tracker only processes surfaced predictions."""
    # Create prediction with was_surfaced = 0
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                "Unreplied message from eve@example.com",
                0.75,
                "suggest",
                json.dumps({"contact_email": "eve@example.com"}),
                0,  # NOT surfaced
                created_at.isoformat(),
            ),
        )

    # Create matching email.sent event
    event_id = str(uuid.uuid4())
    sent_at = created_at + timedelta(hours=1)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event_id,
                "email.sent",
                "google",
                sent_at.isoformat(),
                json.dumps({
                    "to_addresses": ["eve@example.com"],
                }),
            ),
        )

    # Run tracker
    tracker = BehavioralAccuracyTracker(db)
    stats = await tracker.run_inference_cycle()

    # Should NOT process unsurfaced prediction
    assert stats["marked_accurate"] == 0
    assert stats["marked_inaccurate"] == 0


@pytest.mark.asyncio
async def test_within_6_to_48_hour_window(db):
    """Verify tracker waits for full window before marking inaccurate."""
    # Create prediction from 12 hours ago (within 48-hour window)
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=12)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                "Unreplied message from frank@example.com",
                0.75,
                "suggest",
                json.dumps({"contact_email": "frank@example.com"}),
                1,
                created_at.isoformat(),
            ),
        )

    # No outbound event

    # Run tracker
    tracker = BehavioralAccuracyTracker(db)
    stats = await tracker.run_inference_cycle()

    # Should NOT mark anything yet (still within 48-hour window)
    assert stats["marked_accurate"] == 0
    assert stats["marked_inaccurate"] == 0

    # Prediction should still be unresolved
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolved_at FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()

        assert row["resolved_at"] is None


@pytest.mark.asyncio
async def test_handles_missing_to_fields_gracefully(db):
    """Verify tracker doesn't crash on malformed event payloads."""
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                "Unreplied message from grace@example.com",
                0.75,
                "suggest",
                json.dumps({"contact_email": "grace@example.com"}),
                1,
                created_at.isoformat(),
            ),
        )

    # Create event with NEITHER to nor to_addresses
    event_id = str(uuid.uuid4())
    sent_at = created_at + timedelta(hours=1)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event_id,
                "email.sent",
                "google",
                sent_at.isoformat(),
                json.dumps({
                    "message_id": "<test@example.com>",
                    # Missing to_addresses field
                }),
            ),
        )

    # Run tracker (should not crash)
    tracker = BehavioralAccuracyTracker(db)
    stats = await tracker.run_inference_cycle()

    # Should not match (no recipient field)
    assert stats["marked_accurate"] == 0
