"""
Tests for Behavioral Accuracy Tracking of Filtered Predictions

The Problem:
    316K predictions have user_response='filtered' but was_accurate=NULL, meaning they
    never contributed to the learning loop. The behavioral accuracy tracker only looked
    at surfaced predictions (was_surfaced=1), completely ignoring filtered ones.

    This breaks the calibration loop: if the system filters out a prediction but the
    user takes the action anyway, that's a FALSE NEGATIVE that should lower confidence
    in the filter. If the user doesn't take the action, that's a TRUE NEGATIVE that
    should increase confidence.

The Fix:
    Extend BehavioralAccuracyTracker.run_inference_cycle() to also process filtered
    predictions (was_surfaced=0, user_response='filtered'). Check if the user took
    the predicted action anyway:
    - If YES → was_accurate=1 (filter was wrong, false negative)
    - If NO → was_accurate=0 (filter was right, true negative)

Expected Behavior:
    - Filtered predictions aged 48h-7d are evaluated for behavioral signals
    - Predictions with matching behavior are marked was_accurate=1
    - Predictions with no matching behavior are marked was_accurate=0
    - user_response stays 'filtered' to preserve provenance
    - Stats distinguish between surfaced and filtered processing
"""

import pytest
from datetime import datetime, timedelta, timezone
import json

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker
from models.user_model import Prediction


@pytest.mark.asyncio
async def test_filtered_prediction_false_negative_reminder(db):
    """If user takes action on filtered reminder prediction, mark as accurate (false negative)."""
    # Create a filtered reminder prediction (3 days ago)
    three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
    tracker = BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred1",
                "reminder",
                "Unreplied message from alice@example.com: \"Can you review the doc?\" (72 hours ago)",
                0.55,
                "OBSERVE",  # Filtered due to low confidence
                "Reply to alice@example.com",
                json.dumps({"contact_email": "alice@example.com"}),
                0,  # NOT surfaced
                "filtered",
                three_days_ago.isoformat(),
                three_days_ago.isoformat(),  # Auto-resolved when filtered
                None,  # BUG: was_accurate is NULL!
            ),
        )

    # User DID reply to Alice within 24 hours (despite prediction being filtered)
    two_days_ago = three_days_ago + timedelta(hours=12)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "evt1",
                "email.sent",
                "protonmail",
                two_days_ago.isoformat(),
                "normal",
                json.dumps({"to_addresses": ["alice@example.com"], "subject": "Re: Review"}),
                "{}",
            ),
        )

    # Run behavioral accuracy tracker
    stats = await tracker.run_inference_cycle()

    # Verify the filtered prediction was marked as accurate (filter was WRONG)
    with db.get_connection("user_model") as conn:
        pred = conn.execute("SELECT * FROM predictions WHERE id = 'pred1'").fetchone()
        assert pred["was_accurate"] == 1, "User DID take action → filter was wrong (false negative)"
        assert pred["user_response"] == "filtered", "Should preserve 'filtered' provenance"
        assert pred["resolved_at"] is not None
        # resolved_at should be updated to now, not the original auto-resolve time
        resolved_time = datetime.fromisoformat(pred["resolved_at"].replace('Z', '+00:00'))
        assert (datetime.now(timezone.utc) - resolved_time).total_seconds() < 10

    # Verify stats
    assert stats['marked_accurate'] == 1
    assert stats['marked_inaccurate'] == 0
    assert stats['filtered'] == 1
    assert stats['surfaced'] == 0


@pytest.mark.asyncio
async def test_filtered_prediction_true_negative_reminder(db):
    """If user doesn't take action on filtered reminder, mark as inaccurate (true negative)."""
    # Create a filtered reminder prediction (3 days ago)
    three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
    tracker = BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred1",
                "reminder",
                "Unreplied message from bob@example.com: \"Want to grab lunch?\" (72 hours ago)",
                0.45,
                "OBSERVE",
                "Reply to bob@example.com",
                json.dumps({"contact_email": "bob@example.com"}),
                0,  # NOT surfaced
                "filtered",
                three_days_ago.isoformat(),
                three_days_ago.isoformat(),
                None,  # BUG: was_accurate is NULL!
            ),
        )

    # User did NOT reply to Bob (no matching events)
    # Run behavioral accuracy tracker (48+ hours later, so we can infer)
    stats = await tracker.run_inference_cycle()

    # Verify the filtered prediction was marked as inaccurate (filter was RIGHT)
    with db.get_connection("user_model") as conn:
        pred = conn.execute("SELECT * FROM predictions WHERE id = 'pred1'").fetchone()
        assert pred["was_accurate"] == 0, "User didn't take action → filter was right (true negative)"
        assert pred["user_response"] == "filtered", "Should preserve 'filtered' provenance"

    # Verify stats
    assert stats['marked_accurate'] == 0
    assert stats['marked_inaccurate'] == 1
    assert stats['filtered'] == 1


@pytest.mark.asyncio
async def test_filtered_predictions_too_recent_not_processed(db):
    """Filtered predictions <48 hours old should not be processed (need time for behavior)."""
    # Create a filtered prediction 12 hours ago (too recent)
    twelve_hours_ago = datetime.now(timezone.utc) - timedelta(hours=12)
    tracker = BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred1",
                "reminder",
                "Unreplied message from alice@example.com",
                0.40,
                "OBSERVE",
                "Reply to alice@example.com",
                json.dumps({"contact_email": "alice@example.com"}),
                0,
                "filtered",
                twelve_hours_ago.isoformat(),
                twelve_hours_ago.isoformat(),
                None,
            ),
        )

    # Run tracker
    stats = await tracker.run_inference_cycle()

    # Verify it was NOT processed (too recent)
    with db.get_connection("user_model") as conn:
        pred = conn.execute("SELECT * FROM predictions WHERE id = 'pred1'").fetchone()
        assert pred["was_accurate"] is None, "Too recent to infer behavior"

    assert stats['filtered'] == 0


@pytest.mark.asyncio
async def test_filtered_predictions_too_old_not_processed(db):
    """Filtered predictions >7 days old should not be processed (no longer relevant)."""
    # Create a filtered prediction 10 days ago (too old)
    ten_days_ago = datetime.now(timezone.utc) - timedelta(days=10)
    tracker = BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred1",
                "reminder",
                "Unreplied message from alice@example.com",
                0.40,
                "OBSERVE",
                "Reply to alice@example.com",
                json.dumps({"contact_email": "alice@example.com"}),
                0,
                "filtered",
                ten_days_ago.isoformat(),
                ten_days_ago.isoformat(),
                None,
            ),
        )

    # Run tracker
    stats = await tracker.run_inference_cycle()

    # Verify it was NOT processed (too old)
    with db.get_connection("user_model") as conn:
        pred = conn.execute("SELECT * FROM predictions WHERE id = 'pred1'").fetchone()
        assert pred["was_accurate"] is None, "Too old to be relevant"

    assert stats['filtered'] == 0


@pytest.mark.asyncio
async def test_filtered_predictions_already_resolved_skipped(db):
    """Filtered predictions with was_accurate already set should be skipped."""
    three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
    tracker = BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred1",
                "reminder",
                "Unreplied message from alice@example.com",
                0.40,
                "OBSERVE",
                "Reply to alice@example.com",
                json.dumps({"contact_email": "alice@example.com"}),
                0,
                "filtered",
                three_days_ago.isoformat(),
                three_days_ago.isoformat(),
                1,  # Already resolved!
            ),
        )

    # Run tracker
    stats = await tracker.run_inference_cycle()

    # Verify it was NOT processed (already has was_accurate)
    assert stats['filtered'] == 0


@pytest.mark.asyncio
async def test_surfaced_and_filtered_processed_in_same_cycle(db):
    """Both surfaced and filtered predictions should be processed in one cycle."""
    three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
    tracker = BehavioralAccuracyTracker(db)

    # Create one surfaced prediction (not yet resolved)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "surfaced1",
                "reminder",
                "Unreplied message from alice@example.com",
                0.75,
                "SUGGEST",
                "Reply to alice@example.com",
                json.dumps({"contact_email": "alice@example.com"}),
                1,  # Surfaced
                None,
                three_days_ago.isoformat(),
                None,  # NOT resolved yet
                None,
            ),
        )

        # Create one filtered prediction
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "filtered1",
                "reminder",
                "Unreplied message from bob@example.com",
                0.40,
                "OBSERVE",
                "Reply to bob@example.com",
                json.dumps({"contact_email": "bob@example.com"}),
                0,  # Filtered
                "filtered",
                three_days_ago.isoformat(),
                three_days_ago.isoformat(),
                None,
            ),
        )

    # User replied to Alice but not Bob
    two_days_ago = three_days_ago + timedelta(hours=12)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "evt1",
                "email.sent",
                "protonmail",
                two_days_ago.isoformat(),
                "normal",
                json.dumps({"to_addresses": ["alice@example.com"]}),
                "{}",
            ),
        )

    # Run tracker
    stats = await tracker.run_inference_cycle()

    # Verify both were processed
    with db.get_connection("user_model") as conn:
        surfaced = conn.execute("SELECT * FROM predictions WHERE id = 'surfaced1'").fetchone()
        filtered = conn.execute("SELECT * FROM predictions WHERE id = 'filtered1'").fetchone()

        assert surfaced["was_accurate"] == 1, "Surfaced prediction was accurate"
        assert surfaced["user_response"] == "inferred"

        assert filtered["was_accurate"] == 0, "Filtered prediction was inaccurate (no action)"
        assert filtered["user_response"] == "filtered", "Preserves filtered provenance"

    # Verify stats
    assert stats['marked_accurate'] == 1  # Alice reply
    assert stats['marked_inaccurate'] == 1  # Bob no reply
    assert stats['surfaced'] == 1
    assert stats['filtered'] == 1


@pytest.mark.asyncio
async def test_filtered_prediction_partial_email_match(db):
    """Filtered predictions should match complex email addresses correctly."""
    three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
    tracker = BehavioralAccuracyTracker(db)

    # Create filtered prediction with complex email
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred1",
                "reminder",
                "Unreplied message from john.doe+work@company-name.co.uk",
                0.40,
                "OBSERVE",
                "Reply to john.doe+work@company-name.co.uk",
                json.dumps({"contact_email": "john.doe+work@company-name.co.uk"}),
                0,
                "filtered",
                three_days_ago.isoformat(),
                three_days_ago.isoformat(),
                None,
            ),
        )

    # User replied
    two_days_ago = three_days_ago + timedelta(hours=6)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "evt1",
                "email.sent",
                "protonmail",
                two_days_ago.isoformat(),
                "normal",
                json.dumps({"to_addresses": ["john.doe+work@company-name.co.uk"]}),
                "{}",
            ),
        )

    # Run tracker
    stats = await tracker.run_inference_cycle()

    # Verify
    with db.get_connection("user_model") as conn:
        pred = conn.execute("SELECT * FROM predictions WHERE id = 'pred1'").fetchone()
        assert pred["was_accurate"] == 1, "Should match complex email addresses"

    assert stats['marked_accurate'] == 1
    assert stats['filtered'] == 1


@pytest.mark.asyncio
async def test_stats_breakdown_by_type(db):
    """Stats should correctly break down surfaced vs filtered processing."""
    three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
    tracker = BehavioralAccuracyTracker(db)

    # Create 2 surfaced, 3 filtered
    with db.get_connection("user_model") as conn:
        # Surfaced 1 (accurate)
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "s1", "reminder", "Unreplied message from a@example.com", 0.75, "SUGGEST",
                "Reply to a@example.com", json.dumps({"contact_email": "a@example.com"}),
                1, None, three_days_ago.isoformat(), None, None,
            ),
        )
        # Surfaced 2 (inaccurate)
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "s2", "reminder", "Unreplied message from b@example.com", 0.70, "SUGGEST",
                "Reply to b@example.com", json.dumps({"contact_email": "b@example.com"}),
                1, None, three_days_ago.isoformat(), None, None,
            ),
        )
        # Filtered 1 (accurate - false negative)
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "f1", "reminder", "Unreplied message from c@example.com", 0.40, "OBSERVE",
                "Reply to c@example.com", json.dumps({"contact_email": "c@example.com"}),
                0, "filtered", three_days_ago.isoformat(), three_days_ago.isoformat(), None,
            ),
        )
        # Filtered 2 (inaccurate - true negative)
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "f2", "reminder", "Unreplied message from d@example.com", 0.35, "OBSERVE",
                "Reply to d@example.com", json.dumps({"contact_email": "d@example.com"}),
                0, "filtered", three_days_ago.isoformat(), three_days_ago.isoformat(), None,
            ),
        )
        # Filtered 3 (inaccurate - true negative)
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, user_response,
                created_at, resolved_at, was_accurate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "f3", "reminder", "Unreplied message from e@example.com", 0.30, "OBSERVE",
                "Reply to e@example.com", json.dumps({"contact_email": "e@example.com"}),
                0, "filtered", three_days_ago.isoformat(), three_days_ago.isoformat(), None,
            ),
        )

    # User replied to A and C only
    two_days_ago = three_days_ago + timedelta(hours=6)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("e1", "email.sent", "protonmail", two_days_ago.isoformat(), "normal",
             json.dumps({"to_addresses": ["a@example.com"]}), "{}"),
        )
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("e2", "email.sent", "protonmail", two_days_ago.isoformat(), "normal",
             json.dumps({"to_addresses": ["c@example.com"]}), "{}"),
        )

    # Run tracker
    stats = await tracker.run_inference_cycle()

    # Verify stats
    assert stats['marked_accurate'] == 2, "A and C"
    assert stats['marked_inaccurate'] == 3, "B, D, E"
    assert stats['surfaced'] == 2, "S1 and S2"
    assert stats['filtered'] == 3, "F1, F2, F3"
