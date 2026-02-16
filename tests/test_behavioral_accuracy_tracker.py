"""
Tests for Behavioral Prediction Accuracy Tracker.

The behavioral accuracy tracker infers prediction accuracy from user behavior,
closing the feedback loop without requiring explicit notification interaction.
This is critical for bootstrapping the prediction engine's learning on new systems.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


@pytest.mark.asyncio
async def test_reminder_prediction_accurate_when_user_replies(db, user_model_store):
    """When user sends a message to the predicted contact, mark reminder as accurate."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a reminder prediction: "Reply to Alice"
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=2)).isoformat()  # 2 hours ago

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Reply to Alice about dinner plans",
                0.75,
                "DEFAULT",
                "Send message to Alice",
                json.dumps({"contact_name": "Alice", "contact_id": "alice@example.com"}),
                1,
                created_at,
            ),
        )

    # User sends a message to Alice 1 hour after the prediction
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "message.sent",
                "signal_connector",
                (now - timedelta(hours=1)).isoformat(),
                json.dumps({"to": "alice@example.com", "body": "Let's do dinner at 7pm!"}),
            ),
        )

    # Run inference
    stats = await tracker.run_inference_cycle()

    assert stats['marked_accurate'] == 1
    assert stats['marked_inaccurate'] == 0

    # Verify prediction was marked accurate
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

        assert pred["was_accurate"] == 1
        assert pred["user_response"] == "inferred"


@pytest.mark.asyncio
async def test_reminder_prediction_inaccurate_when_user_ignores(db, user_model_store):
    """When 48+ hours pass with no action, mark reminder as inaccurate."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a reminder prediction: "Reply to Bob" from 3 days ago
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(days=3)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Follow up with Bob about the project",
                0.65,
                "DEFAULT",
                "Send message to Bob",
                json.dumps({"contact_name": "Bob", "contact_id": "bob@example.com"}),
                1,
                created_at,
            ),
        )

    # No events sent to Bob — inference should mark as inaccurate

    # Run inference
    stats = await tracker.run_inference_cycle()

    assert stats['marked_accurate'] == 0
    assert stats['marked_inaccurate'] == 1

    # Verify prediction was marked inaccurate
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

        assert pred["was_accurate"] == 0
        assert pred["user_response"] == "inferred"


@pytest.mark.asyncio
async def test_reminder_prediction_pending_within_window(db, user_model_store):
    """Within the 48-hour window, don't mark prediction yet (insufficient evidence)."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a reminder prediction from 6 hours ago (well within 48h window)
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=6)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Reply to Charlie about meeting time",
                0.70,
                "DEFAULT",
                "Send message to Charlie",
                json.dumps({"contact_name": "Charlie"}),
                1,
                created_at,
            ),
        )

    # No message sent yet, but still within the window

    # Run inference
    stats = await tracker.run_inference_cycle()

    assert stats['marked_accurate'] == 0
    assert stats['marked_inaccurate'] == 0

    # Verify prediction is still unresolved
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

        assert pred["was_accurate"] is None
        assert pred["resolved_at"] is None


@pytest.mark.asyncio
async def test_conflict_prediction_accurate_when_event_rescheduled(db, user_model_store):
    """When user resolves a calendar conflict, mark prediction as accurate."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a conflict prediction
    pred_id = str(uuid.uuid4())
    event1_id = "event-meeting-123"
    event2_id = "event-dentist-456"
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=3)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "conflict",
                "Calendar conflict: Team sync overlaps with dentist appointment",
                0.95,
                "AUTONOMOUS",
                "Reschedule one of the events",
                json.dumps({"conflicting_event_ids": [event1_id, event2_id]}),
                1,
                created_at,
            ),
        )

    # User updates one of the conflicting events (reschedules it)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "calendar.event.updated",
                "caldav_connector",
                (now - timedelta(hours=1)).isoformat(),
                json.dumps({"event_id": event1_id, "start": "2026-02-16T15:00:00Z"}),
            ),
        )

    # Run inference
    stats = await tracker.run_inference_cycle()

    assert stats['marked_accurate'] == 1
    assert stats['marked_inaccurate'] == 0

    # Verify prediction was marked accurate
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

        assert pred["was_accurate"] == 1
        assert pred["user_response"] == "inferred"


@pytest.mark.asyncio
async def test_conflict_prediction_accurate_even_if_ignored(db, user_model_store):
    """After 24h, conflict predictions are marked accurate even if user didn't act."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a conflict prediction from 30 hours ago
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=30)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "conflict",
                "Calendar conflict: Standup meeting overlaps with doctor appointment",
                0.90,
                "DEFAULT",
                "Reschedule one of the events",
                json.dumps({"conflicting_event_ids": ["evt-1", "evt-2"]}),
                1,
                created_at,
            ),
        )

    # No calendar updates — user ignored the conflict

    # Run inference
    stats = await tracker.run_inference_cycle()

    assert stats['marked_accurate'] == 1
    assert stats['marked_inaccurate'] == 0

    # Verify: conflict was real even if user didn't fix it
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

        assert pred["was_accurate"] == 1
        assert pred["user_response"] == "inferred"


@pytest.mark.asyncio
async def test_ignores_already_resolved_predictions(db, user_model_store):
    """Don't re-process predictions that already have resolution."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a prediction that was already resolved by explicit user feedback
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=10)).isoformat()
    resolved_at = (now - timedelta(hours=5)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at,
                resolved_at, was_accurate, user_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Reply to David",
                0.70,
                "DEFAULT",
                "Send message",
                json.dumps({"contact_name": "David"}),
                1,
                created_at,
                resolved_at,
                1,
                "acted_on",  # User clicked "Act On" button
            ),
        )

    # Run inference
    stats = await tracker.run_inference_cycle()

    # Should not modify the already-resolved prediction
    assert stats['marked_accurate'] == 0
    assert stats['marked_inaccurate'] == 0

    # Verify prediction unchanged
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT user_response FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

        assert pred["user_response"] == "acted_on"  # Still the original value


@pytest.mark.asyncio
async def test_ignores_predictions_older_than_7_days(db, user_model_store):
    """Don't process very old predictions (those are handled by auto-resolve)."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a prediction from 10 days ago
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(days=10)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Reply to Emma",
                0.65,
                "DEFAULT",
                "Send message",
                json.dumps({"contact_name": "Emma"}),
                1,
                created_at,
            ),
        )

    # Run inference
    stats = await tracker.run_inference_cycle()

    # Should not process predictions older than 7 days
    assert stats['marked_accurate'] == 0
    assert stats['marked_inaccurate'] == 0


@pytest.mark.asyncio
async def test_only_processes_surfaced_predictions(db, user_model_store):
    """Don't infer accuracy for filtered predictions (was_surfaced=0)."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a filtered prediction (never shown to user)
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=5)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Reply to Frank",
                0.15,  # Low confidence, filtered out
                "OBSERVE",
                "Send message",
                json.dumps({"contact_name": "Frank"}),
                0,  # was_surfaced = 0
                created_at,
            ),
        )

    # Run inference
    stats = await tracker.run_inference_cycle()

    # Should not process unsurfaced predictions
    assert stats['marked_accurate'] == 0
    assert stats['marked_inaccurate'] == 0


@pytest.mark.asyncio
async def test_handles_missing_contact_info_gracefully(db, user_model_store):
    """When prediction lacks contact info, skip inference (return None)."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a reminder prediction without contact info in signals
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=10)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Follow up on that thing",  # Vague, no contact name
                0.60,
                "SUGGEST",
                "Send follow-up",
                json.dumps({}),  # Empty signals
                1,
                created_at,
            ),
        )

    # Run inference
    stats = await tracker.run_inference_cycle()

    # Can't infer without contact info
    assert stats['marked_accurate'] == 0
    assert stats['marked_inaccurate'] == 0


@pytest.mark.asyncio
async def test_extracts_contact_from_description_pattern(db, user_model_store):
    """When signals lack contact info, try extracting from description text."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a reminder with contact name in description but not in signals
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=2)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Reply to Grace about the meeting notes",  # Name in description
                0.70,
                "DEFAULT",
                "Send message",
                json.dumps({}),  # No signals
                1,
                created_at,
            ),
        )

    # User sends a message to Grace
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "message.sent",
                "signal_connector",
                (now - timedelta(hours=1)).isoformat(),
                json.dumps({"to": "Grace <grace@example.com>", "body": "Here are the notes"}),
            ),
        )

    # Run inference (should extract "Grace" from description)
    stats = await tracker.run_inference_cycle()

    assert stats['marked_accurate'] == 1
    assert stats['marked_inaccurate'] == 0


@pytest.mark.asyncio
async def test_handles_malformed_signals_json(db, user_model_store):
    """Gracefully handle predictions with malformed supporting_signals JSON."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a prediction with invalid JSON in signals
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=5)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Reply to Henry",
                0.65,
                "DEFAULT",
                "Send message",
                "not valid json {{{",  # Malformed JSON
                1,
                created_at,
            ),
        )

    # Run inference (should handle gracefully, not crash)
    stats = await tracker.run_inference_cycle()

    # Should not crash, but also can't infer without valid signals
    assert stats['marked_accurate'] == 0
    assert stats['marked_inaccurate'] == 0


@pytest.mark.asyncio
async def test_multiple_predictions_processed_in_batch(db, user_model_store):
    """Process multiple predictions in a single inference cycle."""
    tracker = BehavioralAccuracyTracker(db)

    now = datetime.now(timezone.utc)

    # Create 3 predictions:
    # 1. Reminder that user acted on → accurate
    # 2. Reminder that user ignored → inaccurate
    # 3. Reminder still within window → pending

    pred1_id = str(uuid.uuid4())
    pred2_id = str(uuid.uuid4())
    pred3_id = str(uuid.uuid4())

    with db.get_connection("user_model") as conn:
        # Prediction 1: 5 hours ago, user replied
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred1_id,
                "reminder",
                "Reply to Isaac",
                0.70,
                "DEFAULT",
                "Send message",
                json.dumps({"contact_name": "Isaac"}),
                1,
                (now - timedelta(hours=5)).isoformat(),
            ),
        )

        # Prediction 2: 3 days ago, user ignored
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred2_id,
                "reminder",
                "Reply to Jack",
                0.65,
                "DEFAULT",
                "Send message",
                json.dumps({"contact_name": "Jack"}),
                1,
                (now - timedelta(days=3)).isoformat(),
            ),
        )

        # Prediction 3: 10 hours ago, still within window
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred3_id,
                "reminder",
                "Reply to Kate",
                0.75,
                "DEFAULT",
                "Send message",
                json.dumps({"contact_name": "Kate"}),
                1,
                (now - timedelta(hours=10)).isoformat(),
            ),
        )

    # User sent a message to Isaac (prediction 1)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "message.sent",
                "signal_connector",
                (now - timedelta(hours=3)).isoformat(),
                json.dumps({"to": "Isaac <isaac@example.com>", "body": "Got it!"}),
            ),
        )

    # Run inference
    stats = await tracker.run_inference_cycle()

    assert stats['marked_accurate'] == 1  # Prediction 1
    assert stats['marked_inaccurate'] == 1  # Prediction 2

    # Verify each prediction
    with db.get_connection("user_model") as conn:
        pred1 = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?", (pred1_id,)
        ).fetchone()
        assert pred1["was_accurate"] == 1

        pred2 = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?", (pred2_id,)
        ).fetchone()
        assert pred2["was_accurate"] == 0

        pred3 = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?", (pred3_id,)
        ).fetchone()
        assert pred3["was_accurate"] is None  # Still unresolved


@pytest.mark.asyncio
async def test_idempotent_on_repeated_runs(db, user_model_store):
    """Running inference multiple times doesn't change already-inferred predictions."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a prediction that will be inferred as accurate
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=3)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Reply to Laura",
                0.70,
                "DEFAULT",
                "Send message",
                json.dumps({"contact_name": "Laura"}),
                1,
                created_at,
            ),
        )

    # User sent a message to Laura
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "message.sent",
                "signal_connector",
                (now - timedelta(hours=1)).isoformat(),
                json.dumps({"to": "laura@example.com", "body": "Thanks!"}),
            ),
        )

    # Run inference first time
    stats1 = await tracker.run_inference_cycle()
    assert stats1['marked_accurate'] == 1

    # Run inference second time (should not re-process)
    stats2 = await tracker.run_inference_cycle()
    assert stats2['marked_accurate'] == 0  # Already processed

    # Verify prediction still marked as accurate with original response
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

        assert pred["was_accurate"] == 1
        assert pred["user_response"] == "inferred"
