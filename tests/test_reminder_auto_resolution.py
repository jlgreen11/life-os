"""
Test suite for automatic reminder prediction resolution.

This tests the behavioral accuracy tracker's ability to automatically resolve
"reminder" predictions by observing whether the user actually replied to messages,
without requiring explicit user interaction with notifications.

Critical bug fix: The tracker was failing to extract contact information from
prediction descriptions because the regex pattern didn't match the actual format:
"Unreplied message from EMAIL" instead of "reply to NAME".
"""

import json
import pytest
from datetime import datetime, timedelta, timezone

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


@pytest.mark.asyncio
async def test_extract_email_from_unreplied_message_description(db):
    """Tracker should extract email addresses from 'Unreplied message from EMAIL' format."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a prediction with the actual format used by the prediction engine
    prediction_id = "pred-001"
    created_at = datetime.now(timezone.utc) - timedelta(hours=10)
    contact_email = "alice@example.com"

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                f'Unreplied message from {contact_email}: "Project update" (10 hours ago)',
                0.75,
                "SUGGEST",
                "[]",  # Empty - simulating old predictions before the fix
                created_at.isoformat(),
                1,  # Was surfaced
            ),
        )

    # Create an outbound email to this contact within the 48-hour window
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, payload, source, timestamp, priority)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "event-001",
                "email.sent",
                json.dumps({
                    "to": contact_email,
                    "subject": "Re: Project update",
                    "body": "Thanks for the update!",
                }),
                "proton_mail",
                (created_at + timedelta(hours=5)).isoformat(),
                "normal",
            ),
        )

    # Run the behavioral accuracy tracker
    stats = await tracker.run_inference_cycle()

    # Verify the prediction was marked as accurate
    assert stats["marked_accurate"] == 1
    assert stats["marked_inaccurate"] == 0

    # Verify the database was updated
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()

        assert row["was_accurate"] == 1
        assert row["resolved_at"] is not None
        assert row["user_response"] == "inferred"


@pytest.mark.asyncio
async def test_email_extraction_with_special_characters(db):
    """Tracker should handle email addresses with dots, hyphens, and plus signs."""
    tracker = BehavioralAccuracyTracker(db)

    prediction_id = "pred-002"
    created_at = datetime.now(timezone.utc) - timedelta(hours=8)
    # Complex email with special characters
    contact_email = "john.doe+work@company-name.co.uk"

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                f'Unreplied message from {contact_email}: "Urgent: Review needed" (8 hours ago)',
                0.80,
                "DEFAULT",
                "[]",
                created_at.isoformat(),
                1,
            ),
        )

    # User replies within window
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, payload, source, timestamp, priority)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "event-002",
                "email.sent",
                json.dumps({
                    "to": contact_email,
                    "subject": "Re: Urgent: Review needed",
                }),
                "proton_mail",
                (created_at + timedelta(hours=3)).isoformat(),
                "normal",
            ),
        )

    stats = await tracker.run_inference_cycle()

    assert stats["marked_accurate"] == 1

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert row["was_accurate"] == 1


@pytest.mark.asyncio
async def test_no_reply_marks_as_inaccurate_after_48_hours(db):
    """If user doesn't reply within 48 hours, prediction should be marked inaccurate."""
    tracker = BehavioralAccuracyTracker(db)

    prediction_id = "pred-003"
    # Created 50 hours ago (past the 48-hour window)
    created_at = datetime.now(timezone.utc) - timedelta(hours=50)
    contact_email = "bob@example.com"

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                f'Unreplied message from {contact_email}: "Quick question" (50 hours ago)',
                0.70,
                "SUGGEST",
                "[]",
                created_at.isoformat(),
                1,
            ),
        )

    # No outbound messages to this contact

    stats = await tracker.run_inference_cycle()

    assert stats["marked_accurate"] == 0
    assert stats["marked_inaccurate"] == 1

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert row["was_accurate"] == 0


@pytest.mark.asyncio
async def test_still_within_window_returns_none(db):
    """If less than 48 hours have passed and no reply, tracker should not resolve yet."""
    tracker = BehavioralAccuracyTracker(db)

    prediction_id = "pred-004"
    # Created 30 hours ago (still within the 48-hour window)
    created_at = datetime.now(timezone.utc) - timedelta(hours=30)
    contact_email = "carol@example.com"

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                f'Unreplied message from {contact_email}: "Meeting notes" (30 hours ago)',
                0.65,
                "SUGGEST",
                "[]",
                created_at.isoformat(),
                1,
            ),
        )

    # No outbound messages yet

    stats = await tracker.run_inference_cycle()

    # Should not resolve either way yet
    assert stats["marked_accurate"] == 0
    assert stats["marked_inaccurate"] == 0

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert row["was_accurate"] is None
        assert row["resolved_at"] is None


@pytest.mark.asyncio
async def test_case_insensitive_email_matching(db):
    """Email matching should be case-insensitive."""
    tracker = BehavioralAccuracyTracker(db)

    prediction_id = "pred-005"
    created_at = datetime.now(timezone.utc) - timedelta(hours=10)
    contact_email = "David@Example.COM"  # Mixed case

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                f'Unreplied message from {contact_email}: "Feedback request" (10 hours ago)',
                0.72,
                "SUGGEST",
                "[]",
                created_at.isoformat(),
                1,
            ),
        )

    # User replies with lowercase email
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, payload, source, timestamp, priority)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "event-005",
                "email.sent",
                json.dumps({
                    "to": "david@example.com",  # All lowercase
                    "subject": "Re: Feedback request",
                }),
                "proton_mail",
                (created_at + timedelta(hours=6)).isoformat(),
                "normal",
            ),
        )

    stats = await tracker.run_inference_cycle()

    assert stats["marked_accurate"] == 1


@pytest.mark.asyncio
async def test_partial_email_match_in_to_field(db):
    """Should match if contact email appears anywhere in the 'to' field (handles CC/BCC)."""
    tracker = BehavioralAccuracyTracker(db)

    prediction_id = "pred-006"
    created_at = datetime.now(timezone.utc) - timedelta(hours=12)
    contact_email = "emma@example.com"

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                f'Unreplied message from {contact_email}: "Team sync" (12 hours ago)',
                0.77,
                "SUGGEST",
                "[]",
                created_at.isoformat(),
                1,
            ),
        )

    # User sends email to multiple recipients including the contact
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, payload, source, timestamp, priority)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "event-006",
                "email.sent",
                json.dumps({
                    "to": f"team@example.com, {contact_email}, boss@example.com",
                    "subject": "Re: Team sync",
                }),
                "proton_mail",
                (created_at + timedelta(hours=4)).isoformat(),
                "normal",
            ),
        )

    stats = await tracker.run_inference_cycle()

    assert stats["marked_accurate"] == 1


@pytest.mark.asyncio
async def test_message_sent_event_also_counts_as_reply(db):
    """Should match both email.sent and message.sent events."""
    tracker = BehavioralAccuracyTracker(db)

    prediction_id = "pred-007"
    created_at = datetime.now(timezone.utc) - timedelta(hours=15)
    contact_email = "frank@example.com"

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                f'Unreplied message from {contact_email}: "Weekend plans?" (15 hours ago)',
                0.68,
                "SUGGEST",
                "[]",
                created_at.isoformat(),
                1,
            ),
        )

    # User replies via messaging app instead of email
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, payload, source, timestamp, priority)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "event-007",
                "message.sent",
                json.dumps({
                    "to": contact_email,
                    "text": "Free on Saturday!",
                }),
                "signal",
                (created_at + timedelta(hours=8)).isoformat(),
                "normal",
            ),
        )

    stats = await tracker.run_inference_cycle()

    assert stats["marked_accurate"] == 1


@pytest.mark.asyncio
async def test_only_resolves_surfaced_predictions(db):
    """Should only process predictions that were surfaced (was_surfaced=1)."""
    tracker = BehavioralAccuracyTracker(db)

    created_at = datetime.now(timezone.utc) - timedelta(hours=10)

    # Create two predictions: one surfaced, one not
    with db.get_connection("user_model") as conn:
        # Surfaced prediction
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-surfaced",
                "reminder",
                'Unreplied message from grace@example.com: "Test" (10 hours ago)',
                0.75,
                "SUGGEST",
                "[]",
                created_at.isoformat(),
                1,  # Surfaced
            ),
        )

        # Not surfaced prediction (filtered by confidence gate)
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-filtered",
                "reminder",
                'Unreplied message from harry@example.com: "Test" (10 hours ago)',
                0.25,
                "OBSERVE",
                "[]",
                created_at.isoformat(),
                0,  # Not surfaced
            ),
        )

    # Create replies for BOTH (but only the surfaced one should be processed)
    with db.get_connection("events") as conn:
        for email in ["grace@example.com", "harry@example.com"]:
            conn.execute(
                """INSERT INTO events (id, type, payload, source, timestamp, priority)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"event-{email}",
                    "email.sent",
                    json.dumps({"to": email, "subject": "Re: Test"}),
                    "proton_mail",
                    (created_at + timedelta(hours=5)).isoformat(),
                    "normal",
                ),
            )

    stats = await tracker.run_inference_cycle()

    # Only the surfaced prediction should be resolved
    assert stats["marked_accurate"] == 1

    with db.get_connection("user_model") as conn:
        surfaced = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            ("pred-surfaced",),
        ).fetchone()
        filtered = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            ("pred-filtered",),
        ).fetchone()

        assert surfaced["was_accurate"] == 1
        assert filtered["was_accurate"] is None  # Not processed


@pytest.mark.asyncio
async def test_doesnt_double_resolve_already_resolved(db):
    """Should skip predictions that are already resolved."""
    tracker = BehavioralAccuracyTracker(db)

    prediction_id = "pred-008"
    created_at = datetime.now(timezone.utc) - timedelta(hours=10)
    already_resolved_at = (created_at + timedelta(hours=2)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced, was_accurate, resolved_at, user_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                'Unreplied message from iris@example.com: "Test" (10 hours ago)',
                0.75,
                "SUGGEST",
                "[]",
                created_at.isoformat(),
                1,
                1,  # Already marked accurate
                already_resolved_at,
                "explicit",  # User clicked "Act On" button
            ),
        )

    stats = await tracker.run_inference_cycle()

    # Should not re-process
    assert stats["marked_accurate"] == 0
    assert stats["marked_inaccurate"] == 0


@pytest.mark.asyncio
async def test_future_name_based_pattern_compatibility(db):
    """Verify the tracker still supports future name-based descriptions."""
    tracker = BehavioralAccuracyTracker(db)

    prediction_id = "pred-009"
    created_at = datetime.now(timezone.utc) - timedelta(hours=10)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                "Follow up with Alice about the project deadline",  # Name-based format
                0.80,
                "DEFAULT",
                json.dumps({
                    "contact_email": "alice@company.com",
                    "contact_name": "Alice",
                }),
                created_at.isoformat(),
                1,
            ),
        )

    # User sends email to Alice
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, payload, source, timestamp, priority)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "event-009",
                "email.sent",
                json.dumps({
                    "to": "alice@company.com",
                    "subject": "Re: Project deadline",
                }),
                "proton_mail",
                (created_at + timedelta(hours=6)).isoformat(),
                "normal",
            ),
        )

    stats = await tracker.run_inference_cycle()

    assert stats["marked_accurate"] == 1


@pytest.mark.asyncio
async def test_batch_resolution_of_multiple_predictions(db):
    """Should process multiple predictions in a single run."""
    tracker = BehavioralAccuracyTracker(db)

    base_time = datetime.now(timezone.utc) - timedelta(hours=20)

    # Create 5 predictions at different ages
    predictions = [
        ("pred-batch-1", "alice@test.com", 60, True),    # 60h old, no reply → inaccurate
        ("pred-batch-2", "bob@test.com", 50, True),      # 50h old, no reply → inaccurate
        ("pred-batch-3", "carol@test.com", 15, True),    # 15h old, has reply → accurate
        ("pred-batch-4", "david@test.com", 10, True),    # 10h old, has reply → accurate
        ("pred-batch-5", "emma@test.com", 5, False),     # 5h old, no reply → still pending
    ]

    with db.get_connection("user_model") as conn:
        for pred_id, email, hours_ago, _ in predictions:
            created = base_time - timedelta(hours=hours_ago)
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    supporting_signals, created_at, was_surfaced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pred_id,
                    "reminder",
                    f'Unreplied message from {email}: "Test" ({hours_ago} hours ago)',
                    0.75,
                    "SUGGEST",
                    "[]",
                    created.isoformat(),
                    1,
                ),
            )

    # Create replies for carol and david only
    with db.get_connection("events") as conn:
        for email in ["carol@test.com", "david@test.com"]:
            pred_created = base_time - timedelta(hours=15 if email == "carol@test.com" else 10)
            reply_time = pred_created + timedelta(hours=5)
            conn.execute(
                """INSERT INTO events (id, type, payload, source, timestamp, priority)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"event-{email}",
                    "email.sent",
                    json.dumps({"to": email, "subject": "Re: Test"}),
                    "proton_mail",
                    reply_time.isoformat(),
                    "normal",
                ),
            )

    stats = await tracker.run_inference_cycle()

    # Should resolve 4 predictions: 2 accurate (carol, david), 2 inaccurate (alice, bob)
    # Emma's prediction is still within the window
    assert stats["marked_accurate"] == 2
    assert stats["marked_inaccurate"] == 2

    # Verify individual statuses
    with db.get_connection("user_model") as conn:
        results = {}
        for pred_id, _, _, _ in predictions:
            row = conn.execute(
                "SELECT was_accurate FROM predictions WHERE id = ?",
                (pred_id,),
            ).fetchone()
            results[pred_id] = row["was_accurate"]

    assert results["pred-batch-1"] == 0  # Inaccurate (no reply, > 48h)
    assert results["pred-batch-2"] == 0  # Inaccurate (no reply, > 48h)
    assert results["pred-batch-3"] == 1  # Accurate (has reply)
    assert results["pred-batch-4"] == 1  # Accurate (has reply)
    assert results["pred-batch-5"] is None  # Still pending (< 48h, no reply)
