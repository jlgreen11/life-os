"""
Tests for automated-sender fast-path and no-contact timeout in reminder prediction
accuracy inference.

Problem (pre-fix):
    Reminder predictions for automated/marketing senders (e.g., Fidelity, Instagram
    security, HBO Max) were generated before the marketing filter was fully robust.
    These predictions are structurally impossible to fulfill — the user will never
    "reply" to a no-reply mailer by definition. However, _infer_reminder_accuracy()
    had no fast-path for automated senders, causing them to wait the full 48-hour
    window before being marked inaccurate.

    Additionally, 118 reminder predictions had no extractable contact info at all.
    The method returned None indefinitely for these, leaving them permanently
    unresolved and cluttering the unresolved count.

    Combined, this left 218+ predictions in an permanently-unresolved limbo,
    distorting the reminder accuracy stats and slowing the learning loop.

Fix (this iteration):
    1. Automated-sender fast-path: After extracting the contact email, check
       _is_automated_sender(). If matched, return False immediately without
       waiting 48 hours. Mirrors the fast-path added to _infer_opportunity_accuracy()
       in iteration 173 (PR #189).

    2. Empty-contact timeout: When no contact info is found in signals or
       description AND 48+ hours have passed, return False instead of None.
       This ensures stale unresolvable predictions are cleaned up within 48h
       instead of persisting indefinitely.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


# ============================================================================
# _infer_reminder_accuracy(): automated-sender fast-path tests
# ============================================================================


@pytest.mark.asyncio
async def test_reminder_automated_sender_immediately_inaccurate_via_signals(db):
    """Reminder predictions for automated senders are marked INACCURATE immediately.

    The automated-sender fast-path should fire for emails in supporting_signals,
    even when the prediction is only 2 hours old (well within the 48-hour window).
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    # Only 2 hours old — normally still within the 48-hour window
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                'Unreplied message from Fidelity.Investments@mail.fidelity.com: '
                '"Your trade confirmation" (5 hours ago)',
                0.55,
                "suggest",
                "24_hours",
                "Reply to Fidelity.Investments",
                json.dumps({
                    "contact_email": "Fidelity.Investments@mail.fidelity.com",
                    "contact_name": "Fidelity.Investments",
                }),
                1,
                created_at.isoformat(),
            ),
        )

    stats = await tracker.run_inference_cycle()

    # Should be resolved in this cycle — automated sender, no waiting
    assert stats["marked_inaccurate"] >= 1, (
        "Expected automated-sender reminder to be marked inaccurate immediately"
    )

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None
    assert row["was_accurate"] == 0, "Automated sender reminder should be INACCURATE"
    assert row["resolved_at"] is not None, "Prediction should be resolved"


@pytest.mark.asyncio
async def test_reminder_automated_sender_via_description_extraction(db):
    """Automated sender fast-path works even when email is extracted from description.

    For old predictions that stored only empty signals, the email is parsed
    from the description. The fast-path must still trigger for these.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=3)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                'Unreplied message from noreply@instagram.com: '
                '"Your account security update" (8 hours ago)',
                0.45,
                "suggest",
                "24_hours",
                "Reply to instagram",
                "{}",  # No contact in signals — must parse from description
                1,
                created_at.isoformat(),
            ),
        )

    stats = await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["was_accurate"] == 0, (
        "Automated sender parsed from description should be immediately INACCURATE"
    )


@pytest.mark.asyncio
async def test_reminder_fidelity_mail_subdomain_detected(db):
    """Fidelity's marketing subdomain (mail.fidelity.com) is detected as automated.

    This is the most common automated sender in the live prediction backlog.
    The domain uses a marketing subdomain pattern (email.*) which must be caught.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=1)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                'Unreplied message from benefitscenter@mail.fidelity.com: '
                '"Name your beneficiary today" (1 hour ago)',
                0.50,
                "suggest",
                "24_hours",
                "Reply to benefitscenter",
                json.dumps({
                    "contact_email": "benefitscenter@mail.fidelity.com",
                    "contact_name": "benefitscenter",
                }),
                1,
                created_at.isoformat(),
            ),
        )

    stats = await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["was_accurate"] == 0, (
        "Fidelity mail subdomain should be detected as automated and marked INACCURATE"
    )


@pytest.mark.asyncio
async def test_reminder_real_human_waits_for_window(db):
    """Reminder predictions for real humans still wait for the 48-hour window.

    The automated-sender fast-path must NOT affect human contact predictions.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    # Created 2 hours ago — within the 48-hour window, no reply made yet
    created_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                'Unreplied message from alice@gmail.com: "Hey, are you free this weekend?" (4 hours ago)',
                0.65,
                "suggest",
                "24_hours",
                "Reply to Alice",
                json.dumps({
                    "contact_email": "alice@gmail.com",
                    "contact_name": "Alice",
                }),
                1,
                created_at.isoformat(),
            ),
        )

    stats = await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    # Should NOT be resolved yet — still within the 48-hour window, no reply sent
    assert row["was_accurate"] is None, (
        "Human contact reminder should not be resolved within the 48-hour window"
    )
    assert row["resolved_at"] is None, (
        "Human contact reminder should remain unresolved during the window"
    )


@pytest.mark.asyncio
async def test_reminder_real_human_accurate_after_reply(db):
    """Reminder predictions for real humans are ACCURATE when user sends a reply."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    event_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=6)
    reply_at = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                'Unreplied message from bob@company.com: "Project update needed" (3 hours ago)',
                0.70,
                "suggest",
                "24_hours",
                "Reply to Bob",
                json.dumps({
                    "contact_email": "bob@company.com",
                    "contact_name": "Bob",
                }),
                1,
                created_at.isoformat(),
            ),
        )

    # Simulate user sending an email reply to Bob
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                "email.sent",
                "proton_mail",
                reply_at.isoformat(),
                "normal",
                json.dumps({
                    "to_addresses": ["bob@company.com"],
                    "subject": "Re: Project update needed",
                }),
                "{}",
            ),
        )

    stats = await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["was_accurate"] == 1, "Reminder should be ACCURATE after user sends reply"
    assert row["resolved_at"] is not None


@pytest.mark.asyncio
async def test_reminder_multiple_automated_senders_bulk_resolved(db):
    """Multiple automated-sender reminder predictions are resolved in one cycle.

    This mirrors the production backlog: many Fidelity/Instagram/HBO predictions
    that slipped through before the marketing filter was tightened.
    """
    tracker = BehavioralAccuracyTracker(db)

    # Typical automated senders from the live backlog
    cases = [
        "Fidelity.Investments@mail.fidelity.com",
        "security@mail.instagram.com",
        "HBOMax@mail.hbomax.com",
        "wholefoodsmarket@mail.wholefoodsmarket.com",
        "noreply@notifications.service.com",
    ]

    pred_ids = []
    created_at = datetime.now(timezone.utc) - timedelta(hours=3)

    with db.get_connection("user_model") as conn:
        for email in cases:
            pred_id = str(uuid.uuid4())
            pred_ids.append(pred_id)
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    time_horizon, suggested_action, supporting_signals,
                    was_surfaced, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pred_id,
                    "reminder",
                    f'Unreplied message from {email}: "automated message" (2 hours ago)',
                    0.50,
                    "suggest",
                    "24_hours",
                    f"Reply to {email.split('@')[0]}",
                    json.dumps({"contact_email": email, "contact_name": email.split("@")[0]}),
                    1,
                    created_at.isoformat(),
                ),
            )

    stats = await tracker.run_inference_cycle()

    assert stats["marked_inaccurate"] >= len(cases), (
        f"Expected all {len(cases)} automated-sender reminders to be resolved, "
        f"got {stats['marked_inaccurate']}"
    )

    with db.get_connection("user_model") as conn:
        for pred_id in pred_ids:
            row = conn.execute(
                "SELECT was_accurate FROM predictions WHERE id = ?",
                (pred_id,),
            ).fetchone()
            assert row["was_accurate"] == 0, (
                f"Prediction {pred_id} should be marked INACCURATE"
            )


# ============================================================================
# _infer_reminder_accuracy(): no-contact timeout tests
# ============================================================================


@pytest.mark.asyncio
async def test_reminder_no_contact_info_resolves_after_48h(db):
    """Reminder predictions with no contact info resolve as INACCURATE after 48 hours.

    Previously, predictions with empty supporting_signals AND no extractable email
    in the description returned None indefinitely, never resolving. After 48h, they
    should be marked inaccurate instead of staying permanently unresolved.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    # 50 hours old — well past the 48-hour no-contact timeout
    created_at = datetime.now(timezone.utc) - timedelta(hours=50)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "You have an unreplied message",  # No email or name extractable
                0.40,
                "suggest",
                "24_hours",
                "Reply to the message",
                "{}",  # Empty signals — no contact info
                1,
                created_at.isoformat(),
            ),
        )

    stats = await tracker.run_inference_cycle()

    # Should be marked inaccurate due to the 48-hour no-contact timeout
    assert stats["marked_inaccurate"] >= 1, (
        "Expected no-contact prediction older than 48h to be marked inaccurate"
    )

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["was_accurate"] == 0, (
        "No-contact reminder older than 48h should be marked INACCURATE"
    )
    assert row["resolved_at"] is not None, "Prediction should be resolved"


@pytest.mark.asyncio
async def test_reminder_no_contact_info_waits_within_48h(db):
    """Reminder predictions with no contact info still wait within the 48-hour window.

    A prediction with no contact info that is only 10 hours old should not be
    resolved yet — it might still get a reply before the window closes.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    # Only 10 hours old — within the 48-hour no-contact timeout
    created_at = datetime.now(timezone.utc) - timedelta(hours=10)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "You have an unreplied message",  # No email or name extractable
                0.40,
                "suggest",
                "24_hours",
                "Reply to the message",
                "{}",
                1,
                created_at.isoformat(),
            ),
        )

    await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    # Should NOT be resolved yet — within the 48-hour window
    assert row["was_accurate"] is None, (
        "No-contact reminder within 48-hour window should remain unresolved"
    )
    assert row["resolved_at"] is None


@pytest.mark.asyncio
async def test_reminder_no_contact_info_list_format_timeout(db):
    """Empty-list supporting_signals format also triggers no-contact timeout.

    Some older predictions stored supporting_signals as '[]' (empty list JSON)
    instead of '{}' (empty dict). The backward-compatibility handling should
    treat these the same as empty-dict format for the timeout logic.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    # 60 hours old — past the 48-hour timeout
    created_at = datetime.now(timezone.utc) - timedelta(hours=60)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Unreplied message (subject unknown)",
                0.45,
                "suggest",
                "24_hours",
                "Reply",
                "[]",  # Old empty-list format
                1,
                created_at.isoformat(),
            ),
        )

    stats = await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["was_accurate"] == 0, (
        "Empty-list format no-contact reminder older than 48h should be INACCURATE"
    )


# ============================================================================
# Regression: existing passing behavior is unchanged
# ============================================================================


@pytest.mark.asyncio
async def test_reminder_inaccurate_after_48h_with_human_contact(db):
    """Reminder predictions for human contacts are INACCURATE after 48h with no reply.

    This verifies the pre-existing inaccuracy timeout still works correctly
    after the new fast-path and timeout changes were added.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    # 50 hours old — past the 48-hour window, no reply sent
    created_at = datetime.now(timezone.utc) - timedelta(hours=50)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                'Unreplied message from charlie@company.com: "Follow up needed" (52 hours ago)',
                0.60,
                "suggest",
                "24_hours",
                "Reply to Charlie",
                json.dumps({
                    "contact_email": "charlie@company.com",
                    "contact_name": "Charlie",
                }),
                1,
                created_at.isoformat(),
            ),
        )

    # No email.sent event → user did not reply

    stats = await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["was_accurate"] == 0, (
        "Human contact reminder with no reply after 48h should be INACCURATE"
    )
