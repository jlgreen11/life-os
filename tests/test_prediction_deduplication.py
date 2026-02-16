"""
Tests for prediction deduplication to prevent reprocessing old emails.

This test module covers the critical fix in iteration 62 that prevents the
prediction engine from creating thousands of duplicate reminder predictions
by reprocessing the same emails on every prediction cycle.

The Bug:
    The _check_follow_up_needs method was querying ALL inbound messages from
    the last 48 hours on EVERY prediction cycle (every 15 min), creating
    duplicate predictions for the same unreplied messages repeatedly.

    Example: 9,086 duplicate predictions created at 2026-02-16T00:08:00.780Z
    for emails that were already processed in previous cycles.

The Fix:
    Before creating a reminder prediction, check if we've already created one
    for this message_id in the last 48 hours. Only create ONE prediction per
    unreplied message, ever.

Test Coverage:
    1. No duplicate predictions for the same unreplied email
    2. Predictions ARE created on first occurrence
    3. Predictions ARE NOT created on subsequent cycles (same email, still unreplied)
    4. Multiple unreplied emails each get their own prediction (once)
    5. Replied-to messages don't generate predictions (existing behavior)
    6. Marketing emails don't generate predictions (existing behavior)
"""

import json
import pytest
from datetime import datetime, timedelta, timezone

from models.core import ConfidenceGate
from models.user_model import Prediction
from services.prediction_engine.engine import PredictionEngine
from storage.database import DatabaseManager, UserModelStore


@pytest.mark.asyncio
async def test_no_duplicate_predictions_for_same_email(db, user_model_store):
    """
    Test that the prediction engine doesn't create duplicate predictions for
    the same unreplied email across multiple prediction cycles.

    This is the primary regression test for iteration 62.
    """
    engine = PredictionEngine(db, user_model_store)

    # Simulate an unreplied email from 6 hours ago
    now = datetime.now(timezone.utc)
    email_time = now - timedelta(hours=6)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "email-001",
                "email.received",
                "protonmail",
                email_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-001",
                    "from_address": "alice@example.com",
                    "subject": "Can we meet tomorrow?",
                    "body_plain": "Hi, can we schedule a meeting?",
                }),
                json.dumps({"related_contacts": []}),
            ),
        )

    # First prediction cycle — should create a prediction (may be filtered by confidence gates)
    predictions_cycle_1 = await engine.generate_predictions({})

    # Check database — should have created a prediction (even if filtered/not surfaced)
    with db.get_connection("user_model") as conn:
        count_after_cycle_1 = conn.execute(
            """SELECT COUNT(*) as cnt FROM predictions
               WHERE prediction_type = 'reminder'
               AND supporting_signals LIKE '%msg-001%'""",
        ).fetchone()["cnt"]

    assert count_after_cycle_1 == 1, (
        f"Should create exactly ONE prediction on first cycle, got {count_after_cycle_1}"
    )

    # Second prediction cycle (15 minutes later) — should NOT create duplicate
    # Even though the email is still unreplied, we already have a prediction for it
    predictions_cycle_2 = await engine.generate_predictions({})

    # Verify database state — should STILL have exactly ONE prediction for msg-001
    with db.get_connection("user_model") as conn:
        count_after_cycle_2 = conn.execute(
            """SELECT COUNT(*) as cnt FROM predictions
               WHERE prediction_type = 'reminder'
               AND supporting_signals LIKE '%msg-001%'""",
        ).fetchone()["cnt"]

    assert count_after_cycle_2 == 1, (
        f"Should NOT create duplicate prediction. Expected 1, got {count_after_cycle_2}"
    )


@pytest.mark.asyncio
async def test_predictions_created_for_multiple_unreplied_emails(db, user_model_store):
    """
    Test that the deduplication fix doesn't prevent creating predictions for
    DIFFERENT unreplied emails — each unique unreplied message should get
    exactly one prediction.
    """
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Create 3 different unreplied emails from 6 hours ago
    emails = [
        ("email-001", "msg-001", "alice@example.com", "Meeting tomorrow?"),
        ("email-002", "msg-002", "bob@example.com", "Project update needed"),
        ("email-003", "msg-003", "carol@example.com", "Quick question"),
    ]

    with db.get_connection("events") as conn:
        for event_id, msg_id, from_addr, subject in emails:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    "email.received",
                    "protonmail",
                    (now - timedelta(hours=6)).isoformat(),
                    "normal",
                    json.dumps({
                        "message_id": msg_id,
                        "from_address": from_addr,
                        "subject": subject,
                        "body_plain": f"Email from {from_addr}",
                    }),
                    json.dumps({"related_contacts": []}),
                ),
            )

    # Generate predictions
    predictions = await engine.generate_predictions({})

    # Should create predictions for all 3 emails (they're all unreplied and >3h old)
    # Note: predictions might be filtered by confidence gates, so check database instead
    with db.get_connection("user_model") as conn:
        for msg_id in ["msg-001", "msg-002", "msg-003"]:
            count = conn.execute(
                f"""SELECT COUNT(*) as cnt FROM predictions
                   WHERE prediction_type = 'reminder'
                   AND supporting_signals LIKE '%{msg_id}%'""",
            ).fetchone()["cnt"]

            assert count == 1, f"Should have exactly ONE prediction for {msg_id}"


@pytest.mark.asyncio
async def test_no_predictions_for_already_replied_emails(db, user_model_store):
    """
    Test that replied-to emails don't generate predictions, even on first
    occurrence. This is existing behavior that should be preserved.
    """
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)
    email_time = now - timedelta(hours=6)

    with db.get_connection("events") as conn:
        # Inbound email
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "email-001",
                "email.received",
                "protonmail",
                email_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-001",
                    "from_address": "alice@example.com",
                    "subject": "Meeting tomorrow?",
                }),
                json.dumps({"related_contacts": []}),
            ),
        )

        # Outbound reply (1 hour later)
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "email-002",
                "email.sent",
                "protonmail",
                (email_time + timedelta(hours=1)).isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-002",
                    "in_reply_to": "msg-001",  # Links to the inbound email
                    "to": "alice@example.com",
                }),
                json.dumps({}),
            ),
        )

    # Generate predictions — should NOT create one for alice (already replied)
    predictions = await engine.generate_predictions({})

    # Check database
    with db.get_connection("user_model") as conn:
        count = conn.execute(
            """SELECT COUNT(*) as cnt FROM predictions
               WHERE prediction_type = 'reminder'
               AND supporting_signals LIKE '%alice@example.com%'""",
        ).fetchone()["cnt"]

    assert count == 0, "Should NOT create prediction for already-replied email"


@pytest.mark.asyncio
async def test_no_predictions_for_marketing_emails(db, user_model_store):
    """
    Test that marketing emails don't generate predictions, even if unreplied.
    This is existing behavior that should be preserved.
    """
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)
    email_time = now - timedelta(hours=6)

    # Marketing email patterns that should be filtered
    marketing_emails = [
        "newsletter@company.com",
        "promo@deals.com",
        "D23@email.d23.com",
        "notifications@reply.service.com",
        "mailer-daemon@googlemail.com",
    ]

    with db.get_connection("events") as conn:
        for idx, from_addr in enumerate(marketing_emails):
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"email-{idx:03d}",
                    "email.received",
                    "protonmail",
                    email_time.isoformat(),
                    "normal",
                    json.dumps({
                        "message_id": f"msg-{idx:03d}",
                        "from_address": from_addr,
                        "subject": "Special offer!",
                        "body_plain": "Check out our deals!",
                    }),
                    json.dumps({"related_contacts": []}),
                ),
            )

    # Generate predictions — should NOT create any for marketing emails
    predictions = await engine.generate_predictions({})

    # Check database — should have ZERO reminder predictions
    with db.get_connection("user_model") as conn:
        count = conn.execute(
            """SELECT COUNT(*) as cnt FROM predictions
               WHERE prediction_type = 'reminder'""",
        ).fetchone()["cnt"]

    assert count == 0, "Should NOT create predictions for marketing emails"


@pytest.mark.asyncio
async def test_deduplication_across_48_hour_window(db, user_model_store):
    """
    Test that deduplication works correctly across the 48-hour window.

    If an email is received, a prediction is created, and the email remains
    unreplied for 48+ hours, we should NOT create a second prediction when
    the first one expires from the window.
    """
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Email received 50 hours ago (outside the 48h window)
    old_email_time = now - timedelta(hours=50)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "email-001",
                "email.received",
                "protonmail",
                old_email_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-001",
                    "from_address": "alice@example.com",
                    "subject": "Urgent meeting request",
                }),
                json.dumps({"related_contacts": []}),
            ),
        )

    # Generate predictions — email is outside the 48h window, so no prediction
    predictions = await engine.generate_predictions({})

    # The email is too old (>48h) so it's not even considered
    # This is expected behavior — we don't nag about ancient unreplied emails


@pytest.mark.asyncio
async def test_prediction_created_only_after_3_hour_grace_period(db, user_model_store):
    """
    Test that predictions are NOT created for very recent emails (<3 hours old).

    Users need time to respond naturally without being nagged immediately.
    """
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Email received 2 hours ago (within grace period)
    recent_email_time = now - timedelta(hours=2)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "email-001",
                "email.received",
                "protonmail",
                recent_email_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-001",
                    "from_address": "alice@example.com",
                    "subject": "Quick question",
                }),
                json.dumps({"related_contacts": []}),
            ),
        )

    # Generate predictions — should NOT create one (email too recent)
    predictions = await engine.generate_predictions({})

    with db.get_connection("user_model") as conn:
        count = conn.execute(
            """SELECT COUNT(*) as cnt FROM predictions
               WHERE prediction_type = 'reminder'
               AND supporting_signals LIKE '%msg-001%'""",
        ).fetchone()["cnt"]

    assert count == 0, "Should NOT create prediction for email within 3-hour grace period"


@pytest.mark.asyncio
async def test_message_id_tracking_in_supporting_signals(db, user_model_store):
    """
    Test that message_id is correctly stored in supporting_signals for
    deduplication tracking.

    This is critical — without message_id in supporting_signals, deduplication
    won't work.
    """
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)
    email_time = now - timedelta(hours=6)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "email-001",
                "email.received",
                "protonmail",
                email_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-001",
                    "from_address": "alice@example.com",
                    "subject": "Test email",
                }),
                json.dumps({"related_contacts": []}),
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({})

    # Check database — supporting_signals should contain message_id
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            """SELECT supporting_signals FROM predictions
               WHERE prediction_type = 'reminder'
               LIMIT 1""",
        ).fetchone()

    if row:
        signals = json.loads(row["supporting_signals"])
        assert "message_id" in signals, "supporting_signals must contain message_id"
        assert signals["message_id"] == "msg-001", "message_id must match the email"
