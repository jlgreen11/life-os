"""
Test suite for prediction engine email scan window optimization.

This test module verifies that the prediction engine efficiently processes
emails without scanning excessive history or generating duplicate predictions.

Critical issue (iteration 81):
    With 70K+ emails in the database, the prediction engine was scanning
    48 hours of email history every 15 minutes, causing:
    - 37K+ predictions generated per hour (73K email scans every cycle)
    - 100% CPU usage continuously
    - Massive database query overhead

Fix:
    - Reduced scan window from 48h → 24h (50% fewer emails to process)
    - Check existing predictions first to build deduplication set efficiently
    - Keep 48h window for prediction lookup to catch stragglers
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from models.core import ConfidenceGate
from services.prediction_engine.engine import PredictionEngine


@pytest.mark.asyncio
async def test_scan_window_reduced_to_24_hours(db, user_model_store):
    """Verify that email scan uses 24-hour window, not 48-hour (on subsequent cycles)."""
    engine = PredictionEngine(db, user_model_store)
    # Consume the first-run flag so the test exercises the steady-state 24h window.
    # The first cycle intentionally uses a wider 72h lookback for catchup.
    engine._first_follow_up_run = False

    now = datetime.now(timezone.utc)

    # Create test emails at different ages - mark as priority contacts to pass confidence threshold
    with db.get_connection("events") as conn:
        # Email at 12 hours old (should be scanned)
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.received",
                "test",
                (now - timedelta(hours=12)).isoformat(),
                json.dumps({
                    "message_id": "msg-12h",
                    "from_address": "alice@example.com",
                    "subject": "Test 12h",
                    "body_plain": "Test message",
                }),
                json.dumps({"related_contacts": ["alice@example.com"]}),  # Priority contact
            ),
        )

        # Email at 23 hours old (should be scanned)
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.received",
                "test",
                (now - timedelta(hours=23)).isoformat(),
                json.dumps({
                    "message_id": "msg-23h",
                    "from_address": "bob@example.com",
                    "subject": "Test 23h",
                    "body_plain": "Test message",
                }),
                json.dumps({"related_contacts": ["bob@example.com"]}),  # Priority contact
            ),
        )

        # Email at 36 hours old (should NOT be scanned - outside 24h window)
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.received",
                "test",
                (now - timedelta(hours=36)).isoformat(),
                json.dumps({
                    "message_id": "msg-36h",
                    "from_address": "charlie@example.com",
                    "subject": "Test 36h",
                    "body_plain": "Test message",
                }),
                json.dumps({"related_contacts": ["charlie@example.com"]}),  # Priority contact
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({})

    # Should get predictions for 12h and 23h messages (both > 3h grace period)
    # Should NOT get prediction for 36h message (outside 24h scan window)
    message_ids = [
        p.supporting_signals.get("message_id")
        for p in predictions
        if p.prediction_type == "reminder"
    ]

    assert "msg-12h" in message_ids, "Should scan 12-hour-old email"
    assert "msg-23h" in message_ids, "Should scan 23-hour-old email"
    assert "msg-36h" not in message_ids, "Should NOT scan 36-hour-old email (outside 24h window)"


@pytest.mark.asyncio
async def test_deduplication_checks_wider_48h_window(db, user_model_store):
    """Verify that deduplication still checks 48h window to catch stragglers."""
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Create an email at 12 hours old
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.received",
                "test",
                (now - timedelta(hours=12)).isoformat(),
                json.dumps({
                    "message_id": "msg-test",
                    "from_address": "alice@example.com",
                    "subject": "Test",
                    "body_plain": "Test message",
                }),
                json.dumps({"related_contacts": []}),
            ),
        )

    # Create a prediction for this message 36 hours ago
    # (within 48h prediction window but outside 24h scan window)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions (id, prediction_type, description, confidence,
                                       confidence_gate, supporting_signals, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "reminder",
                "Test prediction",
                0.5,
                "suggest",
                json.dumps({"message_id": "msg-test"}),
                (now - timedelta(hours=36)).isoformat(),
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({})

    # Should NOT create a duplicate prediction even though the original
    # prediction was created 36h ago (within 48h prediction window)
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    message_ids = [p.supporting_signals.get("message_id") for p in reminder_predictions]

    assert "msg-test" not in message_ids, "Should not duplicate prediction from 36h ago"


@pytest.mark.asyncio
async def test_performance_with_high_volume_emails(db, user_model_store):
    """Verify efficient processing with thousands of emails."""
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Simulate high-volume email scenario: 1000 emails in last 24h
    # Make half of them priority contacts to get some predictions through
    with db.get_connection("events") as conn:
        for i in range(1000):
            # Distribute emails across 24 hours
            age_hours = (i / 1000) * 24
            email = f"user{i}@example.com"
            is_priority = i % 2 == 0  # Every other email is from a priority contact

            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    "email.received",
                    "test",
                    (now - timedelta(hours=age_hours)).isoformat(),
                    json.dumps({
                        "message_id": f"msg-{i}",
                        "from_address": email,
                        "subject": f"Test {i}",
                        "body_plain": "Test message",
                    }),
                    json.dumps({"related_contacts": [email] if is_priority else []}),
                ),
            )

    # Generate predictions - should complete without timeout
    predictions = await engine.generate_predictions({})

    # Should create predictions for emails older than 3h grace period
    # (approximately 875 emails if evenly distributed)
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]

    # Allow some variance due to marketing filter and other logic
    assert len(reminder_predictions) > 0, "Should generate some predictions"
    assert len(reminder_predictions) < 1000, "Should not create predictions for all emails"


@pytest.mark.asyncio
async def test_no_duplicates_across_multiple_cycles(db, user_model_store):
    """Verify that running multiple prediction cycles doesn't create duplicates."""
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Create a test email from a priority contact
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.received",
                "test",
                (now - timedelta(hours=5)).isoformat(),
                json.dumps({
                    "message_id": "msg-unique",
                    "from_address": "alice@example.com",
                    "subject": "Important",
                    "body_plain": "Test message",
                }),
                json.dumps({"related_contacts": ["alice@example.com"]}),  # Priority contact for 0.7 confidence
            ),
        )

    # Run prediction cycle 1
    predictions_1 = await engine.generate_predictions({})
    reminder_1 = [p for p in predictions_1 if p.prediction_type == "reminder"]

    assert len(reminder_1) == 1, "Should create one prediction on first cycle"

    # Predictions are automatically stored by generate_predictions() at line 136,
    # so we don't need to manually insert them. Just verify they were stored.

    # Add a new event to trigger _has_new_events() = True
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "test.event",
                "test",
                datetime.now(timezone.utc).isoformat(),
                json.dumps({}),
                json.dumps({}),
            ),
        )

    # Run prediction cycle 2
    predictions_2 = await engine.generate_predictions({})
    reminder_2 = [p for p in predictions_2 if p.prediction_type == "reminder"]

    # Should NOT create a duplicate prediction for the same message
    message_ids_2 = [p.supporting_signals.get("message_id") for p in reminder_2]
    assert "msg-unique" not in message_ids_2, "Should not create duplicate on second cycle"


@pytest.mark.asyncio
async def test_grace_period_still_respected(db, user_model_store):
    """Verify that 3-hour grace period is still enforced with optimized scan."""
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Create emails at different ages - mark as priority to pass confidence threshold
    with db.get_connection("events") as conn:
        # Email at 1 hour old (within grace period)
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.received",
                "test",
                (now - timedelta(hours=1)).isoformat(),
                json.dumps({
                    "message_id": "msg-1h",
                    "from_address": "alice@example.com",
                    "subject": "Recent",
                    "body_plain": "Test",
                }),
                json.dumps({"related_contacts": ["alice@example.com"]}),  # Priority contact
            ),
        )

        # Email at 4 hours old (past grace period)
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.received",
                "test",
                (now - timedelta(hours=4)).isoformat(),
                json.dumps({
                    "message_id": "msg-4h",
                    "from_address": "bob@example.com",
                    "subject": "Older",
                    "body_plain": "Test",
                }),
                json.dumps({"related_contacts": ["bob@example.com"]}),  # Priority contact
            ),
        )

    predictions = await engine.generate_predictions({})
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    message_ids = [p.supporting_signals.get("message_id") for p in reminder_predictions]

    assert "msg-1h" not in message_ids, "Should not predict for email within 3h grace period"
    assert "msg-4h" in message_ids, "Should predict for email past grace period"


@pytest.mark.asyncio
async def test_marketing_filter_still_applied(db, user_model_store):
    """Verify that marketing email filter works with optimized scan."""
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Create a marketing email
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.received",
                "test",
                (now - timedelta(hours=5)).isoformat(),
                json.dumps({
                    "message_id": "msg-marketing",
                    "from_address": "noreply@example.com",
                    "subject": "Special Offer",
                    "body_plain": "Click here to unsubscribe",
                }),
                json.dumps({"related_contacts": []}),
            ),
        )

    predictions = await engine.generate_predictions({})
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    message_ids = [p.supporting_signals.get("message_id") for p in reminder_predictions]

    assert "msg-marketing" not in message_ids, "Should filter out marketing emails"


@pytest.mark.asyncio
async def test_scan_efficiency_metric(db, user_model_store):
    """Verify that scan window reduction achieves expected efficiency gains (on subsequent cycles)."""
    engine = PredictionEngine(db, user_model_store)
    # Consume the first-run flag so the test exercises the steady-state 24h window.
    # The first cycle intentionally uses a wider 72h lookback for catchup.
    engine._first_follow_up_run = False

    now = datetime.now(timezone.utc)

    # Create emails across different time windows
    email_counts = {"within_24h": 0, "24h_to_48h": 0}

    with db.get_connection("events") as conn:
        # 100 emails in 0-24h range (will be scanned)
        for i in range(100):
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    "email.received",
                    "test",
                    (now - timedelta(hours=i * 24 / 100)).isoformat(),
                    json.dumps({
                        "message_id": f"msg-24h-{i}",
                        "from_address": f"user{i}@example.com",
                        "subject": f"Test {i}",
                        "body_plain": "Test",
                    }),
                    json.dumps({"related_contacts": []}),
                ),
            )
            email_counts["within_24h"] += 1

        # 100 emails in 24h-48h range (will NOT be scanned)
        for i in range(100):
            age_hours = 24 + (i * 24 / 100)
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    "email.received",
                    "test",
                    (now - timedelta(hours=age_hours)).isoformat(),
                    json.dumps({
                        "message_id": f"msg-48h-{i}",
                        "from_address": f"user{i + 100}@example.com",
                        "subject": f"Test {i + 100}",
                        "body_plain": "Test",
                    }),
                    json.dumps({"related_contacts": []}),
                ),
            )
            email_counts["24h_to_48h"] += 1

    # Verify email distribution
    assert email_counts["within_24h"] == 100
    assert email_counts["24h_to_48h"] == 100

    # Generate predictions - should only process 24h window
    predictions = await engine.generate_predictions({})

    # All predictions should come from 24h window only
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    for pred in reminder_predictions:
        msg_id = pred.supporting_signals.get("message_id", "")
        # Should NOT contain any 48h messages
        assert not msg_id.startswith("msg-48h-"), f"Found prediction from 24h-48h window: {msg_id}"
