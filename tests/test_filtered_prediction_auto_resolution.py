"""
Test suite for automatic resolution of filtered predictions.

The Problem:
    The prediction engine generates hundreds of thousands of predictions per day,
    but 99.95% are filtered out before being shown to the user (via reaction
    prediction gates, confidence thresholds, or top-N caps). Prior to this fix,
    these filtered predictions remained in the database as unresolved (resolved_at=NULL),
    causing massive database bloat and polluting accuracy metrics.

The Fix:
    Predictions that don't pass reaction prediction or confidence gates are
    immediately marked as resolved with user_response='filtered' at generation
    time. This prevents accumulation of unsurfaced predictions while preserving
    the learning signal for accuracy tracking.

Expected Behavior:
    - Predictions that pass all gates → was_surfaced=1, resolved_at=NULL
    - Predictions filtered by reaction prediction → was_surfaced=0, resolved_at=NOW, user_response='filtered'
    - Predictions filtered by confidence threshold → same as above
    - Predictions filtered by top-N cap → same as above
"""

import pytest
from datetime import datetime, timedelta, timezone

from models.user_model import Prediction
from services.prediction_engine.engine import PredictionEngine
from storage.database import DatabaseManager, UserModelStore


@pytest.mark.asyncio
async def test_filtered_predictions_are_auto_resolved(db: DatabaseManager):
    """Predictions that don't pass reaction gates should be immediately resolved."""
    ums = UserModelStore(db)
    engine = PredictionEngine(db, ums)

    # Create a bunch of test events to trigger prediction generation
    with db.get_connection("events") as conn:
        now = datetime.now(timezone.utc)

        # Create multiple unreplied emails from different senders
        for i in range(20):
            hours_ago = i + 1
            timestamp = (now - timedelta(hours=hours_ago)).isoformat()

            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"email-{i}",
                    "email.received",
                    "proton_mail",
                    timestamp,
                    f'{{"from": "person{i}@example.com", "subject": "Test {i}", "body": "Please reply"}}',
                    '{"requires_response": true}',
                ),
            )

    # Generate predictions
    # With reaction prediction filtering and top-5 cap, most will be filtered
    predictions = await engine.generate_predictions({})

    # Verify that we got the top 5 surfaced predictions
    assert len(predictions) <= 5, "Should cap at 5 surfaced predictions"

    # Check database state
    with db.get_connection("user_model") as conn:
        # Count total predictions
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert total >= 5, f"Should have generated predictions, got {total}"

        # Count surfaced predictions
        surfaced = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE was_surfaced = 1"
        ).fetchone()[0]
        assert surfaced == len(predictions), f"Surfaced count mismatch: {surfaced} vs {len(predictions)}"

        # Count filtered predictions (should be auto-resolved)
        filtered = conn.execute(
            """SELECT COUNT(*) FROM predictions
               WHERE was_surfaced = 0
                 AND user_response = 'filtered'
                 AND resolved_at IS NOT NULL"""
        ).fetchone()[0]
        assert filtered == total - surfaced, (
            f"All non-surfaced predictions should be auto-resolved as 'filtered'. "
            f"Total={total}, Surfaced={surfaced}, Filtered={filtered}"
        )

        # Verify no unresolved filtered predictions remain
        unresolved_filtered = conn.execute(
            """SELECT COUNT(*) FROM predictions
               WHERE was_surfaced = 0 AND resolved_at IS NULL"""
        ).fetchone()[0]
        assert unresolved_filtered == 0, (
            f"Found {unresolved_filtered} unresolved filtered predictions - should be 0"
        )


@pytest.mark.asyncio
async def test_surfaced_predictions_not_auto_resolved(db: DatabaseManager):
    """Predictions that pass all gates should NOT be auto-resolved."""
    ums = UserModelStore(db)
    engine = PredictionEngine(db, ums)

    # Create a high-priority event that should generate a surfaced prediction
    with db.get_connection("events") as conn:
        now = datetime.now(timezone.utc)
        timestamp = (now - timedelta(hours=25)).isoformat()

        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "important-email",
                "email.received",
                "proton_mail",
                timestamp,
                '{"from": "boss@company.com", "subject": "URGENT: Need response", "body": "Please reply ASAP"}',
                '{"requires_response": true, "related_contacts": ["boss@company.com"]}',
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({})

    # Should have at least one surfaced prediction
    assert len(predictions) >= 1, "Should generate at least one surfaced prediction"

    # Check database state
    with db.get_connection("user_model") as conn:
        # All surfaced predictions should be unresolved
        surfaced_resolved = conn.execute(
            """SELECT COUNT(*) FROM predictions
               WHERE was_surfaced = 1 AND resolved_at IS NOT NULL"""
        ).fetchone()[0]
        assert surfaced_resolved == 0, (
            f"Surfaced predictions should NOT be auto-resolved, found {surfaced_resolved}"
        )

        # All surfaced predictions should have user_response=NULL
        surfaced_with_response = conn.execute(
            """SELECT COUNT(*) FROM predictions
               WHERE was_surfaced = 1 AND user_response IS NOT NULL"""
        ).fetchone()[0]
        assert surfaced_with_response == 0, (
            f"Surfaced predictions should have user_response=NULL, found {surfaced_with_response}"
        )


@pytest.mark.asyncio
async def test_filtered_predictions_have_timestamp(db: DatabaseManager):
    """Filtered predictions should have a valid resolved_at timestamp."""
    ums = UserModelStore(db)
    engine = PredictionEngine(db, ums)

    # Create test events
    with db.get_connection("events") as conn:
        now = datetime.now(timezone.utc)

        for i in range(10):
            timestamp = (now - timedelta(hours=i+1)).isoformat()
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"email-{i}",
                    "email.received",
                    "proton_mail",
                    timestamp,
                    f'{{"from": "person{i}@example.com", "subject": "Test", "body": "..."}}',
                    '{}',
                ),
            )

    before_generation = datetime.now(timezone.utc)

    # Generate predictions
    await engine.generate_predictions({})

    after_generation = datetime.now(timezone.utc)

    # Check that filtered predictions have valid timestamps
    with db.get_connection("user_model") as conn:
        filtered_preds = conn.execute(
            """SELECT id, created_at, resolved_at
               FROM predictions
               WHERE was_surfaced = 0 AND user_response = 'filtered'"""
        ).fetchall()

        assert len(filtered_preds) > 0, "Should have some filtered predictions"

        for pred in filtered_preds:
            created_at = datetime.fromisoformat(pred["created_at"].replace('Z', '+00:00'))
            resolved_at = datetime.fromisoformat(pred["resolved_at"].replace('Z', '+00:00'))

            # Resolved timestamp should be same as created (or within 1 second)
            delta = abs((resolved_at - created_at).total_seconds())
            assert delta < 1.0, (
                f"Filtered predictions should be resolved immediately. "
                f"created_at={pred['created_at']}, resolved_at={pred['resolved_at']}, delta={delta}s"
            )

            # Timestamp should be within the generation window
            assert before_generation <= resolved_at <= after_generation, (
                f"Resolved timestamp should be within generation window. "
                f"before={before_generation.isoformat()}, "
                f"resolved={pred['resolved_at']}, "
                f"after={after_generation.isoformat()}"
            )


@pytest.mark.asyncio
async def test_multiple_cycles_dont_accumulate_unresolved(db: DatabaseManager):
    """Multiple prediction cycles should not accumulate unresolved predictions."""
    ums = UserModelStore(db)
    engine = PredictionEngine(db, ums)

    # Create events
    with db.get_connection("events") as conn:
        now = datetime.now(timezone.utc)
        for i in range(5):
            timestamp = (now - timedelta(hours=i+1)).isoformat()
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"email-cycle1-{i}",
                    "email.received",
                    "proton_mail",
                    timestamp,
                    f'{{"from": "person{i}@example.com", "subject": "Test", "body": "..."}}',
                    '{}',
                ),
            )

    # First cycle
    await engine.generate_predictions({})

    with db.get_connection("user_model") as conn:
        unresolved_after_cycle1 = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE resolved_at IS NULL AND was_surfaced = 0"
        ).fetchone()[0]
        assert unresolved_after_cycle1 == 0, (
            f"After cycle 1, should have 0 unresolved filtered predictions, got {unresolved_after_cycle1}"
        )

    # Add more events for second cycle
    with db.get_connection("events") as conn:
        for i in range(5):
            timestamp = (now - timedelta(hours=i+0.5)).isoformat()
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"email-cycle2-{i}",
                    "email.received",
                    "proton_mail",
                    timestamp,
                    f'{{"from": "person{i}@example.com", "subject": "Test 2", "body": "..."}}',
                    '{}',
                ),
            )

    # Second cycle
    await engine.generate_predictions({})

    with db.get_connection("user_model") as conn:
        unresolved_after_cycle2 = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE resolved_at IS NULL AND was_surfaced = 0"
        ).fetchone()[0]
        assert unresolved_after_cycle2 == 0, (
            f"After cycle 2, should have 0 unresolved filtered predictions, got {unresolved_after_cycle2}"
        )


@pytest.mark.asyncio
async def test_accuracy_queries_exclude_filtered_predictions(db: DatabaseManager):
    """Accuracy tracking queries should only look at surfaced predictions."""
    ums = UserModelStore(db)
    engine = PredictionEngine(db, ums)

    # Create events
    with db.get_connection("events") as conn:
        now = datetime.now(timezone.utc)
        for i in range(15):
            timestamp = (now - timedelta(hours=i+1)).isoformat()
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"email-{i}",
                    "email.received",
                    "proton_mail",
                    timestamp,
                    f'{{"from": "person{i}@example.com", "subject": "Test", "body": "..."}}',
                    '{}',
                ),
            )

    # Generate predictions
    await engine.generate_predictions({})

    # Verify accuracy queries work correctly
    with db.get_connection("user_model") as conn:
        # Count total predictions
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        # Count surfaced predictions
        surfaced = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE was_surfaced = 1"
        ).fetchone()[0]

        # Count filtered predictions
        filtered = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE user_response = 'filtered'"
        ).fetchone()[0]

        # Accuracy query (from prediction_engine.py _get_accuracy_multiplier)
        # Should only look at surfaced predictions
        accuracy_query_count = conn.execute(
            """SELECT COUNT(*)
               FROM predictions
               WHERE prediction_type = 'reminder'
                 AND was_surfaced = 1
                 AND resolved_at IS NOT NULL"""
        ).fetchone()[0]

        # This should be 0 because we haven't marked any surfaced predictions as resolved yet
        assert accuracy_query_count == 0, (
            f"Accuracy query should return 0 (no surfaced predictions resolved yet), "
            f"got {accuracy_query_count}"
        )

        # Behavioral tracker query (from tracker.py run_inference_cycle)
        # Should also only look at surfaced predictions
        tracker_query_count = conn.execute(
            """SELECT COUNT(*)
               FROM predictions
               WHERE was_surfaced = 1
                 AND resolved_at IS NULL"""
        ).fetchone()[0]

        # This should equal the number of surfaced predictions
        assert tracker_query_count == surfaced, (
            f"Behavioral tracker query should return {surfaced} (all surfaced predictions), "
            f"got {tracker_query_count}"
        )

        # Verify filtered predictions are NOT included in either query
        assert filtered > 0, "Should have some filtered predictions for this test"
        assert total == surfaced + filtered, (
            f"Total should equal surfaced + filtered. "
            f"Total={total}, Surfaced={surfaced}, Filtered={filtered}"
        )


@pytest.mark.asyncio
async def test_filtered_due_to_confidence_threshold(db: DatabaseManager):
    """Predictions filtered by confidence threshold should be auto-resolved."""
    ums = UserModelStore(db)
    engine = PredictionEngine(db, ums)

    # Create events that will generate low-confidence predictions
    # (old messages that are less likely to need followup)
    with db.get_connection("events") as conn:
        now = datetime.now(timezone.utc)

        # Create very old unreplied messages (10+ days old)
        # These should have low confidence due to age penalty
        for i in range(10):
            days_ago = 10 + i
            timestamp = (now - timedelta(days=days_ago)).isoformat()

            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"old-email-{i}",
                    "email.received",
                    "proton_mail",
                    timestamp,
                    f'{{"from": "person{i}@example.com", "subject": "Old message", "body": "..."}}',
                    '{}',
                ),
            )

    # Generate predictions
    predictions = await engine.generate_predictions({})

    # Check that predictions exist and any filtered ones are auto-resolved
    with db.get_connection("user_model") as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        # If predictions were generated, check filtering behavior
        if total > 0:
            surfaced = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE was_surfaced = 1"
            ).fetchone()[0]
            filtered = conn.execute(
                """SELECT COUNT(*) FROM predictions
                   WHERE was_surfaced = 0 AND user_response = 'filtered'"""
            ).fetchone()[0]

            # All non-surfaced should be filtered
            assert filtered == total - surfaced, (
                f"All non-surfaced predictions should be auto-resolved as 'filtered'. "
                f"Total={total}, Surfaced={surfaced}, Filtered={filtered}"
            )
        # If no predictions were generated, that's OK - the events were too old
        # and may have been filtered before generation


@pytest.mark.asyncio
async def test_filtered_due_to_top_n_cap(db: DatabaseManager):
    """Predictions filtered by top-5 cap should be auto-resolved."""
    ums = UserModelStore(db)
    engine = PredictionEngine(db, ums)

    # Create many recent high-confidence events
    # This should generate more than 5 predictions that pass all gates
    with db.get_connection("events") as conn:
        now = datetime.now(timezone.utc)

        # Create 10 recent unreplied messages
        for i in range(10):
            timestamp = (now - timedelta(hours=2+i)).isoformat()

            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"recent-email-{i}",
                    "email.received",
                    "proton_mail",
                    timestamp,
                    f'{{"from": "colleague{i}@company.com", "subject": "Need your input", "body": "Please reply when you can"}}',
                    '{"requires_response": true}',
                ),
            )

    # Generate predictions
    predictions = await engine.generate_predictions({})

    # Should cap at 5
    assert len(predictions) <= 5, f"Should cap at 5 surfaced predictions, got {len(predictions)}"

    with db.get_connection("user_model") as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        surfaced = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE was_surfaced = 1"
        ).fetchone()[0]
        filtered = conn.execute(
            """SELECT COUNT(*) FROM predictions
               WHERE was_surfaced = 0 AND user_response = 'filtered'"""
        ).fetchone()[0]

        # Verify that surfaced predictions respect the cap
        assert surfaced <= 5, f"Should surface at most 5 predictions, got {surfaced}"

        # All non-surfaced predictions should be filtered
        assert filtered == total - surfaced, (
            f"All non-surfaced predictions should be auto-resolved. "
            f"Total={total}, Surfaced={surfaced}, Filtered={filtered}"
        )
