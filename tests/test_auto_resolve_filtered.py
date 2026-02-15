"""
Tests for auto-resolving filtered predictions.

Predictions with was_surfaced=0 were filtered by confidence gates or reaction
prediction and never shown to the user. These should be auto-resolved after a
timeout to prevent database bloat and polluted accuracy metrics.
"""

import pytest
from datetime import datetime, timedelta, timezone


@pytest.mark.asyncio
async def test_auto_resolve_filtered_predictions_basic(db, notification_manager):
    """Filtered predictions older than timeout should be auto-resolved."""
    now = datetime.now(timezone.utc)
    two_hours_ago = (now - timedelta(hours=2)).isoformat()
    thirty_min_ago = (now - timedelta(minutes=30)).isoformat()

    # Create 5 old filtered predictions (was_surfaced=0, created 2h ago)
    with db.get_connection("user_model") as conn:
        for i in range(5):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, created_at)
                   VALUES (?, 'reminder', 'Test prediction', 0.5, 'SUGGEST', 0, ?)""",
                (f"old-filtered-{i}", two_hours_ago),
            )

        # Create 3 recent filtered predictions (created 30 min ago, should not resolve)
        for i in range(3):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, created_at)
                   VALUES (?, 'reminder', 'Test prediction', 0.5, 'SUGGEST', 0, ?)""",
                (f"recent-filtered-{i}", thirty_min_ago),
            )

    # Auto-resolve with 1-hour timeout
    resolved_count = notification_manager.auto_resolve_filtered_predictions(timeout_hours=1)

    # Should resolve 5 old predictions, leave 3 recent ones alone
    assert resolved_count == 5, "Should resolve 5 old filtered predictions"

    # Verify old predictions are resolved
    with db.get_connection("user_model") as conn:
        for i in range(5):
            row = conn.execute(
                "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
                (f"old-filtered-{i}",),
            ).fetchone()
            assert row is not None, "Prediction should exist"
            assert row["was_accurate"] is None, "Filtered predictions should have NULL accuracy"
            assert row["user_response"] == "filtered", "User response should be 'filtered'"
            assert row["resolved_at"] is not None, "Resolved_at should be set"

        # Verify recent predictions are still unresolved
        for i in range(3):
            row = conn.execute(
                "SELECT resolved_at FROM predictions WHERE id = ?",
                (f"recent-filtered-{i}",),
            ).fetchone()
            assert row["resolved_at"] is None, "Recent predictions should remain unresolved"


@pytest.mark.asyncio
async def test_auto_resolve_filtered_predictions_ignores_surfaced(db, notification_manager):
    """Should only resolve unsurfaced predictions, not surfaced ones."""
    now = datetime.now(timezone.utc)
    two_hours_ago = (now - timedelta(hours=2)).isoformat()

    with db.get_connection("user_model") as conn:
        # Create 3 old filtered predictions (was_surfaced=0)
        for i in range(3):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, created_at)
                   VALUES (?, 'reminder', 'Filtered prediction', 0.5, 'SUGGEST', 0, ?)""",
                (f"filtered-{i}", two_hours_ago),
            )

        # Create 3 old surfaced predictions (was_surfaced=1)
        for i in range(3):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, created_at)
                   VALUES (?, 'reminder', 'Surfaced prediction', 0.8, 'DEFAULT', 1, ?)""",
                (f"surfaced-{i}", two_hours_ago),
            )

    # Auto-resolve with 1-hour timeout
    resolved_count = notification_manager.auto_resolve_filtered_predictions(timeout_hours=1)

    # Should only resolve the 3 filtered predictions
    assert resolved_count == 3, "Should only resolve filtered predictions"

    # Verify surfaced predictions are still unresolved
    with db.get_connection("user_model") as conn:
        for i in range(3):
            row = conn.execute(
                "SELECT resolved_at FROM predictions WHERE id = ?",
                (f"surfaced-{i}",),
            ).fetchone()
            assert row["resolved_at"] is None, "Surfaced predictions should remain unresolved"


@pytest.mark.asyncio
async def test_auto_resolve_filtered_predictions_idempotent(db, notification_manager):
    """Running auto-resolve multiple times should not re-resolve predictions."""
    now = datetime.now(timezone.utc)
    two_hours_ago = (now - timedelta(hours=2)).isoformat()

    # Create 3 old filtered predictions
    with db.get_connection("user_model") as conn:
        for i in range(3):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, created_at)
                   VALUES (?, 'reminder', 'Test prediction', 0.5, 'SUGGEST', 0, ?)""",
                (f"filtered-{i}", two_hours_ago),
            )

    # First run should resolve all 3
    first_run = notification_manager.auto_resolve_filtered_predictions(timeout_hours=1)
    assert first_run == 3, "First run should resolve 3 predictions"

    # Second run should resolve 0 (already resolved)
    second_run = notification_manager.auto_resolve_filtered_predictions(timeout_hours=1)
    assert second_run == 0, "Second run should resolve 0 predictions (idempotent)"


@pytest.mark.asyncio
async def test_auto_resolve_filtered_predictions_different_timeouts(db, notification_manager):
    """Different timeout values should resolve different sets of predictions."""
    now = datetime.now(timezone.utc)
    three_hours_ago = (now - timedelta(hours=3)).isoformat()
    ninety_min_ago = (now - timedelta(minutes=90)).isoformat()
    thirty_min_ago = (now - timedelta(minutes=30)).isoformat()

    with db.get_connection("user_model") as conn:
        # Create predictions at different ages
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, created_at)
               VALUES ('very-old', 'reminder', 'Test', 0.5, 'SUGGEST', 0, ?)""",
            (three_hours_ago,),
        )
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, created_at)
               VALUES ('medium-old', 'reminder', 'Test', 0.5, 'SUGGEST', 0, ?)""",
            (ninety_min_ago,),
        )
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, created_at)
               VALUES ('recent', 'reminder', 'Test', 0.5, 'SUGGEST', 0, ?)""",
            (thirty_min_ago,),
        )

    # 2-hour timeout should resolve only the very-old one
    resolved_2h = notification_manager.auto_resolve_filtered_predictions(timeout_hours=2)
    assert resolved_2h == 1, "2-hour timeout should resolve 1 prediction"

    # 1-hour timeout should resolve the medium-old one (very-old already resolved)
    resolved_1h = notification_manager.auto_resolve_filtered_predictions(timeout_hours=1)
    assert resolved_1h == 1, "1-hour timeout should resolve 1 more prediction"

    # Recent one should still be unresolved
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolved_at FROM predictions WHERE id = 'recent'",
        ).fetchone()
        assert row["resolved_at"] is None, "Recent prediction should remain unresolved"


@pytest.mark.asyncio
async def test_auto_resolve_filtered_predictions_accuracy_query(db, notification_manager):
    """Accuracy queries should ignore filtered predictions (was_accurate=NULL)."""
    now = datetime.now(timezone.utc)
    two_hours_ago = (now - timedelta(hours=2)).isoformat()
    one_day_ago = (now - timedelta(days=1)).isoformat()

    with db.get_connection("user_model") as conn:
        # Create 5 filtered predictions (will be auto-resolved with was_accurate=NULL)
        for i in range(5):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, created_at)
                   VALUES (?, 'reminder', 'Filtered', 0.5, 'SUGGEST', 0, ?)""",
                (f"filtered-{i}", two_hours_ago),
            )

        # Create 10 surfaced predictions: 7 accurate, 3 inaccurate
        for i in range(7):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at, user_response, created_at)
                   VALUES (?, 'reminder', 'Surfaced accurate', 0.8, 'DEFAULT',
                           1, 1, ?, 'acted_on', ?)""",
                (f"accurate-{i}", one_day_ago, one_day_ago),
            )
        for i in range(3):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at, user_response, created_at)
                   VALUES (?, 'reminder', 'Surfaced inaccurate', 0.8, 'DEFAULT',
                           1, 0, ?, 'dismissed', ?)""",
                (f"inaccurate-{i}", one_day_ago, one_day_ago),
            )

    # Auto-resolve filtered predictions
    resolved = notification_manager.auto_resolve_filtered_predictions(timeout_hours=1)
    assert resolved == 5, "Should resolve 5 filtered predictions"

    # Query accuracy (mimics _get_accuracy_multiplier logic)
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
               FROM predictions
               WHERE prediction_type = 'reminder'
                 AND was_surfaced = 1
                 AND resolved_at IS NOT NULL""",
        ).fetchone()

        assert row["total"] == 10, "Should count only surfaced predictions"
        assert row["accurate"] == 7, "Should count 7 accurate predictions"
        # Accuracy rate = 7/10 = 70%, which would result in a multiplier > 1.0


@pytest.mark.asyncio
async def test_auto_resolve_filtered_predictions_database_bloat(db, notification_manager):
    """Should significantly reduce unresolved prediction count."""
    now = datetime.now(timezone.utc)
    two_hours_ago = (now - timedelta(hours=2)).isoformat()

    # Simulate the production issue: 270k filtered predictions
    # (We'll use 100 for testing speed, but the pattern is the same)
    with db.get_connection("user_model") as conn:
        for i in range(100):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, created_at)
                   VALUES (?, 'reminder', 'Filtered', 0.5, 'SUGGEST', 0, ?)""",
                (f"filtered-{i}", two_hours_ago),
            )

        # Check initial unresolved count
        before_row = conn.execute(
            "SELECT COUNT(*) as count FROM predictions WHERE resolved_at IS NULL",
        ).fetchone()
        assert before_row["count"] == 100, "Should have 100 unresolved predictions"

    # Auto-resolve
    resolved = notification_manager.auto_resolve_filtered_predictions(timeout_hours=1)
    assert resolved == 100, "Should resolve all 100 filtered predictions"

    # Check final unresolved count
    with db.get_connection("user_model") as conn:
        after_row = conn.execute(
            "SELECT COUNT(*) as count FROM predictions WHERE resolved_at IS NULL",
        ).fetchone()
        assert after_row["count"] == 0, "Should have 0 unresolved predictions"


@pytest.mark.asyncio
async def test_auto_resolve_filtered_predictions_distinguishes_from_ignored(db, notification_manager):
    """Filtered predictions should have user_response='filtered', not 'ignored'."""
    now = datetime.now(timezone.utc)
    two_hours_ago = (now - timedelta(hours=2)).isoformat()

    # Create a filtered prediction
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, created_at)
               VALUES ('filtered', 'reminder', 'Test', 0.5, 'SUGGEST', 0, ?)""",
            (two_hours_ago,),
        )

    # Auto-resolve
    notification_manager.auto_resolve_filtered_predictions(timeout_hours=1)

    # Verify user_response is 'filtered', not 'ignored'
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT user_response FROM predictions WHERE id = 'filtered'",
        ).fetchone()
        assert row["user_response"] == "filtered", "Should be marked 'filtered', not 'ignored'"
        # 'ignored' is used for surfaced predictions that the user never interacted with
        # 'filtered' is for predictions that were never surfaced in the first place
