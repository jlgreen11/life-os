"""
Tests for the prediction backlog cleanup script.

This script cleans up hundreds of thousands of unsurfaced predictions that
accumulated before the auto-resolve system was implemented. These tests verify:

1. Backlog analysis correctly identifies stale predictions
2. Cleanup only touches unsurfaced predictions (was_surfaced=0)
3. Surfaced predictions are preserved for user feedback
4. Dry-run mode doesn't modify the database
5. Cleanup marks predictions with correct metadata
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore

# Import the cleanup script functions by loading the script
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from cleanup_prediction_backlog import analyze_backlog, cleanup_backlog


def create_prediction(db, pred_id, pred_type, created_at, was_surfaced=0):
    """Helper to create a prediction directly in the database."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                pred_type,
                f"Prediction {pred_id}",
                0.7,
                "SUGGEST",
                "next_2_hours",
                None,
                "[]",
                was_surfaced,
                created_at.isoformat(),
            ),
        )


def test_analyze_empty_backlog(db: DatabaseManager):
    """Test that analysis handles an empty database."""
    stats = analyze_backlog(db)

    assert stats["total"] == 0
    assert stats["surfaced"] == 0
    assert stats["unsurfaced"] == 0
    assert stats["resolved"] == 0
    assert stats["unresolved"] == 0
    assert stats["backlog_size"] == 0
    assert stats["oldest_backlog"] is None
    assert stats["newest_backlog"] is None


def test_analyze_backlog_with_stale_predictions(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that analysis correctly identifies stale unsurfaced predictions."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=3)
    recent_time = now - timedelta(minutes=30)

    # Create old unsurfaced predictions (should be in backlog)
    for i in range(5):
        create_prediction(db, f"old-{i}", "reminder", old_time, was_surfaced=0)

    # Create recent unsurfaced predictions (too new for backlog)
    for i in range(3):
        create_prediction(db, f"recent-{i}", "reminder", recent_time, was_surfaced=0)

    # Create surfaced predictions (should NOT be in backlog)
    for i in range(2):
        create_prediction(db, f"surfaced-{i}", "reminder", old_time, was_surfaced=1)

    # Analyze
    stats = analyze_backlog(db)

    assert stats["total"] == 10
    assert stats["surfaced"] == 2
    assert stats["unsurfaced"] == 8
    assert stats["resolved"] == 0
    assert stats["unresolved"] == 10
    assert stats["backlog_size"] == 5  # Only the 5 old unsurfaced ones


def test_cleanup_dry_run_no_changes(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that dry-run mode doesn't modify the database."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=3)

    # Create old unsurfaced predictions
    for i in range(10):
        create_prediction(db, f"pred-{i}", "reminder", old_time, was_surfaced=0)

    # Run dry-run
    count = cleanup_backlog(db, timeout_hours=1, dry_run=True)

    assert count == 10

    # Verify nothing was changed
    with db.get_connection("user_model") as conn:
        resolved = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL"
        ).fetchone()[0]
        assert resolved == 0


def test_cleanup_only_touches_unsurfaced(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that cleanup only resolves unsurfaced predictions."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=3)

    # Create old unsurfaced predictions
    for i in range(5):
        create_prediction(db, f"unsurfaced-{i}", "reminder", old_time, was_surfaced=0)

    # Create old surfaced predictions (should be left alone)
    for i in range(3):
        create_prediction(db, f"surfaced-{i}", "reminder", old_time, was_surfaced=1)

    # Run cleanup
    count = cleanup_backlog(db, timeout_hours=1, dry_run=False)

    assert count == 5  # Only unsurfaced predictions

    # Verify only unsurfaced were resolved
    with db.get_connection("user_model") as conn:
        # Check unsurfaced are resolved
        unsurfaced_resolved = conn.execute(
            """SELECT COUNT(*) FROM predictions
               WHERE was_surfaced = 0 AND resolved_at IS NOT NULL"""
        ).fetchone()[0]
        assert unsurfaced_resolved == 5

        # Check surfaced are still unresolved
        surfaced_unresolved = conn.execute(
            """SELECT COUNT(*) FROM predictions
               WHERE was_surfaced = 1 AND resolved_at IS NULL"""
        ).fetchone()[0]
        assert surfaced_unresolved == 3


def test_cleanup_respects_timeout(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that cleanup only resolves predictions older than timeout."""
    now = datetime.now(timezone.utc)

    # Create predictions at different ages
    create_prediction(db, "very-old-1", "reminder", now - timedelta(hours=5), was_surfaced=0)
    create_prediction(db, "very-old-2", "reminder", now - timedelta(hours=5), was_surfaced=0)
    create_prediction(db, "old-1", "reminder", now - timedelta(hours=2), was_surfaced=0)
    create_prediction(db, "old-2", "reminder", now - timedelta(hours=2), was_surfaced=0)
    create_prediction(db, "recent-1", "reminder", now - timedelta(minutes=30), was_surfaced=0)
    create_prediction(db, "recent-2", "reminder", now - timedelta(minutes=30), was_surfaced=0)
    create_prediction(db, "very-recent-1", "reminder", now - timedelta(minutes=5), was_surfaced=0)
    create_prediction(db, "very-recent-2", "reminder", now - timedelta(minutes=5), was_surfaced=0)

    # Run cleanup with 1-hour timeout
    count = cleanup_backlog(db, timeout_hours=1, dry_run=False)

    # Should only clean up the 4 predictions older than 1 hour
    assert count == 4

    # Verify correct ones were resolved
    with db.get_connection("user_model") as conn:
        resolved_ids = [
            row[0] for row in conn.execute(
                "SELECT id FROM predictions WHERE resolved_at IS NOT NULL"
            ).fetchall()
        ]
        assert len(resolved_ids) == 4
        assert all(pid.startswith(("very-old", "old")) for pid in resolved_ids)


def test_cleanup_sets_correct_metadata(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that cleanup marks predictions with correct metadata."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=3)

    # Create old unsurfaced prediction
    create_prediction(db, "test-pred", "reminder", old_time, was_surfaced=0)

    # Run cleanup
    cleanup_backlog(db, timeout_hours=1, dry_run=False)

    # Verify metadata
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            ("test-pred",)
        ).fetchone()

        assert row["was_accurate"] is None  # Never tested
        assert row["user_response"] == "filtered"  # Filtered out
        assert row["resolved_at"] is not None  # Resolved

        # Verify resolved_at is recent (within last minute)
        resolved_at = datetime.fromisoformat(row["resolved_at"].replace("Z", "+00:00"))
        age_seconds = (now - resolved_at).total_seconds()
        assert age_seconds < 60  # Resolved in the last minute


def test_cleanup_handles_already_resolved(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that cleanup skips predictions that are already resolved."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=3)

    # Create old unsurfaced predictions
    for i in range(3):
        create_prediction(db, f"pred-{i}", "reminder", old_time, was_surfaced=0)

    # Manually resolve one of them first
    with db.get_connection("user_model") as conn:
        conn.execute(
            """UPDATE predictions SET
               resolved_at = ?,
               was_accurate = 1,
               user_response = 'confirmed'
               WHERE id = ?""",
            (now.isoformat(), "pred-0")
        )

    # Run cleanup
    count = cleanup_backlog(db, timeout_hours=1, dry_run=False)

    # Should only clean up the 2 unresolved ones
    assert count == 2

    # Verify the manually resolved one kept its metadata
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT user_response, was_accurate FROM predictions WHERE id = ?",
            ("pred-0",)
        ).fetchone()

        assert row["user_response"] == "confirmed"  # Preserved
        assert row["was_accurate"] == 1  # Preserved


def test_cleanup_massive_backlog_performance(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that cleanup can handle a massive backlog efficiently.

    This simulates the real-world scenario of 266K+ stale predictions.
    We test with 1000 predictions to verify performance without taking too long.
    """
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=3)

    # Create 1000 old unsurfaced predictions
    for i in range(1000):
        create_prediction(db, f"backlog-{i}", "reminder", old_time, was_surfaced=0)

    # Run cleanup and verify it completes quickly
    import time
    start = time.time()
    count = cleanup_backlog(db, timeout_hours=1, dry_run=False)
    elapsed = time.time() - start

    assert count == 1000
    assert elapsed < 5.0  # Should complete in under 5 seconds

    # Verify all were resolved
    with db.get_connection("user_model") as conn:
        resolved = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL"
        ).fetchone()[0]
        assert resolved == 1000
