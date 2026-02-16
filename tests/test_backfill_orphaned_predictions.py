"""
Tests for backfill_orphaned_predictions.py

Verifies that orphaned predictions (surfaced but no notification) are correctly
identified and resolved by the backfill script.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.backfill_orphaned_predictions import backfill_orphaned_predictions


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory with test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create user_model.db with predictions table
        user_model_db = data_dir / "user_model.db"
        user_model_conn = sqlite3.connect(user_model_db)
        user_model_conn.execute("""
            CREATE TABLE predictions (
                id TEXT PRIMARY KEY,
                prediction_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                description TEXT NOT NULL,
                was_surfaced INTEGER DEFAULT 0,
                user_response TEXT,
                was_accurate INTEGER,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
        """)
        user_model_conn.commit()
        user_model_conn.close()

        # Create state.db with notifications table
        state_db = data_dir / "state.db"
        state_conn = sqlite3.connect(state_db)
        state_conn.execute("""
            CREATE TABLE notifications (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT,
                domain TEXT NOT NULL,
                status TEXT NOT NULL,
                source_event_id TEXT,
                created_at TEXT NOT NULL,
                delivered_at TEXT
            )
        """)
        state_conn.commit()
        state_conn.close()

        yield data_dir


def insert_prediction(
    db_path: Path,
    pred_id: str,
    created_at: str,
    was_surfaced: int = 1,
    resolved_at: str = None,
):
    """Helper to insert a test prediction."""
    conn = sqlite3.connect(db_path / "user_model.db")
    conn.execute(
        """INSERT INTO predictions
           (id, prediction_type, confidence, description, was_surfaced, created_at, resolved_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (pred_id, "reminder", 0.5, "Test prediction", was_surfaced, created_at, resolved_at),
    )
    conn.commit()
    conn.close()


def insert_notification(
    db_path: Path,
    notif_id: str,
    source_event_id: str,
    created_at: str,
):
    """Helper to insert a test notification."""
    conn = sqlite3.connect(db_path / "state.db")
    conn.execute(
        """INSERT INTO notifications
           (id, title, body, domain, status, source_event_id, created_at, delivered_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (notif_id, "Test notification", "Body", "prediction", "delivered", source_event_id, created_at, created_at),
    )
    conn.commit()
    conn.close()


def get_prediction(db_path: Path, pred_id: str) -> dict:
    """Helper to fetch a prediction by ID."""
    conn = sqlite3.connect(db_path / "user_model.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def test_no_orphaned_predictions(temp_data_dir):
    """Test that the script handles the case where all predictions have notifications."""
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(hours=30)).isoformat()

    # Create a surfaced prediction with a matching notification
    insert_prediction(temp_data_dir, "pred-1", old_time, was_surfaced=1)
    insert_notification(temp_data_dir, "notif-1", "pred-1", old_time)

    # Run the backfill
    stats = backfill_orphaned_predictions(temp_data_dir, dry_run=False)

    # Assert: No orphaned predictions found
    assert stats["total_unresolved"] == 1
    assert stats["orphaned_24h"] == 0
    assert stats["orphaned_48h"] == 0
    assert stats["resolved"] == 0

    # Verify the prediction was not modified
    pred = get_prediction(temp_data_dir, "pred-1")
    assert pred["resolved_at"] is None
    assert pred["was_accurate"] is None
    assert pred["user_response"] is None


def test_orphaned_prediction_old_enough(temp_data_dir):
    """Test that orphaned predictions older than 24h are resolved."""
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(hours=30)).isoformat()

    # Create a surfaced prediction with NO notification (orphaned)
    insert_prediction(temp_data_dir, "pred-orphan", old_time, was_surfaced=1)

    # Run the backfill
    stats = backfill_orphaned_predictions(temp_data_dir, dry_run=False)

    # Assert: Orphaned prediction was resolved
    assert stats["total_unresolved"] == 1
    assert stats["orphaned_24h"] == 1
    assert stats["resolved"] == 1

    # Verify the prediction was marked as orphaned
    pred = get_prediction(temp_data_dir, "pred-orphan")
    assert pred["resolved_at"] is not None
    assert pred["was_accurate"] == 0  # Marked as inaccurate
    assert pred["user_response"] == "orphaned"


def test_orphaned_prediction_too_recent(temp_data_dir):
    """Test that orphaned predictions younger than 24h are NOT resolved."""
    now = datetime.now(timezone.utc)
    recent_time = (now - timedelta(hours=12)).isoformat()

    # Create a recent surfaced prediction with NO notification (orphaned but recent)
    insert_prediction(temp_data_dir, "pred-recent", recent_time, was_surfaced=1)

    # Run the backfill
    stats = backfill_orphaned_predictions(temp_data_dir, dry_run=False)

    # Assert: Recent prediction was not resolved
    assert stats["total_unresolved"] == 1
    assert stats["orphaned_24h"] == 0  # Not old enough to count
    assert stats["resolved"] == 0

    # Verify the prediction was not modified
    pred = get_prediction(temp_data_dir, "pred-recent")
    assert pred["resolved_at"] is None
    assert pred["was_accurate"] is None
    assert pred["user_response"] is None


def test_multiple_orphaned_predictions(temp_data_dir):
    """Test that multiple orphaned predictions are all resolved."""
    now = datetime.now(timezone.utc)
    old_time_1 = (now - timedelta(hours=30)).isoformat()
    old_time_2 = (now - timedelta(hours=50)).isoformat()
    old_time_3 = (now - timedelta(hours=72)).isoformat()

    # Create three orphaned predictions at different ages
    insert_prediction(temp_data_dir, "pred-1", old_time_1, was_surfaced=1)
    insert_prediction(temp_data_dir, "pred-2", old_time_2, was_surfaced=1)
    insert_prediction(temp_data_dir, "pred-3", old_time_3, was_surfaced=1)

    # Run the backfill
    stats = backfill_orphaned_predictions(temp_data_dir, dry_run=False)

    # Assert: All three predictions were resolved
    assert stats["total_unresolved"] == 3
    assert stats["orphaned_24h"] == 1  # pred-1 (30h old)
    assert stats["orphaned_48h"] == 2  # pred-2 (50h) and pred-3 (72h)
    assert stats["resolved"] == 3

    # Verify all predictions were marked as orphaned
    for pred_id in ["pred-1", "pred-2", "pred-3"]:
        pred = get_prediction(temp_data_dir, pred_id)
        assert pred["resolved_at"] is not None
        assert pred["was_accurate"] == 0
        assert pred["user_response"] == "orphaned"


def test_filtered_predictions_not_touched(temp_data_dir):
    """Test that filtered predictions (was_surfaced=0) are not affected."""
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(hours=30)).isoformat()

    # Create a filtered prediction (was_surfaced=0)
    insert_prediction(temp_data_dir, "pred-filtered", old_time, was_surfaced=0)

    # Run the backfill
    stats = backfill_orphaned_predictions(temp_data_dir, dry_run=False)

    # Assert: Filtered prediction was not counted or resolved
    assert stats["total_unresolved"] == 0
    assert stats["resolved"] == 0

    # Verify the prediction was not modified
    pred = get_prediction(temp_data_dir, "pred-filtered")
    assert pred["resolved_at"] is None


def test_already_resolved_predictions_not_touched(temp_data_dir):
    """Test that already-resolved predictions are not re-processed."""
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(hours=30)).isoformat()

    # Create a prediction that's already resolved
    insert_prediction(
        temp_data_dir,
        "pred-resolved",
        old_time,
        was_surfaced=1,
        resolved_at=old_time,
    )

    # Run the backfill
    stats = backfill_orphaned_predictions(temp_data_dir, dry_run=False)

    # Assert: Already-resolved prediction was not counted
    assert stats["total_unresolved"] == 0
    assert stats["resolved"] == 0


def test_dry_run_mode(temp_data_dir):
    """Test that dry-run mode does not modify the database."""
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(hours=30)).isoformat()

    # Create an orphaned prediction
    insert_prediction(temp_data_dir, "pred-orphan", old_time, was_surfaced=1)

    # Run in dry-run mode
    stats = backfill_orphaned_predictions(temp_data_dir, dry_run=True)

    # Assert: Stats report what would be resolved
    assert stats["total_unresolved"] == 1
    assert stats["orphaned_24h"] == 1
    assert stats["resolved"] == 1

    # Verify the prediction was NOT modified
    pred = get_prediction(temp_data_dir, "pred-orphan")
    assert pred["resolved_at"] is None
    assert pred["was_accurate"] is None
    assert pred["user_response"] is None


def test_mixed_scenario(temp_data_dir):
    """Test a realistic mix: orphaned, with-notification, recent, filtered, resolved."""
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(hours=30)).isoformat()
    recent_time = (now - timedelta(hours=12)).isoformat()

    # 1. Orphaned (old, surfaced, no notification) → should be resolved
    insert_prediction(temp_data_dir, "pred-orphan", old_time, was_surfaced=1)

    # 2. With notification (old, surfaced, has notification) → should NOT be resolved
    insert_prediction(temp_data_dir, "pred-with-notif", old_time, was_surfaced=1)
    insert_notification(temp_data_dir, "notif-1", "pred-with-notif", old_time)

    # 3. Recent orphaned (recent, surfaced, no notification) → should NOT be resolved
    insert_prediction(temp_data_dir, "pred-recent", recent_time, was_surfaced=1)

    # 4. Filtered (old, not surfaced) → should NOT be counted
    insert_prediction(temp_data_dir, "pred-filtered", old_time, was_surfaced=0)

    # 5. Already resolved (old, surfaced, already resolved) → should NOT be counted
    insert_prediction(
        temp_data_dir,
        "pred-resolved",
        old_time,
        was_surfaced=1,
        resolved_at=old_time,
    )

    # Run the backfill
    stats = backfill_orphaned_predictions(temp_data_dir, dry_run=False)

    # Assert: Only the first orphaned prediction was resolved
    assert stats["total_unresolved"] == 3  # orphan, with-notif, recent
    assert stats["orphaned_24h"] == 1  # only pred-orphan
    assert stats["resolved"] == 1  # only pred-orphan

    # Verify pred-orphan was resolved
    pred_orphan = get_prediction(temp_data_dir, "pred-orphan")
    assert pred_orphan["resolved_at"] is not None
    assert pred_orphan["user_response"] == "orphaned"

    # Verify pred-with-notif was NOT resolved
    pred_with_notif = get_prediction(temp_data_dir, "pred-with-notif")
    assert pred_with_notif["resolved_at"] is None

    # Verify pred-recent was NOT resolved
    pred_recent = get_prediction(temp_data_dir, "pred-recent")
    assert pred_recent["resolved_at"] is None


def test_empty_database(temp_data_dir):
    """Test that the script handles an empty database gracefully."""
    # Don't insert any predictions
    stats = backfill_orphaned_predictions(temp_data_dir, dry_run=False)

    # Assert: No errors, all counts are zero
    assert stats["total_unresolved"] == 0
    assert stats["orphaned_24h"] == 0
    assert stats["orphaned_48h"] == 0
    assert stats["resolved"] == 0


def test_48h_age_categorization(temp_data_dir):
    """Test that predictions are correctly categorized by age (24h vs 48h)."""
    now = datetime.now(timezone.utc)
    age_30h = (now - timedelta(hours=30)).isoformat()
    age_50h = (now - timedelta(hours=50)).isoformat()

    # Create two orphaned predictions at different ages
    insert_prediction(temp_data_dir, "pred-30h", age_30h, was_surfaced=1)
    insert_prediction(temp_data_dir, "pred-50h", age_50h, was_surfaced=1)

    # Run the backfill
    stats = backfill_orphaned_predictions(temp_data_dir, dry_run=False)

    # Assert: Both are resolved, but categorized differently
    assert stats["total_unresolved"] == 2
    assert stats["orphaned_24h"] == 1  # pred-30h
    assert stats["orphaned_48h"] == 1  # pred-50h
    assert stats["resolved"] == 2
