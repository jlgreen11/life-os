"""
Tests for the resolution_reason column and its effect on accuracy multiplier calculations.

Background
----------
Opportunity predictions had a measured accuracy of 19% (41/248), but most of the
"inaccurate" resolutions came from the automated-sender fast-path in
BehavioralAccuracyTracker — predictions generated before marketing filter improvements
(PRs #183–#189) that targeted no-reply/automated addresses the user could never reach
out to. These were historical prediction-generation bugs, not real user-behavior signals.

Without distinguishing fast-path resolutions from real behavior, the accuracy multiplier
applied a 0.3 penalty floor. Combined with the opportunity confidence cap of 0.6:
    0.6 × 0.3 = 0.18 → below 0.3 surfacing threshold → ALL opportunity predictions suppressed

This fix:
1. Adds ``resolution_reason`` column to the predictions table (migration 3→4)
2. BehavioralAccuracyTracker sets ``resolution_reason='automated_sender_fast_path'``
   when resolving inaccurate predictions via the automated-sender fast-path
3. ``_get_accuracy_multiplier`` excludes fast-path rows so accuracy reflects real behavior

Tests
-----
1. Schema migration: predictions table has resolution_reason column
2. Tracker: sets resolution_reason='automated_sender_fast_path' for automated senders
3. Tracker: leaves resolution_reason=None for real contacts
4. Tracker: leaves resolution_reason=None for accurate predictions
5. Multiplier: excludes automated_sender_fast_path rows from count
6. Multiplier: opportunity predictions recover to normal multiplier after exclusion
7. Multiplier: non-fast-path inaccurate rows still count against accuracy
8. resolve_prediction: stores resolution_reason via UserModelStore
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker
from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_prediction(db, prediction_type: str, was_accurate: bool = None,
                       was_surfaced: bool = True, resolved_at: str = None,
                       resolution_reason: str = None,
                       contact_email: str = None) -> str:
    """Insert a prediction row and return its ID.

    Args:
        db: DatabaseManager fixture.
        prediction_type: e.g. 'opportunity', 'reminder'.
        was_accurate: True/False/None (unresolved).
        was_surfaced: Whether the prediction reached the notification layer.
        resolved_at: ISO timestamp or None for unresolved.
        resolution_reason: e.g. 'automated_sender_fast_path' or None.
        contact_email: Stored in supporting_signals for tracker tests.
    """
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    signals = json.dumps({"contact_email": contact_email} if contact_email else {})

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, was_accurate, resolution_reason,
                resolved_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                prediction_type,
                f"Test {prediction_type} prediction",
                0.5,
                "suggest",
                signals,
                1 if was_surfaced else 0,
                (1 if was_accurate else 0) if was_accurate is not None else None,
                resolution_reason,
                resolved_at or (now if was_accurate is not None else None),
                now,
            ),
        )
    return pred_id


# ---------------------------------------------------------------------------
# 1. Schema migration: predictions table has resolution_reason column
# ---------------------------------------------------------------------------


def test_predictions_table_has_resolution_reason_column(db):
    """The predictions table must have a resolution_reason column after migration.

    This verifies that migration 3→4 ran correctly and added the column.
    Without this column, the accuracy multiplier query would fail with a
    'no such column' error.
    """
    with db.get_connection("user_model") as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    assert "resolution_reason" in columns, (
        "predictions table is missing 'resolution_reason' column — "
        "migration 3→4 may not have run. Check _migrate_user_model_db."
    )


def test_resolution_reason_column_is_nullable(db):
    """resolution_reason should be nullable (pre-v4 rows have no reason)."""
    pred_id = _insert_prediction(db, "opportunity", was_accurate=False,
                                 resolved_at=datetime.now(timezone.utc).isoformat())
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
    assert row is not None
    assert row["resolution_reason"] is None, (
        "resolution_reason should be NULL for pre-v4 rows (no reason supplied)"
    )


# ---------------------------------------------------------------------------
# 2. Tracker: sets resolution_reason for automated senders
# ---------------------------------------------------------------------------


def test_get_resolution_reason_returns_fast_path_for_automated_sender():
    """_get_resolution_reason returns 'automated_sender_fast_path' for noreply contacts.

    Uses a minimal DatabaseManager mock because _get_resolution_reason only
    uses self._is_automated_sender, which is a pure function.
    """
    from unittest.mock import MagicMock
    mock_db = MagicMock()
    tracker = BehavioralAccuracyTracker(mock_db)

    prediction = {
        "prediction_type": "opportunity",
        "description": "Reach out to noreply@company.com",
        "supporting_signals": json.dumps({"contact_email": "noreply@company.com"}),
    }
    reason = tracker._get_resolution_reason(prediction, was_accurate=False)
    assert reason == "automated_sender_fast_path", (
        f"Expected 'automated_sender_fast_path' for noreply sender, got {reason!r}"
    )


def test_get_resolution_reason_returns_fast_path_for_mailer_daemon():
    """_get_resolution_reason returns 'automated_sender_fast_path' for mailer-daemon."""
    from unittest.mock import MagicMock
    mock_db = MagicMock()
    tracker = BehavioralAccuracyTracker(mock_db)

    prediction = {
        "supporting_signals": json.dumps({"contact_email": "mailer-daemon@mail.example.com"}),
    }
    reason = tracker._get_resolution_reason(prediction, was_accurate=False)
    assert reason == "automated_sender_fast_path"


# ---------------------------------------------------------------------------
# 3. Tracker: leaves resolution_reason=None for real contacts
# ---------------------------------------------------------------------------


def test_get_resolution_reason_returns_none_for_real_contact():
    """_get_resolution_reason returns None for real human contacts.

    A real human contact (alice@example.com) whose prediction was inaccurate
    (user didn't reach out) should have resolution_reason=None, because this
    IS a real behavioral signal that should count against accuracy.
    """
    from unittest.mock import MagicMock
    mock_db = MagicMock()
    tracker = BehavioralAccuracyTracker(mock_db)

    prediction = {
        "supporting_signals": json.dumps({"contact_email": "alice@example.com"}),
    }
    reason = tracker._get_resolution_reason(prediction, was_accurate=False)
    assert reason is None, (
        f"Expected None for real human contact, got {reason!r}. "
        "Real inaccurate predictions should count against accuracy."
    )


def test_get_resolution_reason_returns_none_for_missing_signals():
    """_get_resolution_reason returns None when supporting_signals is empty."""
    from unittest.mock import MagicMock
    mock_db = MagicMock()
    tracker = BehavioralAccuracyTracker(mock_db)

    prediction = {"supporting_signals": "{}"}
    reason = tracker._get_resolution_reason(prediction, was_accurate=False)
    assert reason is None


# ---------------------------------------------------------------------------
# 4. Tracker: leaves resolution_reason=None for accurate predictions
# ---------------------------------------------------------------------------


def test_get_resolution_reason_returns_none_for_accurate_prediction():
    """_get_resolution_reason always returns None for accurate predictions.

    Even if the contact happens to look like an automated sender, an accurate
    resolution means the user DID reach out — this is a real behavioral signal.
    """
    from unittest.mock import MagicMock
    mock_db = MagicMock()
    tracker = BehavioralAccuracyTracker(mock_db)

    prediction = {
        "supporting_signals": json.dumps({"contact_email": "noreply@company.com"}),
    }
    # was_accurate=True: user DID reach out (maybe they replied to a human-forwarded email)
    reason = tracker._get_resolution_reason(prediction, was_accurate=True)
    assert reason is None, (
        "Accurate predictions should always have resolution_reason=None "
        "regardless of contact type — they represent real positive behavior."
    )


# ---------------------------------------------------------------------------
# 5. Multiplier: excludes automated_sender_fast_path rows from count
# ---------------------------------------------------------------------------


def test_accuracy_multiplier_excludes_fast_path_resolutions(db, user_model_store):
    """_get_accuracy_multiplier excludes automated_sender_fast_path rows.

    Setup: 5 accurate + 20 fast-path inaccurate = raw 20% accuracy
    After exclusion: 5 accurate / 5 total = 100% accuracy → multiplier = 1.1
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc).isoformat()

    # 5 accurate predictions (real user behavior)
    for _ in range(5):
        _insert_prediction(db, "opportunity", was_accurate=True,
                           resolved_at=now, resolution_reason=None)

    # 20 fast-path inaccurate predictions (historical bug, not real behavior)
    for _ in range(20):
        _insert_prediction(db, "opportunity", was_accurate=False,
                           resolved_at=now,
                           resolution_reason="automated_sender_fast_path")

    multiplier = engine._get_accuracy_multiplier("opportunity")

    # After excluding fast-path: 5/5 = 100% → 0.5 + 1.0 * 0.6 = 1.1
    assert abs(multiplier - 1.1) < 0.01, (
        f"Expected 1.1 (100% accuracy after excluding fast-path), got {multiplier:.3f}. "
        "The 20 automated-sender fast-path rows should be excluded from the count."
    )


def test_accuracy_multiplier_with_no_real_behavior_rows_returns_default(db, user_model_store):
    """When ALL resolved rows are fast-path, total=0 → default multiplier of 1.0.

    If there's no real behavioral data (only fast-path resolutions), the system
    should apply the 'insufficient data' default (1.0) rather than a penalty.
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc).isoformat()

    # 15 fast-path resolutions only (no real behavior)
    for _ in range(15):
        _insert_prediction(db, "opportunity", was_accurate=False,
                           resolved_at=now,
                           resolution_reason="automated_sender_fast_path")

    multiplier = engine._get_accuracy_multiplier("opportunity")
    assert multiplier == 1.0, (
        f"Expected 1.0 (no real behavioral data after fast-path exclusion), got {multiplier}. "
        "With zero qualifying rows the system should use the insufficient-data default."
    )


# ---------------------------------------------------------------------------
# 6. Multiplier: opportunity predictions recover to normal multiplier
# ---------------------------------------------------------------------------


def test_opportunity_accuracy_multiplier_recovers_with_real_behavior(db, user_model_store):
    """Opportunity multiplier is healthy when real behavior shows reasonable accuracy.

    Simulates the production scenario after the fix:
    - 41 accurate predictions (users did reach out)
    - 33 real inaccurate predictions (users didn't reach out — real signal)
    - 174 fast-path inaccurate (excluded)
    Real accuracy: 41 / (41+33) = 55.4% → multiplier = 0.5 + 0.554 * 0.6 ≈ 0.833
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc).isoformat()

    # 41 accurate (real behavior)
    for _ in range(41):
        _insert_prediction(db, "opportunity", was_accurate=True, resolved_at=now)

    # 33 real inaccurate (user genuinely didn't reach out)
    for _ in range(33):
        _insert_prediction(db, "opportunity", was_accurate=False, resolved_at=now,
                           resolution_reason=None)

    # 174 fast-path inaccurate (excluded from count)
    for _ in range(174):
        _insert_prediction(db, "opportunity", was_accurate=False, resolved_at=now,
                           resolution_reason="automated_sender_fast_path")

    multiplier = engine._get_accuracy_multiplier("opportunity")

    # 41/74 = 55.4% → 0.5 + 0.554 * 0.6 ≈ 0.833
    assert multiplier > 0.8, (
        f"Expected multiplier > 0.8 for 55% real accuracy, got {multiplier:.3f}. "
        "Opportunity predictions should surface after excluding fast-path pollution."
    )
    # Sanity check: multiplier is not penalized
    assert multiplier != 0.3, (
        "Multiplier should NOT be the 0.3 penalty floor — real accuracy is healthy."
    )


# ---------------------------------------------------------------------------
# 7. Non-fast-path inaccurate rows still count against accuracy
# ---------------------------------------------------------------------------


def test_real_inaccurate_rows_still_penalize_accuracy(db, user_model_store):
    """Inaccurate rows without resolution_reason still count against accuracy.

    resolution_reason=None inaccurate rows represent real user behavior (user
    chose not to reach out). These SHOULD penalize the multiplier.
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc).isoformat()

    # 1 accurate, 9 real inaccurate = 10% accuracy → penalty floor
    _insert_prediction(db, "opportunity", was_accurate=True, resolved_at=now)
    for _ in range(9):
        _insert_prediction(db, "opportunity", was_accurate=False, resolved_at=now,
                           resolution_reason=None)

    multiplier = engine._get_accuracy_multiplier("opportunity")
    # 10% accuracy over 10 real samples → penalty floor 0.3
    assert multiplier == 0.3, (
        f"Expected 0.3 penalty floor for 10% real accuracy, got {multiplier}. "
        "Real inaccurate rows (no resolution_reason) should still penalize multiplier."
    )


# ---------------------------------------------------------------------------
# 8. resolve_prediction: stores resolution_reason via UserModelStore
# ---------------------------------------------------------------------------


def test_resolve_prediction_stores_resolution_reason(db, user_model_store):
    """UserModelStore.resolve_prediction persists the resolution_reason field.

    This verifies the storage layer correctly writes the new column, which is
    required for the accuracy multiplier exclusion to work in production.
    """
    pred_id = _insert_prediction(db, "opportunity", was_accurate=None)

    user_model_store.resolve_prediction(
        pred_id,
        was_accurate=False,
        user_response="inferred",
        resolution_reason="automated_sender_fast_path",
    )

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, user_response, resolution_reason, resolved_at "
            "FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None
    assert row["was_accurate"] == 0
    assert row["user_response"] == "inferred"
    assert row["resolution_reason"] == "automated_sender_fast_path"
    assert row["resolved_at"] is not None


def test_resolve_prediction_without_resolution_reason_stores_null(db, user_model_store):
    """resolve_prediction without resolution_reason stores NULL (backward compat)."""
    pred_id = _insert_prediction(db, "reminder", was_accurate=None)

    user_model_store.resolve_prediction(pred_id, was_accurate=True, user_response="inferred")

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()

    assert row["resolution_reason"] is None, (
        "resolution_reason should be NULL when not explicitly provided "
        "(backward compatibility with pre-v4 resolutions)"
    )
