"""
Tests for BehavioralAccuracyTracker._backfill_automated_sender_tags() and the
description-fallback in _get_resolution_reason().

Problem (pre-fix):
    The one-time migration guard in _ensure_resolution_reason_column() tagged
    existing inaccurate predictions using supporting_signals['contact_email'].
    Predictions created before PR #190 (which added supporting_signals) had NULL
    or empty-list signals, so the migration guard skipped them.  Those predictions
    permanently counted as "inaccurate" in _get_accuracy_multiplier, keeping
    opportunity accuracy at 19% and the multiplier at the 0.3 floor — which, when
    combined with the 0.6 confidence cap, produced 0.18 effective confidence (below
    the 0.3 surfacing threshold), silently suppressing all opportunity predictions.

Fix (this iteration):
    1. _backfill_automated_sender_tags() — runs on every BehavioralAccuracyTracker
       __init__, not just during migration.  For each untagged resolved-inaccurate
       opportunity/reminder prediction it tries:
         a. supporting_signals['contact_email'] (works for post-PR #190 predictions)
         b. Regex-extract email from description string (works for pre-PR #190
            predictions that only carry the address in the human-readable text)
       Any automated-sender match is tagged resolution_reason='automated_sender_fast_path'.

    2. _get_resolution_reason() description fallback — same two-strategy pattern
       applied when new predictions are being resolved, so future predictions created
       without supporting_signals are also correctly tagged at resolution time.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


# ============================================================================
# Helpers
# ============================================================================


def _insert_prediction(
    conn,
    pred_id: str,
    description: str,
    supporting_signals: str = "{}",
    was_accurate: int = 0,
    resolved_at: str | None = None,
    resolution_reason: str | None = None,
    was_surfaced: int = 1,
    prediction_type: str = "opportunity",
) -> None:
    """Insert a prediction row into the user_model database for testing."""
    created_at = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    if resolved_at is None:
        resolved_at = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()

    conn.execute(
        """INSERT INTO predictions
           (id, prediction_type, description, confidence, confidence_gate,
            time_horizon, suggested_action, supporting_signals,
            was_surfaced, created_at, was_accurate, resolved_at, resolution_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pred_id,
            prediction_type,
            description,
            0.50,
            "suggest",
            "7_days",
            "Consider reaching out",
            supporting_signals,
            was_surfaced,
            created_at,
            was_accurate,
            resolved_at,
            resolution_reason,
        ),
    )


# ============================================================================
# _backfill_automated_sender_tags() — description-fallback path
# ============================================================================


def test_backfill_tags_old_prediction_no_signals(db):
    """Predictions with no supporting_signals are tagged via description regex fallback.

    This is the primary scenario: predictions created before PR #190 only carry
    the contact address in their description string.  The migration guard previously
    skipped them because signals['contact_email'] was empty.  The new backfill
    method extracts the email from the description text and tags the row.
    """
    pred_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        _insert_prediction(
            conn,
            pred_id,
            description=(
                "It's been 45 days since you last contacted noreply@company.com "
                "(you usually connect every ~14 days)"
            ),
            supporting_signals="[]",  # Old empty-list format — no contact_email
        )

    # Instantiate tracker — backfill runs in __init__
    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["resolution_reason"] == "automated_sender_fast_path", (
        "Prediction with noreply address in description should be tagged via description fallback"
    )


def test_backfill_tags_old_prediction_null_signals(db):
    """Predictions with NULL supporting_signals are tagged via description regex fallback."""
    pred_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        _insert_prediction(
            conn,
            pred_id,
            description=(
                "It's been 30 days since you last contacted "
                "newsletter@marketing-platform.com (avg gap ~20 days)"
            ),
            supporting_signals="null",  # NULL-equivalent JSON
        )

    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["resolution_reason"] == "automated_sender_fast_path", (
        "Prediction with newsletter address in description should be tagged"
    )


def test_backfill_tags_reminder_prediction_with_description_fallback(db):
    """Reminder predictions (not just opportunity) are also tagged via description."""
    pred_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        _insert_prediction(
            conn,
            pred_id,
            description="Reply to notifications@platform.io about your account",
            supporting_signals="{}",  # Empty dict — no contact_email key
            prediction_type="reminder",
        )

    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["resolution_reason"] == "automated_sender_fast_path", (
        "Reminder prediction with automated sender in description should be tagged"
    )


def test_backfill_does_not_tag_human_contact_predictions(db):
    """Human contact predictions must NOT be tagged by the backfill.

    A legitimate "reach out to alice@gmail.com" prediction that the user didn't
    act on should remain untagged so it counts as a real INACCURATE signal.
    """
    pred_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        _insert_prediction(
            conn,
            pred_id,
            description=(
                "It's been 45 days since you last contacted alice@gmail.com "
                "(you usually connect every ~21 days)"
            ),
            supporting_signals="{}",
        )

    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["resolution_reason"] is None, (
        "Human contact predictions must not be tagged as automated_sender_fast_path"
    )


def test_backfill_skips_already_tagged_predictions(db):
    """Backfill is idempotent: already-tagged predictions are not double-updated."""
    pred_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        _insert_prediction(
            conn,
            pred_id,
            description="It's been 30 days since you last contacted noreply@example.com",
            supporting_signals="{}",
            resolution_reason="automated_sender_fast_path",  # Already tagged
        )

    # Should not raise or modify the already-correct row
    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["resolution_reason"] == "automated_sender_fast_path"


def test_backfill_skips_accurate_predictions(db):
    """Accurate predictions (was_accurate=1) should never be tagged as automated-sender.

    Even if an automated-sender email appears in the description of an accurate
    prediction, we must not tag it — accurate predictions represent real behavior.
    """
    pred_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        _insert_prediction(
            conn,
            pred_id,
            description="Reach out to notifications@platform.io (predicted correctly)",
            supporting_signals="{}",
            was_accurate=1,  # Marked accurate
        )

    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["resolution_reason"] is None, (
        "Accurate predictions should never be tagged as automated_sender_fast_path"
    )


def test_backfill_tags_multiple_predictions_in_bulk(db):
    """All matching untagged predictions in the database are tagged in one init call."""
    automated_descs = [
        "It's been 45 days since you last contacted noreply@company.com",
        "It's been 30 days since you last contacted newsletter@brand.com",
        "It's been 22 days since you last contacted mailer-daemon@host.com",
    ]
    human_desc = "It's been 45 days since you last contacted bob.smith@company.com"

    pred_ids_automated = []
    pred_id_human = str(uuid.uuid4())

    with db.get_connection("user_model") as conn:
        for desc in automated_descs:
            pid = str(uuid.uuid4())
            pred_ids_automated.append(pid)
            _insert_prediction(conn, pid, description=desc, supporting_signals="[]")
        _insert_prediction(conn, pred_id_human, description=human_desc, supporting_signals="[]")

    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        for pid in pred_ids_automated:
            row = conn.execute(
                "SELECT resolution_reason FROM predictions WHERE id = ?", (pid,)
            ).fetchone()
            assert row["resolution_reason"] == "automated_sender_fast_path", (
                f"Automated prediction {pid} should be tagged"
            )

        human_row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?", (pred_id_human,)
        ).fetchone()
        assert human_row["resolution_reason"] is None, (
            "Human prediction should not be tagged"
        )


def test_backfill_prefers_signals_over_description(db):
    """When supporting_signals has contact_email, it takes priority over description.

    If signals says alice@gmail.com but description says noreply@example.com, the
    signals value (human) should win — preventing a false-positive tag.
    """
    pred_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        _insert_prediction(
            conn,
            pred_id,
            description=(
                "It's been 45 days since you last contacted noreply@example.com "
                "(you usually connect every ~21 days)"
            ),
            # signals says it's a real human — trust this over description
            supporting_signals=json.dumps({
                "contact_email": "alice@gmail.com",
                "contact_name": "Alice",
            }),
        )

    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["resolution_reason"] is None, (
        "When signals says human contact, description noreply should not override it"
    )


def test_backfill_with_signals_contact_email_automated(db):
    """Backfill still works via signals path when contact_email is present and automated."""
    pred_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        _insert_prediction(
            conn,
            pred_id,
            description="Reach out to Brand Newsletter",  # No email in description
            supporting_signals=json.dumps({
                "contact_email": "newsletter@brand.com",
                "days_since_last_contact": 30,
            }),
        )

    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row["resolution_reason"] == "automated_sender_fast_path", (
        "Automated sender from signals should be tagged even when description has no email"
    )


# ============================================================================
# _get_resolution_reason() — description-fallback for future resolutions
# ============================================================================


def test_get_resolution_reason_description_fallback_automated(db):
    """_get_resolution_reason uses description fallback for predictions without signals.

    When a new prediction without supporting_signals is resolved (e.g. created by
    an older code path), _get_resolution_reason should still correctly tag it as
    automated_sender_fast_path via description regex parsing.
    """
    tracker = BehavioralAccuracyTracker(db)
    prediction = {
        "id": "test-id",
        "prediction_type": "opportunity",
        "description": (
            "It's been 33 days since you last contacted "
            "Fidelity.Investments@shareholderdocs.fidelity.com "
            "(you usually connect every ~21 days)"
        ),
        "supporting_signals": "[]",  # Old empty-list format
    }

    reason = tracker._get_resolution_reason(prediction, was_accurate=False)
    assert reason == "automated_sender_fast_path", (
        "_get_resolution_reason should detect automated sender via description fallback"
    )


def test_get_resolution_reason_description_fallback_human(db):
    """_get_resolution_reason description fallback does not tag human contacts."""
    tracker = BehavioralAccuracyTracker(db)
    prediction = {
        "id": "test-id",
        "prediction_type": "opportunity",
        "description": (
            "It's been 45 days since you last contacted alice@gmail.com "
            "(you usually connect every ~21 days)"
        ),
        "supporting_signals": "{}",
    }

    reason = tracker._get_resolution_reason(prediction, was_accurate=False)
    assert reason is None, (
        "_get_resolution_reason description fallback should not tag human contacts"
    )


def test_get_resolution_reason_accurate_always_none(db):
    """_get_resolution_reason always returns None for accurate predictions."""
    tracker = BehavioralAccuracyTracker(db)
    prediction = {
        "id": "test-id",
        "prediction_type": "opportunity",
        "description": "It's been 30 days since you contacted noreply@company.com",
        "supporting_signals": json.dumps({"contact_email": "noreply@company.com"}),
    }

    # Even though noreply@company.com is automated, accurate=True means real behavior
    reason = tracker._get_resolution_reason(prediction, was_accurate=True)
    assert reason is None, (
        "Accurate predictions always return None regardless of contact address"
    )


def test_get_resolution_reason_signals_override_description(db):
    """Signals contact_email takes priority over description email in _get_resolution_reason.

    When signals says alice@gmail.com but description says noreply@example.com, the
    human signals value wins (no tag).  This prevents a false-positive resolution_reason
    that would exclude a real behavioral signal from the accuracy denominator.
    """
    tracker = BehavioralAccuracyTracker(db)
    prediction = {
        "id": "test-id",
        "prediction_type": "opportunity",
        "description": (
            "It's been 45 days since you last contacted noreply@example.com"
        ),
        "supporting_signals": json.dumps({
            "contact_email": "alice@gmail.com",  # Real human overrides description
        }),
    }

    reason = tracker._get_resolution_reason(prediction, was_accurate=False)
    assert reason is None, (
        "Human contact_email in signals should override automated email in description"
    )
