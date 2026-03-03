"""
Tests for filtered prediction resolution_reason in BehavioralAccuracyTracker.

Verifies that filtered predictions (was_surfaced=0, user_response='filtered')
receive a proper resolution_reason when resolved, just like surfaced predictions.

Without this fix, filtered predictions resolved as inaccurate never received
'automated_sender_fast_path' tagging.  The prediction engine's
_get_accuracy_multiplier() excludes predictions with that resolution_reason from
accuracy calculations — but since filtered predictions lacked the field, they
counted against accuracy scores unfairly, artificially depressing the accuracy
multiplier for entire prediction types.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


def _pred_id() -> str:
    """Generate a unique prediction ID."""
    return str(uuid.uuid4())


def _ts(dt: datetime) -> str:
    """Convert a datetime to an ISO-8601 string."""
    return dt.isoformat()


def _insert_filtered_prediction(
    conn,
    pred_id: str,
    prediction_type: str,
    description: str,
    supporting_signals: dict | None,
    created_at: str,
):
    """Insert a filtered prediction (was_surfaced=0, user_response='filtered').

    These are predictions that were auto-filtered before being shown to the user.
    They are processed in the second loop of run_inference_cycle() to detect
    false negatives (filter mistakes).
    """
    signals_json = json.dumps(supporting_signals) if supporting_signals else None

    conn.execute(
        """INSERT INTO predictions
           (id, prediction_type, description, confidence, confidence_gate,
            suggested_action, supporting_signals, was_surfaced, created_at,
            user_response)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pred_id,
            prediction_type,
            description,
            0.5,
            "SUGGEST",
            f"Take action on {description}",
            signals_json,
            0,  # was_surfaced = False (filtered)
            created_at,
            "filtered",
        ),
    )


@pytest.mark.asyncio
async def test_filtered_automated_sender_gets_resolution_reason(db, user_model_store):
    """Filtered prediction with automated sender contact gets 'automated_sender_fast_path' resolution_reason."""
    tracker = BehavioralAccuracyTracker(db)

    pid = _pred_id()
    now = datetime.now(timezone.utc)
    # Must be 48+ hours old but < 7 days to be picked up by filtered loop
    created_at = _ts(now - timedelta(hours=72))

    with db.get_connection("user_model") as conn:
        _insert_filtered_prediction(
            conn,
            pred_id=pid,
            prediction_type="opportunity",
            description="Reconnect with noreply@marketing.example.com",
            supporting_signals={
                "contact_email": "noreply@marketing.example.com",
                "days_since_contact": 45,
            },
            created_at=created_at,
        )

    # Mock _infer_accuracy to return False (inaccurate — user did not act)
    with patch.object(tracker, "_infer_accuracy", new_callable=AsyncMock, return_value=False):
        stats = await tracker.run_inference_cycle()

    assert stats["marked_inaccurate"] >= 1
    assert stats["filtered"] >= 1

    # Verify the prediction has resolution_reason set
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response, resolution_reason FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()

    assert pred["was_accurate"] == 0
    # user_response should remain 'filtered' — provenance is preserved
    assert pred["user_response"] == "filtered"
    # This is the key assertion: resolution_reason must be set for automated senders
    assert pred["resolution_reason"] == "automated_sender_fast_path"


@pytest.mark.asyncio
async def test_filtered_real_contact_gets_null_resolution_reason(db, user_model_store):
    """Filtered prediction with real human contact gets NULL resolution_reason."""
    tracker = BehavioralAccuracyTracker(db)

    pid = _pred_id()
    now = datetime.now(timezone.utc)
    created_at = _ts(now - timedelta(hours=72))

    with db.get_connection("user_model") as conn:
        _insert_filtered_prediction(
            conn,
            pred_id=pid,
            prediction_type="reminder",
            description="Follow up with alice@example.com about the project",
            supporting_signals={
                "contact_email": "alice@example.com",
                "contact_name": "Alice",
            },
            created_at=created_at,
        )

    # Mock _infer_accuracy to return False (inaccurate — user did not act)
    with patch.object(tracker, "_infer_accuracy", new_callable=AsyncMock, return_value=False):
        stats = await tracker.run_inference_cycle()

    assert stats["marked_inaccurate"] >= 1
    assert stats["filtered"] >= 1

    # Verify resolution_reason is NULL for real human contacts
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response, resolution_reason FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()

    assert pred["was_accurate"] == 0
    assert pred["user_response"] == "filtered"
    # Real contact → no special resolution reason
    assert pred["resolution_reason"] is None


@pytest.mark.asyncio
async def test_filtered_accurate_prediction_gets_null_resolution_reason(db, user_model_store):
    """Filtered prediction resolved as accurate always gets NULL resolution_reason.

    Even if the contact is an automated sender, accurate predictions are always
    tagged as real behavioral signals (resolution_reason = NULL).  This matches
    the behavior of _get_resolution_reason() which returns None for accurate preds.
    """
    tracker = BehavioralAccuracyTracker(db)

    pid = _pred_id()
    now = datetime.now(timezone.utc)
    created_at = _ts(now - timedelta(hours=72))

    with db.get_connection("user_model") as conn:
        _insert_filtered_prediction(
            conn,
            pred_id=pid,
            prediction_type="opportunity",
            description="Reconnect with noreply@marketing.example.com",
            supporting_signals={
                "contact_email": "noreply@marketing.example.com",
                "days_since_contact": 45,
            },
            created_at=created_at,
        )

    # Mock _infer_accuracy to return True (accurate — user did act)
    with patch.object(tracker, "_infer_accuracy", new_callable=AsyncMock, return_value=True):
        stats = await tracker.run_inference_cycle()

    assert stats["marked_accurate"] >= 1

    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response, resolution_reason FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()

    assert pred["was_accurate"] == 1
    assert pred["user_response"] == "filtered"
    # Accurate predictions never get resolution_reason — they're real behavioral signals
    assert pred["resolution_reason"] is None


@pytest.mark.asyncio
async def test_filtered_description_fallback_for_automated_sender(db, user_model_store):
    """Filtered prediction without contact_email in signals falls back to description parsing.

    _get_resolution_reason() parses the description for an email address when
    supporting_signals lacks contact_email.  This handles predictions created
    before PR #190 added supporting_signals.
    """
    tracker = BehavioralAccuracyTracker(db)

    pid = _pred_id()
    now = datetime.now(timezone.utc)
    created_at = _ts(now - timedelta(hours=72))

    with db.get_connection("user_model") as conn:
        _insert_filtered_prediction(
            conn,
            pred_id=pid,
            prediction_type="opportunity",
            # Email embedded in description only, not in supporting_signals
            description="It's been 45 days since you last contacted noreply@automated.example.com",
            supporting_signals={"days_since_contact": 45},  # No contact_email key
            created_at=created_at,
        )

    with patch.object(tracker, "_infer_accuracy", new_callable=AsyncMock, return_value=False):
        stats = await tracker.run_inference_cycle()

    assert stats["marked_inaccurate"] >= 1

    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()

    # Should still detect automated sender via description fallback
    assert pred["resolution_reason"] == "automated_sender_fast_path"


@pytest.mark.asyncio
async def test_filtered_none_result_leaves_prediction_unresolved(db, user_model_store):
    """When _infer_accuracy returns None, filtered prediction stays unresolved."""
    tracker = BehavioralAccuracyTracker(db)

    pid = _pred_id()
    now = datetime.now(timezone.utc)
    created_at = _ts(now - timedelta(hours=72))

    with db.get_connection("user_model") as conn:
        _insert_filtered_prediction(
            conn,
            pred_id=pid,
            prediction_type="reminder",
            description="Follow up with alice@example.com",
            supporting_signals={"contact_email": "alice@example.com"},
            created_at=created_at,
        )

    # None means insufficient evidence — prediction should remain unresolved
    with patch.object(tracker, "_infer_accuracy", new_callable=AsyncMock, return_value=None):
        stats = await tracker.run_inference_cycle()

    assert stats["marked_accurate"] == 0
    assert stats["marked_inaccurate"] == 0

    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, resolution_reason FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()

    # Prediction should remain unresolved
    assert pred["was_accurate"] is None
    assert pred["resolved_at"] is None
    assert pred["resolution_reason"] is None
