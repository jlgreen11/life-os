"""
Tests for the filtered-prediction self-correction loop in BehavioralAccuracyTracker.

The behavioral accuracy tracker processes filtered predictions (was_surfaced=0,
user_response='filtered') in a second pass of run_inference_cycle() to detect
when the system's own filters incorrectly reject valuable predictions. This is
the prediction engine's self-correction mechanism:

    - If a prediction was filtered but the user took the action anyway,
      the filter was WRONG (false negative) → was_accurate = 1
    - If a prediction was filtered and the user did NOT take the action,
      the filter was RIGHT (true negative) → was_accurate = 0

These tests cover:
    1. Filtered prediction resolved as accurate when user acts (false negative detection)
    2. Filtered prediction resolved as inaccurate when user does not act (true negative)
    3. Time window boundaries (48h minimum, 7d maximum)
    4. Automated sender tagging via resolution_reason
    5. Stats dict correctly separates surfaced vs. filtered counters
    6. Filtered conflict prediction resolved via calendar events
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

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
    *,
    confidence: float = 0.5,
    confidence_gate: str = "SUGGEST",
    suggested_action: str | None = None,
    filter_reason: str | None = None,
):
    """Insert a filtered prediction (was_surfaced=0, user_response='filtered').

    Filtered predictions are those that were auto-filtered before being shown to
    the user. They are processed in the second loop of run_inference_cycle() to
    detect false negatives (filter mistakes).

    Args:
        conn: Database connection for the user_model database.
        pred_id: Unique prediction ID.
        prediction_type: Type of prediction (reminder, conflict, opportunity, etc.).
        description: Human-readable description of the prediction.
        supporting_signals: Dict of machine-readable context (contact info, event IDs, etc.).
        created_at: ISO-8601 timestamp of when the prediction was created.
        confidence: Prediction confidence score (default 0.5).
        confidence_gate: Confidence gate level (default SUGGEST).
        suggested_action: What action was suggested. Auto-generated if None.
        filter_reason: Why the prediction was filtered (optional).
    """
    signals_json = json.dumps(supporting_signals) if supporting_signals else None
    if suggested_action is None:
        suggested_action = f"Take action on {description}"

    conn.execute(
        """INSERT INTO predictions
           (id, prediction_type, description, confidence, confidence_gate,
            suggested_action, supporting_signals, was_surfaced, created_at,
            user_response, filter_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pred_id,
            prediction_type,
            description,
            confidence,
            confidence_gate,
            suggested_action,
            signals_json,
            0,  # was_surfaced = False (filtered)
            created_at,
            "filtered",
            filter_reason,
        ),
    )


def _insert_event(conn, event_type: str, source: str, timestamp: str, payload: dict):
    """Insert an event into the events database.

    Args:
        conn: Database connection for the events database.
        event_type: Event type string (e.g. 'email.sent', 'calendar.event.updated').
        source: Source connector name.
        timestamp: ISO-8601 timestamp of the event.
        payload: Event payload dict.
    """
    conn.execute(
        """INSERT INTO events (id, type, source, timestamp, payload)
           VALUES (?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            event_type,
            source,
            timestamp,
            json.dumps(payload),
        ),
    )


# ---------------------------------------------------------------------------
# Test 1: Filtered prediction — user took action → was_accurate = 1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filtered_prediction_user_took_action_marked_accurate(db, user_model_store):
    """When a filtered prediction's action was taken by the user anyway, mark it accurate.

    This is the core false-negative detection path: the filter rejected a
    prediction, but the user went ahead and did the predicted thing. This means
    the filter was WRONG — the prediction was actually valuable.
    """
    tracker = BehavioralAccuracyTracker(db)

    pid = _pred_id()
    now = datetime.now(timezone.utc)
    # 3 days ago — within the 48h–7d filtered window
    created_at = _ts(now - timedelta(days=3))

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

    # User sends an email to Alice — the predicted follow-up actually happened
    with db.get_connection("events") as conn:
        _insert_event(
            conn,
            event_type="email.sent",
            source="proton_mail_connector",
            timestamp=_ts(now - timedelta(days=2, hours=20)),
            payload={
                "to_addresses": ["alice@example.com"],
                "subject": "Re: Project update",
                "body": "Hey Alice, following up on our discussion...",
            },
        )

    stats = await tracker.run_inference_cycle()

    # Filter was WRONG — user took action → accurate
    assert stats["marked_accurate"] >= 1
    assert stats["filtered"] >= 1

    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()

    assert pred["was_accurate"] == 1
    # user_response stays 'filtered' to preserve provenance (not overwritten to 'inferred')
    assert pred["user_response"] == "filtered"
    assert pred["resolved_at"] is not None


# ---------------------------------------------------------------------------
# Test 2: Filtered prediction — user did NOT act → was_accurate = 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filtered_prediction_user_did_not_act_marked_inaccurate(db, user_model_store):
    """When a filtered prediction's action was NOT taken, mark it inaccurate.

    This is the true-negative path: the filter rejected a prediction, and
    the user indeed did NOT take the action. The filter was RIGHT.
    """
    tracker = BehavioralAccuracyTracker(db)

    pid = _pred_id()
    now = datetime.now(timezone.utc)
    # 3 days ago — within the 48h–7d filtered window, and past the 48h
    # inaction threshold for reminder predictions
    created_at = _ts(now - timedelta(days=3))

    with db.get_connection("user_model") as conn:
        _insert_filtered_prediction(
            conn,
            pred_id=pid,
            prediction_type="reminder",
            description="Follow up with bob@example.com about the report",
            supporting_signals={
                "contact_email": "bob@example.com",
                "contact_name": "Bob",
            },
            created_at=created_at,
        )

    # No email.sent or message.sent events to Bob — user did NOT act

    stats = await tracker.run_inference_cycle()

    # Filter was RIGHT — user didn't act → inaccurate
    assert stats["marked_inaccurate"] >= 1
    assert stats["filtered"] >= 1

    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()

    assert pred["was_accurate"] == 0
    assert pred["user_response"] == "filtered"
    assert pred["resolved_at"] is not None


# ---------------------------------------------------------------------------
# Test 3: Filtered predictions outside the 48h–7d window are NOT processed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filtered_prediction_outside_window_not_processed(db, user_model_store):
    """Filtered predictions outside the 48h–7d window should not be processed.

    The filtered prediction SQL query uses:
        created_at > (now - 7 days)   -- must be newer than 7 days
        created_at < (now - 48 hours) -- must be older than 48 hours

    A prediction from 8 days ago (too old) and one from 24 hours ago (too new)
    should both remain unresolved (was_accurate = NULL).
    """
    tracker = BehavioralAccuracyTracker(db)

    now = datetime.now(timezone.utc)
    pid_old = _pred_id()
    pid_new = _pred_id()

    with db.get_connection("user_model") as conn:
        # Too old — 8 days ago (exceeds the 7-day maximum)
        _insert_filtered_prediction(
            conn,
            pred_id=pid_old,
            prediction_type="reminder",
            description="Follow up with charlie@example.com",
            supporting_signals={
                "contact_email": "charlie@example.com",
                "contact_name": "Charlie",
            },
            created_at=_ts(now - timedelta(days=8)),
        )

        # Too new — 24 hours ago (below the 48-hour minimum)
        _insert_filtered_prediction(
            conn,
            pred_id=pid_new,
            prediction_type="reminder",
            description="Follow up with diana@example.com",
            supporting_signals={
                "contact_email": "diana@example.com",
                "contact_name": "Diana",
            },
            created_at=_ts(now - timedelta(hours=24)),
        )

    stats = await tracker.run_inference_cycle()

    # Neither prediction should have been processed
    with db.get_connection("user_model") as conn:
        pred_old = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pid_old,),
        ).fetchone()
        pred_new = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pid_new,),
        ).fetchone()

    assert pred_old["was_accurate"] is None, "8-day-old prediction should not be processed"
    assert pred_old["resolved_at"] is None
    assert pred_new["was_accurate"] is None, "24-hour-old prediction should not be processed"
    assert pred_new["resolved_at"] is None


# ---------------------------------------------------------------------------
# Test 4: Automated sender gets 'automated_sender_fast_path' resolution_reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filtered_automated_sender_gets_resolution_reason_tag(db, user_model_store):
    """Filtered prediction for an automated sender gets resolution_reason='automated_sender_fast_path'.

    Without this tag, the prediction would pollute the accuracy denominator in
    _get_accuracy_multiplier(), artificially depressing confidence for the entire
    prediction type. The resolution_reason tag allows the prediction engine to
    exclude these structurally-unfulfillable predictions from accuracy calculations.
    """
    tracker = BehavioralAccuracyTracker(db)

    pid = _pred_id()
    now = datetime.now(timezone.utc)
    # 3 days ago — within the filtered window
    created_at = _ts(now - timedelta(days=3))

    with db.get_connection("user_model") as conn:
        _insert_filtered_prediction(
            conn,
            pred_id=pid,
            prediction_type="reminder",
            description="Reply to noreply@company.com about subscription renewal",
            supporting_signals={
                "contact_email": "noreply@company.com",
                "contact_name": "Company Notifications",
            },
            created_at=created_at,
        )

    stats = await tracker.run_inference_cycle()

    # Automated sender → inaccurate (user will never "reply" to noreply@)
    assert stats["marked_inaccurate"] >= 1
    assert stats["filtered"] >= 1

    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolution_reason, user_response FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()

    assert pred["was_accurate"] == 0
    assert pred["resolution_reason"] == "automated_sender_fast_path"
    # Provenance preserved — still 'filtered', not overwritten to 'inferred'
    assert pred["user_response"] == "filtered"


# ---------------------------------------------------------------------------
# Test 5: Stats dict separates 'surfaced' and 'filtered' counters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filtered_prediction_stats_counted_separately(db, user_model_store):
    """Run a cycle with both surfaced and filtered predictions; verify separate counters.

    The stats dict returned by run_inference_cycle() must have:
    - 'surfaced': count of surfaced predictions processed
    - 'filtered': count of filtered predictions processed
    - 'marked_accurate' / 'marked_inaccurate': shared totals

    Filtered predictions must increment 'filtered', not 'surfaced'.
    """
    tracker = BehavioralAccuracyTracker(db)

    now = datetime.now(timezone.utc)
    pid_surfaced = _pred_id()
    pid_filtered = _pred_id()

    with db.get_connection("user_model") as conn:
        # Surfaced prediction — 3 days ago, user ignored → inaccurate
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pid_surfaced,
                "reminder",
                "Reply to surfaced-contact@example.com",
                0.70,
                "DEFAULT",
                "Send message",
                json.dumps({"contact_email": "surfaced-contact@example.com"}),
                1,  # was_surfaced = True (surfaced)
                _ts(now - timedelta(days=3)),
            ),
        )

        # Filtered prediction — 3 days ago, user also ignored → inaccurate
        _insert_filtered_prediction(
            conn,
            pred_id=pid_filtered,
            prediction_type="reminder",
            description="Reply to filtered-contact@example.com",
            supporting_signals={
                "contact_email": "filtered-contact@example.com",
            },
            created_at=_ts(now - timedelta(days=3)),
        )

    stats = await tracker.run_inference_cycle()

    # Both resolved as inaccurate (no matching events)
    assert stats["marked_inaccurate"] == 2

    # Counters must be separate
    assert stats["surfaced"] == 1, "Surfaced prediction should increment 'surfaced' counter"
    assert stats["filtered"] == 1, "Filtered prediction should increment 'filtered' counter"

    # Sanity check: the accurate counter should be zero
    assert stats["marked_accurate"] == 0


# ---------------------------------------------------------------------------
# Test 6: Filtered conflict prediction resolved as accurate from calendar
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filtered_conflict_prediction_accuracy_from_calendar(db, user_model_store):
    """Filtered calendar.conflict prediction is resolved as accurate when the conflict was real.

    Insert a filtered conflict prediction referencing two overlapping calendar events.
    Insert a calendar.event.updated event showing the user resolved the conflict.
    Run the cycle and verify the conflict prediction is resolved as accurate —
    the filter was WRONG to suppress it (the conflict was real and actionable).
    """
    tracker = BehavioralAccuracyTracker(db)

    now = datetime.now(timezone.utc)
    pid = _pred_id()
    event1_id = "event-standup-789"
    event2_id = "event-dentist-012"

    # 3 days ago — within the filtered 48h–7d window
    created_at = _ts(now - timedelta(days=3))

    with db.get_connection("user_model") as conn:
        _insert_filtered_prediction(
            conn,
            pred_id=pid,
            prediction_type="conflict",
            description="Calendar conflict: Daily standup overlaps with dentist appointment",
            supporting_signals={
                "conflicting_event_ids": [event1_id, event2_id],
            },
            created_at=created_at,
        )

    # User resolves the conflict by rescheduling one of the events
    with db.get_connection("events") as conn:
        _insert_event(
            conn,
            event_type="calendar.event.updated",
            source="caldav_connector",
            timestamp=_ts(now - timedelta(days=2, hours=22)),
            payload={
                "event_id": event1_id,
                "start": "2026-02-28T15:00:00Z",
                "end": "2026-02-28T15:30:00Z",
            },
        )

    stats = await tracker.run_inference_cycle()

    # Filter was WRONG — conflict was real and user resolved it → accurate
    assert stats["marked_accurate"] >= 1
    assert stats["filtered"] >= 1

    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            (pid,),
        ).fetchone()

    assert pred["was_accurate"] == 1
    assert pred["user_response"] == "filtered"
    assert pred["resolved_at"] is not None
