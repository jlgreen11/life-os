"""
Tests for CC/BCC recipient support in reminder prediction accuracy inference.

Problem (pre-fix):
    _infer_reminder_accuracy() only checked 'to_addresses' when scanning outbound
    email.sent events to determine if the user followed up with a predicted contact.
    This caused false INACCURATE outcomes when the user replied-all to a group thread
    and the predicted contact was in CC or BCC rather than To.

    Example:
        Prediction: "Unreplied message from alice@example.com"
        User's reply-all: email.sent with to_addresses=["bob@example.com"],
                          cc_addresses=["alice@example.com"]
        Old result: INACCURATE (alice not in to_addresses → no match)
        Correct result: ACCURATE (alice is in cc_addresses → user did follow up)

    The same class of bug was fixed for opportunity predictions in PR #201.
    This iteration closes the identical gap in _infer_reminder_accuracy().

Fix:
    Replace the to_addresses-only check with the same all_recipients pattern used
    in _infer_opportunity_accuracy():
        - Collect addresses from to_addresses, cc_addresses, bcc_addresses
        - Fall back to legacy "to" string for older connector payloads
        - Match contact_email/contact_name against the full recipient set
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


# ============================================================================
# Helpers
# ============================================================================


def _insert_reminder_prediction(
    db,
    pred_id: str,
    contact_email: str,
    created_at: datetime,
    *,
    contact_name: str = "",
) -> None:
    """Insert a reminder prediction with the given contact into the test DB."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                f'Unreplied message from {contact_email}: "Hello" (2 hours ago)',
                0.6,
                "suggest",
                "24_hours",
                f"Reply to {contact_email}",
                json.dumps({
                    "contact_email": contact_email,
                    "contact_name": contact_name,
                }),
                1,
                created_at.isoformat(),
            ),
        )


def _insert_email_sent(
    db,
    *,
    timestamp: datetime,
    to_addresses: list[str] | None = None,
    cc_addresses: list[str] | None = None,
    bcc_addresses: list[str] | None = None,
    legacy_to: str | None = None,
) -> None:
    """Insert an email.sent event into the events DB."""
    payload: dict = {}
    if to_addresses is not None:
        payload["to_addresses"] = to_addresses
    if cc_addresses is not None:
        payload["cc_addresses"] = cc_addresses
    if bcc_addresses is not None:
        payload["bcc_addresses"] = bcc_addresses
    if legacy_to is not None:
        payload["to"] = legacy_to

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.sent",
                "protonmail",
                timestamp.isoformat(),
                "normal",
                json.dumps(payload),
                "{}",
            ),
        )


# ============================================================================
# CC recipient tests
# ============================================================================


@pytest.mark.asyncio
async def test_reminder_accurate_when_contact_in_cc(db):
    """Reminder is ACCURATE when user replied-all and predicted contact is in CC.

    This is the core regression test: before the fix, the contact in CC would
    not match the to_addresses check, causing a false INACCURATE outcome.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=4)
    contact = "alice@example.com"

    _insert_reminder_prediction(db, pred_id, contact, created_at)

    # User sent a reply-all — alice is in CC, not in To
    _insert_email_sent(
        db,
        timestamp=created_at + timedelta(hours=2),
        to_addresses=["bob@example.com"],
        cc_addresses=[contact],
    )

    stats = await tracker.run_inference_cycle()

    assert stats["marked_accurate"] >= 1, (
        "Expected reminder to be ACCURATE when contact was in CC of sent email"
    )

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None
    assert row["was_accurate"] == 1, "Contact in CC should yield ACCURATE reminder"
    assert row["resolved_at"] is not None


@pytest.mark.asyncio
async def test_reminder_accurate_when_contact_in_bcc(db):
    """Reminder is ACCURATE when user sent email with predicted contact in BCC.

    BCC recipients are invisible to other recipients but still count as a
    deliberate outreach by the user — the prediction was accurate.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=6)
    contact = "carol@example.com"

    _insert_reminder_prediction(db, pred_id, contact, created_at)

    # User sent email with carol silently BCC'd
    _insert_email_sent(
        db,
        timestamp=created_at + timedelta(hours=3),
        to_addresses=["dave@example.com"],
        bcc_addresses=[contact],
    )

    stats = await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None
    assert row["was_accurate"] == 1, "Contact in BCC should yield ACCURATE reminder"


@pytest.mark.asyncio
async def test_reminder_accurate_when_contact_in_to(db):
    """Reminder is ACCURATE when contact is in To field (baseline / regression guard).

    The original to_addresses-only path must still work correctly after the fix.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    contact = "eve@example.com"

    _insert_reminder_prediction(db, pred_id, contact, created_at)

    _insert_email_sent(
        db,
        timestamp=created_at + timedelta(hours=1),
        to_addresses=[contact],
    )

    await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None
    assert row["was_accurate"] == 1, "Contact in To should yield ACCURATE reminder"


@pytest.mark.asyncio
async def test_reminder_inaccurate_when_contact_not_in_any_field(db):
    """Reminder is INACCURATE when outbound email was sent to a different contact.

    Even with CC/BCC scanning, emails to completely unrelated contacts should
    NOT resolve a reminder as accurate.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    # Set created_at more than 48h ago so the timeout fires
    created_at = datetime.now(timezone.utc) - timedelta(hours=50)
    contact = "frank@example.com"

    _insert_reminder_prediction(db, pred_id, contact, created_at)

    # User sent email — but to someone completely unrelated
    _insert_email_sent(
        db,
        timestamp=created_at + timedelta(hours=2),
        to_addresses=["unrelated@other.com"],
        cc_addresses=["also-unrelated@other.com"],
    )

    await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None
    assert row["was_accurate"] == 0, (
        "Email to unrelated contact should leave reminder INACCURATE after 48h timeout"
    )


@pytest.mark.asyncio
async def test_reminder_accurate_via_legacy_to_field(db):
    """Reminder is ACCURATE when legacy 'to' string field contains the contact.

    Older iMessage connector events (and some third-party connectors) emit a
    plain 'to' string rather than a 'to_addresses' list.  The CC/BCC rewrite
    must preserve the legacy fallback path.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=5)
    contact = "grace@example.com"

    _insert_reminder_prediction(db, pred_id, contact, created_at)

    # Legacy payload: 'to' is a plain string, no to_addresses key at all
    _insert_email_sent(
        db,
        timestamp=created_at + timedelta(hours=1),
        legacy_to=contact,
    )

    await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None
    assert row["was_accurate"] == 1, (
        "Legacy 'to' string field should still yield ACCURATE reminder"
    )


@pytest.mark.asyncio
async def test_reminder_accurate_when_contact_in_multiple_recipient_fields(db):
    """All-fields check is correct when contact appears in both To and CC.

    Edge case: user sends a follow-up and accidentally includes the contact
    both in To and CC (e.g. a forwarded-then-replied thread).  Must still
    resolve as ACCURATE, not error out or double-count.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    contact = "henry@example.com"

    _insert_reminder_prediction(db, pred_id, contact, created_at)

    _insert_email_sent(
        db,
        timestamp=created_at + timedelta(hours=1),
        to_addresses=[contact],
        cc_addresses=[contact],  # duplicate — still fine
    )

    await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None
    assert row["was_accurate"] == 1, (
        "Contact in both To and CC should yield ACCURATE reminder (no double-count error)"
    )


@pytest.mark.asyncio
async def test_reminder_still_pending_within_window_no_matching_email(db):
    """Reminder stays unresolved (None) when still within 48h window and no matching email.

    If no outbound email to the contact has been sent within the 48h window,
    and the prediction is still fresh, _infer_reminder_accuracy should return
    None — not prematurely mark it inaccurate.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    # Only 10 hours old — well within the 48h window
    created_at = datetime.now(timezone.utc) - timedelta(hours=10)
    contact = "iris@example.com"

    _insert_reminder_prediction(db, pred_id, contact, created_at)

    # No outbound email to iris has been sent at all

    await tracker.run_inference_cycle()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None
    # Should remain unresolved — was_accurate=None and resolved_at=None
    assert row["was_accurate"] is None, (
        "Reminder within 48h window with no matching email should remain unresolved"
    )
    assert row["resolved_at"] is None
