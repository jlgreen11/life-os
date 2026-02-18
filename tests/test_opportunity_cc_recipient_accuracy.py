"""
Tests for CC/BCC recipient matching in opportunity prediction accuracy inference.

Problem (pre-fix):
    _infer_opportunity_accuracy() only checked to_addresses when determining
    if the user contacted a predicted person. This meant that any email where
    the predicted contact was in the CC or BCC field was silently counted as
    "no contact made" and the prediction was marked INACCURATE after 7 days.

    This artificially depressed the opportunity accuracy rate (reported as 19%)
    because group emails and reply-all scenarios often CC the relevant contact
    rather than placing them in the To field.

    Example failure case:
        Prediction: "Reach out to alice@example.com — it's been 30 days"
        User sends: Reply-all to a group thread where alice is CC'd
        Old behavior: Prediction marked INACCURATE (alice not in to_addresses)
        Correct behavior: Prediction should be marked ACCURATE

Fix (this iteration):
    Expanded recipient matching in _infer_opportunity_accuracy() to include
    cc_addresses and bcc_addresses fields from the email payload, in addition
    to to_addresses. The 'contact_name' check was also extended to search
    all recipient fields.

    Both ProtonMail and Gmail connectors have always included cc_addresses
    in their email payloads (see connectors/proton_mail/connector.py:253 and
    connectors/google/connector.py:284), so historical data can be reprocessed
    without any connector changes.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


# ============================================================================
# Helpers
# ============================================================================


def _make_tracker(db):
    """Create a BehavioralAccuracyTracker with a real test database."""
    return BehavioralAccuracyTracker(db=db)


def _insert_email_sent_event(db, to_addresses, cc_addresses=None, bcc_addresses=None,
                              timestamp=None):
    """Insert an email.sent event into the events table.

    Args:
        db: DatabaseManager with an events database.
        to_addresses: List of To: recipients.
        cc_addresses: List of CC: recipients (or None for empty list).
        bcc_addresses: List of BCC: recipients (or None for empty list).
        timestamp: Event timestamp (defaults to 1 hour ago).

    Returns:
        The event id.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc) - timedelta(hours=1)

    payload = {
        "message_id": str(uuid.uuid4()),
        "channel": "proton_mail",
        "direction": "outbound",
        "from_address": "me@example.com",
        "to_addresses": to_addresses,
        "cc_addresses": cc_addresses or [],
        "bcc_addresses": bcc_addresses or [],
        "subject": "Re: Group thread",
        "body_plain": "test body",
    }

    event_id = str(uuid.uuid4())
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, 'email.sent', 'proton_mail', ?, 'normal', ?, '{}')""",
            (event_id, timestamp.isoformat(), json.dumps(payload)),
        )
    return event_id


def _make_prediction(description, created_at=None, signals=None):
    """Build a minimal prediction dict for testing."""
    if created_at is None:
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
    return {
        "id": str(uuid.uuid4()),
        "type": "opportunity",
        "description": description,
        "created_at": created_at.isoformat(),
        "signals": json.dumps(signals or {}),
    }


# ============================================================================
# CC recipient matching tests
# ============================================================================


@pytest.mark.asyncio
async def test_contact_in_cc_marks_prediction_accurate(db):
    """Contact in CC field counts as reaching out → prediction should be ACCURATE.

    This is the primary regression test. Before the fix, a contact in CC would
    be silently ignored and the prediction would be marked INACCURATE after 7 days.
    """
    tracker = _make_tracker(db)

    # User sends email where alice is CC'd (not in To:)
    _insert_email_sent_event(
        db,
        to_addresses=["bob@example.com"],
        cc_addresses=["alice@example.com"],
    )

    created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    prediction = _make_prediction(
        "Reach out to alice@example.com — it's been 30 days",
        created_at=created_at,
        signals={"contact_email": "alice@example.com"},
    )

    result = await tracker._infer_opportunity_accuracy(
        prediction, {"contact_email": "alice@example.com"}, created_at
    )

    assert result is True, (
        "Contact in CC field should count as a successful reach-out. "
        f"Got: {result!r}"
    )


@pytest.mark.asyncio
async def test_contact_in_bcc_marks_prediction_accurate(db):
    """Contact in BCC field also counts as reaching out → prediction should be ACCURATE.

    BCC is used for blind copies but still represents direct contact.
    """
    tracker = _make_tracker(db)

    _insert_email_sent_event(
        db,
        to_addresses=["group@example.com"],
        bcc_addresses=["alice@example.com"],
    )

    created_at = datetime.now(timezone.utc) - timedelta(hours=3)

    result = await tracker._infer_opportunity_accuracy(
        _make_prediction("Reach out to alice@example.com", created_at=created_at,
                         signals={"contact_email": "alice@example.com"}),
        {"contact_email": "alice@example.com"},
        created_at,
    )

    assert result is True, f"Contact in BCC should mark prediction accurate. Got: {result!r}"


@pytest.mark.asyncio
async def test_contact_in_to_still_marks_prediction_accurate(db):
    """Existing behavior: contact in To field must still be detected correctly."""
    tracker = _make_tracker(db)

    _insert_email_sent_event(
        db,
        to_addresses=["alice@example.com"],
        cc_addresses=["other@example.com"],
    )

    created_at = datetime.now(timezone.utc) - timedelta(hours=3)

    result = await tracker._infer_opportunity_accuracy(
        _make_prediction("Reach out to alice@example.com", created_at=created_at,
                         signals={"contact_email": "alice@example.com"}),
        {"contact_email": "alice@example.com"},
        created_at,
    )

    assert result is True, f"Contact in To field must still work. Got: {result!r}"


@pytest.mark.asyncio
async def test_unrelated_email_does_not_trigger_accurate(db):
    """Email to unrelated contacts should NOT mark an opportunity prediction accurate."""
    tracker = _make_tracker(db)

    _insert_email_sent_event(
        db,
        to_addresses=["bob@example.com"],
        cc_addresses=["carol@example.com"],
    )

    created_at = datetime.now(timezone.utc) - timedelta(hours=3)

    # Still within 7-day window → should be None (undecided), not True
    result = await tracker._infer_opportunity_accuracy(
        _make_prediction("Reach out to alice@example.com", created_at=created_at,
                         signals={"contact_email": "alice@example.com"}),
        {"contact_email": "alice@example.com"},
        created_at,
    )

    assert result is None, (
        "Email to unrelated parties should not mark prediction accurate. "
        f"Got: {result!r}"
    )


@pytest.mark.asyncio
async def test_expired_window_no_contact_marks_inaccurate(db):
    """After 7-day window with no contact in any field, prediction is INACCURATE."""
    tracker = _make_tracker(db)

    # No sent events for alice at all
    created_at = datetime.now(timezone.utc) - timedelta(days=8)

    result = await tracker._infer_opportunity_accuracy(
        _make_prediction("Reach out to alice@example.com", created_at=created_at,
                         signals={"contact_email": "alice@example.com"}),
        {"contact_email": "alice@example.com"},
        created_at,
    )

    assert result is False, (
        "After 7-day window with no contact, prediction should be INACCURATE. "
        f"Got: {result!r}"
    )


@pytest.mark.asyncio
async def test_contact_name_in_cc_marks_accurate(db):
    """Contact name matched against all recipient fields (not just to_addresses).

    When contact_email is unavailable but contact_name is known, the name
    should be searched across to_addresses, cc_addresses, and bcc_addresses.
    """
    tracker = _make_tracker(db)

    # The CC field has 'alice' in the address
    _insert_email_sent_event(
        db,
        to_addresses=["bob@example.com"],
        cc_addresses=["alice.smith@example.com"],
    )

    created_at = datetime.now(timezone.utc) - timedelta(hours=3)

    result = await tracker._infer_opportunity_accuracy(
        _make_prediction(
            "Reach out to Alice — it's been 45 days",
            created_at=created_at,
            signals={"contact_name": "Alice"},
        ),
        {"contact_name": "Alice"},
        created_at,
    )

    assert result is True, (
        "Contact name in CC address should mark prediction accurate. "
        f"Got: {result!r}"
    )


@pytest.mark.asyncio
async def test_empty_cc_bcc_falls_back_to_to_field(db):
    """When cc_addresses and bcc_addresses are empty/absent, to_addresses is still checked."""
    tracker = _make_tracker(db)

    # Email with no CC or BCC fields in payload at all
    payload = {
        "message_id": str(uuid.uuid4()),
        "channel": "imessage",
        "direction": "outbound",
        "from_address": "me@example.com",
        "to_addresses": ["alice@example.com"],
        # cc_addresses and bcc_addresses are absent
        "subject": "Hi",
        "body_plain": "Hey alice",
    }
    event_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc) - timedelta(hours=1)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, 'email.sent', 'imessage', ?, 'normal', ?, '{}')""",
            (event_id, ts.isoformat(), json.dumps(payload)),
        )

    created_at = datetime.now(timezone.utc) - timedelta(hours=3)

    result = await tracker._infer_opportunity_accuracy(
        _make_prediction("Reach out to alice@example.com", created_at=created_at,
                         signals={"contact_email": "alice@example.com"}),
        {"contact_email": "alice@example.com"},
        created_at,
    )

    assert result is True, (
        "Should still find contact in to_addresses when CC/BCC absent. "
        f"Got: {result!r}"
    )


@pytest.mark.asyncio
async def test_contact_in_multiple_recipient_fields_accurate(db):
    """Contact appearing in both to_addresses and cc_addresses: still ACCURATE."""
    tracker = _make_tracker(db)

    _insert_email_sent_event(
        db,
        to_addresses=["alice@example.com", "bob@example.com"],
        cc_addresses=["alice@example.com", "carol@example.com"],
    )

    created_at = datetime.now(timezone.utc) - timedelta(hours=3)

    result = await tracker._infer_opportunity_accuracy(
        _make_prediction("Reach out to alice@example.com", created_at=created_at,
                         signals={"contact_email": "alice@example.com"}),
        {"contact_email": "alice@example.com"},
        created_at,
    )

    assert result is True, f"Contact in multiple fields: should be ACCURATE. Got: {result!r}"
