"""
Tests for prediction follow-up stale-data fallback.

When a connector is down (e.g. Google auth failure), the standard 24h lookback
window yields 0 inbound emails.  The stale-data fallback extends the lookback
to cover the last active email period (up to 14 days) so accumulated unreplied
emails still surface as predictions.

Also tests the reduced grace period for priority contacts (1h vs 3h).
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine


def _insert_email(event_store, *, from_address, subject, message_id, hours_ago, snippet=""):
    """Helper to insert an email.received event at a specific age."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": ts,
        "payload": {
            "from_address": from_address,
            "subject": subject,
            "snippet": snippet,
            "message_id": message_id,
        },
        "metadata": {},
    })


def _insert_sent_email(event_store, *, in_reply_to, hours_ago):
    """Helper to insert an email.sent event (reply)."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.sent",
        "source": "google",
        "timestamp": ts,
        "payload": {
            "in_reply_to": in_reply_to,
        },
        "metadata": {},
    })


# -------------------------------------------------------------------------
# Stale-data fallback activation
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_activates_when_no_recent_emails_and_old_data_exists(
    db, event_store, user_model_store
):
    """When no emails in 24h but emails exist 5 days ago, fallback should activate
    and surface unreplied ones."""
    engine = PredictionEngine(db, user_model_store)
    # Consume the first-run 72h window so we test the subsequent-cycle path
    engine._first_follow_up_run = False

    # Insert emails from 5 days ago (120 hours) — well outside the 24h window
    _insert_email(
        event_store,
        from_address="colleague@work.com",
        subject="Project update needed",
        message_id="msg-stale-1",
        hours_ago=120,
    )
    _insert_email(
        event_store,
        from_address="manager@work.com",
        subject="Review my proposal",
        message_id="msg-stale-2",
        hours_ago=118,
    )

    predictions = await engine._check_follow_up_needs({})

    # The fallback should have found these emails
    contacts = {p.relevant_contacts[0] for p in predictions}
    assert "colleague@work.com" in contacts, (
        "Stale fallback should surface unreplied email from 5 days ago"
    )
    assert "manager@work.com" in contacts, (
        "Stale fallback should surface all unreplied emails in last active period"
    )


@pytest.mark.asyncio
async def test_fallback_does_not_activate_when_recent_emails_exist(
    db, event_store, user_model_store
):
    """When there ARE emails within 24h, the fallback should NOT activate."""
    engine = PredictionEngine(db, user_model_store)
    engine._first_follow_up_run = False

    # Insert a recent email (6 hours ago — inside 24h window)
    _insert_email(
        event_store,
        from_address="recent@work.com",
        subject="Quick question",
        message_id="msg-recent-1",
        hours_ago=6,
    )

    # Also insert an old email (10 days ago)
    _insert_email(
        event_store,
        from_address="old@work.com",
        subject="Ancient thread",
        message_id="msg-old-1",
        hours_ago=240,
    )

    predictions = await engine._check_follow_up_needs({})

    contacts = {p.relevant_contacts[0] for p in predictions}
    # Recent email should be found normally
    assert "recent@work.com" in contacts
    # Old email should NOT be found (standard window works, no fallback needed)
    assert "old@work.com" not in contacts


# -------------------------------------------------------------------------
# 14-day cap enforcement
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_14_day_cap_enforced(db, event_store, user_model_store):
    """Emails older than 14 days should NOT surface even in fallback mode."""
    engine = PredictionEngine(db, user_model_store)
    engine._first_follow_up_run = False

    # Insert an email from 16 days ago — beyond the 14-day cap
    _insert_email(
        event_store,
        from_address="ancient@work.com",
        subject="Very old thread",
        message_id="msg-ancient-1",
        hours_ago=16 * 24,
    )

    predictions = await engine._check_follow_up_needs({})

    contacts = {p.relevant_contacts[0] for p in predictions}
    assert "ancient@work.com" not in contacts, (
        "Emails older than 14 days should not surface even with fallback"
    )


@pytest.mark.asyncio
async def test_fallback_emails_within_14_day_cap_surface(db, event_store, user_model_store):
    """Emails within the 14-day cap should surface in fallback mode."""
    engine = PredictionEngine(db, user_model_store)
    engine._first_follow_up_run = False

    # Insert an email from 10 days ago — within 14-day cap
    _insert_email(
        event_store,
        from_address="tenday@work.com",
        subject="Important request",
        message_id="msg-10day-1",
        hours_ago=10 * 24,
    )

    predictions = await engine._check_follow_up_needs({})

    contacts = {p.relevant_contacts[0] for p in predictions}
    assert "tenday@work.com" in contacts, (
        "Emails within 14-day cap should surface via fallback"
    )


# -------------------------------------------------------------------------
# Dedup still works in fallback mode
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_works_in_fallback_mode(db, event_store, user_model_store):
    """Already-predicted messages should not re-surface in fallback mode."""
    engine = PredictionEngine(db, user_model_store)
    engine._first_follow_up_run = False

    # Insert email from 5 days ago
    _insert_email(
        event_store,
        from_address="boss@work.com",
        subject="Urgent review",
        message_id="msg-dedup-fallback-1",
        hours_ago=120,
    )

    # First call — should generate a prediction
    first_preds = await engine._check_follow_up_needs({})
    assert len(first_preds) >= 1, "First fallback call should generate prediction"

    # Store the prediction in the DB (simulate generate_predictions storing it)
    with db.get_connection("user_model") as conn:
        for pred in first_preds:
            conn.execute(
                """INSERT INTO predictions (id, prediction_type, description,
                   confidence, confidence_gate, time_horizon, supporting_signals,
                   created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pred.id,
                    pred.prediction_type,
                    pred.description,
                    pred.confidence,
                    pred.confidence_gate.value if hasattr(pred.confidence_gate, "value") else str(pred.confidence_gate),
                    pred.time_horizon,
                    json.dumps(pred.supporting_signals),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    # Second call — same message should NOT generate again
    second_preds = await engine._check_follow_up_needs({})
    second_msg_ids = {
        p.supporting_signals.get("message_id") for p in second_preds
    }
    assert "msg-dedup-fallback-1" not in second_msg_ids, (
        "Deduplication should prevent re-predicting in fallback mode"
    )


# -------------------------------------------------------------------------
# Marketing emails still filtered in fallback mode
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_marketing_filtered_in_fallback_mode(db, event_store, user_model_store):
    """Marketing/automated emails should still be filtered in fallback mode."""
    engine = PredictionEngine(db, user_model_store)
    engine._first_follow_up_run = False

    # Insert a marketing email from 5 days ago
    _insert_email(
        event_store,
        from_address="newsletter@marketing.example.com",
        subject="Weekly deals!",
        message_id="msg-marketing-fallback-1",
        hours_ago=120,
        snippet="Click here to unsubscribe from our mailing list",
    )

    # Also insert a legitimate email so fallback activates
    _insert_email(
        event_store,
        from_address="real-person@work.com",
        subject="Can you help?",
        message_id="msg-real-fallback-1",
        hours_ago=119,
    )

    predictions = await engine._check_follow_up_needs({})

    contacts = {p.relevant_contacts[0] for p in predictions}
    assert "newsletter@marketing.example.com" not in contacts, (
        "Marketing emails should be filtered even in fallback mode"
    )
    assert "real-person@work.com" in contacts, (
        "Legitimate emails should still surface in fallback mode"
    )


# -------------------------------------------------------------------------
# Replied-to emails excluded in fallback mode
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replied_emails_excluded_in_fallback_mode(db, event_store, user_model_store):
    """Emails the user already replied to should not surface in fallback mode."""
    engine = PredictionEngine(db, user_model_store)
    engine._first_follow_up_run = False

    # Insert an email from 5 days ago
    _insert_email(
        event_store,
        from_address="colleague@work.com",
        subject="Design feedback",
        message_id="msg-replied-fallback-1",
        hours_ago=120,
    )

    # Insert a reply to that email
    _insert_sent_email(
        event_store,
        in_reply_to="msg-replied-fallback-1",
        hours_ago=119,
    )

    # Insert an unreplied email from the same period
    _insert_email(
        event_store,
        from_address="other@work.com",
        subject="Unreplied request",
        message_id="msg-unreplied-fallback-1",
        hours_ago=118,
    )

    predictions = await engine._check_follow_up_needs({})

    contacts = {p.relevant_contacts[0] for p in predictions}
    assert "colleague@work.com" not in contacts, (
        "Replied-to emails should be excluded even in fallback mode"
    )
    assert "other@work.com" in contacts, (
        "Unreplied emails should still surface in fallback mode"
    )


# -------------------------------------------------------------------------
# First-run flag is not interfered with
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_run_flag_not_interfered_with(db, event_store, user_model_store):
    """The stale-data fallback should not interfere with _first_follow_up_run logic."""
    engine = PredictionEngine(db, user_model_store)
    assert engine._first_follow_up_run is True

    # First run uses 72h lookback — insert email at 60h
    _insert_email(
        event_store,
        from_address="first-run@work.com",
        subject="First run test",
        message_id="msg-first-run-1",
        hours_ago=60,
    )

    predictions = await engine._check_follow_up_needs({})
    assert engine._first_follow_up_run is False
    contacts = {p.relevant_contacts[0] for p in predictions}
    assert "first-run@work.com" in contacts, (
        "First-run 72h lookback should still work independently of fallback"
    )


# -------------------------------------------------------------------------
# Priority contact grace period (1h vs 3h)
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_priority_contact_1h_grace_period(db, event_store, user_model_store):
    """Priority contacts should get a 1-hour grace period (not 3 hours)."""
    engine = PredictionEngine(db, user_model_store)

    # Set up a priority contact in the relationships signal profile
    user_model_store.update_signal_profile("relationships", {
        "contacts": {
            "priority-sender@work.com": {
                "outbound_count": 5,
                "inbound_count": 10,
            }
        }
    })

    # Insert an email from a priority contact 1.5 hours ago
    # (past the 1h priority grace, but before the 3h standard grace)
    _insert_email(
        event_store,
        from_address="priority-sender@work.com",
        subject="Urgent: need your review",
        message_id="msg-priority-grace-1",
        hours_ago=1.5,
    )

    predictions = await engine._check_follow_up_needs({})

    contacts = {p.relevant_contacts[0] for p in predictions}
    assert "priority-sender@work.com" in contacts, (
        "Priority contacts should surface after 1h grace period (1.5h > 1h)"
    )


@pytest.mark.asyncio
async def test_non_priority_contact_3h_grace_period(db, event_store, user_model_store):
    """Non-priority contacts should still use the 3-hour grace period."""
    engine = PredictionEngine(db, user_model_store)

    # Insert an email from a non-priority contact 1.5 hours ago
    # (past 1h but before 3h — should NOT surface for non-priority)
    _insert_email(
        event_store,
        from_address="stranger@external.com",
        subject="Random inquiry",
        message_id="msg-nonpriority-grace-1",
        hours_ago=1.5,
    )

    predictions = await engine._check_follow_up_needs({})

    contacts = {p.relevant_contacts[0] for p in predictions}
    assert "stranger@external.com" not in contacts, (
        "Non-priority contacts should not surface within the 3h grace period"
    )


@pytest.mark.asyncio
async def test_priority_contact_within_1h_grace_not_surfaced(db, event_store, user_model_store):
    """Even priority contacts should NOT surface within the 1-hour grace period."""
    engine = PredictionEngine(db, user_model_store)

    # Set up a priority contact
    user_model_store.update_signal_profile("relationships", {
        "contacts": {
            "vip@work.com": {
                "outbound_count": 3,
                "inbound_count": 7,
            }
        }
    })

    # Insert an email from a priority contact 30 minutes ago
    _insert_email(
        event_store,
        from_address="vip@work.com",
        subject="Quick sync",
        message_id="msg-vip-too-recent",
        hours_ago=0.5,
    )

    predictions = await engine._check_follow_up_needs({})

    contacts = {p.relevant_contacts[0] for p in predictions}
    assert "vip@work.com" not in contacts, (
        "Priority contacts within 1h grace should NOT surface"
    )
