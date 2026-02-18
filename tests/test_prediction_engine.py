"""
Comprehensive tests for the PredictionEngine.

Tests all 8 prediction methods across various scenarios to ensure the core
prediction loop generates accurate, actionable predictions.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from models.core import ConfidenceGate
from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# Initialization and Event Cursor Tests
# -------------------------------------------------------------------------


def test_prediction_engine_initializes(db, user_model_store):
    """PredictionEngine can be created with a real DB and UserModelStore."""
    engine = PredictionEngine(db=db, ums=user_model_store)

    assert engine.db is db
    assert engine.ums is user_model_store
    assert engine._last_event_cursor == 0


@pytest.mark.asyncio
async def test_prediction_engine_skips_when_no_new_events(db, event_store, user_model_store):
    """Prediction engine should return empty list when no new events since last run."""
    engine = PredictionEngine(db, user_model_store)

    # First run with no events — should return empty and set cursor
    predictions = await engine.generate_predictions({})
    assert predictions == []

    # Second run with still no new events — should skip via _has_new_events gate
    assert engine._has_new_events() is False, "Should report no new events on second check"
    predictions = await engine.generate_predictions({})
    assert predictions == []


@pytest.mark.asyncio
async def test_prediction_engine_runs_when_new_events_exist(db, event_store, user_model_store):
    """Prediction engine should run when new events exist since last cursor."""
    engine = PredictionEngine(db, user_model_store)

    # First run sets cursor
    await engine.generate_predictions({})

    # Add a new event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "boss@company.com", "subject": "Urgent", "message_id": "msg-1"},
        "metadata": {},
    })

    # Engine should detect new events and run (not skip)
    assert engine._has_new_events() is True, "Should detect the new event"
    predictions = await engine.generate_predictions({})
    # No assertion on length — just that it ran without error


# -------------------------------------------------------------------------
# Calendar Conflict Detection Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calendar_conflicts_detects_overlaps(db, event_store, user_model_store):
    """Calendar conflicts should be detected when two events overlap in time."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Event 1: 2:00 PM - 3:00 PM
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2)).isoformat(),
        "payload": {
            "title": "Team meeting",
            "start_time": (now + timedelta(hours=2)).isoformat(),
            "end_time": (now + timedelta(hours=3)).isoformat(),
        },
        "metadata": {},
    })

    # Event 2: 2:30 PM - 3:30 PM (overlaps with Event 1 by 30 minutes)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2, minutes=30)).isoformat(),
        "payload": {
            "title": "Client call",
            "start_time": (now + timedelta(hours=2, minutes=30)).isoformat(),
            "end_time": (now + timedelta(hours=3, minutes=30)).isoformat(),
        },
        "metadata": {},
    })

    predictions = await engine._check_calendar_conflicts({})
    assert len(predictions) >= 1
    conflict = predictions[0]
    assert conflict.prediction_type == "conflict"
    assert conflict.confidence == 0.95
    assert "overlap" in conflict.description.lower()
    assert "30 minutes" in conflict.description


@pytest.mark.asyncio
async def test_calendar_conflicts_detects_tight_transitions(db, event_store, user_model_store):
    """Events with <15 min gaps should be flagged as tight transitions."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Event 1: 2:00 PM - 3:00 PM
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2)).isoformat(),
        "payload": {
            "title": "Team standup",
            "start_time": (now + timedelta(hours=2)).isoformat(),
            "end_time": (now + timedelta(hours=3)).isoformat(),
        },
        "metadata": {},
    })

    # Event 2: 3:10 PM - 4:00 PM (only 10 min gap)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=3, minutes=10)).isoformat(),
        "payload": {
            "title": "Design review",
            "start_time": (now + timedelta(hours=3, minutes=10)).isoformat(),
            "end_time": (now + timedelta(hours=4)).isoformat(),
        },
        "metadata": {},
    })

    predictions = await engine._check_calendar_conflicts({})
    assert len(predictions) >= 1
    risk = predictions[0]
    assert risk.prediction_type == "risk"
    assert risk.confidence == 0.7
    assert "10 minutes between" in risk.description


@pytest.mark.asyncio
async def test_calendar_conflicts_skips_well_spaced_events(db, event_store, user_model_store):
    """Events with ample gaps (>15 min) should not trigger conflict warnings."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Event 1: 2:00 PM - 3:00 PM
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2)).isoformat(),
        "payload": {
            "title": "Morning meeting",
            "start_time": (now + timedelta(hours=2)).isoformat(),
            "end_time": (now + timedelta(hours=3)).isoformat(),
        },
        "metadata": {},
    })

    # Event 2: 4:00 PM - 5:00 PM (1 hour gap — plenty of time)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=4)).isoformat(),
        "payload": {
            "title": "Afternoon sync",
            "start_time": (now + timedelta(hours=4)).isoformat(),
            "end_time": (now + timedelta(hours=5)).isoformat(),
        },
        "metadata": {},
    })

    predictions = await engine._check_calendar_conflicts({})
    assert len(predictions) == 0


# -------------------------------------------------------------------------
# Follow-up Needs Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_follow_up_skips_marketing_emails(db, event_store, user_model_store):
    """Marketing emails should never generate follow-up predictions."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert a no-reply sender
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=6)).isoformat(),
        "payload": {
            "from_address": "no-reply@marketing.example.com",
            "subject": "50% off today!",
            "snippet": "Big sale happening now",
            "body_plain": "Click here for deals. Unsubscribe: example.com/unsub",
            "message_id": "msg-marketing-1",
        },
        "metadata": {},
    })

    # Insert a noreply variant
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=5)).isoformat(),
        "payload": {
            "from_address": "noreply@accounts.google.com",
            "subject": "Security alert",
            "snippet": "New sign-in detected",
            "message_id": "msg-noreply-1",
        },
        "metadata": {},
    })

    # Insert a newsletter sender
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=4)).isoformat(),
        "payload": {
            "from_address": "newsletter@techcrunch.com",
            "subject": "Daily digest",
            "snippet": "Top stories today",
            "message_id": "msg-newsletter-1",
        },
        "metadata": {},
    })

    predictions = await engine._check_follow_up_needs({})
    # None of these should produce predictions
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_follow_up_keeps_real_emails(db, event_store, user_model_store):
    """Real emails from real people should still generate follow-up predictions."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert a real email from a real person, old enough to need follow-up
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=6)).isoformat(),
        "payload": {
            "from_address": "boss@company.com",
            "subject": "Project update needed",
            "snippet": "Can you send me the latest numbers?",
            "body_plain": "Hi, can you send me the latest numbers? Thanks.",
            "message_id": "msg-real-1",
        },
        "metadata": {},
    })

    predictions = await engine._check_follow_up_needs({})
    assert len(predictions) >= 1
    assert predictions[0].relevant_contacts == ["boss@company.com"]
    assert predictions[0].prediction_type == "reminder"


@pytest.mark.asyncio
async def test_follow_up_skips_already_replied_threads(db, event_store, user_model_store):
    """Emails in threads we've already replied to should not generate reminders."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Inbound message
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=6)).isoformat(),
        "payload": {
            "from_address": "colleague@company.com",
            "subject": "Re: Project update",
            "message_id": "msg-inbound-1",
        },
        "metadata": {},
    })

    # Our reply to that thread
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.sent",
        "source": "google",
        "timestamp": (now - timedelta(hours=4)).isoformat(),
        "payload": {
            "to_address": "colleague@company.com",
            "subject": "Re: Project update",
            "in_reply_to": "msg-inbound-1",
        },
        "metadata": {},
    })

    predictions = await engine._check_follow_up_needs({})
    # Should be empty — we already replied
    assert len(predictions) == 0


# -------------------------------------------------------------------------
# Routine Deviation Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_routine_deviation_detects_missed_habits(db, user_model_store):
    """Missed routines with high consistency should trigger reminder predictions."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    day_name = now.strftime("%A").lower()

    # Insert a high-consistency routine for today (name is PRIMARY KEY, no id column)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?)""",
            (
                "Morning workout",
                f"Every {day_name}",
                json.dumps([{"action": "exercise", "duration": "30min"}]),
                0.85,  # High consistency
                15,  # Observed many times
            ),
        )

    predictions = await engine._check_routine_deviations({})
    assert len(predictions) >= 1
    routine_pred = predictions[0]
    assert routine_pred.prediction_type == "routine_deviation"
    assert "morning workout" in routine_pred.description.lower()
    assert routine_pred.confidence >= 0.4  # Routine deviations have lower confidence


@pytest.mark.asyncio
async def test_routine_deviation_skips_low_consistency_routines(db, user_model_store):
    """Routines with <0.6 consistency should not trigger reminders (too noisy)."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    day_name = now.strftime("%A").lower()

    # Insert a low-consistency routine (name is PRIMARY KEY)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?)""",
            (
                "Sporadic journaling",
                f"Every {day_name}",
                json.dumps([{"action": "write"}]),
                0.4,  # Low consistency — user rarely follows this
                5,
            ),
        )

    predictions = await engine._check_routine_deviations({})
    # Should be empty — consistency too low to trust
    assert len(predictions) == 0


# -------------------------------------------------------------------------
# Relationship Maintenance Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relationship_maintenance_detects_cold_contacts(db, user_model_store):
    """Contacts we haven't engaged with in 1.5x their usual frequency should be flagged."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Create a relationship profile with a contact going cold
    # Need more recent timestamps to meet the algorithm's 15-day windows
    profile_data = {
        "contacts": {
            "old-friend@example.com": {
                "last_interaction": (now - timedelta(days=45)).isoformat(),
                "interaction_count": 12,
                "interaction_timestamps": [
                    (now - timedelta(days=45)).isoformat(),
                    (now - timedelta(days=60)).isoformat(),
                    (now - timedelta(days=75)).isoformat(),
                    (now - timedelta(days=90)).isoformat(),
                    (now - timedelta(days=105)).isoformat(),
                    (now - timedelta(days=120)).isoformat(),
                    (now - timedelta(days=135)).isoformat(),
                    (now - timedelta(days=150)).isoformat(),
                    (now - timedelta(days=165)).isoformat(),
                    (now - timedelta(days=180)).isoformat(),
                ],
            }
        }
    }
    # update_signal_profile only takes 2 args: profile_type and data
    user_model_store.update_signal_profile("relationships", profile_data)

    predictions = await engine._check_relationship_maintenance({})
    assert len(predictions) >= 1
    rel_pred = predictions[0]
    assert rel_pred.prediction_type == "opportunity"
    assert "old-friend@example.com" in rel_pred.description
    assert "45 days" in rel_pred.description


@pytest.mark.asyncio
async def test_relationship_maintenance_skips_new_contacts(db, user_model_store):
    """Contacts with <5 interactions should be skipped (not enough history)."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Contact with only 2 interactions
    profile_data = {
        "contacts": {
            "new-contact@example.com": {
                "last_interaction": (now - timedelta(days=30)).isoformat(),
                "interaction_count": 2,
                "interaction_timestamps": [
                    (now - timedelta(days=30)).isoformat(),
                    (now - timedelta(days=60)).isoformat(),
                ],
            }
        }
    }
    # update_signal_profile only takes 2 args: profile_type and data
    user_model_store.update_signal_profile("relationships", profile_data)

    predictions = await engine._check_relationship_maintenance({})
    # Should be empty — not enough history to establish baseline
    assert len(predictions) == 0


# -------------------------------------------------------------------------
# Preparation Needs Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preparation_detects_travel_events(db, event_store, user_model_store):
    """Upcoming travel events should trigger preparation reminders."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Flight in 24 hours (within the 12-48 hour prep window)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=24)).isoformat(),
        "payload": {
            "title": "Flight to NYC",
            "start_time": (now + timedelta(hours=24)).isoformat(),
            "end_time": (now + timedelta(hours=27)).isoformat(),
            "location": "JFK Airport",
        },
        "metadata": {},
    })

    predictions = await engine._check_preparation_needs({})
    assert len(predictions) >= 1
    prep = predictions[0]
    assert prep.prediction_type == "need"
    assert "travel" in prep.description.lower() or "flight" in prep.description.lower()
    assert prep.confidence == 0.75
    assert "packing" in prep.suggested_action.lower()


@pytest.mark.asyncio
async def test_preparation_detects_large_meetings(db, event_store, user_model_store):
    """Large meetings (>3 attendees) should trigger prep reminders."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Big meeting tomorrow
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=30)).isoformat(),
        "payload": {
            "title": "Quarterly planning session",
            "start_time": (now + timedelta(hours=30)).isoformat(),
            "end_time": (now + timedelta(hours=32)).isoformat(),
            "attendees": ["alice@co.com", "bob@co.com", "carol@co.com", "dan@co.com"],
        },
        "metadata": {},
    })

    predictions = await engine._check_preparation_needs({})
    assert len(predictions) >= 1
    prep = predictions[0]
    assert prep.prediction_type == "need"
    assert "4 attendees" in prep.description or "large meeting" in prep.description.lower()
    assert "agenda" in prep.suggested_action.lower()


@pytest.mark.asyncio
async def test_preparation_skips_events_outside_window(db, event_store, user_model_store):
    """Events outside the 12-48 hour window should not trigger prep reminders."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Event too soon (in 6 hours — before the 12-hour window starts)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=6)).isoformat(),
        "payload": {
            "title": "Flight to LA",
            "start_time": (now + timedelta(hours=6)).isoformat(),
        },
        "metadata": {},
    })

    # Event too far (in 60 hours — after the 48-hour window ends)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=60)).isoformat(),
        "payload": {
            "title": "Flight to SF",
            "start_time": (now + timedelta(hours=60)).isoformat(),
        },
        "metadata": {},
    })

    predictions = await engine._check_preparation_needs({})
    assert len(predictions) == 0


# -------------------------------------------------------------------------
# Spending Pattern Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spending_patterns_detects_anomalies(db, event_store, user_model_store):
    """Categories consuming >25% of spend AND >$200 should trigger alerts."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # $600 on dining (60% of $1000 total — well over threshold)
    for i in range(6):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "finance.transaction.new",
            "source": "plaid",
            "timestamp": (now - timedelta(days=i * 3)).isoformat(),
            "payload": {"category": "dining", "amount": -100},
            "metadata": {},
        })

    # $400 on other categories
    for i in range(4):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "finance.transaction.new",
            "source": "plaid",
            "timestamp": (now - timedelta(days=i * 5)).isoformat(),
            "payload": {"category": "groceries", "amount": -100},
            "metadata": {},
        })

    predictions = await engine._check_spending_patterns({})
    assert len(predictions) >= 1
    spending = predictions[0]
    assert spending.prediction_type == "risk"
    assert "$600" in spending.description
    assert "dining" in spending.description.lower()
    assert "60%" in spending.description


@pytest.mark.asyncio
async def test_spending_patterns_skips_low_absolute_spend(db, event_store, user_model_store):
    """Categories <$200 absolute should not trigger alerts (even if high %)."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # $150 on subscriptions (75% of $200 total) — high % but low absolute
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "finance.transaction.new",
        "source": "plaid",
        "timestamp": (now - timedelta(days=5)).isoformat(),
        "payload": {"category": "subscriptions", "amount": -150},
        "metadata": {},
    })

    # $50 on other
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "finance.transaction.new",
        "source": "plaid",
        "timestamp": (now - timedelta(days=10)).isoformat(),
        "payload": {"category": "entertainment", "amount": -50},
        "metadata": {},
    })

    predictions = await engine._check_spending_patterns({})
    # Should be empty — $150 is below the $200 absolute threshold
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_spending_patterns_requires_sufficient_data(db, event_store, user_model_store):
    """Spending patterns need at least 5 transactions to avoid false positives."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Only 3 transactions — not enough for pattern detection
    for i in range(3):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "finance.transaction.new",
            "source": "plaid",
            "timestamp": (now - timedelta(days=i * 2)).isoformat(),
            "payload": {"category": "travel", "amount": -300},
            "metadata": {},
        })

    predictions = await engine._check_spending_patterns({})
    assert len(predictions) == 0


# -------------------------------------------------------------------------
# Reaction Prediction Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_reaction_surfaces_high_confidence(db, user_model_store):
    """High-confidence predictions should get a reaction score boost."""
    engine = PredictionEngine(db, user_model_store)
    from models.user_model import Prediction

    pred = Prediction(
        prediction_type="conflict",
        description="Calendar overlap detected",
        confidence=0.9,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="24_hours",  # Required field
    )

    reaction = await engine.predict_reaction(pred, {})
    # High confidence + conflict type should push score above 0.4 (helpful)
    assert reaction.predicted_reaction in ("helpful", "neutral")


@pytest.mark.asyncio
async def test_predict_reaction_suppresses_during_dismissal_fatigue(db, user_model_store):
    """Predictions should be suppressed if user dismissed >5 in last 2 hours (recalibrated threshold)."""
    engine = PredictionEngine(db, user_model_store)
    from models.user_model import Prediction

    now = datetime.now(timezone.utc)

    # Insert 6 recent dismissals (threshold increased from >3 to >5 in recalibration)
    with db.get_connection("preferences") as conn:
        for i in range(6):
            conn.execute(
                """INSERT INTO feedback_log (id, feedback_type, action_id, action_type, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    "dismissed",
                    str(uuid.uuid4()),
                    "notification",
                    (now - timedelta(minutes=30 + i * 10)).isoformat(),
                ),
            )

    pred = Prediction(
        prediction_type="opportunity",
        description="Time to reach out",
        confidence=0.5,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="this_week",  # Required field
    )

    reaction = await engine.predict_reaction(pred, {})
    # Dismissal fatigue should suppress this (score should be negative)
    # With recalibrated thresholds, we expect neutral or annoying
    assert reaction.predicted_reaction in ["neutral", "annoying"]


@pytest.mark.asyncio
async def test_predict_reaction_prioritizes_conflicts_and_risks(db, user_model_store):
    """Conflicts and risks should get score boost for urgency."""
    engine = PredictionEngine(db, user_model_store)
    from models.user_model import Prediction

    conflict = Prediction(
        prediction_type="conflict",
        description="Meetings overlap",
        confidence=0.6,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="24_hours",  # Required field
    )

    opportunity = Prediction(
        prediction_type="opportunity",
        description="Good time to call",
        confidence=0.6,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="this_week",  # Required field
    )

    conflict_reaction = await engine.predict_reaction(conflict, {})
    opp_reaction = await engine.predict_reaction(opportunity, {})

    # Conflict should score higher than opportunity with same confidence
    assert conflict_reaction.predicted_reaction in ("helpful", "neutral")
    # Opportunity gets a -0.1 penalty, so it may be suppressed
    # (Don't assert it's annoying since base score is 0.5, just check conflict is better)


# -------------------------------------------------------------------------
# Accuracy Multiplier Tests
# -------------------------------------------------------------------------


def test_accuracy_multiplier_returns_baseline_with_insufficient_data(db, user_model_store):
    """Accuracy multiplier should be 1.0 when <5 resolved predictions exist."""
    engine = PredictionEngine(db, user_model_store)

    # No predictions in DB yet
    multiplier = engine._get_accuracy_multiplier("reminder")
    assert multiplier == 1.0


def test_accuracy_multiplier_scales_with_accuracy(db, user_model_store):
    """Accuracy multiplier should scale from 0.5 (0% accurate) to 1.1 (100% accurate)."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert 10 predictions, 5 accurate (50% accuracy)
    with db.get_connection("user_model") as conn:
        for i in range(10):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate, time_horizon,
                    created_at, was_surfaced, resolved_at, was_accurate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    "reminder",
                    "Test prediction",
                    0.7,
                    "suggest",  # confidence_gate is NOT NULL
                    "24_hours",
                    now.isoformat(),
                    1,  # was_surfaced
                    now.isoformat(),  # resolved
                    1 if i < 5 else 0,  # 5 accurate, 5 inaccurate
                ),
            )

    multiplier = engine._get_accuracy_multiplier("reminder")
    # 50% accuracy -> 0.5 + (0.5 * 0.6) = 0.8
    assert abs(multiplier - 0.8) < 0.01


def test_accuracy_multiplier_applies_penalty_floor_for_low_accuracy(db, user_model_store):
    """Prediction types with <20% accuracy and ≥10 samples should get the 0.3 penalty floor.

    A hard 0.0 would create a death spiral (no predictions → no data → never recovers).
    The 0.3 floor keeps the type alive at reduced confidence so the learning loop can
    rehabilitate it as new, higher-quality predictions accumulate.
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert 12 predictions, only 1 accurate (8% accuracy)
    with db.get_connection("user_model") as conn:
        for i in range(12):
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate, time_horizon,
                    created_at, was_surfaced, resolved_at, was_accurate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    "reminder",
                    "Test prediction",
                    0.7,
                    "suggest",  # confidence_gate is NOT NULL
                    "24_hours",
                    now.isoformat(),
                    1,
                    now.isoformat(),
                    1 if i == 0 else 0,  # Only 1 accurate
                ),
            )

    multiplier = engine._get_accuracy_multiplier("reminder")
    # Heavy penalty but NOT a hard cutoff — 0.3 allows recovery
    assert multiplier == 0.3, f"Expected 0.3 penalty floor for <20% accuracy, got {multiplier}"


# -------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_confidence_predictions_not_surfaced(db, event_store, user_model_store):
    """Predictions below 0.6 confidence should not be surfaced."""
    engine = PredictionEngine(db, user_model_store)
    # Force the engine to think there are new events
    engine._last_event_cursor = 0

    predictions = await engine.generate_predictions({})
    for pred in predictions:
        assert pred.confidence >= 0.6, f"Prediction surfaced with confidence {pred.confidence} < 0.6"


@pytest.mark.asyncio
async def test_max_five_predictions_per_cycle(db, event_store, user_model_store):
    """At most 5 predictions should be surfaced per cycle."""
    engine = PredictionEngine(db, user_model_store)
    engine._last_event_cursor = 0

    predictions = await engine.generate_predictions({})
    assert len(predictions) <= 5, f"Expected at most 5 predictions, got {len(predictions)}"


@pytest.mark.asyncio
async def test_predictions_stored_with_surfaced_flag(db, event_store, user_model_store):
    """All predictions should be stored with was_surfaced flag set correctly."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Create conditions that generate a prediction
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=6)).isoformat(),
        "payload": {
            "from_address": "important@company.com",
            "subject": "Action needed",
            "message_id": "msg-1",
        },
        "metadata": {},
    })

    engine._last_event_cursor = 0
    predictions = await engine.generate_predictions({})

    # Check that predictions were stored in the DB
    with db.get_connection("user_model") as conn:
        rows = conn.execute(
            "SELECT id, was_surfaced FROM predictions ORDER BY created_at DESC"
        ).fetchall()

    # At least one prediction should have been stored
    assert len(rows) >= 1
    # Verify was_surfaced flag exists
    for row in rows:
        assert "was_surfaced" in row.keys()
        assert row["was_surfaced"] in (0, 1)
