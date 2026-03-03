"""
Tests for InsightEngine source weight mapping gaps and calendar alert timestamp fix.

Validates:
1. email_timing insights have source weight applied (not bypassed)
2. meeting_density insights have source weight applied (not bypassed)
3. _actionable_alert_insights() finds calendar events synced long ago but
   starting soon (payload start_time filter vs timestamp filter)
4. _actionable_alert_insights() does NOT alert for events starting >24h away
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.engine import InsightEngine
from services.insight_engine.models import Insight
from services.insight_engine.source_weights import SourceWeightManager


# =============================================================================
# Helpers
# =============================================================================


def _insert_event(db, event_type, payload, timestamp=None, source="test"):
    """Insert an event into the events database with the given type and payload."""
    event_id = str(uuid.uuid4())
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (event_id, event_type, source, ts, json.dumps(payload)),
        )
    return event_id


# =============================================================================
# Bug 1: Missing source weight mappings
# =============================================================================


@pytest.mark.asyncio
async def test_email_timing_insight_has_source_weight_applied(db, user_model_store):
    """email_timing insights should be weighted by the email.work source weight.

    When the user has set email.work weight to 0.5, an email_timing insight
    with confidence 0.8 should have its confidence halved to 0.4.
    """
    swm = SourceWeightManager(db)
    swm.seed_defaults()
    # Set email.work weight to 0.5
    swm.set_user_weight("email.work", 0.5)

    engine = InsightEngine(db, user_model_store, source_weight_manager=swm, timezone="UTC")

    # Create an email_timing insight directly and pass through source weight filter
    insight = Insight(
        type="behavioral_pattern",
        summary="Your email peaks between 10:00-11:00 (25 emails vs ~8 average)",
        confidence=0.8,
        evidence=["peak_hour=10", "peak_count=25", "avg_hourly=8"],
        category="email_timing",
        entity="10",
    )

    weighted = engine._apply_source_weights([insight])

    assert len(weighted) == 1
    # Confidence should be multiplied by the 0.5 weight
    assert weighted[0].confidence == pytest.approx(0.4, abs=0.05)
    # Evidence should record the source weight application
    assert any("source_weight=" in e for e in weighted[0].evidence)


@pytest.mark.asyncio
async def test_meeting_density_insight_has_source_weight_applied(db, user_model_store):
    """meeting_density insights should be weighted by the calendar.meetings source weight.

    When the user has set calendar.meetings weight to 0.5, a meeting_density
    insight with confidence 0.8 should have its confidence halved to 0.4.
    """
    swm = SourceWeightManager(db)
    swm.seed_defaults()
    # Set calendar.meetings weight to 0.5
    swm.set_user_weight("calendar.meetings", 0.5)

    engine = InsightEngine(db, user_model_store, source_weight_manager=swm, timezone="UTC")

    insight = Insight(
        type="behavioral_pattern",
        summary="Your meeting-heaviest day is Tuesday (12 meetings vs ~5 average)",
        confidence=0.8,
        evidence=["peak_day=Tuesday", "peak_count=12", "avg_daily=5"],
        category="meeting_density",
        entity="Tuesday",
    )

    weighted = engine._apply_source_weights([insight])

    assert len(weighted) == 1
    # Confidence should be multiplied by the 0.5 weight
    assert weighted[0].confidence == pytest.approx(0.4, abs=0.05)
    assert any("source_weight=" in e for e in weighted[0].evidence)


@pytest.mark.asyncio
async def test_email_timing_insight_dropped_when_weight_very_low(db, user_model_store):
    """email_timing insights should be dropped when source weight makes confidence < 0.1."""
    swm = SourceWeightManager(db)
    swm.seed_defaults()
    # Set email.work weight very low so the insight gets filtered out
    swm.set_user_weight("email.work", 0.05)

    engine = InsightEngine(db, user_model_store, source_weight_manager=swm, timezone="UTC")

    insight = Insight(
        type="behavioral_pattern",
        summary="Your email peaks between 10:00-11:00",
        confidence=0.5,
        category="email_timing",
        entity="10",
    )

    weighted = engine._apply_source_weights([insight])

    # 0.5 * 0.05 = 0.025 < 0.1 threshold → should be dropped
    assert len(weighted) == 0


@pytest.mark.asyncio
async def test_meeting_density_insight_dropped_when_weight_very_low(db, user_model_store):
    """meeting_density insights should be dropped when source weight makes confidence < 0.1."""
    swm = SourceWeightManager(db)
    swm.seed_defaults()
    swm.set_user_weight("calendar.meetings", 0.05)

    engine = InsightEngine(db, user_model_store, source_weight_manager=swm, timezone="UTC")

    insight = Insight(
        type="behavioral_pattern",
        summary="Your meeting-heaviest day is Tuesday",
        confidence=0.5,
        category="meeting_density",
        entity="Tuesday",
    )

    weighted = engine._apply_source_weights([insight])

    # 0.5 * 0.05 = 0.025 < 0.1 threshold → should be dropped
    assert len(weighted) == 0


# =============================================================================
# Bug 2: Calendar alert queries sync timestamp, not event start time
# =============================================================================


@pytest.mark.asyncio
async def test_actionable_alert_finds_old_synced_event_starting_soon(db, user_model_store):
    """Calendar events synced 48h ago but starting in 2h should generate alerts.

    This tests the fix: the SQL now filters on payload start_time rather than
    the event's sync timestamp, so events added to the calendar well before
    they occur are correctly surfaced.
    """
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    now = datetime.now(timezone.utc)
    # Event was synced 48 hours ago...
    sync_time = (now - timedelta(hours=48)).isoformat()
    # ...but it starts in 2 hours
    start_time = (now + timedelta(hours=2)).isoformat()

    _insert_event(
        db,
        "calendar.event.created",
        {
            "event_id": "cal-old-sync-123",
            "title": "Team Standup",
            "start_time": start_time,
        },
        timestamp=sync_time,
    )

    insights = engine._actionable_alert_insights()

    # Should find the event despite old sync timestamp
    calendar_insights = [i for i in insights if i.category == "upcoming_calendar"]
    assert len(calendar_insights) == 1
    assert "Team Standup" in calendar_insights[0].summary


@pytest.mark.asyncio
async def test_actionable_alert_ignores_event_starting_beyond_24h(db, user_model_store):
    """Calendar events starting more than 24h from now should NOT generate alerts."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    now = datetime.now(timezone.utc)
    # Event synced recently but starts in 30 hours (beyond the 24h window)
    sync_time = (now - timedelta(hours=1)).isoformat()
    start_time = (now + timedelta(hours=30)).isoformat()

    _insert_event(
        db,
        "calendar.event.created",
        {
            "event_id": "cal-far-future-456",
            "title": "Future Planning Session",
            "start_time": start_time,
        },
        timestamp=sync_time,
    )

    insights = engine._actionable_alert_insights()

    # Should NOT find the event since it starts >24h from now
    calendar_insights = [i for i in insights if i.category == "upcoming_calendar"]
    assert len(calendar_insights) == 0


@pytest.mark.asyncio
async def test_actionable_alert_ignores_past_events(db, user_model_store):
    """Calendar events whose start_time is in the past should NOT generate alerts."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    now = datetime.now(timezone.utc)
    sync_time = (now - timedelta(hours=2)).isoformat()
    # Event already started 1 hour ago
    start_time = (now - timedelta(hours=1)).isoformat()

    _insert_event(
        db,
        "calendar.event.created",
        {
            "event_id": "cal-past-789",
            "title": "Already Happened",
            "start_time": start_time,
        },
        timestamp=sync_time,
    )

    insights = engine._actionable_alert_insights()

    calendar_insights = [i for i in insights if i.category == "upcoming_calendar"]
    assert len(calendar_insights) == 0


@pytest.mark.asyncio
async def test_actionable_alert_uses_start_field_fallback(db, user_model_store):
    """Calendar events with 'start' field (instead of 'start_time') should be found.

    The SQL uses COALESCE(json_extract(payload, '$.start_time'),
    json_extract(payload, '$.start')) to handle both field names.
    """
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    now = datetime.now(timezone.utc)
    sync_time = (now - timedelta(hours=48)).isoformat()
    start_time = (now + timedelta(hours=3)).isoformat()

    _insert_event(
        db,
        "calendar.event.created",
        {
            "event_id": "cal-start-field-101",
            "title": "Board Meeting",
            "start": start_time,  # Using 'start' instead of 'start_time'
        },
        timestamp=sync_time,
    )

    insights = engine._actionable_alert_insights()

    calendar_insights = [i for i in insights if i.category == "upcoming_calendar"]
    assert len(calendar_insights) == 1
    assert "Board Meeting" in calendar_insights[0].summary
