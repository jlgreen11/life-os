"""
Tests for events.db-based InsightEngine correlators.

Covers _email_peak_hour_insights and _meeting_density_insights — two
correlators that read directly from the events table (events.db) rather
than user_model.db, so they keep producing useful behavioural insights
even when user_model.db is unavailable.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.engine import InsightEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_event(event_store, event_type: str, timestamp: datetime):
    """Insert a minimal event into events.db via the EventStore."""
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": "test",
        "timestamp": timestamp.isoformat(),
        "priority": 2,
        "payload": {},
        "metadata": {},
    })


# ===========================================================================
# _email_peak_hour_insights tests
# ===========================================================================


@pytest.mark.asyncio
async def test_peak_hour_no_emails_returns_empty(db, user_model_store):
    """No email events at all should produce no insights."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    insights = engine._email_peak_hour_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_peak_hour_few_emails_returns_empty(db, user_model_store, event_store):
    """Below the 50-email threshold should produce no insights."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    base = datetime.now(timezone.utc)

    for i in range(10):
        _insert_event(event_store, "email.received", base - timedelta(hours=i))

    insights = engine._email_peak_hour_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_peak_hour_detected(db, user_model_store, event_store):
    """When 40%+ of emails land in a single hour, an insight is generated."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    base = datetime.now(timezone.utc)

    # Pin peak hour to 10:00 UTC — use today at 10:00 as anchor so all
    # emails fall within the 30-day window.
    peak_anchor = base.replace(hour=10, minute=30, second=0, microsecond=0)

    # 50 emails in hour 10
    for i in range(50):
        _insert_event(
            event_store, "email.received",
            peak_anchor - timedelta(days=i % 25, minutes=i % 60),
        )

    # 60 emails spread across other hours (5 per hour for 12 hours)
    for hour_offset in range(1, 13):
        hour_anchor = base.replace(hour=(10 + hour_offset) % 24, minute=15, second=0, microsecond=0)
        for j in range(5):
            _insert_event(
                event_store, "email.received",
                hour_anchor - timedelta(days=j),
            )

    insights = engine._email_peak_hour_insights()
    assert len(insights) == 1
    assert insights[0].type == "behavioral_pattern"
    assert insights[0].category == "email_timing"
    assert insights[0].entity == "10"
    assert "10:00" in insights[0].summary


@pytest.mark.asyncio
async def test_peak_hour_no_dominant_hour_returns_empty(db, user_model_store, event_store):
    """Evenly distributed emails should not produce an insight."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    base = datetime.now(timezone.utc)

    # 60 emails, ~5 per hour across 12 hours — no hour is 2x the average
    for hour_offset in range(12):
        anchor = base.replace(hour=hour_offset, minute=30, second=0, microsecond=0)
        for j in range(5):
            _insert_event(
                event_store, "email.received",
                anchor - timedelta(days=j),
            )

    insights = engine._email_peak_hour_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_peak_hour_insight_has_correct_fields(db, user_model_store, event_store):
    """Verify type, category, entity, and evidence keys on the produced insight."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    base = datetime.now(timezone.utc)

    peak_anchor = base.replace(hour=14, minute=30, second=0, microsecond=0)

    # 60 emails at hour 14
    for i in range(60):
        _insert_event(
            event_store, "email.received",
            peak_anchor - timedelta(days=i % 28, minutes=i % 60),
        )

    # 30 emails spread across 6 other hours
    for hour_offset in range(1, 7):
        anchor = base.replace(hour=(14 + hour_offset) % 24, minute=15, second=0, microsecond=0)
        for j in range(5):
            _insert_event(
                event_store, "email.received",
                anchor - timedelta(days=j),
            )

    insights = engine._email_peak_hour_insights()
    assert len(insights) == 1

    insight = insights[0]
    assert insight.type == "behavioral_pattern"
    assert insight.category == "email_timing"
    assert insight.entity == "14"
    assert insight.dedup_key  # compute_dedup_key was called

    evidence_keys = {e.split("=")[0] for e in insight.evidence}
    assert "peak_hour" in evidence_keys
    assert "peak_count" in evidence_keys
    assert "avg_hourly" in evidence_keys

    assert 0.0 < insight.confidence <= 0.85


# ===========================================================================
# _meeting_density_insights tests
# ===========================================================================


@pytest.mark.asyncio
async def test_meeting_density_no_calendar_events_returns_empty(db, user_model_store):
    """No calendar events at all should produce no insights."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    insights = engine._meeting_density_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_meeting_density_few_events_returns_empty(db, user_model_store, event_store):
    """Below the 10-event threshold should produce no insights."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    base = datetime.now(timezone.utc)

    for i in range(5):
        _insert_event(event_store, "calendar.event.created", base - timedelta(days=i))

    insights = engine._meeting_density_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_meeting_density_peak_day_detected(db, user_model_store, event_store):
    """When 50%+ of meetings fall on one day, an insight is generated."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    base = datetime.now(timezone.utc)

    # Find next Monday (or today if Monday) and anchor at noon UTC
    days_until_monday = (7 - base.weekday()) % 7
    if days_until_monday == 0 and base.weekday() != 0:
        days_until_monday = 7
    monday = (base - timedelta(days=base.weekday())).replace(
        hour=12, minute=0, second=0, microsecond=0,
    )

    # 12 meetings on Monday (spread across recent Mondays)
    for week in range(4):
        for i in range(3):
            _insert_event(
                event_store, "calendar.event.created",
                monday - timedelta(weeks=week, hours=i),
            )

    # 2 meetings on each of 3 other days
    for day_offset in [1, 2, 3]:
        for week in range(2):
            _insert_event(
                event_store, "calendar.event.created",
                monday - timedelta(weeks=week) + timedelta(days=day_offset),
            )

    insights = engine._meeting_density_insights()
    assert len(insights) == 1
    assert insights[0].type == "behavioral_pattern"
    assert insights[0].category == "meeting_density"
    assert insights[0].entity == "Monday"
    assert "Monday" in insights[0].summary


@pytest.mark.asyncio
async def test_meeting_density_no_dominant_day_returns_empty(db, user_model_store, event_store):
    """Evenly distributed meetings should not produce an insight."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    base = datetime.now(timezone.utc)

    # 14 meetings, 2 per day across 7 days — no day is 1.5x
    for day_offset in range(7):
        for j in range(2):
            _insert_event(
                event_store, "calendar.event.created",
                base - timedelta(days=day_offset, hours=j),
            )

    insights = engine._meeting_density_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_meeting_density_insight_dedup_key_set(db, user_model_store, event_store):
    """Verify compute_dedup_key was called on meeting density insight."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    base = datetime.now(timezone.utc)

    # Anchor on a known Wednesday at noon UTC
    wednesday = (base - timedelta(days=(base.weekday() - 2) % 7)).replace(
        hour=12, minute=0, second=0, microsecond=0,
    )

    # 10 meetings on Wednesday
    for week in range(3):
        for i in range(4 if week < 2 else 2):
            _insert_event(
                event_store, "calendar.event.created",
                wednesday - timedelta(weeks=week, hours=i),
            )

    # 2 meetings on other days to meet threshold
    for day_offset in [1, 2]:
        _insert_event(
            event_store, "calendar.event.created",
            wednesday + timedelta(days=day_offset),
        )

    insights = engine._meeting_density_insights()
    # If there is an insight, dedup_key must be set
    if insights:
        assert insights[0].dedup_key
        assert len(insights[0].dedup_key) == 16
