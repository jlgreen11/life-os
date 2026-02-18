"""
Tests for the InsightEngine ``_actionable_alert_insights`` correlator.

The ``actionable_alert`` insight type surfaces time-sensitive items the user
should act on now: overdue/soon-due tasks and calendar events starting within
the next 24 hours.

This test suite validates:
- Overdue task detection and confidence scaling
- Soon-due task detection (due within 24 hours)
- Tasks with no due_date are skipped
- Completed/cancelled tasks are skipped
- Calendar events starting within 24 hours generate insights
- Calendar events starting in < 1 hour get high_urgency flag
- Calendar events beyond 24 hours are excluded
- Calendar events already started (past) are excluded
- Duplicate calendar event IDs within one batch are deduplicated
- Malformed due_date / start_time values are handled gracefully
- Integration: correlator is wired into generate_insights()
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.engine import InsightEngine
from services.insight_engine.models import Insight


# =============================================================================
# Task helpers
# =============================================================================


def _insert_task(db, title: str, status: str, due_date: str | None,
                 priority: int = 2, task_id: str | None = None) -> str:
    """Insert a task into state.db and return its id."""
    tid = task_id or str(uuid.uuid4())
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO tasks
               (id, title, status, priority, due_date, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                tid, title, status, priority, due_date,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return tid


def _insert_cal_event(db, title: str, start_time: str,
                      event_id: str | None = None) -> str:
    """Insert a calendar.event.created event and return the event_id."""
    eid = event_id or str(uuid.uuid4())
    payload = json.dumps({
        "event_id": eid,
        "title": title,
        "start_time": start_time,
    })
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events
               (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "calendar.event.created",
                "test",
                datetime.now(timezone.utc).isoformat(),
                2, payload, "{}",
            ),
        )
    return eid


# =============================================================================
# Overdue / soon-due task tests
# =============================================================================


@pytest.mark.asyncio
async def test_actionable_alert_no_tasks(db, user_model_store):
    """No tasks → no actionable_alert insights."""
    engine = InsightEngine(db, user_model_store)
    insights = engine._actionable_alert_insights()
    task_insights = [i for i in insights if i.category == "overdue_task"]
    assert task_insights == []


@pytest.mark.asyncio
async def test_actionable_alert_overdue_task(db, user_model_store):
    """Overdue task (past due_date) should generate an actionable_alert insight."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    due = (now - timedelta(days=2)).isoformat()
    tid = _insert_task(db, "Submit Q1 report", "pending", due)

    insights = engine._actionable_alert_insights()
    task_insights = [i for i in insights if i.category == "overdue_task"]

    assert len(task_insights) == 1
    assert task_insights[0].type == "actionable_alert"
    assert "overdue" in task_insights[0].summary.lower()
    assert "Submit Q1 report" in task_insights[0].summary
    assert task_insights[0].entity == tid
    assert any("overdue" in e for e in task_insights[0].evidence)


@pytest.mark.asyncio
async def test_actionable_alert_due_soon_task(db, user_model_store):
    """Task due in < 24 hours should generate an actionable_alert insight."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    due = (now + timedelta(hours=3)).isoformat()
    _insert_task(db, "Prepare presentation", "pending", due)

    insights = engine._actionable_alert_insights()
    task_insights = [i for i in insights if i.category == "overdue_task"]

    assert len(task_insights) == 1
    assert "due in" in task_insights[0].summary.lower()
    assert "Prepare presentation" in task_insights[0].summary
    assert any("due_soon" in e for e in task_insights[0].evidence)


@pytest.mark.asyncio
async def test_actionable_alert_task_no_due_date_skipped(db, user_model_store):
    """Tasks with no due_date should not generate insights."""
    engine = InsightEngine(db, user_model_store)
    _insert_task(db, "Eventually refactor code", "pending", None)

    insights = engine._actionable_alert_insights()
    task_insights = [i for i in insights if i.category == "overdue_task"]
    assert task_insights == []


@pytest.mark.asyncio
async def test_actionable_alert_completed_task_skipped(db, user_model_store):
    """Completed tasks should not generate insights even if overdue."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    due = (now - timedelta(days=3)).isoformat()
    _insert_task(db, "Done task", "completed", due)

    insights = engine._actionable_alert_insights()
    task_insights = [i for i in insights if i.category == "overdue_task"]
    assert task_insights == []


@pytest.mark.asyncio
async def test_actionable_alert_cancelled_task_skipped(db, user_model_store):
    """Cancelled tasks should not generate insights."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    due = (now - timedelta(days=1)).isoformat()
    _insert_task(db, "Cancelled task", "cancelled", due)

    insights = engine._actionable_alert_insights()
    task_insights = [i for i in insights if i.category == "overdue_task"]
    assert task_insights == []


@pytest.mark.asyncio
async def test_actionable_alert_future_task_beyond_24h_skipped(db, user_model_store):
    """Tasks due more than 24 hours in the future should not generate insights."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    due = (now + timedelta(hours=36)).isoformat()
    _insert_task(db, "Future task", "pending", due)

    insights = engine._actionable_alert_insights()
    task_insights = [i for i in insights if i.category == "overdue_task"]
    assert task_insights == []


@pytest.mark.asyncio
async def test_actionable_alert_overdue_confidence_scaling(db, user_model_store):
    """More-overdue tasks should get higher confidence, capped at 0.9."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    _insert_task(db, "Slightly overdue", "pending",
                 (now - timedelta(hours=1)).isoformat(), task_id="t1")
    _insert_task(db, "Very overdue", "pending",
                 (now - timedelta(days=10)).isoformat(), task_id="t2")

    insights = engine._actionable_alert_insights()
    task_insights = {i.entity: i for i in insights if i.category == "overdue_task"}

    assert "t1" in task_insights
    assert "t2" in task_insights
    assert task_insights["t1"].confidence < task_insights["t2"].confidence
    assert task_insights["t2"].confidence <= 0.9


@pytest.mark.asyncio
async def test_actionable_alert_in_progress_task_included(db, user_model_store):
    """In-progress tasks that are overdue should still generate insights."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    due = (now - timedelta(days=1)).isoformat()
    _insert_task(db, "In-progress overdue", "in_progress", due)

    insights = engine._actionable_alert_insights()
    task_insights = [i for i in insights if i.category == "overdue_task"]
    assert len(task_insights) == 1


@pytest.mark.asyncio
async def test_actionable_alert_malformed_due_date_skipped(db, user_model_store):
    """Tasks with malformed due_date should be skipped gracefully."""
    engine = InsightEngine(db, user_model_store)
    _insert_task(db, "Bad due date task", "pending", "not-a-date")

    # Should not raise; just returns empty
    insights = engine._actionable_alert_insights()
    task_insights = [i for i in insights if i.category == "overdue_task"]
    assert task_insights == []


# =============================================================================
# Calendar event tests
# =============================================================================


@pytest.mark.asyncio
async def test_actionable_alert_calendar_no_events(db, user_model_store):
    """No calendar events → no calendar insights."""
    engine = InsightEngine(db, user_model_store)
    insights = engine._actionable_alert_insights()
    cal_insights = [i for i in insights if i.category == "upcoming_calendar"]
    assert cal_insights == []


@pytest.mark.asyncio
async def test_actionable_alert_calendar_upcoming_event(db, user_model_store):
    """Calendar event starting in < 24h should generate an actionable_alert insight."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    start_time = (now + timedelta(hours=3)).isoformat()
    eid = _insert_cal_event(db, "Team standup", start_time)

    insights = engine._actionable_alert_insights()
    cal_insights = [i for i in insights if i.category == "upcoming_calendar"]

    assert len(cal_insights) == 1
    assert cal_insights[0].type == "actionable_alert"
    assert "Team standup" in cal_insights[0].summary
    assert cal_insights[0].entity == eid
    assert any(f"event_id={eid}" in e for e in cal_insights[0].evidence)


@pytest.mark.asyncio
async def test_actionable_alert_calendar_high_urgency(db, user_model_store):
    """Event starting in < 1 hour should have high_urgency flag and higher confidence."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    start_time = (now + timedelta(minutes=30)).isoformat()
    _insert_cal_event(db, "Imminent meeting", start_time)

    insights = engine._actionable_alert_insights()
    cal_insights = [i for i in insights if i.category == "upcoming_calendar"]

    assert len(cal_insights) == 1
    assert any("high_urgency=true" in e for e in cal_insights[0].evidence)
    assert cal_insights[0].confidence >= 0.8


@pytest.mark.asyncio
async def test_actionable_alert_calendar_beyond_24h_skipped(db, user_model_store):
    """Calendar events starting more than 24 hours away should be excluded."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    start_time = (now + timedelta(hours=36)).isoformat()
    _insert_cal_event(db, "Future meeting", start_time)

    insights = engine._actionable_alert_insights()
    cal_insights = [i for i in insights if i.category == "upcoming_calendar"]
    assert cal_insights == []


@pytest.mark.asyncio
async def test_actionable_alert_calendar_past_event_skipped(db, user_model_store):
    """Calendar events that have already started should be excluded."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    start_time = (now - timedelta(hours=2)).isoformat()
    _insert_cal_event(db, "Past meeting", start_time)

    insights = engine._actionable_alert_insights()
    cal_insights = [i for i in insights if i.category == "upcoming_calendar"]
    assert cal_insights == []


@pytest.mark.asyncio
async def test_actionable_alert_calendar_duplicate_event_id_deduped(db, user_model_store):
    """Multiple events rows with the same event_id produce only one insight."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    start_time = (now + timedelta(hours=2)).isoformat()
    shared_eid = str(uuid.uuid4())

    # Insert two rows with the same event_id (simulates re-sync)
    for _ in range(2):
        _insert_cal_event(db, "Weekly sync", start_time, event_id=shared_eid)

    insights = engine._actionable_alert_insights()
    cal_insights = [i for i in insights if i.category == "upcoming_calendar"]
    assert len(cal_insights) == 1


@pytest.mark.asyncio
async def test_actionable_alert_calendar_malformed_start_time_skipped(db, user_model_store):
    """Calendar events with malformed start_time should be skipped gracefully."""
    engine = InsightEngine(db, user_model_store)
    # Insert event with bad start_time directly
    payload = json.dumps({
        "event_id": str(uuid.uuid4()),
        "title": "Bad event",
        "start_time": "not-a-timestamp",
    })
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events
               (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()), "calendar.event.created", "test",
                datetime.now(timezone.utc).isoformat(), 2, payload, "{}",
            ),
        )

    insights = engine._actionable_alert_insights()
    cal_insights = [i for i in insights if i.category == "upcoming_calendar"]
    assert cal_insights == []


# =============================================================================
# Integration tests
# =============================================================================


@pytest.mark.asyncio
async def test_generate_insights_includes_actionable_alerts(db, user_model_store):
    """generate_insights should include actionable_alert insights when tasks are overdue."""
    engine = InsightEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    due = (now - timedelta(days=1)).isoformat()
    _insert_task(db, "Overdue task", "pending", due)

    insights = await engine.generate_insights()
    types = {i.type for i in insights}
    assert "actionable_alert" in types


@pytest.mark.asyncio
async def test_generate_insights_actionable_alert_not_source_weight_filtered(db, user_model_store):
    """Actionable alerts should never be filtered by source weights (confidence stays unchanged)."""
    from services.insight_engine.source_weights import SourceWeightManager
    engine = InsightEngine(db, user_model_store, source_weight_manager=SourceWeightManager(db))

    now = datetime.now(timezone.utc)
    due = (now - timedelta(days=1)).isoformat()
    _insert_task(db, "Alert task", "pending", due)

    insights = await engine.generate_insights()
    alert_insights = [i for i in insights if i.type == "actionable_alert"]

    # Should appear even when source weights are active because actionable_alert
    # categories are intentionally excluded from the source-weight map.
    assert len(alert_insights) >= 1
