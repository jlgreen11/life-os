"""
Tests for calendar event date filtering in the dashboard feed endpoint.

Verifies that the /api/dashboard/feed?topic=calendar endpoint correctly
handles all-day events (date-only timestamps), full datetime timestamps,
timezone-aware and naive timestamps, and properly excludes past and
far-future events.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_calendar_row(event_id: str, payload: dict, timestamp: str | None = None):
    """Build a mock database row matching the events table schema."""
    return {
        "id": event_id,
        "payload": json.dumps(payload),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }


def _make_mock_life_os(calendar_rows: list[dict]):
    """Create a minimal LifeOS mock wired for dashboard feed calendar tests.

    ``calendar_rows`` is the list of row dicts returned by the events DB query.
    """
    life_os = Mock()

    # Database mock — returns ``calendar_rows`` for the calendar query
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_cursor.fetchall.return_value = calendar_rows
    mock_cursor.fetchone.return_value = (0,)
    mock_conn.execute.return_value = mock_cursor
    life_os.db = Mock()
    life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    # Event bus / event store
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-1")

    # Notification manager — empty
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0})

    # Task manager — empty
    life_os.task_manager = Mock()
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.create_task = Mock(return_value="task-1")
    life_os.task_manager.update_task = Mock()
    life_os.task_manager.complete_task = Mock()

    # Signal extractor
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(
        return_value=Mock(
            energy_level=0.5, stress_level=0.3, social_battery=0.4,
            cognitive_load=0.3, emotional_valence=0.5, confidence=0.6, trend="stable",
        )
    )

    # Vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    # AI engine
    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = Mock(return_value="Briefing")
    life_os.ai_engine.draft_reply = Mock(return_value="Draft")
    life_os.ai_engine.search_life = Mock(return_value="Result")

    # Rules engine
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.rules_engine.add_rule = Mock(return_value="rule-1")
    life_os.rules_engine.remove_rule = Mock()

    # User model store
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.user_model_store.resolve_prediction = Mock()

    # Feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})
    life_os.feedback_collector.process_explicit_feedback = Mock()

    # Connector / browser stubs
    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    # Onboarding
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.onboarding.submit_answer = Mock()
    life_os.onboarding.finalize = Mock(return_value={})

    # Connector management
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})
    life_os.save_connector_config = Mock()
    life_os.test_connector = Mock(return_value={"success": True})
    life_os.enable_connector = Mock(return_value={"status": "started"})
    life_os.disable_connector = Mock(return_value={"status": "stopped"})

    return life_os


def _get_calendar_items(response_json: dict) -> list[dict]:
    """Extract only calendar/event items from the dashboard feed response."""
    return [item for item in response_json.get("items", []) if item.get("kind") == "event"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAllDayEventsInFeed:
    """All-day events with date-only start_time must appear when within 7 days."""

    def test_allday_event_within_window_appears(self):
        """An all-day event 3 days from now should appear in the calendar feed."""
        future_date = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")
        rows = [
            _make_calendar_row("evt-allday", {
                "event_id": "allday-1",
                "title": "Team Offsite",
                "start_time": future_date,
                "end_time": future_date,
                "is_all_day": True,
            }),
        ]
        life_os = _make_mock_life_os(rows)
        client = TestClient(create_web_app(life_os))

        resp = client.get("/api/dashboard/feed?topic=calendar")
        assert resp.status_code == 200
        cal_items = _get_calendar_items(resp.json())
        assert len(cal_items) == 1
        assert cal_items[0]["title"] == "Team Offsite"
        assert cal_items[0]["metadata"]["is_all_day"] is True

    def test_allday_event_tomorrow_appears(self):
        """An all-day event for tomorrow should appear."""
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        rows = [
            _make_calendar_row("evt-tomorrow", {
                "event_id": "allday-tomorrow",
                "title": "Birthday",
                "start_time": tomorrow,
                "end_time": tomorrow,
                "is_all_day": True,
            }),
        ]
        life_os = _make_mock_life_os(rows)
        client = TestClient(create_web_app(life_os))

        resp = client.get("/api/dashboard/feed?topic=calendar")
        cal_items = _get_calendar_items(resp.json())
        assert len(cal_items) == 1
        assert cal_items[0]["title"] == "Birthday"


class TestFullDatetimeEventsInFeed:
    """Events with full datetime timestamps work correctly."""

    def test_future_datetime_event_appears(self):
        """An event 2 days from now with full ISO datetime should appear."""
        future_dt = (datetime.now(timezone.utc) + timedelta(days=2, hours=3)).isoformat()
        rows = [
            _make_calendar_row("evt-dt", {
                "event_id": "dt-1",
                "title": "Team Meeting",
                "start_time": future_dt,
                "end_time": (datetime.now(timezone.utc) + timedelta(days=2, hours=4)).isoformat(),
            }),
        ]
        life_os = _make_mock_life_os(rows)
        client = TestClient(create_web_app(life_os))

        resp = client.get("/api/dashboard/feed?topic=calendar")
        cal_items = _get_calendar_items(resp.json())
        assert len(cal_items) == 1
        assert cal_items[0]["title"] == "Team Meeting"

    def test_event_with_z_suffix_appears(self):
        """An event with 'Z' timezone suffix should be parsed correctly."""
        future_dt = (datetime.now(timezone.utc) + timedelta(days=1, hours=5))
        start_str = future_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = [
            _make_calendar_row("evt-z", {
                "event_id": "z-1",
                "title": "Sync Call",
                "start_time": start_str,
                "end_time": (future_dt + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }),
        ]
        life_os = _make_mock_life_os(rows)
        client = TestClient(create_web_app(life_os))

        resp = client.get("/api/dashboard/feed?topic=calendar")
        cal_items = _get_calendar_items(resp.json())
        assert len(cal_items) == 1
        assert cal_items[0]["title"] == "Sync Call"

    def test_event_with_naive_datetime_appears(self):
        """An event with a naive datetime (no tz) should be treated as UTC."""
        future_dt = (datetime.now(timezone.utc) + timedelta(days=2))
        start_str = future_dt.strftime("%Y-%m-%dT%H:%M:%S")
        rows = [
            _make_calendar_row("evt-naive", {
                "event_id": "naive-1",
                "title": "Standup",
                "start_time": start_str,
                "end_time": (future_dt + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
            }),
        ]
        life_os = _make_mock_life_os(rows)
        client = TestClient(create_web_app(life_os))

        resp = client.get("/api/dashboard/feed?topic=calendar")
        cal_items = _get_calendar_items(resp.json())
        assert len(cal_items) == 1
        assert cal_items[0]["title"] == "Standup"


class TestPastEventsExcluded:
    """Events in the past should be filtered out of the calendar feed."""

    def test_yesterday_event_excluded(self):
        """An event from yesterday should NOT appear."""
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        rows = [
            _make_calendar_row("evt-past", {
                "event_id": "past-1",
                "title": "Old Meeting",
                "start_time": yesterday,
                "end_time": yesterday,
            }),
        ]
        life_os = _make_mock_life_os(rows)
        client = TestClient(create_web_app(life_os))

        resp = client.get("/api/dashboard/feed?topic=calendar")
        cal_items = _get_calendar_items(resp.json())
        assert len(cal_items) == 0

    def test_yesterday_allday_excluded(self):
        """An all-day event from yesterday (date-only) should NOT appear."""
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        rows = [
            _make_calendar_row("evt-past-allday", {
                "event_id": "past-allday-1",
                "title": "Past Holiday",
                "start_time": yesterday,
                "end_time": yesterday,
                "is_all_day": True,
            }),
        ]
        life_os = _make_mock_life_os(rows)
        client = TestClient(create_web_app(life_os))

        resp = client.get("/api/dashboard/feed?topic=calendar")
        cal_items = _get_calendar_items(resp.json())
        assert len(cal_items) == 0


class TestFarFutureEventsExcluded:
    """Events beyond the 7-day window should be excluded."""

    def test_event_10_days_out_excluded(self):
        """An event 10 days from now should NOT appear."""
        far_future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        rows = [
            _make_calendar_row("evt-far", {
                "event_id": "far-1",
                "title": "Future Conference",
                "start_time": far_future,
                "end_time": far_future,
            }),
        ]
        life_os = _make_mock_life_os(rows)
        client = TestClient(create_web_app(life_os))

        resp = client.get("/api/dashboard/feed?topic=calendar")
        cal_items = _get_calendar_items(resp.json())
        assert len(cal_items) == 0

    def test_allday_event_10_days_out_excluded(self):
        """An all-day event 10 days out (date-only) should NOT appear."""
        far_date = (datetime.now(timezone.utc) + timedelta(days=10)).strftime("%Y-%m-%d")
        rows = [
            _make_calendar_row("evt-far-allday", {
                "event_id": "far-allday-1",
                "title": "Future Holiday",
                "start_time": far_date,
                "end_time": far_date,
                "is_all_day": True,
            }),
        ]
        life_os = _make_mock_life_os(rows)
        client = TestClient(create_web_app(life_os))

        resp = client.get("/api/dashboard/feed?topic=calendar")
        cal_items = _get_calendar_items(resp.json())
        assert len(cal_items) == 0


class TestMixedEventFiltering:
    """Multiple events with different date formats are correctly filtered."""

    def test_mixed_events_correct_filtering(self):
        """Only future events within 7 days should appear, regardless of format."""
        now = datetime.now(timezone.utc)
        rows = [
            # Should appear: all-day event in 2 days
            _make_calendar_row("evt-1", {
                "event_id": "allday-ok",
                "title": "Team Offsite",
                "start_time": (now + timedelta(days=2)).strftime("%Y-%m-%d"),
                "end_time": (now + timedelta(days=2)).strftime("%Y-%m-%d"),
                "is_all_day": True,
            }),
            # Should appear: datetime event in 5 days
            _make_calendar_row("evt-2", {
                "event_id": "dt-ok",
                "title": "Weekly Sync",
                "start_time": (now + timedelta(days=5, hours=2)).isoformat(),
                "end_time": (now + timedelta(days=5, hours=3)).isoformat(),
            }),
            # Should NOT appear: past datetime event
            _make_calendar_row("evt-3", {
                "event_id": "dt-past",
                "title": "Past Standup",
                "start_time": (now - timedelta(days=2)).isoformat(),
                "end_time": (now - timedelta(days=2, hours=-1)).isoformat(),
            }),
            # Should NOT appear: far-future all-day event
            _make_calendar_row("evt-4", {
                "event_id": "allday-far",
                "title": "Far Away Event",
                "start_time": (now + timedelta(days=14)).strftime("%Y-%m-%d"),
                "end_time": (now + timedelta(days=14)).strftime("%Y-%m-%d"),
                "is_all_day": True,
            }),
        ]
        life_os = _make_mock_life_os(rows)
        client = TestClient(create_web_app(life_os))

        resp = client.get("/api/dashboard/feed?topic=calendar")
        assert resp.status_code == 200
        cal_items = _get_calendar_items(resp.json())
        titles = {item["title"] for item in cal_items}
        assert titles == {"Team Offsite", "Weekly Sync"}
