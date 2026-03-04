"""
Tests for GET /api/dashboard/badges endpoint.

Verifies the lightweight badge-count endpoint returns correct per-topic
counts without loading full feed payloads.  This endpoint replaces the
previous pattern of 5 separate /api/dashboard/feed requests just to
count items per topic.
"""

from contextlib import contextmanager
from unittest.mock import Mock

from fastapi.testclient import TestClient

from web.app import create_web_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_life_os(notifications=None, tasks=None, cal_rows=0, insight_rows=0):
    """Build a mock life_os with controllable notification/task/calendar data.

    Args:
        notifications: list of notification dicts to return from get_pending.
        tasks: list of task dicts to return from get_pending_tasks.
        cal_rows: number of upcoming calendar events to simulate.
        insight_rows: number of active insights to simulate.
    """
    life_os = Mock()
    life_os.config = {}

    # Notification manager
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=notifications or [])

    # Task manager
    life_os.task_manager = Mock()
    life_os.task_manager.get_pending_tasks = Mock(return_value=tasks or [])

    # Database connections for calendar and insights queries
    def _mock_get_connection(db_name):
        """Return a context-manager mock connection per database."""
        conn = Mock()

        if db_name == "events":
            # Calendar count query returns cal_rows
            row = (cal_rows,)
            cursor = Mock()
            cursor.fetchone = Mock(return_value=row)
            conn.execute = Mock(return_value=cursor)
        elif db_name == "user_model":
            # Insights count query returns insight_rows
            row = (insight_rows,)
            cursor = Mock()
            cursor.fetchone = Mock(return_value=row)
            conn.execute = Mock(return_value=cursor)
        else:
            cursor = Mock()
            cursor.fetchone = Mock(return_value=(0,))
            conn.execute = Mock(return_value=cursor)

        @contextmanager
        def _ctx_mgr(name):
            yield conn

        return conn

    # Make get_connection a context manager that dispatches by db name
    @contextmanager
    def _get_connection(db_name):
        yield _mock_get_connection(db_name)

    life_os.db = Mock()
    life_os.db.get_connection = _get_connection

    return life_os


EXPECTED_TOPICS = ["inbox", "messages", "email", "calendar", "tasks", "insights"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDashboardBadges:
    """Tests for GET /api/dashboard/badges."""

    def test_returns_all_topic_keys(self):
        """Response contains badge counts for all expected topics."""
        life_os = _make_life_os()
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/dashboard/badges")
        assert resp.status_code == 200

        data = resp.json()
        assert "badges" in data
        for topic in EXPECTED_TOPICS:
            assert topic in data["badges"], f"Missing topic: {topic}"

    def test_all_counts_are_non_negative_integers(self):
        """Every badge count must be a non-negative integer."""
        life_os = _make_life_os()
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/dashboard/badges")
        data = resp.json()

        for topic, count in data["badges"].items():
            assert isinstance(count, int), f"{topic} count is not int: {type(count)}"
            assert count >= 0, f"{topic} count is negative: {count}"

    def test_empty_state_returns_all_zeros(self):
        """With no notifications, tasks, or calendar events, all counts are 0."""
        life_os = _make_life_os()
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/dashboard/badges")
        data = resp.json()

        for topic in EXPECTED_TOPICS:
            assert data["badges"][topic] == 0, f"{topic} should be 0"

    def test_message_notifications_counted(self):
        """Notifications with 'message' or 'signal' domain count under messages."""
        notifications = [
            {"id": "1", "domain": "imessage", "title": "Hey"},
            {"id": "2", "domain": "signal", "title": "Hi"},
            {"id": "3", "domain": "email.inbox", "title": "Newsletter"},
        ]
        life_os = _make_life_os(notifications=notifications)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/dashboard/badges")
        data = resp.json()["badges"]

        assert data["messages"] == 2  # imessage + signal
        assert data["email"] == 1  # email.inbox
        assert data["inbox"] >= 3  # all notifications + tasks + calendar

    def test_task_count(self):
        """Pending tasks are counted under the tasks topic."""
        tasks = [
            {"id": "t1", "title": "Buy groceries", "priority": "normal"},
            {"id": "t2", "title": "Call dentist", "priority": "high"},
        ]
        life_os = _make_life_os(tasks=tasks)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/dashboard/badges")
        data = resp.json()["badges"]

        assert data["tasks"] == 2

    def test_calendar_count(self):
        """Upcoming calendar events are counted under the calendar topic."""
        life_os = _make_life_os(cal_rows=5)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/dashboard/badges")
        data = resp.json()["badges"]

        assert data["calendar"] == 5

    def test_inbox_aggregates_all(self):
        """Inbox count is the sum of notifications + tasks + calendar."""
        notifications = [
            {"id": "1", "domain": "email.inbox", "title": "Email 1"},
            {"id": "2", "domain": "imessage", "title": "Msg 1"},
        ]
        tasks = [{"id": "t1", "title": "Task", "priority": "normal"}]
        life_os = _make_life_os(notifications=notifications, tasks=tasks, cal_rows=3)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/dashboard/badges")
        data = resp.json()["badges"]

        # inbox = 2 notifications + 1 task + 3 calendar = 6
        assert data["inbox"] == 6

    def test_service_errors_produce_zero_counts(self):
        """If a service raises an exception, its count defaults to 0."""
        life_os = _make_life_os()
        # Make notification_manager.get_pending raise
        life_os.notification_manager.get_pending = Mock(side_effect=RuntimeError("DB down"))
        life_os.task_manager.get_pending_tasks = Mock(side_effect=RuntimeError("DB down"))

        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/dashboard/badges")
        assert resp.status_code == 200

        data = resp.json()["badges"]
        assert data["messages"] == 0
        assert data["email"] == 0
        assert data["tasks"] == 0
        # inbox = 0 (notifications) + 0 (tasks) + calendar
        assert data["inbox"] >= 0
