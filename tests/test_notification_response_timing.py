"""
Life OS — Notification Response Timing Tests

Verifies that the notification feedback pipeline computes real response
times from the notification's ``created_at`` timestamp instead of
hardcoding ``response_time_seconds=0``.
"""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from main import LifeOS


# ---------------------------------------------------------------------------
# Helper: build a minimal LifeOS instance with only the database wired up
# ---------------------------------------------------------------------------

def _make_lifeos(db, user_model_store, event_bus):
    """Return a LifeOS whose DB, user-model store, and event bus are the
    test fixtures, bypassing the full startup sequence."""
    return LifeOS(
        db=db,
        event_bus=event_bus,
        user_model_store=user_model_store,
        config={"data_dir": "/tmp/lifeos-test", "ai": {}},
    )


# ---------------------------------------------------------------------------
# _get_notification_response_time unit tests
# ---------------------------------------------------------------------------

class TestGetNotificationResponseTime:
    """Unit tests for LifeOS._get_notification_response_time."""

    def test_returns_positive_for_existing_notification(self, db, user_model_store, event_bus):
        """A notification created a few seconds ago should yield a positive response time."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)

        # Insert a notification created 60 seconds ago
        created = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, domain, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("notif-timing-1", "Test", "body", "normal", "email", "delivered", created),
            )

        result = lifeos._get_notification_response_time("notif-timing-1")

        # Should be roughly 60 seconds (allow some tolerance for test execution)
        assert result > 55.0
        assert result < 120.0

    def test_returns_zero_for_nonexistent_notification(self, db, user_model_store, event_bus):
        """A missing notification ID should fall back to 0.0."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)

        result = lifeos._get_notification_response_time("does-not-exist")

        assert result == 0.0

    def test_returns_zero_when_created_at_is_empty(self, db, user_model_store, event_bus):
        """If created_at is an empty string the helper should return 0.0."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)

        # The schema enforces NOT NULL, so we test with an empty string
        # which is falsy and should be caught by the truthiness check.
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, domain, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("notif-empty-ts", "Test", "body", "normal", "email", "delivered", ""),
            )

        result = lifeos._get_notification_response_time("notif-empty-ts")

        assert result == 0.0

    def test_returns_zero_when_created_at_is_garbage(self, db, user_model_store, event_bus):
        """If created_at is a non-ISO string the helper should return 0.0."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)

        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, domain, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("notif-garbage-ts", "Test", "body", "normal", "email", "delivered", "not-a-date"),
            )

        result = lifeos._get_notification_response_time("notif-garbage-ts")

        assert result == 0.0

    def test_handles_utc_z_suffix(self, db, user_model_store, event_bus):
        """Timestamps ending with 'Z' (common in SQLite defaults) should parse correctly."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)

        created = "2025-01-01T00:00:00.000Z"
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, domain, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("notif-z-suffix", "Test", "body", "normal", "email", "delivered", created),
            )

        result = lifeos._get_notification_response_time("notif-z-suffix")

        # Should be a large positive number (over a year ago)
        assert result > 0.0


# ---------------------------------------------------------------------------
# Integration: full feedback path with real response times
# ---------------------------------------------------------------------------

class TestNotificationFeedbackIntegration:
    """End-to-end tests verifying that notification feedback events receive
    a computed response_time_seconds instead of 0."""

    @pytest.mark.asyncio
    async def test_acted_on_passes_real_response_time(self, db, user_model_store, event_bus):
        """Simulating a notification.acted_on event should result in a
        non-zero response_time_seconds being stored in the feedback log."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)

        # Create a notification that was created 10 seconds ago
        notif_id = "notif-acted-integ"
        created = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, domain, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (notif_id, "Test", "body", "normal", "email", "delivered", created),
            )

        # Directly call feedback_collector with the computed response time
        # (mirrors what master_event_handler now does)
        response_time = lifeos._get_notification_response_time(notif_id)
        await lifeos.feedback_collector.process_notification_response(
            notification_id=notif_id,
            response_type="engaged",
            response_time_seconds=response_time,
        )

        # Verify the stored feedback has a non-zero latency
        with db.get_connection("preferences") as conn:
            feedback = conn.execute(
                "SELECT * FROM feedback_log WHERE action_id = ?",
                (notif_id,),
            ).fetchone()

        assert feedback is not None
        assert feedback["response_latency_seconds"] > 5.0, (
            "response_latency_seconds should reflect real elapsed time, not 0"
        )

    @pytest.mark.asyncio
    async def test_dismissed_passes_real_response_time(self, db, user_model_store, event_bus):
        """Simulating a notification.dismissed event should result in a
        non-zero response_time_seconds being stored in the feedback log."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)

        notif_id = "notif-dismissed-integ"
        created = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, domain, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (notif_id, "Test", "body", "low", "calendar", "delivered", created),
            )

        response_time = lifeos._get_notification_response_time(notif_id)
        await lifeos.feedback_collector.process_notification_response(
            notification_id=notif_id,
            response_type="dismissed",
            response_time_seconds=response_time,
        )

        with db.get_connection("preferences") as conn:
            feedback = conn.execute(
                "SELECT * FROM feedback_log WHERE action_id = ?",
                (notif_id,),
            ).fetchone()

        assert feedback is not None
        assert feedback["response_latency_seconds"] > 2.0

    @pytest.mark.asyncio
    async def test_nonexistent_notification_falls_back_to_zero(self, db, user_model_store, event_bus):
        """When the notification doesn't exist, the feedback path should
        still work with response_time_seconds=0.0 (fail-open)."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)

        response_time = lifeos._get_notification_response_time("ghost-notif")

        assert response_time == 0.0
