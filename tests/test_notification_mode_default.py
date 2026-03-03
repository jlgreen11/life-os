"""
Tests for notification mode default behavior.

Verifies that fresh installations (no onboarding, no notification_mode
preference) default to 'immediate' delivery so notifications actually
appear, while explicit user preferences are still honored.
"""

import pytest

from services.notification_manager.manager import NotificationManager


@pytest.fixture
def nm(db, event_bus):
    """A NotificationManager with a fresh (empty) preferences DB."""
    return NotificationManager(db, event_bus, config={}, timezone="UTC")


class TestNotificationModeDefault:
    """Tests for _get_notification_mode() default and explicit preferences."""

    def test_default_mode_is_immediate(self, nm):
        """Fresh install with no notification_mode preference returns 'immediate'."""
        mode = nm._get_notification_mode()
        assert mode == "immediate"

    def test_explicit_batched_preference_honored(self, db, nm):
        """When user sets notification_mode to 'batched', it is respected."""
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_preferences (key, value) VALUES (?, ?)",
                ("notification_mode", "batched"),
            )
        mode = nm._get_notification_mode()
        assert mode == "batched"

    def test_explicit_minimal_preference_honored(self, db, nm):
        """When user sets notification_mode to 'minimal', it is respected."""
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_preferences (key, value) VALUES (?, ?)",
                ("notification_mode", "minimal"),
            )
        mode = nm._get_notification_mode()
        assert mode == "minimal"


class TestNotificationDeliveryWithDefault:
    """Tests that the default 'immediate' mode actually delivers notifications."""

    @pytest.mark.asyncio
    async def test_normal_priority_delivered_immediately_by_default(self, db, nm):
        """Normal-priority notification is delivered (not batched) on fresh install."""
        notif_id = await nm.create_notification(
            title="Test notification",
            body="This should be delivered immediately",
            priority="normal",
            domain="test",
        )

        assert notif_id is not None

        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT status FROM notifications WHERE id = ?",
                (notif_id,),
            ).fetchone()

        assert row is not None
        assert row["status"] == "delivered"

    @pytest.mark.asyncio
    async def test_notification_survives_restart_cycle(self, db, event_bus):
        """Notifications persist across NotificationManager restarts."""
        # Create a notification with the first manager instance
        nm1 = NotificationManager(db, event_bus, config={}, timezone="UTC")

        # Set batched mode so the notification stays pending
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_preferences (key, value) VALUES (?, ?)",
                ("notification_mode", "batched"),
            )

        notif_id = await nm1.create_notification(
            title="Restart test",
            body="Should survive restart",
            priority="normal",
            domain="test",
        )
        assert notif_id is not None

        # Simulate restart: create a new NotificationManager with the same DB
        nm2 = NotificationManager(db, event_bus, config={}, timezone="UTC")
        pending = nm2.get_pending()

        pending_ids = [n["id"] for n in pending]
        assert notif_id in pending_ids
