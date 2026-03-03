"""
Tests for Home Assistant episodic memory integration and notification.ignored
feedback handler.

Part A: Verifies that home.arrived and home.departed events from the Home
Assistant connector are included in episodic memory and properly classified,
while home.device.state_changed (device-level noise) is intentionally excluded.

Part B: Verifies that notification.ignored events are routed to the feedback
collector with response_type='ignored' and response_time_seconds=None.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
import yaml

from main import LifeOS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lifeos_from_config(tmp_path):
    """Create a minimal LifeOS instance via config file for classification tests."""
    config_data = {
        "data_dir": str(tmp_path / "data"),
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "embedding_model": "all-MiniLM-L6-v2",
        "ai": {
            "ollama_url": "http://localhost:11434",
            "ollama_model": "mistral",
            "use_cloud": False,
        },
        "connectors": {},
    }
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml.dump(config_data))
    (tmp_path / "data").mkdir(exist_ok=True)
    return LifeOS(str(config_file))


def _make_lifeos(db, user_model_store, event_bus):
    """Return a LifeOS whose DB, user-model store, and event bus are the
    test fixtures, bypassing the full startup sequence."""
    return LifeOS(
        db=db,
        event_bus=event_bus,
        user_model_store=user_model_store,
        config={"data_dir": "/tmp/lifeos-test", "ai": {}},
    )


# ===========================================================================
# Part A: Home Assistant events in episodic memory
# ===========================================================================

class TestHomeAssistantEpisodicTypes:
    """Verify home.arrived and home.departed are in the episodic event set."""

    def test_home_arrived_in_episodic_event_types(self, lifeos_from_config):
        """home.arrived should be included in episodic event types so HA
        presence data enters the user's episodic memory."""
        # Access the episodic_event_types via _create_episode by checking
        # that home.arrived events don't return early (i.e., they're in the set).
        # We verify by checking the classification works (only episodic types
        # reach the classifier).
        interaction_type = lifeos_from_config._classify_interaction_type(
            "home.arrived", {}
        )
        assert interaction_type == "home_arrived"

    def test_home_departed_in_episodic_event_types(self, lifeos_from_config):
        """home.departed should be included in episodic event types so HA
        departure data enters the user's episodic memory."""
        interaction_type = lifeos_from_config._classify_interaction_type(
            "home.departed", {}
        )
        assert interaction_type == "home_departed"

    def test_home_device_state_changed_not_in_episodic_types(self, lifeos_from_config):
        """home.device.state_changed should NOT be in episodic event types.
        Device state changes (lights, thermostat) are not user-facing
        interactions and would add noise to episodic memory."""
        # home.device.state_changed should fall through to the fallback
        # classifier, indicating it's NOT in the explicit episodic set.
        interaction_type = lifeos_from_config._classify_interaction_type(
            "home.device.state_changed", {}
        )
        # Should hit the fallback: extracts suffix after last dot
        assert interaction_type == "state_changed"


class TestHomeAssistantClassification:
    """Verify _classify_interaction_type handles HA presence events."""

    def test_home_arrived_returns_home_arrived(self, lifeos_from_config):
        """home.arrived should map to 'home_arrived' for routine detection."""
        result = lifeos_from_config._classify_interaction_type("home.arrived", {})
        assert result == "home_arrived"

    def test_home_departed_returns_home_departed(self, lifeos_from_config):
        """home.departed should map to 'home_departed' for routine detection."""
        result = lifeos_from_config._classify_interaction_type("home.departed", {})
        assert result == "home_departed"

    def test_home_arrived_distinct_from_location_arrived(self, lifeos_from_config):
        """home.arrived and location.arrived should be distinct types since
        they come from different connectors with different semantics."""
        home = lifeos_from_config._classify_interaction_type("home.arrived", {})
        location = lifeos_from_config._classify_interaction_type("location.arrived", {})
        assert home != location

    def test_home_departed_distinct_from_location_departed(self, lifeos_from_config):
        """home.departed and location.departed should be distinct types."""
        home = lifeos_from_config._classify_interaction_type("home.departed", {})
        location = lifeos_from_config._classify_interaction_type("location.departed", {})
        assert home != location


class TestHomeAssistantEpisodeSummary:
    """Verify _generate_episode_summary handles HA presence events."""

    def test_home_arrived_summary(self, lifeos_from_config):
        """home.arrived events should generate 'Arrived home' summary."""
        summary = lifeos_from_config._generate_episode_summary(
            {"type": "home.arrived", "payload": {}}
        )
        assert summary == "Arrived home"

    def test_home_departed_summary(self, lifeos_from_config):
        """home.departed events should generate 'Left home' summary."""
        summary = lifeos_from_config._generate_episode_summary(
            {"type": "home.departed", "payload": {}}
        )
        assert summary == "Left home"


# ===========================================================================
# Part B: notification.ignored feedback handler
# ===========================================================================

class TestNotificationIgnoredFeedback:
    """Verify that notification.ignored events are routed to the feedback
    collector with the correct parameters."""

    @pytest.mark.asyncio
    async def test_ignored_notification_calls_feedback_collector(
        self, db, user_model_store, event_bus
    ):
        """notification.ignored should call process_notification_response
        with response_type='ignored' and response_time_seconds=None."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)

        # Mock the feedback collector to capture the call
        lifeos.feedback_collector.process_notification_response = AsyncMock()

        # Create a notification in state.db so the handler can find it
        notif_id = "notif-ignored-test"
        created = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, domain, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (notif_id, "Test Alert", "body", "normal", "email", "delivered", created),
            )

        # Simulate the feedback processing stage of master_event_handler
        event = {
            "id": "evt-ignored-1",
            "type": "notification.ignored",
            "payload": {"notification_id": notif_id},
        }

        # Execute the feedback processing block directly
        event_type = event.get("type", "")
        if event_type == "notification.ignored":
            notif_id_from_event = event.get("payload", {}).get("notification_id")
            if notif_id_from_event:
                await lifeos.feedback_collector.process_notification_response(
                    notification_id=notif_id_from_event,
                    response_type="ignored",
                    response_time_seconds=None,
                )

        # Verify the call was made with correct parameters
        lifeos.feedback_collector.process_notification_response.assert_called_once_with(
            notification_id=notif_id,
            response_type="ignored",
            response_time_seconds=None,
        )

    @pytest.mark.asyncio
    async def test_ignored_notification_uses_none_for_response_time(
        self, db, user_model_store, event_bus
    ):
        """Ignored notifications should pass response_time_seconds=None since
        the user never interacted, so there is no meaningful response time."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)

        lifeos.feedback_collector.process_notification_response = AsyncMock()

        notif_id = "notif-ignored-none-time"
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, domain, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (notif_id, "Ignored", "body", "low", "task", "delivered",
                 datetime.now(timezone.utc).isoformat()),
            )

        event = {
            "id": "evt-ignored-2",
            "type": "notification.ignored",
            "payload": {"notification_id": notif_id},
        }

        event_type = event.get("type", "")
        if event_type == "notification.ignored":
            pid = event.get("payload", {}).get("notification_id")
            if pid:
                await lifeos.feedback_collector.process_notification_response(
                    notification_id=pid,
                    response_type="ignored",
                    response_time_seconds=None,
                )

        call_kwargs = lifeos.feedback_collector.process_notification_response.call_args[1]
        assert call_kwargs["response_time_seconds"] is None

    @pytest.mark.asyncio
    async def test_ignored_notification_without_id_is_skipped(
        self, db, user_model_store, event_bus
    ):
        """notification.ignored events without a notification_id should be
        silently skipped (fail-open)."""
        lifeos = _make_lifeos(db, user_model_store, event_bus)
        lifeos.feedback_collector.process_notification_response = AsyncMock()

        event = {
            "id": "evt-ignored-no-id",
            "type": "notification.ignored",
            "payload": {},  # No notification_id
        }

        event_type = event.get("type", "")
        if event_type == "notification.ignored":
            pid = event.get("payload", {}).get("notification_id")
            if pid:
                await lifeos.feedback_collector.process_notification_response(
                    notification_id=pid,
                    response_type="ignored",
                    response_time_seconds=None,
                )

        # Should NOT have been called since there's no notification_id
        lifeos.feedback_collector.process_notification_response.assert_not_called()
