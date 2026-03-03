"""
Tests for the LifeOS._db_health_loop() runtime DB health check.

Verifies that the background loop:
- detects user_model.db corruption via probe queries and triggers rebuild
- skips rebuild when the database is healthy
- caps runtime rebuilds at 3 to prevent infinite repair loops
- sends user notifications and WebSocket broadcasts on corruption events
"""

import asyncio
import sqlite3
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def lifeos_instance(db):
    """Create a minimal LifeOS instance with injected test DB.

    Uses dependency injection to provide a real DatabaseManager backed
    by temporary SQLite databases, avoiding any NATS or Ollama dependency.
    Includes mocked notification_manager and event_bus for notification tests.
    """
    from main import LifeOS

    instance = LifeOS(
        config={"data_dir": "./data", "ai": {}, "connectors": {}},
        db=db,
    )
    instance.shutdown_event = asyncio.Event()
    instance.notification_manager = AsyncMock()
    instance.event_bus = AsyncMock()
    instance.event_bus.is_connected = True
    return instance


async def _run_loop_once(lifeos):
    """Run _db_health_loop for a single iteration then stop.

    Patches asyncio.sleep so that the first call sets the shutdown event
    and returns immediately, causing the loop to exit after one iteration.
    """
    original_sleep = asyncio.sleep

    async def sleep_then_shutdown(seconds):
        """On first call, signal shutdown so the loop exits after one pass."""
        lifeos.shutdown_event.set()
        # Yield control to allow event loop to process
        await original_sleep(0)

    with patch("asyncio.sleep", side_effect=sleep_then_shutdown):
        await lifeos._db_health_loop()


async def test_healthy_db_skips_rebuild(lifeos_instance):
    """When all probe queries succeed, no rebuild should be triggered."""
    with patch.object(
        lifeos_instance, "_rebuild_user_model_db_if_corrupted", new_callable=AsyncMock
    ) as mock_rebuild:
        await _run_loop_once(lifeos_instance)

    mock_rebuild.assert_not_called()
    assert lifeos_instance._runtime_db_rebuilds == 0


async def test_healthy_db_sends_no_notification(lifeos_instance):
    """When the DB is healthy, no notifications should be created."""
    with (
        patch.object(
            lifeos_instance, "_rebuild_user_model_db_if_corrupted", new_callable=AsyncMock
        ),
        patch("main.ws_manager") as mock_ws,
    ):
        await _run_loop_once(lifeos_instance)

    lifeos_instance.notification_manager.create_notification.assert_not_called()
    lifeos_instance.event_bus.publish.assert_not_called()
    mock_ws.broadcast.assert_not_called()


async def test_corrupted_db_triggers_rebuild(lifeos_instance):
    """When a probe query raises an exception, rebuild should be called
    and a success notification should be created."""
    original_get_connection = lifeos_instance.db.get_connection

    @contextmanager
    def corrupted_connection(db_name):
        if db_name == "user_model":
            mock_conn = MagicMock()
            mock_conn.execute.side_effect = sqlite3.DatabaseError(
                "database disk image is malformed"
            )
            yield mock_conn
        else:
            with original_get_connection(db_name) as conn:
                yield conn

    with (
        patch.object(lifeos_instance.db, "get_connection", side_effect=corrupted_connection),
        patch.object(
            lifeos_instance, "_rebuild_user_model_db_if_corrupted", new_callable=AsyncMock
        ) as mock_rebuild,
        patch.object(
            lifeos_instance, "_verify_and_retry_backfills", new_callable=AsyncMock
        ) as mock_backfill,
        patch("main.ws_manager") as mock_ws,
    ):
        await _run_loop_once(lifeos_instance)

    mock_rebuild.assert_called_once()
    mock_backfill.assert_called_once()
    assert lifeos_instance._runtime_db_rebuilds == 1

    # Verify success notification was created
    lifeos_instance.notification_manager.create_notification.assert_called_once()
    call_kwargs = lifeos_instance.notification_manager.create_notification.call_args[1]
    assert "auto-repaired" in call_kwargs["title"].lower()
    assert call_kwargs["priority"] == "normal"
    assert call_kwargs["domain"] == "system"

    # Verify system event was published
    lifeos_instance.event_bus.publish.assert_called_once()
    pub_args = lifeos_instance.event_bus.publish.call_args
    assert pub_args[0][0] == "system.database.corruption_detected"
    assert pub_args[0][1]["database"] == "user_model"
    assert pub_args[1]["priority"] == "critical"

    # Verify WebSocket broadcast was sent
    mock_ws.broadcast.assert_called_once()
    ws_data = mock_ws.broadcast.call_args[0][0]
    assert ws_data["type"] == "db_corruption"
    assert ws_data["database"] == "user_model"


async def test_rebuild_counter_limits_retries(lifeos_instance):
    """After 3 rebuilds, the loop should stop attempting further repairs."""
    # Pre-set counter above limit — loop should skip probing entirely
    lifeos_instance._runtime_db_rebuilds = 4

    with patch.object(
        lifeos_instance, "_rebuild_user_model_db_if_corrupted", new_callable=AsyncMock
    ) as mock_rebuild:
        await _run_loop_once(lifeos_instance)

    mock_rebuild.assert_not_called()


async def test_rebuild_counter_increments_on_each_corruption(lifeos_instance):
    """Each corruption detection should increment the counter by 1."""
    original_get_connection = lifeos_instance.db.get_connection

    @contextmanager
    def corrupted_connection(db_name):
        if db_name == "user_model":
            mock_conn = MagicMock()
            mock_conn.execute.side_effect = sqlite3.DatabaseError(
                "database disk image is malformed"
            )
            yield mock_conn
        else:
            with original_get_connection(db_name) as conn:
                yield conn

    # Run the loop 3 times to simulate repeated corruption
    for expected_count in (1, 2, 3):
        # Reset shutdown event for next iteration
        lifeos_instance.shutdown_event = asyncio.Event()

        with (
            patch.object(lifeos_instance.db, "get_connection", side_effect=corrupted_connection),
            patch.object(
                lifeos_instance, "_rebuild_user_model_db_if_corrupted", new_callable=AsyncMock
            ),
            patch.object(
                lifeos_instance, "_verify_and_retry_backfills", new_callable=AsyncMock
            ),
            patch("main.ws_manager"),
        ):
            await _run_loop_once(lifeos_instance)

        assert lifeos_instance._runtime_db_rebuilds == expected_count


async def test_fourth_corruption_stops_rebuilds(lifeos_instance):
    """On the 4th corruption, the counter exceeds limit, rebuild stops,
    and a critical notification is created for the user."""
    # Set counter to 3, one more corruption should increment to 4 and log error
    lifeos_instance._runtime_db_rebuilds = 3

    original_get_connection = lifeos_instance.db.get_connection

    @contextmanager
    def corrupted_connection(db_name):
        if db_name == "user_model":
            mock_conn = MagicMock()
            mock_conn.execute.side_effect = sqlite3.DatabaseError(
                "database disk image is malformed"
            )
            yield mock_conn
        else:
            with original_get_connection(db_name) as conn:
                yield conn

    with (
        patch.object(lifeos_instance.db, "get_connection", side_effect=corrupted_connection),
        patch.object(
            lifeos_instance, "_rebuild_user_model_db_if_corrupted", new_callable=AsyncMock
        ) as mock_rebuild,
        patch.object(
            lifeos_instance, "_verify_and_retry_backfills", new_callable=AsyncMock
        ) as mock_backfill,
        patch("main.ws_manager"),
    ):
        await _run_loop_once(lifeos_instance)

    # Counter incremented to 4, which exceeds the limit → no rebuild
    mock_rebuild.assert_not_called()
    mock_backfill.assert_not_called()
    assert lifeos_instance._runtime_db_rebuilds == 4

    # Verify critical notification was created for the user
    lifeos_instance.notification_manager.create_notification.assert_called_once()
    call_kwargs = lifeos_instance.notification_manager.create_notification.call_args[1]
    assert "manual intervention" in call_kwargs["title"].lower()
    assert call_kwargs["priority"] == "critical"
    assert call_kwargs["domain"] == "system"


async def test_connection_failure_triggers_rebuild(lifeos_instance):
    """If get_connection itself raises, the loop should detect corruption."""
    original_get_connection = lifeos_instance.db.get_connection

    def raise_on_user_model(db_name):
        if db_name == "user_model":
            raise sqlite3.DatabaseError("unable to open database file")
        return original_get_connection(db_name)

    with (
        patch.object(lifeos_instance.db, "get_connection", side_effect=raise_on_user_model),
        patch.object(
            lifeos_instance, "_rebuild_user_model_db_if_corrupted", new_callable=AsyncMock
        ) as mock_rebuild,
        patch.object(
            lifeos_instance, "_verify_and_retry_backfills", new_callable=AsyncMock
        ) as mock_backfill,
        patch("main.ws_manager"),
    ):
        await _run_loop_once(lifeos_instance)

    mock_rebuild.assert_called_once()
    mock_backfill.assert_called_once()
    assert lifeos_instance._runtime_db_rebuilds == 1
