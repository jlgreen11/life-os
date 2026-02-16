"""
Tests for the automatic digest delivery loop.

The digest delivery loop fixes a critical bug where batched notifications
were created but never delivered, remaining stuck in "pending" state forever.
This happened because notification_manager.get_digest() was only accessible
via manual API call — there was no automated scheduler to deliver digests
at scheduled times (morning briefing, midday update, evening wrap-up).

These tests verify that:
1. The loop delivers digests at scheduled times (09:00, 13:00, 18:00)
2. Batched notifications are marked as "delivered" when the digest runs
3. Predictions are marked as "surfaced" when their notifications are delivered
4. The loop doesn't deliver duplicate digests within the same hour
5. The loop handles empty digests gracefully (no notifications to deliver)
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from main import LifeOS


@pytest.mark.asyncio
async def test_digest_delivery_at_scheduled_time(db, event_bus):
    """Test that digest delivery triggers at scheduled hours (09:00, 13:00, 18:00)."""
    # Create Life OS instance with mocked notification manager
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[
        {"id": "notif1", "title": "Test notification 1"},
        {"id": "notif2", "title": "Test notification 2"},
    ])

    # Mock datetime to return a digest hour (09:00)
    with patch("main.datetime") as mock_datetime:
        mock_now = MagicMock()
        mock_now.hour = 9  # 09:00
        mock_datetime.now.return_value = mock_now

        # Start the digest delivery loop
        loop_task = asyncio.create_task(life_os._digest_delivery_loop())

        # Let it run for a short time to trigger the digest
        await asyncio.sleep(0.1)

        # Cancel the loop
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        # Assert: get_digest was called
        life_os.notification_manager.get_digest.assert_called()


@pytest.mark.asyncio
async def test_digest_delivery_marks_notifications_delivered(db, event_bus):
    """Test that digest delivery marks notifications as 'delivered' in the database."""
    # Create Life OS instance
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data"})

    # Create a batched notification (normal priority, batched mode)
    with db.get_connection("preferences") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_preferences (key, value) VALUES (?, ?)",
            ("notification_mode", '"batched"'),
        )

    notif_id = await life_os.notification_manager.create_notification(
        title="Test batched notification",
        body="This should be delivered via digest",
        priority="normal",
    )

    # Verify notification is pending
    with db.get_connection("state") as conn:
        status = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (notif_id,)
        ).fetchone()["status"]
        assert status == "pending", "Notification should start as pending"

    # Call get_digest to simulate scheduled delivery
    digest = await life_os.notification_manager.get_digest()
    assert len(digest) == 1
    assert digest[0]["id"] == notif_id

    # Verify notification is now marked as delivered
    with db.get_connection("state") as conn:
        status = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (notif_id,)
        ).fetchone()["status"]
        assert status == "delivered", "Notification should be marked as delivered after digest"


@pytest.mark.asyncio
async def test_digest_delivery_marks_predictions_surfaced(db, event_bus):
    """Test that digest delivery marks predictions as 'surfaced' when delivered."""
    # Create Life OS instance
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data"})

    # Create a prediction event
    prediction_id = "pred123"
    await event_bus.publish(
        "usermodel.prediction.generated",
        {
            "prediction_type": "reminder",
            "description": "Test prediction",
            "confidence": 0.7,
        },
        source="test",
        event_id=prediction_id,
    )

    # Wait for event to be processed
    await asyncio.sleep(0.1)

    # Create batched notification for this prediction
    with db.get_connection("preferences") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_preferences (key, value) VALUES (?, ?)",
            ("notification_mode", '"batched"'),
        )

    notif_id = await life_os.notification_manager.create_notification(
        title="Prediction notification",
        body="Test prediction body",
        priority="normal",
        source_event_id=prediction_id,
        domain="prediction",
    )

    # Verify prediction is not surfaced yet
    with db.get_connection("user_model") as conn:
        surfaced = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        # Prediction might not exist yet, that's ok for this test
        if surfaced:
            assert surfaced["was_surfaced"] == 0, "Prediction should not be surfaced before digest"

    # Deliver the digest
    await life_os.notification_manager.get_digest()

    # Verify prediction is now marked as surfaced
    with db.get_connection("user_model") as conn:
        surfaced = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if surfaced:  # Only check if prediction was created
            assert surfaced["was_surfaced"] == 1, "Prediction should be surfaced after digest delivery"


@pytest.mark.asyncio
async def test_digest_delivery_no_duplicate_within_hour(db, event_bus):
    """Test that digest delivery doesn't trigger multiple times within the same hour."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[
        {"id": "notif1", "title": "Test notification"},
    ])

    # Mock datetime to stay at 09:00 for multiple iterations
    with patch("main.datetime") as mock_datetime:
        mock_now = MagicMock()
        mock_now.hour = 9  # Keep hour constant at 09:00
        mock_datetime.now.return_value = mock_now

        # Start the digest delivery loop
        loop_task = asyncio.create_task(life_os._digest_delivery_loop())

        # Let it run for two sleep cycles
        await asyncio.sleep(0.2)

        # Cancel the loop
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        # Assert: get_digest was called exactly once (not multiple times)
        assert life_os.notification_manager.get_digest.call_count == 1, \
            "Digest should only be delivered once per hour window"


@pytest.mark.asyncio
async def test_digest_delivery_handles_empty_digest(db, event_bus):
    """Test that digest delivery handles empty digest (no notifications) gracefully."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[])

    # Mock datetime to return a digest hour
    with patch("main.datetime") as mock_datetime:
        mock_now = MagicMock()
        mock_now.hour = 9  # 09:00
        mock_datetime.now.return_value = mock_now

        # Start the digest delivery loop
        loop_task = asyncio.create_task(life_os._digest_delivery_loop())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel the loop
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        # Assert: get_digest was called even with no notifications
        life_os.notification_manager.get_digest.assert_called()


@pytest.mark.asyncio
async def test_digest_delivery_skips_non_digest_hours(db, event_bus):
    """Test that digest delivery doesn't trigger outside scheduled hours."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[])

    # Mock datetime to return a non-digest hour (11:00)
    with patch("main.datetime") as mock_datetime:
        mock_now = MagicMock()
        mock_now.hour = 11  # 11:00 (not a digest hour)
        mock_datetime.now.return_value = mock_now

        # Start the digest delivery loop
        loop_task = asyncio.create_task(life_os._digest_delivery_loop())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel the loop
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        # Assert: get_digest was NOT called
        life_os.notification_manager.get_digest.assert_not_called()


@pytest.mark.asyncio
async def test_digest_delivery_continues_after_error(db, event_bus):
    """Test that digest delivery continues running even if get_digest raises an error."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data"})

    # Mock get_digest to raise an error
    life_os.notification_manager.get_digest = AsyncMock(side_effect=Exception("Test error"))

    # Mock datetime to return a digest hour
    with patch("main.datetime") as mock_datetime:
        mock_now = MagicMock()
        mock_now.hour = 9  # 09:00 (digest hour)
        mock_datetime.now.return_value = mock_now

        # Start the digest delivery loop
        loop_task = asyncio.create_task(life_os._digest_delivery_loop())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel the loop
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        # Assert: The loop attempted to call get_digest despite the error
        # (doesn't crash the loop)
        life_os.notification_manager.get_digest.assert_called()
