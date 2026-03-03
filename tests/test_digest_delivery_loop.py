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
6. Digest items are broadcast via WebSocket to connected dashboard clients
7. A digest summary message is broadcast after individual notifications
8. No WebSocket broadcast happens when digest is empty
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from main import LifeOS


def _make_utc_datetime(hour):
    """Return a timezone-aware UTC datetime at the given hour."""
    return datetime(2026, 2, 16, hour, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_digest_delivery_at_scheduled_time(db, event_bus):
    """Test that digest delivery triggers at scheduled hours (09:00, 13:00, 18:00)."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data", "timezone": "UTC"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[
        {"id": "notif1", "title": "Test notification 1"},
        {"id": "notif2", "title": "Test notification 2"},
    ])

    with patch("main.datetime") as mock_datetime:
        mock_datetime.now.return_value = _make_utc_datetime(9)
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)

        loop_task = asyncio.create_task(life_os._digest_delivery_loop())
        await asyncio.sleep(0.1)
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        life_os.notification_manager.get_digest.assert_called()


@pytest.mark.asyncio
async def test_digest_delivery_marks_notifications_delivered(db, event_bus):
    """Test that digest delivery marks notifications as 'delivered' in the database."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data", "timezone": "UTC"})

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

    with db.get_connection("state") as conn:
        status = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (notif_id,)
        ).fetchone()["status"]
        assert status == "pending", "Notification should start as pending"

    digest = await life_os.notification_manager.get_digest()
    assert len(digest) == 1
    assert digest[0]["id"] == notif_id

    with db.get_connection("state") as conn:
        status = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (notif_id,)
        ).fetchone()["status"]
        assert status == "delivered", "Notification should be marked as delivered after digest"


@pytest.mark.asyncio
async def test_digest_delivery_marks_predictions_surfaced(db, event_bus):
    """Test that digest delivery marks predictions as 'surfaced' when delivered."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data", "timezone": "UTC"})

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

    await asyncio.sleep(0.1)

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

    with db.get_connection("user_model") as conn:
        surfaced = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if surfaced:
            assert surfaced["was_surfaced"] == 0, "Prediction should not be surfaced before digest"

    await life_os.notification_manager.get_digest()

    with db.get_connection("user_model") as conn:
        surfaced = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if surfaced:
            assert surfaced["was_surfaced"] == 1, "Prediction should be surfaced after digest delivery"


@pytest.mark.asyncio
async def test_digest_delivery_no_duplicate_within_hour(db, event_bus):
    """Test that digest delivery doesn't trigger multiple times within the same hour."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data", "timezone": "UTC"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[
        {"id": "notif1", "title": "Test notification"},
    ])

    with patch("main.datetime") as mock_datetime:
        mock_datetime.now.return_value = _make_utc_datetime(9)
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)

        loop_task = asyncio.create_task(life_os._digest_delivery_loop())
        await asyncio.sleep(0.2)
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        assert life_os.notification_manager.get_digest.call_count == 1, \
            "Digest should only be delivered once per hour window"


@pytest.mark.asyncio
async def test_digest_delivery_skips_non_digest_hours(db, event_bus):
    """Test that digest delivery doesn't trigger outside scheduled hours."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data", "timezone": "UTC"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[])

    with patch("main.datetime") as mock_datetime:
        mock_datetime.now.return_value = _make_utc_datetime(11)
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)

        loop_task = asyncio.create_task(life_os._digest_delivery_loop())
        await asyncio.sleep(0.1)
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        life_os.notification_manager.get_digest.assert_not_called()


@pytest.mark.asyncio
async def test_digest_delivery_handles_empty_digest(db, event_bus):
    """Test that digest delivery handles empty digest (no notifications) gracefully."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data", "timezone": "UTC"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[])

    with patch("main.datetime") as mock_datetime:
        mock_datetime.now.return_value = _make_utc_datetime(9)
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)

        loop_task = asyncio.create_task(life_os._digest_delivery_loop())
        await asyncio.sleep(0.1)
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        life_os.notification_manager.get_digest.assert_called()


@pytest.mark.asyncio
async def test_digest_broadcasts_notifications_via_websocket(db, event_bus):
    """Test that digest items are broadcast to dashboard clients via WebSocket."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data", "timezone": "UTC"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[
        {"id": "notif1", "title": "Morning email summary", "source_event_id": "ev1"},
        {"id": "notif2", "title": "Calendar reminder", "source_event_id": "ev2"},
    ])

    with patch("main.datetime") as mock_datetime, \
         patch("main.ws_manager") as mock_ws:
        mock_datetime.now.return_value = _make_utc_datetime(9)
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        mock_ws.broadcast = AsyncMock()

        loop_task = asyncio.create_task(life_os._digest_delivery_loop())
        await asyncio.sleep(0.1)
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        # Should broadcast each notification item + one digest summary = 3 calls
        assert mock_ws.broadcast.call_count == 3

        # Verify individual notification broadcasts
        calls = mock_ws.broadcast.call_args_list
        assert calls[0].args[0] == {
            "type": "notification",
            "title": "Morning email summary",
            "source_event_id": "ev1",
        }
        assert calls[1].args[0] == {
            "type": "notification",
            "title": "Calendar reminder",
            "source_event_id": "ev2",
        }

        # Verify digest summary broadcast
        assert calls[2].args[0] == {
            "type": "digest",
            "count": 2,
        }


@pytest.mark.asyncio
async def test_digest_no_broadcast_when_empty(db, event_bus):
    """Test that no WebSocket broadcast happens when the digest is empty."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data", "timezone": "UTC"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[])

    with patch("main.datetime") as mock_datetime, \
         patch("main.ws_manager") as mock_ws:
        mock_datetime.now.return_value = _make_utc_datetime(13)
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        mock_ws.broadcast = AsyncMock()

        loop_task = asyncio.create_task(life_os._digest_delivery_loop())
        await asyncio.sleep(0.1)
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        # get_digest was called but returned empty, so no broadcast should happen
        life_os.notification_manager.get_digest.assert_called()
        mock_ws.broadcast.assert_not_called()


@pytest.mark.asyncio
async def test_digest_broadcast_error_does_not_block_delivery(db, event_bus):
    """Test that a WebSocket broadcast error doesn't prevent digest delivery from completing."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data", "timezone": "UTC"})
    life_os.notification_manager.get_digest = AsyncMock(return_value=[
        {"id": "notif1", "title": "Test notification", "source_event_id": "ev1"},
    ])

    with patch("main.datetime") as mock_datetime, \
         patch("main.ws_manager") as mock_ws:
        mock_datetime.now.return_value = _make_utc_datetime(18)
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        # Simulate WebSocket broadcast failure
        mock_ws.broadcast = AsyncMock(side_effect=Exception("WebSocket error"))

        loop_task = asyncio.create_task(life_os._digest_delivery_loop())
        await asyncio.sleep(0.1)
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        # get_digest should still have been called successfully despite broadcast failure
        life_os.notification_manager.get_digest.assert_called()
        # broadcast was attempted (even though it failed)
        mock_ws.broadcast.assert_called()


@pytest.mark.asyncio
async def test_digest_delivery_continues_after_error(db, event_bus):
    """Test that digest delivery continues running even if get_digest raises an error."""
    life_os = LifeOS(db=db, event_bus=event_bus, config={"data_dir": "./test_data", "timezone": "UTC"})
    life_os.notification_manager.get_digest = AsyncMock(side_effect=Exception("Test error"))

    with patch("main.datetime") as mock_datetime:
        mock_datetime.now.return_value = _make_utc_datetime(9)
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)

        loop_task = asyncio.create_task(life_os._digest_delivery_loop())
        await asyncio.sleep(0.1)
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

        life_os.notification_manager.get_digest.assert_called()
