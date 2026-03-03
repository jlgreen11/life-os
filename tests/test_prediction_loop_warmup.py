"""
Tests for background loop warmup delays and NATS error handling.

Verifies that:
  1. _prediction_loop() waits 180s before calling generate_predictions.
  2. _insight_loop() waits 180s before calling generate_insights.
  3. _semantic_inference_loop() waits 180s before calling run_all_inference.
  4. _behavioral_accuracy_loop() waits 180s before calling run_inference_cycle.
  5. _task_overdue_loop() still creates notifications when event_bus.publish
     raises (NATS outage resilience).
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.notification_manager.manager import NotificationManager
from services.task_manager.manager import TaskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lifeos_stub(**overrides):
    """Create a minimal object that has the attributes _prediction_loop etc. need.

    Instead of instantiating the full LifeOS class (which requires NATS,
    Ollama, config files, etc.), we build a lightweight stub with only the
    fields referenced by the individual loop methods and then bind the
    unbound method to this stub.
    """
    stub = MagicMock()
    stub.shutdown_event = asyncio.Event()

    # Default mocks that individual tests can override
    stub.prediction_engine = MagicMock()
    stub.prediction_engine.generate_predictions = AsyncMock(return_value=[])

    stub.notification_manager = MagicMock(spec=NotificationManager)
    stub.notification_manager.create_notification = AsyncMock(return_value="notif-id")
    stub.notification_manager.auto_resolve_stale_predictions = AsyncMock(return_value=0)
    stub.notification_manager.auto_resolve_filtered_predictions = MagicMock(return_value=0)

    stub.insight_engine = MagicMock()
    stub.insight_engine.generate_insights = AsyncMock(return_value=[])

    stub.source_weight_manager = MagicMock()
    stub.source_weight_manager.bulk_recalculate_drift = MagicMock()

    stub.semantic_fact_inferrer = MagicMock()
    stub.semantic_fact_inferrer.run_all_inference = MagicMock()

    stub.behavioral_tracker = MagicMock()
    stub.behavioral_tracker.run_inference_cycle = AsyncMock(
        return_value={"marked_accurate": 0, "marked_inaccurate": 0}
    )

    stub.event_bus = MagicMock()
    stub.event_bus.publish = AsyncMock()

    stub.task_manager = MagicMock()
    stub.task_manager.get_overdue_tasks = MagicMock(return_value=[])

    stub._notified_overdue_tasks = set()

    # Apply any overrides
    for key, value in overrides.items():
        setattr(stub, key, value)

    return stub


def _bind_loop(loop_method, stub):
    """Bind an unbound LifeOS loop method to a stub instance.

    The loop methods are defined as ``async def _prediction_loop(self)``
    on the LifeOS class.  We import the class and call the method with
    our stub as ``self``.
    """
    from main import LifeOS

    method = getattr(LifeOS, loop_method)
    return method(stub)


# ---------------------------------------------------------------------------
# Warmup delay tests
# ---------------------------------------------------------------------------


class TestPredictionLoopWarmup:
    """Verify _prediction_loop sleeps 180s before first generate_predictions call."""

    @pytest.mark.asyncio
    async def test_prediction_loop_warmup_delay(self):
        """The loop should sleep 180s before calling generate_predictions."""
        stub = _make_lifeos_stub()
        sleep_durations: list[float] = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(duration):
            """Record sleep durations and stop after the inter-cycle sleep."""
            sleep_durations.append(duration)
            # Stop on the inter-cycle sleep (900s), not the warmup (180s)
            if duration == 900:
                stub.shutdown_event.set()
            await original_sleep(0)  # yield control

        with patch("asyncio.sleep", side_effect=tracking_sleep):
            await asyncio.wait_for(_bind_loop("_prediction_loop", stub), timeout=5)

        # The first sleep should be the 180s warmup
        assert len(sleep_durations) >= 2
        assert sleep_durations[0] == 180
        assert sleep_durations[1] == 900

        # generate_predictions should have been called once (after warmup)
        stub.prediction_engine.generate_predictions.assert_called_once()

    @pytest.mark.asyncio
    async def test_prediction_loop_no_call_during_warmup(self):
        """generate_predictions must NOT be called before the warmup sleep."""
        stub = _make_lifeos_stub()
        call_order: list[str] = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(duration):
            call_order.append(f"sleep({duration})")
            if duration == 900:
                stub.shutdown_event.set()
            await original_sleep(0)

        async def tracking_generate(context):
            call_order.append("generate_predictions")
            return []

        stub.prediction_engine.generate_predictions = AsyncMock(side_effect=tracking_generate)

        with patch("asyncio.sleep", side_effect=tracking_sleep):
            await asyncio.wait_for(_bind_loop("_prediction_loop", stub), timeout=5)

        # sleep(180) must come before generate_predictions
        assert call_order[0] == "sleep(180)"
        assert "generate_predictions" in call_order
        assert call_order.index("sleep(180)") < call_order.index("generate_predictions")


class TestInsightLoopWarmup:
    """Verify _insight_loop sleeps 180s before first generate_insights call."""

    @pytest.mark.asyncio
    async def test_insight_loop_warmup_delay(self):
        """The insight loop should sleep 180s before the first cycle."""
        stub = _make_lifeos_stub()
        sleep_durations: list[float] = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(duration):
            sleep_durations.append(duration)
            # Stop on the inter-cycle sleep (900s)
            if duration == 900:
                stub.shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=tracking_sleep):
            await asyncio.wait_for(_bind_loop("_insight_loop", stub), timeout=5)

        assert sleep_durations[0] == 180
        stub.insight_engine.generate_insights.assert_called_once()


class TestSemanticInferenceLoopWarmup:
    """Verify _semantic_inference_loop sleeps 180s before first run_all_inference call."""

    @pytest.mark.asyncio
    async def test_semantic_inference_loop_warmup_delay(self):
        """The semantic inference loop should sleep 180s before the first cycle."""
        stub = _make_lifeos_stub()
        sleep_durations: list[float] = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(duration):
            sleep_durations.append(duration)
            # Stop on the inter-cycle sleep (3600s)
            if duration == 3600:
                stub.shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=tracking_sleep):
            await asyncio.wait_for(_bind_loop("_semantic_inference_loop", stub), timeout=5)

        assert sleep_durations[0] == 180
        stub.semantic_fact_inferrer.run_all_inference.assert_called_once()


class TestBehavioralAccuracyLoopWarmup:
    """Verify _behavioral_accuracy_loop sleeps 180s before first run_inference_cycle call."""

    @pytest.mark.asyncio
    async def test_behavioral_accuracy_loop_warmup_delay(self):
        """The behavioral accuracy loop should sleep 180s before the first cycle."""
        stub = _make_lifeos_stub()
        sleep_durations: list[float] = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(duration):
            sleep_durations.append(duration)
            # Stop on the inter-cycle sleep (900s)
            if duration == 900:
                stub.shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=tracking_sleep):
            await asyncio.wait_for(_bind_loop("_behavioral_accuracy_loop", stub), timeout=5)

        assert sleep_durations[0] == 180
        stub.behavioral_tracker.run_inference_cycle.assert_called_once()


# ---------------------------------------------------------------------------
# NATS error handling in task overdue loop
# ---------------------------------------------------------------------------


def _insert_task(db, task_id=None, title="Test task", status="pending",
                 due_date=None, priority="normal", domain="personal"):
    """Helper to insert a task row directly into the state database."""
    if task_id is None:
        task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO tasks (id, title, description, status, source, domain,
                                  priority, due_date, created_at, updated_at)
               VALUES (?, ?, '', ?, 'manual', ?, ?, ?, ?, ?)""",
            (task_id, title, status, domain, priority, due_date, now, now),
        )
    return task_id


class TestTaskOverdueLoopNATSResilience:
    """Verify _task_overdue_loop creates notifications even when event_bus.publish fails."""

    @pytest.mark.asyncio
    async def test_notification_created_when_publish_raises(self, db, event_bus):
        """When event_bus.publish raises, the notification should still be created."""
        past_due = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        task_id = _insert_task(db, title="Overdue report", due_date=past_due,
                               priority="high", domain="work")

        task_manager = TaskManager(db, event_bus=MagicMock(), ai_engine=None)
        mock_notification_mgr = MagicMock(spec=NotificationManager)
        mock_notification_mgr.create_notification = AsyncMock(return_value="notif-id")

        # Make event_bus.publish raise to simulate NATS outage
        failing_event_bus = MagicMock()
        failing_event_bus.publish = AsyncMock(side_effect=ConnectionError("NATS disconnected"))

        stub = _make_lifeos_stub(
            task_manager=task_manager,
            notification_manager=mock_notification_mgr,
            event_bus=failing_event_bus,
        )

        loop_iterations = 0
        original_sleep = asyncio.sleep

        async def stop_after_first(duration):
            nonlocal loop_iterations
            loop_iterations += 1
            stub.shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=stop_after_first):
            await asyncio.wait_for(_bind_loop("_task_overdue_loop", stub), timeout=5)

        # event_bus.publish was called and raised — but notification should still happen
        failing_event_bus.publish.assert_called_once()
        mock_notification_mgr.create_notification.assert_called_once()

        # Verify notification content
        call_kwargs = mock_notification_mgr.create_notification.call_args
        assert "Overdue report" in call_kwargs.kwargs.get("title", call_kwargs[1].get("title", ""))

    @pytest.mark.asyncio
    async def test_notification_still_created_on_nats_timeout(self, db, event_bus):
        """A NATS timeout should not prevent the overdue notification from being created."""
        past_due = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        task_id = _insert_task(db, title="Call client", due_date=past_due,
                               priority="normal", domain="personal")

        task_manager = TaskManager(db, event_bus=MagicMock(), ai_engine=None)
        mock_notification_mgr = MagicMock(spec=NotificationManager)
        mock_notification_mgr.create_notification = AsyncMock(return_value="notif-id")

        # Simulate a NATS timeout
        failing_event_bus = MagicMock()
        failing_event_bus.publish = AsyncMock(side_effect=TimeoutError("NATS publish timed out"))

        stub = _make_lifeos_stub(
            task_manager=task_manager,
            notification_manager=mock_notification_mgr,
            event_bus=failing_event_bus,
        )

        original_sleep = asyncio.sleep

        async def stop_after_first(duration):
            stub.shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=stop_after_first):
            await asyncio.wait_for(_bind_loop("_task_overdue_loop", stub), timeout=5)

        # Notification should have been created despite NATS failure
        mock_notification_mgr.create_notification.assert_called_once()
        call_kwargs = mock_notification_mgr.create_notification.call_args
        assert call_kwargs.kwargs.get("priority", call_kwargs[1].get("priority", "")) == "high"

    @pytest.mark.asyncio
    async def test_task_added_to_notified_set_despite_publish_failure(self, db, event_bus):
        """The task ID should be added to _notified_overdue_tasks even if publish fails,
        so that subsequent loop iterations don't re-notify for the same task."""
        past_due = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        task_id = _insert_task(db, title="Buy groceries", due_date=past_due)

        task_manager = TaskManager(db, event_bus=MagicMock(), ai_engine=None)
        mock_notification_mgr = MagicMock(spec=NotificationManager)
        mock_notification_mgr.create_notification = AsyncMock(return_value="notif-id")

        failing_event_bus = MagicMock()
        failing_event_bus.publish = AsyncMock(side_effect=RuntimeError("NATS broken"))

        stub = _make_lifeos_stub(
            task_manager=task_manager,
            notification_manager=mock_notification_mgr,
            event_bus=failing_event_bus,
        )

        original_sleep = asyncio.sleep

        async def stop_after_first(duration):
            stub.shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=stop_after_first):
            await asyncio.wait_for(_bind_loop("_task_overdue_loop", stub), timeout=5)

        # Task should be in the notified set
        assert task_id in stub._notified_overdue_tasks

    @pytest.mark.asyncio
    async def test_happy_path_both_publish_and_notification(self, db, event_bus):
        """When NATS is healthy, both the event and notification should succeed."""
        past_due = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
        task_id = _insert_task(db, title="Submit timesheet", due_date=past_due,
                               priority="high", domain="work")

        task_manager = TaskManager(db, event_bus=MagicMock(), ai_engine=None)
        mock_notification_mgr = MagicMock(spec=NotificationManager)
        mock_notification_mgr.create_notification = AsyncMock(return_value="notif-id")

        healthy_event_bus = MagicMock()
        healthy_event_bus.publish = AsyncMock(return_value="event-id")

        stub = _make_lifeos_stub(
            task_manager=task_manager,
            notification_manager=mock_notification_mgr,
            event_bus=healthy_event_bus,
        )

        original_sleep = asyncio.sleep

        async def stop_after_first(duration):
            stub.shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=stop_after_first):
            await asyncio.wait_for(_bind_loop("_task_overdue_loop", stub), timeout=5)

        # Both should succeed
        healthy_event_bus.publish.assert_called_once()
        mock_notification_mgr.create_notification.assert_called_once()
        assert task_id in stub._notified_overdue_tasks
