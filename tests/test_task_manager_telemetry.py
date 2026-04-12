"""
Tests for TaskManager extraction-pipeline telemetry counters.

Covers the in-memory diagnostic counters added to TaskManager so that
operators can determine *why* task extraction is producing zero results —
AI engine unavailable, events filtered out by text/marketing rules, or
genuine AI call failures — without having to grep through log files.

Test groups:
  1. No-AI skip counting — process_event increments _events_skipped_no_ai
     and the legacy _ai_engine_skip_count when ai_engine is None.
  2. Text/marketing filter counting — increments the right counter per filter.
  3. Mock AI extraction counting — successful extractions update
     _tasks_extracted and _last_extraction_time.
  4. Error counting — AI engine exceptions increment _extraction_errors.
  5. get_diagnostics() telemetry — snapshot reflects current counters and
     computes skip_rate / extraction_rate correctly.
  6. 500-skip telemetry event — _publish_telemetry is called at multiples
     of 500 skipped-no-AI events.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.task_manager.manager import TaskManager

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_LONG_TEXT = "Please review the attached document and send me feedback by end of day."
_MARKETING_PAYLOAD = {
    "from_address": "noreply@deals.bigstore.com",
    "body": (
        "You have been selected for an exclusive offer! "
        "Save 50% on your next purchase at BigStore. "
        "To unsubscribe from these promotional emails click here."
    ),
}


def _make_email_event(body: str, from_address: str = "alice@example.com") -> dict:
    """Build a minimal email.received event dict."""
    return {
        "id": "evt-test",
        "type": "email.received",
        "payload": {"from_address": from_address, "body": body},
    }


def _make_message_event(body: str) -> dict:
    """Build a minimal message.received event dict."""
    return {
        "id": "evt-msg",
        "type": "message.received",
        "payload": {"body": body},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task_manager_no_ai(db):
    """TaskManager with no AI engine — all events should be skipped."""
    return TaskManager(db=db, event_bus=None, ai_engine=None)


@pytest.fixture()
def mock_ai_engine():
    """AI engine that returns an empty list by default."""
    engine = AsyncMock()
    engine.extract_action_items = AsyncMock(return_value=[])
    return engine


@pytest.fixture()
def task_manager_with_ai(db, mock_ai_engine):
    """TaskManager wired to a mock AI engine."""
    return TaskManager(db=db, event_bus=None, ai_engine=mock_ai_engine)


# ---------------------------------------------------------------------------
# 1. No-AI skip counting
# ---------------------------------------------------------------------------


class TestNoAISkipCounting:
    """When ai_engine is None every call to process_event should increment
    _events_skipped_no_ai (and _events_processed)."""

    @pytest.mark.asyncio
    async def test_single_event_increments_skip_counter(self, task_manager_no_ai):
        """One processed event → _events_skipped_no_ai == 1."""
        await task_manager_no_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_no_ai._events_skipped_no_ai == 1

    @pytest.mark.asyncio
    async def test_events_processed_increments_before_ai_check(self, task_manager_no_ai):
        """_events_processed is incremented even when AI engine is absent."""
        await task_manager_no_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_no_ai._events_processed == 1

    @pytest.mark.asyncio
    async def test_multiple_events_accumulate(self, task_manager_no_ai):
        """Counter accumulates across multiple calls."""
        for _ in range(5):
            await task_manager_no_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_no_ai._events_skipped_no_ai == 5
        assert task_manager_no_ai._events_processed == 5

    @pytest.mark.asyncio
    async def test_legacy_skip_count_also_increments(self, task_manager_no_ai):
        """Legacy _ai_engine_skip_count kept for backward compatibility."""
        await task_manager_no_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_no_ai._ai_engine_skip_count == 1

    @pytest.mark.asyncio
    async def test_no_text_counter_not_incremented_when_no_ai(self, task_manager_no_ai):
        """The no-text counter must NOT increment: the early return for missing
        AI engine fires before text extraction is reached."""
        await task_manager_no_ai.process_event(_make_email_event("ok"))
        # No-AI skip should be counted, but text-skip should be zero because
        # we never reached that code path.
        assert task_manager_no_ai._events_skipped_no_ai == 1
        assert task_manager_no_ai._events_skipped_no_text == 0

    @pytest.mark.asyncio
    async def test_ai_engine_available_flag_is_false(self, task_manager_no_ai):
        """_ai_engine_available should reflect False after a no-AI process call."""
        await task_manager_no_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_no_ai._ai_engine_available is False

    @pytest.mark.asyncio
    async def test_last_ai_check_time_is_set(self, task_manager_no_ai):
        """_last_ai_check_time is updated even when AI engine is absent."""
        assert task_manager_no_ai._last_ai_check_time is None
        await task_manager_no_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_no_ai._last_ai_check_time is not None


# ---------------------------------------------------------------------------
# 2. Text and marketing filter counting
# ---------------------------------------------------------------------------


class TestFilterCounters:
    """Events that fail text or marketing checks should increment the correct
    counter without touching the AI counters."""

    @pytest.mark.asyncio
    async def test_short_text_increments_no_text_counter(self, task_manager_with_ai):
        """Text shorter than 20 chars → _events_skipped_no_text."""
        await task_manager_with_ai.process_event(_make_email_event("ok"))
        assert task_manager_with_ai._events_skipped_no_text == 1
        assert task_manager_with_ai._tasks_extracted == 0
        assert task_manager_with_ai._extraction_errors == 0

    @pytest.mark.asyncio
    async def test_empty_text_increments_no_text_counter(self, task_manager_with_ai):
        """Empty body → _events_skipped_no_text."""
        await task_manager_with_ai.process_event(_make_email_event(""))
        assert task_manager_with_ai._events_skipped_no_text == 1

    @pytest.mark.asyncio
    async def test_marketing_email_increments_marketing_counter(self, task_manager_with_ai):
        """Marketing sender → _events_skipped_marketing."""
        event = {
            "id": "evt-mkt",
            "type": "email.received",
            "payload": _MARKETING_PAYLOAD,
        }
        await task_manager_with_ai.process_event(event)
        assert task_manager_with_ai._events_skipped_marketing == 1
        assert task_manager_with_ai._events_skipped_no_text == 0
        assert task_manager_with_ai._tasks_extracted == 0

    @pytest.mark.asyncio
    async def test_non_actionable_type_does_not_increment_any_filter_counter(
        self, task_manager_with_ai
    ):
        """System events are dropped before counters for text/marketing filters —
        but _events_processed IS incremented."""
        await task_manager_with_ai.process_event({
            "id": "evt-sys",
            "type": "system.rule.triggered",
            "payload": {"body": _LONG_TEXT},
        })
        # The event was processed (counted) but all filter counters stay zero.
        assert task_manager_with_ai._events_processed == 1
        assert task_manager_with_ai._events_skipped_no_text == 0
        assert task_manager_with_ai._events_skipped_marketing == 0
        assert task_manager_with_ai._events_skipped_no_ai == 0


# ---------------------------------------------------------------------------
# 3. Successful extraction counting
# ---------------------------------------------------------------------------


class TestExtractionCounting:
    """When the AI returns action items the extraction counters should update."""

    @pytest.mark.asyncio
    async def test_tasks_extracted_counts_individual_tasks(
        self, task_manager_with_ai, mock_ai_engine
    ):
        """_tasks_extracted is the sum of task objects across all events, not
        just a count of events that produced at least one task."""
        mock_ai_engine.extract_action_items.return_value = [
            {"title": "Send report", "priority": "normal"},
            {"title": "Schedule follow-up", "priority": "low"},
        ]
        await task_manager_with_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_with_ai._tasks_extracted == 2

    @pytest.mark.asyncio
    async def test_tasks_extracted_accumulates_across_events(
        self, task_manager_with_ai, mock_ai_engine
    ):
        """Counter accumulates when multiple events each produce tasks."""
        mock_ai_engine.extract_action_items.return_value = [
            {"title": "Task A", "priority": "normal"}
        ]
        await task_manager_with_ai.process_event(_make_email_event(_LONG_TEXT))
        await task_manager_with_ai.process_event(_make_message_event(_LONG_TEXT))
        assert task_manager_with_ai._tasks_extracted == 2

    @pytest.mark.asyncio
    async def test_last_extraction_time_updated_on_success(
        self, task_manager_with_ai, mock_ai_engine
    ):
        """_last_extraction_time is set after a successful extraction."""
        assert task_manager_with_ai._last_extraction_time is None
        mock_ai_engine.extract_action_items.return_value = [
            {"title": "Follow up with client", "priority": "high"}
        ]
        await task_manager_with_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_with_ai._last_extraction_time is not None

    @pytest.mark.asyncio
    async def test_last_extraction_time_not_set_when_no_tasks(
        self, task_manager_with_ai, mock_ai_engine
    ):
        """_last_extraction_time remains None when AI returns empty list."""
        mock_ai_engine.extract_action_items.return_value = []
        await task_manager_with_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_with_ai._last_extraction_time is None

    @pytest.mark.asyncio
    async def test_events_processed_increments_regardless_of_extraction_result(
        self, task_manager_with_ai, mock_ai_engine
    ):
        """_events_processed reflects total calls even when AI returns nothing."""
        mock_ai_engine.extract_action_items.return_value = []
        await task_manager_with_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_with_ai._events_processed == 1
        assert task_manager_with_ai._tasks_extracted == 0


# ---------------------------------------------------------------------------
# 4. Extraction error counting
# ---------------------------------------------------------------------------


class TestExtractionErrorCounting:
    """AI engine exceptions should increment _extraction_errors without crashing."""

    @pytest.mark.asyncio
    async def test_ai_exception_increments_error_counter(
        self, task_manager_with_ai, mock_ai_engine
    ):
        """RuntimeError from AI engine → _extraction_errors == 1."""
        mock_ai_engine.extract_action_items.side_effect = RuntimeError("Model down")
        await task_manager_with_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_with_ai._extraction_errors == 1

    @pytest.mark.asyncio
    async def test_ai_exception_does_not_increment_tasks_extracted(
        self, task_manager_with_ai, mock_ai_engine
    ):
        """On AI failure, _tasks_extracted must remain zero."""
        mock_ai_engine.extract_action_items.side_effect = ValueError("Parse error")
        await task_manager_with_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_with_ai._tasks_extracted == 0

    @pytest.mark.asyncio
    async def test_error_counter_accumulates(self, task_manager_with_ai, mock_ai_engine):
        """Multiple failures accumulate in the counter."""
        mock_ai_engine.extract_action_items.side_effect = RuntimeError("Timeout")
        for _ in range(3):
            await task_manager_with_ai.process_event(_make_email_event(_LONG_TEXT))
        assert task_manager_with_ai._extraction_errors == 3


# ---------------------------------------------------------------------------
# 5. get_diagnostics() telemetry
# ---------------------------------------------------------------------------


class TestGetDiagnosticsTelemetry:
    """get_diagnostics() should expose all telemetry counters and computed rates."""

    def test_telemetry_present_in_diagnostics(self, task_manager_no_ai):
        """extraction_telemetry key is always present in get_diagnostics()."""
        result = task_manager_no_ai.get_diagnostics()
        assert "extraction_telemetry" in result

    def test_telemetry_all_fields_present(self, task_manager_no_ai):
        """All expected sub-fields are present in extraction_telemetry."""
        telemetry = task_manager_no_ai.get_diagnostics()["extraction_telemetry"]
        required = {
            "events_processed", "events_skipped_no_ai", "events_skipped_no_text",
            "events_skipped_marketing", "tasks_extracted", "extraction_errors",
            "last_extraction_time", "last_ai_check_time",
            "ai_engine_available", "ai_engine_type", "skip_rate", "extraction_rate",
        }
        assert required.issubset(set(telemetry.keys()))

    @pytest.mark.asyncio
    async def test_skip_rate_reflects_no_ai_skips(self, task_manager_no_ai):
        """skip_rate = events_skipped_no_ai / events_processed."""
        for _ in range(4):
            await task_manager_no_ai.process_event(_make_email_event(_LONG_TEXT))
        telemetry = task_manager_no_ai.get_diagnostics()["extraction_telemetry"]
        assert telemetry["events_processed"] == 4
        assert telemetry["events_skipped_no_ai"] == 4
        assert telemetry["skip_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_extraction_rate_tasks_per_ai_eligible_event(
        self, task_manager_with_ai, mock_ai_engine
    ):
        """extraction_rate = tasks_extracted / (events_processed - skipped_no_ai).

        Two events each produce 3 tasks → rate == 3.0 tasks/event.
        """
        mock_ai_engine.extract_action_items.return_value = [
            {"title": "Task A", "priority": "normal"},
            {"title": "Task B", "priority": "low"},
            {"title": "Task C", "priority": "high"},
        ]
        await task_manager_with_ai.process_event(_make_email_event(_LONG_TEXT))
        await task_manager_with_ai.process_event(_make_message_event(_LONG_TEXT))
        telemetry = task_manager_with_ai.get_diagnostics()["extraction_telemetry"]
        assert telemetry["tasks_extracted"] == 6
        # 6 tasks / 2 AI-eligible events = 3.0
        assert telemetry["extraction_rate"] == 3.0

    def test_ai_engine_type_reflects_class_name(self, db):
        """ai_engine_type should return the class name of the engine."""
        class FakeAIEngine:
            pass

        tm = TaskManager(db=db, event_bus=None, ai_engine=FakeAIEngine())
        telemetry = tm.get_diagnostics()["extraction_telemetry"]
        assert telemetry["ai_engine_available"] is True
        assert telemetry["ai_engine_type"] == "FakeAIEngine"

    def test_ai_engine_type_none_when_no_engine(self, task_manager_no_ai):
        """ai_engine_type is None when no engine is configured."""
        telemetry = task_manager_no_ai.get_diagnostics()["extraction_telemetry"]
        assert telemetry["ai_engine_available"] is False
        assert telemetry["ai_engine_type"] is None

    @pytest.mark.asyncio
    async def test_extraction_rate_zero_when_no_tasks_extracted(
        self, task_manager_with_ai, mock_ai_engine
    ):
        """extraction_rate stays 0.0 when AI engine returns nothing."""
        mock_ai_engine.extract_action_items.return_value = []
        await task_manager_with_ai.process_event(_make_email_event(_LONG_TEXT))
        telemetry = task_manager_with_ai.get_diagnostics()["extraction_telemetry"]
        assert telemetry["extraction_rate"] == 0.0


# ---------------------------------------------------------------------------
# 6. 500-skip telemetry event threshold
# ---------------------------------------------------------------------------


class TestDegradedTelemetryThreshold:
    """process_event publishes a 'system.task_extraction.degraded' NATS event
    at every multiple of 500 events skipped due to missing AI engine."""

    @pytest.mark.asyncio
    async def test_no_publish_before_500_skips(self, db):
        """No telemetry event should be published before the 500-skip mark."""
        published: list[dict] = []

        class MockBus:
            is_connected = True

            async def publish(self, event_type: str, payload: dict, **kwargs):
                published.append({"type": event_type, "payload": payload})

        tm = TaskManager(db=db, event_bus=MockBus(), ai_engine=None)
        # Process 499 events — just under the threshold
        for _ in range(499):
            await tm.process_event(_make_email_event(_LONG_TEXT))

        degraded_events = [e for e in published if e["type"] == "system.task_extraction.degraded"]
        assert len(degraded_events) == 0

    @pytest.mark.asyncio
    async def test_publish_at_exactly_500_skips(self, db):
        """A 'system.task_extraction.degraded' event should fire at 500 skips."""
        published: list[dict] = []

        class MockBus:
            is_connected = True

            async def publish(self, event_type: str, payload: dict, **kwargs):
                published.append({"type": event_type, "payload": payload})

        tm = TaskManager(db=db, event_bus=MockBus(), ai_engine=None)
        for _ in range(500):
            await tm.process_event(_make_email_event(_LONG_TEXT))

        degraded_events = [e for e in published if e["type"] == "system.task_extraction.degraded"]
        assert len(degraded_events) == 1

    @pytest.mark.asyncio
    async def test_publish_again_at_1000_skips(self, db):
        """Telemetry event fires at every multiple of 500 (500, 1000, …)."""
        published: list[dict] = []

        class MockBus:
            is_connected = True

            async def publish(self, event_type: str, payload: dict, **kwargs):
                published.append({"type": event_type, "payload": payload})

        tm = TaskManager(db=db, event_bus=MockBus(), ai_engine=None)
        for _ in range(1000):
            await tm.process_event(_make_email_event(_LONG_TEXT))

        degraded_events = [e for e in published if e["type"] == "system.task_extraction.degraded"]
        assert len(degraded_events) == 2  # once at 500, once at 1000

    @pytest.mark.asyncio
    async def test_degraded_event_payload_structure(self, db):
        """Payload of the degraded event should contain diagnostic counters."""
        published: list[dict] = []

        class MockBus:
            is_connected = True

            async def publish(self, event_type: str, payload: dict, **kwargs):
                published.append({"type": event_type, "payload": payload})

        tm = TaskManager(db=db, event_bus=MockBus(), ai_engine=None)
        for _ in range(500):
            await tm.process_event(_make_email_event(_LONG_TEXT))

        degraded = next(e for e in published if e["type"] == "system.task_extraction.degraded")
        payload = degraded["payload"]
        assert payload["events_skipped_no_ai"] == 500
        assert payload["events_processed"] == 500
        assert payload["skip_rate"] == 1.0
        assert payload["reason"] == "ai_engine_unavailable"

    @pytest.mark.asyncio
    async def test_no_publish_without_event_bus(self, task_manager_no_ai):
        """When event_bus is None the 500-skip threshold must not raise."""
        # task_manager_no_ai has event_bus=None
        for _ in range(500):
            await task_manager_no_ai.process_event(_make_email_event(_LONG_TEXT))
        # No exception raised — pass
        assert task_manager_no_ai._events_skipped_no_ai == 500
