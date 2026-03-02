"""
Tests for observability logging in signal extractors and task manager.

Verifies that previously-silent exception handlers now emit debug/warning
log messages when encountering malformed data, following the pattern
established in PRs #364 and #366.
"""

import logging

import pytest

from models.core import EventType
from services.signal_extractor.cadence import CadenceExtractor
from services.signal_extractor.spatial import SpatialExtractor
from services.signal_extractor.temporal import TemporalExtractor
from services.task_manager.manager import TaskManager


class TestTemporalExtractorObservability:
    """Verify TemporalExtractor logs debug messages for malformed timestamps."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        """Create a TemporalExtractor with test database."""
        return TemporalExtractor(db, user_model_store)

    def test_logs_malformed_outer_timestamp(self, extractor, caplog):
        """Malformed event timestamp should log a debug message and return empty signals."""
        event = {
            "id": "test-temporal-bad-ts",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "not-a-timestamp",
            "payload": {},
        }

        with caplog.at_level(logging.DEBUG):
            signals = extractor.extract(event)

        assert signals == []
        assert "temporal_extractor" in caplog.text
        assert "malformed timestamp" in caplog.text

    def test_logs_malformed_calendar_start_time(self, extractor, caplog):
        """Malformed calendar start_time should log a debug message but still
        return the outer temporal_activity signal."""
        event = {
            "id": "test-temporal-bad-start",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "timestamp": "2026-01-15T10:00:00Z",
            "payload": {"start_time": "garbage-date"},
        }

        with caplog.at_level(logging.DEBUG):
            signals = extractor.extract(event)

        # The outer timestamp is valid, so we should get a temporal_activity signal
        assert len(signals) >= 1
        assert signals[0]["type"] == "temporal_activity"
        # But the inner start_time handler should have logged
        assert "temporal_extractor" in caplog.text
        assert "malformed start_time" in caplog.text


class TestSpatialExtractorObservability:
    """Verify SpatialExtractor logs debug messages for malformed time fields."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        """Create a SpatialExtractor with test database."""
        return SpatialExtractor(db, user_model_store)

    def test_logs_malformed_duration_calc(self, extractor, caplog):
        """Malformed start_time/end_time in calendar event should log a debug
        message but still produce a spatial signal."""
        event = {
            "id": "test-spatial-bad-time",
            "type": "calendar.event.created",
            "timestamp": "2026-01-15T10:00:00Z",
            "payload": {
                "location": "Conference Room A",
                "start_time": "not-a-time",
                "end_time": "also-not-a-time",
            },
        }

        with caplog.at_level(logging.DEBUG):
            signals = extractor.extract(event)

        # Should still produce a spatial signal (location is valid)
        assert len(signals) == 1
        assert signals[0]["location"] is not None
        # Duration calc failure should have been logged
        assert "spatial_extractor" in caplog.text
        assert "malformed time" in caplog.text


class TestCadenceExtractorObservability:
    """Verify CadenceExtractor logs debug messages for malformed timestamps."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        """Create a CadenceExtractor with test database."""
        return CadenceExtractor(db, user_model_store)

    def test_logs_malformed_timestamp(self, extractor, caplog):
        """Malformed event timestamp should log a debug message and skip
        the activity signal without raising."""
        event = {
            "id": "test-cadence-bad-ts",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "google",
            "timestamp": "not-a-timestamp",
            "payload": {
                "body": "test",
                "sender": "alice@example.com",
            },
        }

        with caplog.at_level(logging.DEBUG):
            signals = extractor.extract(event)

        # No cadence_activity signal should be produced (timestamp is bad)
        activity_signals = [s for s in signals if s["type"] == "cadence_activity"]
        assert len(activity_signals) == 0
        # But the handler should have logged
        assert "cadence_extractor" in caplog.text
        assert "malformed timestamp" in caplog.text


class TestTaskManagerObservability:
    """Verify TaskManager logs warnings and debug messages for silent handlers."""

    @pytest.fixture
    def task_manager(self, db):
        """Create a TaskManager with no AI engine."""
        return TaskManager(db, event_bus=None, ai_engine=None)

    @pytest.mark.asyncio
    async def test_warns_no_ai_engine_once(self, task_manager, caplog):
        """TaskManager should log a warning when AI engine is missing, but only once."""
        event = {
            "id": "test-no-ai-1",
            "type": "email.received",
            "payload": {"body": "Please send the report by Friday."},
        }

        with caplog.at_level(logging.WARNING):
            await task_manager.process_event(event)

        assert "task_manager" in caplog.text
        assert "AI engine not available" in caplog.text
        warning_count = caplog.text.count("AI engine not available")
        assert warning_count == 1

        # Call again — should NOT log a second warning
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            await task_manager.process_event(event)

        assert "AI engine not available" not in caplog.text

    @pytest.mark.asyncio
    async def test_ai_engine_warned_flag_is_instance_attr(self, db):
        """The _ai_engine_warned flag should be per-instance, not shared."""
        tm1 = TaskManager(db, ai_engine=None)
        tm2 = TaskManager(db, ai_engine=None)

        assert tm1._ai_engine_warned is False
        assert tm2._ai_engine_warned is False

        # Trigger warning on tm1
        await tm1.process_event({"id": "e1", "type": "email.received", "payload": {"body": "test"}})
        assert tm1._ai_engine_warned is True
        # tm2 should be unaffected
        assert tm2._ai_engine_warned is False
