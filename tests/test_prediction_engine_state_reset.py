"""Tests for PredictionEngine.reset_state() — ensures in-memory state is
properly cleared after DB corruption recovery so that event-based and
time-based predictions resume correctly."""

import logging
from datetime import datetime, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine
from storage.database import UserModelStore


@pytest.fixture()
def prediction_engine(db, event_bus):
    """A PredictionEngine wired to a temporary DatabaseManager."""
    ums = UserModelStore(db, event_bus)
    return PredictionEngine(db, ums, timezone="America/Los_Angeles")


class TestResetStateFieldValues:
    """Verify reset_state() sets all fields back to initial defaults."""

    def test_reset_clears_cursor_and_counters(self, prediction_engine):
        """After mutating state, reset_state() should restore operational fields
        to their initial (fresh-start) values, while preserving persistence
        failure diagnostic fields (_store_failure_count, _persistence_failure_detected,
        _last_store_errors) so that recovery logic can trigger after DB corruption."""
        pe = prediction_engine

        # Mutate all in-memory state to simulate a long-running engine
        pe._last_event_cursor = 99999
        pe._last_time_based_run = datetime(2025, 1, 1, tzinfo=timezone.utc)
        pe._first_follow_up_run = False
        pe._total_runs = 42
        pe._total_predictions_generated = 100
        pe._total_predictions_surfaced = 50
        pe._consecutive_zero_runs = 3
        pe._store_failure_count = 5
        pe._persistence_failure_detected = True
        pe._last_store_errors = [{"error": "test"}]
        pe._last_generation_stats = {"need": 2}
        pe._last_generation_timestamp = "2025-01-01T00:00:00"
        pe._last_run_diagnostics = {"total_runs": 42}
        pe._zero_surfacing_cycles = 7

        pe.reset_state()

        # Operational fields are cleared
        assert pe._last_event_cursor == 0
        assert pe._last_time_based_run is None
        assert pe._first_follow_up_run is True
        assert pe._total_runs == 0
        assert pe._total_predictions_generated == 0
        assert pe._total_predictions_surfaced == 0
        assert pe._consecutive_zero_runs == 0
        assert pe._last_generation_stats == {}
        assert pe._last_generation_timestamp is None
        assert pe._last_run_diagnostics == {}
        assert pe._zero_surfacing_cycles == 0
        assert pe._surfacing_diagnostics == pe._empty_surfacing_diagnostics()

        # Persistence failure fields are PRESERVED across reset so that
        # recovery logic can detect and correct data loss after DB corruption
        assert pe._store_failure_count == 5
        assert pe._persistence_failure_detected is True
        assert pe._last_store_errors == [{"error": "test"}]

    def test_reset_does_not_clear_injected_dependencies(self, prediction_engine):
        """reset_state() must NOT touch db, ums, or timezone config."""
        pe = prediction_engine
        original_db = pe.db
        original_ums = pe.ums
        original_tz = pe._tz_name

        pe.reset_state()

        assert pe.db is original_db
        assert pe.ums is original_ums
        assert pe._tz_name == original_tz


class TestHasNewEventsAfterReset:
    """Verify _has_new_events() behaves correctly after reset."""

    def test_has_new_events_true_after_reset_with_events(self, prediction_engine, db):
        """After reset (cursor=0), _has_new_events() should return True if
        events exist in the database."""
        pe = prediction_engine

        # Insert a test event
        with db.get_connection("events") as conn:
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
                ("test-1", "test.event", "test", datetime.now(timezone.utc).isoformat(), "{}"),
            )

        # Simulate stale cursor (higher than any real rowid)
        pe._last_event_cursor = 999999

        # _has_new_events should return False with stale cursor
        assert pe._has_new_events() is False

        # Reset and verify it now returns True
        pe.reset_state()
        assert pe._has_new_events() is True

    def test_has_new_events_false_after_reset_empty_db(self, prediction_engine):
        """After reset, _has_new_events() should return False if no events exist."""
        pe = prediction_engine
        pe.reset_state()
        assert pe._has_new_events() is False


class TestShouldRunTimeBasedAfterReset:
    """Verify _should_run_time_based_predictions() after reset."""

    def test_time_based_returns_true_after_reset(self, prediction_engine):
        """After reset (last_run=None), time-based predictions should fire
        immediately."""
        pe = prediction_engine

        # Simulate a recent run so it would normally NOT fire
        pe._last_time_based_run = datetime.now(timezone.utc)
        assert pe._should_run_time_based_predictions() is False

        # Reset clears last_run, so next call should return True
        pe.reset_state()
        assert pe._should_run_time_based_predictions() is True


class TestResetStateLogging:
    """Verify reset_state() logs pre-reset values for debugging."""

    def test_logs_pre_reset_values(self, prediction_engine, caplog):
        """reset_state() should log the old cursor, last_run, total_runs, and
        consecutive_zero_runs values before clearing them."""
        pe = prediction_engine
        pe._last_event_cursor = 12345
        pe._last_time_based_run = datetime(2025, 6, 15, tzinfo=timezone.utc)
        pe._total_runs = 99
        pe._consecutive_zero_runs = 5

        with caplog.at_level(logging.INFO, logger="services.prediction_engine.engine"):
            pe.reset_state()

        assert "cursor=12345" in caplog.text
        assert "total_runs=99" in caplog.text
        assert "consecutive_zero=5" in caplog.text


class TestResetIntegrationWithDBRebuild:
    """Integration test: simulate the DB rebuild + reset_state() flow."""

    def test_predictions_resume_after_simulated_rebuild(self, prediction_engine, db):
        """Simulate the _db_health_loop scenario: predictions exist, DB is
        rebuilt (state table wiped), reset_state() is called, and the engine
        can detect new events again."""
        pe = prediction_engine

        # 1. Insert some events
        with db.get_connection("events") as conn:
            for i in range(5):
                conn.execute(
                    "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
                    (f"evt-{i}", "test.event", "test", datetime.now(timezone.utc).isoformat(), "{}"),
                )

        # 2. Simulate engine having processed them (cursor at max)
        pe._last_event_cursor = 999999
        pe._total_runs = 50
        pe._first_follow_up_run = False

        # _has_new_events should return False (cursor too high)
        assert pe._has_new_events() is False

        # 3. Simulate DB rebuild: wipe prediction_engine_state table
        with db.get_connection("user_model") as conn:
            conn.execute("DELETE FROM prediction_engine_state")

        # 4. Call reset_state() as _db_health_loop would
        pe.reset_state()

        # 5. Verify engine is back to fresh-start state
        assert pe._last_event_cursor == 0
        assert pe._first_follow_up_run is True
        assert pe._total_runs == 0

        # 6. _has_new_events should now return True (events exist, cursor=0)
        assert pe._has_new_events() is True
