"""
Tests for RoutineDetector fail-open resilience under database corruption.

Validates that individual detection strategies are isolated — a failure in one
(e.g. corrupted user_model.db causing sqlite3.DatabaseError) does not prevent
the others from running.  This mirrors the isolation already present in the
InsightEngine correlators.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.routine_detector.detector import RoutineDetector


def _seed_morning_routine(user_model_store, days=10):
    """Seed episodic memory with a consistent morning routine pattern.

    Creates check_email + review_calendar episodes at 8am on each day,
    giving both temporal and event-triggered detection strategies data to
    work with.
    """
    base_date = datetime.now(timezone.utc) - timedelta(days=days)
    for day_offset in range(days):
        day_start = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
        user_model_store.store_episode({
            "id": str(uuid.uuid4()),
            "timestamp": day_start.isoformat(),
            "event_id": str(uuid.uuid4()),
            "interaction_type": "check_email",
            "content_summary": "Check Email",
        })
        user_model_store.store_episode({
            "id": str(uuid.uuid4()),
            "timestamp": (day_start + timedelta(minutes=15)).isoformat(),
            "event_id": str(uuid.uuid4()),
            "interaction_type": "review_calendar",
            "content_summary": "Review Calendar",
        })


def _seed_location_routine(user_model_store, days=10):
    """Seed episodic memory with a consistent location-based routine.

    Creates location + smart_home episodes at Home on each evening.
    """
    base_date = datetime.now(timezone.utc) - timedelta(days=days)
    for day_offset in range(days):
        arrive_time = base_date + timedelta(days=day_offset, hours=17)
        user_model_store.store_episode({
            "id": str(uuid.uuid4()),
            "timestamp": arrive_time.isoformat(),
            "event_id": str(uuid.uuid4()),
            "interaction_type": "location",
            "location": "Home",
        })
        user_model_store.store_episode({
            "id": str(uuid.uuid4()),
            "timestamp": (arrive_time + timedelta(minutes=5)).isoformat(),
            "event_id": str(uuid.uuid4()),
            "interaction_type": "smart_home",
            "location": "Home",
        })


class TestDetectRoutinesSurvivesStrategyCrash:
    """Verify detect_routines() returns partial results when individual
    strategies raise exceptions (e.g. from a corrupted database)."""

    def test_survives_temporal_strategy_crash(self, db, user_model_store):
        """When _detect_temporal_routines raises, the other strategies still run."""
        detector = RoutineDetector(db, user_model_store)

        # Seed data for location and event-triggered strategies
        _seed_location_routine(user_model_store)
        _seed_morning_routine(user_model_store)

        with patch.object(
            detector,
            "_detect_temporal_routines",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            routines = detector.detect_routines(lookback_days=30)

        # Should NOT raise; should return results from the surviving strategies
        assert isinstance(routines, list)
        # Location routines should still be present
        location_routines = [r for r in routines if "Home" in r.get("name", "")]
        assert len(location_routines) >= 1

    def test_survives_location_strategy_crash(self, db, user_model_store):
        """When _detect_location_routines raises, temporal and event strategies still run."""
        detector = RoutineDetector(db, user_model_store)

        _seed_morning_routine(user_model_store)

        with patch.object(
            detector,
            "_detect_location_routines",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            routines = detector.detect_routines(lookback_days=30)

        assert isinstance(routines, list)
        # Temporal routines should still be present
        morning_routines = [r for r in routines if r.get("trigger") == "morning"]
        assert len(morning_routines) >= 1

    def test_survives_event_strategy_crash(self, db, user_model_store):
        """When _detect_event_triggered_routines raises, temporal and location still run."""
        detector = RoutineDetector(db, user_model_store)

        _seed_morning_routine(user_model_store)
        _seed_location_routine(user_model_store)

        with patch.object(
            detector,
            "_detect_event_triggered_routines",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            routines = detector.detect_routines(lookback_days=30)

        assert isinstance(routines, list)
        # Both temporal and location should survive
        morning_routines = [r for r in routines if r.get("trigger") == "morning"]
        location_routines = [r for r in routines if "Home" in r.get("name", "")]
        assert len(morning_routines) >= 1
        assert len(location_routines) >= 1

    def test_survives_all_strategies_crash(self, db, user_model_store):
        """When ALL strategies raise, detect_routines returns empty list (not exception)."""
        detector = RoutineDetector(db, user_model_store)

        _seed_morning_routine(user_model_store)

        with (
            patch.object(
                detector,
                "_detect_temporal_routines",
                side_effect=sqlite3.DatabaseError("database disk image is malformed"),
            ),
            patch.object(
                detector,
                "_detect_location_routines",
                side_effect=sqlite3.DatabaseError("database disk image is malformed"),
            ),
            patch.object(
                detector,
                "_detect_event_triggered_routines",
                side_effect=sqlite3.DatabaseError("database disk image is malformed"),
            ),
        ):
            routines = detector.detect_routines(lookback_days=30)

        assert routines == []


class TestHelperMethodResilience:
    """Verify that helper methods return safe defaults when the database
    is corrupted, allowing the calling strategy to continue."""

    def test_count_active_days_returns_default_on_corruption(self, db, user_model_store):
        """_count_active_days should return 1 (safe default) when DB is corrupted.

        The try/except wrapper inside each strategy catches the exception and
        falls back to active_days=1, preventing division-by-zero downstream.
        """
        detector = RoutineDetector(db, user_model_store)

        # Seed some data so the detector has something to work with
        _seed_morning_routine(user_model_store, days=3)

        # Patch _count_active_days to raise DatabaseError, then verify
        # _detect_temporal_routines still returns results (using fallback=1)
        with patch.object(
            detector,
            "_count_active_days",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            # The strategy should catch the error and use active_days=1
            routines = detector._detect_temporal_routines(lookback_days=30)

        # Should not raise; may return routines (consistency will be high
        # with active_days=1 since every action appears on at least 1 day)
        assert isinstance(routines, list)

    def test_compute_step_duration_map_returns_empty_on_corruption(self, db, user_model_store):
        """_compute_step_duration_map should return {} (empty) when DB is corrupted.

        The try/except wrapper inside each strategy catches the exception and
        falls back to step_duration_map={}, causing all steps to use the 5.0
        minute default duration.
        """
        detector = RoutineDetector(db, user_model_store)

        _seed_morning_routine(user_model_store, days=3)

        # Patch _compute_step_duration_map to raise, verify strategy still works
        with patch.object(
            detector,
            "_compute_step_duration_map",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            routines = detector._detect_temporal_routines(lookback_days=30)

        assert isinstance(routines, list)
        # If routines were detected, step durations should be the fallback values
        for routine in routines:
            for step in routine.get("steps", []):
                # Duration should be either 5.0 (fallback) or 15.0 (last-step default)
                assert step["typical_duration_minutes"] in (5.0, 15.0), (
                    f"Expected fallback duration (5.0 or 15.0), got {step['typical_duration_minutes']}"
                )

    def test_location_strategy_survives_helper_corruption(self, db, user_model_store):
        """Location strategy should survive when both helpers fail."""
        detector = RoutineDetector(db, user_model_store)

        _seed_location_routine(user_model_store)

        with (
            patch.object(
                detector,
                "_count_active_days",
                side_effect=sqlite3.DatabaseError("database disk image is malformed"),
            ),
            patch.object(
                detector,
                "_compute_step_duration_map",
                side_effect=sqlite3.DatabaseError("database disk image is malformed"),
            ),
        ):
            routines = detector._detect_location_routines(lookback_days=30)

        assert isinstance(routines, list)

    def test_event_triggered_strategy_survives_helper_corruption(self, db, user_model_store):
        """Event-triggered strategy should survive when _compute_step_duration_map fails."""
        detector = RoutineDetector(db, user_model_store)

        _seed_morning_routine(user_model_store)

        with patch.object(
            detector,
            "_compute_step_duration_map",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            routines = detector._detect_event_triggered_routines(lookback_days=30)

        assert isinstance(routines, list)
