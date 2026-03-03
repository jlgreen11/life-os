"""
Tests for routine detector resilience to user_model.db corruption.

Verifies that all RoutineDetector methods that query user_model.db
gracefully degrade when the database is corrupted (e.g., 'database
disk image is malformed') by returning safe default values instead
of raising exceptions.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from services.routine_detector.detector import RoutineDetector


@pytest.fixture()
def corrupted_db(db):
    """A DatabaseManager whose user_model connections raise sqlite3.DatabaseError.

    Wraps the real DatabaseManager so that get_connection("user_model") raises
    sqlite3.DatabaseError (simulating a corrupted database file), while other
    databases remain functional.
    """
    original_get_connection = db.get_connection

    @contextmanager
    def corrupted_get_connection(db_name: str):
        if db_name == "user_model":
            raise sqlite3.DatabaseError("database disk image is malformed")
        with original_get_connection(db_name) as conn:
            yield conn

    db.get_connection = corrupted_get_connection
    return db


@pytest.fixture()
def detector(corrupted_db, user_model_store):
    """A RoutineDetector wired to a corrupted user_model database."""
    return RoutineDetector(corrupted_db, user_model_store, timezone="UTC")


@pytest.fixture()
def healthy_detector(db, user_model_store):
    """A RoutineDetector wired to a healthy database (for baseline tests)."""
    return RoutineDetector(db, user_model_store, timezone="UTC")


class TestCountActiveDaysCorruption:
    """Tests for _count_active_days resilience to DB corruption."""

    def test_returns_1_on_corruption(self, detector):
        """_count_active_days returns 1 (safe fallback) when DB is corrupted."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        result = detector._count_active_days(cutoff)
        assert result == 1

    def test_does_not_raise_on_corruption(self, detector):
        """_count_active_days does not propagate DatabaseError."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        # Should not raise — the exception is caught internally
        detector._count_active_days(cutoff)

    def test_returns_correct_value_when_healthy(self, healthy_detector):
        """_count_active_days returns real data when DB is healthy (baseline)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        result = healthy_detector._count_active_days(cutoff)
        # No episodes inserted, so should return 1 (max(1, 0))
        assert result == 1


class TestComputeStepDurationMapCorruption:
    """Tests for _compute_step_duration_map resilience to DB corruption."""

    def test_returns_empty_dict_on_corruption(self, detector):
        """_compute_step_duration_map returns {} when DB is corrupted."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        result = detector._compute_step_duration_map(cutoff)
        assert result == {}

    def test_does_not_raise_on_corruption(self, detector):
        """_compute_step_duration_map does not propagate DatabaseError."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        detector._compute_step_duration_map(cutoff)


class TestDetectTemporalRoutinesCorruption:
    """Tests for _detect_temporal_routines resilience to DB corruption."""

    def test_returns_empty_list_on_corruption(self, detector):
        """_detect_temporal_routines returns [] when DB is corrupted."""
        result = detector._detect_temporal_routines(lookback_days=30)
        assert result == []

    def test_does_not_raise_on_corruption(self, detector):
        """_detect_temporal_routines does not propagate DatabaseError."""
        detector._detect_temporal_routines(lookback_days=30)


class TestDetectLocationRoutinesCorruption:
    """Tests for _detect_location_routines resilience to DB corruption."""

    def test_returns_empty_list_on_corruption(self, detector):
        """_detect_location_routines returns [] when DB is corrupted."""
        result = detector._detect_location_routines(lookback_days=30)
        assert result == []

    def test_does_not_raise_on_corruption(self, detector):
        """_detect_location_routines does not propagate DatabaseError."""
        detector._detect_location_routines(lookback_days=30)


class TestDetectEventTriggeredRoutinesCorruption:
    """Tests for _detect_event_triggered_routines resilience to DB corruption."""

    def test_returns_empty_list_on_corruption(self, detector):
        """_detect_event_triggered_routines returns [] when DB is corrupted."""
        result = detector._detect_event_triggered_routines(lookback_days=30)
        assert result == []

    def test_does_not_raise_on_corruption(self, detector):
        """_detect_event_triggered_routines does not propagate DatabaseError."""
        detector._detect_event_triggered_routines(lookback_days=30)


class TestDetectRoutinesPublicMethodCorruption:
    """Tests for the public detect_routines() method with a corrupted DB."""

    def test_returns_empty_list_on_corruption(self, detector):
        """detect_routines() returns [] when all DB queries fail due to corruption."""
        result = detector.detect_routines(lookback_days=30)
        assert result == []

    def test_does_not_raise_on_corruption(self, detector):
        """detect_routines() completes without raising when DB is corrupted."""
        detector.detect_routines(lookback_days=30)

    def test_logs_warnings_on_corruption(self, detector, caplog):
        """detect_routines() logs warnings when DB queries fail."""
        import logging

        with caplog.at_level(logging.WARNING):
            detector.detect_routines(lookback_days=30)

        # At least the three main detection methods should log warnings
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_messages) > 0, "Expected at least one warning about DB corruption"

        # Check that corruption-related messages were logged
        corruption_warnings = [m for m in warning_messages if "user_model.db" in m or "malformed" in m]
        assert len(corruption_warnings) > 0, "Expected warnings mentioning user_model.db or malformed"


class TestOperationalErrorCorruption:
    """Tests that sqlite3.OperationalError (a subclass of DatabaseError) is also caught."""

    @pytest.fixture()
    def operational_error_db(self, db):
        """A DatabaseManager whose user_model connections raise OperationalError."""
        original_get_connection = db.get_connection

        @contextmanager
        def error_get_connection(db_name: str):
            if db_name == "user_model":
                raise sqlite3.OperationalError("database disk image is malformed")
            with original_get_connection(db_name) as conn:
                yield conn

        db.get_connection = error_get_connection
        return db

    def test_count_active_days_catches_operational_error(self, operational_error_db, user_model_store):
        """OperationalError (subclass of DatabaseError) is caught by _count_active_days."""
        detector = RoutineDetector(operational_error_db, user_model_store, timezone="UTC")
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        result = detector._count_active_days(cutoff)
        assert result == 1

    def test_detect_routines_catches_operational_error(self, operational_error_db, user_model_store):
        """OperationalError is caught by detect_routines(), returning [] without raising."""
        detector = RoutineDetector(operational_error_db, user_model_store, timezone="UTC")
        result = detector.detect_routines(lookback_days=30)
        assert result == []
