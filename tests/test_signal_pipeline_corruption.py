"""
Tests for SignalExtractorPipeline.get_user_summary() resilience to corrupted user_model.db.

Verifies that get_user_summary() gracefully degrades when the underlying
UserModelStore raises database errors (e.g., 'database disk image is malformed')
by returning a safe empty summary with a ``degraded`` flag instead of
propagating the exception to callers like briefing generation or web API routes.
"""

import sqlite3
from unittest.mock import patch

import pytest

from services.signal_extractor.pipeline import SignalExtractorPipeline


@pytest.fixture()
def pipeline(db, user_model_store):
    """A SignalExtractorPipeline wired to healthy temporary databases."""
    return SignalExtractorPipeline(db, user_model_store)


class TestGetUserSummaryCorruptedDb:
    """Tests for get_user_summary() resilience to DB corruption."""

    def test_returns_empty_on_corrupted_db(self, pipeline):
        """get_user_summary() returns a degraded dict when get_signal_profile raises DatabaseError."""
        with patch.object(
            pipeline.ums,
            "get_signal_profile",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            result = pipeline.get_user_summary()

        assert result["profiles"] == {}
        assert result["semantic_facts_count"] == 0
        assert result["high_confidence_facts"] == []
        assert result["degraded"] is True

    def test_returns_empty_on_operational_error(self, pipeline):
        """get_user_summary() catches OperationalError (subclass of DatabaseError)."""
        with patch.object(
            pipeline.ums,
            "get_signal_profile",
            side_effect=sqlite3.OperationalError("database disk image is malformed"),
        ):
            result = pipeline.get_user_summary()

        assert result["profiles"] == {}
        assert result["semantic_facts_count"] == 0
        assert result["high_confidence_facts"] == []
        assert result["degraded"] is True

    def test_partial_failure_returns_degraded(self, pipeline):
        """When only get_semantic_facts raises, the entire method returns a degraded response.

        Since both calls are in a single try block, a failure in get_semantic_facts
        means the whole summary is degraded (not a partial result).
        """
        with patch.object(
            pipeline.ums,
            "get_semantic_facts",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            result = pipeline.get_user_summary()

        assert result["profiles"] == {}
        assert result["semantic_facts_count"] == 0
        assert result["high_confidence_facts"] == []
        assert result["degraded"] is True

    def test_logs_warning_on_corruption(self, pipeline, caplog):
        """get_user_summary() logs a warning when the DB is unavailable."""
        import logging

        with caplog.at_level(logging.WARNING):
            with patch.object(
                pipeline.ums,
                "get_signal_profile",
                side_effect=sqlite3.DatabaseError("database disk image is malformed"),
            ):
                pipeline.get_user_summary()

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        corruption_warnings = [m for m in warning_messages if "user_model.db unavailable" in m]
        assert len(corruption_warnings) > 0, "Expected a warning about user_model.db being unavailable"


class TestGetUserSummaryNormalOperation:
    """Sanity tests verifying get_user_summary() works correctly with a healthy DB."""

    def test_normal_operation_returns_valid_summary(self, pipeline):
        """get_user_summary() returns a dict with profiles and facts on a healthy DB."""
        result = pipeline.get_user_summary()

        # On a fresh DB there are no profiles or facts, but the structure is correct
        assert "profiles" in result
        assert "semantic_facts_count" in result
        assert "high_confidence_facts" in result
        assert isinstance(result["profiles"], dict)
        assert isinstance(result["semantic_facts_count"], int)
        assert isinstance(result["high_confidence_facts"], list)

    def test_normal_operation_has_no_degraded_flag(self, pipeline):
        """A successful get_user_summary() does not include the degraded flag."""
        result = pipeline.get_user_summary()
        assert "degraded" not in result
