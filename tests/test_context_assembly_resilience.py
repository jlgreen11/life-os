"""
Tests for ContextAssembler fail-open resilience.

Verifies that each context method gracefully degrades (returns a fallback
string instead of propagating an exception) when its backing database is
corrupted or unavailable. This is critical because a single malformed
SQLite DB (e.g. user_model.db) should NOT crash the entire morning briefing.
"""

import sqlite3
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from services.ai_engine.context import ContextAssembler
from storage.user_model_store import UserModelStore


@pytest.fixture()
def context_assembler(db, event_bus):
    """A ContextAssembler wired to a temporary DatabaseManager."""
    ums = UserModelStore(db, event_bus=event_bus)
    return ContextAssembler(db, ums)


def _make_broken_connection(db_name):
    """Return a patched get_connection that raises OperationalError for a specific DB.

    All other database names pass through to the real implementation.
    """

    def _broken_get_connection(original_get_connection, target_db_name):
        @contextmanager
        def wrapper(name):
            if name == target_db_name:
                raise sqlite3.OperationalError("database disk image is malformed")
            with original_get_connection(name) as conn:
                yield conn

        return wrapper

    return _broken_get_connection


class TestPredictionsContextResilience:
    """_get_predictions_context returns '' when user_model.db is corrupted."""

    def test_returns_empty_on_db_error(self, context_assembler):
        """Corrupted user_model.db should not propagate exceptions."""
        original = context_assembler.db.get_connection
        broken = _make_broken_connection("user_model")(original, "user_model")
        with patch.object(context_assembler.db, "get_connection", broken):
            result = context_assembler._get_predictions_context()
        assert result == ""


class TestInsightsContextResilience:
    """_get_insights_context returns '' when user_model.db is corrupted."""

    def test_returns_empty_on_db_error(self, context_assembler):
        """Corrupted user_model.db should not propagate exceptions."""
        original = context_assembler.db.get_connection
        broken = _make_broken_connection("user_model")(original, "user_model")
        with patch.object(context_assembler.db, "get_connection", broken):
            result = context_assembler._get_insights_context()
        assert result == ""


class TestPreferenceContextResilience:
    """_get_preference_context returns fallback string when preferences.db is corrupted."""

    def test_returns_fallback_on_db_error(self, context_assembler):
        """Corrupted preferences.db should return the 'not yet configured' fallback."""
        original = context_assembler.db.get_connection
        broken = _make_broken_connection("preferences")(original, "preferences")
        with patch.object(context_assembler.db, "get_connection", broken):
            result = context_assembler._get_preference_context()
        assert result == "User preferences: not yet configured"


class TestTaskContextResilience:
    """_get_task_context returns '' when state.db is corrupted."""

    def test_returns_empty_on_db_error(self, context_assembler):
        """Corrupted state.db should not propagate exceptions."""
        original = context_assembler.db.get_connection
        broken = _make_broken_connection("state")(original, "state")
        with patch.object(context_assembler.db, "get_connection", broken):
            result = context_assembler._get_task_context()
        assert result == ""


class TestRecentCompletionsContextResilience:
    """_get_recent_completions_context returns '' when state.db is corrupted."""

    def test_returns_empty_on_db_error(self, context_assembler):
        """Corrupted state.db should not propagate exceptions."""
        original = context_assembler.db.get_connection
        broken = _make_broken_connection("state")(original, "state")
        with patch.object(context_assembler.db, "get_connection", broken):
            result = context_assembler._get_recent_completions_context()
        assert result == ""


class TestBriefingContextIntegration:
    """assemble_briefing_context still produces a usable result when user_model.db is corrupted."""

    def test_briefing_survives_user_model_corruption(self, context_assembler):
        """The briefing should still include the timestamp section even when
        user_model.db raises OperationalError on every query.

        This simulates the real-world scenario where user_model.db is malformed
        (as reported by data quality analysis) — the briefing must degrade
        gracefully rather than crash outright.
        """
        original = context_assembler.db.get_connection
        broken = _make_broken_connection("user_model")(original, "user_model")
        with patch.object(context_assembler.db, "get_connection", broken):
            result = context_assembler.assemble_briefing_context()

        # The briefing must still be a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0
        # The timestamp section should always be present since it doesn't
        # depend on any database
        assert "Current time:" in result
