"""
Tests for semantic fact telemetry ordering.

Verifies that telemetry events for semantic fact updates only fire AFTER the
database transaction commits successfully, preventing phantom telemetry when
the user_model.db transaction rolls back.
"""

from unittest.mock import patch, MagicMock

import pytest

from storage.user_model_store import UserModelStore


@pytest.fixture
def user_model_store_with_spy(db):
    """UserModelStore with a spy on _emit_telemetry for verifying call ordering."""
    store = UserModelStore(db)
    store._emit_telemetry = MagicMock()
    return store


class TestSemanticFactTelemetryOrdering:
    """Ensure telemetry fires only after a successful commit."""

    def test_telemetry_emitted_on_successful_insert(self, user_model_store_with_spy):
        """Happy path: new fact is stored AND telemetry event is emitted."""
        store = user_model_store_with_spy

        store.update_semantic_fact(
            key="prefers_morning_coffee",
            category="preference",
            value={"detail": "drinks coffee every morning"},
            confidence=0.6,
            episode_id="ep-001",
        )

        # Verify the fact row exists in user_model.db
        fact = store.get_semantic_fact("prefers_morning_coffee")
        assert fact is not None
        assert fact["category"] == "preference"
        assert fact["confidence"] == 0.6

        # Verify telemetry was emitted
        store._emit_telemetry.assert_called_once()
        call_args = store._emit_telemetry.call_args
        assert call_args[0][0] == "usermodel.fact.learned"
        payload = call_args[0][1]
        assert payload["key"] == "prefers_morning_coffee"
        assert payload["is_new"] is True
        assert payload["confidence"] == 0.6

    def test_telemetry_emitted_on_successful_update(self, user_model_store_with_spy):
        """Existing fact update emits telemetry with incremented confidence."""
        store = user_model_store_with_spy

        # Insert initial fact
        store.update_semantic_fact(
            key="likes_jazz",
            category="preference",
            value={"detail": "enjoys jazz music"},
            confidence=0.5,
        )
        store._emit_telemetry.reset_mock()

        # Update the same fact (re-confirmation)
        store.update_semantic_fact(
            key="likes_jazz",
            category="preference",
            value={"detail": "enjoys jazz music"},
            confidence=0.5,
            episode_id="ep-002",
        )

        # Verify confidence was incremented in the DB
        fact = store.get_semantic_fact("likes_jazz")
        assert fact["confidence"] == pytest.approx(0.55)

        # Verify telemetry reflects the incremented confidence
        store._emit_telemetry.assert_called_once()
        payload = store._emit_telemetry.call_args[0][1]
        assert payload["confidence"] == pytest.approx(0.55)
        assert payload["is_new"] is False

    def test_no_telemetry_on_commit_failure(self, user_model_store_with_spy):
        """When the DB transaction fails, no telemetry event should be emitted.

        This is the phantom telemetry scenario: previously, telemetry fired
        inside the `with` block before commit, so it persisted in events.db
        even when user_model.db rolled back.
        """
        store = user_model_store_with_spy
        original_get_connection = store.db.get_connection

        class FailingConnection:
            """Context manager that raises on exit (simulating commit failure)."""

            def __init__(self, real_cm):
                self._real_cm = real_cm
                self._conn = None

            def __enter__(self):
                self._conn = self._real_cm.__enter__()
                return self._conn

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Force a rollback by raising before commit
                raise RuntimeError("Simulated commit failure")

        def patched_get_connection(db_name):
            if db_name == "user_model":
                return FailingConnection(original_get_connection(db_name))
            return original_get_connection(db_name)

        with patch.object(store.db, "get_connection", side_effect=patched_get_connection):
            # This should NOT raise (fail-open pattern) and should NOT emit telemetry
            store.update_semantic_fact(
                key="phantom_fact",
                category="preference",
                value={"detail": "should not persist"},
                confidence=0.7,
            )

        # Verify NO telemetry was emitted
        store._emit_telemetry.assert_not_called()

        # Verify the fact was NOT stored
        fact = store.get_semantic_fact("phantom_fact")
        assert fact is None
