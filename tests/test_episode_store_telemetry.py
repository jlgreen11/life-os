"""
Tests for episode store telemetry correctness and WAL checkpoint resilience.

Background
----------
The data quality report showed 1,865 ``usermodel.episode.stored`` telemetry
events while the episodes table in user_model.db contained 0 rows.  This was
caused by the ``_emit_telemetry()`` call in ``store_episode()`` sitting OUTSIDE
the try/except block — telemetry fired even when the DB INSERT failed, producing
phantom telemetry that masked the underlying persistence failure.

The fix mirrors the pattern used by ``update_signal_profile()`` and
``update_semantic_fact()`` in the same file:

* Telemetry fires ONLY after the ``with`` block exits (transaction committed).
* Telemetry is skipped (and a warning is logged) when the INSERT raises.
* A WAL checkpoint runs every 50 episode writes (same throttle as signal profiles).
* A post-write row-count check logs CRITICAL if the episode is not found after INSERT.

These tests guard against a regression.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(episode_id: str = "ep-001", event_id: str = "evt-001") -> dict:
    """Return a minimal, valid episode dict suitable for store_episode()."""
    return {
        "id": episode_id,
        "timestamp": "2026-04-12T00:00:00Z",
        "event_id": event_id,
        "interaction_type": "message",
        "active_domain": "communication",
        "content_summary": "test episode",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(db):
    """A UserModelStore backed by a real temporary DatabaseManager (no event bus)."""
    return UserModelStore(db)


@pytest.fixture()
def store_with_bus(db, event_bus):
    """A UserModelStore backed by a real temporary DatabaseManager with mock EventBus."""
    return UserModelStore(db, event_bus=event_bus)


# ---------------------------------------------------------------------------
# Telemetry gating: telemetry must NOT fire on DB failure
# ---------------------------------------------------------------------------


class TestTelemetryNotEmittedOnDBFailure:
    """Telemetry must be suppressed when the episode INSERT raises an exception."""

    def test_telemetry_not_called_when_get_connection_raises(self, store):
        """If get_connection() raises, _emit_telemetry must not be called."""
        episode = _make_episode()

        with patch.object(store.db, "get_connection", side_effect=Exception("DB error")), \
             patch.object(store, "_emit_telemetry") as mock_telemetry:
            store.store_episode(episode)

        mock_telemetry.assert_not_called()

    def test_telemetry_not_called_when_execute_raises(self, store):
        """If conn.execute() raises, _emit_telemetry must not be called."""
        episode = _make_episode()

        # Build a mock context manager that raises inside __enter__
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("UNIQUE constraint failed")
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch.object(store.db, "get_connection", return_value=mock_ctx), \
             patch.object(store, "_emit_telemetry") as mock_telemetry:
            store.store_episode(episode)

        mock_telemetry.assert_not_called()

    def test_store_does_not_raise_on_db_failure(self, store):
        """store_episode must catch DB errors internally (fail-open contract)."""
        episode = _make_episode()

        with patch.object(store.db, "get_connection", side_effect=OSError("disk full")):
            # Must not propagate the exception to the caller
            store.store_episode(episode)

    def test_warning_logged_on_db_failure(self, store, caplog):
        """A WARNING must be logged when the INSERT raises."""
        episode = _make_episode()

        with patch.object(store.db, "get_connection", side_effect=RuntimeError("WAL locked")):
            with caplog.at_level(logging.WARNING, logger="storage.user_model_store"):
                store.store_episode(episode)

        assert any(
            "store_episode failed" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# Telemetry gating: telemetry MUST fire on successful write
# ---------------------------------------------------------------------------


class TestTelemetryEmittedOnSuccess:
    """Telemetry must fire exactly once after every successful episode write."""

    def test_telemetry_called_on_successful_write(self, store):
        """_emit_telemetry is called once after a successful store_episode()."""
        episode = _make_episode()

        with patch.object(store, "_emit_telemetry") as mock_telemetry:
            store.store_episode(episode)

        mock_telemetry.assert_called_once()

    def test_telemetry_event_type_correct(self, store):
        """The telemetry event type must be 'usermodel.episode.stored'."""
        episode = _make_episode()

        with patch.object(store, "_emit_telemetry") as mock_telemetry:
            store.store_episode(episode)

        event_type = mock_telemetry.call_args[0][0]
        assert event_type == "usermodel.episode.stored"

    def test_telemetry_payload_contains_episode_id(self, store):
        """Telemetry payload must include the episode's id."""
        episode = _make_episode(episode_id="ep-verify-01")

        with patch.object(store, "_emit_telemetry") as mock_telemetry:
            store.store_episode(episode)

        payload = mock_telemetry.call_args[0][1]
        assert payload["episode_id"] == "ep-verify-01"

    def test_telemetry_not_called_for_failed_write_but_called_for_next_success(self, store):
        """After a failed write (no telemetry), a successful write still emits telemetry."""
        episode_fail = _make_episode(episode_id="ep-fail")
        episode_ok = _make_episode(episode_id="ep-ok")

        calls = []

        def _real_emit(event_type, payload):
            calls.append((event_type, payload))

        # Patch get_connection to fail on the first call only
        original_get_connection = store.db.get_connection
        call_count = {"n": 0}

        def fail_first_call(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated first-call failure")
            return original_get_connection(*args, **kwargs)

        with patch.object(store.db, "get_connection", side_effect=fail_first_call), \
             patch.object(store, "_emit_telemetry", side_effect=_real_emit):
            store.store_episode(episode_fail)   # fails → no telemetry
            store.store_episode(episode_ok)     # succeeds → telemetry fires

        assert len(calls) == 1
        assert calls[0][1]["episode_id"] == "ep-ok"

    def test_episode_actually_stored_in_db(self, store):
        """After store_episode, the row must be retrievable from the database."""
        episode = _make_episode(episode_id="ep-db-check")
        store.store_episode(episode)

        with store.db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE id = ?", ("ep-db-check",)
            ).fetchone()[0]

        assert count == 1


# ---------------------------------------------------------------------------
# WAL checkpoint tests
# ---------------------------------------------------------------------------


class TestEpisodeWALCheckpoint:
    """checkpoint_wal is called throttled to every 50 episode writes."""

    def test_no_checkpoint_before_50th_write(self, store):
        """checkpoint_wal must NOT be called for writes 1–49."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            for i in range(49):
                store.store_episode(_make_episode(f"ep-{i:04d}", f"evt-{i:04d}"))
            mock_ckpt.assert_not_called()

    def test_checkpoint_called_on_50th_write(self, store):
        """checkpoint_wal MUST be called exactly once on the 50th write."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            for i in range(50):
                store.store_episode(_make_episode(f"ep-{i:04d}", f"evt-{i:04d}"))
            mock_ckpt.assert_called_once_with("user_model")

    def test_checkpoint_called_on_100th_write(self, store):
        """checkpoint_wal is called again at 100 writes (every 50)."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            for i in range(100):
                store.store_episode(_make_episode(f"ep-{i:04d}", f"evt-{i:04d}"))
            assert mock_ckpt.call_count == 2
            for call in mock_ckpt.call_args_list:
                assert call.args[0] == "user_model"

    def test_checkpoint_called_at_correct_multiples(self, store):
        """Checkpoint fires exactly at multiples of 50 up to 200 writes."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            for i in range(200):
                store.store_episode(_make_episode(f"ep-{i:04d}", f"evt-{i:04d}"))
            assert mock_ckpt.call_count == 4  # 50, 100, 150, 200

    def test_checkpoint_counter_is_cumulative(self, store):
        """Counter accumulates across calls — not per-batch."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            # 30 writes → no checkpoint yet
            for i in range(30):
                store.store_episode(_make_episode(f"ep-a{i:03d}", f"evt-a{i:03d}"))
            assert mock_ckpt.call_count == 0

            # 20 more → total 50, checkpoint fires
            for i in range(20):
                store.store_episode(_make_episode(f"ep-b{i:03d}", f"evt-b{i:03d}"))
            assert mock_ckpt.call_count == 1

    def test_episode_write_count_starts_at_zero(self, store):
        """_episode_write_count is 0 on a freshly created store."""
        assert store._episode_write_count == 0

    def test_episode_write_count_increments_on_success(self, store):
        """_episode_write_count increments with each successful write."""
        store.store_episode(_make_episode("ep-cnt-01", "evt-cnt-01"))
        assert store._episode_write_count == 1
        store.store_episode(_make_episode("ep-cnt-02", "evt-cnt-02"))
        assert store._episode_write_count == 2

    def test_checkpoint_failure_does_not_crash_write(self, store):
        """A checkpoint exception must NOT propagate out of store_episode."""
        with patch.object(
            store.db, "checkpoint_wal", side_effect=RuntimeError("disk full")
        ):
            # Drive to the 50th write, which triggers checkpoint — must not raise
            for i in range(50):
                store.store_episode(_make_episode(f"ep-{i:04d}", f"evt-{i:04d}"))

        # Data must still be in the database
        with store.db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert count == 50

    def test_checkpoint_failure_logs_warning(self, store, caplog):
        """A checkpoint failure must log a WARNING without re-raising."""
        with patch.object(
            store.db, "checkpoint_wal", side_effect=RuntimeError("disk full")
        ):
            with caplog.at_level(logging.WARNING, logger="storage.user_model_store"):
                for i in range(50):
                    store.store_episode(_make_episode(f"ep-{i:04d}", f"evt-{i:04d}"))

        assert any(
            "WAL checkpoint" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# Post-write verification tests
# ---------------------------------------------------------------------------


class TestPostWriteVerification:
    """The row-count read-back after INSERT must log CRITICAL when row is missing."""

    def test_no_critical_log_on_successful_write(self, store, caplog):
        """No CRITICAL log when the episode row is found after INSERT."""
        episode = _make_episode(episode_id="ep-ok-verify")

        with caplog.at_level(logging.CRITICAL, logger="storage.user_model_store"):
            store.store_episode(episode)

        # Should be no CRITICAL messages about verification failure
        critical_msgs = [
            r for r in caplog.records
            if r.levelno == logging.CRITICAL and "post-write verification" in r.message
        ]
        assert critical_msgs == []

    def test_critical_logged_when_row_missing_after_insert(self, store, caplog):
        """CRITICAL must be logged when the post-write count check returns 0."""
        episode = _make_episode(episode_id="ep-ghost")

        # We need to simulate the INSERT succeeding but the verification SELECT
        # finding 0 rows.  Achieve this by making the second get_connection call
        # (the verification SELECT) return a mock connection that reports count 0.
        original = store.db.get_connection
        call_n = {"n": 0}

        def patched_get_connection(*args, **kwargs):
            call_n["n"] += 1
            if call_n["n"] == 2:
                # Verification call — return mock conn reporting 0 rows
                mock_conn = MagicMock()
                mock_conn.execute.return_value.fetchone.return_value = [0]
                ctx = MagicMock()
                ctx.__enter__ = MagicMock(return_value=mock_conn)
                ctx.__exit__ = MagicMock(return_value=False)
                return ctx
            return original(*args, **kwargs)

        with patch.object(store.db, "get_connection", side_effect=patched_get_connection):
            with caplog.at_level(logging.CRITICAL, logger="storage.user_model_store"):
                store.store_episode(episode)

        critical_msgs = [
            r for r in caplog.records
            if r.levelno == logging.CRITICAL and "post-write verification FAILED" in r.message
        ]
        assert len(critical_msgs) == 1
        assert "ep-ghost" in critical_msgs[0].message
