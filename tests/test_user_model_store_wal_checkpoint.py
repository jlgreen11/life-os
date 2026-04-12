"""
Tests for WAL checkpoint resilience in UserModelStore.

Verifies that:
1. checkpoint_wal is called after every 50th signal profile write.
2. checkpoint_wal is called after every communication template store.
3. checkpoint_wal is called after every communication template update.
4. A checkpoint failure (exception) does NOT crash the write operation.

Background: March 5-6 2026 corruption wiped signal profiles and templates
because data accumulated in the WAL file without being flushed to the main
database.  These tests guard against a regression.
"""

from unittest.mock import MagicMock, patch

import pytest

from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(db):
    """A UserModelStore backed by the shared real DatabaseManager."""
    return UserModelStore(db)


def _make_template(template_id: str = "tpl-001") -> dict:
    """Return a minimal valid communication_template dict."""
    return {
        "id": template_id,
        "context": "professional_email",
        "contact_id": "contact-001",
        "channel": "email",
        "formality": 0.8,
        "samples_analyzed": 5,
    }


# ---------------------------------------------------------------------------
# Signal profile WAL checkpoint tests
# ---------------------------------------------------------------------------


class TestSignalProfileWALCheckpoint:
    """checkpoint_wal is called throttled to every 50 signal-profile writes."""

    def test_no_checkpoint_before_50th_write(self, store):
        """checkpoint_wal must NOT be called for writes 1–49."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            for i in range(49):
                store.update_signal_profile("linguistic", {"vocab": i})
            mock_ckpt.assert_not_called()

    def test_checkpoint_called_on_50th_write(self, store):
        """checkpoint_wal MUST be called exactly once on the 50th write."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            for i in range(50):
                store.update_signal_profile("linguistic", {"vocab": i})
            mock_ckpt.assert_called_once_with("user_model")

    def test_checkpoint_called_on_100th_write(self, store):
        """checkpoint_wal is called again at 100 writes (every 50)."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            for i in range(100):
                store.update_signal_profile("cadence", {"interval": i})
            assert mock_ckpt.call_count == 2
            for call in mock_ckpt.call_args_list:
                assert call.args[0] == "user_model"

    def test_checkpoint_called_at_correct_multiples(self, store):
        """Checkpoint is called exactly at multiples of 50 up to 200 writes."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            for i in range(200):
                store.update_signal_profile("mood_proxy", {"energy": i})
            assert mock_ckpt.call_count == 4  # 50, 100, 150, 200

    def test_checkpoint_counter_is_cumulative(self, store):
        """Counter does not reset between calls — tracks lifetime writes."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            # 30 writes → no checkpoint yet
            for i in range(30):
                store.update_signal_profile("linguistic", {"vocab": i})
            assert mock_ckpt.call_count == 0

            # 20 more → total 50, checkpoint fires
            for i in range(20):
                store.update_signal_profile("linguistic", {"vocab": i + 30})
            assert mock_ckpt.call_count == 1

    def test_checkpoint_failure_does_not_crash_write(self, store):
        """A checkpoint exception must NOT propagate out of update_signal_profile."""
        with patch.object(
            store.db, "checkpoint_wal", side_effect=RuntimeError("disk full")
        ):
            # Drive to the 50th write, which triggers checkpoint
            for i in range(50):
                # Should never raise, even when checkpoint raises
                store.update_signal_profile("linguistic", {"vocab": i})

        # The data should still be in the database
        profile = store.get_signal_profile("linguistic")
        assert profile is not None

    def test_checkpoint_failure_logs_warning(self, store, caplog):
        """A checkpoint failure must log a WARNING without re-raising."""
        import logging

        with patch.object(
            store.db, "checkpoint_wal", side_effect=RuntimeError("disk full")
        ):
            with caplog.at_level(logging.WARNING, logger="storage.user_model_store"):
                for i in range(50):
                    store.update_signal_profile("cadence", {"interval": i})

        # At least one warning about the checkpoint failure
        assert any(
            "WAL checkpoint" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# store_communication_template WAL checkpoint tests
# ---------------------------------------------------------------------------


class TestStoreCommunicationTemplateWALCheckpoint:
    """checkpoint_wal is called on every store_communication_template call."""

    def test_checkpoint_called_on_first_store(self, store):
        """checkpoint_wal is called immediately after the first template store."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            store.store_communication_template(_make_template("tpl-001"))
            mock_ckpt.assert_called_once_with("user_model")

    def test_checkpoint_called_on_every_store(self, store):
        """checkpoint_wal is called once per store_communication_template call."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            for i in range(5):
                store.store_communication_template(_make_template(f"tpl-{i:03d}"))
            assert mock_ckpt.call_count == 5

    def test_checkpoint_failure_does_not_crash_store(self, store):
        """A checkpoint exception must NOT propagate out of store_communication_template."""
        with patch.object(
            store.db, "checkpoint_wal", side_effect=OSError("read-only filesystem")
        ):
            # Should not raise
            store.store_communication_template(_make_template("tpl-safe"))

        # Verify template was stored despite checkpoint failure
        result = store.get_communication_template(contact_id="contact-001")
        assert result is not None
        assert result["id"] == "tpl-safe"

    def test_checkpoint_failure_logs_warning(self, store, caplog):
        """A checkpoint failure during store logs a WARNING."""
        import logging

        with patch.object(
            store.db, "checkpoint_wal", side_effect=OSError("read-only filesystem")
        ):
            with caplog.at_level(logging.WARNING, logger="storage.user_model_store"):
                store.store_communication_template(_make_template("tpl-warn"))

        assert any(
            "WAL checkpoint" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        )

    def test_template_write_count_increments(self, store):
        """_template_write_count increments with each store call."""
        assert store._template_write_count == 0
        store.store_communication_template(_make_template("tpl-cnt-1"))
        assert store._template_write_count == 1
        store.store_communication_template(_make_template("tpl-cnt-2"))
        assert store._template_write_count == 2


# ---------------------------------------------------------------------------
# update_communication_template WAL checkpoint tests
# ---------------------------------------------------------------------------


class TestUpdateCommunicationTemplateWALCheckpoint:
    """checkpoint_wal is called on every update_communication_template call."""

    def test_checkpoint_called_on_successful_update(self, store):
        """checkpoint_wal is called once after a successful template update."""
        store.store_communication_template(_make_template("tpl-upd-001"))

        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            result = store.update_communication_template(
                "tpl-upd-001", {"formality": 0.9}
            )
            assert result is not None
            mock_ckpt.assert_called_once_with("user_model")

    def test_no_checkpoint_when_template_not_found(self, store):
        """checkpoint_wal is NOT called when the template ID does not exist."""
        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            result = store.update_communication_template(
                "nonexistent-id", {"formality": 0.9}
            )
            assert result is None
            mock_ckpt.assert_not_called()

    def test_no_checkpoint_when_no_valid_fields(self, store):
        """checkpoint_wal is NOT called when updates dict has no allowed fields."""
        store.store_communication_template(_make_template("tpl-noop"))

        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            # "id" is not an allowed field, so no SQL UPDATE is issued
            result = store.update_communication_template(
                "tpl-noop", {"id": "something-else"}
            )
            # Returns existing template without issuing an UPDATE
            assert result is not None
            mock_ckpt.assert_not_called()

    def test_checkpoint_called_on_every_update(self, store):
        """checkpoint_wal is called once per successful update call."""
        store.store_communication_template(_make_template("tpl-multi"))

        with patch.object(store.db, "checkpoint_wal") as mock_ckpt:
            for formality in [0.1, 0.2, 0.3]:
                store.update_communication_template(
                    "tpl-multi", {"formality": formality}
                )
            assert mock_ckpt.call_count == 3

    def test_checkpoint_failure_does_not_crash_update(self, store):
        """A checkpoint exception must NOT propagate out of update_communication_template."""
        store.store_communication_template(_make_template("tpl-crash"))

        with patch.object(
            store.db, "checkpoint_wal", side_effect=RuntimeError("WAL locked")
        ):
            result = store.update_communication_template(
                "tpl-crash", {"formality": 0.7}
            )
            # Update should still succeed and return the updated template
            assert result is not None
            assert abs(result["formality"] - 0.7) < 1e-9

    def test_checkpoint_failure_logs_warning(self, store, caplog):
        """A checkpoint failure during update logs a WARNING."""
        import logging

        store.store_communication_template(_make_template("tpl-log"))

        with patch.object(
            store.db, "checkpoint_wal", side_effect=RuntimeError("WAL locked")
        ):
            with caplog.at_level(logging.WARNING, logger="storage.user_model_store"):
                store.update_communication_template("tpl-log", {"formality": 0.6})

        assert any(
            "WAL checkpoint" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        )

    def test_template_write_count_increments_on_update(self, store):
        """_template_write_count increments on each successful update call."""
        store.store_communication_template(_make_template("tpl-wc"))
        initial = store._template_write_count

        store.update_communication_template("tpl-wc", {"formality": 0.5})
        assert store._template_write_count == initial + 1

        store.update_communication_template("tpl-wc", {"formality": 0.6})
        assert store._template_write_count == initial + 2
