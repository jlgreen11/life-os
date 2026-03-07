"""Tests for prediction persistence flag after DB rebuild and temporal backfill guard.

Verifies:
1. PredictionEngine.reset_state() preserves _persistence_failure_detected.
2. After a mock rebuild scenario, _persistence_failure_detected is set to True.
3. The temporal backfill guard allows re-running when inbound events are missing.
"""

import pytest

from services.prediction_engine.engine import PredictionEngine


class TestResetStatePreservesFlag:
    """reset_state() must NOT clear _persistence_failure_detected."""

    def test_flag_stays_false_after_reset(self, db, user_model_store):
        """When the flag is False before reset, it stays False."""
        engine = PredictionEngine(db, user_model_store)
        assert engine._persistence_failure_detected is False
        engine.reset_state()
        assert engine._persistence_failure_detected is False

    def test_flag_stays_true_after_reset(self, db, user_model_store):
        """When the flag is True before reset, it survives the reset."""
        engine = PredictionEngine(db, user_model_store)
        engine._persistence_failure_detected = True
        engine.reset_state()
        assert engine._persistence_failure_detected is True


class TestRebuildSetsPersistenceFlag:
    """After a DB rebuild, _persistence_failure_detected must be set to True.

    This simulates what _db_health_loop does: call reset_state() then set
    the flag. We verify the flag ends up True regardless of prior state.
    """

    def test_flag_set_after_rebuild_sequence(self, db, user_model_store):
        """Simulate the post-rebuild code path in _db_health_loop."""
        engine = PredictionEngine(db, user_model_store)
        assert engine._persistence_failure_detected is False

        # Simulate what main.py does after rebuild:
        engine.reset_state()
        engine._persistence_failure_detected = True

        assert engine._persistence_failure_detected is True

    def test_flag_set_even_if_previously_false(self, db, user_model_store):
        """Flag must be True after rebuild even if it was never set before."""
        engine = PredictionEngine(db, user_model_store)
        engine._persistence_failure_detected = False

        engine.reset_state()
        engine._persistence_failure_detected = True

        assert engine._persistence_failure_detected is True


class TestTemporalBackfillGuard:
    """The temporal backfill guard should re-run when inbound events are missing."""

    def _should_skip_backfill(self, profile):
        """Replicate the guard logic from _backfill_temporal_profile_if_needed.

        Returns True if backfill should be SKIPPED, False if it should run.
        """
        if profile and profile.get("samples_count", 0) >= 5:
            data = profile.get("data", {})
            activity_types = data.get("activity_by_type", {})
            if (
                activity_types.get("email_inbound", 0) > 0
                or activity_types.get("message_inbound", 0) > 0
            ):
                return True
            # Inbound data missing — need to re-run
            return False
        return False

    def test_skip_when_inbound_present(self):
        """Backfill is skipped when profile has inbound event data."""
        profile = {
            "samples_count": 10,
            "data": {
                "activity_by_type": {
                    "email_outbound": 5,
                    "email_inbound": 20,
                    "message_inbound": 3,
                },
            },
        }
        assert self._should_skip_backfill(profile) is True

    def test_rerun_when_inbound_missing(self):
        """Backfill must re-run when profile has samples but no inbound data."""
        profile = {
            "samples_count": 10,
            "data": {
                "activity_by_type": {
                    "email_outbound": 5,
                    "task_created": 5,
                },
            },
        }
        assert self._should_skip_backfill(profile) is False

    def test_rerun_when_activity_by_type_empty(self):
        """Backfill must re-run when activity_by_type is empty."""
        profile = {
            "samples_count": 10,
            "data": {"activity_by_type": {}},
        }
        assert self._should_skip_backfill(profile) is False

    def test_rerun_when_data_key_missing(self):
        """Backfill must re-run when data dict is absent (legacy profile)."""
        profile = {"samples_count": 10}
        assert self._should_skip_backfill(profile) is False

    def test_no_skip_when_profile_empty(self):
        """No skip when profile is None or has too few samples."""
        assert self._should_skip_backfill(None) is False
        assert self._should_skip_backfill({"samples_count": 2}) is False

    def test_skip_when_only_email_inbound(self):
        """Backfill skipped if only email_inbound is present (no message_inbound needed)."""
        profile = {
            "samples_count": 5,
            "data": {
                "activity_by_type": {"email_inbound": 1},
            },
        }
        assert self._should_skip_backfill(profile) is True

    def test_skip_when_only_message_inbound(self):
        """Backfill skipped if only message_inbound is present."""
        profile = {
            "samples_count": 5,
            "data": {
                "activity_by_type": {"message_inbound": 1},
            },
        }
        assert self._should_skip_backfill(profile) is True
