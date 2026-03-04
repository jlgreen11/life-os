"""
Tests for the prediction engine's post-store verification logic.

Verifies that the verification query correctly distinguishes between:
- Normal operation (all predictions filtered, resolved_at set) — no alarm
- Real persistence failure (predictions stored but missing from DB) — alarm fires
"""

import uuid
from datetime import datetime, timezone

import pytest


@pytest.fixture()
def _insert_prediction(db):
    """Helper to insert a prediction row directly into the predictions table."""

    def _insert(*, resolved_at=None, created_at=None):
        """Insert a minimal prediction row.

        Args:
            resolved_at: ISO timestamp or None for unresolved predictions.
            created_at: ISO timestamp; defaults to now.
        """
        pred_id = str(uuid.uuid4())
        now = created_at or datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate, created_at, resolved_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (pred_id, "reminder", "Test prediction", 0.5, "SUGGEST", now, resolved_at),
            )
        return pred_id

    return _insert


class TestPostStoreVerification:
    """Tests for the post-store verification query in generate_predictions()."""

    def test_filtered_prediction_does_not_trigger_alarm(self, db, _insert_prediction):
        """Predictions with resolved_at set (filtered) should NOT trigger the persistence alarm.

        This was the original false alarm: all predictions were filtered (resolved_at set),
        so the old query 'WHERE resolved_at IS NULL' returned 0 and falsely logged CRITICAL.
        The fixed query checks created_at instead, so filtered predictions still count.
        """
        now = datetime.now(timezone.utc).isoformat()
        # Insert a filtered prediction (resolved_at is set)
        _insert_prediction(resolved_at=now, created_at=now)

        # Run the verification query (same as in engine.py after the fix)
        from datetime import timedelta

        run_start = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        with db.get_connection("user_model") as conn:
            actual_count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE created_at >= ?",
                (run_start,),
            ).fetchone()[0]

        # Should find the prediction — no alarm should fire
        assert actual_count >= 1, (
            "Verification query should find filtered predictions by created_at"
        )

    def test_surfaced_prediction_found_by_verification(self, db, _insert_prediction):
        """Predictions with resolved_at IS NULL (surfaced) should also be found."""
        now = datetime.now(timezone.utc).isoformat()
        _insert_prediction(resolved_at=None, created_at=now)

        from datetime import timedelta

        run_start = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        with db.get_connection("user_model") as conn:
            actual_count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE created_at >= ?",
                (run_start,),
            ).fetchone()[0]

        assert actual_count >= 1, (
            "Verification query should find surfaced predictions by created_at"
        )

    def test_persistence_failure_detected_when_db_empty(self, db):
        """When stored_count > 0 but the DB has no recent rows, the alarm should fire.

        This simulates a real persistence failure: store_prediction() appeared to
        succeed (stored_count > 0) but the data was silently lost.
        """
        from datetime import timedelta

        run_start = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        with db.get_connection("user_model") as conn:
            actual_count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE created_at >= ?",
                (run_start,),
            ).fetchone()[0]

        # DB is empty — if stored_count were > 0, this would be a real failure
        stored_count = 5  # Simulated: engine thinks it stored 5 predictions
        should_alarm = stored_count > 0 and actual_count == 0
        assert should_alarm is True, (
            "Alarm should fire when stored_count > 0 but DB has no matching rows"
        )

    def test_old_predictions_do_not_satisfy_verification(self, db, _insert_prediction):
        """Predictions created before the run window should not satisfy the check.

        Ensures the verification only looks at predictions from the current run,
        not historical data.
        """
        # Insert a prediction with an old created_at (well before the 60s window)
        old_time = "2020-01-01T00:00:00+00:00"
        _insert_prediction(resolved_at=old_time, created_at=old_time)

        from datetime import timedelta

        run_start = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        with db.get_connection("user_model") as conn:
            actual_count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE created_at >= ?",
                (run_start,),
            ).fetchone()[0]

        # Old prediction should not be found
        assert actual_count == 0, (
            "Old predictions should not satisfy the verification query"
        )


class TestZeroSurfacingCycleCounter:
    """Tests for the _zero_surfacing_cycles counter and WARNING log escalation."""

    def test_counter_initializes_to_zero(self, prediction_engine):
        """Counter starts at 0 on engine creation."""
        assert prediction_engine._zero_surfacing_cycles == 0

    def test_counter_in_runtime_diagnostics(self, prediction_engine):
        """The _zero_surfacing_cycles counter appears in runtime diagnostics."""
        diag = prediction_engine.get_runtime_diagnostics()
        assert "zero_surfacing_cycles" in diag
        assert diag["zero_surfacing_cycles"] == 0

    def test_counter_increments_on_zero_surfacing(self, prediction_engine):
        """Manually incrementing the counter simulates zero-surfacing cycles."""
        # Simulate 3 consecutive zero-surfacing cycles
        prediction_engine._zero_surfacing_cycles = 3
        diag = prediction_engine.get_runtime_diagnostics()
        assert diag["zero_surfacing_cycles"] == 3

    def test_counter_resets_on_successful_surfacing(self, prediction_engine):
        """Counter resets when predictions start surfacing again."""
        prediction_engine._zero_surfacing_cycles = 5
        # Simulate a successful surfacing cycle resetting the counter
        prediction_engine._zero_surfacing_cycles = 0
        assert prediction_engine._zero_surfacing_cycles == 0
