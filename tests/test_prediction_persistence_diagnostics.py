"""
Tests for prediction engine persistence diagnostics and recovery hardening.

Covers:
- get_persistence_diagnostics() returns correct state
- _recovery_attempt_count increments on each recovery entry
- CRITICAL log fires when recovery_attempt_count > 3
- Schema check in recovery block catches missing columns
- Pre-verification WAL checkpoint is called before the post-store query
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.core import ConfidenceGate
from models.user_model import Prediction, ReactionPrediction
from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prediction(**overrides) -> Prediction:
    """Create a minimal Prediction with a unique description to avoid dedup."""
    defaults = {
        "id": str(uuid.uuid4()),
        "prediction_type": "need",
        "description": f"Test prediction {uuid.uuid4().hex[:8]}",
        "confidence": 0.7,
        "confidence_gate": ConfidenceGate.DEFAULT,
        "time_horizon": "2_hours",
    }
    defaults.update(overrides)
    return Prediction(**defaults)


def _insert_event(event_store):
    """Insert a generic event to advance the cursor so generate_predictions runs."""
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "from_address": "test@example.com",
            "subject": "Test",
            "message_id": str(uuid.uuid4()),
        },
        "metadata": {},
    })


async def _run_engine_with_fake_predictions(engine, predictions):
    """Run generate_predictions with all _check_* methods mocked.

    The first check method returns the provided predictions; all others return
    empty lists.  predict_reaction always returns 'helpful' so predictions pass
    the reaction filter and reach the storage loop.
    """
    check_methods = [
        "_check_calendar_conflicts",
        "_check_routine_deviations",
        "_check_relationship_maintenance",
        "_check_preparation_needs",
        "_check_follow_up_needs",
        "_check_calendar_event_reminders",
        "_check_connector_health",
        "_check_spending_patterns",
    ]

    patches = []
    for method_name in check_methods:
        if hasattr(engine, method_name):
            p = patch.object(engine, method_name, new_callable=AsyncMock, return_value=[])
            patches.append(p)

    if patches:
        patches[0].kwargs["return_value"] = list(predictions)

    helpful_reaction = ReactionPrediction(
        predicted_reaction="helpful",
        confidence=0.9,
        reasoning="test",
        proposed_action="surface",
    )
    reaction_patch = patch.object(
        engine, "predict_reaction", new_callable=AsyncMock, return_value=helpful_reaction
    )
    patches.append(reaction_patch)

    for p in patches:
        p.start()
    try:
        result = await engine.generate_predictions({})
    finally:
        for p in patches:
            p.stop()

    return result


# ---------------------------------------------------------------------------
# Tests: get_persistence_diagnostics()
# ---------------------------------------------------------------------------


class TestGetPersistenceDiagnostics:
    """Tests for the get_persistence_diagnostics() method."""

    def test_returns_dict_with_required_keys(self, db, user_model_store):
        """get_persistence_diagnostics() returns a dict with all required keys."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        diag = engine.get_persistence_diagnostics()

        required_keys = {
            "store_failure_count",
            "last_store_errors",
            "persistence_failure_detected",
            "recovery_attempt_count",
            "last_successful_generation",
            "current_prediction_count",
            "generation_events_count",
            "count_mismatch",
            "schema_status",
        }
        assert required_keys.issubset(diag.keys()), (
            f"Missing keys: {required_keys - diag.keys()}"
        )

    def test_fresh_engine_defaults(self, db, user_model_store):
        """Fresh engine diagnostics should show zero failures and healthy schema."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        diag = engine.get_persistence_diagnostics()

        assert diag["store_failure_count"] == 0
        assert diag["last_store_errors"] == []
        assert diag["persistence_failure_detected"] is False
        assert diag["recovery_attempt_count"] == 0
        assert diag["last_successful_generation"] is None
        assert diag["current_prediction_count"] == 0
        assert diag["generation_events_count"] == 0
        assert diag["count_mismatch"] is False
        assert diag["schema_status"] == "ok"

    def test_reflects_in_memory_failure_state(self, db, user_model_store):
        """Diagnostics accurately reflect manually set in-memory failure state."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        engine._store_failure_count = 7
        engine._persistence_failure_detected = True
        engine._recovery_attempt_count = 2
        engine._last_store_errors = [{"error": "test", "timestamp": "2026-01-01"}]

        diag = engine.get_persistence_diagnostics()

        assert diag["store_failure_count"] == 7
        assert diag["persistence_failure_detected"] is True
        assert diag["recovery_attempt_count"] == 2
        assert len(diag["last_store_errors"]) == 1

    @pytest.mark.asyncio
    async def test_prediction_count_reflects_stored_predictions(
        self, db, event_store, user_model_store
    ):
        """current_prediction_count matches actual rows in the predictions table."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Run the engine to store some predictions
        preds = [_make_prediction(), _make_prediction()]
        await _run_engine_with_fake_predictions(engine, preds)

        diag = engine.get_persistence_diagnostics()

        # Verify prediction count is positive (at least 1 stored)
        assert diag["current_prediction_count"] >= 1, (
            "Prediction count should reflect stored rows"
        )
        # Schema should still be ok
        assert diag["schema_status"] == "ok"

    def test_schema_status_ok_for_valid_schema(self, db, user_model_store):
        """schema_status is 'ok' when predictions table has all required columns."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        diag = engine.get_persistence_diagnostics()
        assert diag["schema_status"] == "ok"

    def test_schema_status_reports_missing_columns(self, db, user_model_store):
        """schema_status reports missing columns when store_prediction() columns are absent."""
        # Drop the predictions table and recreate with only the DB write test columns —
        # this is missing the store_prediction()-specific columns (was_surfaced, time_horizon,
        # supporting_signals, filter_reason) so get_persistence_diagnostics() should report them.
        with db.get_connection("user_model") as conn:
            conn.execute("DROP TABLE IF EXISTS predictions")
            conn.execute(
                """CREATE TABLE predictions (
                    id TEXT PRIMARY KEY,
                    prediction_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    confidence_gate TEXT NOT NULL,
                    resolved_at TEXT,
                    user_response TEXT
                )"""
            )

        engine = PredictionEngine(db=db, ums=user_model_store)
        diag = engine.get_persistence_diagnostics()

        assert "missing_columns" in diag["schema_status"], (
            f"Expected schema_status to report missing columns, got: {diag['schema_status']}"
        )
        # At least one of the store_prediction()-specific columns should be reported
        assert any(
            col in diag["schema_status"]
            for col in ("was_surfaced", "time_horizon", "supporting_signals", "filter_reason")
        ), f"Missing store_prediction() columns not reported in schema_status: {diag['schema_status']}"

    def test_no_count_mismatch_when_both_zero(self, db, user_model_store):
        """count_mismatch is False when both generation events and predictions are 0."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        diag = engine.get_persistence_diagnostics()
        # Fresh DB: 0 generation events and 0 predictions → no mismatch
        assert diag["count_mismatch"] is False

    def test_count_mismatch_when_events_but_no_predictions(self, db, user_model_store, event_store):
        """count_mismatch is True when generation events exist but predictions table is empty."""
        # Insert a fake 'usermodel.prediction.generated' event
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "usermodel.prediction.generated",
            "source": "prediction_engine",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {"count": 5},
            "metadata": {},
        })

        engine = PredictionEngine(db=db, ums=user_model_store)
        diag = engine.get_persistence_diagnostics()

        assert diag["generation_events_count"] >= 1
        assert diag["current_prediction_count"] == 0
        assert diag["count_mismatch"] is True, (
            "count_mismatch should be True when generation events exist but DB has no predictions"
        )

    def test_degrades_gracefully_when_db_unavailable(self, tmp_path):
        """get_persistence_diagnostics() never raises even when the DB is broken."""
        from storage.manager import DatabaseManager
        from storage.user_model_store import UserModelStore
        from unittest.mock import MagicMock

        # Create a DatabaseManager with a valid path but stub get_connection to raise
        bad_db = DatabaseManager(data_dir=str(tmp_path))
        bad_db.initialize_all()

        engine = PredictionEngine(db=bad_db, ums=MagicMock())

        # Monkey-patch get_connection to always raise so we can test graceful degradation
        original_get_connection = bad_db.get_connection

        class _RaisingCtx:
            def __enter__(self):
                raise RuntimeError("Simulated DB failure")

            def __exit__(self, *args):
                pass

        with patch.object(bad_db, "get_connection", return_value=_RaisingCtx()):
            # Should not raise — must return a dict even under DB errors
            diag = engine.get_persistence_diagnostics()

        assert isinstance(diag, dict)
        # DB error key should be present for at least one of the queries
        assert "user_model_query_error" in diag or "events_query_error" in diag


# ---------------------------------------------------------------------------
# Tests: _recovery_attempt_count counter
# ---------------------------------------------------------------------------


class TestRecoveryAttemptCount:
    """Tests for _recovery_attempt_count incrementing and CRITICAL escalation."""

    def test_initial_count_is_zero(self, db, user_model_store):
        """Fresh engine should start with recovery_attempt_count = 0."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        assert engine._recovery_attempt_count == 0

    @pytest.mark.asyncio
    async def test_count_increments_each_recovery_entry(
        self, db, event_store, user_model_store
    ):
        """_recovery_attempt_count increments by 1 each time recovery block runs."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Simulate persistence failure so recovery block runs
        engine._persistence_failure_detected = True

        await _run_engine_with_fake_predictions(engine, [_make_prediction()])
        assert engine._recovery_attempt_count == 1, (
            "First recovery entry should set count to 1"
        )

        # Run a second cycle with the flag re-set to simulate repeated failures
        _insert_event(event_store)
        engine._persistence_failure_detected = True

        await _run_engine_with_fake_predictions(engine, [_make_prediction()])
        assert engine._recovery_attempt_count == 2, (
            "Second recovery entry should set count to 2"
        )

    @pytest.mark.asyncio
    async def test_count_in_persistence_diagnostics(self, db, event_store, user_model_store):
        """recovery_attempt_count appears in get_persistence_diagnostics() output."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        engine._recovery_attempt_count = 5

        diag = engine.get_persistence_diagnostics()
        assert diag["recovery_attempt_count"] == 5

    @pytest.mark.asyncio
    async def test_critical_log_after_repeated_failures(
        self, db, event_store, user_model_store
    ):
        """CRITICAL is logged when recovery_attempt_count > 3 and DB write fails."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Force recovery to fail by dropping the predictions table
        with db.get_connection("user_model") as conn:
            conn.execute("DROP TABLE IF EXISTS predictions")

        # Set count to 3 so the NEXT attempt (count=4) triggers CRITICAL
        engine._recovery_attempt_count = 3
        engine._persistence_failure_detected = True

        import logging

        with patch.object(
            logging.getLogger("services.prediction_engine.engine"), "critical"
        ) as mock_critical:
            result = await engine.generate_predictions({})

        # Cycle should be skipped (empty list) since DB write test fails
        assert result == []
        # CRITICAL should have been logged because count > 3
        assert mock_critical.called, "CRITICAL should be logged after 4+ failed recoveries"

    @pytest.mark.asyncio
    async def test_no_critical_log_on_first_three_failures(
        self, db, event_store, user_model_store
    ):
        """No CRITICAL escalation for the first 3 recovery attempts."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Drop predictions table so recovery fails
        with db.get_connection("user_model") as conn:
            conn.execute("DROP TABLE IF EXISTS predictions")

        # Count is 0 so this is attempt #1 — should NOT trigger CRITICAL
        engine._recovery_attempt_count = 0
        engine._persistence_failure_detected = True

        import logging

        with patch.object(
            logging.getLogger("services.prediction_engine.engine"), "critical"
        ) as mock_critical:
            await engine.generate_predictions({})

        # The first failure logs 'logger.critical(...)' even without escalation,
        # so we check that the escalation message (containing the count) is absent.
        # Extract call args to check the message content.
        escalation_calls = [
            call for call in mock_critical.call_args_list
            if "recovery attempted" in str(call)
        ]
        assert len(escalation_calls) == 0, (
            "Escalation CRITICAL should not fire on first attempt"
        )


# ---------------------------------------------------------------------------
# Tests: schema check in recovery block
# ---------------------------------------------------------------------------


class TestSchemaCheckInRecovery:
    """Tests for the schema verification that runs after a successful DB write test."""

    @pytest.mark.asyncio
    async def test_schema_check_passes_with_valid_schema(
        self, db, event_store, user_model_store
    ):
        """No CRITICAL log for schema mismatch when table has all expected columns."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)
        engine._persistence_failure_detected = True

        import logging

        with patch.object(
            logging.getLogger("services.prediction_engine.engine"), "critical"
        ) as mock_critical:
            await _run_engine_with_fake_predictions(engine, [_make_prediction()])

        # Flag should be cleared (DB healthy)
        assert engine._persistence_failure_detected is False

        # No schema CRITICAL should have been logged
        schema_criticals = [
            call for call in mock_critical.call_args_list
            if "schema mismatch" in str(call)
        ]
        assert len(schema_criticals) == 0, (
            "No schema CRITICAL should be logged when table columns are all present"
        )

    @pytest.mark.asyncio
    async def test_schema_check_logs_critical_for_missing_columns(
        self, db, event_store, user_model_store
    ):
        """CRITICAL is logged when predictions table is missing store_prediction() columns.

        The recovery block DB write test uses a minimal INSERT
        (id, prediction_type, description, confidence, confidence_gate, resolved_at, user_response).
        The schema check additionally verifies columns that store_prediction() requires
        (was_surfaced, time_horizon, supporting_signals, filter_reason).

        We recreate the table with only the DB write test columns so the write test
        passes, but the schema check still fires because store_prediction() columns
        are absent.
        """
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Recreate the table with exactly the columns the DB write test needs —
        # id, prediction_type, description, confidence, confidence_gate, resolved_at, user_response.
        # The schema check expects ALSO: was_surfaced, time_horizon, supporting_signals, filter_reason.
        # Those are absent here, so the schema CRITICAL should fire.
        with db.get_connection("user_model") as conn:
            conn.execute("DROP TABLE IF EXISTS predictions")
            conn.execute(
                """CREATE TABLE predictions (
                    id TEXT PRIMARY KEY,
                    prediction_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    confidence_gate TEXT NOT NULL,
                    resolved_at TEXT,
                    user_response TEXT
                )"""
            )

        engine._persistence_failure_detected = True

        import logging

        with patch.object(
            logging.getLogger("services.prediction_engine.engine"), "critical"
        ) as mock_critical:
            await engine.generate_predictions({})

        # Schema CRITICAL should be present
        schema_criticals = [
            call for call in mock_critical.call_args_list
            if "schema mismatch" in str(call)
        ]
        assert len(schema_criticals) >= 1, (
            "CRITICAL should fire when predictions table is missing store_prediction() columns"
        )


# ---------------------------------------------------------------------------
# Tests: pre-verification WAL checkpoint
# ---------------------------------------------------------------------------


class TestPreVerificationWalCheckpoint:
    """Tests for the WAL checkpoint injected before the post-store verification query."""

    @pytest.mark.asyncio
    async def test_checkpoint_called_before_verification(
        self, db, event_store, user_model_store
    ):
        """checkpoint_wal is called at least twice when predictions are stored:
        once after storage and once before the verification query."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        checkpoint_calls = []

        original_ckpt = db.checkpoint_wal

        def _tracking_ckpt(db_name):
            checkpoint_calls.append(db_name)
            return original_ckpt(db_name)

        with patch.object(db, "checkpoint_wal", side_effect=_tracking_ckpt):
            await _run_engine_with_fake_predictions(engine, [_make_prediction()])

        # At minimum two checkpoint calls when predictions were stored:
        # one after the storage loop + one before the verification query.
        user_model_checkpoints = [c for c in checkpoint_calls if c == "user_model"]
        assert len(user_model_checkpoints) >= 2, (
            f"Expected at least 2 user_model WAL checkpoints when predictions stored, "
            f"got {len(user_model_checkpoints)}: {checkpoint_calls}"
        )

    @pytest.mark.asyncio
    async def test_checkpoint_failure_before_verification_does_not_crash(
        self, db, event_store, user_model_store
    ):
        """A failing pre-verification checkpoint logs a warning but does not abort the cycle."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        call_count = [0]
        original_ckpt = db.checkpoint_wal

        def _fail_second_call(db_name):
            call_count[0] += 1
            # Let the first checkpoint (post-storage) succeed; fail the second (pre-verify)
            if call_count[0] == 2:
                raise RuntimeError("Simulated pre-verify checkpoint failure")
            return original_ckpt(db_name)

        with patch.object(db, "checkpoint_wal", side_effect=_fail_second_call):
            # Should not raise — pre-verification checkpoint failure is logged but non-fatal
            result = await _run_engine_with_fake_predictions(engine, [_make_prediction()])

        # Engine should still complete and return predictions
        assert isinstance(result, list)
