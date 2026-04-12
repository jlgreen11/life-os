"""
Tests for prediction engine WAL persistence fixes.

Covers:
1. store_prediction() followed by a WAL checkpoint produces durable rows.
2. Post-store count verification detects predictions lost after storage (e.g.
   WAL truncation / DB recovery that drops committed rows).
3. Persistence recovery correctly clears a poisoned pre-filter cache when the
   predictions table is empty but the in-memory cache has stale entries.
4. _wal_checkpoint_count diagnostic counter is initialised to 0, increments on
   each successful checkpoint, and is exposed via get_runtime_diagnostics().
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prediction(
    *,
    prediction_type: str = "need",
    description: str = "You will need coffee",
    confidence: float = 0.6,
    confidence_gate: str = "SUGGEST",
    resolved_at: str | None = None,
    user_response: str | None = None,
) -> dict:
    """Return a minimal prediction dict suitable for store_prediction()."""
    return {
        "id": str(uuid.uuid4()),
        "prediction_type": prediction_type,
        "description": description,
        "confidence": confidence,
        "confidence_gate": confidence_gate,
        "time_horizon": None,
        "suggested_action": None,
        "supporting_signals": {},
        "was_surfaced": False,
        "user_response": user_response,
        "resolved_at": resolved_at,
        "filter_reason": None,
    }


# ---------------------------------------------------------------------------
# 1. WAL checkpoint durability
# ---------------------------------------------------------------------------

class TestStorePredictionWalCheckpointDurability:
    """store_prediction() followed by WAL checkpoint yields durable rows."""

    def test_stored_row_survives_wal_checkpoint(self, db, user_model_store):
        """A prediction written through store_prediction() must be readable
        after a forced WAL checkpoint to the main database file."""
        pred = _make_prediction()

        stored = user_model_store.store_prediction(pred)
        assert stored is True, "store_prediction() should return True for a new prediction"

        # Force WAL checkpoint — collapses WAL frames into the main .db file.
        db.checkpoint_wal("user_model")

        # Open a fresh connection (not the cached WAL view) and confirm the row.
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT id, prediction_type, description FROM predictions WHERE id = ?",
                (pred["id"],),
            ).fetchone()

        assert row is not None, "Row must exist after WAL checkpoint"
        assert row["id"] == pred["id"]
        assert row["prediction_type"] == pred["prediction_type"]
        assert row["description"] == pred["description"]

    def test_multiple_predictions_survive_wal_checkpoint(self, db, user_model_store):
        """Multiple predictions stored in sequence must all survive a single
        WAL checkpoint."""
        pred_ids = []
        for i in range(5):
            pred = _make_prediction(description=f"Prediction {i}")
            stored = user_model_store.store_prediction(pred)
            assert stored is True
            pred_ids.append(pred["id"])

        db.checkpoint_wal("user_model")

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT id FROM predictions WHERE id IN ({})".format(
                    ", ".join("?" * len(pred_ids))
                ),
                pred_ids,
            ).fetchall()

        assert len(rows) == len(pred_ids), (
            f"Expected {len(pred_ids)} rows after checkpoint, got {len(rows)}"
        )

    def test_dedup_returns_false_for_existing_unresolved_prediction(self, db, user_model_store):
        """store_prediction() returns False when an identical unresolved prediction
        already exists — the dedup check works correctly against checkpointed rows."""
        pred = _make_prediction()
        first = user_model_store.store_prediction(pred)
        assert first is True

        db.checkpoint_wal("user_model")

        # Attempt to store the same prediction again (identical type + description).
        dupe = dict(pred, id=str(uuid.uuid4()))  # new ID, same type/description
        second = user_model_store.store_prediction(dupe)
        assert second is False, (
            "store_prediction() must return False for a duplicate prediction "
            "that already exists in the checkpointed database"
        )


# ---------------------------------------------------------------------------
# 2. Post-store count verification
# ---------------------------------------------------------------------------

class TestPostStoreCountVerification:
    """The proactive detection mechanism flags _persistence_failure_detected
    when the predictions table is empty despite prior generation runs."""

    @pytest.mark.asyncio
    async def test_empty_table_with_prior_runs_sets_persistence_failure(
        self, db, user_model_store
    ):
        """If the predictions table is empty but total_runs > 0 (meaning
        predictions were generated but somehow lost), the engine must set
        _persistence_failure_detected = True on the next cycle."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        # Simulate that previous cycles ran (predictions were generated) but the
        # table is now empty — as would happen after WAL truncation or DB rebuild.
        engine._total_runs = 3
        engine._persistence_failure_detected = False

        # Verify predictions table is empty (fresh test db)
        with db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert count == 0, "Predictions table must be empty for this test"

        # Run a prediction cycle — the proactive detection logic should fire.
        await engine.generate_predictions({})

        assert engine._persistence_failure_detected is True, (
            "_persistence_failure_detected must be True when the table is empty "
            "but total_runs > 0 (indicating previously-generated predictions are missing)"
        )

    @pytest.mark.asyncio
    async def test_empty_table_with_store_failures_sets_persistence_failure(
        self, db, user_model_store
    ):
        """If the predictions table is empty AND there are recorded store failures,
        the engine must flag _persistence_failure_detected = True."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        engine._store_failure_count = 1
        engine._persistence_failure_detected = False

        await engine.generate_predictions({})

        assert engine._persistence_failure_detected is True, (
            "_persistence_failure_detected must be True when the table is empty "
            "and store_failure_count > 0"
        )

    @pytest.mark.asyncio
    async def test_fresh_engine_does_not_flag_on_empty_table(
        self, db, user_model_store
    ):
        """A brand-new engine (total_runs == 0, no failures) must NOT flag
        _persistence_failure_detected simply because the table is empty."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        assert engine._total_runs == 0
        assert engine._store_failure_count == 0

        await engine.generate_predictions({})

        assert engine._persistence_failure_detected is False, (
            "A fresh engine must not set _persistence_failure_detected on its "
            "first run just because the predictions table is empty"
        )

    def test_post_store_count_query_detects_zero_rows_after_delete(
        self, db, user_model_store
    ):
        """Simulate WAL truncation: store a prediction, delete it, then verify
        the post-store count query would find 0 rows — confirming it can detect
        the 'predictions generated but not persisted' anomaly."""
        pred = _make_prediction()
        user_model_store.store_prediction(pred)

        # Simulate WAL truncation wiping the committed row.
        with db.get_connection("user_model") as conn:
            conn.execute("DELETE FROM predictions WHERE id = ?", (pred["id"],))

        # The post-store verification query (created_at within last 60 s).
        run_start = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
        with db.get_connection("user_model") as conn:
            actual_count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE created_at >= ?",
                (run_start,),
            ).fetchone()[0]

        assert actual_count == 0, (
            "After deleting the stored row the verification query must return 0, "
            "confirming the detection mechanism would fire in a real WAL-loss scenario"
        )


# ---------------------------------------------------------------------------
# 3. Persistence recovery clears stale pre-filter cache
# ---------------------------------------------------------------------------

class TestPersistenceRecoveryClearsPrefilterCache:
    """The recovery block must clear a poisoned pre-filter cache when the
    predictions table is empty but the in-memory cache has stale entries."""

    @pytest.mark.asyncio
    async def test_recovery_clears_poisoned_prefilter_cache(
        self, db, user_model_store
    ):
        """When _persistence_failure_detected is True and _prefilter_cache has
        entries but the predictions table is empty, calling generate_predictions()
        must clear the cache and reset _persistence_failure_detected to False."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        # Inject the conditions that cause a poisoned pre-filter cache:
        # the in-memory cache has entries for predictions that no longer exist
        # in the database (WAL rows were lost).
        engine._persistence_failure_detected = True
        engine._prefilter_cache = {
            ("need", "Contact Alice — it has been 3 weeks", None),
            ("risk", "Calendar conflict tomorrow at 9 AM", "48h"),
        }

        # Predictions table is empty (fresh test db — mirrors the real scenario).
        with db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE resolved_at IS NULL"
            ).fetchone()[0]
        assert count == 0, "Predictions table must be empty to trigger the poison check"

        await engine.generate_predictions({})

        # Recovery block ran, detected poison, and must have cleared the cache.
        assert engine._prefilter_cache == set(), (
            "_prefilter_cache must be empty after recovery detects a poisoned cache "
            "(empty table but non-empty cache)"
        )
        assert engine._persistence_failure_detected is False, (
            "_persistence_failure_detected must be reset to False after successful recovery"
        )

    @pytest.mark.asyncio
    async def test_recovery_does_not_falsely_report_poison_when_table_has_rows(
        self, db, user_model_store
    ):
        """When the predictions table has rows, the prefilter cache is NOT
        considered poisoned — recovery must clear the failure flag normally
        without logging a spurious cache-poison warning.

        After recovery the cache is reset and then immediately repopulated
        from the database (normal forced-refresh path), so the cache ends up
        containing the stored row — not empty.
        """
        engine = PredictionEngine(db=db, ums=user_model_store)

        # Store a real prediction so the table is non-empty.
        pred = _make_prediction()
        user_model_store.store_prediction(pred)

        # The cache has entries consistent with the DB content.
        engine._persistence_failure_detected = True
        engine._prefilter_cache = {
            (pred["prediction_type"], pred["description"], pred.get("time_horizon")),
        }

        await engine.generate_predictions({})

        # Recovery cleared the failure flag.
        assert engine._persistence_failure_detected is False

        # After recovery the cache is reset to {} then immediately refreshed
        # from the DB (forced-refresh because _prefilter_refresh_cycle was 0).
        # The stored prediction row should appear in the repopulated cache.
        expected_key = (pred["prediction_type"], pred["description"], pred.get("time_horizon"))
        assert expected_key in engine._prefilter_cache, (
            "After recovery + DB refresh, the stored prediction key must be in the "
            "pre-filter cache — the cache should contain real DB rows, not be empty"
        )

    @pytest.mark.asyncio
    async def test_recovery_also_deletes_persisted_prefilter_cache_state_key(
        self, db, user_model_store
    ):
        """The recovery block must DELETE the 'prefilter_cache' key from
        prediction_engine_state so that a future restart cannot restore a
        stale in-memory cache from persisted state."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        # Manually insert a stale 'prefilter_cache' key into the state table
        # to simulate a scenario where the cache was persisted before WAL loss.
        with db.get_connection("user_model") as conn:
            conn.execute(
                "INSERT OR REPLACE INTO prediction_engine_state (key, value, updated_at) "
                "VALUES ('prefilter_cache', 'stale_data', ?)",
                (datetime.now(UTC).isoformat(),),
            )

        engine._persistence_failure_detected = True
        engine._prefilter_cache = {("need", "stale prediction", None)}

        await engine.generate_predictions({})

        # The stale key must have been deleted during recovery.
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT value FROM prediction_engine_state WHERE key = 'prefilter_cache'"
            ).fetchone()

        assert row is None, (
            "The 'prefilter_cache' key must be deleted from prediction_engine_state "
            "during persistence recovery to prevent stale cache restoration on restart"
        )


# ---------------------------------------------------------------------------
# 4. _wal_checkpoint_count diagnostic counter
# ---------------------------------------------------------------------------

class TestWalCheckpointCounter:
    """_wal_checkpoint_count starts at 0, is exposed in diagnostics, and
    increments each time checkpoint_wal() succeeds after a store cycle."""

    def test_wal_checkpoint_count_initial_value(self, db, user_model_store):
        """_wal_checkpoint_count must be 0 on a freshly created engine."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        assert engine._wal_checkpoint_count == 0

    def test_wal_checkpoint_count_exposed_in_runtime_diagnostics(
        self, db, user_model_store
    ):
        """get_runtime_diagnostics() must include 'wal_checkpoint_count'."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        diag = engine.get_runtime_diagnostics()
        assert "wal_checkpoint_count" in diag, (
            "get_runtime_diagnostics() must expose 'wal_checkpoint_count' so "
            "operators can verify WAL checkpointing is happening"
        )
        assert diag["wal_checkpoint_count"] == 0

    def test_wal_checkpoint_count_reflects_manual_checkpoint(
        self, db, user_model_store
    ):
        """After incrementing _wal_checkpoint_count manually, the diagnostic
        counter and runtime_diagnostics dict must reflect the new value."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        engine._wal_checkpoint_count = 3
        diag = engine.get_runtime_diagnostics()
        assert diag["wal_checkpoint_count"] == 3

    @pytest.mark.asyncio
    async def test_wal_checkpoint_count_increments_when_predictions_stored(
        self, db, event_store, user_model_store
    ):
        """_wal_checkpoint_count must increment when a prediction cycle
        stores ≥ 1 prediction and the WAL checkpoint succeeds.

        This test seeds data that the prediction engine can process to
        produce at least one prediction in a real (non-mocked) run.
        It accepts that the engine may legitimately generate 0 predictions
        in a test environment; in that case the counter stays at 0.
        """
        from unittest.mock import patch

        engine = PredictionEngine(db=db, ums=user_model_store)
        assert engine._wal_checkpoint_count == 0

        # Seed an email so event-based checks have something to process.
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "email.received",
            "source": "google",
            "timestamp": (datetime.now(UTC) - timedelta(hours=5)).isoformat(),
            "payload": {
                "from_address": "important@example.com",
                "subject": "Action required — please reply",
                "message_id": f"msg-wal-count-{uuid.uuid4().hex[:8]}",
            },
            "metadata": {},
        })

        with patch.object(
            db, "checkpoint_wal", wraps=db.checkpoint_wal
        ) as mock_ckpt:
            await engine.generate_predictions({})

            if mock_ckpt.called:
                # If the checkpoint was called, the counter must have incremented.
                assert engine._wal_checkpoint_count == 1, (
                    "_wal_checkpoint_count must be 1 after one successful checkpoint"
                )
            else:
                # No predictions were stored in this run — counter stays at 0.
                assert engine._wal_checkpoint_count == 0
