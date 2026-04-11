"""
Tests for prediction engine intra-batch dedup and pre-filter cache persistence.

Two related improvements are tested here:

1. **Intra-batch dedup**: When two _check_* methods generate the same prediction
   in a single cycle (same type+description+time_horizon), only the first is kept.
   Without this, duplicates pass the pre-filter (loaded before generation starts)
   and hit store_prediction()'s dedup, wasting DB queries and telemetry events.

2. **Pre-filter cache persistence**: The existing-predictions set is now maintained
   in-memory (_prefilter_cache) and refreshed from DB only every 4th cycle instead
   of every 15-minute cycle.  Predictions stored during each cycle are appended to
   the cache immediately so cross-cycle dedup stays accurate without a DB round-trip.
"""

import logging
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from models.core import ConfidenceGate
from models.user_model import Prediction
from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_prediction(
    description: str = "Test prediction",
    prediction_type: str = "opportunity",
    confidence: float = 0.65,
    time_horizon: str | None = "this_week",
) -> Prediction:
    """Create a minimal Prediction instance with the given fields.

    Confidence 0.65 puts the prediction in the DEFAULT gate (0.6–0.8), ensuring
    it passes the confidence floor check (>= 0.3) in generate_predictions().
    """
    return Prediction(
        prediction_type=prediction_type,
        description=description,
        confidence=confidence,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon=time_horizon,
    )


# ---------------------------------------------------------------------------
# Intra-batch dedup tests
# ---------------------------------------------------------------------------


class TestIntraBatchDedup:
    """Verify that duplicate predictions within a single cycle are eliminated."""

    @pytest.mark.asyncio
    async def test_intra_batch_dedup_removes_second_identical_prediction(
        self, db, user_model_store
    ):
        """Two _check_* methods returning identical predictions → only one stored in DB.

        Patches _check_relationship_maintenance and _check_follow_up_needs to both
        return the same prediction (same type, description, time_horizon).  After
        generate_predictions() runs, the DB should contain exactly one row for that
        description, and _last_run_diagnostics should report intra_batch_dedup=1.
        """
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        duplicate_desc = "Reach out to alice@example.com (overdue by about 2 weeks)"

        shared_pred_1 = _make_prediction(description=duplicate_desc)
        shared_pred_2 = _make_prediction(description=duplicate_desc)

        async def mock_relationship_preds(context):
            """Return the first copy of the duplicate prediction."""
            return [shared_pred_1]

        async def mock_followup_preds(context):
            """Return the second copy of the duplicate prediction."""
            return [shared_pred_2]

        async def mock_empty(context):
            """Stub for _check_* methods that should return nothing."""
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_empty),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_relationship_preds),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_followup_preds),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            # Ensure the time-based trigger fires (first run, last_time_based_run=None)
            await engine.generate_predictions({})

        # Exactly one row should be stored for the duplicate description
        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT id FROM predictions WHERE description = ?",
                (duplicate_desc,),
            ).fetchall()
        assert len(rows) == 1, (
            f"Expected 1 prediction stored, got {len(rows)}. "
            "Intra-batch dedup should have removed the second copy before storage."
        )

        # Diagnostics should record that 1 intra-batch duplicate was removed
        assert engine._last_run_diagnostics.get("intra_batch_dedup") == 1, (
            f"Expected intra_batch_dedup=1 in diagnostics, "
            f"got {engine._last_run_diagnostics.get('intra_batch_dedup')!r}"
        )

    @pytest.mark.asyncio
    async def test_intra_batch_dedup_preserves_distinct_predictions(
        self, db, user_model_store
    ):
        """Distinct predictions from different _check_* methods are NOT deduplicated.

        When two methods return predictions with different descriptions, both should
        survive intra-batch dedup and be available for further filtering/storage.
        """
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        desc_a = "Follow up on the project proposal email"
        desc_b = "Reach out to bob@example.com (overdue by about a month)"

        async def mock_method_a(context):
            """Return first unique prediction."""
            return [_make_prediction(description=desc_a, prediction_type="reminder")]

        async def mock_method_b(context):
            """Return second, distinct prediction."""
            return [_make_prediction(description=desc_b, prediction_type="opportunity")]

        async def mock_empty(context):
            """Stub that returns nothing."""
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_empty),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_method_a),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_method_b),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        # Both distinct predictions should reach the DB (no intra-batch removal)
        with db.get_connection("user_model") as conn:
            count_a = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE description = ?", (desc_a,)
            ).fetchone()[0]
            count_b = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE description = ?", (desc_b,)
            ).fetchone()[0]

        assert count_a == 1, f"Expected 1 row for desc_a, got {count_a}"
        assert count_b == 1, f"Expected 1 row for desc_b, got {count_b}"

        # No intra-batch dedup should have occurred
        assert engine._last_run_diagnostics.get("intra_batch_dedup", 0) == 0, (
            "No intra-batch dedup expected for distinct predictions"
        )

    @pytest.mark.asyncio
    async def test_intra_batch_dedup_key_includes_time_horizon(
        self, db, user_model_store
    ):
        """Same description but different time_horizon → NOT considered a duplicate.

        The dedup key is (prediction_type, description, time_horizon), so two
        predictions with the same type+description but different time horizons are
        distinct and should both be stored.
        """
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        common_desc = "Prepare for quarterly review"
        pred_short = _make_prediction(
            description=common_desc,
            prediction_type="need",
            time_horizon="24_hours",
        )
        pred_long = _make_prediction(
            description=common_desc,
            prediction_type="need",
            time_horizon="this_week",
        )

        async def mock_returns_two(context):
            """Return two predictions that differ only in time_horizon."""
            return [pred_short, pred_long]

        async def mock_empty(context):
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_returns_two),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT time_horizon FROM predictions WHERE description = ?",
                (common_desc,),
            ).fetchall()

        # Both should be stored: they differ by time_horizon
        horizons = {r[0] for r in rows}
        assert len(rows) == 2, (
            f"Expected 2 rows (different time_horizon), got {len(rows)}: {horizons}"
        )
        assert "24_hours" in horizons
        assert "this_week" in horizons

        # No intra-batch dedup since descriptions differ in effective key
        assert engine._last_run_diagnostics.get("intra_batch_dedup", 0) == 0

    @pytest.mark.asyncio
    async def test_intra_batch_dedup_multiple_duplicates(
        self, db, user_model_store
    ):
        """Three copies of the same prediction → only one kept, dedup count=2."""
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        desc = "Triplicate reminder"

        async def mock_three_copies(context):
            """Return three identical predictions."""
            return [
                _make_prediction(description=desc, prediction_type="reminder"),
                _make_prediction(description=desc, prediction_type="reminder"),
                _make_prediction(description=desc, prediction_type="reminder"),
            ]

        async def mock_empty(context):
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_three_copies),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        with db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE description = ?", (desc,)
            ).fetchone()[0]
        assert count == 1, f"Expected 1 row after removing 2 duplicates, got {count}"
        assert engine._last_run_diagnostics.get("intra_batch_dedup") == 2, (
            f"Expected intra_batch_dedup=2, "
            f"got {engine._last_run_diagnostics.get('intra_batch_dedup')!r}"
        )

    @pytest.mark.asyncio
    async def test_intra_batch_dedup_logged_at_debug(
        self, db, user_model_store, caplog
    ):
        """Intra-batch dedup count should be logged at DEBUG level."""
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        desc = "Logged dedup prediction"

        async def mock_two_copies(context):
            return [
                _make_prediction(description=desc),
                _make_prediction(description=desc),
            ]

        async def mock_empty(context):
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_two_copies),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
                await engine.generate_predictions({})

        assert "Intra-batch dedup" in caplog.text, (
            "Expected 'Intra-batch dedup' in debug log output"
        )
        assert "1" in caplog.text  # "removed 1 duplicate predictions"


# ---------------------------------------------------------------------------
# Pre-filter cache persistence tests
# ---------------------------------------------------------------------------


class TestPrefilterCachePersistence:
    """Verify that the pre-filter cache avoids redundant DB queries across cycles."""

    def test_prefilter_cache_initially_empty(self, db, user_model_store):
        """A fresh PredictionEngine should have an empty pre-filter cache."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        assert engine._prefilter_cache == set(), "Cache should start empty"
        assert engine._prefilter_refresh_cycle == 0, "Cycle counter should start at 0"

    @pytest.mark.asyncio
    async def test_prefilter_cache_populated_after_first_run(
        self, db, user_model_store
    ):
        """After the first generate_predictions() run, _prefilter_cache should contain
        keys for any predictions stored during that cycle.

        This ensures the second cycle can skip re-loading from DB for those entries.
        """
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        desc = "Cache population test prediction"

        async def mock_one_pred(context):
            return [_make_prediction(description=desc, prediction_type="reminder")]

        async def mock_empty(context):
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_one_pred),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        # The cache should now contain the stored prediction's key
        cache_key = ("reminder", desc, "this_week")
        assert cache_key in engine._prefilter_cache, (
            f"Expected key {cache_key!r} in _prefilter_cache after first run. "
            f"Cache: {engine._prefilter_cache}"
        )
        # Cycle counter should have advanced past 0
        assert engine._prefilter_refresh_cycle > 0

    @pytest.mark.asyncio
    async def test_prefilter_cache_used_on_second_run(
        self, db, user_model_store
    ):
        """After first run, the second cycle uses the in-memory cache (no DB query).

        We verify this by manually injecting a key into _prefilter_cache that
        does NOT exist in the DB, then generating the same prediction on the second
        run.  If the cache is used, the prediction is pre-filtered; if a DB query
        was used, it would NOT be in existing_predictions and would be generated.
        """
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        desc = "Cache-only prediction — never in DB"
        cache_only_key = ("opportunity", desc, "this_week")

        # Run first cycle to initialize (force DB refresh on cycle 0)
        async def mock_empty(context):
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_empty),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        # Manually inject the key into the in-memory cache (not present in DB)
        engine._prefilter_cache.add(cache_only_key)
        # Confirm it's not in the DB
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT id FROM predictions WHERE description = ?", (desc,)
            ).fetchone()
        assert row is None, "Prediction must not exist in DB for this test to be valid"

        # Second cycle: returns the "cache-only" prediction
        async def mock_returns_cached_pred(context):
            return [_make_prediction(description=desc, prediction_type="opportunity")]

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_returns_cached_pred),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        # The prediction should NOT have been stored because the cache said it exists
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT id FROM predictions WHERE description = ?", (desc,)
            ).fetchone()
        assert row is None, (
            "Prediction was stored even though it was in the in-memory cache. "
            "The second cycle should use _prefilter_cache, not query the DB."
        )

    @pytest.mark.asyncio
    async def test_prefilter_cache_db_refresh_on_fourth_cycle(
        self, db, user_model_store
    ):
        """Cache is rebuilt from DB when _prefilter_refresh_cycle % 4 == 0.

        We simulate being at cycle 4 (would normally use cache) by directly setting
        _prefilter_refresh_cycle = 4, then inserting a prediction into the DB that
        the cache doesn't know about.  After the run, the cache should reflect the
        DB state (picking up the new entry from the DB refresh).
        """
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        # Insert a prediction directly into the DB (simulating one from a previous cycle)
        existing_desc = "Pre-existing prediction from a previous cycle"
        pred_id = str(uuid.uuid4())
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    time_horizon, was_surfaced)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (pred_id, "reminder", existing_desc, 0.5, "suggest", "24_hours", 0),
            )

        # Manually set engine state to cycle 4 (next run will trigger periodic refresh)
        # and clear the cache so it doesn't already contain the DB entry
        engine._prefilter_cache = set()
        engine._prefilter_refresh_cycle = 4
        engine._last_time_based_run = None  # Ensure time-based trigger fires

        async def mock_empty(context):
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_empty),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        # After the periodic refresh on cycle 4, the cache should contain the
        # DB prediction's key (loaded from the DB query)
        db_pred_key = ("reminder", existing_desc, "24_hours")
        assert db_pred_key in engine._prefilter_cache, (
            f"Expected cache to contain DB prediction key {db_pred_key!r} "
            f"after periodic refresh on cycle 4. Cache: {engine._prefilter_cache}"
        )

    @pytest.mark.asyncio
    async def test_prefilter_cache_reset_on_persistence_failure_recovery(
        self, db, user_model_store
    ):
        """When persistence failure recovery runs, the pre-filter cache is rebuilt.

        If _persistence_failure_detected is True at entry, the recovery block:
        1. Runs a DB write test
        2. On success, clears _persistence_failure_detected AND resets the cache
           (_prefilter_cache = set(), _prefilter_refresh_cycle = 0)
        3. The pre-filter block then does a forced DB refresh (cycle == 0)

        We verify that stale cache entries (not in DB) are evicted after recovery.
        """
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        # Inject a stale key into the cache (not in DB)
        stale_key = ("opportunity", "Stale prediction not in DB", "this_week")
        engine._prefilter_cache = {stale_key}
        engine._prefilter_refresh_cycle = 2  # Would normally use cache (not refresh)

        # Trigger persistence failure recovery path
        engine._persistence_failure_detected = True
        engine._last_time_based_run = None  # Ensure time-based trigger fires

        async def mock_empty(context):
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_empty),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        # The stale key should no longer be in the cache (rebuilt from DB)
        assert stale_key not in engine._prefilter_cache, (
            f"Stale key {stale_key!r} should have been evicted when cache was "
            f"rebuilt after persistence failure recovery. Cache: {engine._prefilter_cache}"
        )
        # Failure flag should be cleared (DB write test passed)
        assert engine._persistence_failure_detected is False


# ---------------------------------------------------------------------------
# Diagnostics integration
# ---------------------------------------------------------------------------


class TestPrefilterDiagnostics:
    """Verify that pre-filter cache metrics appear in _last_run_diagnostics."""

    @pytest.mark.asyncio
    async def test_diagnostics_include_prefilter_cache_size(
        self, db, user_model_store
    ):
        """_last_run_diagnostics should expose prefilter_cache_size and
        prefilter_refresh_cycle for monitoring."""
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        async def mock_empty(context):
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_empty),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        diag = engine._last_run_diagnostics
        assert "prefilter_cache_size" in diag, (
            f"Expected 'prefilter_cache_size' in diagnostics. Keys: {list(diag)}"
        )
        assert "prefilter_refresh_cycle" in diag, (
            f"Expected 'prefilter_refresh_cycle' in diagnostics. Keys: {list(diag)}"
        )
        assert isinstance(diag["prefilter_cache_size"], int)
        assert isinstance(diag["prefilter_refresh_cycle"], int)

    @pytest.mark.asyncio
    async def test_diagnostics_intra_batch_dedup_zero_when_no_duplicates(
        self, db, user_model_store
    ):
        """When no intra-batch duplicates occur, intra_batch_dedup should be 0."""
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        async def mock_empty(context):
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_empty),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        assert engine._last_run_diagnostics.get("intra_batch_dedup", 0) == 0

    @pytest.mark.asyncio
    async def test_diagnostics_intra_batch_dedup_present_when_duplicates_exist(
        self, db, user_model_store
    ):
        """When intra-batch duplicates are removed, the count is in diagnostics."""
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        desc = "Duplicate prediction for diagnostics test"

        async def mock_two_copies(context):
            return [
                _make_prediction(description=desc),
                _make_prediction(description=desc),
            ]

        async def mock_empty(context):
            return []

        with (
            patch.object(engine, "_check_calendar_conflicts", new=mock_two_copies),
            patch.object(engine, "_check_routine_deviations", new=mock_empty),
            patch.object(engine, "_check_relationship_maintenance", new=mock_empty),
            patch.object(engine, "_check_preparation_needs", new=mock_empty),
            patch.object(engine, "_check_calendar_event_reminders", new=mock_empty),
            patch.object(engine, "_check_connector_health", new=mock_empty),
            patch.object(engine, "_check_follow_up_needs", new=mock_empty),
            patch.object(engine, "_check_spending_patterns", new=mock_empty),
        ):
            await engine.generate_predictions({})

        assert engine._last_run_diagnostics["intra_batch_dedup"] == 1


# ---------------------------------------------------------------------------
# reset_state() coverage for new fields
# ---------------------------------------------------------------------------


class TestResetStatePrefilterFields:
    """Verify reset_state() clears pre-filter cache fields."""

    def test_reset_clears_prefilter_cache(self, db, user_model_store):
        """reset_state() should empty _prefilter_cache and reset the cycle counter."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        # Populate cache and advance cycle counter
        engine._prefilter_cache = {("need", "Some old prediction", "24_hours")}
        engine._prefilter_refresh_cycle = 7

        engine.reset_state()

        assert engine._prefilter_cache == set(), (
            "reset_state() should clear _prefilter_cache"
        )
        assert engine._prefilter_refresh_cycle == 0, (
            "reset_state() should reset _prefilter_refresh_cycle to 0"
        )
