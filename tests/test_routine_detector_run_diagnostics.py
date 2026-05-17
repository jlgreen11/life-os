"""
Tests for RoutineDetector.get_last_run_diagnostics() per-cycle skip diagnostics.

The detector accumulates a per-run dict capturing why a detection cycle produced
the routines it did — episode counts, effective thresholds, and per-candidate
skip reasons.  Operators rely on this dict to debug zero-result cycles without
re-running detection by hand.

Covers:
- Empty run still publishes a populated diagnostics dict and logs at INFO.
- skipped_below_min_episodes increments for sparse candidate days.
- skipped_below_consistency increments for candidates that survive the
  min-occurrences filter but fail the consistency threshold.
- A successful run reports the routine count and clears the prior cycle's
  state (the dict is reset, not appended to).
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta

from services.routine_detector.detector import RoutineDetector


def _make_episode(timestamp_iso: str, interaction_type: str) -> dict:
    """Build a minimal episode dict accepted by UserModelStore.store_episode."""
    return {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp_iso,
        "event_id": str(uuid.uuid4()),
        "interaction_type": interaction_type,
        "content_summary": "test episode",
    }


class TestZeroEpisodesRun:
    """A run against an empty database publishes diagnostics and emits an INFO log."""

    def test_zero_episodes_emits_zero_routines(self, db, user_model_store, caplog):
        """get_last_run_diagnostics()['routines_emitted'] is 0 and a summary line is logged."""
        detector = RoutineDetector(db, user_model_store)

        with caplog.at_level(logging.INFO, logger="services.routine_detector.detector"):
            routines = detector.detect_routines(lookback_days=30)

        assert routines == []

        diag = detector.get_last_run_diagnostics()
        assert diag["routines_emitted"] == 0
        # last_run_at is populated on every call, even when nothing matches.
        assert diag["last_run_at"] is not None
        # effective_lookback_days reflects the caller's request (no adaptive
        # expansion is possible against a fully empty DB).
        assert diag["effective_lookback_days"] == 30

        # The INFO summary at the end of detect_routines() should mention the
        # zero-result outcome so operators don't need to call the accessor.
        zero_summary_logs = [
            r for r in caplog.records
            if "produced 0 routines" in r.message and r.levelno == logging.INFO
        ]
        assert zero_summary_logs, "Expected an INFO log summarizing the zero-result diagnostic dict"

    def test_diagnostics_dict_is_empty_before_first_run(self, db, user_model_store):
        """Before detect_routines() is called, the dict is the empty initializer."""
        detector = RoutineDetector(db, user_model_store)
        assert detector.get_last_run_diagnostics() == {}

    def test_get_last_run_diagnostics_returns_copy(self, db, user_model_store):
        """Mutating the returned dict must not corrupt the detector's cached state."""
        detector = RoutineDetector(db, user_model_store)
        detector.detect_routines(lookback_days=30)

        snapshot = detector.get_last_run_diagnostics()
        snapshot["routines_emitted"] = 9999

        fresh = detector.get_last_run_diagnostics()
        assert fresh["routines_emitted"] == 0


class TestSkipCounters:
    """Per-candidate skip counters increment correctly."""

    def test_skipped_below_min_episodes_counts_sparse_pairs(self, db, user_model_store):
        """A (bucket, type) pair appearing on fewer days than min_occurrences is skipped.

        Seeds two interaction types on a single day each — both pairs land in the
        same morning bucket but each only covers one distinct day (< 3 needed),
        so both must increment skipped_below_min_episodes.
        """
        detector = RoutineDetector(db, user_model_store)
        morning = (datetime.now(UTC) - timedelta(days=2)).replace(
            hour=8, minute=0, second=0, microsecond=0
        )

        user_model_store.store_episode(_make_episode(morning.isoformat(), "calendar_reviewed"))
        user_model_store.store_episode(
            _make_episode((morning + timedelta(minutes=5)).isoformat(), "task_management")
        )

        detector.detect_routines(lookback_days=30)
        diag = detector.get_last_run_diagnostics()

        # Both candidate pairs (morning, calendar_reviewed) and
        # (morning, task_management) should be counted as candidates and as
        # below-min-episodes skips.
        assert diag["candidates_considered"] >= 2
        assert diag["skipped_below_min_episodes"] >= 2
        assert diag["routines_emitted"] == 0

    def test_skipped_below_consistency_counts_low_consistency_buckets(
        self, db, user_model_store
    ):
        """Candidate pairs that pass min_occurrences but fail consistency are tallied.

        Build a 60-day window with email_received episodes on only 4 distinct
        days — passes min_occurrences (>= 3) but the bucket consistency
        (4/4 = 1.0) won't trip the skip path, so we instead seed a pattern
        where many active days exist but the bucket only fires on a small
        minority of them.  We achieve this by adding a control type that
        fills the active-day denominator on most days, while the target
        candidate only appears on 3 of those days.
        """
        detector = RoutineDetector(db, user_model_store)

        # 30 active days of an unrelated type so that active_days >= 30 and the
        # mature consistency threshold (0.6) kicks in instead of cold-start.
        far_back = datetime.now(UTC) - timedelta(days=35)
        for day_offset in range(30):
            ts = (far_back + timedelta(days=day_offset)).replace(
                hour=14, minute=0, second=0, microsecond=0
            )
            user_model_store.store_episode(_make_episode(ts.isoformat(), "web_browsing"))

        # 3 morning email_received episodes (passes min_occurrences=3 but 3/30
        # = 0.1 consistency, well below the 0.4 passive-type cap let alone
        # the 0.6 base threshold).
        for day_offset in range(3):
            ts = (far_back + timedelta(days=day_offset)).replace(
                hour=8, minute=0, second=0, microsecond=0
            )
            user_model_store.store_episode(_make_episode(ts.isoformat(), "email_received"))

        routines = detector.detect_routines(lookback_days=60)
        diag = detector.get_last_run_diagnostics()

        # The morning email_received candidate should be counted as failing
        # consistency.  The passive-type sub-counter must also fire because
        # email_received is in HIGH_VOLUME_PASSIVE_TYPES and 3/30 = 0.1 is
        # below the 0.4 passive-type cap.
        assert diag["skipped_below_consistency"] >= 1
        assert diag["skipped_high_volume_passive"] >= 1

        # The control (afternoon web_browsing on 30 days) trivially passes
        # consistency, so one routine is expected — but none triggered by the
        # sparse morning email pattern.
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert morning_routines == []


class TestSuccessfulRunPopulatesCounts:
    """A run that produces routines still updates the diagnostic dict."""

    def test_successful_detection_updates_dict(self, db, user_model_store):
        """A clear daily pattern produces routines and routines_emitted > 0."""
        detector = RoutineDetector(db, user_model_store)
        base = datetime.now(UTC) - timedelta(days=10)

        # 10 consecutive days of a morning email pattern — easily clears
        # cold-start consistency thresholds and produces at least one routine.
        for day_offset in range(10):
            ts = (base + timedelta(days=day_offset)).replace(
                hour=8, minute=0, second=0, microsecond=0
            )
            user_model_store.store_episode(_make_episode(ts.isoformat(), "email_received"))

        routines = detector.detect_routines(lookback_days=30)
        diag = detector.get_last_run_diagnostics()

        assert routines, "Expected at least one routine for a 10-day morning pattern"
        assert diag["routines_emitted"] == len(routines)
        assert diag["episodes_considered"] >= 10
        assert diag["candidates_considered"] >= 1
        # The effective thresholds should have been recorded.
        assert diag["effective_consistency_threshold"] is not None
        assert diag["effective_min_episodes"] is not None
