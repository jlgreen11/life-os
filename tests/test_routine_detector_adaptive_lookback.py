"""
Tests for the RoutineDetector adaptive lookback feature.

When a connector outage (or any gap in data ingestion) pushes all recent
episodes outside the default 30-day window, `detect_routines()` used to
return 0 results despite hundreds of historical episodes.

The adaptive lookback logic detects this scenario by:
1. Counting episodes in the default window.
2. If the count is below `min_episodes_for_detection`, querying the oldest
   timestamp among the 200 most recent episodes.
3. Extending the window to cover those episodes (+ 1-day buffer, capped at 180).

These tests validate that the extension fires when it should, stays quiet when
it shouldn't, and that routines are actually detected once the window is extended.

Test patterns mirror `tests/test_routine_detector.py`:
- Real DatabaseManager + UserModelStore via conftest fixtures (no DB mocks).
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.routine_detector.detector import RoutineDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _make_episode(
    user_model_store,
    interaction_type: str,
    timestamp: datetime,
    content_summary: str = "Test episode",
) -> None:
    """Insert a single episode into the user-model store."""
    user_model_store.store_episode(
        {
            "id": str(uuid.uuid4()),
            "timestamp": timestamp.isoformat(),
            "event_id": str(uuid.uuid4()),
            "interaction_type": interaction_type,
            "content_summary": content_summary,
        }
    )


def _insert_episodes_at_offset(
    user_model_store,
    *,
    days_ago: float,
    interaction_type: str,
    num_unique_days: int,
    episodes_per_day: int = 3,
    hour: int = 9,
) -> None:
    """Insert a block of episodes centred around `days_ago` days before now.

    Episodes are spread across `num_unique_days` consecutive days so that
    consistency calculations have meaningful denominators.

    Args:
        user_model_store: Store fixture from conftest.
        days_ago: How many days before now the *first* batch starts.
        interaction_type: Episode interaction_type to assign.
        num_unique_days: How many distinct calendar days to insert episodes on.
        episodes_per_day: How many episodes per day (default 3).
        hour: Hour of day for all inserted episodes (default 9 = morning).
    """
    base = datetime.now(UTC) - timedelta(days=days_ago)
    base = base.replace(hour=hour, minute=0, second=0, microsecond=0)
    for day_offset in range(num_unique_days):
        for ep_idx in range(episodes_per_day):
            ts = base + timedelta(days=day_offset, minutes=ep_idx * 5)
            _make_episode(user_model_store, interaction_type, ts)


# ---------------------------------------------------------------------------
# Unit tests for _compute_adaptive_lookback_days
# ---------------------------------------------------------------------------


class TestComputeAdaptiveLookbackDays:
    """Unit-level tests for the adaptive lookback computation helper."""

    def test_no_extension_when_episodes_within_window(self, db, user_model_store):
        """When the default window has enough episodes, no extension is applied."""
        detector = RoutineDetector(db, user_model_store)

        # Insert 60 episodes within the last 15 days (well inside a 30-day window).
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=15,
            interaction_type="email_check",
            num_unique_days=10,
            episodes_per_day=6,
        )

        effective = detector._compute_adaptive_lookback_days(30)
        assert effective == 30, (
            "Lookback should stay at 30 when enough episodes exist in the default window"
        )

    def test_extension_when_all_episodes_outside_window(self, db, user_model_store):
        """When all episodes are older than the default window, lookback is extended."""
        detector = RoutineDetector(db, user_model_store)

        # Insert 60 episodes at 40 days ago (outside a 30-day window).
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=40,
            interaction_type="email_check",
            num_unique_days=10,
            episodes_per_day=6,
        )

        effective = detector._compute_adaptive_lookback_days(30)
        assert effective > 30, "Lookback should extend beyond 30 when all episodes are older"
        assert effective >= 41, "Lookback should be at least 41 days to cover episodes at day 40"
        assert effective <= 180, "Lookback should never exceed the 180-day cap"

    def test_no_extension_when_db_is_empty(self, db, user_model_store):
        """With no episodes in the database, no extension is applied (nothing to cover)."""
        detector = RoutineDetector(db, user_model_store)

        effective = detector._compute_adaptive_lookback_days(30)
        assert effective == 30, (
            "Lookback should stay at 30 when the database has no episodes at all"
        )

    def test_extension_capped_at_180_days(self, db, user_model_store):
        """Even if episodes are 500 days old, the lookback caps at 180 days."""
        detector = RoutineDetector(db, user_model_store)

        # Insert a handful of episodes 300 days ago.
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=300,
            interaction_type="email_check",
            num_unique_days=5,
            episodes_per_day=4,
        )

        effective = detector._compute_adaptive_lookback_days(30)
        assert effective == 180, "Lookback must be capped at 180 days regardless of episode age"

    def test_extension_not_triggered_when_any_episodes_in_window(self, db, user_model_store):
        """Even a single episode in the default window prevents extension.

        The adaptive logic only fires when the window is completely empty (count == 0).
        This preserves the lookback boundary guarantee: callers that pass lookback_days=30
        and have even one episode within that window should not see episodes from day 40.
        """
        detector = RoutineDetector(db, user_model_store)

        # Insert just 3 recent episodes — far below min_episodes_for_detection (50),
        # but count > 0 so no extension should fire.
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=10,
            interaction_type="email_check",
            num_unique_days=1,
            episodes_per_day=3,
        )
        # Also insert old episodes to make sure they'd be found if extension fired
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=50,
            interaction_type="old_email_check",
            num_unique_days=10,
            episodes_per_day=6,
        )

        effective = detector._compute_adaptive_lookback_days(30)
        assert effective == 30, (
            "Lookback should stay at 30 when there is at least one episode in the default window"
        )

    def test_extension_triggered_when_window_is_completely_empty(self, db, user_model_store):
        """When the default window has 0 episodes (connector outage), extension fires."""
        detector = RoutineDetector(db, user_model_store)

        # Insert ONLY old episodes — simulates a connector outage that stopped
        # feeding new data, so all existing episodes are older than 30 days.
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=50,
            interaction_type="email_received",
            num_unique_days=10,
            episodes_per_day=6,
        )

        effective = detector._compute_adaptive_lookback_days(30)
        assert effective > 30, (
            "Lookback should extend when the default window has 0 episodes "
            "(complete connector outage scenario)"
        )


# ---------------------------------------------------------------------------
# Integration tests: detect_routines() with adaptive lookback
# ---------------------------------------------------------------------------


class TestDetectRoutinesAdaptiveLookback:
    """Integration tests verifying that detect_routines() uses adaptive lookback correctly."""

    def test_routines_detected_when_episodes_outside_default_window(self, db, user_model_store):
        """The primary regression test: routines ARE detected even when all episodes are
        older than the default 30-day window.

        Scenario mirrors the Google connector outage: 50+ email_received episodes
        spread across 10 unique days, all created 35 days ago.  Without adaptive
        lookback, detect_routines(30) would return [].  With it, the window extends
        and the episodes are included in detection.
        """
        detector = RoutineDetector(db, user_model_store)

        # Insert 60 episodes (> min_episodes_for_detection=50) across 10 unique days
        # centred at 35 days ago — just outside the default 30-day window.
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=35,
            interaction_type="email_received",
            num_unique_days=10,
            episodes_per_day=6,
            hour=9,  # morning bucket
        )

        routines = detector.detect_routines(lookback_days=30)

        assert len(routines) >= 1, (
            "detect_routines(30) should detect routines from episodes at day 35 "
            "after adaptive lookback extends the window"
        )

    def test_no_false_extension_when_episodes_in_window(self, db, user_model_store):
        """When the default window has enough episodes, no spurious extension occurs
        and routines are still detected correctly."""
        detector = RoutineDetector(db, user_model_store)

        # Insert 60 episodes within the last 10 days (clearly inside 30-day window).
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=10,
            interaction_type="calendar_review",
            num_unique_days=10,
            episodes_per_day=6,
            hour=8,
        )

        routines = detector.detect_routines(lookback_days=30)

        # Routines should be detected normally — the adaptive logic should not
        # interfere when the default window already has enough data.
        assert len(routines) >= 1, (
            "Routines within the default window should still be detected normally"
        )

    def test_empty_database_returns_empty_list(self, db, user_model_store):
        """With no episodes at all, detect_routines() returns [] without error."""
        detector = RoutineDetector(db, user_model_store)
        routines = detector.detect_routines(lookback_days=30)
        assert routines == [], "Empty database should produce 0 routines"

    def test_episodes_across_multiple_buckets_all_detected(self, db, user_model_store):
        """All time-of-day buckets are detected when adaptive lookback extends the window.

        Morning episodes (9am) and evening episodes (6pm) are both placed at 40 days
        ago.  After extension, both buckets should produce routines.
        """
        detector = RoutineDetector(db, user_model_store)

        # Morning routine at 40 days ago
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=40,
            interaction_type="morning_email",
            num_unique_days=10,
            episodes_per_day=6,
            hour=9,
        )
        # Evening routine at 40 days ago
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=40,
            interaction_type="evening_review",
            num_unique_days=10,
            episodes_per_day=6,
            hour=18,
        )

        routines = detector.detect_routines(lookback_days=30)

        triggers = {r["trigger"] for r in routines}
        assert "morning" in triggers, "Morning routine should be detected after lookback extension"
        assert "evening" in triggers, "Evening routine should be detected after lookback extension"

    def test_adaptive_extension_logged(self, db, user_model_store, caplog):
        """The adaptive lookback extension should emit an INFO log message."""
        import logging

        detector = RoutineDetector(db, user_model_store)

        # Insert episodes outside the 30-day default window
        _insert_episodes_at_offset(
            user_model_store,
            days_ago=50,
            interaction_type="browser_visit",
            num_unique_days=10,
            episodes_per_day=6,
        )

        with caplog.at_level(logging.INFO, logger="services.routine_detector.detector"):
            detector.detect_routines(lookback_days=30)

        # The log message from _compute_adaptive_lookback_days should appear
        assert any(
            "Adaptive lookback: extended from" in record.message
            for record in caplog.records
        ), "Expected an INFO log about adaptive lookback extension"

    def test_no_extension_log_when_episodes_in_window(self, db, user_model_store, caplog):
        """No adaptive-extension log should appear when the default window suffices."""
        import logging

        detector = RoutineDetector(db, user_model_store)

        _insert_episodes_at_offset(
            user_model_store,
            days_ago=5,
            interaction_type="task_check",
            num_unique_days=10,
            episodes_per_day=6,
        )

        with caplog.at_level(logging.INFO, logger="services.routine_detector.detector"):
            detector.detect_routines(lookback_days=30)

        assert not any(
            "Adaptive lookback: extended from" in record.message
            for record in caplog.records
        ), "Should NOT emit an adaptive-extension log when the default window has enough data"

    def test_db_error_during_adaptive_check_falls_back_gracefully(self, db, user_model_store, monkeypatch):
        """If the adaptive lookback query raises, detect_routines() continues with
        the default window (fail-open behaviour).

        We monkeypatch `_compute_adaptive_lookback_days` to raise an exception so
        the real DB is unaffected.  The fallback is tested by verifying that
        detect_routines() completes and returns a list (empty or otherwise)
        rather than propagating the exception.
        """
        detector = RoutineDetector(db, user_model_store)

        # Force the adaptive helper to raise an exception
        def _raise(*_args, **_kwargs):
            raise RuntimeError("Simulated DB error during adaptive lookback")

        monkeypatch.setattr(detector, "_compute_adaptive_lookback_days", _raise)

        # detect_routines() should NOT propagate — the exception is caught within
        # the helper itself and defaults to lookback_days.  But since we patched
        # the helper to raise unconditionally (bypassing the try/except inside it),
        # we should see the outer detect_routines() catch it and continue.
        # This tests that detect_routines() is not broken by a failing helper.
        # Note: the actual _compute_adaptive_lookback_days wraps everything in
        # try/except, so an error inside it returns lookback_days.  Monkeypatching
        # bypasses that wrapper.  We verify the broader detect_routines() is safe.
        result = detector.detect_routines(lookback_days=30)
        assert isinstance(result, list), (
            "detect_routines() should always return a list, even when adaptive check fails"
        )

    def test_minimum_50_episodes_across_5_days_triggers_detection(self, db, user_model_store):
        """Full end-to-end scenario: 50+ episodes at 35 days ago, across 5+ unique dates.

        This is the exact scenario described in the task rationale:
        - Default 30-day window → 0 episodes → no routines
        - Adaptive lookback extends to 36 days → 50+ episodes found → routines detected
        """
        detector = RoutineDetector(db, user_model_store)

        # Create 60 episodes (> 50) across exactly 10 unique days.
        # Use days_ago=40 so all 10 days land between 40–31 days ago —
        # all strictly outside the 30-day default window.
        # Uses 6 distinct interaction types to create several candidate routines.
        interaction_types = [
            "email_received",
            "calendar_event",
            "task_created",
            "message_received",
            "browser_visit",
            "notification_check",
        ]
        base = datetime.now(UTC) - timedelta(days=40)
        base = base.replace(hour=8, minute=0, second=0, microsecond=0)

        for day_offset in range(10):
            for ep_idx, itype in enumerate(interaction_types):
                ts = base + timedelta(days=day_offset, minutes=ep_idx * 10)
                _make_episode(user_model_store, itype, ts)

        # Verify the setup: 60 episodes exist, none within 30 days
        from datetime import UTC as DT_UTC

        now = datetime.now(timezone.utc)
        thirty_day_cutoff = now - timedelta(days=30)

        # All episodes should be outside the 30-day window
        with db.get_connection("user_model") as conn:
            recent_count = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE timestamp > ?",
                (thirty_day_cutoff.isoformat(),),
            ).fetchone()[0]
            total_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

        assert recent_count == 0, f"Expected 0 episodes in 30-day window, got {recent_count}"
        assert total_count == 60, f"Expected 60 total episodes, got {total_count}"

        # Now run detection — adaptive lookback should extend to cover day 35
        routines = detector.detect_routines(lookback_days=30)

        assert len(routines) >= 1, (
            f"Expected at least 1 routine from 60 episodes at day 35, got 0. "
            f"Adaptive lookback may not have fired or detection thresholds not met."
        )
