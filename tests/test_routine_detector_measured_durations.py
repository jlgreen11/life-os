"""
Tests for RoutineDetector step-duration measurement improvements.

Verifies that _detect_location_routines() and _detect_event_triggered_routines()
use actual observed inter-episode gaps instead of the old hardcoded 5.0-minute
placeholder, and that the shared _compute_step_duration_map() helper returns
correct values.

Before this fix, all location and event-triggered routine steps reported
``typical_duration_minutes: 5.0`` regardless of the real cadence.  After the
fix, durations reflect the average measured gap to the next episode on the same
day, giving accurate timing data to the UI and AI briefings.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.routine_detector.detector import RoutineDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _episode(user_model_store, interaction_type: str, ts: datetime, location: str | None = None) -> None:
    """Insert a minimal episode row into the user-model database."""
    user_model_store.store_episode({
        "id": str(uuid.uuid4()),
        "timestamp": ts.isoformat(),
        "event_id": str(uuid.uuid4()),
        "interaction_type": interaction_type,
        "content_summary": interaction_type.replace("_", " ").title(),
        "location": location,
    })


# ---------------------------------------------------------------------------
# _compute_step_duration_map
# ---------------------------------------------------------------------------

class TestComputeStepDurationMap:
    """Unit tests for the shared duration-measurement helper."""

    def test_returns_empty_map_when_no_episodes(self, db, user_model_store):
        """Should return an empty dict when the episodes table is empty."""
        detector = RoutineDetector(db, user_model_store)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        result = detector._compute_step_duration_map(cutoff)
        assert result == {}

    def test_measures_gap_between_two_actions(self, db, user_model_store):
        """Average gap should equal the observed gap when there is exactly one pair."""
        detector = RoutineDetector(db, user_model_store)
        base = datetime.now(timezone.utc) - timedelta(days=2)
        day = base.replace(hour=9, minute=0, second=0, microsecond=0)

        # check_email at 9:00, review_calendar at 9:20 → 20-min gap for check_email
        _episode(user_model_store, "check_email", day)
        _episode(user_model_store, "review_calendar", day + timedelta(minutes=20))

        cutoff = base - timedelta(days=1)
        result = detector._compute_step_duration_map(cutoff)

        assert "check_email" in result
        # Allow ±1 minute for floating-point JULIANDAY arithmetic
        assert abs(result["check_email"] - 20.0) < 1.0

    def test_averages_gaps_across_multiple_days(self, db, user_model_store):
        """Duration should be the mean gap across all observed days."""
        detector = RoutineDetector(db, user_model_store)
        base = datetime.now(timezone.utc) - timedelta(days=10)

        # Day 1: gap = 10 min; Day 2: gap = 30 min → average = 20 min
        for day_offset, gap_minutes in enumerate([10, 30]):
            day = (base + timedelta(days=day_offset)).replace(hour=8, minute=0, second=0, microsecond=0)
            _episode(user_model_store, "read_news", day)
            _episode(user_model_store, "make_coffee", day + timedelta(minutes=gap_minutes))

        cutoff = base - timedelta(days=1)
        result = detector._compute_step_duration_map(cutoff)

        assert "read_news" in result
        assert abs(result["read_news"] - 20.0) < 1.0

    def test_excludes_episodes_before_cutoff(self, db, user_model_store):
        """Episodes older than the cutoff must not influence the result."""
        detector = RoutineDetector(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Old episode pair (outside window)
        old_day = now - timedelta(days=60)
        _episode(user_model_store, "old_action", old_day)
        _episode(user_model_store, "old_follow", old_day + timedelta(minutes=99))

        cutoff = now - timedelta(days=30)
        result = detector._compute_step_duration_map(cutoff)

        assert "old_action" not in result

    def test_types_with_no_successor_absent_from_map(self, db, user_model_store):
        """An interaction type that is always the last episode of the day has no gap."""
        detector = RoutineDetector(db, user_model_store)
        base = datetime.now(timezone.utc) - timedelta(days=5)

        # Only one episode per day — no successor to compute a gap against
        for i in range(3):
            day = (base + timedelta(days=i)).replace(hour=23, minute=0, second=0, microsecond=0)
            _episode(user_model_store, "end_of_day_review", day)

        cutoff = base - timedelta(days=1)
        result = detector._compute_step_duration_map(cutoff)

        assert "end_of_day_review" not in result


# ---------------------------------------------------------------------------
# _detect_location_routines — measured durations
# ---------------------------------------------------------------------------

class TestLocationRoutineMeasuredDurations:
    """Verify that location routines use measured, not hardcoded, step durations."""

    def _build_location_pattern(self, user_model_store, location: str, days: int = 7,
                                 gap_minutes: float = 25.0) -> None:
        """Insert <days> days of a two-step location routine with a fixed gap."""
        base = datetime.now(timezone.utc) - timedelta(days=days)
        for day_offset in range(days):
            day = (base + timedelta(days=day_offset)).replace(
                hour=18, minute=0, second=0, microsecond=0
            )
            _episode(user_model_store, "arrive_home", day, location=location)
            _episode(user_model_store, "check_mail", day + timedelta(minutes=gap_minutes), location=location)

    def test_step_duration_reflects_measured_gap(self, db, user_model_store):
        """Step typical_duration_minutes should match the observed inter-episode gap."""
        detector = RoutineDetector(db, user_model_store)
        self._build_location_pattern(user_model_store, location="Home", gap_minutes=20.0)

        routines = detector.detect_routines(lookback_days=30)
        location_routines = [r for r in routines if "Home" in r.get("name", "")]

        assert location_routines, "Expected at least one Home location routine"
        routine = location_routines[0]

        # Verify the arrive_home step does NOT use the old 5.0-minute placeholder.
        arrive_step = next(
            (s for s in routine["steps"] if s["action"] == "arrive_home"), None
        )
        assert arrive_step is not None
        # The measured gap from arrive_home to check_mail is 20 min; must not be 5.0.
        assert arrive_step["typical_duration_minutes"] != 5.0, (
            "Step duration should be measured (~20 min), not the old hardcoded 5.0"
        )
        assert abs(arrive_step["typical_duration_minutes"] - 20.0) < 2.0, (
            f"Expected ~20.0 min, got {arrive_step['typical_duration_minutes']}"
        )

    def test_total_routine_duration_reflects_measured_steps(self, db, user_model_store):
        """The routine's total duration should sum the measured step durations."""
        detector = RoutineDetector(db, user_model_store)
        self._build_location_pattern(user_model_store, location="Office", gap_minutes=15.0)

        routines = detector.detect_routines(lookback_days=30)
        location_routines = [r for r in routines if "Office" in r.get("name", "")]

        assert location_routines
        routine = location_routines[0]

        # total_duration should equal the sum of all step durations
        step_sum = sum(s["typical_duration_minutes"] for s in routine["steps"])
        assert abs(routine["typical_duration_minutes"] - step_sum) < 0.01


# ---------------------------------------------------------------------------
# _detect_event_triggered_routines — measured durations
# ---------------------------------------------------------------------------

class TestEventTriggeredRoutineMeasuredDurations:
    """Verify that event-triggered routines use measured, not hardcoded, step durations."""

    def _build_event_trigger_pattern(self, user_model_store, trigger: str, days: int = 8,
                                      gap1_min: float = 12.0, gap2_min: float = 25.0) -> None:
        """Insert <days> days of a 3-step event-triggered routine with known gaps."""
        base = datetime.now(timezone.utc) - timedelta(days=days)
        for day_offset in range(days):
            day = (base + timedelta(days=day_offset)).replace(
                hour=10, minute=0, second=0, microsecond=0
            )
            # Trigger
            _episode(user_model_store, trigger, day)
            # Step 1 follows at gap1_min
            _episode(user_model_store, "write_notes", day + timedelta(minutes=gap1_min))
            # Step 2 follows at gap1_min + gap2_min
            _episode(user_model_store, "send_followup", day + timedelta(minutes=gap1_min + gap2_min))

    def test_step_duration_reflects_measured_gap(self, db, user_model_store):
        """Event-triggered steps should report measured durations, not hardcoded 5.0.

        ``_compute_step_duration_map`` measures the gap from each action to the
        *immediately following* episode on the same day.  In the test pattern:
          end_meeting → write_notes (gap1_min=10) → send_followup (gap2_min=20)

        So ``write_notes`` has a measured gap of ~20 min to ``send_followup``.
        The important assertion is that neither step uses the old 5.0 placeholder.
        """
        detector = RoutineDetector(db, user_model_store)
        self._build_event_trigger_pattern(
            user_model_store, trigger="end_meeting", gap1_min=10.0, gap2_min=20.0
        )

        routines = detector.detect_routines(lookback_days=30)
        triggered = [r for r in routines if "End Meeting" in r.get("name", "")]

        assert triggered, "Expected at least one 'After End Meeting' routine"
        routine = triggered[0]

        write_step = next((s for s in routine["steps"] if s["action"] == "write_notes"), None)
        assert write_step is not None
        # write_notes → send_followup gap is ~20 min; must NOT be the old 5.0 placeholder
        assert write_step["typical_duration_minutes"] != 5.0, (
            "Step duration should be measured (~20 min from write_notes→send_followup), "
            "not the old hardcoded 5.0"
        )
        # The measured gap (write_notes → send_followup) is gap2_min = 20 min
        assert abs(write_step["typical_duration_minutes"] - 20.0) < 2.0, (
            f"Expected ~20 min (write_notes→send_followup gap), "
            f"got {write_step['typical_duration_minutes']}"
        )

    def test_total_duration_reflects_measured_steps(self, db, user_model_store):
        """Event-triggered routine total duration should equal sum of measured steps."""
        detector = RoutineDetector(db, user_model_store)
        self._build_event_trigger_pattern(
            user_model_store, trigger="finish_sprint", gap1_min=8.0, gap2_min=16.0
        )

        routines = detector.detect_routines(lookback_days=30)
        triggered = [r for r in routines if "Finish Sprint" in r.get("name", "")]

        assert triggered
        routine = triggered[0]

        step_sum = sum(s["typical_duration_minutes"] for s in routine["steps"])
        # total_duration == step_sum (no last-step default applied to event-triggered)
        assert abs(routine["typical_duration_minutes"] - step_sum) < 0.01

    def test_old_hardcoded_total_no_longer_applies(self, db, user_model_store):
        """Old code computed total as len(steps) * 5.0; new code uses measured sums."""
        detector = RoutineDetector(db, user_model_store)
        # 3-step routine with 30-min gaps: old total = 3 * 5.0 = 15.0 (wrong)
        # new total = ~30.0 + ~30.0 + ~30.0 = ~90.0 (correct, or at least ≠ 15)
        self._build_event_trigger_pattern(
            user_model_store, trigger="deploy_release", gap1_min=30.0, gap2_min=30.0
        )

        routines = detector.detect_routines(lookback_days=30)
        triggered = [r for r in routines if "Deploy Release" in r.get("name", "")]

        assert triggered
        routine = triggered[0]
        # Total must be much larger than the old hardcoded 15.0 (3 steps × 5 min)
        assert routine["typical_duration_minutes"] > 15.0, (
            f"Expected total > 15 min (measured gaps), got {routine['typical_duration_minutes']}"
        )
