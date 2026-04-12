"""
Tests for RoutineDetector behavior with email-dominated episode data.

Real-world Life OS deployments often have >95% of episodes classified as
``email_received`` because that connector is always-on.  With the default
60% consistency threshold, a genuine morning-email habit (user checks email
every morning) was not being detected because:

  - Email arrival time is driven by *external senders*, not the user.
  - Even with 30 emails per day, only some land in the 5–10 AM morning
    bucket — the rest arrive in afternoon/evening.
  - A bucket with ~50–55% day-coverage fails the 0.6 threshold by a hair.

The fix: ``HIGH_VOLUME_PASSIVE_TYPES`` (``email_received``, ``email_sent``,
``message_received``, ``notification_received``) use a lower consistency cap
(``PASSIVE_TYPE_CONSISTENCY_THRESHOLD = 0.4``).  This cap only applies when
the day-tier threshold exceeds 0.4 — cold-start scaling is preserved.

These tests verify three scenarios:

1. **Morning concentration** — 100 emails at 9 AM across 30 days produce a
   "Morning routine" detection.

2. **Uniform distribution** — 50 emails spread evenly across ALL five time
   buckets (10 emails × 5 buckets, each on distinct days) do NOT produce
   any routine because per-bucket day coverage is too low.

3. **Calendar below threshold** — 6 calendar_blocked episodes in a window
   dominated by 90 email episodes do NOT produce a calendar routine because
   6/30 = 0.2 < 0.3 (minimum threshold).

Test patterns follow ``tests/test_routine_detector_adaptive_lookback.py``:
- Real ``DatabaseManager`` + ``UserModelStore`` via conftest fixtures.
- No mocking of the storage layer.
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
    """Insert a single episode into the user-model store.

    Args:
        user_model_store: Store fixture from conftest.
        interaction_type: Episode interaction type (e.g. ``email_received``).
        timestamp: UTC datetime for the episode.
        content_summary: Human-readable summary (default generic string).
    """
    user_model_store.store_episode(
        {
            "id": str(uuid.uuid4()),
            "timestamp": timestamp.isoformat(),
            "event_id": str(uuid.uuid4()),
            "interaction_type": interaction_type,
            "content_summary": content_summary,
        }
    )


def _insert_morning_emails(
    user_model_store,
    *,
    num_days: int,
    days_ago_start: int,
    hour: int = 9,
    emails_per_day: int = 3,
) -> None:
    """Insert emails concentrated in the morning bucket (hour 5–10 AM).

    Creates ``emails_per_day`` episodes per day at the given ``hour`` across
    ``num_days`` consecutive days ending at ``days_ago_start`` days before now.

    Args:
        user_model_store: Store fixture from conftest.
        num_days: Number of consecutive days to insert emails on.
        days_ago_start: How many days before now the FIRST day falls.
        hour: UTC hour for all inserted emails (default 9).
        emails_per_day: How many email episodes per day (default 3).
    """
    base = datetime.now(UTC) - timedelta(days=days_ago_start)
    base = base.replace(hour=hour, minute=0, second=0, microsecond=0)
    for day_offset in range(num_days):
        for ep_idx in range(emails_per_day):
            ts = base + timedelta(days=day_offset, minutes=ep_idx * 10)
            _make_episode(user_model_store, "email_received", ts)


# ---------------------------------------------------------------------------
# Threshold unit tests
# ---------------------------------------------------------------------------


class TestPassiveTypeThreshold:
    """Verify the type-aware threshold logic in isolation."""

    def test_passive_type_cap_applied_at_maturity(self, db, user_model_store):
        """email_received at >= 30 active days uses 0.4, not the base 0.6."""
        detector = RoutineDetector(db, user_model_store)
        # 30+ days → base threshold would be 0.6; cap should bring it to 0.4.
        threshold = detector._effective_consistency_threshold(30, "email_received")
        assert threshold == 0.4, f"Expected 0.4 for email_received at 30 days, got {threshold}"

    def test_passive_type_cap_applied_at_near_maturity(self, db, user_model_store):
        """email_received at 20 active days (tier=0.5) is capped to 0.4."""
        detector = RoutineDetector(db, user_model_store)
        threshold = detector._effective_consistency_threshold(20, "email_received")
        assert threshold == 0.4, f"Expected 0.4 for email_received at 20 days, got {threshold}"

    def test_passive_type_cold_start_not_overridden(self, db, user_model_store):
        """Cold-start scaling (< 7 days, tier=0.3) is preserved for email_received."""
        detector = RoutineDetector(db, user_model_store)
        # Cold-start tier (0.3) is already below the passive cap (0.4).
        # min(0.3, 0.4) = 0.3 — should NOT be overridden upward.
        threshold = detector._effective_consistency_threshold(5, "email_received")
        assert threshold == 0.3, f"Expected cold-start 0.3 preserved, got {threshold}"

    def test_non_passive_type_unaffected(self, db, user_model_store):
        """calendar_blocked at 30 days still uses the full 0.6 threshold."""
        detector = RoutineDetector(db, user_model_store)
        threshold = detector._effective_consistency_threshold(30, "calendar_blocked")
        assert threshold == 0.6, f"Expected 0.6 for calendar_blocked at 30 days, got {threshold}"

    def test_none_type_unaffected(self, db, user_model_store):
        """None interaction_type leaves the day-tier threshold unchanged."""
        detector = RoutineDetector(db, user_model_store)
        threshold = detector._effective_consistency_threshold(30, None)
        assert threshold == 0.6, f"Expected 0.6 for None type at 30 days, got {threshold}"

    def test_message_received_also_passive(self, db, user_model_store):
        """message_received is in HIGH_VOLUME_PASSIVE_TYPES and receives the cap."""
        detector = RoutineDetector(db, user_model_store)
        threshold = detector._effective_consistency_threshold(30, "message_received")
        assert threshold == 0.4

    def test_email_sent_also_passive(self, db, user_model_store):
        """email_sent is in HIGH_VOLUME_PASSIVE_TYPES and receives the cap."""
        detector = RoutineDetector(db, user_model_store)
        threshold = detector._effective_consistency_threshold(30, "email_sent")
        assert threshold == 0.4

    def test_existing_tier_values_unchanged_without_type(self, db, user_model_store):
        """Pre-existing cold-start tier values are unchanged when type is omitted."""
        detector = RoutineDetector(db, user_model_store)
        # These mirror the assertions in test_routine_detector.py to confirm
        # the new optional parameter does not break the base behavior.
        assert detector._effective_consistency_threshold(3) == 0.3
        assert detector._effective_consistency_threshold(7) == 0.4
        assert detector._effective_consistency_threshold(14) == 0.5
        assert detector._effective_consistency_threshold(30) == 0.6


# ---------------------------------------------------------------------------
# Integration tests: morning concentration produces a routine
# ---------------------------------------------------------------------------


class TestMorningEmailRoutineDetected:
    """100 email_received episodes at 9 AM across 30 days → morning routine."""

    def test_morning_email_routine_detected(self, db, user_model_store):
        """Concentrated morning emails should produce at least one morning routine.

        Setup: 100 episodes (3–4 per day) at 9 AM UTC over 30 consecutive days.
        Expectation: detect_routines() returns >= 1 routine whose trigger is
        "morning", demonstrating that the type-aware threshold enables detection.
        """
        # Insert 30 days × ~3 emails at 9 AM = 90 email_received episodes.
        _insert_morning_emails(
            user_model_store,
            num_days=30,
            days_ago_start=30,
            hour=9,
            emails_per_day=3,
        )

        detector = RoutineDetector(db, user_model_store)
        routines = detector.detect_routines(lookback_days=35)

        morning_routines = [r for r in routines if r.get("trigger") == "morning"]
        assert len(morning_routines) >= 1, (
            f"Expected at least 1 morning routine from concentrated 9 AM emails, "
            f"got {len(routines)} total routines: {[r.get('trigger') for r in routines]}"
        )

    def test_morning_email_step_contains_email_received(self, db, user_model_store):
        """The detected morning routine's first step should be email_received."""
        _insert_morning_emails(
            user_model_store,
            num_days=30,
            days_ago_start=30,
            hour=9,
            emails_per_day=3,
        )

        detector = RoutineDetector(db, user_model_store)
        routines = detector.detect_routines(lookback_days=35)

        morning_routines = [r for r in routines if r.get("trigger") == "morning"]
        assert morning_routines, "No morning routine detected"

        steps = morning_routines[0].get("steps", [])
        assert steps, "Morning routine has no steps"

        # The dominant step action should be email_received.
        actions_in_routine = [s.get("action") for s in steps]
        assert "email_received" in actions_in_routine, (
            f"Expected email_received in morning routine steps, got: {actions_in_routine}"
        )

    def test_consistency_score_above_passive_threshold(self, db, user_model_store):
        """Detected routine's consistency_score should reflect the actual day coverage."""
        _insert_morning_emails(
            user_model_store,
            num_days=30,
            days_ago_start=30,
            hour=9,
            emails_per_day=3,
        )

        detector = RoutineDetector(db, user_model_store)
        routines = detector.detect_routines(lookback_days=35)

        morning_routines = [r for r in routines if r.get("trigger") == "morning"]
        assert morning_routines, "No morning routine detected"

        consistency = morning_routines[0].get("consistency_score", 0.0)
        assert consistency >= 0.4, (
            f"Morning routine consistency_score={consistency:.3f} is below "
            f"passive threshold 0.4; routine appears spuriously weak"
        )


# ---------------------------------------------------------------------------
# Integration tests: uniform distribution does NOT produce spurious routines
# ---------------------------------------------------------------------------


class TestUniformEmailNoSpuriousRoutines:
    """50 emails spread evenly across all time buckets should NOT form a routine.

    The key insight: if emails are spread uniformly across all five time
    buckets (morning/midday/afternoon/evening/night), each bucket covers only
    ~20% of active days.  With active_days = 25 (one unique day per episode)
    and per-bucket day_count ≈ 5, consistency ≈ 5/25 = 0.20 < 0.3 (even the
    most lenient threshold) → no routines.
    """

    def test_uniform_email_no_routines(self, db, user_model_store):
        """Emails spread uniformly across all time buckets produce no routines.

        Setup: 50 email_received episodes, 10 per time bucket, each episode on
        a distinct calendar day so that no bucket achieves meaningful day
        coverage relative to the total active-day count.
        """
        # Five canonical bucket hours (one per bucket).
        # morning=9, midday=12, afternoon=14, evening=19, night=23
        bucket_hours = [9, 12, 14, 19, 23]
        now = datetime.now(UTC)

        for bucket_idx, hour in enumerate(bucket_hours):
            for day_offset in range(10):
                # Place each bucket's 10 episodes on 10 distinct days,
                # interleaved with other buckets so active_days is large.
                # Bucket 0 uses days 0,5,10,...45; bucket 1 uses 1,6,11,...46
                absolute_day = bucket_idx + day_offset * 5
                ts = now - timedelta(days=absolute_day + 5)
                ts = ts.replace(hour=hour, minute=0, second=0, microsecond=0)
                _make_episode(user_model_store, "email_received", ts)

        detector = RoutineDetector(db, user_model_store)
        routines = detector.detect_routines(lookback_days=60)

        assert len(routines) == 0, (
            f"Expected 0 routines for uniformly distributed emails, "
            f"got {len(routines)}: {[r.get('trigger') for r in routines]}"
        )


# ---------------------------------------------------------------------------
# Integration tests: sparse calendar episodes do NOT produce routines
# ---------------------------------------------------------------------------


class TestCalendarBelowThresholdNoRoutine:
    """6 calendar_blocked episodes with 30-day email context → no calendar routine.

    With active_days = 30 (driven by the email episodes) and calendar on only
    6 distinct days, consistency = 6/30 = 0.20 < 0.3 (minimum threshold),
    so no calendar routine should be detected.
    """

    def test_sparse_calendar_no_routine(self, db, user_model_store):
        """6 calendar_blocked episodes do not form a routine.

        A dominant email presence pushes active_days to 30, making the calendar
        episodes appear sparse even though 6 > min_occurrences (3).
        """
        now = datetime.now(UTC)

        # Insert 30 days of morning emails to establish active_days = 30.
        _insert_morning_emails(
            user_model_store,
            num_days=30,
            days_ago_start=30,
            hour=9,
            emails_per_day=2,
        )

        # Insert 6 calendar_blocked episodes spread across 6 different days,
        # also in the morning bucket — same time bucket as emails.
        for day_i in range(6):
            ts = now - timedelta(days=day_i * 5 + 2)
            ts = ts.replace(hour=10, minute=0, second=0, microsecond=0)
            _make_episode(user_model_store, "calendar_blocked", ts)

        detector = RoutineDetector(db, user_model_store)
        routines = detector.detect_routines(lookback_days=35)

        # Check that no routine exists where calendar_blocked is the dominant step.
        calendar_routines = [
            r
            for r in routines
            if r.get("steps") and r["steps"][0].get("action") == "calendar_blocked"
        ]
        assert len(calendar_routines) == 0, (
            f"Expected 0 calendar_blocked routines (sparse data, 6/30 active days), "
            f"got {len(calendar_routines)}"
        )

    def test_email_routine_still_detected_alongside_calendar(self, db, user_model_store):
        """Even with sparse calendar data, the email morning routine IS still detected.

        This verifies that fixing the calendar case doesn't suppress legitimate
        email routines — both outcomes (email routine detected, calendar not) hold.
        """
        now = datetime.now(UTC)

        # Dense morning emails.
        _insert_morning_emails(
            user_model_store,
            num_days=30,
            days_ago_start=30,
            hour=9,
            emails_per_day=3,
        )

        # Sparse calendar events (6 days only).
        for day_i in range(6):
            ts = now - timedelta(days=day_i * 5 + 2)
            ts = ts.replace(hour=10, minute=0, second=0, microsecond=0)
            _make_episode(user_model_store, "calendar_blocked", ts)

        detector = RoutineDetector(db, user_model_store)
        routines = detector.detect_routines(lookback_days=35)

        # Email morning routine should be detected.
        morning_routines = [r for r in routines if r.get("trigger") == "morning"]
        assert len(morning_routines) >= 1, (
            f"Expected at least 1 morning routine for dense email data, "
            f"got {len(routines)} total"
        )

        # Calendar routine should NOT dominate any routine.
        calendar_routines = [
            r
            for r in routines
            if r.get("steps") and r["steps"][0].get("action") == "calendar_blocked"
        ]
        assert len(calendar_routines) == 0, (
            f"Sparse calendar (6/30 days) should not form its own routine, "
            f"but got {len(calendar_routines)}"
        )
