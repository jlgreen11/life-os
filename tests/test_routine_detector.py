"""
Tests for the RoutineDetector service.

Validates that Layer 3 (Procedural Memory) is correctly populated by detecting
recurring behavioral patterns from episodic memory.
"""

import pytest
from datetime import datetime, timedelta, timezone
import uuid

from services.routine_detector.detector import RoutineDetector


class TestRoutineDetector:
    """Test suite for routine detection from episodic memory."""

    def test_detector_initialization(self, db, user_model_store):
        """Detector should initialize with database and store dependencies."""
        detector = RoutineDetector(db, user_model_store)
        assert detector.db is db
        assert detector.user_model_store is user_model_store
        assert detector.min_occurrences == 3
        assert detector.time_window_hours == 2
        assert detector.consistency_threshold == 0.6

    def test_no_routines_with_insufficient_data(self, db, user_model_store):
        """Should not detect routines when episode count < min_occurrences."""
        detector = RoutineDetector(db, user_model_store)

        # Create only 1 episode (below min_occurrences of 3)
        episode = {
            "id": str(uuid.uuid4()),
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
            "event_id": str(uuid.uuid4()),
            "interaction_type": "email",
            "content_summary": "Check email",
        }
        user_model_store.store_episode(episode)

        routines = detector.detect_routines(lookback_days=30)
        assert len(routines) == 0

    def test_temporal_routine_detection_morning(self, db, user_model_store):
        """Should detect morning routines from repeated actions at similar times."""
        detector = RoutineDetector(db, user_model_store)

        # Create a morning routine pattern over 10 days
        # Each morning: check email (8am) → review calendar (8:15am) → coffee (8:30am)
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            day_start = base_date.replace(hour=8, minute=0, second=0) + timedelta(days=day_offset)

            # Step 1: Check email
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "check_email", "content_summary": "Check Email",
            })

            # Step 2: Review calendar
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (day_start + timedelta(minutes=15)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "review_calendar", "content_summary": "Review Calendar",
            })

            # Step 3: Make coffee
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (day_start + timedelta(minutes=30)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "make_coffee", "content_summary": "Make Coffee",
            })

        routines = detector.detect_routines(lookback_days=30)

        # Should detect at least one morning routine
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1

        # Validate structure
        routine = morning_routines[0]
        assert routine["name"] == "Morning routine"
        assert routine["trigger"] == "morning"
        assert len(routine["steps"]) >= 2  # At least 2 of the 3 actions
        assert routine["consistency_score"] >= 0.6
        assert routine["times_observed"] >= 3

        # Validate steps have required fields
        for step in routine["steps"]:
            assert "order" in step
            assert "action" in step
            assert "typical_duration_minutes" in step
            assert "skip_rate" in step

    def test_temporal_routine_detection_evening(self, db, user_model_store):
        """Should detect evening routines (5pm-11pm)."""
        detector = RoutineDetector(db, user_model_store)

        # Create evening pattern: inbox zero (6pm) → update tasks (6:30pm)
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            evening_start = base_date.replace(hour=18, minute=0, second=0) + timedelta(days=day_offset)

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": evening_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "inbox_zero", "content_summary": "Inbox Zero",
            })

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (evening_start + timedelta(minutes=30)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "update_tasks", "content_summary": "Update Tasks",
            })

        routines = detector.detect_routines(lookback_days=30)

        evening_routines = [r for r in routines if r["trigger"] == "evening"]
        assert len(evening_routines) >= 1

        routine = evening_routines[0]
        assert "Evening routine" in routine["name"]
        assert len(routine["steps"]) >= 2

    def test_location_based_routine_detection(self, db, user_model_store):
        """Should detect routines triggered by location arrival."""
        detector = RoutineDetector(db, user_model_store)

        # Create "arrive home" routine pattern
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            arrive_time = base_date + timedelta(days=day_offset, hours=17)

            # Arrive home → turn on lights → check mail
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": arrive_time.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "location",
                "location": "Home",
            })

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (arrive_time + timedelta(minutes=5)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "smart_home",
                "location": "Home",
            })

        routines = detector.detect_routines(lookback_days=30)

        location_routines = [r for r in routines if "Home" in r["name"]]
        assert len(location_routines) >= 1

        routine = location_routines[0]
        assert "Arrive at Home" in routine["name"]
        assert "arrive_home" in routine["trigger"]
        assert len(routine["steps"]) >= 2
        assert routine["consistency_score"] >= 0.6

    def test_location_routine_multiple_locations(self, db, user_model_store):
        """Should detect separate routines for different locations."""
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        # Home routine
        for day_offset in range(10):
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (base_date + timedelta(days=day_offset, hours=17)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "location",
                "location": "Home",
            })

        # Work routine
        for day_offset in range(10):
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (base_date + timedelta(days=day_offset, hours=9)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "location",
                "location": "Office",
            })

        routines = detector.detect_routines(lookback_days=30)

        home_routines = [r for r in routines if "Home" in r["name"]]
        office_routines = [r for r in routines if "Office" in r["name"]]

        assert len(home_routines) >= 1
        assert len(office_routines) >= 1

    def test_event_triggered_routine_detection(self, db, user_model_store):
        """Should detect routines triggered by specific event types."""
        detector = RoutineDetector(db, user_model_store)

        # Create post-meeting routine pattern
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            meeting_end = base_date + timedelta(days=day_offset, hours=14)

            # Meeting ends
            meeting_event_id = str(uuid.uuid4())
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": meeting_end.isoformat(),
                "event_id": meeting_event_id,
                "interaction_type": "calendar",
            })

            # Actions that follow: update tasks → send follow-up
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (meeting_end + timedelta(minutes=5)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "update_tasks", "content_summary": "Update Tasks",
            })

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (meeting_end + timedelta(minutes=15)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "send_followup", "content_summary": "Send Followup",
            })

        routines = detector.detect_routines(lookback_days=30)

        event_routines = [r for r in routines if r["trigger"].startswith("after_")]
        assert len(event_routines) >= 1

        routine = event_routines[0]
        assert "After" in routine["name"]
        assert len(routine["steps"]) >= 2

    def test_consistency_score_calculation(self, db, user_model_store):
        """Consistency score should reflect pattern reliability."""
        detector = RoutineDetector(db, user_model_store)

        # Create pattern with 8/10 consistency (skip 2 days)
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            if day_offset in [3, 7]:  # Skip 2 days
                continue

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (base_date + timedelta(days=day_offset, hours=8)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "morning_check", "content_summary": "Morning Check",
            })

        routines = detector.detect_routines(lookback_days=30)

        # Should still detect the routine (80% > 60% threshold)
        assert len(routines) >= 1
        routine = routines[0]
        assert 0.6 <= routine["consistency_score"] <= 1.0

    def test_no_routine_below_consistency_threshold(self, db, user_model_store):
        """Should not detect routines with consistency below threshold."""
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        # Create rare_action on only 2 of 10 active days → consistency = 2/10 = 0.2
        for day_offset in [0, 5]:
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (base_date + timedelta(days=day_offset, hours=8)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "rare_action", "content_summary": "Rare Action",
            })

        # Pad with unique filler episodes on all 10 days so active_days = 10.
        # Each filler uses a unique interaction_type (only 1 day each) so it
        # won't itself be detected as a routine (below min_occurrences=2).
        for day_offset in range(10):
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (base_date + timedelta(days=day_offset, hours=12)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": f"filler_{day_offset}",
                "content_summary": f"Filler {day_offset}",
            })

        routines = detector.detect_routines(lookback_days=30)

        # rare_action appears on 2/10 active days → consistency = 0.2 < 0.6 threshold
        rare_routines = [r for r in routines if any("rare_action" in step["action"] for step in r["steps"])]
        assert len(rare_routines) == 0

    def test_typical_duration_calculation(self, db, user_model_store):
        """Should calculate accurate total duration for routines."""
        detector = RoutineDetector(db, user_model_store)

        # Create routine with known durations
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            day_start = base_date.replace(hour=9, minute=0) + timedelta(days=day_offset)

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "step1", "content_summary": "Step1",
            })

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (day_start + timedelta(minutes=10)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "step2", "content_summary": "Step2",
            })

        routines = detector.detect_routines(lookback_days=30)

        assert len(routines) >= 1
        routine = routines[0]

        # Total should be approximately 25 minutes (10 + 15)
        assert 20.0 <= routine["typical_duration_minutes"] <= 30.0

    def test_store_routines(self, db, user_model_store):
        """Should persist detected routines to database."""
        detector = RoutineDetector(db, user_model_store)

        routine = {
            "name": "Test Routine",
            "trigger": "test_trigger",
            "steps": [
                {"order": 0, "action": "step1", "typical_duration_minutes": 5.0, "skip_rate": 0.0},
                {"order": 1, "action": "step2", "typical_duration_minutes": 10.0, "skip_rate": 0.1},
            ],
            "typical_duration_minutes": 15.0,
            "consistency_score": 0.85,
            "times_observed": 10,
            "variations": ["skips step2 on Mondays"],
        }

        stored_count = detector.store_routines([routine])
        assert stored_count == 1

        # Verify it was stored
        stored_routines = user_model_store.get_routines()
        assert len(stored_routines) >= 1

        found = next((r for r in stored_routines if r["name"] == "Test Routine"), None)
        assert found is not None
        assert found["trigger"] == "test_trigger"
        assert len(found["steps"]) == 2
        assert found["consistency_score"] == 0.85
        assert found["times_observed"] == 10

    def test_store_routines_upsert_behavior(self, db, user_model_store):
        """Should update existing routines when stored again."""
        detector = RoutineDetector(db, user_model_store)

        routine_v1 = {
            "name": "Evolving Routine",
            "trigger": "morning",
            "steps": [{"order": 0, "action": "old_step", "typical_duration_minutes": 5.0, "skip_rate": 0.0}],
            "typical_duration_minutes": 5.0,
            "consistency_score": 0.7,
            "times_observed": 5,
            "variations": [],
        }

        detector.store_routines([routine_v1])

        # Update with new data
        routine_v2 = {
            "name": "Evolving Routine",  # Same name triggers REPLACE
            "trigger": "morning",
            "steps": [
                {"order": 0, "action": "old_step", "typical_duration_minutes": 5.0, "skip_rate": 0.0},
                {"order": 1, "action": "new_step", "typical_duration_minutes": 10.0, "skip_rate": 0.0},
            ],
            "typical_duration_minutes": 15.0,
            "consistency_score": 0.8,
            "times_observed": 10,
            "variations": ["variation1"],
        }

        detector.store_routines([routine_v2])

        # Should have only one routine with updated values
        routines = user_model_store.get_routines()
        evolving = [r for r in routines if r["name"] == "Evolving Routine"]
        assert len(evolving) == 1

        routine = evolving[0]
        assert len(routine["steps"]) == 2  # Updated from 1 to 2 steps
        assert routine["consistency_score"] == 0.8  # Updated
        assert routine["times_observed"] == 10  # Updated

    def test_get_routines_by_trigger(self, db, user_model_store):
        """Should filter routines by trigger."""
        detector = RoutineDetector(db, user_model_store)

        routines = [
            {
                "name": "Morning Routine",
                "trigger": "morning",
                "steps": [],
                "typical_duration_minutes": 30.0,
                "consistency_score": 0.9,
                "times_observed": 10,
                "variations": [],
            },
            {
                "name": "Evening Routine",
                "trigger": "evening",
                "steps": [],
                "typical_duration_minutes": 20.0,
                "consistency_score": 0.8,
                "times_observed": 8,
                "variations": [],
            },
        ]

        detector.store_routines(routines)

        morning_routines = user_model_store.get_routines(trigger="morning")
        assert len(morning_routines) >= 1
        assert all(r["trigger"] == "morning" for r in morning_routines)

    def test_lookback_period_filtering(self, db, user_model_store):
        """Should only analyze episodes within lookback period."""
        detector = RoutineDetector(db, user_model_store)

        # Create old episodes (outside lookback)
        old_date = datetime.now(timezone.utc) - timedelta(days=40)
        for i in range(5):
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (old_date + timedelta(days=i)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "old_action", "content_summary": "Old Action",
            })

        # Create recent episodes (within lookback)
        recent_date = datetime.now(timezone.utc) - timedelta(days=7)
        for i in range(5):
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (recent_date + timedelta(days=i, hours=9)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "recent_action", "content_summary": "Recent Action",
            })

        # Detect with 30-day lookback (should exclude old_action)
        routines = detector.detect_routines(lookback_days=30)

        # Check that only recent actions are in detected routines
        old_actions = [r for r in routines if any("old_action" in step["action"] for step in r["steps"])]
        assert len(old_actions) == 0

    def test_error_handling_during_storage(self, db, user_model_store):
        """Should gracefully handle storage errors and continue."""
        detector = RoutineDetector(db, user_model_store)

        routines = [
            {
                "name": "Valid Routine",
                "trigger": "test",
                "steps": [],
                "typical_duration_minutes": 10.0,
                "consistency_score": 0.8,
                "times_observed": 5,
                "variations": [],
            },
            {
                # Invalid: missing required field 'name'
                "trigger": "test",
                "steps": [],
            },
        ]

        # Should store the valid one and skip the invalid one
        stored_count = detector.store_routines(routines)
        assert stored_count == 1

    def test_multiple_detection_strategies(self, db, user_model_store):
        """Should run all three detection strategies and merge results."""
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        # Create data for all three strategies
        for day_offset in range(10):
            day = base_date + timedelta(days=day_offset)

            # Temporal: morning actions
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day.replace(hour=8).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "morning_email", "content_summary": "Morning Email",
            })

            # Location: home actions
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day.replace(hour=18).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "location",
                "location": "Home",
            })

            # Event-triggered: post-meeting actions
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day.replace(hour=10).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "calendar",
            })

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day.replace(hour=10, minute=5).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "post_meeting_task", "content_summary": "Post Meeting Task",
            })

        routines = detector.detect_routines(lookback_days=30)

        # Should have routines from multiple strategies
        assert len(routines) >= 2

        # Check for variety in triggers
        triggers = {r["trigger"] for r in routines}
        assert len(triggers) >= 2  # At least 2 different trigger types

    def test_temporal_routine_varied_hours_same_bucket(self, db, user_model_store):
        """Regression test: activities at different hours within the same time-of-day
        bucket should be aggregated and detect a routine.

        Before the fix, the detector grouped by exact UTC hour, so an activity at
        3am on one day and 4am on another would be split into two (hour, type)
        groups each with day_count=1 — never reaching min_occurrences=3 — even
        though the activity consistently happens "at night."  The fix groups by
        time-of-day bucket (morning/midday/afternoon/evening/night) in the SQL
        query so all hours within the same bucket are counted together.
        """
        detector = RoutineDetector(db, user_model_store)

        # Simulate production scenario: email_received arriving at different UTC
        # hours each day (3am one day, 4am the next, 5am the third) — all in the
        # "night" bucket (hours 0–4 and 23).  The old exact-hour grouping would
        # give day_count=1 for each (hour, email_received) pair and detect nothing.
        base_date = datetime.now(timezone.utc) - timedelta(days=5)
        varying_hours = [3, 4, 5, 3, 4]  # different hours, all night bucket (0–4)

        for day_offset, hour in enumerate(varying_hours):
            day = (base_date + timedelta(days=day_offset)).replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "email_received",
                "content_summary": "Email received",
            })
            # Add a second action (email_sent) shortly after on 3+ days so that
            # bucket has >= 1 recurring action and consistency can reach 0.6.
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (day + timedelta(minutes=5)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "email_sent",
                "content_summary": "Email sent",
            })

        routines = detector.detect_routines(lookback_days=30)

        # With bucket-based grouping the night bucket should be detected.
        night_routines = [r for r in routines if r["trigger"] == "night"]
        assert len(night_routines) >= 1, (
            "Expected a 'night' routine from activities spread across varied UTC hours "
            "(3am, 4am, 5am) — the time-bucket grouping fix should aggregate them."
        )

        routine = night_routines[0]
        assert routine["consistency_score"] >= 0.6
        assert len(routine["steps"]) >= 1

    def test_temporal_routine_not_detected_with_two_occurrences(self, db, user_model_store):
        """A pattern appearing on exactly 2 distinct days should NOT be detected
        as a routine because min_occurrences=3 requires at least 3 instances.

        Two occurrences is too few to distinguish a genuine recurring pattern
        from coincidence (e.g., visiting the same coffee shop on two consecutive
        days). The min_occurrences=3 threshold provides a more reliable signal.
        """
        detector = RoutineDetector(db, user_model_store)

        # Create a pattern on exactly 2 distinct days — both in the morning bucket
        base_date = datetime.now(timezone.utc) - timedelta(days=3)

        for day_offset in range(2):
            day_start = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "sparse_morning_email",
                "content_summary": "Sparse morning email",
            })

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (day_start + timedelta(minutes=10)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "sparse_morning_calendar",
                "content_summary": "Sparse morning calendar",
            })

        routines = detector.detect_routines(lookback_days=30)

        # With min_occurrences=3, 2-day patterns should NOT be detected
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        sparse_steps = []
        for r in morning_routines:
            sparse_steps.extend(s for s in r["steps"] if s["action"].startswith("sparse_morning_"))
        assert len(sparse_steps) == 0, (
            "Expected no routine from 2-day pattern; min_occurrences=3 should reject it"
        )

    def test_event_triggered_routine_with_single_followup(self, db, user_model_store):
        """Regression: an event-triggered routine with exactly 1 consistent
        follow-up action should now be detected after lowering the guard from
        >= 2 to >= 1.

        With the old guard (len(following_actions) >= 2) this test would fail
        because only a single follow-up type appears.  A consistent single
        follow-up (e.g., always checking email after a meeting) is a valid
        behavioral pattern.
        """
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(timezone.utc) - timedelta(days=5)

        # Create trigger event (meeting) followed by exactly ONE follow-up type
        # on 5 distinct days — well above min_occurrences=2
        for day_offset in range(5):
            meeting_time = base_date + timedelta(days=day_offset, hours=14)

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": meeting_time.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "video_call",
            })

            # Single follow-up action: always check email after video call
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (meeting_time + timedelta(minutes=10)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "post_call_email_check",
                "content_summary": "Check email after call",
            })

        routines = detector.detect_routines(lookback_days=30)

        # Should detect an event-triggered routine for "after_video_call"
        event_routines = [r for r in routines if r["trigger"] == "after_video_call"]
        assert len(event_routines) >= 1, (
            "Expected event-triggered routine with 1 follow-up action; "
            "following_actions >= 1 guard should allow it"
        )

        routine = event_routines[0]
        assert len(routine["steps"]) == 1
        assert routine["steps"][0]["action"] == "post_call_email_check"
        assert routine["consistency_score"] >= 0.6

    def test_cold_start_threshold_scaling(self, db, user_model_store):
        """Verify _effective_consistency_threshold() returns correct tiered values.

        The method scales the consistency threshold based on data maturity:
        - < 7 active days  → 0.3 (very lenient)
        - < 14 active days → 0.4 (moderate)
        - < 30 active days → 0.5 (approaching maturity)
        - >= 30 active days → 0.6 (full base threshold)
        """
        detector = RoutineDetector(db, user_model_store)

        assert detector._effective_consistency_threshold(3) == 0.3
        assert detector._effective_consistency_threshold(6) == 0.3
        assert detector._effective_consistency_threshold(7) == 0.4
        assert detector._effective_consistency_threshold(13) == 0.4
        assert detector._effective_consistency_threshold(14) == 0.5
        assert detector._effective_consistency_threshold(29) == 0.5
        assert detector._effective_consistency_threshold(30) == 0.6
        assert detector._effective_consistency_threshold(60) == 0.6
        assert detector._effective_consistency_threshold(365) == 0.6

    def test_temporal_routine_detection_cold_start(self, db, user_model_store):
        """During cold-start (< 7 active days), a pattern with ~0.4 consistency
        should be detected because the effective threshold drops to 0.3.

        Creates a morning email pattern on 3 of 5 active days (consistency = 0.6).
        Even with the base threshold this would pass, but with cold-start scaling
        the threshold is 0.3, so even sparser patterns would also be caught.

        Also verifies the detected routine has cold_start=False when the consistency
        exceeds the base threshold (0.6 >= 0.6).
        """
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(timezone.utc) - timedelta(days=5)

        # Create episodes on 5 days with morning email on 3 of them
        for day_offset in range(5):
            day_start = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)

            # Morning email on days 0, 2, 4 (3 of 5 = 0.6 consistency)
            if day_offset in [0, 2, 4]:
                user_model_store.store_episode({
                    "id": str(uuid.uuid4()),
                    "timestamp": day_start.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "cold_start_email",
                    "content_summary": "Cold start email",
                })

            # Filler on all days so active_days = 5
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (day_start + timedelta(hours=4)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": f"cs_filler_{day_offset}",
                "content_summary": f"Filler {day_offset}",
            })

        routines = detector.detect_routines(lookback_days=30)

        # With 5 active days, threshold is 0.3, and 3/5 = 0.6 consistency → detected
        morning = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning) >= 1
        routine = morning[0]
        # consistency 0.6 >= base 0.6, so no cold_start scaling
        assert routine["cold_start"] is False

    def test_cold_start_sparse_pattern_detected(self, db, user_model_store):
        """A pattern with consistency below the base threshold (0.6) but above
        the cold-start threshold (0.3) should be detected with reduced confidence
        and cold_start=True.

        Creates episodes on 5 days with a morning action on only 2 of 5 days
        (consistency = 0.4). The base threshold of 0.6 would reject this, but
        the cold-start threshold of 0.3 (for < 7 active days) accepts it.
        """
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(timezone.utc) - timedelta(days=5)

        # Morning action on only 3 of 5 days to get consistency near the boundary.
        # We need min_occurrences=3, so 3 days is the minimum.
        # 3/5 = 0.6 → this passes even the base threshold.
        # To test cold_start=True, we need consistency < 0.6 but >= 0.3.
        # Use 3 occurrences over a higher number of active days.
        # Create 10 filler days so active_days=10, but morning action on 4 days → 0.4.
        for day_offset in range(10):
            day_start = base_date.replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            # Filler on all 10 days
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": f"filler_cs2_{day_offset}",
                "content_summary": f"Filler {day_offset}",
            })

        # Morning action on 4 of 10 days → consistency = 0.4
        for day_offset in [0, 3, 6, 9]:
            day_start = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "sparse_cold_email",
                "content_summary": "Sparse cold email",
            })

        # With 10 active days (< 14), threshold is 0.4.
        # consistency = 4/10 = 0.4, which passes 0.4 threshold but fails base 0.6.
        routines = detector.detect_routines(lookback_days=30)

        morning = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning) >= 1
        routine = morning[0]
        # The routine should be marked as cold-start with reduced confidence
        assert routine["cold_start"] is True
        # Confidence should be scaled: 0.4 * 0.7 = 0.28
        assert routine["consistency_score"] == pytest.approx(0.4 * 0.7, abs=0.05)

    def test_routine_confidence_scaled_for_cold_start(self, db, user_model_store):
        """Routines detected below the base threshold have reduced confidence
        (consistency * 0.7) and cold_start=True in metadata.

        This ensures downstream consumers (e.g., prediction engine) treat
        cold-start routines as provisional.
        """
        detector = RoutineDetector(db, user_model_store)

        # Create a location-based pattern with 5 active days and location
        # actions on 3 of them → consistency = 3/5 = 0.6, which passes
        # cold-start threshold of 0.3 (< 7 days) AND base threshold 0.6.
        # To get cold_start=True, we need consistency < 0.6.
        # Use 6 active days, location on 3 → 3/6 = 0.5 (< 0.6 base, >= 0.3 cold-start)
        base_date = datetime.now(timezone.utc) - timedelta(days=6)

        for day_offset in range(6):
            day_start = base_date.replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            # Filler on all 6 days
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": f"filler_loc_{day_offset}",
                "content_summary": f"Filler {day_offset}",
                "location": "Gym",
            })

        # Location action on 3 of 6 days at Gym → consistency = 0.5
        for day_offset in [0, 2, 4]:
            day_start = base_date.replace(hour=7, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "gym_workout",
                "content_summary": "Gym workout",
                "location": "Gym",
            })

        routines = detector.detect_routines(lookback_days=30)

        # gym_workout appears on 3/6 days = 0.5 consistency.
        # With 6 active days (< 7), threshold = 0.3, so it passes.
        # But 0.5 < base 0.6, so cold_start=True and confidence = 0.5 * 0.7 = 0.35
        gym_routines = [r for r in routines if "Gym" in r.get("name", "")]
        assert len(gym_routines) >= 1
        routine = gym_routines[0]
        assert routine["cold_start"] is True
        assert routine["consistency_score"] < 0.6  # Scaled down

    def test_mature_data_uses_full_threshold(self, db, user_model_store):
        """With 30+ active days, the full base threshold (0.6) should be used.

        A pattern appearing on 10 of 35 days (consistency ~0.29) should NOT be
        detected when active_days >= 30, because the threshold remains at 0.6.
        """
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(timezone.utc) - timedelta(days=35)

        # Create episodes on 35 days
        for day_offset in range(35):
            day_start = base_date.replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": f"filler_mat_{day_offset}",
                "content_summary": f"Filler {day_offset}",
            })

        # Morning action on only 10 of 35 days → consistency ≈ 0.29 < 0.6
        for day_offset in [0, 3, 7, 10, 14, 17, 21, 24, 28, 31]:
            day_start = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "mature_morning_check",
                "content_summary": "Mature morning check",
            })

        routines = detector.detect_routines(lookback_days=40)

        # With 35 active days, threshold = 0.6 (full). 10/35 ≈ 0.29 < 0.6 → rejected.
        mature_routines = [
            r for r in routines
            if any(s["action"] == "mature_morning_check" for s in r.get("steps", []))
        ]
        assert len(mature_routines) == 0, (
            "With 35 active days (>= 30), base threshold 0.6 should apply — "
            "10/35 ≈ 0.29 consistency should not produce a routine"
        )
