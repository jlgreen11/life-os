"""
Tests for location and event-triggered routine detector fallbacks.

Verifies that _detect_location_routines and _detect_event_triggered_routines
can recover episodes with NULL interaction_type by deriving the type from
the linked event's type field in events.db, following the same pattern as
_fallback_temporal_episodes (added in PR #546).
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.routine_detector.detector import RoutineDetector


def _store_event(event_store, event_id: str, event_type: str, timestamp: str):
    """Helper to insert a raw event into events.db for fallback derivation."""
    event_store.store_event({
        "id": event_id,
        "type": event_type,
        "source": "test",
        "timestamp": timestamp,
        "priority": "normal",
        "payload": json.dumps({}),
        "metadata": json.dumps({}),
    })


def _store_episode_with_null_type(
    db, event_id: str, timestamp: str, location: str | None = None
):
    """Helper to store an episode with NULL interaction_type.

    In production, many episodes have NULL interaction_type from before the
    granular classification was deployed. The episodes table may have a NOT NULL
    constraint in the test schema, so we insert directly via SQL to simulate
    the production state. If direct NULL insert fails (NOT NULL constraint),
    we fall back to inserting with 'unknown' which the fallback logic treats
    identically to NULL.
    """
    episode_id = str(uuid.uuid4())
    try:
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO episodes
                   (id, timestamp, event_id, interaction_type, content_summary, location)
                   VALUES (?, ?, ?, NULL, ?, ?)""",
                (episode_id, timestamp, event_id, "Episode with null type", location),
            )
    except Exception:
        # NOT NULL constraint — use 'unknown' placeholder which the fallback
        # treats the same as NULL
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO episodes
                   (id, timestamp, event_id, interaction_type, content_summary, location)
                   VALUES (?, ?, ?, 'unknown', ?, ?)""",
                (episode_id, timestamp, event_id, "Episode with null type", location),
            )


class TestLocationRoutineFallback:
    """Tests for the location-based routine detector fallback path."""

    def test_location_routine_detected_via_fallback(self, db, event_store, user_model_store):
        """Episodes with NULL interaction_type but non-NULL location should
        produce location routines via the fallback derivation path.

        Creates episodes at 'Home' with NULL interaction_type and matching
        events in events.db with type 'smart_home.toggled'. The fallback
        should derive 'smart_home_toggled' and detect a location routine.
        """
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            ts = (base_date + timedelta(days=day_offset, hours=17)).isoformat()
            event_id = str(uuid.uuid4())

            # Store the event in events.db so fallback can derive the type
            _store_event(event_store, event_id, "smart_home.toggled", ts)

            # Store episode with NULL interaction_type but with location
            _store_episode_with_null_type(db, event_id, ts, location="Home")

        routines = detector.detect_routines(lookback_days=30)

        # Should detect a location routine for "Home" via fallback derivation
        home_routines = [r for r in routines if "Home" in r.get("name", "")]
        assert len(home_routines) >= 1, (
            "Expected location routine for 'Home' from NULL-type episodes "
            "recovered via event_type derivation fallback"
        )
        routine = home_routines[0]
        assert "Arrive at Home" in routine["name"]
        assert routine["consistency_score"] > 0

    def test_location_fallback_not_used_when_main_query_has_results(
        self, db, event_store, user_model_store
    ):
        """When the main location query returns results, the fallback should
        NOT be used (performance guard).

        Creates episodes with proper interaction_type set. The fallback
        code path should be skipped entirely.
        """
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            ts = (base_date + timedelta(days=day_offset, hours=17)).isoformat()

            # Store episode WITH a proper interaction_type and location
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": ts,
                "event_id": str(uuid.uuid4()),
                "interaction_type": "smart_home",
                "location": "Office",
                "content_summary": "Smart home action at office",
            })

        routines = detector.detect_routines(lookback_days=30)

        # Should detect location routine via main query (not fallback)
        office_routines = [r for r in routines if "Office" in r.get("name", "")]
        assert len(office_routines) >= 1

    def test_location_fallback_skips_underivable_episodes(
        self, db, event_store, user_model_store
    ):
        """Episodes where _derive_interaction_type_from_event returns None
        (no matching event in events.db) should be silently skipped.

        Only episodes with derivable types should count toward routines.
        """
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            ts = (base_date + timedelta(days=day_offset, hours=17)).isoformat()
            event_id = str(uuid.uuid4())

            # Do NOT store a matching event in events.db
            # The fallback should skip these episodes

            _store_episode_with_null_type(db, event_id, ts, location="GhostPlace")

        routines = detector.detect_routines(lookback_days=30)

        # No routines should be detected for GhostPlace since derivation fails
        ghost_routines = [r for r in routines if "GhostPlace" in r.get("name", "")]
        assert len(ghost_routines) == 0, (
            "Episodes with underivable interaction_type should be skipped"
        )

    def test_location_fallback_mixed_derivable_and_underivable(
        self, db, event_store, user_model_store
    ):
        """Mix of derivable and underivable episodes: only derivable ones
        should contribute to routine detection.
        """
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            ts = (base_date + timedelta(days=day_offset, hours=17)).isoformat()

            if day_offset < 5:
                # First 5 days: derivable (event exists in events.db)
                event_id = str(uuid.uuid4())
                _store_event(event_store, event_id, "location.arrived", ts)
                _store_episode_with_null_type(db, event_id, ts, location="Gym")
            else:
                # Last 5 days: underivable (no matching event)
                event_id = str(uuid.uuid4())
                _store_episode_with_null_type(db, event_id, ts, location="Gym")

        routines = detector.detect_routines(lookback_days=30)

        # 5 derivable episodes on 5 distinct days >= min_occurrences (3)
        gym_routines = [r for r in routines if "Gym" in r.get("name", "")]
        assert len(gym_routines) >= 1, (
            "5 derivable episodes on distinct days should produce a routine"
        )

    def test_fallback_location_episodes_method_directly(self, db, event_store, user_model_store):
        """Test the _fallback_location_episodes method directly to verify
        it returns the correct (location, interaction_type, timestamp) tuples.
        """
        detector = RoutineDetector(db, user_model_store)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)

        ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        event_id = str(uuid.uuid4())

        # Store event with type that should be converted
        _store_event(event_store, event_id, "email.received", ts)

        # Store episode with NULL interaction_type and location
        _store_episode_with_null_type(db, event_id, ts, location="Office")

        results = detector._fallback_location_episodes(cutoff)

        assert len(results) >= 1
        # Should be (location, derived_type, timestamp) tuple
        location, derived_type, result_ts = results[0]
        assert location == "Office"
        assert derived_type == "email_received"  # dots converted to underscores

    def test_fallback_preserves_existing_interaction_types(
        self, db, event_store, user_model_store
    ):
        """Episodes with existing non-null, non-placeholder interaction_type
        should use their existing type in the fallback, not re-derive.
        """
        detector = RoutineDetector(db, user_model_store)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)

        ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        # Store episode with existing interaction_type "smart_home" and location
        user_model_store.store_episode({
            "id": str(uuid.uuid4()),
            "timestamp": ts,
            "event_id": str(uuid.uuid4()),
            "interaction_type": "smart_home",
            "location": "Home",
            "content_summary": "Smart home action",
        })

        results = detector._fallback_location_episodes(cutoff)

        assert len(results) >= 1
        location, itype, _ = results[0]
        assert location == "Home"
        assert itype == "smart_home"  # Original type preserved


class TestEventTriggeredRoutineFallback:
    """Tests for the event-triggered routine detector fallback path."""

    def test_event_triggered_routine_detected_via_fallback(
        self, db, event_store, user_model_store
    ):
        """Episodes with NULL/unknown interaction_type should produce
        event-triggered routines via the fallback derivation path.

        Creates a pattern: calendar event (derived from 'calendar.event_ended')
        followed by task update (derived from 'task.created') across 10 days.
        ALL episodes have 'unknown' type so the main query returns nothing
        and the fallback is triggered. Both triggers and follow-ups are derived
        from their linked events in events.db.
        """
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            meeting_time = base_date + timedelta(days=day_offset, hours=14)
            followup_time = meeting_time + timedelta(minutes=10)

            # Trigger event (calendar meeting) — unknown type, derived via fallback
            trigger_event_id = str(uuid.uuid4())
            trigger_ts = meeting_time.isoformat()
            _store_event(event_store, trigger_event_id, "calendar.event_ended", trigger_ts)
            _store_episode_with_null_type(db, trigger_event_id, trigger_ts)

            # Follow-up action (task created) — also unknown type, derived via fallback
            followup_event_id = str(uuid.uuid4())
            followup_ts = followup_time.isoformat()
            _store_event(event_store, followup_event_id, "task.created", followup_ts)
            _store_episode_with_null_type(db, followup_event_id, followup_ts)

        routines = detector.detect_routines(lookback_days=30)

        # Should detect an event-triggered routine via fallback derivation
        event_routines = [r for r in routines if r.get("trigger", "").startswith("after_")]
        assert len(event_routines) >= 1, (
            "Expected event-triggered routine from unknown-type episodes "
            "recovered via event_type derivation fallback"
        )

    def test_event_triggered_fallback_not_used_when_main_query_has_results(
        self, db, event_store, user_model_store
    ):
        """When the main trigger query returns results, the fallback should
        NOT be used (performance guard).
        """
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            meeting_time = base_date + timedelta(days=day_offset, hours=14)

            # Trigger event WITH proper interaction_type
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": meeting_time.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "video_call",
            })

            # Follow-up action
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (meeting_time + timedelta(minutes=5)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "post_call_notes",
                "content_summary": "Post call notes",
            })

        routines = detector.detect_routines(lookback_days=30)

        # Should detect routine via main query (not fallback)
        event_routines = [r for r in routines if r.get("trigger", "").startswith("after_")]
        assert len(event_routines) >= 1

    def test_event_triggered_fallback_skips_underivable_episodes(
        self, db, event_store, user_model_store
    ):
        """Episodes where _derive_interaction_type_from_event returns None
        should be silently skipped in the fallback.
        """
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(timezone.utc) - timedelta(days=5)

        for day_offset in range(5):
            ts = (base_date + timedelta(days=day_offset, hours=10)).isoformat()
            event_id = str(uuid.uuid4())

            # Do NOT store event in events.db — derivation will fail
            _store_episode_with_null_type(db, event_id, ts)

        routines = detector.detect_routines(lookback_days=30)

        # No event-triggered routines should be detected from underivable episodes
        event_routines = [r for r in routines if r.get("trigger", "").startswith("after_")]
        # These episodes would need both a trigger and follow-up to form a routine
        # With no derivable types, no triggers are found
        assert len(event_routines) == 0

    def test_fallback_event_triggered_episodes_method_directly(
        self, db, event_store, user_model_store
    ):
        """Test the _fallback_event_triggered_episodes method directly to verify
        it returns the correct (interaction_type, timestamp) tuples.
        """
        detector = RoutineDetector(db, user_model_store)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)

        ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        event_id = str(uuid.uuid4())

        # Store event with dotted type
        _store_event(event_store, event_id, "email.sent", ts)

        # Store episode with NULL interaction_type
        _store_episode_with_null_type(db, event_id, ts)

        results = detector._fallback_event_triggered_episodes(cutoff)

        assert len(results) >= 1
        derived_type, result_ts = results[0]
        assert derived_type == "email_sent"  # dots converted to underscores

    def test_event_triggered_fallback_across_three_days(
        self, db, event_store, user_model_store
    ):
        """Verify that fallback-derived trigger types need at least 3 distinct
        days (min_occurrences) to be considered candidates.

        Creates episodes on exactly 3 distinct days with unknown interaction_type.
        Both trigger and follow-up have 'unknown' type so the main query returns
        nothing and the fallback is triggered.
        """
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(timezone.utc) - timedelta(days=5)

        for day_offset in [0, 2, 4]:  # 3 distinct days
            meeting_time = base_date + timedelta(days=day_offset, hours=14)
            followup_time = meeting_time + timedelta(minutes=10)

            # Trigger: unknown type, derived from events.db
            trigger_event_id = str(uuid.uuid4())
            trigger_ts = meeting_time.isoformat()
            _store_event(event_store, trigger_event_id, "calendar.meeting", trigger_ts)
            _store_episode_with_null_type(db, trigger_event_id, trigger_ts)

            # Follow-up: also unknown type, derived from events.db
            followup_event_id = str(uuid.uuid4())
            followup_ts = followup_time.isoformat()
            _store_event(event_store, followup_event_id, "task.created", followup_ts)
            _store_episode_with_null_type(db, followup_event_id, followup_ts)

        routines = detector.detect_routines(lookback_days=30)

        # 3 days meets min_occurrences=3, so trigger should be found
        calendar_routines = [
            r for r in routines
            if r.get("trigger", "") == "after_calendar_meeting"
        ]
        assert len(calendar_routines) >= 1, (
            "3 distinct days of fallback-derived trigger should meet min_occurrences=3"
        )

    def test_fallback_preserves_existing_event_interaction_types(
        self, db, event_store, user_model_store
    ):
        """Episodes with existing non-null, non-placeholder interaction_type
        should use their existing type in the fallback, not re-derive.
        """
        detector = RoutineDetector(db, user_model_store)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)

        ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        # Store episode with existing interaction_type
        user_model_store.store_episode({
            "id": str(uuid.uuid4()),
            "timestamp": ts,
            "event_id": str(uuid.uuid4()),
            "interaction_type": "video_call",
            "content_summary": "Video call",
        })

        results = detector._fallback_event_triggered_episodes(cutoff)

        assert len(results) >= 1
        itype, _ = results[0]
        assert itype == "video_call"  # Original type preserved
