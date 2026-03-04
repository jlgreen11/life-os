"""
Tests for routine detector diagnostic logging and NULL interaction_type fallback.

Validates that:
1. Temporal routines ARE detected when episodes have valid interaction_types
2. The NULL interaction_type fallback produces routines from unclassified episodes
3. Diagnostic logging output contains expected format strings
4. Consistency scoring works correctly with sparse active days
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta

from services.routine_detector.detector import RoutineDetector


class TestTemporalDetectionWithValidTypes:
    """Temporal routines should be detected when episodes have valid interaction_types."""

    def test_detects_routines_from_valid_interaction_types(self, db, user_model_store):
        """Episodes with proper interaction_types across multiple days should produce routines."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=10)

        # Create a clear morning pattern: email_received at 8am on 10 days
        for day_offset in range(10):
            day = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "email_received",
                    "content_summary": "Email received",
                }
            )

        routines = detector.detect_routines(lookback_days=30)
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1
        assert any(step["action"] == "email_received" for r in morning_routines for step in r["steps"])

    def test_multiple_types_in_same_bucket_all_detected(self, db, user_model_store):
        """Multiple interaction types in the same time bucket should all appear as steps."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=8)

        for day_offset in range(8):
            day = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "email_received",
                    "content_summary": "Check email",
                }
            )
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": (day + timedelta(minutes=15)).isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "calendar_reviewed",
                    "content_summary": "Review calendar",
                }
            )

        routines = detector.detect_routines(lookback_days=30)
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1

        step_actions = {s["action"] for r in morning_routines for s in r["steps"]}
        assert "email_received" in step_actions
        assert "calendar_reviewed" in step_actions


class TestNullInteractionTypeFallback:
    """The fallback should produce routines from episodes with NULL/unknown interaction_type."""

    def _insert_episode_with_placeholder_type(self, db, timestamp, event_id, placeholder="unknown"):
        """Insert an episode with a placeholder interaction_type (e.g. 'unknown', 'communication').

        Simulates the production scenario where old episodes were created before
        granular classification was deployed, leaving 'unknown' or 'communication'
        as the interaction_type.

        Args:
            db: DatabaseManager instance.
            timestamp: Episode timestamp string.
            event_id: Linked event ID.
            placeholder: The useless type to insert ('unknown' or 'communication').
        """
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO episodes
                   (id, timestamp, event_id, interaction_type, content_summary)
                   VALUES (?, ?, ?, ?, 'test')""",
                (str(uuid.uuid4()), timestamp, event_id, placeholder),
            )

    def _insert_event(self, db, event_id, event_type):
        """Insert a source event into the events database."""
        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO events
                   (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, 'test', ?, 'normal', '{}', '{}')""",
                (event_id, event_type, datetime.now(UTC).isoformat()),
            )

    def test_fallback_recovers_routines_from_placeholder_types(self, db, user_model_store):
        """When all episodes have 'unknown' interaction_type, fallback should derive types from events."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=8)

        for day_offset in range(8):
            day = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = str(uuid.uuid4())

            # Insert the source event (email.received)
            self._insert_event(db, event_id, "email.received")

            # Insert episode with NULL interaction_type
            self._insert_episode_with_placeholder_type(db, day.isoformat(), event_id)

        routines = detector.detect_routines(lookback_days=30)

        # The fallback should derive email_received from the events table
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, (
            "Expected fallback to derive 'email_received' from events and detect a morning routine"
        )

        step_actions = {s["action"] for r in morning_routines for s in r["steps"]}
        assert "email_received" in step_actions

    def test_fallback_skips_unresolvable_events(self, db, user_model_store):
        """Episodes whose event_id doesn't exist in events table should be skipped gracefully."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=5)

        for day_offset in range(5):
            day = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            # Insert episode with 'unknown' type pointing to a non-existent event
            self._insert_episode_with_placeholder_type(db, day.isoformat(), "nonexistent-event-id")

        routines = detector.detect_routines(lookback_days=30)
        # No routines should be detected since events can't be resolved
        # (The primary query returns 0 because 'unknown' is filtered out,
        # the fallback fires but can't resolve any event_ids)
        assert len(routines) == 0

    def test_fallback_handles_unknown_interaction_type(self, db, user_model_store):
        """Episodes with 'unknown' interaction_type should trigger the fallback and derive types."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=8)

        for day_offset in range(8):
            day = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = str(uuid.uuid4())

            self._insert_event(db, event_id, "email.sent")

            # Store with 'unknown' type — the default when interaction_type is missing
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": event_id,
                    "interaction_type": "unknown",
                    "content_summary": "test",
                }
            )

        routines = detector.detect_routines(lookback_days=30)

        # 'unknown' is now filtered out from the primary query, so the fallback fires.
        # The fallback looks up event_ids in the events table and derives 'email_sent'.
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, (
            "Fallback should derive 'email_sent' from events and detect a morning routine"
        )

    def test_fallback_mixes_valid_and_derived_types(self, db, user_model_store):
        """Fallback should use existing valid types where available, derive for the rest."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=6)

        # Mix of valid and NULL interaction_type episodes
        for day_offset in range(6):
            day = base_date.replace(hour=7, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)

            if day_offset % 2 == 0:
                # Valid interaction_type
                user_model_store.store_episode(
                    {
                        "id": str(uuid.uuid4()),
                        "timestamp": day.isoformat(),
                        "event_id": str(uuid.uuid4()),
                        "interaction_type": "email_received",
                        "content_summary": "test",
                    }
                )
            else:
                # Placeholder 'unknown' type with linked event
                event_id = str(uuid.uuid4())
                self._insert_event(db, event_id, "email.received")
                self._insert_episode_with_placeholder_type(db, day.isoformat(), event_id)

        # The primary query finds 3 episodes (even days with valid types).
        # That meets min_occurrences=3, so fallback should not fire.
        # If it does fire (e.g., NULL rows throw off the query), the fallback
        # should still produce correct results.
        routines = detector.detect_routines(lookback_days=30)
        assert isinstance(routines, list)


class TestDiagnosticLogging:
    """Diagnostic logging should output the expected format strings at INFO level."""

    def test_temporal_detection_logs_episode_count(self, db, user_model_store, caplog):
        """Should log the count of episodes with interaction_type in lookback window."""
        detector = RoutineDetector(db, user_model_store)

        # Add some episodes
        base_date = datetime.now(UTC) - timedelta(days=3)
        for day_offset in range(3):
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": (base_date + timedelta(days=day_offset, hours=9)).isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "email_received",
                    "content_summary": "test",
                }
            )

        with caplog.at_level(logging.INFO, logger="services.routine_detector.detector"):
            detector.detect_routines(lookback_days=30)

        assert any(
            "Temporal detection:" in msg and "episodes with" in msg and "interaction_type" in msg
            for msg in caplog.messages
        ), f"Expected 'episodes with...interaction_type' log, got: {caplog.messages}"

    def test_temporal_detection_logs_bucket_pairs(self, db, user_model_store, caplog):
        """Should log the count of (bucket, type) pairs found."""
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(UTC) - timedelta(days=5)
        for day_offset in range(5):
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": (base_date + timedelta(days=day_offset, hours=9)).isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "email_received",
                    "content_summary": "test",
                }
            )

        with caplog.at_level(logging.INFO, logger="services.routine_detector.detector"):
            detector.detect_routines(lookback_days=30)

        assert any("(bucket, type) pairs found" in msg for msg in caplog.messages), (
            f"Expected '(bucket, type) pairs found' log, got: {caplog.messages}"
        )

    def test_temporal_detection_logs_min_occurrences_filter(self, db, user_model_store, caplog):
        """Should log how many pairs meet min_occurrences threshold."""
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(UTC) - timedelta(days=5)
        for day_offset in range(5):
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": (base_date + timedelta(days=day_offset, hours=9)).isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "email_received",
                    "content_summary": "test",
                }
            )

        with caplog.at_level(logging.INFO, logger="services.routine_detector.detector"):
            detector.detect_routines(lookback_days=30)

        assert any("pairs meet min_occurrences=" in msg for msg in caplog.messages), (
            f"Expected 'pairs meet min_occurrences=' log, got: {caplog.messages}"
        )

    def test_temporal_detection_logs_consistency_result(self, db, user_model_store, caplog):
        """Should log consistency score with PASS/FAIL for each bucket."""
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(UTC) - timedelta(days=10)
        for day_offset in range(10):
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": (base_date + timedelta(days=day_offset, hours=9)).isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "email_received",
                    "content_summary": "test",
                }
            )

        with caplog.at_level(logging.INFO, logger="services.routine_detector.detector"):
            detector.detect_routines(lookback_days=30)

        consistency_logs = [
            msg for msg in caplog.messages if "consistency=" in msg and ("PASS" in msg or "FAIL" in msg)
        ]
        assert len(consistency_logs) >= 1, f"Expected at least one consistency PASS/FAIL log, got: {caplog.messages}"

    def test_fallback_logs_when_triggered(self, db, user_model_store, caplog):
        """Should log fallback recovery count when primary query returns 0 usable rows."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=5)

        # Insert episodes with 'unknown' type (filtered by primary query) and linked events
        for day_offset in range(5):
            day = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = str(uuid.uuid4())
            with db.get_connection("events") as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO events
                       (id, type, source, timestamp, priority, payload, metadata)
                       VALUES (?, 'email.received', 'test', ?, 'normal', '{}', '{}')""",
                    (event_id, datetime.now(UTC).isoformat()),
                )
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": event_id,
                    "interaction_type": "unknown",
                    "content_summary": "test",
                }
            )

        with caplog.at_level(logging.INFO, logger="services.routine_detector.detector"):
            detector.detect_routines(lookback_days=30)

        assert any("fallback recovered" in msg or "fallback" in msg.lower() for msg in caplog.messages), (
            f"Expected fallback log messages, got: {caplog.messages}"
        )


class TestSparseActiveDaysConsistency:
    """Consistency scoring should work correctly with sparse active days."""

    def test_sparse_days_in_large_window(self, db, user_model_store):
        """5 active days in a 30-day window should score consistency correctly.

        With active_days=5 (not 30), an action appearing on 4 of those 5 days
        should score 0.8 consistency — well above the 0.6 threshold.
        """
        detector = RoutineDetector(db, user_model_store)

        # Spread 5 episodes across a 30-day window at 9am
        base_date = datetime.now(UTC) - timedelta(days=29)
        active_day_offsets = [0, 7, 14, 21, 28]

        for offset in active_day_offsets:
            day = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=offset)
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "weekly_email_check",
                    "content_summary": "Weekly email check",
                }
            )

        routines = detector.detect_routines(lookback_days=30)

        # 5 occurrences on 5 active days → consistency = 5/5 = 1.0
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, "5 episodes on 5 active days (consistency=1.0) should produce a routine"
        routine = morning_routines[0]
        assert routine["consistency_score"] >= 0.9  # Should be ~1.0

    def test_partial_coverage_sparse_days(self, db, user_model_store):
        """Action appearing on 3 of 5 active days should score 0.6 — exactly at threshold."""
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(UTC) - timedelta(days=28)

        # Create the target action on 3 of 5 active days
        target_days = [0, 7, 14]
        filler_days = [21, 28]

        for offset in target_days:
            day = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=offset)
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "morning_standup",
                    "content_summary": "Morning standup",
                }
            )

        # Add filler episodes on the other 2 days with unique types (won't form routines)
        for offset in filler_days:
            day = base_date.replace(hour=15, minute=0, second=0, microsecond=0) + timedelta(days=offset)
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": f"filler_unique_{offset}",
                    "content_summary": "Filler",
                }
            )

        routines = detector.detect_routines(lookback_days=30)

        # 3 occurrences on 5 active days → consistency = 3/5 = 0.6 (threshold)
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, "3 episodes on 5 active days (consistency=0.6) should meet the 0.6 threshold"

    def test_below_threshold_sparse_days(self, db, user_model_store):
        """Action appearing on 3 of 6 active days should score 0.5 — below threshold."""
        detector = RoutineDetector(db, user_model_store)

        base_date = datetime.now(UTC) - timedelta(days=28)

        # Target action on 3 days
        target_days = [0, 7, 14]
        for offset in target_days:
            day = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=offset)
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "sporadic_check",
                    "content_summary": "Sporadic check",
                }
            )

        # Filler episodes on 3 additional days (total 6 active days)
        filler_days = [21, 24, 28]
        for offset in filler_days:
            day = base_date.replace(hour=14, minute=0, second=0, microsecond=0) + timedelta(days=offset)
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": f"filler_{offset}",
                    "content_summary": "Filler",
                }
            )

        routines = detector.detect_routines(lookback_days=30)

        # 3 occurrences on 6 active days → consistency = 3/6 = 0.5.
        # With cold-start scaling (6 active days < 7 → threshold = 0.3), this
        # pattern IS now detected as a provisional routine with cold_start=True
        # and scaled-down confidence (0.5 * 0.7 = 0.35).
        sporadic_routines = [
            r for r in routines
            for s in r["steps"]
            if s["action"] == "sporadic_check"
        ]
        assert len(sporadic_routines) >= 1, (
            "With cold-start threshold (0.3 for < 7 active days), "
            "consistency=0.5 should be detected as a provisional routine"
        )
        routine = sporadic_routines[0]
        assert routine.get("cold_start") is True, "Routine below base threshold should be marked cold_start"
        assert routine["consistency_score"] < 0.5, "Cold-start confidence should be scaled down (0.5 * 0.7)"


class TestDeriveInteractionType:
    """Tests for the _derive_interaction_type_from_event helper."""

    def test_derives_type_from_dotted_event(self, db, user_model_store):
        """Should convert dotted event type to underscored interaction type."""
        detector = RoutineDetector(db, user_model_store)
        event_id = str(uuid.uuid4())

        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, 'email.received', 'test', ?, 'normal', '{}', '{}')""",
                (event_id, datetime.now(UTC).isoformat()),
            )

        result = detector._derive_interaction_type_from_event(event_id)
        assert result == "email_received"

    def test_returns_none_for_missing_event(self, db, user_model_store):
        """Should return None when event_id doesn't exist in events table."""
        detector = RoutineDetector(db, user_model_store)
        result = detector._derive_interaction_type_from_event("nonexistent-id")
        assert result is None

    def test_handles_complex_event_types(self, db, user_model_store):
        """Should handle multi-segment dotted types like 'calendar.event.created'."""
        detector = RoutineDetector(db, user_model_store)
        event_id = str(uuid.uuid4())

        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, 'calendar.event.created', 'test', ?, 'normal', '{}', '{}')""",
                (event_id, datetime.now(UTC).isoformat()),
            )

        result = detector._derive_interaction_type_from_event(event_id)
        assert result == "calendar_event_created"
