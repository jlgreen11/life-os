"""Tests for the calendar conflict detection service."""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.conflict_detector import ConflictDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _future_iso(hours_from_now: float) -> str:
    """Return an ISO 8601 UTC timestamp *hours_from_now* in the future."""
    return (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).isoformat()


def _past_iso(hours_ago: float) -> str:
    """Return an ISO 8601 UTC timestamp *hours_ago* in the past."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _insert_calendar_event(
    db,
    *,
    summary: str = "Test Event",
    start_time: str | None = None,
    end_time: str | None = None,
    is_all_day: bool = False,
    event_id: str | None = None,
    timestamp: str | None = None,
) -> str:
    """Insert a calendar.event.created row into events.db and return its ID.

    Uses the same schema that the real event pipeline writes.
    """
    eid = event_id or str(uuid.uuid4())
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    payload: dict = {"summary": summary}
    if start_time is not None:
        payload["start_time"] = start_time
    if end_time is not None:
        payload["end_time"] = end_time
    if is_all_day:
        payload["is_all_day"] = True

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                eid,
                "calendar.event.created",
                "google_calendar",
                ts,
                "normal",
                json.dumps(payload),
                json.dumps({}),
            ),
        )
    return eid


# ---------------------------------------------------------------------------
# detect_conflicts() tests
# ---------------------------------------------------------------------------

class TestDetectConflicts:
    """Tests for ConflictDetector.detect_conflicts()."""

    def test_no_events_returns_empty(self, db):
        """An empty calendar should return no conflicts."""
        detector = ConflictDetector(db)
        assert detector.detect_conflicts() == []

    def test_single_event_returns_empty(self, db):
        """A single event cannot conflict with anything."""
        _insert_calendar_event(
            db,
            summary="Solo Meeting",
            start_time=_future_iso(2),
            end_time=_future_iso(3),
        )
        detector = ConflictDetector(db)
        assert detector.detect_conflicts() == []

    def test_non_overlapping_events_no_conflict(self, db):
        """Two events that don't overlap should return no conflicts."""
        _insert_calendar_event(
            db,
            summary="Morning Standup",
            start_time=_future_iso(1),
            end_time=_future_iso(1.5),
        )
        _insert_calendar_event(
            db,
            summary="Afternoon Review",
            start_time=_future_iso(6),
            end_time=_future_iso(7),
        )
        detector = ConflictDetector(db)
        assert detector.detect_conflicts() == []

    def test_overlapping_events_detected(self, db):
        """Two events that overlap should be detected as a conflict."""
        id_a = _insert_calendar_event(
            db,
            summary="Team Sync",
            start_time=_future_iso(2),
            end_time=_future_iso(4),
        )
        id_b = _insert_calendar_event(
            db,
            summary="1:1 with Manager",
            start_time=_future_iso(3),
            end_time=_future_iso(5),
        )
        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()

        assert len(conflicts) == 1
        c = conflicts[0]
        assert {c["event_a_id"], c["event_b_id"]} == {id_a, id_b}
        # Overlap is 1 hour (from +3h to +4h)
        assert abs(c["overlap_minutes"] - 60.0) < 1.0
        assert c["event_a_summary"] in ("Team Sync", "1:1 with Manager")
        assert c["event_b_summary"] in ("Team Sync", "1:1 with Manager")

    def test_back_to_back_events_not_conflict(self, db):
        """Events where one ends exactly when the other starts are NOT conflicts."""
        _insert_calendar_event(
            db,
            summary="First Meeting",
            start_time=_future_iso(2),
            end_time=_future_iso(3),
        )
        _insert_calendar_event(
            db,
            summary="Second Meeting",
            start_time=_future_iso(3),
            end_time=_future_iso(4),
        )
        detector = ConflictDetector(db)
        assert detector.detect_conflicts() == []

    def test_no_self_conflict(self, db):
        """An event should never conflict with itself."""
        eid = str(uuid.uuid4())
        _insert_calendar_event(
            db,
            event_id=eid,
            summary="Only Event",
            start_time=_future_iso(2),
            end_time=_future_iso(3),
        )
        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()
        # Only one event → can't self-conflict
        assert len(conflicts) == 0

    def test_multiple_overlapping_events(self, db):
        """Three events overlapping pairwise should yield three conflicts."""
        _insert_calendar_event(
            db,
            summary="Event A",
            start_time=_future_iso(2),
            end_time=_future_iso(6),
        )
        _insert_calendar_event(
            db,
            summary="Event B",
            start_time=_future_iso(4),
            end_time=_future_iso(8),
        )
        _insert_calendar_event(
            db,
            summary="Event C",
            start_time=_future_iso(5),
            end_time=_future_iso(7),
        )
        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()
        # A-B overlap, A-C overlap, B-C overlap
        assert len(conflicts) == 3

    def test_missing_start_time_handled_gracefully(self, db):
        """Events with missing start_time should be skipped, not crash."""
        _insert_calendar_event(
            db,
            summary="Has Times",
            start_time=_future_iso(2),
            end_time=_future_iso(3),
        )
        # Insert event with no start_time
        _insert_calendar_event(
            db,
            summary="No Start",
            end_time=_future_iso(3),
        )
        detector = ConflictDetector(db)
        # Should not crash, and only one parseable event → no conflicts
        assert detector.detect_conflicts() == []

    def test_missing_end_time_handled_gracefully(self, db):
        """Events with missing end_time should be skipped, not crash."""
        _insert_calendar_event(
            db,
            summary="Has Times",
            start_time=_future_iso(2),
            end_time=_future_iso(3),
        )
        _insert_calendar_event(
            db,
            summary="No End",
            start_time=_future_iso(2),
        )
        detector = ConflictDetector(db)
        assert detector.detect_conflicts() == []

    def test_all_day_events_excluded(self, db):
        """All-day events should not trigger conflicts (they're informational markers)."""
        _insert_calendar_event(
            db,
            summary="Holiday",
            start_time=_future_iso(2),
            end_time=_future_iso(26),
            is_all_day=True,
        )
        _insert_calendar_event(
            db,
            summary="Birthday",
            start_time=_future_iso(2),
            end_time=_future_iso(26),
            is_all_day=True,
        )
        detector = ConflictDetector(db)
        assert detector.detect_conflicts() == []

    def test_overlap_minutes_calculated_correctly(self, db):
        """Overlap should be calculated as the intersection duration."""
        # 3-hour meeting from +2h to +5h
        _insert_calendar_event(
            db,
            summary="Long Meeting",
            start_time=_future_iso(2),
            end_time=_future_iso(5),
        )
        # 45-min meeting from +4h to +4.75h — overlaps from +4h to +4.75h = 45 min
        _insert_calendar_event(
            db,
            summary="Short Meeting",
            start_time=_future_iso(4),
            end_time=_future_iso(4.75),
        )
        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()
        assert len(conflicts) == 1
        assert abs(conflicts[0]["overlap_minutes"] - 45.0) < 1.0

    def test_event_fully_contained_in_another(self, db):
        """An event entirely inside another should be detected as a conflict."""
        _insert_calendar_event(
            db,
            summary="All-Day Workshop",
            start_time=_future_iso(1),
            end_time=_future_iso(10),
        )
        _insert_calendar_event(
            db,
            summary="Lunch Meeting",
            start_time=_future_iso(5),
            end_time=_future_iso(6),
        )
        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()
        assert len(conflicts) == 1
        assert abs(conflicts[0]["overlap_minutes"] - 60.0) < 1.0

    def test_alternative_field_names(self, db):
        """Events using 'start'/'end' instead of 'start_time'/'end_time' should work."""
        eid1 = str(uuid.uuid4())
        eid2 = str(uuid.uuid4())
        start1 = _future_iso(2)
        end1 = _future_iso(3)
        start2 = _future_iso(2.5)
        end2 = _future_iso(3.5)

        with db.get_connection("events") as conn:
            for eid, summary, start, end in [
                (eid1, "Event With Start", start1, end1),
                (eid2, "Overlapping", start2, end2),
            ]:
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        eid,
                        "calendar.event.created",
                        "caldav",
                        datetime.now(timezone.utc).isoformat(),
                        "normal",
                        json.dumps({"summary": summary, "start": start, "end": end}),
                        json.dumps({}),
                    ),
                )

        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()
        assert len(conflicts) == 1

    def test_double_encoded_json_payload(self, db):
        """Payloads that are double-JSON-encoded should still be parsed."""
        eid1 = str(uuid.uuid4())
        eid2 = str(uuid.uuid4())
        start1 = _future_iso(2)
        end1 = _future_iso(3)
        start2 = _future_iso(2.5)
        end2 = _future_iso(3.5)

        with db.get_connection("events") as conn:
            for eid, summary, start, end in [
                (eid1, "Double A", start1, end1),
                (eid2, "Double B", start2, end2),
            ]:
                # Double-encode: json.dumps(json.dumps(...))
                inner = json.dumps({"summary": summary, "start_time": start, "end_time": end})
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        eid,
                        "calendar.event.created",
                        "google_calendar",
                        datetime.now(timezone.utc).isoformat(),
                        "normal",
                        json.dumps(inner),  # double-encoded
                        json.dumps({}),
                    ),
                )

        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()
        assert len(conflicts) == 1


# ---------------------------------------------------------------------------
# Forward-looking window tests (regression tests for the query strategy fix)
# ---------------------------------------------------------------------------

class TestForwardLookingWindow:
    """Tests verifying the forward-looking query strategy works correctly.

    The original implementation filtered by sync timestamp (``WHERE timestamp > cutoff``),
    which missed conflicts between events synced at different times. The fix queries all
    calendar events and filters by start_time in the upcoming window.
    """

    def test_detects_conflict_for_old_synced_events(self, db):
        """Events synced days ago should still be checked for upcoming conflicts.

        This is the core regression test: two overlapping events synced 3 days apart
        but both starting within the next 24 hours should be detected.
        """
        three_days_ago = _past_iso(72)
        one_day_ago = _past_iso(24)

        id_a = _insert_calendar_event(
            db,
            summary="Planning Review",
            start_time=_future_iso(4),
            end_time=_future_iso(5),
            timestamp=three_days_ago,  # Synced 3 days ago
        )
        id_b = _insert_calendar_event(
            db,
            summary="Sprint Demo",
            start_time=_future_iso(4.5),
            end_time=_future_iso(5.5),
            timestamp=one_day_ago,  # Synced 1 day ago
        )

        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()

        assert len(conflicts) == 1
        assert {conflicts[0]["event_a_id"], conflicts[0]["event_b_id"]} == {id_a, id_b}

    def test_skips_past_events(self, db):
        """Events whose end_time is in the past should not be considered."""
        _insert_calendar_event(
            db,
            summary="Yesterday's Standup",
            start_time=_past_iso(25),
            end_time=_past_iso(24.5),
        )
        _insert_calendar_event(
            db,
            summary="Yesterday's Review",
            start_time=_past_iso(25),
            end_time=_past_iso(24.5),
        )

        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()
        assert conflicts == []

    def test_skips_far_future_events(self, db):
        """Events starting beyond the 48h window should not be considered."""
        _insert_calendar_event(
            db,
            summary="Next Week Planning",
            start_time=_future_iso(72),
            end_time=_future_iso(73),
        )
        _insert_calendar_event(
            db,
            summary="Next Week Review",
            start_time=_future_iso(72.5),
            end_time=_future_iso(73.5),
        )

        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()
        assert conflicts == []

    def test_custom_forward_window(self, db):
        """The forward_hours parameter should control the look-ahead window size."""
        # Events at +72h — outside default 48h window but inside 96h window
        _insert_calendar_event(
            db,
            summary="Far Meeting A",
            start_time=_future_iso(72),
            end_time=_future_iso(73),
        )
        _insert_calendar_event(
            db,
            summary="Far Meeting B",
            start_time=_future_iso(72.5),
            end_time=_future_iso(73.5),
        )

        detector = ConflictDetector(db)

        # Default 48h window should NOT find them
        assert detector.detect_conflicts(forward_hours=48) == []

        # 96h window SHOULD find them
        conflicts = detector.detect_conflicts(forward_hours=96)
        assert len(conflicts) == 1

    def test_event_currently_in_progress_included(self, db):
        """An event that started in the past but hasn't ended yet should be included."""
        id_a = _insert_calendar_event(
            db,
            summary="Ongoing Workshop",
            start_time=_past_iso(1),
            end_time=_future_iso(2),
        )
        id_b = _insert_calendar_event(
            db,
            summary="Overlapping Call",
            start_time=_future_iso(1),
            end_time=_future_iso(3),
        )

        detector = ConflictDetector(db)
        conflicts = detector.detect_conflicts()

        assert len(conflicts) == 1
        assert {conflicts[0]["event_a_id"], conflicts[0]["event_b_id"]} == {id_a, id_b}


# ---------------------------------------------------------------------------
# check_and_publish() tests
# ---------------------------------------------------------------------------

class TestCheckAndPublish:
    """Tests for ConflictDetector.check_and_publish()."""

    @pytest.mark.asyncio
    async def test_publishes_conflict_events(self, db, event_bus):
        """Detected conflicts should be published to the event bus."""
        _insert_calendar_event(
            db,
            summary="Meeting A",
            start_time=_future_iso(2),
            end_time=_future_iso(3),
        )
        _insert_calendar_event(
            db,
            summary="Meeting B",
            start_time=_future_iso(2.5),
            end_time=_future_iso(3.5),
        )

        detector = ConflictDetector(db)
        count = await detector.check_and_publish(event_bus)

        assert count == 1
        event_bus.publish.assert_called()
        assert event_bus.publish.call_count == 1

        # Verify the published event structure
        call_args = event_bus.publish.call_args_list[0]
        assert call_args[0][0] == "calendar.conflict.detected"
        payload = call_args[0][1]
        assert "event_a_id" in payload
        assert "event_b_id" in payload
        assert "overlap_minutes" in payload

    @pytest.mark.asyncio
    async def test_no_conflicts_no_publish(self, db, event_bus):
        """When there are no conflicts, nothing should be published."""
        _insert_calendar_event(
            db,
            summary="Morning",
            start_time=_future_iso(1),
            end_time=_future_iso(2),
        )
        _insert_calendar_event(
            db,
            summary="Afternoon",
            start_time=_future_iso(6),
            end_time=_future_iso(7),
        )

        detector = ConflictDetector(db)
        count = await detector.check_and_publish(event_bus)

        assert count == 0
        event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplication_across_runs(self, db, event_bus):
        """The same conflict should not be published twice across multiple runs."""
        _insert_calendar_event(
            db,
            summary="Meeting A",
            start_time=_future_iso(2),
            end_time=_future_iso(3),
        )
        _insert_calendar_event(
            db,
            summary="Meeting B",
            start_time=_future_iso(2.5),
            end_time=_future_iso(3.5),
        )

        detector = ConflictDetector(db)

        first_count = await detector.check_and_publish(event_bus)
        assert first_count == 1

        event_bus.publish.reset_mock()
        second_count = await detector.check_and_publish(event_bus)
        assert second_count == 0
        event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_uses_high_priority(self, db, event_bus):
        """Conflict events should be published with high priority."""
        _insert_calendar_event(
            db,
            summary="A",
            start_time=_future_iso(2),
            end_time=_future_iso(3),
        )
        _insert_calendar_event(
            db,
            summary="B",
            start_time=_future_iso(2.5),
            end_time=_future_iso(3.5),
        )

        detector = ConflictDetector(db)
        await detector.check_and_publish(event_bus)

        call_kwargs = event_bus.publish.call_args_list[0][1]
        assert call_kwargs["source"] == "conflict_detector"
        assert call_kwargs["priority"] == "high"


# ---------------------------------------------------------------------------
# _parse_datetime() tests
# ---------------------------------------------------------------------------

class TestParseDatetime:
    """Tests for the datetime parsing helper."""

    def test_parse_z_suffix(self):
        """ISO timestamps with 'Z' suffix should parse correctly."""
        dt = ConflictDetector._parse_datetime("2026-03-02T10:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.hour == 10

    def test_parse_offset_format(self):
        """ISO timestamps with +00:00 offset should parse correctly."""
        dt = ConflictDetector._parse_datetime("2026-03-02T10:00:00+00:00")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_parse_date_only(self):
        """Date-only strings should parse as midnight UTC."""
        dt = ConflictDetector._parse_datetime("2026-03-02")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.hour == 0
        assert dt.minute == 0

    def test_parse_invalid_returns_none(self):
        """Invalid strings should return None, not raise."""
        assert ConflictDetector._parse_datetime("not-a-date") is None
        assert ConflictDetector._parse_datetime("") is None

    def test_parse_none_returns_none(self):
        """None input should return None (AttributeError caught)."""
        assert ConflictDetector._parse_datetime(None) is None
