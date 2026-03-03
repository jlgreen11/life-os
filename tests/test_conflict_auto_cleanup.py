"""Tests for automatic daily cleanup of stale published_conflicts.

Verifies that ConflictDetector.check_and_publish() triggers cleanup of
conflict pairs older than 30 days, running at most once per 24 hours,
and that cleanup failures never block conflict detection.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.conflict_detector import ConflictDetector


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_conflict_dedup_persistence.py)
# ---------------------------------------------------------------------------


def _future_iso(hours_from_now: float) -> str:
    """Return an ISO 8601 UTC timestamp *hours_from_now* in the future."""
    return (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).isoformat()


def _insert_calendar_event(db, *, summary="Test Event", start_time=None, end_time=None) -> str:
    """Insert a calendar.event.created row into events.db and return its ID."""
    eid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    payload = {"summary": summary, "title": summary}
    if start_time is not None:
        payload["start_time"] = start_time
    if end_time is not None:
        payload["end_time"] = end_time
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (eid, "calendar.event.created", "google_calendar", ts, "normal", json.dumps(payload), json.dumps({})),
        )
    return eid


def _insert_published_conflict(db, id_a, id_b, *, detected_at=None):
    """Insert a conflict_detector row into published_conflicts."""
    ids = sorted([id_a, id_b])
    ts = detected_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT OR IGNORE INTO published_conflicts (event_id_a, event_id_b, detected_at, source) VALUES (?, ?, ?, ?)",
            (ids[0], ids[1], ts, "conflict_detector"),
        )


def _count_published_conflicts(db) -> int:
    """Count conflict_detector rows in the published_conflicts table."""
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM published_conflicts WHERE source = 'conflict_detector'"
        ).fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConflictAutoCleanup:
    """Tests for automatic cleanup of stale conflicts during check_and_publish."""

    async def test_cleanup_called_on_first_invocation(self, db, event_bus):
        """First call to check_and_publish() should trigger cleanup, removing old conflicts."""
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        _insert_published_conflict(db, "old-a", "old-b", detected_at=old_ts)
        _insert_published_conflict(db, "old-c", "old-d", detected_at=old_ts)

        # Insert a recent conflict that should survive cleanup
        _insert_published_conflict(db, "new-a", "new-b")

        assert _count_published_conflicts(db) == 3

        detector = ConflictDetector(db)
        assert detector._last_cleanup is None

        await detector.check_and_publish(event_bus)

        # Old conflicts should be deleted, recent one should remain
        assert _count_published_conflicts(db) == 1
        assert detector._last_cleanup is not None

    async def test_cleanup_not_called_within_24h(self, db, event_bus):
        """Cleanup should not run again within 24 hours of the last run."""
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        _insert_published_conflict(db, "old-a", "old-b", detected_at=old_ts)

        detector = ConflictDetector(db)

        # First call triggers cleanup
        await detector.check_and_publish(event_bus)
        first_cleanup_time = detector._last_cleanup
        assert first_cleanup_time is not None
        assert _count_published_conflicts(db) == 0

        # Insert another old conflict
        _insert_published_conflict(db, "old-c", "old-d", detected_at=old_ts)
        assert _count_published_conflicts(db) == 1

        # Second call should NOT trigger cleanup (within 24h)
        await detector.check_and_publish(event_bus)

        # The old conflict should still be there (cleanup didn't run)
        assert _count_published_conflicts(db) == 1
        # _last_cleanup should be unchanged
        assert detector._last_cleanup == first_cleanup_time

    async def test_cleanup_called_after_24h(self, db, event_bus):
        """Cleanup should run again once 24 hours have elapsed since the last run."""
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        detector = ConflictDetector(db)

        # Simulate that cleanup ran 25 hours ago
        detector._last_cleanup = datetime.now(timezone.utc) - timedelta(hours=25)

        # Insert an old conflict that should be cleaned up
        _insert_published_conflict(db, "old-a", "old-b", detected_at=old_ts)
        assert _count_published_conflicts(db) == 1

        await detector.check_and_publish(event_bus)

        # Cleanup should have run and removed the old conflict
        assert _count_published_conflicts(db) == 0
        # _last_cleanup should be updated to roughly now
        assert (datetime.now(timezone.utc) - detector._last_cleanup).total_seconds() < 5

    async def test_cleanup_error_does_not_block_detection(self, db, event_bus):
        """If cleanup_old_conflicts raises, check_and_publish should still detect conflicts."""
        # Insert two overlapping events so there's a conflict to detect
        _insert_calendar_event(
            db, summary="Meeting A", start_time=_future_iso(2), end_time=_future_iso(3)
        )
        _insert_calendar_event(
            db, summary="Meeting B", start_time=_future_iso(2.5), end_time=_future_iso(3.5)
        )

        detector = ConflictDetector(db)

        with patch.object(detector, "cleanup_old_conflicts", side_effect=RuntimeError("DB locked")):
            count = await detector.check_and_publish(event_bus)

        # Conflict detection should still have worked despite cleanup failure
        assert count == 1
        event_bus.publish.assert_called()

    def test_cleanup_returns_count(self, db):
        """cleanup_old_conflicts should return the number of rows deleted."""
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Insert 5 old conflicts
        for i in range(5):
            _insert_published_conflict(db, f"old-a-{i}", f"old-b-{i}", detected_at=old_ts)

        # Insert 3 recent conflicts
        for i in range(3):
            _insert_published_conflict(db, f"new-a-{i}", f"new-b-{i}")

        assert _count_published_conflicts(db) == 8

        detector = ConflictDetector(db)
        removed = detector.cleanup_old_conflicts(days=30)

        assert removed == 5
        assert _count_published_conflicts(db) == 3
