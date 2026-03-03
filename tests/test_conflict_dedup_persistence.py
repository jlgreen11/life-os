"""Tests for conflict dedup persistence across restarts.

Verifies that both ConflictDetector and CalDAVConnector persist their
published-conflict dedup sets to state.db so that process restarts don't
trigger duplicate conflict notifications.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from connectors.caldav.connector import CalDAVConnector
from services.conflict_detector import ConflictDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _future_iso(hours_from_now: float) -> str:
    """Return an ISO 8601 UTC timestamp *hours_from_now* in the future."""
    return (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).isoformat()


def _insert_calendar_event(
    db,
    *,
    summary: str = "Test Event",
    start_time: str | None = None,
    end_time: str | None = None,
    is_all_day: bool = False,
    event_id: str | None = None,
) -> str:
    """Insert a calendar.event.created row into events.db and return its ID."""
    eid = event_id or str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    payload: dict = {"summary": summary, "title": summary}
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


def _insert_published_conflict(db, id_a: str, id_b: str, source: str = "conflict_detector", detected_at: str | None = None):
    """Directly insert a row into the published_conflicts table."""
    ids = sorted([id_a, id_b])
    ts = detected_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT OR IGNORE INTO published_conflicts (event_id_a, event_id_b, detected_at, source) VALUES (?, ?, ?, ?)",
            (ids[0], ids[1], ts, source),
        )


def _count_published_conflicts(db, source: str | None = None) -> int:
    """Count rows in the published_conflicts table, optionally filtered by source."""
    with db.get_connection("state") as conn:
        if source:
            row = conn.execute(
                "SELECT COUNT(*) FROM published_conflicts WHERE source = ?", (source,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM published_conflicts").fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# ConflictDetector persistence tests
# ---------------------------------------------------------------------------

class TestConflictDetectorPersistence:
    """Tests for ConflictDetector persisting dedup state to state.db."""

    async def test_conflict_persisted_to_db(self, db, event_bus):
        """Publishing a conflict via check_and_publish should persist it to state.db."""
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
        # No conflicts persisted yet
        assert _count_published_conflicts(db, source="conflict_detector") == 0

        count = await detector.check_and_publish(event_bus)

        assert count == 1
        # Conflict should now be persisted in state.db
        assert _count_published_conflicts(db, source="conflict_detector") == 1

    def test_persisted_conflicts_loaded_on_init(self, db):
        """Pre-populated conflict pairs should be loaded into memory on init."""
        # Pre-populate 3 conflict pairs
        _insert_published_conflict(db, "evt-a", "evt-b", source="conflict_detector")
        _insert_published_conflict(db, "evt-c", "evt-d", source="conflict_detector")
        _insert_published_conflict(db, "evt-e", "evt-f", source="conflict_detector")

        detector = ConflictDetector(db)

        assert len(detector._published_conflicts) == 3
        assert frozenset(["evt-a", "evt-b"]) in detector._published_conflicts
        assert frozenset(["evt-c", "evt-d"]) in detector._published_conflicts
        assert frozenset(["evt-e", "evt-f"]) in detector._published_conflicts

    async def test_persisted_conflict_prevents_republish(self, db, event_bus):
        """A conflict pair already in state.db should NOT be re-published."""
        id_a = _insert_calendar_event(
            db,
            summary="Meeting A",
            start_time=_future_iso(2),
            end_time=_future_iso(3),
        )
        id_b = _insert_calendar_event(
            db,
            summary="Meeting B",
            start_time=_future_iso(2.5),
            end_time=_future_iso(3.5),
        )

        # Pre-populate the conflict pair in state.db (simulating a previous run)
        _insert_published_conflict(db, id_a, id_b, source="conflict_detector")

        # Create a NEW ConflictDetector instance (simulating restart)
        detector = ConflictDetector(db)

        count = await detector.check_and_publish(event_bus)

        # Should not re-publish the already-persisted conflict
        assert count == 0
        event_bus.publish.assert_not_called()

    def test_cleanup_old_conflicts(self, db):
        """cleanup_old_conflicts should remove old pairs but keep recent ones."""
        # Insert a conflict detected 60 days ago
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        _insert_published_conflict(db, "old-a", "old-b", source="conflict_detector", detected_at=old_ts)

        # Insert a recent conflict
        _insert_published_conflict(db, "new-a", "new-b", source="conflict_detector")

        assert _count_published_conflicts(db, source="conflict_detector") == 2

        detector = ConflictDetector(db)
        detector.cleanup_old_conflicts(days=30)

        # Old pair should be gone, recent one should remain
        assert _count_published_conflicts(db, source="conflict_detector") == 1

        with db.get_connection("state") as conn:
            rows = conn.execute(
                "SELECT event_id_a, event_id_b FROM published_conflicts WHERE source = 'conflict_detector'"
            ).fetchall()
            remaining = {frozenset([rows[0][0], rows[0][1]])}
            assert frozenset(["new-a", "new-b"]) in remaining

    def test_only_loads_own_source(self, db):
        """ConflictDetector should only load pairs with source='conflict_detector'."""
        _insert_published_conflict(db, "cd-a", "cd-b", source="conflict_detector")
        _insert_published_conflict(db, "cal-a", "cal-b", source="caldav")

        detector = ConflictDetector(db)

        assert len(detector._published_conflicts) == 1
        assert frozenset(["cd-a", "cd-b"]) in detector._published_conflicts
        assert frozenset(["cal-a", "cal-b"]) not in detector._published_conflicts


# ---------------------------------------------------------------------------
# CalDAV connector persistence tests
# ---------------------------------------------------------------------------

class TestCalDAVConnectorPersistence:
    """Tests for CalDAVConnector persisting conflict dedup state to state.db."""

    def _make_connector(self, db):
        """Create a CalDAVConnector with minimal mock dependencies."""
        mock_event_bus = AsyncMock()
        config = {"url": "https://example.com", "username": "test", "password": "test"}
        return CalDAVConnector(mock_event_bus, db, config)

    def test_caldav_loads_persisted_conflicts(self, db):
        """CalDAVConnector should load published conflicts from state.db on init."""
        _insert_published_conflict(db, "cal-a", "cal-b", source="caldav")
        _insert_published_conflict(db, "cal-c", "cal-d", source="caldav")

        connector = self._make_connector(db)

        assert len(connector._published_conflicts) == 2
        assert frozenset(["cal-a", "cal-b"]) in connector._published_conflicts
        assert frozenset(["cal-c", "cal-d"]) in connector._published_conflicts

    def test_caldav_persists_conflict(self, db):
        """CalDAVConnector._persist_conflict should write to state.db."""
        connector = self._make_connector(db)

        pair = frozenset(["evt-x", "evt-y"])
        connector._persist_conflict(pair)

        assert _count_published_conflicts(db, source="caldav") == 1

        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT event_id_a, event_id_b, source FROM published_conflicts"
            ).fetchone()
            # Should be stored in sorted order
            assert row[0] == "evt-x"
            assert row[1] == "evt-y"
            assert row[2] == "caldav"

    def test_caldav_only_loads_own_source(self, db):
        """CalDAVConnector should only load pairs with source='caldav'."""
        _insert_published_conflict(db, "cd-a", "cd-b", source="conflict_detector")
        _insert_published_conflict(db, "cal-a", "cal-b", source="caldav")

        connector = self._make_connector(db)

        assert len(connector._published_conflicts) == 1
        assert frozenset(["cal-a", "cal-b"]) in connector._published_conflicts
        assert frozenset(["cd-a", "cd-b"]) not in connector._published_conflicts

    async def test_caldav_detect_conflicts_persists(self, db):
        """CalDAV _detect_conflicts should persist new conflict pairs to state.db."""
        # Insert two overlapping calendar events in events.db
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

        connector = self._make_connector(db)

        await connector._detect_conflicts()

        # The conflict pair should be persisted
        assert _count_published_conflicts(db, source="caldav") == 1

    async def test_caldav_persisted_conflict_prevents_republish(self, db):
        """A conflict pair already in state.db should not be re-published by CalDAV."""
        id_a = _insert_calendar_event(
            db,
            summary="Meeting A",
            start_time=_future_iso(2),
            end_time=_future_iso(3),
        )
        id_b = _insert_calendar_event(
            db,
            summary="Meeting B",
            start_time=_future_iso(2.5),
            end_time=_future_iso(3.5),
        )

        # Pre-populate the conflict pair (simulating previous run)
        _insert_published_conflict(db, id_a, id_b, source="caldav")

        connector = self._make_connector(db)

        await connector._detect_conflicts()

        # Still only 1 row (the pre-populated one)
        assert _count_published_conflicts(db, source="caldav") == 1
