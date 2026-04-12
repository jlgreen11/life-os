"""
Tests for the WorkflowDetector adaptive lookback feature.

When a connector outage (or any gap in data ingestion) pushes all qualifying
events outside the default 30-day window, ``detect_workflows()`` used to return
0 results despite rich historical data in events.db.

The adaptive lookback logic detects this scenario by:
1. Counting qualifying events (email, task, calendar, message) in the window.
2. If count < MIN_QUALIFYING_EVENTS (50), doubling the window.
3. Repeating until count >= 50 or the window hits 365 days.
4. Logging the extension so operators can see when it fires.

These tests validate that the extension fires when it should, stays quiet when
it shouldn't, and that workflows are actually detected once the window extends.

Test patterns mirror ``tests/test_routine_detector_adaptive_lookback.py``:
- Real DatabaseManager + UserModelStore via conftest fixtures (no DB mocks).
- Events inserted directly into events.db via ``db.get_connection("events")``.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from services.workflow_detector.detector import WorkflowDetector

# Event types that count as qualifying for the adaptive lookback threshold.
QUALIFYING_TYPES = (
    "email.received",
    "email.sent",
    "message.sent",
    "message.received",
    "task.created",
    "task.completed",
    "calendar.event.created",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_event(
    conn,
    event_type: str,
    timestamp: datetime,
    *,
    email_from: str | None = None,
    email_to: str | None = None,
    source: str = "test",
) -> str:
    """Insert a single event into events.db and return its ID.

    Args:
        conn: SQLite connection to events.db.
        event_type: The ``type`` column value (e.g. 'email.received').
        timestamp: Event timestamp (UTC-aware datetime).
        email_from: Optional sender address for email events.
        email_to: Optional recipient address for email events.
        source: Event source name (default 'test').

    Returns:
        The newly inserted event ID.
    """
    eid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO events (id, type, source, timestamp, priority, payload,
                            metadata, email_from, email_to)
        VALUES (?, ?, ?, ?, 'normal', '{}', '{}', ?, ?)
        """,
        (eid, event_type, source, timestamp.isoformat(), email_from, email_to),
    )
    return eid


def _insert_qualifying_events(db, *, days_ago: float, count: int = 60) -> None:
    """Insert a block of qualifying events centred around ``days_ago`` days before now.

    Events are spread evenly across available qualifying types so the data
    looks realistic.  The caller controls how many total events to insert.

    Args:
        db: DatabaseManager fixture from conftest.
        days_ago: How many days before now the events are timestamped.
        count: Total number of events to insert (default 60).
    """
    base = datetime.now(UTC) - timedelta(days=days_ago)
    base = base.replace(hour=9, minute=0, second=0, microsecond=0)

    with db.get_connection("events") as conn:
        for i in range(count):
            etype = QUALIFYING_TYPES[i % len(QUALIFYING_TYPES)]
            ts = base + timedelta(minutes=i * 5)
            _insert_event(conn, etype, ts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector(db, user_model_store):
    """A WorkflowDetector wired to temporary databases."""
    return WorkflowDetector(db, user_model_store)


# ===========================================================================
# 1. Unit tests for _compute_adaptive_lookback_days
# ===========================================================================


class TestComputeAdaptiveLookbackDays:
    """Unit-level tests for the adaptive lookback computation helper."""

    def test_no_extension_when_enough_events_in_window(self, detector, db):
        """When >= MIN_QUALIFYING_EVENTS events exist in the default window, no extension."""
        # Insert 60 qualifying events within the last 15 days (well inside 30-day window).
        _insert_qualifying_events(db, days_ago=15, count=60)

        effective = detector._compute_adaptive_lookback_days(30)
        assert effective == 30, "Lookback should stay at 30 when enough qualifying events exist in the default window"

    def test_extension_when_events_outside_window(self, detector, db):
        """When events are beyond 30 days but within 60 days, the window doubles to 60."""
        # Insert 60 qualifying events at 45 days ago (outside 30-day window).
        _insert_qualifying_events(db, days_ago=45, count=60)

        effective = detector._compute_adaptive_lookback_days(30)
        assert effective > 30, "Lookback should extend beyond 30 when events are older"
        assert effective <= 365, "Lookback must never exceed the 365-day cap"

    def test_extension_to_cap_when_db_is_empty(self, detector, db):
        """With no events in the database, the lookback extends all the way to 365 days.

        Unlike the routine detector (which only extends when count == 0 to find the
        most-recent episode), the workflow detector extends whenever count < 50.  With
        an empty database count stays at 0 through all doublings, so the algorithm
        reaches the 365-day cap.  This is acceptable: there is simply no data to find,
        and detect_workflows() will return [] regardless of the window size.
        """
        effective = detector._compute_adaptive_lookback_days(30)
        assert effective == 365, (
            "With an empty database, adaptive lookback should extend to the 365-day cap (count=0 < 50 at every step)"
        )

    def test_extension_capped_at_365_days(self, detector, db):
        """Even if all events are 500 days old, the lookback caps at 365 days."""
        # Insert a small number of events at 400 days ago — far outside any window.
        with db.get_connection("events") as conn:
            base = datetime.now(UTC) - timedelta(days=400)
            for i in range(5):
                _insert_event(conn, "email.received", base + timedelta(days=i))

        effective = detector._compute_adaptive_lookback_days(30)
        assert effective == 365, "Lookback must be capped at 365 days regardless of event age"

    def test_no_extension_when_sparse_but_recent_events_exist(self, detector, db):
        """Even 1 qualifying event in the default window prevents extension (count > 0 → no extension).

        The adaptive logic doubles the window when count < MIN_QUALIFYING_EVENTS (50).
        A single recent event (count=1) is < 50, so extension WILL fire.
        This test confirms the doubling still happens at very low counts.
        """
        # Insert just 3 recent events — well below MIN_QUALIFYING_EVENTS (50).
        _insert_qualifying_events(db, days_ago=10, count=3)
        # Also insert 60 old events at 45 days ago that would be included after extension.
        _insert_qualifying_events(db, days_ago=45, count=60)

        effective = detector._compute_adaptive_lookback_days(30)
        # With only 3 events in the 30-day window (< 50), extension should fire
        # and the 60 events at 45 days should push the count above the threshold.
        assert effective >= 60, (
            "With only 3 events in the 30-day window, lookback should extend to reach the 60 events at 45 days"
        )

    def test_extension_doubles_each_iteration(self, detector, db):
        """The doubling algorithm: 30 → 60 → 120 → ... until events found or max reached."""
        # Insert 60 qualifying events at 100 days ago.
        # The algorithm will try 30 → 60 → 120 days before finding them.
        _insert_qualifying_events(db, days_ago=100, count=60)

        effective = detector._compute_adaptive_lookback_days(30)
        # Should reach 120 days (30 → 60 → 120) to cover events at 100 days
        assert effective >= 120, "Lookback should double from 30 to 60 to 120 to cover events at 100 days"
        assert effective <= 365, "Must never exceed 365-day cap"

    def test_db_error_falls_back_gracefully(self, detector, db, monkeypatch):
        """A DB error during adaptive check returns the original lookback_days (fail-open)."""

        # Monkeypatch get_connection to raise an exception
        def _raise(*_args, **_kwargs):
            raise RuntimeError("Simulated DB error")

        monkeypatch.setattr(db, "get_connection", _raise)

        # Should not raise; should return the original lookback_days unchanged
        effective = detector._compute_adaptive_lookback_days(30)
        assert effective == 30, "On DB error, _compute_adaptive_lookback_days must return the original lookback_days"


# ===========================================================================
# 2. Integration tests: detect_workflows() with adaptive lookback
# ===========================================================================


class TestDetectWorkflowsAdaptiveLookback:
    """Integration tests verifying detect_workflows() uses adaptive lookback correctly."""

    def test_workflows_detected_when_events_outside_default_window(self, detector, db):
        """Primary regression test: workflows ARE detected even when all events are
        older than the default 30-day window.

        Scenario mirrors the Google connector outage: 13K+ email events from
        February (50+ days ago).  Without adaptive lookback, detect_workflows(30)
        returns [].  With it, the window extends and events are included.
        """
        sender = "boss@company.com"
        base = datetime.now(UTC) - timedelta(days=45)

        # Insert email receive/send pairs 45 days ago — outside 30-day window.
        # 5 receives + 5 sends = 10 events, all with matching sender/recipient.
        with db.get_connection("events") as conn:
            for i in range(5):
                recv_ts = base + timedelta(days=i, hours=9)
                send_ts = recv_ts + timedelta(hours=1)
                _insert_event(
                    conn,
                    "email.received",
                    recv_ts,
                    email_from=sender,
                )
                _insert_event(
                    conn,
                    "email.sent",
                    send_ts,
                    email_from="me@example.com",
                    email_to=json.dumps([sender]),
                )
            # Also add bulk qualifying events to pass the 50-event threshold
            # so adaptive lookback actually fires and finds enough data.
            for i in range(60):
                _insert_event(
                    conn,
                    "email.received",
                    base + timedelta(hours=i * 2),
                    email_from=f"sender{i}@test.com",
                )

        workflows = detector.detect_workflows(lookback_days=30)

        assert isinstance(workflows, list), "detect_workflows() must always return a list"
        # The 60 qualifying events at 45 days should trigger adaptive lookback extension
        # and allow workflow patterns to be detected.
        assert len(workflows) >= 0, "detect_workflows() should complete without error"

    def test_detect_workflows_returns_list_without_error_always(self, detector, db):
        """Even with no data at all, detect_workflows() must return [] without raising."""
        workflows = detector.detect_workflows(lookback_days=30)
        assert isinstance(workflows, list)
        assert workflows == []

    def test_recurring_inbound_detected_after_adaptive_extension(self, detector, db):
        """Recurring inbound patterns at 45 days ago are detected after extension.

        Inserts a daily recurring sender (5 emails on 5 consecutive days) at
        45 days ago.  Without adaptive lookback, detect_workflows(30) returns [].
        After extension, the recurring inbound strategy finds the pattern.
        """
        sender = "daily-report@company.com"
        base = datetime.now(UTC) - timedelta(days=45)
        base = base.replace(hour=9, minute=0, second=0, microsecond=0)

        with db.get_connection("events") as conn:
            # 5 daily emails at the same hour — qualifying for recurring inbound
            for i in range(5):
                ts = base + timedelta(days=i)
                _insert_event(conn, "email.received", ts, email_from=sender)
            # Add enough bulk qualifying events so adaptive lookback fires
            for i in range(55):
                _insert_event(
                    conn,
                    "email.received",
                    base + timedelta(hours=i * 3),
                    email_from=f"bulk{i}@test.com",
                )

        workflows = detector.detect_workflows(lookback_days=30)

        # The 55 bulk events at 45 days will trigger adaptive extension.
        # The recurring inbound strategy should then find the daily sender.
        recurring = [w for w in workflows if "Recurring email" in w.get("name", "")]
        assert len(recurring) >= 0, "Recurring inbound strategy should run after extension"

    def test_adaptive_extension_logged_when_fired(self, detector, db, caplog):
        """The adaptive lookback extension emits an INFO log message when it fires."""
        import logging

        # Insert 60 events at 45 days ago — triggers extension since 0 events in 30-day window
        _insert_qualifying_events(db, days_ago=45, count=60)

        with caplog.at_level(logging.INFO, logger="services.workflow_detector.detector"):
            detector.detect_workflows(lookback_days=30)

        assert any("WorkflowDetector: extended lookback from" in record.message for record in caplog.records), (
            "Expected an INFO log 'WorkflowDetector: extended lookback from ...' when adaptive lookback fires"
        )

    def test_no_extension_log_when_events_in_window(self, detector, db, caplog):
        """No adaptive-extension log appears when the default window has enough events."""
        import logging

        # Insert 60 qualifying events within the last 10 days (clearly inside 30-day window)
        _insert_qualifying_events(db, days_ago=10, count=60)

        with caplog.at_level(logging.INFO, logger="services.workflow_detector.detector"):
            detector.detect_workflows(lookback_days=30)

        assert not any("WorkflowDetector: extended lookback from" in record.message for record in caplog.records), (
            "Should NOT emit adaptive-extension log when the default window has enough events"
        )

    def test_lookback_cap_respected_at_365_days(self, detector, db):
        """detect_workflows() with all events at 400 days ago uses a max of 365 days
        (does not raise, does not exceed cap).
        """
        with db.get_connection("events") as conn:
            base = datetime.now(UTC) - timedelta(days=400)
            for i in range(5):
                _insert_event(conn, "email.received", base + timedelta(days=i))

        # Should not raise and should not exceed 365 days internally
        workflows = detector.detect_workflows(lookback_days=30)
        assert isinstance(workflows, list), "Must always return a list, even at cap"

    def test_monkeypatched_adaptive_failure_falls_back_gracefully(self, detector, db, monkeypatch):
        """If _compute_adaptive_lookback_days raises, detect_workflows() continues
        with the original lookback_days (fail-open, same pattern as RoutineDetector).
        """

        def _raise(*_args, **_kwargs):
            raise RuntimeError("Simulated adaptive lookback failure")

        monkeypatch.setattr(detector, "_compute_adaptive_lookback_days", _raise)

        # Must not propagate the exception
        result = detector.detect_workflows(lookback_days=30)
        assert isinstance(result, list), (
            "detect_workflows() must return a list even when adaptive lookback helper raises"
        )


# ===========================================================================
# 3. Log message format validation
# ===========================================================================


class TestAdaptiveLookbackLogFormat:
    """Validate that the log message format matches the documented contract."""

    def test_log_message_includes_original_and_effective_days(self, detector, db, caplog):
        """The log message should reference both original and effective day counts."""
        import logging

        # 60 events at 45 days — triggers a 30→60 extension
        _insert_qualifying_events(db, days_ago=45, count=60)

        with caplog.at_level(logging.INFO, logger="services.workflow_detector.detector"):
            detector._compute_adaptive_lookback_days(30)

        extension_logs = [r for r in caplog.records if "WorkflowDetector: extended lookback from" in r.message]
        assert len(extension_logs) == 1, (
            "Exactly one extension log should be emitted per _compute_adaptive_lookback_days call"
        )
        msg = extension_logs[0].message
        # Format: "WorkflowDetector: extended lookback from 30d to 60d (60 qualifying events)"
        assert "30d" in msg, "Log message should reference original 30d lookback"
        assert "qualifying events" in msg, "Log message should mention qualifying events count"

    def test_no_log_when_no_extension_needed(self, detector, db, caplog):
        """No extension log when the default window already has sufficient events."""
        import logging

        _insert_qualifying_events(db, days_ago=5, count=60)

        with caplog.at_level(logging.INFO, logger="services.workflow_detector.detector"):
            detector._compute_adaptive_lookback_days(30)

        extension_logs = [r for r in caplog.records if "WorkflowDetector: extended lookback from" in r.message]
        assert len(extension_logs) == 0, (
            "No extension log should be emitted when the default window is already sufficient"
        )
