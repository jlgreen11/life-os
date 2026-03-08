"""
Tests for the recurring inbound email pattern detection strategy.

Validates that _detect_recurring_inbound_patterns correctly identifies
emails from the same sender that arrive on a predictable schedule (daily,
weekly, or consistent interval), without requiring any outbound replies.

This strategy addresses the zero-workflow detection issue caused by
heavily inbound-skewed email data (e.g. 12,429 received vs 10 sent).
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.workflow_detector.detector import WorkflowDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(dt: datetime) -> str:
    """Format a datetime as ISO 8601 string for SQLite storage."""
    return dt.isoformat()


def _insert_event(conn, event_type: str, timestamp: datetime, *,
                  email_from: str | None = None,
                  email_to: str | None = None,
                  source: str = "test"):
    """Insert a single event into the events table with denormalized columns."""
    eid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                            email_from, email_to)
        VALUES (?, ?, ?, ?, 'normal', '{}', '{}', ?, ?)
        """,
        (eid, event_type, source, _ts(timestamp), email_from, email_to),
    )
    return eid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector(db, user_model_store):
    """A WorkflowDetector wired to temporary databases."""
    return WorkflowDetector(db, user_model_store)


# ===========================================================================
# 1. Daily recurring sender detection
# ===========================================================================


class TestDailyRecurringSender:
    """Test detection of daily recurring email patterns."""

    def test_detects_daily_pattern(self, detector, db):
        """5 emails at the same hour over 5 consecutive days = daily pattern."""
        now = datetime.now(timezone.utc)
        sender = "daily-report@company.com"

        with db.get_connection("events") as conn:
            for i in range(5):
                # Same hour each day (9:00 AM)
                ts = now.replace(hour=9, minute=0, second=0) - timedelta(days=5 - i)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        assert len(workflows) == 1
        wf = workflows[0]
        assert sender in wf["name"]
        assert wf["steps"] == ["email_received"]
        assert wf["times_observed"] == 5
        assert wf["metadata"]["cadence"] == "daily"
        assert wf["metadata"]["sender"] == sender
        assert "email" in wf["tools_used"]
        assert wf["success_rate"] == 1.0

    def test_detects_daily_pattern_with_hour_jitter(self, detector, db):
        """Daily emails with ±1h jitter should still be detected as daily."""
        now = datetime.now(timezone.utc)
        sender = "standup@team.com"

        with db.get_connection("events") as conn:
            hours = [9, 10, 8, 9, 10]  # ±1h around 9 AM
            for i, hour in enumerate(hours):
                ts = now.replace(hour=hour, minute=15, second=0) - timedelta(days=5 - i)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        assert len(workflows) == 1
        assert workflows[0]["metadata"]["cadence"] == "daily"


# ===========================================================================
# 2. Weekly recurring sender detection
# ===========================================================================


class TestWeeklyRecurringSender:
    """Test detection of weekly recurring email patterns."""

    def test_detects_weekly_pattern(self, detector, db):
        """4 emails on the same day of week over 4 weeks = weekly pattern."""
        now = datetime.now(timezone.utc)
        sender = "weekly-digest@newsletter.com"

        with db.get_connection("events") as conn:
            for i in range(4):
                # Same day of week, 7 days apart
                ts = now - timedelta(weeks=4 - i)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=60)

        assert len(workflows) == 1
        wf = workflows[0]
        assert wf["metadata"]["cadence"] == "weekly"
        assert wf["times_observed"] == 4


# ===========================================================================
# 3. Irregular sender rejection
# ===========================================================================


class TestIrregularSenderRejection:
    """Test that irregular/random senders are not detected as workflows."""

    def test_rejects_random_timestamps(self, detector, db):
        """Emails at random intervals should NOT produce a workflow."""
        now = datetime.now(timezone.utc)
        sender = "random@example.com"

        with db.get_connection("events") as conn:
            # Deliberately irregular intervals: 1h, 72h, 3h, 200h, 5h
            offsets_hours = [0, 1, 73, 76, 276, 281]
            for offset in offsets_hours:
                ts = now - timedelta(hours=300 - offset)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        # Random intervals should not match any cadence
        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 0

    def test_rejects_sender_below_min_occurrences(self, detector, db):
        """Senders with fewer than 3 emails are rejected."""
        now = datetime.now(timezone.utc)
        sender = "rare@example.com"

        with db.get_connection("events") as conn:
            for i in range(2):
                ts = now - timedelta(days=2 - i)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)
        assert len(workflows) == 0


# ===========================================================================
# 4. High-volume marketing sender rejection
# ===========================================================================


class TestHighVolumeSenderRejection:
    """Test that extreme-volume senders are filtered by the dynamic threshold.

    The dynamic threshold is max(200, total_emails // 5), so only senders
    exceeding that ceiling are excluded.
    """

    def test_rejects_extreme_volume_sender(self, detector, db):
        """A sender exceeding the dynamic threshold is filtered out."""
        now = datetime.now(timezone.utc)
        spam_sender = "promo@marketing-spam.com"

        with db.get_connection("events") as conn:
            # 201 emails from the spam sender — only sender, so
            # max_volume = max(200, 201 // 5) = 200, and 201 > 200.
            for i in range(201):
                ts = now - timedelta(hours=i * 3)
                _insert_event(conn, "email.received", ts, email_from=spam_sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        matching = [w for w in workflows if spam_sender in w["name"]]
        assert len(matching) == 0

    def test_accepts_sender_at_dynamic_threshold(self, detector, db):
        """A sender with 100 emails is accepted (well below 200 floor)."""
        now = datetime.now(timezone.utc)
        sender = "borderline@reports.com"

        with db.get_connection("events") as conn:
            # 100 emails, daily cadence — below the 200 floor threshold
            for i in range(100):
                ts = now.replace(hour=8, minute=0, second=0) - timedelta(days=100 - i)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=120)

        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 1


# ===========================================================================
# 5. Diagnostics integration
# ===========================================================================


class TestDiagnosticsIncludesRecurringInbound:
    """Test that the recurring_inbound strategy appears in diagnostics output."""

    def test_diagnostics_includes_recurring_inbound(self, detector):
        """get_diagnostics() should include recurring_inbound in detection_results."""
        diag = detector.get_diagnostics(lookback_days=30)

        assert "recurring_inbound" in diag["detection_results"]
        assert "detected" in diag["detection_results"]["recurring_inbound"]

    def test_diagnostics_counts_recurring_inbound(self, detector, db):
        """Detected recurring inbound workflows appear in diagnostics total."""
        now = datetime.now(timezone.utc)
        sender = "diag-test@example.com"

        with db.get_connection("events") as conn:
            for i in range(5):
                ts = now.replace(hour=10, minute=0, second=0) - timedelta(days=5 - i)
                _insert_event(conn, "email.received", ts, email_from=sender)

        diag = detector.get_diagnostics(lookback_days=30)

        ri = diag["detection_results"]["recurring_inbound"]
        assert ri["detected"] >= 1
        assert diag["total_detected"] >= 1


# ===========================================================================
# 6. Integration with detect_workflows()
# ===========================================================================


class TestRecurringInboundIntegration:
    """Test that recurring inbound patterns are included in detect_workflows()."""

    def test_detect_workflows_includes_recurring(self, detector, db):
        """detect_workflows() should return recurring inbound patterns."""
        now = datetime.now(timezone.utc)
        sender = "integration-test@example.com"

        with db.get_connection("events") as conn:
            for i in range(5):
                ts = now.replace(hour=14, minute=0, second=0) - timedelta(days=5 - i)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector.detect_workflows(lookback_days=30)

        recurring = [w for w in workflows if "Recurring" in w["name"]]
        assert len(recurring) >= 1
        assert recurring[0]["metadata"]["cadence"] == "daily"

    def test_workflow_schema_consistency(self, detector, db):
        """Recurring inbound workflows have all required schema keys."""
        now = datetime.now(timezone.utc)
        sender = "schema-test@example.com"

        with db.get_connection("events") as conn:
            for i in range(5):
                ts = now.replace(hour=8, minute=0, second=0) - timedelta(days=5 - i)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        assert len(workflows) >= 1
        wf = workflows[0]
        # All keys required by store_workflows()
        assert "name" in wf
        assert "trigger_conditions" in wf
        assert "steps" in wf
        assert "tools_used" in wf
        assert "success_rate" in wf
        assert "times_observed" in wf

    def test_consistent_interval_detection(self, detector, db):
        """Emails at a consistent non-daily, non-weekly interval are detected."""
        now = datetime.now(timezone.utc)
        sender = "every-3-days@example.com"

        with db.get_connection("events") as conn:
            for i in range(5):
                # Every 3 days, 72-hour intervals
                ts = now - timedelta(days=15 - i * 3)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 1
        assert matching[0]["metadata"]["cadence"] == "interval"
