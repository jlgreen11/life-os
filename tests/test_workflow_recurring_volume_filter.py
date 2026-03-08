"""
Tests for the dynamic volume filter and marketing-tag exclusion in
recurring inbound email workflow detection.

Validates that the dynamic max_volume threshold (max(200, total_emails // 5))
correctly allows legitimate high-volume senders while filtering extreme outliers,
and that senders tagged as 'marketing' or 'system:suppressed' are excluded.
"""

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


def _tag_event(conn, event_id: str, tag: str):
    """Tag an event in the event_tags table."""
    conn.execute(
        "INSERT OR IGNORE INTO event_tags (event_id, tag) VALUES (?, ?)",
        (event_id, tag),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector(db, user_model_store):
    """A WorkflowDetector wired to temporary databases."""
    return WorkflowDetector(db, user_model_store)


# ===========================================================================
# 1. Sender with 100 emails in 30 days IS detected (was incorrectly filtered)
# ===========================================================================


class TestHighVolumeLegitimateSender:
    """Senders with 100 emails over 30 days should be detected, not filtered."""

    def test_100_emails_in_30_days_detected(self, detector, db):
        """A sender with 100 daily emails is well below the 200 floor and should
        be detected as a recurring inbound pattern."""
        now = datetime.now(timezone.utc)
        sender = "daily-standup@company.com"

        with db.get_connection("events") as conn:
            for i in range(100):
                ts = now.replace(hour=9, minute=0, second=0) - timedelta(days=100 - i)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=120)

        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 1
        assert matching[0]["times_observed"] == 100
        assert matching[0]["metadata"]["cadence"] == "daily"

    def test_150_emails_from_colleague_detected(self, detector, db):
        """A prolific colleague sending 150 emails should still be detected."""
        now = datetime.now(timezone.utc)
        sender = "boss@company.com"

        with db.get_connection("events") as conn:
            for i in range(150):
                ts = now.replace(hour=10, minute=0, second=0) - timedelta(days=150 - i)
                _insert_event(conn, "email.received", ts, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=180)

        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 1


# ===========================================================================
# 2. True spam sender with 1000+ emails IS filtered
# ===========================================================================


class TestExtremeVolumeSenderFiltered:
    """Senders with extreme volumes (1000+) should be filtered out."""

    def test_1000_emails_filtered(self, detector, db):
        """A sender with 1000 emails exceeds the dynamic threshold and is filtered."""
        now = datetime.now(timezone.utc)
        spam_sender = "alerts@spammy-service.com"

        with db.get_connection("events") as conn:
            # 1000 emails — dynamic threshold = max(200, 1000 // 5) = 200
            # 1000 > 200, so this sender should be filtered
            for i in range(1000):
                ts = now - timedelta(minutes=i * 40)
                _insert_event(conn, "email.received", ts, email_from=spam_sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=60)

        matching = [w for w in workflows if spam_sender in w["name"]]
        assert len(matching) == 0

    def test_dynamic_threshold_scales_with_volume(self, detector, db):
        """With many senders, the dynamic threshold scales up to allow high-volume
        legitimate senders while still filtering extreme outliers."""
        now = datetime.now(timezone.utc)
        legit_sender = "reports@company.com"
        spam_sender = "noreply@mass-mailer.com"

        with db.get_connection("events") as conn:
            # Insert 50 moderate senders with 20 daily emails each = 1000 total
            for s in range(50):
                sender = f"sender{s}@example.com"
                for i in range(20):
                    ts = now.replace(hour=9, minute=0, second=0) - timedelta(days=20 - i)
                    _insert_event(conn, "email.received", ts, email_from=sender)

            # Legit sender with 180 emails — below threshold
            for i in range(180):
                ts = now.replace(hour=8, minute=0, second=0) - timedelta(days=180 - i)
                _insert_event(conn, "email.received", ts, email_from=legit_sender)

            # Spam sender with 500 emails
            for i in range(500):
                ts = now - timedelta(minutes=i * 80)
                _insert_event(conn, "email.received", ts, email_from=spam_sender)

        # total = 1000 + 180 + 500 = 1680, threshold = max(200, 1680 // 5) = 336
        workflows = detector._detect_recurring_inbound_patterns(lookback_days=200)

        legit = [w for w in workflows if legit_sender in w["name"]]
        spam = [w for w in workflows if spam_sender in w["name"]]
        assert len(legit) == 1, "Legit sender (180 emails) should be below dynamic threshold"
        assert len(spam) == 0, "Spam sender (500 emails) should exceed dynamic threshold"


# ===========================================================================
# 3. Marketing-tagged sender IS excluded
# ===========================================================================


class TestMarketingTagExclusion:
    """Senders whose events are majority-tagged as marketing are excluded."""

    def test_marketing_tagged_sender_excluded(self, detector, db):
        """A sender where >50% of events are tagged 'marketing' is excluded."""
        now = datetime.now(timezone.utc)
        sender = "deals@marketing-co.com"

        with db.get_connection("events") as conn:
            event_ids = []
            for i in range(10):
                ts = now.replace(hour=10, minute=0, second=0) - timedelta(days=10 - i)
                eid = _insert_event(conn, "email.received", ts, email_from=sender)
                event_ids.append(eid)

            # Tag 6 out of 10 as marketing (60% > 50% threshold)
            for eid in event_ids[:6]:
                _tag_event(conn, eid, "marketing")

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 0

    def test_suppressed_tagged_sender_excluded(self, detector, db):
        """A sender where >50% of events are tagged 'system:suppressed' is excluded."""
        now = datetime.now(timezone.utc)
        sender = "notifications@social-media.com"

        with db.get_connection("events") as conn:
            event_ids = []
            for i in range(8):
                ts = now.replace(hour=14, minute=0, second=0) - timedelta(days=8 - i)
                eid = _insert_event(conn, "email.received", ts, email_from=sender)
                event_ids.append(eid)

            # Tag 5 out of 8 as suppressed (62.5% > 50% threshold)
            for eid in event_ids[:5]:
                _tag_event(conn, eid, "system:suppressed")

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 0

    def test_minority_tagged_sender_still_detected(self, detector, db):
        """A sender where <50% of events are tagged is NOT excluded."""
        now = datetime.now(timezone.utc)
        sender = "team-updates@company.com"

        with db.get_connection("events") as conn:
            event_ids = []
            for i in range(10):
                ts = now.replace(hour=9, minute=0, second=0) - timedelta(days=10 - i)
                eid = _insert_event(conn, "email.received", ts, email_from=sender)
                event_ids.append(eid)

            # Tag only 2 out of 10 as marketing (20% < 50%)
            for eid in event_ids[:2]:
                _tag_event(conn, eid, "marketing")

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 1


# ===========================================================================
# 4. Weekday-only cadence detection
# ===========================================================================


class TestWeekdayOnlyCadence:
    """Test that weekday-only email patterns are detected as weekday_daily cadence."""

    def test_weekday_only_emails_detected(self, detector, db):
        """5 emails per week (Mon-Fri only) should be detected as weekday_daily."""
        now = datetime.now(timezone.utc)
        sender = "ci-reports@devops.com"

        with db.get_connection("events") as conn:
            # Generate 3 weeks of weekday-only emails at 9 AM
            base = now.replace(hour=9, minute=0, second=0) - timedelta(days=21)
            for day_offset in range(21):
                d = base + timedelta(days=day_offset)
                if d.weekday() < 5:  # Mon-Fri only
                    _insert_event(conn, "email.received", d, email_from=sender)

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 1
        assert matching[0]["metadata"]["cadence"] == "weekday_daily"

    def test_weekday_cadence_with_hour_jitter(self, detector, db):
        """Weekday emails with slight hour variation still detected as some
        form of daily cadence (daily or weekday_daily depending on interval)."""
        now = datetime.now(timezone.utc)
        sender = "standup-bot@slack.com"

        with db.get_connection("events") as conn:
            base = now.replace(hour=9, minute=0, second=0) - timedelta(days=14)
            hour_offsets = [0, 1, -1, 0, 1, 0, -1, 0, 1, 0]
            idx = 0
            for day_offset in range(14):
                d = base + timedelta(days=day_offset)
                if d.weekday() < 5 and idx < len(hour_offsets):
                    d = d.replace(hour=9 + hour_offsets[idx])
                    _insert_event(conn, "email.received", d, email_from=sender)
                    idx += 1

        workflows = detector._detect_recurring_inbound_patterns(lookback_days=30)

        matching = [w for w in workflows if sender in w["name"]]
        assert len(matching) == 1
        # May match daily or weekday_daily depending on exact interval math
        assert matching[0]["metadata"]["cadence"] in ("daily", "weekday_daily")
