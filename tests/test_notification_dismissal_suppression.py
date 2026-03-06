"""
Tests for notification dismissal-rate suppression.

Validates that NotificationManager suppresses notifications from domains
with high dismissal rates (70%+ with 3+ data points), while always
allowing critical notifications and failing open for unknown domains.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from services.notification_manager.manager import NotificationManager


@pytest.fixture()
def nm(db, event_bus):
    """NotificationManager wired to temp databases."""
    return NotificationManager(db, event_bus, config={}, timezone="UTC")


def _insert_feedback(db, domain, feedback_type, count=1):
    """Insert feedback_log entries for a given domain and feedback_type."""
    with db.get_connection("preferences") as conn:
        for _ in range(count):
            conn.execute(
                """INSERT INTO feedback_log
                   (id, timestamp, action_id, action_type, feedback_type, context)
                   VALUES (?, ?, ?, 'notification', ?, ?)""",
                (
                    str(uuid.uuid4()),
                    datetime.now(timezone.utc).isoformat(),
                    str(uuid.uuid4()),
                    feedback_type,
                    json.dumps({"domain": domain}),
                ),
            )


def _insert_notification(db, notif_id, domain="test", status="pending"):
    """Insert a notification row into state.db."""
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications (id, title, body, priority, domain, status)
               VALUES (?, 'Test', 'body', 'normal', ?, ?)""",
            (notif_id, domain, status),
        )


class TestDismissalSuppression:
    """Tests for _check_dismissal_suppression()."""

    def test_no_dismissals_not_suppressed(self, nm, db):
        """Domain with zero feedback history should not be suppressed."""
        assert nm._check_dismissal_suppression("clean_domain") is False

    def test_low_dismissal_rate_not_suppressed(self, nm, db):
        """Domain with <70% dismissal rate should not be suppressed."""
        # 1 dismissed, 3 engaged = 25% dismissal rate
        _insert_feedback(db, "mixed_domain", "dismissed", count=1)
        _insert_feedback(db, "mixed_domain", "engaged", count=3)
        assert nm._check_dismissal_suppression("mixed_domain") is False

    def test_high_dismissal_rate_with_enough_data_suppressed(self, nm, db):
        """Domain with 70%+ dismissal rate and 3+ data points should be suppressed."""
        # 3 dismissed, 1 engaged = 75% dismissal rate, 4 total >= 3
        _insert_feedback(db, "noisy_domain", "dismissed", count=3)
        _insert_feedback(db, "noisy_domain", "engaged", count=1)
        assert nm._check_dismissal_suppression("noisy_domain") is True

    def test_high_dismissal_rate_sparse_data_not_suppressed(self, nm, db):
        """Domain with high dismissal rate but <3 data points should not suppress."""
        # 2 dismissed, 0 engaged = 100% rate but only 2 data points
        _insert_feedback(db, "sparse_domain", "dismissed", count=2)
        assert nm._check_dismissal_suppression("sparse_domain") is False

    def test_none_domain_not_suppressed(self, nm, db):
        """None domain should never be suppressed (fail-open)."""
        assert nm._check_dismissal_suppression(None) is False

    def test_empty_string_domain_not_suppressed(self, nm, db):
        """Empty string domain should not be suppressed (fail-open)."""
        assert nm._check_dismissal_suppression("") is False


class TestCreateNotificationDismissalSuppression:
    """Tests that create_notification() integrates dismissal suppression."""

    @pytest.mark.asyncio
    async def test_suppressed_domain_returns_none(self, nm, db):
        """Notification from a heavily-dismissed domain returns None."""
        _insert_feedback(db, "spam_domain", "dismissed", count=5)
        result = await nm.create_notification(
            title="Spammy",
            body="Should be suppressed",
            priority="normal",
            domain="spam_domain",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_critical_never_suppressed(self, nm, db):
        """Critical notifications bypass dismissal suppression entirely."""
        _insert_feedback(db, "spam_domain", "dismissed", count=10)
        result = await nm.create_notification(
            title="Critical Alert",
            body="Must get through",
            priority="critical",
            domain="spam_domain",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_clean_domain_not_suppressed(self, nm, db):
        """Notification from a domain with no dismissals is delivered normally."""
        result = await nm.create_notification(
            title="Normal Notif",
            body="Should go through",
            priority="normal",
            domain="clean_domain",
        )
        assert result is not None


class TestDismissFeedbackIncludesDomain:
    """Tests that dismiss() includes domain in the feedback context."""

    @pytest.mark.asyncio
    async def test_dismiss_logs_domain_in_feedback(self, nm, db):
        """Dismissing a notification should log the domain in feedback context."""
        notif_id = str(uuid.uuid4())
        _insert_notification(db, notif_id, domain="calendar")

        await nm.dismiss(notif_id)

        # Check feedback_log for the domain
        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT context FROM feedback_log WHERE action_id = ?",
                (notif_id,),
            ).fetchone()

        assert row is not None
        ctx = json.loads(row["context"])
        assert ctx["domain"] == "calendar"

    @pytest.mark.asyncio
    async def test_dismiss_handles_missing_notification(self, nm, db):
        """Dismissing a non-existent notification should still log feedback (fail-open)."""
        fake_id = str(uuid.uuid4())
        # No notification inserted — dismiss should not crash
        await nm.dismiss(fake_id)

        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT context FROM feedback_log WHERE action_id = ?",
                (fake_id,),
            ).fetchone()

        assert row is not None
        ctx = json.loads(row["context"])
        assert ctx["domain"] is None
