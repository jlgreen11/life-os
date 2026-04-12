"""
Tests for notification expiry_reason column and delivery_diagnostics().

Covers:
- expire_stale_notifications() sets expiry_reason='age_exceeded' on expired rows
- Notifications that do not expire retain NULL expiry_reason
- delivery_diagnostics() returns correct reason breakdowns for expired notifications
- delivery_diagnostics() handles the 'unknown' bucket (NULL expiry_reason from
  pre-migration rows)
- The ALTER TABLE migration is idempotent: calling _init_state_db() on a fresh
  database (column already in schema) does not raise an error
- delivery_diagnostics() handles empty tables gracefully (no expired notifications)
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.notification_manager.manager import NotificationManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_event_bus():
    """Mock event bus with is_connected and publish."""
    bus = MagicMock()
    bus.is_connected = True
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def notification_manager(db, mock_event_bus):
    """Create a NotificationManager with the shared test database."""
    return NotificationManager(db, mock_event_bus, config={}, timezone="UTC")


def _insert_notification(db, notif_id: str, status: str, hours_ago: float,
                         expiry_reason: str | None = None):
    """Insert a notification row directly with a backdated created_at.

    Args:
        db: DatabaseManager fixture
        notif_id: Notification ID string
        status: 'pending', 'batched', 'delivered', 'expired', etc.
        hours_ago: How many hours in the past to set created_at
        expiry_reason: Optional expiry_reason value (NULL if omitted)
    """
    created_at = (
        datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, status, created_at, expiry_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, f"Notification {notif_id}", "body", "normal",
             status, created_at, expiry_reason),
        )


# ============================================================================
# expiry_reason is set on expired notifications
# ============================================================================


class TestExpiryReasonOnExpire:
    """expire_stale_notifications() must write expiry_reason='age_exceeded'."""

    def test_expired_notification_has_age_exceeded_reason(self, notification_manager, db):
        """A notification expired by the 48h cutoff gets expiry_reason='age_exceeded'."""
        _insert_notification(db, "notif-old", "pending", hours_ago=72)

        expired_count, expired_ids = notification_manager.expire_stale_notifications(max_age_hours=48)

        assert expired_count == 1
        assert "notif-old" in expired_ids

        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT status, expiry_reason FROM notifications WHERE id = ?",
                ("notif-old",),
            ).fetchone()

        assert row["status"] == "expired"
        assert row["expiry_reason"] == "age_exceeded"

    def test_batched_notification_gets_age_exceeded_reason(self, notification_manager, db):
        """Batched notifications expired by the cutoff also get expiry_reason='age_exceeded'."""
        _insert_notification(db, "notif-batched-old", "batched", hours_ago=72)

        expired_count, _ = notification_manager.expire_stale_notifications(max_age_hours=48)

        assert expired_count == 1

        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT expiry_reason FROM notifications WHERE id = ?",
                ("notif-batched-old",),
            ).fetchone()

        assert row["expiry_reason"] == "age_exceeded"

    def test_recent_notification_keeps_null_expiry_reason(self, notification_manager, db):
        """A notification younger than the cutoff is not expired and has NULL expiry_reason."""
        _insert_notification(db, "notif-recent", "pending", hours_ago=1)

        expired_count, expired_ids = notification_manager.expire_stale_notifications(max_age_hours=48)

        assert expired_count == 0
        assert expired_ids == []

        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT status, expiry_reason FROM notifications WHERE id = ?",
                ("notif-recent",),
            ).fetchone()

        assert row["status"] == "pending"
        assert row["expiry_reason"] is None

    def test_multiple_expirations_all_get_age_exceeded(self, notification_manager, db):
        """All notifications expired in one pass get expiry_reason='age_exceeded'."""
        for i in range(3):
            _insert_notification(db, f"notif-multi-{i}", "pending", hours_ago=72)

        expired_count, expired_ids = notification_manager.expire_stale_notifications(max_age_hours=48)

        assert expired_count == 3

        with db.get_connection("state") as conn:
            rows = conn.execute(
                "SELECT expiry_reason FROM notifications WHERE status = 'expired'"
            ).fetchall()

        assert all(row["expiry_reason"] == "age_exceeded" for row in rows)

    def test_delivered_notification_not_expired(self, notification_manager, db):
        """Delivered notifications are never expired, so they get no expiry_reason."""
        _insert_notification(db, "notif-delivered", "delivered", hours_ago=72)
        _insert_notification(db, "notif-pending", "pending", hours_ago=72)

        notification_manager.expire_stale_notifications(max_age_hours=48)

        with db.get_connection("state") as conn:
            delivered = conn.execute(
                "SELECT expiry_reason FROM notifications WHERE id = ?",
                ("notif-delivered",),
            ).fetchone()
            pending = conn.execute(
                "SELECT expiry_reason FROM notifications WHERE id = ?",
                ("notif-pending",),
            ).fetchone()

        # Delivered row should not have an expiry_reason set
        assert delivered["expiry_reason"] is None
        # The pending row that crossed the cutoff should be age_exceeded
        assert pending["expiry_reason"] == "age_exceeded"


# ============================================================================
# delivery_diagnostics() breakdown
# ============================================================================


class TestDeliveryDiagnosticsBreakdown:
    """delivery_diagnostics() returns correct reason breakdowns."""

    def test_empty_table_returns_zero_totals(self, notification_manager, db):
        """delivery_diagnostics() with no expired notifications returns empty breakdown."""
        result = notification_manager.delivery_diagnostics()

        assert result["total_expired"] == 0
        assert result["expiry_reason_breakdown"] == {}

    def test_age_exceeded_bucket_populated_after_expiry(self, notification_manager, db):
        """After expire_stale_notifications(), breakdown contains 'age_exceeded'."""
        _insert_notification(db, "notif-diag-1", "pending", hours_ago=72)
        _insert_notification(db, "notif-diag-2", "pending", hours_ago=72)

        notification_manager.expire_stale_notifications(max_age_hours=48)

        result = notification_manager.delivery_diagnostics()

        assert result["total_expired"] == 2
        assert result["expiry_reason_breakdown"]["age_exceeded"] == 2

    def test_unknown_bucket_for_pre_migration_rows(self, notification_manager, db):
        """Rows with NULL expiry_reason (pre-migration) appear in the 'unknown' bucket."""
        # Insert a row already expired but without expiry_reason (simulates pre-migration data)
        _insert_notification(db, "notif-legacy", "expired", hours_ago=72, expiry_reason=None)

        result = notification_manager.delivery_diagnostics()

        assert result["total_expired"] == 1
        assert result["expiry_reason_breakdown"]["unknown"] == 1

    def test_mixed_reasons_breakdown(self, notification_manager, db):
        """Breakdown correctly splits between known reasons and legacy unknown rows."""
        # Simulate pre-migration expired rows (NULL expiry_reason)
        _insert_notification(db, "notif-legacy-1", "expired", hours_ago=96, expiry_reason=None)
        _insert_notification(db, "notif-legacy-2", "expired", hours_ago=96, expiry_reason=None)

        # Now expire two more via the normal path (gets 'age_exceeded')
        _insert_notification(db, "notif-new-1", "pending", hours_ago=72)
        _insert_notification(db, "notif-new-2", "pending", hours_ago=72)
        notification_manager.expire_stale_notifications(max_age_hours=48)

        result = notification_manager.delivery_diagnostics()

        assert result["total_expired"] == 4
        breakdown = result["expiry_reason_breakdown"]
        assert breakdown["unknown"] == 2
        assert breakdown["age_exceeded"] == 2

    def test_non_expired_notifications_not_counted(self, notification_manager, db):
        """Pending, delivered, and dismissed notifications are excluded from the breakdown."""
        _insert_notification(db, "notif-pending", "pending", hours_ago=1)
        _insert_notification(db, "notif-delivered", "delivered", hours_ago=72)
        _insert_notification(db, "notif-dismissed", "dismissed", hours_ago=72)
        # One genuinely expired row
        _insert_notification(db, "notif-expired", "expired", hours_ago=72,
                             expiry_reason="age_exceeded")

        result = notification_manager.delivery_diagnostics()

        assert result["total_expired"] == 1
        assert result["expiry_reason_breakdown"] == {"age_exceeded": 1}

    def test_total_expired_matches_sum_of_breakdown(self, notification_manager, db):
        """total_expired must equal the sum of all values in expiry_reason_breakdown."""
        # Mix of reasons
        _insert_notification(db, "n1", "expired", hours_ago=72, expiry_reason="age_exceeded")
        _insert_notification(db, "n2", "expired", hours_ago=72, expiry_reason="age_exceeded")
        _insert_notification(db, "n3", "expired", hours_ago=72, expiry_reason=None)

        result = notification_manager.delivery_diagnostics()

        total = sum(result["expiry_reason_breakdown"].values())
        assert result["total_expired"] == total
        assert result["total_expired"] == 3


# ============================================================================
# Migration idempotency
# ============================================================================


class TestMigrationIdempotency:
    """The ALTER TABLE migration must not fail on a fresh database."""

    def test_init_state_db_idempotent_with_existing_column(self, db):
        """Calling _init_state_db() on an already-migrated database does not raise.

        The conftest.py db fixture calls initialize_databases() which calls
        _init_state_db() and then runs the ALTER TABLE migration.  Calling
        _init_state_db() a second time must be safe (the try/except catches
        the OperationalError for duplicate column).
        """
        # Re-initialise the database — should not raise even though the column exists
        db._init_state_db()  # noqa: SLF001

        # Verify the column is still present and functional
        _insert_notification(db, "notif-idempotent", "pending", hours_ago=1)
        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT expiry_reason FROM notifications WHERE id = ?",
                ("notif-idempotent",),
            ).fetchone()
        assert row["expiry_reason"] is None

    def test_expiry_reason_column_exists_after_init(self, db):
        """After initialization, PRAGMA table_info shows expiry_reason column."""
        with db.get_connection("state") as conn:
            columns = [
                row[1]
                for row in conn.execute("PRAGMA table_info(notifications)").fetchall()
            ]
        assert "expiry_reason" in columns
