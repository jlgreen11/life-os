"""
Tests for NotificationManager.get_diagnostics().

Covers:
- Diagnostics with empty notifications table (baseline health)
- Correct status counts with mixed statuses
- Health assessment: 'noisy' when most notifications expire
- Health assessment: 'degraded' with old pending notifications
- Health assessment: 'degraded' with large pending backlog
- In-memory batch queue depth reporting
- Domain breakdown correctness
- Read rate calculation
"""

import uuid
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
    """Create a NotificationManager instance with test database."""
    config = {}
    return NotificationManager(db, mock_event_bus, config, timezone="UTC")


def _insert_notification(
    db,
    *,
    status="pending",
    priority="normal",
    domain="rule",
    created_at=None,
    delivered_at=None,
    read_at=None,
    title="Test notification",
):
    """Helper to insert a notification directly into the DB."""
    nid = str(uuid.uuid4())
    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, domain, status, created_at, delivered_at, read_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (nid, title, "body", priority, domain, status, created_at, delivered_at, read_at),
        )
    return nid


# ============================================================================
# Tests
# ============================================================================


class TestDiagnosticsEmpty:
    """Diagnostics with an empty notifications table."""

    def test_empty_table_returns_ok_health(self, notification_manager):
        """Empty table should report health='ok' with all counts at 0."""
        diag = notification_manager.get_diagnostics()
        assert diag["health"] == "ok"
        assert diag["status_counts"] == {}
        assert diag["in_memory_batch_depth"] == 0
        assert diag["oldest_pending_hours"] is None
        assert diag["recommendations"] == []
        assert diag["recent_activity"]["created_24h"] == 0
        assert diag["recent_activity"]["delivered_24h"] == 0
        assert diag["recent_activity"]["expired_24h"] == 0
        assert diag["recent_activity"]["read_rate_7d"] == 0.0

    def test_empty_table_domain_breakdown_empty(self, notification_manager):
        """Empty table should have empty domain breakdown."""
        diag = notification_manager.get_diagnostics()
        assert diag["domain_breakdown"] == {}


class TestDiagnosticsStatusCounts:
    """Diagnostics correctly count notifications by status."""

    def test_mixed_statuses(self, notification_manager, db):
        """Various statuses should be counted correctly."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        _insert_notification(db, status="pending", created_at=now_str)
        _insert_notification(db, status="pending", created_at=now_str)
        _insert_notification(db, status="delivered", created_at=now_str, delivered_at=now_str)
        _insert_notification(db, status="read", created_at=now_str, read_at=now_str)
        _insert_notification(db, status="acted_on", created_at=now_str)
        _insert_notification(db, status="dismissed", created_at=now_str)
        _insert_notification(db, status="expired", created_at=now_str)

        diag = notification_manager.get_diagnostics()
        assert diag["status_counts"]["pending"] == 2
        assert diag["status_counts"]["delivered"] == 1
        assert diag["status_counts"]["read"] == 1
        assert diag["status_counts"]["acted_on"] == 1
        assert diag["status_counts"]["dismissed"] == 1
        assert diag["status_counts"]["expired"] == 1


class TestDiagnosticsDomainBreakdown:
    """Domain breakdown groups notifications by domain."""

    def test_domain_counts(self, notification_manager, db):
        """Notifications should be grouped by domain."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        _insert_notification(db, domain="prediction", created_at=now_str)
        _insert_notification(db, domain="prediction", created_at=now_str)
        _insert_notification(db, domain="rule", created_at=now_str)
        _insert_notification(db, domain="connector", created_at=now_str)
        _insert_notification(db, domain=None, created_at=now_str)

        diag = notification_manager.get_diagnostics()
        assert diag["domain_breakdown"]["prediction"] == 2
        assert diag["domain_breakdown"]["rule"] == 1
        assert diag["domain_breakdown"]["connector"] == 1
        assert diag["domain_breakdown"]["unknown"] == 1


class TestDiagnosticsReadRate:
    """Read rate calculation over 7 days."""

    def test_read_rate_calculation(self, notification_manager, db):
        """Read rate = (read + acted_on) / (delivered + read + acted_on + dismissed)."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        # 2 read, 1 acted_on = 3 "read"
        _insert_notification(db, status="read", created_at=now_str, read_at=now_str)
        _insert_notification(db, status="read", created_at=now_str, read_at=now_str)
        _insert_notification(db, status="acted_on", created_at=now_str)
        # 3 delivered, 4 dismissed = 7 not "read"
        for _ in range(3):
            _insert_notification(db, status="delivered", created_at=now_str, delivered_at=now_str)
        for _ in range(4):
            _insert_notification(db, status="dismissed", created_at=now_str)

        diag = notification_manager.get_diagnostics()
        # 3 / (3 + 2 + 1 + 4) = 3/10 = 0.3
        assert diag["recent_activity"]["read_rate_7d"] == 0.3


class TestDiagnosticsBatchDepth:
    """In-memory batch queue depth reporting."""

    def test_batch_depth_reflects_pending_batch(self, notification_manager):
        """Batch depth should match len(self._pending_batch)."""
        notification_manager._pending_batch = [
            {"id": "a", "title": "t1"},
            {"id": "b", "title": "t2"},
            {"id": "c", "title": "t3"},
        ]
        diag = notification_manager.get_diagnostics()
        assert diag["in_memory_batch_depth"] == 3

    def test_empty_batch(self, notification_manager):
        """Empty batch should report depth 0."""
        notification_manager._pending_batch = []
        diag = notification_manager.get_diagnostics()
        assert diag["in_memory_batch_depth"] == 0


class TestDiagnosticsHealthNoisy:
    """Health assessment: 'noisy' when most notifications expire."""

    def test_noisy_when_high_expiry_rate(self, notification_manager, db):
        """Health should be 'noisy' when >70% of recent notifications expired."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        # 10 notifications created in last 24h, 8 expired
        for _ in range(8):
            _insert_notification(db, status="expired", created_at=now_str)
        for _ in range(2):
            _insert_notification(db, status="delivered", created_at=now_str, delivered_at=now_str)

        diag = notification_manager.get_diagnostics()
        assert diag["health"] == "noisy"
        assert any("expiry rate" in r.lower() for r in diag["recommendations"])


class TestDiagnosticsHealthDegraded:
    """Health assessment: 'degraded' conditions."""

    def test_degraded_with_old_pending(self, notification_manager, db):
        """Health should be 'degraded' when oldest pending notification is >48h old."""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=72)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )
        _insert_notification(db, status="pending", created_at=old_time)

        diag = notification_manager.get_diagnostics()
        assert diag["health"] == "degraded"
        assert diag["oldest_pending_hours"] is not None
        assert diag["oldest_pending_hours"] > 48
        assert any("oldest pending" in r.lower() for r in diag["recommendations"])

    def test_degraded_with_large_pending_backlog(self, notification_manager, db):
        """Health should be 'degraded' when pending count exceeds 50."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        for _ in range(55):
            _insert_notification(db, status="pending", created_at=now_str)

        diag = notification_manager.get_diagnostics()
        assert diag["health"] == "degraded"
        assert any("pending notifications" in r.lower() for r in diag["recommendations"])

    def test_degraded_with_low_read_rate(self, notification_manager, db):
        """Health should be 'degraded' when read rate is below 10%."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        # 20 delivered, 0 read = 0% read rate
        for _ in range(20):
            _insert_notification(db, status="delivered", created_at=now_str, delivered_at=now_str)

        diag = notification_manager.get_diagnostics()
        assert diag["health"] == "degraded"
        assert diag["recent_activity"]["read_rate_7d"] == 0.0
        assert any("read rate" in r.lower() for r in diag["recommendations"])


class TestDiagnosticsDeliveryMode:
    """Delivery mode reporting."""

    def test_delivery_mode_unknown_by_default(self, notification_manager):
        """Delivery mode should be 'unknown' when _delivery_mode is not set."""
        diag = notification_manager.get_diagnostics()
        assert diag["delivery_mode"] == "unknown"

    def test_delivery_mode_when_set(self, notification_manager):
        """Delivery mode should reflect _delivery_mode attribute."""
        notification_manager._delivery_mode = "batched"
        diag = notification_manager.get_diagnostics()
        assert diag["delivery_mode"] == "batched"
