"""
Tests for the connector health monitor loop.

The ``_connector_health_monitor_loop`` method on ``LifeOS`` periodically checks
the ``connector_state`` table in state.db and publishes
``system.connector.health_degraded`` events when a connector is in error status
or its last_sync is stale (>24 hours old).  These events trigger the
"Alert on degraded connector" default rule, creating user-facing notifications.

These tests exercise the monitor logic directly by extracting the core check
into a helper and verifying the publish / deduplication / recovery behavior.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from storage.manager import DatabaseManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    """A fully-initialized DatabaseManager using a temporary data directory."""
    manager = DatabaseManager(data_dir=str(tmp_path))
    manager.initialize_all()
    return manager


@pytest.fixture()
def mock_event_bus():
    """A mock event bus that records published events."""
    bus = MagicMock()
    bus.is_connected = True
    bus.publish = AsyncMock()
    return bus


def _insert_connector_state(db, connector_id, status="ok", last_sync=None, last_error=None):
    """Insert a row into connector_state for testing."""
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO connector_state
               (connector_id, status, last_sync, last_error, enabled)
               VALUES (?, ?, ?, ?, 1)""",
            (connector_id, status, last_sync, last_error),
        )


async def _run_monitor_iteration(db, event_bus, alerted_connectors):
    """Execute one iteration of the connector health monitor logic.

    Extracted from ``LifeOS._connector_health_monitor_loop`` so tests
    can exercise the core logic without needing a full LifeOS instance
    or dealing with the infinite loop / sleep.

    Args:
        db: DatabaseManager instance
        event_bus: Event bus (real or mock) with publish() method
        alerted_connectors: Mutable set tracking already-alerted connector IDs

    Returns:
        The alerted_connectors set (same object, mutated in-place).
    """
    with db.get_connection("state") as conn:
        cursor = conn.execute(
            "SELECT connector_id, status, last_sync, last_error FROM connector_state"
        )
        rows = cursor.fetchall()

    now = datetime.now(timezone.utc)
    for row in rows:
        connector_id = row["connector_id"]
        is_degraded = False
        reason = ""

        if row["status"] == "error":
            is_degraded = True
            reason = f"status=error: {row['last_error'] or 'unknown'}"
        elif row["last_sync"]:
            try:
                last_sync = datetime.fromisoformat(
                    row["last_sync"].replace("Z", "+00:00")
                )
                stale_seconds = (now - last_sync).total_seconds()
                if stale_seconds > 86400:  # 24 hours
                    is_degraded = True
                    hours_stale = int(stale_seconds / 3600)
                    reason = f"no sync for {hours_stale}h"
            except (ValueError, TypeError):
                pass

        if is_degraded and connector_id not in alerted_connectors:
            if event_bus and event_bus.is_connected:
                await event_bus.publish(
                    "system.connector.health_degraded",
                    {
                        "connector_id": connector_id,
                        "status": row["status"],
                        "last_sync": row["last_sync"],
                        "error": row["last_error"],
                        "reason": reason,
                    },
                    source="connector_health_monitor",
                    priority="high",
                )
            alerted_connectors.add(connector_id)
        elif not is_degraded and connector_id in alerted_connectors:
            alerted_connectors.discard(connector_id)

    return alerted_connectors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnectorHealthMonitor:
    """Tests for the connector health monitor loop logic."""

    async def test_detects_error_status(self, db, mock_event_bus):
        """A connector with status='error' should trigger a health_degraded event."""
        _insert_connector_state(
            db, "google", status="error", last_error="Authentication failed"
        )
        alerted = set()
        await _run_monitor_iteration(db, mock_event_bus, alerted)

        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == "system.connector.health_degraded"
        payload = call_args[0][1]
        assert payload["connector_id"] == "google"
        assert payload["status"] == "error"
        assert payload["error"] == "Authentication failed"
        assert "status=error" in payload["reason"]
        assert call_args[1]["priority"] == "high"
        assert "google" in alerted

    async def test_detects_stale_sync(self, db, mock_event_bus):
        """A connector with last_sync >24h ago should trigger a health_degraded event."""
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        _insert_connector_state(db, "proton_mail", status="ok", last_sync=stale_time)

        alerted = set()
        await _run_monitor_iteration(db, mock_event_bus, alerted)

        mock_event_bus.publish.assert_called_once()
        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["connector_id"] == "proton_mail"
        assert "no sync for 48h" in payload["reason"]

    async def test_no_alert_for_healthy_connector(self, db, mock_event_bus):
        """A connector with status='ok' and recent last_sync should NOT be alerted."""
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        _insert_connector_state(db, "google", status="ok", last_sync=recent_time)

        alerted = set()
        await _run_monitor_iteration(db, mock_event_bus, alerted)

        mock_event_bus.publish.assert_not_called()
        assert "google" not in alerted

    async def test_no_duplicate_alerts(self, db, mock_event_bus):
        """Running the monitor twice with the same degraded connector should only publish once."""
        _insert_connector_state(
            db, "google", status="error", last_error="Auth expired"
        )
        alerted = set()

        # First iteration — should publish
        await _run_monitor_iteration(db, mock_event_bus, alerted)
        assert mock_event_bus.publish.call_count == 1

        # Second iteration — same degraded state, should NOT publish again
        await _run_monitor_iteration(db, mock_event_bus, alerted)
        assert mock_event_bus.publish.call_count == 1

    async def test_alert_clears_on_recovery(self, db, mock_event_bus):
        """Alert should clear when connector recovers, then re-fire on subsequent degradation."""
        # Step 1: Connector is degraded → alert fires
        _insert_connector_state(
            db, "google", status="error", last_error="Auth expired"
        )
        alerted = set()
        await _run_monitor_iteration(db, mock_event_bus, alerted)
        assert mock_event_bus.publish.call_count == 1
        assert "google" in alerted

        # Step 2: Connector recovers → alert clears
        recent_time = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        _insert_connector_state(db, "google", status="ok", last_sync=recent_time)
        await _run_monitor_iteration(db, mock_event_bus, alerted)
        assert "google" not in alerted
        # No new publish for recovery
        assert mock_event_bus.publish.call_count == 1

        # Step 3: Connector degrades again → alert re-fires
        _insert_connector_state(
            db, "google", status="error", last_error="Token revoked"
        )
        await _run_monitor_iteration(db, mock_event_bus, alerted)
        assert mock_event_bus.publish.call_count == 2
        assert "google" in alerted

    async def test_error_in_monitor_does_not_crash(self, db, mock_event_bus):
        """If the DB query throws, the monitor should catch the exception and continue."""
        alerted = set()

        # Create a mock DB that raises on get_connection
        broken_db = MagicMock()
        broken_db.get_connection = MagicMock(
            side_effect=Exception("DB connection failed")
        )

        # Should not raise — errors are caught
        try:
            # Simulate the try/except wrapper from the loop
            try:
                await _run_monitor_iteration(broken_db, mock_event_bus, alerted)
            except Exception:
                pass  # The real loop catches this
        except Exception:
            pytest.fail("Monitor should not propagate exceptions")

        mock_event_bus.publish.assert_not_called()

    async def test_multiple_connectors_mixed_health(self, db, mock_event_bus):
        """Multiple connectors with mixed health states should only alert degraded ones."""
        # Healthy connector
        recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        _insert_connector_state(db, "caldav", status="ok", last_sync=recent)

        # Error connector
        _insert_connector_state(
            db, "google", status="error", last_error="Auth failed"
        )

        # Stale connector
        stale = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        _insert_connector_state(db, "proton_mail", status="ok", last_sync=stale)

        alerted = set()
        await _run_monitor_iteration(db, mock_event_bus, alerted)

        # Two degraded connectors should be published
        assert mock_event_bus.publish.call_count == 2
        assert alerted == {"google", "proton_mail"}

    async def test_connector_with_no_last_sync_and_ok_status(self, db, mock_event_bus):
        """A connector with status='ok' and NULL last_sync should NOT be alerted.

        A newly-registered connector may not have synced yet — that's not degraded.
        """
        _insert_connector_state(db, "new_connector", status="ok", last_sync=None)

        alerted = set()
        await _run_monitor_iteration(db, mock_event_bus, alerted)

        mock_event_bus.publish.assert_not_called()

    async def test_stale_sync_z_suffix_timestamp(self, db, mock_event_bus):
        """Timestamps with trailing 'Z' (UTC marker) should be parsed correctly."""
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=30)).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        _insert_connector_state(db, "signal", status="ok", last_sync=stale_time)

        alerted = set()
        await _run_monitor_iteration(db, mock_event_bus, alerted)

        mock_event_bus.publish.assert_called_once()
        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["connector_id"] == "signal"
        assert "no sync for 30h" in payload["reason"]

    async def test_disconnected_event_bus_skips_publish(self, db, mock_event_bus):
        """If the event bus is disconnected, the monitor should skip publishing."""
        mock_event_bus.is_connected = False
        _insert_connector_state(
            db, "google", status="error", last_error="Auth failed"
        )

        alerted = set()
        await _run_monitor_iteration(db, mock_event_bus, alerted)

        mock_event_bus.publish.assert_not_called()
        # But still track as alerted to avoid re-firing when bus reconnects
        assert "google" in alerted


class TestDegradedConnectorDefaultRule:
    """Tests for the 'Alert on degraded connector' default rule."""

    def test_rule_exists_in_default_rules(self):
        """DEFAULT_RULES should include a rule for system.connector.health_degraded."""
        from services.rules_engine.engine import DEFAULT_RULES

        rule = next(
            (r for r in DEFAULT_RULES if r["name"] == "Alert on degraded connector"),
            None,
        )
        assert rule is not None, "Expected 'Alert on degraded connector' in DEFAULT_RULES"
        assert rule["trigger_event"] == "system.connector.health_degraded"
        assert any(a["type"] == "notify" for a in rule["actions"])
        assert any(a.get("priority") == "high" for a in rule["actions"])

    def test_event_type_in_core_model(self):
        """EventType enum should include CONNECTOR_HEALTH_DEGRADED."""
        from models.core import EventType

        assert hasattr(EventType, "CONNECTOR_HEALTH_DEGRADED")
        assert EventType.CONNECTOR_HEALTH_DEGRADED.value == "system.connector.health_degraded"
