"""
Tests for notification mode and quiet-hours indicator in dashboard status bar.

Covers:
- get_diagnostics() returns delivery_mode field
- get_diagnostics() returns quiet_hours_active field
- Different delivery modes are correctly reflected
- Quiet hours active state is correctly detected
- Dashboard template includes notification mode HTML elements
"""

import json
import uuid
from datetime import datetime, timezone
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
def notif_manager(db, mock_event_bus):
    """Create a NotificationManager instance with test database."""
    return NotificationManager(db, mock_event_bus, config={}, timezone="UTC")


# ============================================================================
# Tests: delivery_mode in diagnostics
# ============================================================================


def test_diagnostics_returns_delivery_mode(notif_manager):
    """get_diagnostics() should include a delivery_mode field."""
    diag = notif_manager.get_diagnostics()
    assert "delivery_mode" in diag
    assert isinstance(diag["delivery_mode"], str)


def test_diagnostics_delivery_mode_reflects_setting(db, mock_event_bus):
    """delivery_mode in diagnostics should reflect the configured mode."""
    # NotificationManager reads _delivery_mode from config or defaults
    mgr = NotificationManager(db, mock_event_bus, config={}, timezone="UTC")
    mgr._delivery_mode = "minimal"
    diag = mgr.get_diagnostics()
    assert diag["delivery_mode"] == "minimal"


def test_diagnostics_delivery_mode_batched(db, mock_event_bus):
    """delivery_mode 'batched' should be reflected in diagnostics."""
    mgr = NotificationManager(db, mock_event_bus, config={}, timezone="UTC")
    mgr._delivery_mode = "batched"
    diag = mgr.get_diagnostics()
    assert diag["delivery_mode"] == "batched"


def test_diagnostics_delivery_mode_frequent(db, mock_event_bus):
    """delivery_mode 'frequent' should be reflected in diagnostics."""
    mgr = NotificationManager(db, mock_event_bus, config={}, timezone="UTC")
    mgr._delivery_mode = "frequent"
    diag = mgr.get_diagnostics()
    assert diag["delivery_mode"] == "frequent"


# ============================================================================
# Tests: quiet_hours_active in diagnostics
# ============================================================================


def test_diagnostics_returns_quiet_hours_active(notif_manager):
    """get_diagnostics() should include a quiet_hours_active field."""
    diag = notif_manager.get_diagnostics()
    assert "quiet_hours_active" in diag
    assert isinstance(diag["quiet_hours_active"], bool)


def test_diagnostics_quiet_hours_inactive_by_default(notif_manager):
    """Without quiet hours configured, quiet_hours_active should be False."""
    diag = notif_manager.get_diagnostics()
    assert diag["quiet_hours_active"] is False


def test_diagnostics_quiet_hours_active_when_configured(db, mock_event_bus):
    """quiet_hours_active should be True when current time falls in quiet hours."""
    mgr = NotificationManager(db, mock_event_bus, config={}, timezone="UTC")

    # Configure quiet hours covering ALL days and ALL hours (00:00–23:59)
    now = datetime.now(timezone.utc)
    current_day = now.strftime("%A").lower()
    quiet_hours_config = json.dumps([{
        "start": "00:00",
        "end": "23:59",
        "days": [current_day],
    }])

    with db.get_connection("preferences") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_preferences (key, value) VALUES (?, ?)",
            ("quiet_hours", quiet_hours_config),
        )
        conn.commit()

    diag = mgr.get_diagnostics()
    assert diag["quiet_hours_active"] is True


# ============================================================================
# Tests: Dashboard template contains notification mode elements
# ============================================================================


def test_template_contains_notif_mode_element():
    """Dashboard HTML should contain the notifMode span element."""
    from web.template import HTML_TEMPLATE
    html = HTML_TEMPLATE
    assert 'id="notifMode"' in html


def test_template_contains_notif_separator():
    """Dashboard HTML should contain the notification separator element."""
    from web.template import HTML_TEMPLATE
    html = HTML_TEMPLATE
    assert 'id="notifSep"' in html


def test_template_contains_diagnostics_fetch():
    """Dashboard JS should fetch the diagnostics endpoint for notification mode."""
    from web.template import HTML_TEMPLATE
    html = HTML_TEMPLATE
    assert "/api/diagnostics/user-model" in html


def test_template_contains_mode_labels():
    """Dashboard JS should contain labels for each notification mode."""
    from web.template import HTML_TEMPLATE
    html = HTML_TEMPLATE
    assert "All notifications" in html
    assert "Batched" in html
    assert "Minimal" in html
    assert "Quiet hours" in html


def test_template_contains_diag_cache_guard():
    """Dashboard JS should cache diagnostics fetch with a timestamp guard."""
    from web.template import HTML_TEMPLATE
    html = HTML_TEMPLATE
    assert "_lastDiagFetch" in html
    assert "290000" in html
