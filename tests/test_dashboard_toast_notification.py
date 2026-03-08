"""
Tests for the dashboard real-time notification toast feature.

Verifies that the dashboard template contains the required DOM elements,
CSS styles, and JavaScript functions for showing toast notifications
when WebSocket events arrive for notification.created / notification.delivered.
"""

from web.template import HTML_TEMPLATE


# ---------------------------------------------------------------------------
# Toast container
# ---------------------------------------------------------------------------


def test_template_contains_toast_container():
    """The dashboard must include a #toastContainer element for toast display."""
    assert 'id="toastContainer"' in HTML_TEMPLATE


def test_template_has_toast_container_class():
    """The toast container must use the .toast-container CSS class."""
    assert "toast-container" in HTML_TEMPLATE


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


def test_template_defines_toast_css():
    """Toast-related CSS classes must be defined in the template."""
    assert ".toast {" in HTML_TEMPLATE or ".toast{" in HTML_TEMPLATE


def test_template_defines_priority_css_classes():
    """Priority-specific CSS classes for color-coding toasts."""
    assert ".toast.priority-critical" in HTML_TEMPLATE
    assert ".toast.priority-high" in HTML_TEMPLATE
    assert ".toast.priority-normal" in HTML_TEMPLATE
    assert ".toast.priority-low" in HTML_TEMPLATE


def test_template_defines_toast_slide_in_animation():
    """A slide-in animation keyframe must be defined for toasts."""
    assert "toast-slide-in" in HTML_TEMPLATE


def test_template_defines_toast_fade_out():
    """A fade-out class must be defined for dismissing toasts."""
    assert ".toast.fade-out" in HTML_TEMPLATE


def test_template_has_responsive_toast_styles():
    """Toast styles should adapt on mobile viewports (<768px)."""
    # The media query for mobile toast styling should exist.
    assert "max-width: 768px" in HTML_TEMPLATE


# ---------------------------------------------------------------------------
# JavaScript functions
# ---------------------------------------------------------------------------


def test_template_defines_show_notification_toast_function():
    """The showNotificationToast() JS function must be defined."""
    assert "function showNotificationToast" in HTML_TEMPLATE


def test_template_defines_dismiss_toast_function():
    """The dismissToast() JS function must be defined."""
    assert "function dismissToast" in HTML_TEMPLATE


def test_template_has_max_toasts_constant():
    """MAX_TOASTS constant should limit visible toasts."""
    assert "MAX_TOASTS" in HTML_TEMPLATE


def test_template_has_toast_duration_constant():
    """TOAST_DURATION constant should control auto-dismiss timing."""
    assert "TOAST_DURATION" in HTML_TEMPLATE


# ---------------------------------------------------------------------------
# WebSocket integration
# ---------------------------------------------------------------------------


def test_websocket_handler_handles_notification_created():
    """The WS message handler must trigger toasts for notification.created events."""
    assert "notification.created" in HTML_TEMPLATE
    assert "showNotificationToast" in HTML_TEMPLATE


def test_websocket_handler_handles_notification_delivered():
    """The WS message handler must trigger toasts for notification.delivered events."""
    assert "notification.delivered" in HTML_TEMPLATE


def test_toast_fetches_from_inbox_feed():
    """The toast function should fetch notification details from the inbox feed endpoint."""
    assert "/api/dashboard/feed?topic=inbox&limit=1" in HTML_TEMPLATE


def test_toast_uses_safe_dom_methods():
    """Toast creation should use safe DOM methods (textContent) instead of innerHTML."""
    # The toast-building code uses createElement + textContent, not innerHTML.
    assert "titleEl.textContent" in HTML_TEMPLATE
    assert "bodyEl.textContent" in HTML_TEMPLATE
