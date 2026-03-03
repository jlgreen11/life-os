"""
Tests for client-side calendar conflict detection and visual indicators.

The dashboard design doc specifies "conflict warning if overlapping" for
calendar cards.  The backend's ConflictDetector publishes
``calendar.conflict.detected`` events, but the calendar *UI* also needs
visual indicators for overlapping events.

These tests verify that the HTML template (``web/template.py``) contains:
  - CSS classes for conflict styling (``.conflict``, ``.conflict-badge``,
    ``.calendar-conflict-dot``)
  - The ``detectDayConflicts()`` JavaScript helper function
  - Integration of conflict detection into ``renderDayDetail()`` and
    ``renderCalendarEvents()``

Pattern: string/pattern matching against the template source, following
the approach used in ``tests/web/test_toggle_card_dom.py``.
"""

import re

import pytest

from web.template import HTML_TEMPLATE as TEMPLATE


# ---------------------------------------------------------------------------
# CSS class tests
# ---------------------------------------------------------------------------


class TestConflictCSS:
    """Verify conflict-related CSS classes exist in the template."""

    def test_calendar_event_conflict_class(self):
        """The `.calendar-event.conflict` CSS selector must exist for month-grid pills."""
        assert ".calendar-event.conflict" in TEMPLATE, (
            "CSS must contain .calendar-event.conflict for styling conflicting event pills"
        )

    def test_calendar_detail_card_conflict_class(self):
        """The `.calendar-detail-card.conflict` CSS selector must exist for day detail cards."""
        assert ".calendar-detail-card.conflict" in TEMPLATE, (
            "CSS must contain .calendar-detail-card.conflict for styling conflicting day-detail cards"
        )

    def test_conflict_badge_class(self):
        """The `.conflict-badge` CSS class must exist for the overlap indicator chip."""
        assert ".conflict-badge" in TEMPLATE, (
            "CSS must contain .conflict-badge class for the overlap warning chip"
        )

    def test_calendar_conflict_dot_class(self):
        """The `.calendar-conflict-dot` CSS class must exist for month-grid day indicators."""
        assert ".calendar-conflict-dot" in TEMPLATE, (
            "CSS must contain .calendar-conflict-dot class for day-level conflict dots"
        )

    def test_conflict_badge_has_red_color(self):
        """The conflict-badge should use a red/warning color scheme."""
        # Find the .conflict-badge CSS block
        idx = TEMPLATE.find(".conflict-badge")
        assert idx >= 0
        # Read the next ~200 chars to check for red color
        block = TEMPLATE[idx : idx + 300]
        assert "#e74c3c" in block, (
            "conflict-badge CSS should use the red accent color #e74c3c"
        )

    def test_conflict_event_has_red_border(self):
        """The .calendar-event.conflict class should have a red left border."""
        idx = TEMPLATE.find(".calendar-event.conflict")
        assert idx >= 0
        block = TEMPLATE[idx : idx + 200]
        assert "border-left" in block, (
            ".calendar-event.conflict CSS should include a border-left for visual distinction"
        )


# ---------------------------------------------------------------------------
# JavaScript helper function tests
# ---------------------------------------------------------------------------


class TestDetectDayConflictsFunction:
    """Verify the detectDayConflicts() JavaScript function exists and is correct."""

    def test_function_exists(self):
        """The detectDayConflicts function must be defined in the template."""
        assert "function detectDayConflicts" in TEMPLATE, (
            "Template must define a detectDayConflicts() function"
        )

    def test_function_returns_set(self):
        """detectDayConflicts must create and return a Set of conflicting indices."""
        func_idx = TEMPLATE.find("function detectDayConflicts")
        assert func_idx >= 0
        # Extract the function body
        func_start = TEMPLATE.find("{", func_idx)
        depth, pos = 0, func_start
        while pos < len(TEMPLATE):
            if TEMPLATE[pos] == "{":
                depth += 1
            elif TEMPLATE[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        func_body = TEMPLATE[func_idx : pos + 1]

        assert "new Set()" in func_body, (
            "detectDayConflicts must create a Set to track conflicting indices"
        )
        assert "return conflicting" in func_body, (
            "detectDayConflicts must return the conflicting Set"
        )

    def test_function_skips_all_day_events(self):
        """detectDayConflicts must skip all-day events from conflict detection."""
        func_idx = TEMPLATE.find("function detectDayConflicts")
        assert func_idx >= 0
        func_start = TEMPLATE.find("{", func_idx)
        depth, pos = 0, func_start
        while pos < len(TEMPLATE):
            if TEMPLATE[pos] == "{":
                depth += 1
            elif TEMPLATE[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        func_body = TEMPLATE[func_idx : pos + 1]

        assert "is_all_day" in func_body, (
            "detectDayConflicts must check is_all_day to skip all-day events"
        )

    def test_function_uses_overlap_logic(self):
        """detectDayConflicts must implement the standard overlap check: A.start < B.end AND B.start < A.end."""
        func_idx = TEMPLATE.find("function detectDayConflicts")
        assert func_idx >= 0
        func_start = TEMPLATE.find("{", func_idx)
        depth, pos = 0, func_start
        while pos < len(TEMPLATE):
            if TEMPLATE[pos] == "{":
                depth += 1
            elif TEMPLATE[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        func_body = TEMPLATE[func_idx : pos + 1]

        assert "start_time" in func_body and "end_time" in func_body, (
            "detectDayConflicts must compare start_time and end_time for overlap detection"
        )


# ---------------------------------------------------------------------------
# Integration with renderDayDetail
# ---------------------------------------------------------------------------


class TestRenderDayDetailConflicts:
    """Verify renderDayDetail uses conflict detection to mark overlapping events."""

    def _get_render_day_detail_body(self):
        """Extract the renderDayDetail function body."""
        func_idx = TEMPLATE.find("function renderDayDetail(day)")
        assert func_idx >= 0, "renderDayDetail function must exist"
        func_start = TEMPLATE.find("{", func_idx)
        depth, pos = 0, func_start
        while pos < len(TEMPLATE):
            if TEMPLATE[pos] == "{":
                depth += 1
            elif TEMPLATE[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        return TEMPLATE[func_idx : pos + 1]

    def test_calls_detect_day_conflicts(self):
        """renderDayDetail must call detectDayConflicts() on the day's events."""
        body = self._get_render_day_detail_body()
        assert "detectDayConflicts" in body, (
            "renderDayDetail must call detectDayConflicts to identify overlapping events"
        )

    def test_adds_conflict_class_to_detail_card(self):
        """renderDayDetail must conditionally add the 'conflict' class to detail cards."""
        body = self._get_render_day_detail_body()
        assert "conflict" in body, (
            "renderDayDetail must add a 'conflict' class to calendar-detail-card elements"
        )

    def test_adds_conflict_badge(self):
        """renderDayDetail must add a conflict-badge span for overlapping events."""
        body = self._get_render_day_detail_body()
        assert "conflict-badge" in body, (
            "renderDayDetail must include a conflict-badge indicator for overlapping events"
        )

    def test_overlap_label_text(self):
        """The conflict badge should display 'Overlap' text."""
        body = self._get_render_day_detail_body()
        assert "Overlap" in body, (
            "renderDayDetail must show 'Overlap' text in the conflict badge"
        )


# ---------------------------------------------------------------------------
# Integration with renderCalendarEvents (month grid)
# ---------------------------------------------------------------------------


class TestRenderCalendarEventsConflicts:
    """Verify renderCalendarEvents marks conflicting events in the month grid."""

    def _get_render_calendar_events_body(self):
        """Extract the renderCalendarEvents function body."""
        func_idx = TEMPLATE.find("function renderCalendarEvents()")
        assert func_idx >= 0, "renderCalendarEvents function must exist"
        func_start = TEMPLATE.find("{", func_idx)
        depth, pos = 0, func_start
        while pos < len(TEMPLATE):
            if TEMPLATE[pos] == "{":
                depth += 1
            elif TEMPLATE[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        return TEMPLATE[func_idx : pos + 1]

    def test_uses_detect_day_conflicts(self):
        """renderCalendarEvents must use detectDayConflicts for month grid conflict detection."""
        body = self._get_render_calendar_events_body()
        assert "detectDayConflicts" in body, (
            "renderCalendarEvents must call detectDayConflicts"
        )

    def test_adds_conflict_class_to_event_pills(self):
        """renderCalendarEvents must conditionally add the 'conflict' CSS class to event pills."""
        body = self._get_render_calendar_events_body()
        assert "conflict" in body, (
            "renderCalendarEvents must add a 'conflict' class to conflicting event pills"
        )

    def test_adds_conflict_dot_to_day_cells(self):
        """renderCalendarEvents must add a conflict dot indicator to day cells with overlaps."""
        body = self._get_render_calendar_events_body()
        assert "calendar-conflict-dot" in body, (
            "renderCalendarEvents must add a calendar-conflict-dot element to days with conflicts"
        )
