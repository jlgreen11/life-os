"""
Tests for the mobile sidebar feature (Task 8 from ui-engagement-fixes plan).

The AI sidebar is hidden via CSS at < 900px viewports.  This feature adds:
  - A floating action button (#mobileSidebarFab) that opens the sidebar as a
    full-screen overlay by toggling the `mobile-open` class on #aiSidebar.
  - A close button (.mobile-sidebar-close) inside the sidebar.
  - The `toggleMobileSidebar()` JS function that manages open/close state.
  - Restored badge counts in the 900px breakpoint (smaller font, still visible).

These tests validate the template HTML/CSS/JS in web/template.py.
"""

import pytest

from web.template import HTML_TEMPLATE as TEMPLATE


# ---------------------------------------------------------------------------
# HTML structure tests
# ---------------------------------------------------------------------------


class TestMobileSidebarHTML:
    """Verify the required HTML elements are present in the template."""

    def test_mobile_fab_button_exists(self):
        """The FAB button #mobileSidebarFab must be present in the HTML body."""
        assert 'id="mobileSidebarFab"' in TEMPLATE, (
            "Missing mobile sidebar FAB button: expected id=\"mobileSidebarFab\""
        )

    def test_mobile_fab_has_correct_class(self):
        """The FAB must use the .mobile-sidebar-fab CSS class."""
        assert 'class="mobile-sidebar-fab"' in TEMPLATE, (
            "FAB button must have class=\"mobile-sidebar-fab\" for CSS targeting"
        )

    def test_mobile_fab_calls_toggle_function(self):
        """FAB onclick must invoke toggleMobileSidebar()."""
        assert 'onclick="toggleMobileSidebar()"' in TEMPLATE, (
            "FAB button must call toggleMobileSidebar() via onclick"
        )

    def test_mobile_sidebar_close_button_exists(self):
        """A close button (.mobile-sidebar-close) must be inside #aiSidebar."""
        sidebar_start = TEMPLATE.find('id="aiSidebar"')
        sidebar_end = TEMPLATE.find('</aside>', sidebar_start)
        sidebar_html = TEMPLATE[sidebar_start:sidebar_end]
        assert 'mobile-sidebar-close' in sidebar_html, (
            "Expected .mobile-sidebar-close div inside #aiSidebar for the mobile close button"
        )

    def test_mobile_close_calls_toggle_function(self):
        """The close button inside the sidebar must also call toggleMobileSidebar()."""
        # Search for the HTML <div class="mobile-sidebar-close"> element,
        # not the CSS rule.  The div comes after the <aside> tag.
        sidebar_start = TEMPLATE.find('id="aiSidebar"')
        sidebar_end = TEMPLATE.find('</aside>', sidebar_start)
        sidebar_html = TEMPLATE[sidebar_start:sidebar_end]
        # The close button must call toggleMobileSidebar()
        assert 'toggleMobileSidebar()' in sidebar_html, (
            "Close button inside #aiSidebar must call toggleMobileSidebar()"
        )


# ---------------------------------------------------------------------------
# CSS tests
# ---------------------------------------------------------------------------


class TestMobileSidebarCSS:
    """Verify the required CSS rules are present."""

    def test_mobile_open_overlay_css_exists(self):
        """CSS rule for .ai-sidebar.mobile-open must define overlay positioning."""
        assert '.ai-sidebar.mobile-open' in TEMPLATE, (
            "Missing .ai-sidebar.mobile-open CSS rule for the full-screen overlay"
        )

    def test_mobile_open_uses_fixed_positioning(self):
        """The overlay must use position:fixed to cover the full viewport."""
        # Find the mobile-open block and look for position:fixed nearby
        idx = TEMPLATE.find('.ai-sidebar.mobile-open')
        snippet = TEMPLATE[idx:idx + 400]
        assert 'position: fixed' in snippet or 'position:fixed' in snippet, (
            ".ai-sidebar.mobile-open must use position:fixed for full-screen overlay"
        )

    def test_mobile_sidebar_fab_hidden_by_default(self):
        """FAB must be display:none outside the 900px media query."""
        # The base .mobile-sidebar-fab rule sets display:none; the 900px query
        # overrides it to display:block.
        fab_rule_idx = TEMPLATE.find('.mobile-sidebar-fab {')
        assert fab_rule_idx != -1, "Missing base .mobile-sidebar-fab CSS rule"
        snippet = TEMPLATE[fab_rule_idx:fab_rule_idx + 80]
        assert 'display: none' in snippet or 'display:none' in snippet, (
            ".mobile-sidebar-fab must be display:none in the base rule (hidden on wide screens)"
        )

    def test_mobile_fab_shown_in_900px_breakpoint(self):
        """FAB must be set to display:block inside the 900px media query."""
        # Find the 900px media query block
        media_idx = TEMPLATE.find('@media (max-width: 900px)')
        assert media_idx != -1, "Missing @media (max-width: 900px) breakpoint"
        # The FAB display:block rule must appear after the media query open
        media_snippet = TEMPLATE[media_idx:media_idx + 2000]
        assert 'display: block' in media_snippet, (
            ".mobile-sidebar-fab must be display:block inside the 900px media query"
        )

    def test_topic_badge_visible_in_900px_breakpoint(self):
        """topic-badge must NOT use display:none !important in the 900px breakpoint.

        Previous behaviour hid all badge counts on narrow screens.  After this fix
        badges are resized (smaller font/height) but remain visible.
        """
        media_idx = TEMPLATE.find('@media (max-width: 900px)')
        assert media_idx != -1
        media_snippet = TEMPLATE[media_idx:media_idx + 2000]
        # The old rule was: .topic-badge { display: none !important; }
        # It must NOT appear inside the 900px breakpoint any more.
        assert '.topic-badge { display: none !important; }' not in media_snippet, (
            "topic badges must not be hidden with 'display:none !important' in the 900px breakpoint"
        )
        assert 'display: none !important' not in media_snippet, (
            "No element should use display:none !important inside the 900px media query after this fix"
        )


# ---------------------------------------------------------------------------
# JavaScript tests
# ---------------------------------------------------------------------------


class TestMobileSidebarJS:
    """Verify the toggleMobileSidebar JS function is correctly defined."""

    def test_toggle_function_defined(self):
        """toggleMobileSidebar() must be defined as a JS function."""
        assert 'function toggleMobileSidebar()' in TEMPLATE, (
            "Missing toggleMobileSidebar() function in template JS"
        )

    def test_toggle_function_toggles_mobile_open_class(self):
        """The function must toggle the mobile-open class on #aiSidebar."""
        fn_idx = TEMPLATE.find('function toggleMobileSidebar()')
        assert fn_idx != -1
        fn_body = TEMPLATE[fn_idx:fn_idx + 500]
        assert 'mobile-open' in fn_body, (
            "toggleMobileSidebar() must toggle the 'mobile-open' class"
        )

    def test_toggle_function_loads_sidebar_data_on_open(self):
        """On open, the function should trigger sidebar data loads (mood, predictions)."""
        fn_idx = TEMPLATE.find('function toggleMobileSidebar()')
        fn_body = TEMPLATE[fn_idx:fn_idx + 600]
        # Must call at least one data-load function when isOpen is true
        assert 'loadMood' in fn_body or 'loadPredictions' in fn_body, (
            "toggleMobileSidebar() must trigger sidebar data loads when opening"
        )

    def test_toggle_function_before_end_of_script(self):
        """The function must be inside the <script> block (not after </script>)."""
        fn_idx = TEMPLATE.find('function toggleMobileSidebar()')
        script_end = TEMPLATE.find('</script>', fn_idx)
        # Check it's between the function definition and </script>
        assert script_end > fn_idx, (
            "toggleMobileSidebar() must appear before </script>"
        )
