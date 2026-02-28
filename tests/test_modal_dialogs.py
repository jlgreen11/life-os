"""
Life OS — Tests for Inline Modal Dialog System

Verifies that:
    1. The dashboard template contains the modal overlay HTML component.
    2. The modal CSS classes are present and cover the key structural rules.
    3. The JS helper functions (closeModal, showConfirmModal, showPromptModal)
       exist in the template.
    4. deleteFact() and correctFact() no longer use native browser dialogs
       (confirm() / prompt()) which are blocked on mobile Safari.
    5. iOS APIClient uses the correct endpoint paths and field names.

Background
----------
Previously deleteFact() called window.confirm() and correctFact() called
window.prompt().  Both are synchronous browser dialogs that modern mobile
Safari may suppress.  The fix replaces them with showConfirmModal() and
showPromptModal(), which are styled overlay modals that work everywhere.

The iOS APIClient previously sent the wrong JSON field name for sendCommand
("command" instead of "text") and hit the wrong URL for createTask
("/api/task" instead of "/api/tasks"), causing silent 422 / 404 failures.
"""

import re


def _get_template_source() -> str:
    """Return the full HTML/JS/CSS template source string from web.template."""
    from web.template import HTML_TEMPLATE

    return HTML_TEMPLATE


class TestModalHTMLPresent:
    """Verify the modal overlay HTML exists in the rendered template."""

    def test_modal_overlay_div_present(self):
        """The modal overlay container must be present in the template."""
        src = _get_template_source()
        assert 'id="modalOverlay"' in src, (
            "Modal overlay element missing — required for mobile-safe dialogs"
        )

    def test_modal_title_div_present(self):
        """The modal title placeholder must exist inside the overlay."""
        src = _get_template_source()
        assert 'id="modalTitle"' in src

    def test_modal_body_div_present(self):
        """The modal body placeholder must exist inside the overlay."""
        src = _get_template_source()
        assert 'id="modalBody"' in src

    def test_modal_actions_div_present(self):
        """The modal actions placeholder must exist inside the overlay."""
        src = _get_template_source()
        assert 'id="modalActions"' in src

    def test_modal_overlay_has_close_handler(self):
        """Clicking the overlay backdrop must close the modal."""
        src = _get_template_source()
        assert 'onclick="closeModal()"' in src


class TestModalCSSPresent:
    """Verify that the key CSS rules for the modal are in the template."""

    def test_modal_overlay_css_class(self):
        """.modal-overlay CSS must be defined."""
        src = _get_template_source()
        assert ".modal-overlay" in src

    def test_modal_box_css_class(self):
        """.modal-box CSS must be defined."""
        src = _get_template_source()
        assert ".modal-box" in src

    def test_modal_btn_danger_css_class(self):
        """.modal-btn-danger CSS must be defined for destructive actions."""
        src = _get_template_source()
        assert ".modal-btn-danger" in src

    def test_modal_overlay_visible_class(self):
        """.modal-overlay.visible must toggle display: flex to show the modal."""
        src = _get_template_source()
        assert ".modal-overlay.visible" in src

    def test_modal_uses_design_tokens(self):
        """Modal CSS must reference CSS custom properties for theming."""
        src = _get_template_source()
        # The box uses --bg-card so it matches the app's colour scheme.
        assert "--bg-card" in src


class TestModalJSFunctions:
    """Verify that the three modal JS functions are defined in the template."""

    def test_close_modal_function_defined(self):
        """closeModal() must be defined as a named function."""
        src = _get_template_source()
        assert "function closeModal()" in src

    def test_show_confirm_modal_function_defined(self):
        """showConfirmModal() must be defined as a named function."""
        src = _get_template_source()
        assert "function showConfirmModal(" in src

    def test_show_prompt_modal_function_defined(self):
        """showPromptModal() must be defined as a named function."""
        src = _get_template_source()
        assert "function showPromptModal(" in src

    def test_close_modal_removes_visible_class(self):
        """closeModal() must remove the 'visible' class from the overlay."""
        src = _get_template_source()
        assert "classList.remove('visible')" in src

    def test_show_confirm_modal_adds_visible_class(self):
        """showConfirmModal() must add the 'visible' class to show the overlay."""
        src = _get_template_source()
        assert "classList.add('visible')" in src

    def test_prompt_modal_supports_enter_key(self):
        """showPromptModal() must allow Enter key to confirm without mouse click."""
        src = _get_template_source()
        assert "'Enter'" in src


class TestNativeBrowserDialogsRemoved:
    """Verify that deleteFact() and correctFact() no longer call native dialogs."""

    def test_delete_fact_does_not_call_confirm(self):
        """deleteFact() must use showConfirmModal(), not window.confirm().

        Native confirm() is blocked on mobile Safari in certain contexts
        and produces an ugly unstyled dialog that breaks the app's UI.
        """
        src = _get_template_source()

        # Find the deleteFact function body
        match = re.search(
            r"function deleteFact\(key\)\s*\{(.+?)(?=\n    function )",
            src,
            re.DOTALL,
        )
        assert match, "deleteFact() function not found in template"
        body = match.group(1)

        assert "showConfirmModal(" in body, (
            "deleteFact() must use showConfirmModal() — not confirm()"
        )
        assert "confirm(" not in body, (
            "deleteFact() must not call native confirm() — blocked on mobile Safari"
        )

    def test_correct_fact_does_not_call_prompt(self):
        """correctFact() must use showPromptModal(), not window.prompt().

        Native prompt() is blocked on mobile Safari in certain contexts
        and produces an ugly unstyled dialog that breaks the app's UI.
        """
        src = _get_template_source()

        # Find the correctFact function body
        match = re.search(
            r"function correctFact\(key\)\s*\{(.+?)(?=\n    function )",
            src,
            re.DOTALL,
        )
        assert match, "correctFact() function not found in template"
        body = match.group(1)

        assert "showPromptModal(" in body, (
            "correctFact() must use showPromptModal() — not prompt()"
        )
        assert "= prompt(" not in body, (
            "correctFact() must not call native prompt() — blocked on mobile Safari"
        )

    def test_delete_fact_uses_danger_style(self):
        """deleteFact modal must use danger styling (red confirm button)."""
        src = _get_template_source()
        match = re.search(
            r"function deleteFact\(key\)\s*\{(.+?)(?=\n    function )",
            src,
            re.DOTALL,
        )
        assert match
        body = match.group(1)
        # The fifth argument to showConfirmModal is the danger flag
        assert "true" in body, (
            "deleteFact must pass danger=true to showConfirmModal for red confirm button"
        )


class TestIOSAPIClientFixes:
    """Verify that the iOS APIClient uses correct endpoint paths and field names."""

    def _get_swift_source(self) -> str:
        """Read the Swift API client source."""
        with open("ios/LifeOS/Services/APIClient.swift") as f:
            return f.read()

    def test_send_command_uses_text_field(self):
        """sendCommand() must send 'text' key, not 'command' key.

        The backend CommandRequest Pydantic schema has a 'text' field.
        Sending 'command' returns 422 Unprocessable Entity and the command
        bar silently fails on iOS.
        """
        src = self._get_swift_source()
        # Find the sendCommand method
        match = re.search(r"func sendCommand\([^}]+\}", src, re.DOTALL)
        assert match, "sendCommand function not found in APIClient.swift"
        body = match.group(0)
        # Check the dictionary literal contains "text" as a key
        assert '["text"' in body or '"text":' in body, (
            "sendCommand must use 'text' field to match CommandRequest schema"
        )
        # Make sure the dict key is not "command" — check for Swift dict literal syntax
        assert '["command"' not in body and '"command":' not in body, (
            "sendCommand must not use 'command' as a dict key — that field causes 422 errors"
        )

    def test_create_task_uses_plural_endpoint(self):
        """createTask() must hit /api/tasks (plural), not /api/task.

        /api/task does not exist — the correct endpoint is /api/tasks.
        Using the wrong path returns 404 Not Found and task creation
        silently fails on iOS.
        """
        src = self._get_swift_source()
        match = re.search(r"func createTask\([^}]+\}", src, re.DOTALL)
        assert match, "createTask function not found in APIClient.swift"
        body = match.group(0)
        assert '"/api/tasks"' in body, (
            "createTask must hit /api/tasks — the plural endpoint that exists"
        )
        assert '"/api/task"' not in body, (
            "createTask must not hit /api/task — that path returns 404"
        )
