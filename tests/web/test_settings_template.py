"""Smoke + contract tests for ``web/templates/settings.html``
and the two detail-pane partials (``connector_edit_form``,
``connector_test_result``).

Scope (NEXT_TASKS.md Week 10 task 3 acceptance):

- ``settings.html`` declares ``data-active-tab="settings"`` so base.html
  marks the Settings nav anchor current.
- Renders all three top-level sections (header, connectors, preferences)
  plus the detail-pane aside.
- Connector rows render the status dot, kind, last-sync, enabled
  checkbox, Test / Edit buttons, and a per-row test-result slot.
- Empty-state copy renders when the connectors list is empty.
- Preferences form renders the three fields (quiet hours, autonomy,
  proactivity) with defaults pre-filled.
- HTMX wiring on the enabled toggle / Test / Edit / preference inputs
  points at the locked routes.
- No progress bars / SVG charts — DESIGN.md § "No horoscope".
- Edit-form partial renders config and secret fields, with secret
  values blanked and placeholder ``********``.
- Test-result partial renders the success / failure variants.

Structural assertions only — no full-HTML snapshots.
"""

from __future__ import annotations

import re

from web.rendering import render

REF_NOW = 1_777_204_800  # 2026-04-22T12:00:00Z


def _empty_settings() -> dict:
    return {
        "connectors": [],
        "preferences": {},
    }


def _populated_settings() -> dict:
    return {
        "connectors": [
            {
                "id": "proton_mail",
                "kind": "proton_mail",
                "enabled": True,
                "status": "ok",
                "last_sync_at": REF_NOW - 3600,
                "last_error": None,
            },
            {
                "id": "signal",
                "kind": "signal",
                "enabled": False,
                "status": "disabled",
                "last_sync_at": None,
                "last_error": None,
            },
            {
                "id": "imessage",
                "kind": "imessage",
                "enabled": True,
                "status": "error",
                "last_sync_at": REF_NOW - 86400,
                "last_error": "auth failed",
            },
        ],
        "preferences": {
            "quiet_hours_start": "21:30",
            "quiet_hours_end": "06:45",
            "autonomy_level": 0.7,
            "proactivity": 0.3,
        },
    }


# ---------------------------------------------------------------------------
# Base wiring
# ---------------------------------------------------------------------------


def test_settings_page_declares_active_tab() -> None:
    html = render("settings.html", _empty_settings())
    assert 'data-active-tab="settings"' in html


def test_settings_page_marks_settings_tab_active_in_nav() -> None:
    html = render("settings.html", _empty_settings())
    settings_anchor = html.split('data-tab="settings"', 1)[1].split("</a>", 1)[0]
    assert 'aria-current="page"' in settings_anchor
    now_anchor = html.split('data-tab="now"', 1)[1].split("</a>", 1)[0]
    assert 'aria-current="page"' not in now_anchor


def test_settings_page_renders_all_sections() -> None:
    html = render("settings.html", _empty_settings())
    assert 'data-section="settings-header"' in html
    assert 'data-section="connectors"' in html
    assert 'data-section="preferences"' in html
    assert 'data-section="detail-pane"' in html


def test_settings_page_has_no_progress_or_pie_chart_markup() -> None:
    html = render("settings.html", _populated_settings())
    assert "progressbar" not in html
    assert 'role="progressbar"' not in html
    assert "<svg" not in html


# ---------------------------------------------------------------------------
# Connectors list
# ---------------------------------------------------------------------------


def test_connector_list_renders_each_row() -> None:
    html = render("settings.html", _populated_settings())
    assert html.count('data-slot="connector-row"') == 3
    assert 'data-connector-id="proton_mail"' in html
    assert 'data-connector-id="signal"' in html
    assert 'data-connector-id="imessage"' in html


def test_connector_list_renders_status_dot_colors() -> None:
    html = render("settings.html", _populated_settings())
    # Each connector row has one status dot.
    assert html.count('data-slot="status-dot"') == 3
    # Status "ok" → success, "error" → error, "disabled" → tertiary.
    ok_row = html.split('data-connector-id="proton_mail"', 1)[1].split("</li>", 1)[0]
    err_row = html.split('data-connector-id="imessage"', 1)[1].split("</li>", 1)[0]
    disabled_row = html.split('data-connector-id="signal"', 1)[1].split("</li>", 1)[0]
    assert "var(--success)" in ok_row
    assert "var(--error)" in err_row
    assert "var(--text-tertiary)" in disabled_row


def test_connector_row_renders_last_sync_and_never_label() -> None:
    html = render("settings.html", _populated_settings())
    proton_row = html.split('data-connector-id="proton_mail"', 1)[1].split("</li>", 1)[0]
    signal_row = html.split('data-connector-id="signal"', 1)[1].split("</li>", 1)[0]
    # proton_mail was synced — renders a "Last sync" label.
    assert "Last sync" in proton_row
    # signal was never synced — renders the "Never" empty-state.
    assert "Never synced" in signal_row


def test_connector_row_renders_last_error_when_present() -> None:
    html = render("settings.html", _populated_settings())
    err_row = html.split('data-connector-id="imessage"', 1)[1].split("</li>", 1)[0]
    assert 'data-slot="last-error"' in err_row
    assert "auth failed" in err_row


def test_connector_row_enabled_toggle_wired_to_patch() -> None:
    html = render("settings.html", _populated_settings())
    proton_row = html.split('data-connector-id="proton_mail"', 1)[1].split("</li>", 1)[0]
    assert 'hx-patch="/api/connectors/proton_mail"' in proton_row
    # Checkbox honours the incoming enabled flag.
    assert re.search(r"<input[^>]*type=\"checkbox\"[^>]*checked", proton_row)
    signal_row = html.split('data-connector-id="signal"', 1)[1].split("</li>", 1)[0]
    # Disabled row must NOT have a "checked" attribute on its enabled checkbox.
    enabled_input = re.search(
        r"<input[^>]*data-slot=\"enabled-checkbox\"[^>]*>",
        signal_row,
    )
    assert enabled_input is not None
    assert "checked" not in enabled_input.group(0)


def test_connector_row_test_button_wired_to_html_endpoint() -> None:
    html = render("settings.html", _populated_settings())
    proton_row = html.split('data-connector-id="proton_mail"', 1)[1].split("</li>", 1)[0]
    assert 'data-slot="test-button"' in proton_row
    assert 'hx-post="/settings/connectors/proton_mail/test"' in proton_row
    # The Test button targets the per-row result slot, not the detail pane,
    # so a Test action never displaces an open edit form.
    assert 'data-slot="test-result"' in proton_row
    assert 'data-for="proton_mail"' in proton_row


def test_connector_row_edit_button_targets_detail_pane() -> None:
    html = render("settings.html", _populated_settings())
    proton_row = html.split('data-connector-id="proton_mail"', 1)[1].split("</li>", 1)[0]
    assert 'data-slot="edit-button"' in proton_row
    assert 'hx-get="/settings/connectors/proton_mail/edit"' in proton_row
    assert 'hx-target="#settings-detail-pane"' in proton_row


def test_connector_list_empty_state() -> None:
    html = render("settings.html", _empty_settings())
    assert 'data-empty="connectors"' in html
    assert "No connectors configured yet." in html


# ---------------------------------------------------------------------------
# Detail pane
# ---------------------------------------------------------------------------


def test_detail_pane_initial_state() -> None:
    html = render("settings.html", _empty_settings())
    assert 'id="settings-detail-pane"' in html
    assert 'data-empty="detail-pane"' in html
    assert "Select a connector to edit." in html


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


def test_preferences_form_renders_all_three_fields() -> None:
    html = render("settings.html", _populated_settings())
    assert 'data-pref-key="quiet_hours"' in html
    assert 'data-pref-key="autonomy_level"' in html
    assert 'data-pref-key="proactivity"' in html
    assert 'data-slot="quiet-hours-start"' in html
    assert 'data-slot="quiet-hours-end"' in html
    assert 'data-slot="autonomy-slider"' in html
    assert 'data-slot="proactivity-slider"' in html


def test_preferences_form_prefills_values() -> None:
    html = render("settings.html", _populated_settings())
    assert 'value="21:30"' in html
    assert 'value="06:45"' in html
    # Sliders render their numeric value verbatim.
    assert 'value="0.7"' in html
    assert 'value="0.3"' in html
    # Output monitors display the pretty-formatted value.
    assert ">0.70<" in html
    assert ">0.30<" in html


def test_preferences_form_defaults_when_empty() -> None:
    html = render("settings.html", _empty_settings())
    # Defaults come from the template's |default() filters.
    assert 'value="22:00"' in html
    assert 'value="07:00"' in html
    assert 'value="0.5"' in html
    assert ">0.50<" in html


def test_preferences_form_posts_each_field_to_preferences_endpoint() -> None:
    html = render("settings.html", _populated_settings())
    # Each field uses hx-post to /api/preferences with its own (key, value).
    assert html.count('hx-post="/api/preferences"') >= 4
    assert "quiet_hours_start" in html
    assert "quiet_hours_end" in html
    assert "autonomy_level" in html
    assert "proactivity" in html


# ---------------------------------------------------------------------------
# Edit-form partial
# ---------------------------------------------------------------------------


def _connector_edit_view() -> dict:
    return {
        "id": "proton_mail",
        "kind": "proton_mail",
        "enabled": True,
        "status": "ok",
        "config": {
            "username": "alice@example.com",
            "server": "127.0.0.1",
        },
        "secret_keys": ["password"],
    }


def test_edit_form_renders_config_and_secret_fields() -> None:
    html = render(
        "partials/connector_edit_form.html",
        {"connector": _connector_edit_view()},
    )
    # Non-secret config fields render as text inputs with values.
    assert 'data-slot="config-field"' in html
    assert 'name="config.username"' in html
    assert 'value="alice@example.com"' in html
    assert 'name="config.server"' in html
    # Secret field renders as password input with blank value + placeholder.
    assert 'data-slot="secret-field"' in html
    assert 'name="secrets.password"' in html
    secret_input = re.search(
        r'<input[^>]*data-slot="secret-field"[^>]*>',
        html,
    )
    assert secret_input is not None
    # The existing Fernet ciphertext must NEVER round-trip through the client.
    assert 'value=""' in secret_input.group(0)
    assert 'placeholder="********"' in secret_input.group(0)


def test_edit_form_submits_patch_to_api_connectors() -> None:
    html = render(
        "partials/connector_edit_form.html",
        {"connector": _connector_edit_view()},
    )
    assert 'hx-patch="/api/connectors/proton_mail"' in html
    assert 'data-slot="save-button"' in html
    assert 'data-slot="cancel-button"' in html


def test_edit_form_cancel_resets_detail_pane() -> None:
    html = render(
        "partials/connector_edit_form.html",
        {"connector": _connector_edit_view()},
    )
    cancel = re.search(
        r'<button[^>]*data-slot="cancel-button"[^>]*>',
        html,
    )
    assert cancel is not None
    assert 'hx-get="/settings/detail-empty"' in cancel.group(0)
    assert 'hx-target="#settings-detail-pane"' in cancel.group(0)


# ---------------------------------------------------------------------------
# Test-result partial
# ---------------------------------------------------------------------------


def test_test_result_renders_success_variant() -> None:
    html = render(
        "partials/connector_test_result.html",
        {"result": {"ok": True, "message": "dry-run complete"}},
    )
    assert 'data-slot="test-ok"' in html
    assert "dry-run complete" in html
    assert "var(--success)" in html


def test_test_result_renders_failure_variant() -> None:
    html = render(
        "partials/connector_test_result.html",
        {"result": {"ok": False, "message": "auth failed"}},
    )
    assert 'data-slot="test-fail"' in html
    assert "auth failed" in html
    assert "var(--error)" in html


def test_test_result_renders_none_as_empty_state() -> None:
    html = render("partials/connector_test_result.html", {"result": None})
    assert "No test result." in html
