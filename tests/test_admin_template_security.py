"""Tests for XSS prevention and HTTP error handling in the admin connector template.

Validates defense-in-depth measures:
1. escapeHtml() function exists in the template
2. field.name values are escaped via escapeHtml() in all render paths
3. field.help_text values are escaped via escapeHtml() in all render paths
4. testConnector() checks res.ok before inspecting data.success
"""

import re

from web.admin_template import ADMIN_HTML_TEMPLATE


# ---------------------------------------------------------------------------
# 1. escapeHtml function exists
# ---------------------------------------------------------------------------


def test_escape_html_function_defined():
    """The template must define an escapeHtml() function that escapes &, \", <, >."""
    assert "function escapeHtml(s)" in ADMIN_HTML_TEMPLATE, (
        "ADMIN_HTML_TEMPLATE should define an escapeHtml() function"
    )
    # Verify it escapes the critical characters
    assert "&amp;" in ADMIN_HTML_TEMPLATE
    assert "&lt;" in ADMIN_HTML_TEMPLATE
    assert "&gt;" in ADMIN_HTML_TEMPLATE


# ---------------------------------------------------------------------------
# 2. field.name is escaped in all render paths
# ---------------------------------------------------------------------------


def test_field_name_escaped_in_boolean_label():
    """The boolean checkbox label must use escapeHtml(field.name), not raw field.name."""
    # The boolean branch has: <label for="f_${field.name}">${escapeHtml(field.name)}...
    # We want to ensure the label *text* is escaped. The `for` attribute value
    # doesn't need HTML escaping (it's a JS reference, not rendered HTML text).
    pattern = r'<label\s+for="f_\$\{field\.name\}">\$\{escapeHtml\(field\.name\)\}'
    assert re.search(pattern, ADMIN_HTML_TEMPLATE), (
        "Boolean field label text should use escapeHtml(field.name)"
    )


def test_field_name_escaped_in_standard_label():
    """The standard (non-boolean) label must use escapeHtml(field.name)."""
    pattern = r'<label\s+class="form-label">\$\{escapeHtml\(field\.name\)\}'
    assert re.search(pattern, ADMIN_HTML_TEMPLATE), (
        "Standard field label text should use escapeHtml(field.name)"
    )


def test_field_name_not_raw_in_labels():
    """No label text should contain raw ${field.name} without escapeHtml wrapping."""
    # Find all label elements that render field.name as text content.
    # We look for patterns like >...${field.name}... inside label tags.
    # Exclude the `for` attribute usage which is fine unescaped.
    raw_in_label = re.findall(
        r'<label[^>]*>\$\{field\.name\}', ADMIN_HTML_TEMPLATE
    )
    assert len(raw_in_label) == 0, (
        f"Found {len(raw_in_label)} label(s) with raw ${{field.name}} "
        f"(should use escapeHtml): {raw_in_label}"
    )


# ---------------------------------------------------------------------------
# 3. field.help_text is escaped in all render paths
# ---------------------------------------------------------------------------


def test_help_text_escaped_in_boolean_path():
    """Boolean field help_text must be wrapped in escapeHtml()."""
    # The boolean path: <div class="form-help">${escapeHtml(field.help_text)}</div>
    pattern = r'form-help">\$\{escapeHtml\(field\.help_text\)\}'
    matches = re.findall(pattern, ADMIN_HTML_TEMPLATE)
    assert len(matches) >= 1, (
        "Boolean field help_text should use escapeHtml(field.help_text)"
    )


def test_help_text_escaped_in_standard_path():
    """Standard field help_text must be wrapped in escapeHtml()."""
    # The standard path: `<div class="form-help">${escapeHtml(field.help_text)}</div>`
    pattern = r'form-help">\$\{escapeHtml\(field\.help_text\)\}'
    matches = re.findall(pattern, ADMIN_HTML_TEMPLATE)
    # There are two render paths (boolean inline and standard variable assignment)
    assert len(matches) >= 2, (
        f"Expected at least 2 escapeHtml(field.help_text) usages (boolean + standard), "
        f"found {len(matches)}"
    )


def test_help_text_not_raw():
    """No help_text should be rendered raw without escapeHtml wrapping."""
    raw_help = re.findall(
        r'form-help">\$\{field\.help_text\}', ADMIN_HTML_TEMPLATE
    )
    assert len(raw_help) == 0, (
        f"Found {len(raw_help)} raw ${{field.help_text}} usage(s) "
        f"(should use escapeHtml): {raw_help}"
    )


# ---------------------------------------------------------------------------
# 4. testConnector checks res.ok
# ---------------------------------------------------------------------------


def test_test_connector_checks_res_ok():
    """testConnector() must check res.ok before inspecting data.success.

    This matches the pattern used by saveConfig(), which correctly throws
    on HTTP error responses before inspecting the JSON body.
    """
    # Find the testConnector function body
    tc_match = re.search(
        r'async function testConnector\b.*?\n    \}',
        ADMIN_HTML_TEMPLATE,
        re.DOTALL,
    )
    assert tc_match, "testConnector function not found in template"

    tc_body = tc_match.group()

    # res.ok check must appear before data.success check
    res_ok_pos = tc_body.find("!res.ok")
    data_success_pos = tc_body.find("data.success")

    assert res_ok_pos != -1, (
        "testConnector should check !res.ok (HTTP status)"
    )
    assert data_success_pos != -1, (
        "testConnector should check data.success"
    )
    assert res_ok_pos < data_success_pos, (
        "testConnector should check !res.ok BEFORE data.success "
        f"(res.ok at {res_ok_pos}, data.success at {data_success_pos})"
    )


def test_save_config_still_checks_res_ok():
    """Regression: saveConfig() must continue to check res.ok."""
    sc_match = re.search(
        r'async function saveConfig\b.*?\n    \}',
        ADMIN_HTML_TEMPLATE,
        re.DOTALL,
    )
    assert sc_match, "saveConfig function not found in template"
    assert "!res.ok" in sc_match.group(), (
        "saveConfig should check !res.ok"
    )
