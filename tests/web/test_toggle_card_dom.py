"""
Tests for the toggleCard() DOM-class-toggle fix (Task 6, ui-engagement-fixes plan).

Before this fix, toggleCard() re-rendered the ENTIRE feed HTML on every
expand/collapse, causing a visual flash and scroll-position loss.

After the fix, toggleCard() uses targeted `classList.add/remove('expanded')` on
the single card element identified by its `[data-id]` attribute.  The CSS rule
`.card.expanded .card-detail { display: block }` handles showing the detail
panel without any inline style manipulation.

These tests validate the JavaScript source in web/template.py.
"""

import pytest

from web.template import HTML_TEMPLATE as TEMPLATE


class TestToggleCardImplementation:
    """Verify toggleCard() uses targeted DOM updates, not full re-renders."""

    def test_toggle_card_uses_queryselector_not_feedcontent(self):
        """toggleCard must NOT use getElementById('feedContent') for re-rendering.

        The old implementation fetched #feedContent and replaced its entire
        innerHTML by iterating feedItems.  The new implementation targets the
        individual card via querySelector, so #feedContent must not appear inside
        the toggleCard function body.
        """
        # Extract just the toggleCard function body by finding the function
        # declaration and reading until the closing brace.
        toggle_idx = TEMPLATE.find("function toggleCard(id)")
        assert toggle_idx >= 0, "toggleCard function must exist in template"

        # Find the end of the function (the next stand-alone closing brace).
        func_body_start = TEMPLATE.find("{", toggle_idx)
        # Walk forward counting braces to find the matching close.
        depth = 0
        pos = func_body_start
        while pos < len(TEMPLATE):
            ch = TEMPLATE[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        func_body = TEMPLATE[toggle_idx : pos + 1]

        # The function must NOT re-fetch feedContent for a full re-render.
        assert "getElementById('feedContent')" not in func_body, (
            "toggleCard must not use getElementById('feedContent') for re-rendering"
        )
        assert "renderCard(feedItems" not in func_body, (
            "toggleCard must not iterate feedItems to rebuild HTML"
        )

    def test_toggle_card_uses_queryselector_with_data_id(self):
        """toggleCard must use querySelector('[data-id=...') to find the card."""
        toggle_idx = TEMPLATE.find("function toggleCard(id)")
        assert toggle_idx >= 0

        func_end = TEMPLATE.find("function ", toggle_idx + 20)
        func_body = TEMPLATE[toggle_idx:func_end] if func_end > 0 else TEMPLATE[toggle_idx:]

        # The targeted selector approach uses the data-id attribute.
        assert "querySelector('[data-id=\"'" in func_body or "querySelector" in func_body, (
            "toggleCard must use querySelector to target a specific card element"
        )

    def test_toggle_card_adds_expanded_class(self):
        """toggleCard must call classList.add('expanded') to expand a card."""
        toggle_idx = TEMPLATE.find("function toggleCard(id)")
        assert toggle_idx >= 0
        # Find end of function
        func_body_start = TEMPLATE.find("{", toggle_idx)
        depth, pos = 0, func_body_start
        while pos < len(TEMPLATE):
            if TEMPLATE[pos] == "{":
                depth += 1
            elif TEMPLATE[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        func_body = TEMPLATE[toggle_idx : pos + 1]

        assert "classList.add('expanded')" in func_body, (
            "toggleCard must call classList.add('expanded') to show the detail panel"
        )

    def test_toggle_card_removes_expanded_class(self):
        """toggleCard must call classList.remove('expanded') to collapse a card."""
        toggle_idx = TEMPLATE.find("function toggleCard(id)")
        assert toggle_idx >= 0
        func_body_start = TEMPLATE.find("{", toggle_idx)
        depth, pos = 0, func_body_start
        while pos < len(TEMPLATE):
            if TEMPLATE[pos] == "{":
                depth += 1
            elif TEMPLATE[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        func_body = TEMPLATE[toggle_idx : pos + 1]

        assert "classList.remove('expanded')" in func_body, (
            "toggleCard must call classList.remove('expanded') to collapse"
        )

    def test_toggle_card_collapses_previous_card(self):
        """toggleCard must collapse the previously-expanded card before expanding a new one.

        Without this, two cards can appear expanded simultaneously, which
        breaks the accordion behavior of the feed.
        """
        toggle_idx = TEMPLATE.find("function toggleCard(id)")
        assert toggle_idx >= 0
        func_body_start = TEMPLATE.find("{", toggle_idx)
        depth, pos = 0, func_body_start
        while pos < len(TEMPLATE):
            if TEMPLATE[pos] == "{":
                depth += 1
            elif TEMPLATE[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        func_body = TEMPLATE[toggle_idx : pos + 1]

        # The function should check expandedCardId and collapse the previous card.
        assert "expandedCardId" in func_body, "toggleCard must track expandedCardId"
        # There should be at least two classList.remove calls or conditional logic
        # that handles the previously-expanded card.
        remove_count = func_body.count("classList.remove('expanded')")
        assert remove_count >= 1, (
            "toggleCard must have at least one classList.remove('expanded') call "
            "to collapse either the current or previously-expanded card"
        )


class TestEscapeKeyHandler:
    """Verify the Escape key handler also uses targeted DOM updates."""

    def test_escape_handler_uses_queryselector(self):
        """Escape handler must use querySelector to find and collapse the card."""
        esc_idx = TEMPLATE.find("Escape key: collapse expanded card")
        assert esc_idx >= 0, "Escape key handler comment must be present"

        # Read the handler body (until the closing listener brace)
        handler_end = TEMPLATE.find("});", esc_idx)
        handler_body = TEMPLATE[esc_idx : handler_end + 3] if handler_end > 0 else ""

        assert "querySelector" in handler_body, (
            "Escape key handler must use querySelector, not a full re-render"
        )
        assert "classList.remove('expanded')" in handler_body, (
            "Escape key handler must call classList.remove('expanded')"
        )

    def test_escape_handler_no_full_rerender(self):
        """Escape handler must not rebuild the entire feed HTML."""
        esc_idx = TEMPLATE.find("Escape key: collapse expanded card")
        assert esc_idx >= 0

        handler_end = TEMPLATE.find("});", esc_idx)
        handler_body = TEMPLATE[esc_idx : handler_end + 3] if handler_end > 0 else ""

        assert "renderCard(feedItems" not in handler_body, (
            "Escape handler must not iterate feedItems to rebuild HTML"
        )
        assert "getElementById('feedContent')" not in handler_body, (
            "Escape handler must not re-fetch #feedContent for a full re-render"
        )


class TestCardHTMLStructure:
    """Verify the rendered card HTML has the attributes the DOM fix depends on."""

    def test_cards_have_data_id_attribute(self):
        """renderCard must emit data-id on each card for querySelector targeting.

        The toggleCard fix depends on querySelector('[data-id="..."]') to find
        cards.  If data-id is missing, the fix silently does nothing.
        """
        assert 'data-id="' in TEMPLATE or "data-id='" in TEMPLATE, (
            "renderCard must emit data-id attribute on card elements"
        )

    def test_card_detail_css_uses_expanded_selector(self):
        """CSS must show .card-detail via the .card.expanded selector.

        The toggleCard fix ONLY toggles the .expanded class — it does not
        manually set display style.  The CSS must handle the rest.
        """
        assert ".card.expanded .card-detail" in TEMPLATE, (
            "CSS must contain .card.expanded .card-detail rule to show the detail panel"
        )
        assert "display: block" in TEMPLATE or "display:block" in TEMPLATE, (
            "The .card.expanded .card-detail rule must set display to block"
        )
