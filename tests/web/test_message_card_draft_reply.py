"""
Tests for the Draft Reply button on message cards (Signal / iMessage / WhatsApp).

Before this fix, only email cards had a "Draft Reply" button.  Message cards
exposed "Create Task" but gave the user no way to generate an AI-drafted reply,
creating inconsistency and leaving a core design-doc feature unimplemented.

After the fix:
- Message cards in the drill-down detail section include a "Draft Reply" button
  that calls draftReply(), matching the email card UX exactly.
- A <div id="draft-{id}"> placeholder is rendered so draftReply() has a DOM
  target to write the generated draft into.
- The draftReply() function now also passes contact_id (sender address) to
  /api/draft so the AI engine can apply per-contact communication templates.

These tests inspect the HTML/JS source in web/template.py directly.
"""

import re

import pytest

from web.template import HTML_TEMPLATE as TEMPLATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_message_card_block() -> str:
    """Extract the JavaScript block that renders a 'message' channel card.

    The renderCard() function contains a chain of if/else if branches, one per
    channel type.  This helper locates the ``} else if (item.channel === 'message') {``
    branch and returns the text up to (but not including) the next branch.
    """
    start_marker = "} else if (item.channel === 'message') {"
    end_marker = "} else if (item.channel === 'calendar') {"

    start_idx = TEMPLATE.find(start_marker)
    assert start_idx >= 0, f"Could not find message-channel branch: '{start_marker}'"

    end_idx = TEMPLATE.find(end_marker, start_idx)
    assert end_idx > start_idx, f"Could not find end of message-channel branch: '{end_marker}'"

    return TEMPLATE[start_idx:end_idx]


def _extract_draft_reply_function() -> str:
    """Extract the full draftReply() function body from the template."""
    func_marker = "function draftReply(id, context)"
    start_idx = TEMPLATE.find(func_marker)
    assert start_idx >= 0, "draftReply function must exist in template"

    # Walk braces to find the matching closing brace for the function body.
    brace_start = TEMPLATE.find("{", start_idx)
    depth = 0
    pos = brace_start
    while pos < len(TEMPLATE):
        if TEMPLATE[pos] == "{":
            depth += 1
        elif TEMPLATE[pos] == "}":
            depth -= 1
            if depth == 0:
                break
        pos += 1

    return TEMPLATE[start_idx : pos + 1]


# ---------------------------------------------------------------------------
# Message card structure tests
# ---------------------------------------------------------------------------

class TestMessageCardDraftReplyButton:
    """Verify that message cards include a Draft Reply button in their detail view."""

    def test_message_card_has_draft_reply_button(self):
        """Message card detail section must include a Draft Reply button.

        The button must call draftReply() passing the card id and title,
        matching the pattern used by email cards.
        """
        block = _extract_message_card_block()
        assert "draftReply(" in block, (
            "Message card must include a call to draftReply() in its action buttons"
        )

    def test_message_card_draft_button_is_primary(self):
        """Draft Reply button on message cards must use btn-primary class.

        Consistency with email cards: the Draft Reply button is the primary
        action, styled with btn-primary so it's visually prominent.
        """
        block = _extract_message_card_block()
        # The Draft Reply button must have the btn-primary class
        assert "btn-primary" in block, (
            "Message card Draft Reply button must use btn-primary class"
        )
        # And that btn-primary button must call draftReply
        btn_primary_idx = block.find("btn-primary")
        draft_reply_near = "draftReply(" in block[btn_primary_idx:btn_primary_idx + 200]
        assert draft_reply_near, (
            "btn-primary button must call draftReply() — it should be the Draft Reply button"
        )

    def test_message_card_has_draft_placeholder_div(self):
        """Message card must render a <div id='draft-{id}'> placeholder.

        draftReply() uses getElementById('draft-' + id) to find the DOM node
        to write the generated draft into.  Without this placeholder, draftReply
        exits early and nothing is displayed.
        """
        block = _extract_message_card_block()
        assert "draft-' +" in block or "'draft-' + escAttr(id)" in block, (
            "Message card must render a <div id='draft-{id}'> placeholder "
            "for draftReply() to target"
        )

    def test_message_card_draft_placeholder_before_actions(self):
        """Draft placeholder must appear before the card-actions div.

        The placeholder must be above the buttons so the generated draft text
        appears between the message body and the action buttons — consistent
        with the email card layout.
        """
        block = _extract_message_card_block()
        placeholder_idx = block.find("draft-' +")
        actions_idx = block.find("card-actions")
        assert placeholder_idx >= 0, "Draft placeholder must exist in message card block"
        assert actions_idx >= 0, "card-actions div must exist in message card block"
        assert placeholder_idx < actions_idx, (
            "Draft placeholder (#draft-{id}) must appear before card-actions in the HTML"
        )

    def test_message_card_still_has_create_task_button(self):
        """Message card must still have the Create Task button after this change.

        Adding Draft Reply must not remove the pre-existing Create Task button.
        """
        block = _extract_message_card_block()
        assert "createTaskFrom(" in block, (
            "Message card must still include the Create Task button"
        )


# ---------------------------------------------------------------------------
# draftReply() function tests
# ---------------------------------------------------------------------------

class TestDraftReplyContactId:
    """Verify draftReply() passes contact_id to /api/draft."""

    def test_draft_reply_sends_contact_id(self):
        """draftReply() must include contact_id in the POST body to /api/draft.

        Passing the sender address as contact_id allows the AI engine to look up
        per-contact communication templates (formality, greeting style, etc.)
        so the generated draft mirrors how the user typically writes to that contact.
        """
        func_body = _extract_draft_reply_function()
        assert "contact_id" in func_body, (
            "draftReply() must pass contact_id in the POST body to /api/draft"
        )

    def test_draft_reply_contact_id_from_metadata_sender(self):
        """draftReply() must derive contact_id from item.metadata.sender.

        For message cards the sender address lives in item.metadata.sender.
        Email cards populate the same field.  Using this as contact_id ensures
        the AI engine can apply per-contact templates for both channel types.
        """
        func_body = _extract_draft_reply_function()
        assert "metadata" in func_body and "sender" in func_body, (
            "draftReply() must extract sender from item.metadata.sender for contact_id"
        )

    def test_draft_reply_contact_id_nullable(self):
        """draftReply() must handle the case where contact_id is null/missing.

        Not all items have a sender address (e.g., system notifications).  The
        function must fall back to null so the backend still works without it.
        """
        func_body = _extract_draft_reply_function()
        # Should have null or || null pattern to handle missing sender
        assert "null" in func_body, (
            "draftReply() must fall back to null when contact_id is not available"
        )


# ---------------------------------------------------------------------------
# Email card parity tests (regression: email cards must still work)
# ---------------------------------------------------------------------------

class TestEmailCardDraftReplyUnchanged:
    """Verify the email card draft reply behaviour is unaffected by the message card change."""

    def test_email_card_still_has_draft_reply_button(self):
        """Email card must still include its Draft Reply button after the change."""
        email_start = TEMPLATE.find("if (item.channel === 'email') {")
        assert email_start >= 0

        # Find the end of the email branch (next else if)
        email_end = TEMPLATE.find("} else if (item.channel === 'message') {", email_start)
        email_block = TEMPLATE[email_start:email_end] if email_end > email_start else ""

        assert "draftReply(" in email_block, (
            "Email card must still include its Draft Reply button"
        )

    def test_email_card_still_has_draft_placeholder(self):
        """Email card must still render its #draft-{id} placeholder."""
        email_start = TEMPLATE.find("if (item.channel === 'email') {")
        email_end = TEMPLATE.find("} else if (item.channel === 'message') {", email_start)
        email_block = TEMPLATE[email_start:email_end] if email_end > email_start else ""

        assert "draft-' +" in email_block or "draft-' + escAttr(id)" in email_block, (
            "Email card must still render its #draft-{id} placeholder"
        )
