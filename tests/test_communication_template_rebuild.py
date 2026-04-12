"""Tests for communication template extraction reliability during profile rebuild.

Regression test suite for the "0 communication templates despite 13K+ emails"
data quality issue.

Root causes fixed:
  1. store_communication_template() inside _extract_communication_templates() lacked
     try/except — any DB error propagated through extract() to the pipeline's
     fail-open handler, silently losing the template.
  2. extract() had no isolation around _extract_communication_templates(), so a
     single template write failure would abort signal extraction for the whole event.
  3. HTML email bodies were not stripped before style analysis, poisoning
     common_phrases with HTML tag names ("div", "span", "href").

These tests verify:
  1. Templates are stored from email.received events (inbound, human sender)
  2. Templates accumulate correctly across multiple events (rebuild scenario)
  3. HTML bodies are stripped before analysis; common_phrases have no HTML tags
  4. Marketing/noreply senders are still filtered for inbound events
  5. A DB write failure (simulated) does NOT abort relationship signal extraction
  6. A DB write failure (simulated) is logged but does NOT propagate to callers
  7. Empty and very-short bodies are skipped without error
"""

from __future__ import annotations

import json
import unittest.mock
from datetime import UTC, datetime

import pytest

from models.core import EventType
from services.signal_extractor.relationship import RelationshipExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rel_extractor(db, user_model_store):
    """RelationshipExtractor wired to real (temporary) test databases."""
    return RelationshipExtractor(db=db, user_model_store=user_model_store)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_received(from_addr: str, body: str, *, source: str = "email") -> dict:
    """Build a minimal email.received event for testing."""
    return {
        "id": f"evt-recv-{hash((from_addr, body)) % 10_000_000:07d}",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": source,
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {
            "from_address": from_addr,
            "from_name": from_addr.split("@")[0].capitalize(),
            "body_plain": body,
            "body": body,
            "channel": source,
        },
    }


def _make_sent(to_addr: str, body: str, *, source: str = "email") -> dict:
    """Build a minimal email.sent event for testing."""
    return {
        "id": f"evt-sent-{hash((to_addr, body)) % 10_000_000:07d}",
        "type": EventType.EMAIL_SENT.value,
        "source": source,
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {
            "to_addresses": [to_addr],
            "body_plain": body,
            "body": body,
            "channel": source,
        },
    }


def _make_html_received(from_addr: str, html_body: str, *, source: str = "email") -> dict:
    """Build an email.received event with an HTML-only body (no body_plain).

    This simulates the common case where an email connector stores HTML in
    ``body`` but has no ``body_plain`` alternative — e.g. a newsletter or
    richly formatted email from a contact.
    """
    return {
        "id": f"evt-html-{hash((from_addr, html_body)) % 10_000_000:07d}",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": source,
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {
            "from_address": from_addr,
            "body_plain": "",          # Empty — connector found no plain-text part
            "body": html_body,         # Full HTML body
            "channel": source,
        },
    }


HUMAN_BODY = (
    "Hi Jeremy, just following up on our meeting yesterday. "
    "I reviewed the proposal and I think we should move forward with option B. "
    "Could you send me the updated timeline when you get a chance? "
    "Best regards, Alice"
)

HUMAN_BODY_2 = (
    "Hey there, just a quick note — I will be out of office next week. "
    "Please reach out to my colleague if anything urgent comes up. "
    "Thanks for understanding. Cheers, Alice"
)

HUMAN_BODY_3 = (
    "Good morning, I wanted to check in on the status of the project. "
    "We are approaching the deadline and I want to make sure everything is on track. "
    "Let me know if you need any support from my side. Kind regards, Alice"
)

HTML_BODY = (
    "<html><head><style>body { font-family: Arial; }</style></head>"
    "<body><p>Hi Jeremy,</p>"
    "<p>I wanted to reach out about the upcoming meeting. "
    "Could we reschedule to Thursday afternoon? "
    "Please let me know if that works for you.</p>"
    "<p>Best regards,<br/>Alice</p></body></html>"
)


# ---------------------------------------------------------------------------
# 1. Basic inbound template creation
# ---------------------------------------------------------------------------


class TestBasicInboundTemplateCreation:
    """A single email.received from a human creates a template."""

    def test_inbound_human_creates_template(self, rel_extractor, db):
        """One email.received from a human sender stores a contact_to_user template."""
        event = _make_received("alice@corp.com", HUMAN_BODY)
        rel_extractor.extract(event)

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT contact_id, context, samples_analyzed "
                "FROM communication_templates WHERE contact_id = ?",
                ("alice@corp.com",),
            ).fetchall()

        assert len(rows) == 1, (
            "Expected one template for alice@corp.com after processing one inbound email; "
            f"got {len(rows)}"
        )
        assert rows[0]["context"] == "contact_to_user"
        assert rows[0]["samples_analyzed"] == 1

    def test_inbound_human_template_fields(self, rel_extractor, user_model_store):
        """Template created from inbound email has all expected fields populated."""
        event = _make_received("bob@example.com", HUMAN_BODY)
        rel_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="bob@example.com")
        assert template is not None

        required = [
            "id", "context", "contact_id", "channel",
            "greeting", "closing", "formality", "typical_length",
            "uses_emoji", "common_phrases", "avoids_phrases",
            "tone_notes", "example_message_ids", "samples_analyzed",
        ]
        for field in required:
            assert field in template, f"Template missing required field: {field!r}"

    def test_outbound_creates_user_to_contact_template(self, rel_extractor, db):
        """email.sent stores a user_to_contact template."""
        event = _make_sent("carol@agency.com", HUMAN_BODY)
        rel_extractor.extract(event)

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT context FROM communication_templates WHERE contact_id = ?",
                ("carol@agency.com",),
            ).fetchall()

        assert len(rows) == 1
        assert rows[0]["context"] == "user_to_contact"


# ---------------------------------------------------------------------------
# 2. Rebuild scenario: templates accumulate across repeated events
# ---------------------------------------------------------------------------


class TestRebuildScenario:
    """Multiple events from the same contact accumulate into one template.

    This is the core rebuild test: when rebuild_profiles_from_events() replays
    historical email events, each call to extract() must update the same template
    (keyed on contact + channel + direction) rather than overwriting it with a
    single-sample snapshot.
    """

    def test_samples_accumulate_across_events(self, rel_extractor, user_model_store):
        """Three emails from the same contact → samples_analyzed == 3."""
        bodies = [HUMAN_BODY, HUMAN_BODY_2, HUMAN_BODY_3]
        for body in bodies:
            rel_extractor.extract(_make_received("dave@startup.io", body))

        template = user_model_store.get_communication_template(contact_id="dave@startup.io")
        assert template is not None
        assert template["samples_analyzed"] == 3

    def test_ten_events_rebuild_produces_single_template(self, rel_extractor, db):
        """Replaying 10 historical events (rebuild) creates exactly one template per contact."""
        for i in range(10):
            body = (
                f"Hi Jeremy, this is message {i + 1}. "
                "Just checking in on the project status and wanted to share some updates. "
                "Please let me know your thoughts when you get a moment. Best regards, Eve"
            )
            rel_extractor.extract(_make_received("eve@enterprise.com", body))

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT samples_analyzed FROM communication_templates WHERE contact_id = ?",
                ("eve@enterprise.com",),
            ).fetchall()

        assert len(rows) == 1, "Rebuild must produce exactly ONE template per contact-channel-direction"
        assert rows[0]["samples_analyzed"] == 10

    def test_separate_templates_per_contact(self, rel_extractor, user_model_store):
        """Different contacts each get their own template."""
        contacts = ["alpha@a.com", "beta@b.com", "gamma@c.com"]
        for addr in contacts:
            for body in [HUMAN_BODY, HUMAN_BODY_2]:
                rel_extractor.extract(_make_received(addr, body))

        for addr in contacts:
            tmpl = user_model_store.get_communication_template(contact_id=addr)
            assert tmpl is not None, f"Expected template for {addr}"
            assert tmpl["contact_id"] == addr
            assert tmpl["samples_analyzed"] == 2

    def test_inbound_and_outbound_get_separate_templates(self, rel_extractor, db):
        """Same contact generates two templates: one inbound, one outbound."""
        rel_extractor.extract(_make_received("frank@co.com", HUMAN_BODY))
        rel_extractor.extract(_make_sent("frank@co.com", HUMAN_BODY_2))

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT context FROM communication_templates WHERE contact_id = ?",
                ("frank@co.com",),
            ).fetchall()

        contexts = {r["context"] for r in rows}
        assert "contact_to_user" in contexts, "Missing inbound template"
        assert "user_to_contact" in contexts, "Missing outbound template"


# ---------------------------------------------------------------------------
# 3. HTML body stripping
# ---------------------------------------------------------------------------


class TestHtmlBodyStripping:
    """HTML bodies are stripped before analysis to prevent tag pollution."""

    def test_html_body_creates_template(self, rel_extractor, db):
        """An email with an HTML-only body (empty body_plain) still creates a template."""
        event = _make_html_received("grace@design.com", HTML_BODY)
        rel_extractor.extract(event)

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT contact_id FROM communication_templates WHERE contact_id = ?",
                ("grace@design.com",),
            ).fetchall()

        assert len(rows) == 1, (
            "HTML-only email body must create a template after HTML stripping; "
            "if body_plain is empty and stripping is absent, the body check sees "
            "short/empty content and silently skips template creation"
        )

    def test_html_common_phrases_no_html_tags(self, rel_extractor, user_model_store):
        """common_phrases extracted from an HTML body must not include HTML tag words."""
        event = _make_html_received("henry@media.net", HTML_BODY)
        rel_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="henry@media.net")
        assert template is not None

        # "body" is a valid English word so we don't test for it.
        # The key words to exclude are CSS/JS tokens from the <style> block.
        html_words = {"html", "head", "style", "font", "family", "arial", "href", "div", "span", "class"}
        phrases_set = set(template.get("common_phrases", []))
        polluted = html_words & phrases_set
        assert not polluted, (
            f"common_phrases contains HTML/CSS keywords after stripping: {polluted!r}. "
            "CSS block stripping must remove <style> content before word extraction."
        )

    def test_html_strip_helper(self, rel_extractor):
        """_strip_html removes tags, decodes safe entities, and collapses whitespace.

        Note: &lt; and &gt; are intentionally NOT decoded to avoid re-introducing
        angle brackets that could look like new HTML tags to downstream code.
        """
        raw = "<p>Hello  &amp;  <b>world</b>!</p>\n<p>See &lt;you&gt; soon.</p>"
        result = rel_extractor._strip_html(raw)
        # All HTML tags must be removed
        assert "<p>" not in result
        assert "<b>" not in result
        assert "</b>" not in result
        assert "</p>" not in result
        # &amp; is a safe entity and must be decoded to &
        assert "&amp;" not in result
        assert "&" in result
        # Content words must be preserved
        assert "Hello" in result
        assert "world" in result
        # Consecutive whitespace collapsed to a single space
        assert "  " not in result

    def test_strip_html_passthrough_for_plain_text(self, rel_extractor):
        """_strip_html returns plain text unchanged (no angle brackets present)."""
        plain = "Hi there, how are you doing today?"
        assert rel_extractor._strip_html(plain) == plain

    def test_strip_html_passthrough_for_empty(self, rel_extractor):
        """_strip_html handles empty string and None gracefully."""
        assert rel_extractor._strip_html("") == ""
        assert rel_extractor._strip_html(None) is None


# ---------------------------------------------------------------------------
# 4. Marketing filter still applied for inbound
# ---------------------------------------------------------------------------


class TestMarketingFilterInbound:
    """Inbound marketing emails must NOT create templates."""

    @pytest.mark.parametrize("addr", [
        "newsletter@bigco.com",
        "noreply@service.com",
        "no-reply@platform.io",
        "donotreply@corp.org",
        "notifications@app.com",
    ])
    def test_inbound_marketing_no_template(self, rel_extractor, db, addr):
        """Automated/marketing sender produces no template."""
        event = _make_received(addr, HUMAN_BODY)
        rel_extractor.extract(event)

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT * FROM communication_templates WHERE contact_id = ?",
                (addr,),
            ).fetchall()

        assert len(rows) == 0, f"Marketing/noreply sender {addr!r} must not create a template"

    def test_inbound_human_sender_passes_filter(self, rel_extractor, db):
        """A legitimate human sender is NOT filtered and creates a template."""
        event = _make_received("colleague@company.com", HUMAN_BODY)
        rel_extractor.extract(event)

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT contact_id FROM communication_templates WHERE contact_id = ?",
                ("colleague@company.com",),
            ).fetchall()

        assert len(rows) == 1


# ---------------------------------------------------------------------------
# 5. Error resilience: DB write failure does not abort signal extraction
# ---------------------------------------------------------------------------


class TestErrorResilience:
    """A template write failure must not abort relationship signal extraction.

    Prior to the fix, store_communication_template() lacked try/except.  Any
    DB error inside _extract_communication_templates() propagated through
    extract(), aborting signal extraction and causing the pipeline's fail-open
    handler to drop both the template AND the relationship signal update.
    """

    def test_db_write_failure_does_not_raise(self, db, user_model_store):
        """extract() completes normally even when store_communication_template raises.

        Simulates a DB failure by patching store_communication_template to raise,
        then verifies extract() returns signals without propagating the exception.
        """
        rel_extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

        with unittest.mock.patch.object(
            user_model_store,
            "store_communication_template",
            side_effect=Exception("Simulated DB write error"),
        ):
            # Must not raise — the try/except in extract() must catch this
            signals = rel_extractor.extract(_make_received("zara@test.com", HUMAN_BODY))

        # extract() should still return the relationship signals
        assert isinstance(signals, list), "extract() must return a list even when template store fails"

    def test_db_write_failure_logs_warning(self, db, user_model_store, caplog):
        """A DB write failure during template store is logged at WARNING level."""
        import logging

        rel_extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

        with unittest.mock.patch.object(
            user_model_store,
            "store_communication_template",
            side_effect=Exception("WAL corruption"),
        ):
            with caplog.at_level(logging.WARNING, logger="services.signal_extractor.relationship"):
                rel_extractor.extract(_make_received("yuki@test.com", HUMAN_BODY))

        # A WARNING must be emitted so operators can diagnose the failure
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("WAL corruption" in msg or "template" in msg.lower() for msg in warning_messages), (
            f"Expected a warning about template store failure, got: {warning_messages!r}"
        )

    def test_db_write_failure_relationship_signals_still_saved(self, db, user_model_store):
        """Relationship signal profile is persisted even when template store fails.

        _update_contact_profiles() commits before _extract_communication_templates()
        is called.  A failure in the latter must not roll back the former.
        """
        rel_extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

        with unittest.mock.patch.object(
            user_model_store,
            "store_communication_template",
            side_effect=Exception("Simulated DB write error"),
        ):
            rel_extractor.extract(_make_received("xander@test.com", HUMAN_BODY))

        # The relationships signal profile must still have the contact recorded
        profile = user_model_store.get_signal_profile("relationships")
        assert profile is not None, "Relationship signal profile must be written"
        contacts = (profile.get("data") or {}).get("contacts", {})
        assert "xander@test.com" in contacts, (
            "Contact must appear in relationship profile even when template store fails"
        )


# ---------------------------------------------------------------------------
# 6. Edge cases in email payloads
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Handle missing or unusual payload fields without raising."""

    def test_empty_body_plain_and_body_skipped_gracefully(self, rel_extractor, db):
        """Event with empty body is skipped without error or template."""
        event = {
            "id": "evt-empty-body",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "from_address": "sender@example.com",
                "body_plain": "",
                "body": "",
                "channel": "email",
            },
        }
        # Must not raise
        rel_extractor.extract(event)

        with db.get_connection("user_model") as conn:
            rows = conn.execute("SELECT * FROM communication_templates").fetchall()
        assert len(rows) == 0

    def test_very_short_body_skipped(self, rel_extractor, db):
        """Body shorter than 10 chars is skipped without error or template."""
        event = _make_received("brief@example.com", "Hi.")
        rel_extractor.extract(event)

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT * FROM communication_templates WHERE contact_id = ?",
                ("brief@example.com",),
            ).fetchall()
        assert len(rows) == 0

    def test_missing_from_address_no_template(self, rel_extractor, db):
        """Event without from_address produces no template (nothing to key on)."""
        event = {
            "id": "evt-no-from",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "body": HUMAN_BODY,
                "channel": "email",
                # No from_address key
            },
        }
        rel_extractor.extract(event)

        with db.get_connection("user_model") as conn:
            rows = conn.execute("SELECT * FROM communication_templates").fetchall()
        assert len(rows) == 0

    def test_missing_body_keys_no_crash(self, rel_extractor, db):
        """Event with no body keys at all is silently skipped."""
        event = {
            "id": "evt-no-body",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "from_address": "partial@example.com",
                "channel": "email",
                # Neither body nor body_plain
            },
        }
        # Must not raise
        rel_extractor.extract(event)

    def test_html_only_body_with_html_encoded_text(self, rel_extractor, db):
        """HTML body containing &amp; and &lt; entities is decoded before analysis."""
        html = (
            "<p>Hi Jeremy &amp; Team,</p>"
            "<p>Please review the document at &lt;docs.example.com&gt; "
            "and let me know your thoughts on the proposal. "
            "I appreciate your help with this important matter. Best regards.</p>"
        )
        event = _make_html_received("ian@client.com", html)
        rel_extractor.extract(event)

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT common_phrases FROM communication_templates WHERE contact_id = ?",
                ("ian@client.com",),
            ).fetchall()

        assert len(rows) == 1
        phrases = json.loads(rows[0]["common_phrases"])
        # "&amp;" entity should not appear as a word; decoded form "amp" might appear
        # but raw entity strings must not be in common_phrases
        for phrase in phrases:
            assert "&amp;" not in phrase
            assert "&lt;" not in phrase
            assert "&gt;" not in phrase
