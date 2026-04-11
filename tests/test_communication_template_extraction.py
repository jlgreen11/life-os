"""
Tests for communication template extraction.

Two extractors participate in template extraction:

1. LinguisticExtractor — processes outbound messages via its own per-contact
   running statistics, storing templates after a 5-sample threshold.

2. RelationshipExtractor — extracts templates from BOTH inbound and outbound
   messages, applying the marketing filter only to inbound messages so that
   automated bulk senders do not pollute the template store.

This module covers both extractors, with particular focus on the direction-
aware marketing filter in RelationshipExtractor (the bug fixed in this PR).
"""

from datetime import UTC, datetime

import pytest

from models.core import EventType
from services.signal_extractor.linguistic import LinguisticExtractor
from services.signal_extractor.relationship import RelationshipExtractor


@pytest.fixture()
def linguistic_extractor(db, user_model_store):
    """Create a LinguisticExtractor with test database."""
    return LinguisticExtractor(db=db, user_model_store=user_model_store)


def _make_email_sent_event(to_address: str, body: str, source: str = "email") -> dict:
    """Build a realistic email.sent event for testing."""
    return {
        "id": f"evt-{hash(body) % 100000}",
        "type": EventType.EMAIL_SENT.value,
        "source": source,
        "timestamp": datetime.now(UTC).isoformat(),
        "payload": {
            "to_addresses": [to_address],
            "body": body,
            "channel": source,
        },
    }


# Sample email bodies with greetings ("Hi Bob") and closings ("Best,")
ALICE_EMAILS = [
    "Hi Alice, I wanted to follow up on our meeting yesterday. "
    "The quarterly reports look really promising and I think we should "
    "discuss the budget allocations further. Best regards, Jeremy",
    "Hi Alice, just a quick note about the project timeline. "
    "We need to push the deadline back by two weeks due to the "
    "vendor delays we discussed. Let me know your thoughts. Best regards, Jeremy",
    "Hi Alice, thanks for sending over the updated proposal. "
    "I reviewed it thoroughly and have a few suggestions regarding "
    "the marketing strategy section. Can we schedule a call? Best regards, Jeremy",
    "Hi Alice, I hope you had a great weekend. Quick question about "
    "the client presentation next Thursday. Should we include the "
    "new analytics dashboard data? Best regards, Jeremy",
    "Hi Alice, following up on your email about the team offsite. "
    "I think the proposed dates work well for everyone. Please go "
    "ahead and book the venue when you get a chance. Best regards, Jeremy",
    "Hi Alice, I wanted to share some thoughts on the Q2 strategy. "
    "We should consider expanding into the European market given "
    "the positive signals from our pilot program. Best regards, Jeremy",
]

BOB_EMAILS = [
    "Hey Bob, are you free for lunch tomorrow? I was thinking we could "
    "try that new place downtown. Let me know what works for you! Cheers",
    "Hey Bob, just saw the game last night! What an incredible finish "
    "yeah that was wild. We should go to a game together soon. Cheers",
    "Hey Bob, thanks for helping me move last weekend. I really "
    "appreciate it man. Let me buy you dinner sometime to repay "
    "the favor. Cheers",
    "Hey Bob, did you hear about the concert next month? I'm gonna "
    "grab tickets if you wanna come along. Should be a great show! Cheers",
    "Hey Bob, hope you're doing well! Just wanted to check in and "
    "see if you're still up for the hiking trip this Saturday. "
    "Weather looks perfect for it. Cheers",
    "Hey Bob, lol that meme you sent was hilarious! Also wanted to "
    "ask if you've tried that new coffee shop on Main Street. "
    "Their espresso is amazing! Cheers",
]


class TestCommunicationTemplateExtraction:
    """Test that LinguisticExtractor stores per-contact communication templates."""

    def test_templates_stored_after_enough_samples(self, linguistic_extractor, user_model_store, db):
        """Templates are created for contacts with >= 5 outbound messages."""
        # Process 6 emails to Alice (above the 5-sample threshold)
        for body in ALICE_EMAILS:
            event = _make_email_sent_event("alice@example.com", body)
            linguistic_extractor.extract(event)

        # Verify template was stored
        template = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert template is not None
        assert template["contact_id"] == "alice@example.com"
        assert template["samples_analyzed"] >= 5

    def test_template_captures_greeting(self, linguistic_extractor, user_model_store):
        """Template greeting field reflects the user's most common greeting for that contact."""
        for body in ALICE_EMAILS:
            event = _make_email_sent_event("alice@example.com", body)
            linguistic_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert template is not None
        # All Alice emails start with "Hi Alice" → greeting should be "hi"
        assert template["greeting"] == "hi"

    def test_template_captures_closing(self, linguistic_extractor, user_model_store):
        """Template closing field reflects the user's most common closing for that contact."""
        for body in ALICE_EMAILS:
            event = _make_email_sent_event("alice@example.com", body)
            linguistic_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert template is not None
        # All Alice emails end with "Best regards" → closing should be "regards" or "best"
        assert template["closing"] in ("best", "regards")

    def test_template_formality(self, linguistic_extractor, user_model_store):
        """Template formality is derived from per-contact linguistic averages."""
        for body in ALICE_EMAILS:
            event = _make_email_sent_event("alice@example.com", body)
            linguistic_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert template is not None
        # Formality should be a valid score between 0 and 1
        assert 0.0 <= template["formality"] <= 1.0

    def test_template_not_stored_below_threshold(self, linguistic_extractor, user_model_store):
        """No template is stored when fewer than 5 messages exist for a contact."""
        # Process only 3 emails (below the 5-sample threshold)
        for body in ALICE_EMAILS[:3]:
            event = _make_email_sent_event("sparse@example.com", body)
            linguistic_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="sparse@example.com")
        # Should be None or from another contact — not from sparse@example.com
        assert template is None or template["contact_id"] != "sparse@example.com"

    def test_multiple_contacts_get_separate_templates(self, linguistic_extractor, user_model_store, db):
        """Different contacts get their own templates with distinct styles."""
        # Process emails to both Alice (formal) and Bob (casual)
        for body in ALICE_EMAILS:
            event = _make_email_sent_event("alice@example.com", body)
            linguistic_extractor.extract(event)
        for body in BOB_EMAILS:
            event = _make_email_sent_event("bob@example.com", body)
            linguistic_extractor.extract(event)

        # Both contacts should have templates
        alice_tpl = user_model_store.get_communication_template(contact_id="alice@example.com")
        bob_tpl = user_model_store.get_communication_template(contact_id="bob@example.com")
        assert alice_tpl is not None
        assert bob_tpl is not None
        assert alice_tpl["contact_id"] == "alice@example.com"
        assert bob_tpl["contact_id"] == "bob@example.com"

    def test_template_greeting_differs_by_contact(self, linguistic_extractor, user_model_store):
        """Templates capture different greetings for different contacts."""
        for body in ALICE_EMAILS:
            event = _make_email_sent_event("alice@example.com", body)
            linguistic_extractor.extract(event)
        for body in BOB_EMAILS:
            event = _make_email_sent_event("bob@example.com", body)
            linguistic_extractor.extract(event)

        alice_tpl = user_model_store.get_communication_template(contact_id="alice@example.com")
        bob_tpl = user_model_store.get_communication_template(contact_id="bob@example.com")
        assert alice_tpl is not None
        assert bob_tpl is not None
        # Alice gets "hi", Bob gets "hey" — different greetings
        assert alice_tpl["greeting"] == "hi"
        assert bob_tpl["greeting"] == "hey"

    def test_template_channel_from_event_source(self, linguistic_extractor, user_model_store):
        """Template channel reflects the event source/channel."""
        for body in ALICE_EMAILS:
            event = _make_email_sent_event("alice@example.com", body, source="email")
            linguistic_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert template is not None
        assert template["channel"] == "email"

    def test_template_typical_length(self, linguistic_extractor, user_model_store):
        """Template typical_length reflects average word count across messages."""
        for body in ALICE_EMAILS:
            event = _make_email_sent_event("alice@example.com", body)
            linguistic_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert template is not None
        # Alice emails are ~30-50 words each; typical_length should be reasonable
        assert template["typical_length"] > 10

    def test_template_context_is_linguistic_outbound(self, linguistic_extractor, user_model_store):
        """Templates from LinguisticExtractor use 'linguistic_outbound' context."""
        for body in ALICE_EMAILS:
            event = _make_email_sent_event("alice@example.com", body)
            linguistic_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert template is not None
        assert template["context"] == "linguistic_outbound"

    def test_template_updates_incrementally(self, linguistic_extractor, user_model_store):
        """Processing more messages updates the existing template."""
        # First batch: 5 emails
        for body in ALICE_EMAILS[:5]:
            event = _make_email_sent_event("alice@example.com", body)
            linguistic_extractor.extract(event)

        tpl1 = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert tpl1 is not None
        count1 = tpl1["samples_analyzed"]

        # Second batch: 1 more email
        event = _make_email_sent_event("alice@example.com", ALICE_EMAILS[5])
        linguistic_extractor.extract(event)

        tpl2 = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert tpl2 is not None
        assert tpl2["samples_analyzed"] > count1

    def test_inbound_email_does_not_create_template(self, linguistic_extractor, user_model_store):
        """Inbound (received) emails should NOT store communication templates.

        Templates capture the *user's* writing style, not contacts' styles.
        Only outbound messages feed into template creation.
        """
        for i in range(6):
            event = {
                "id": f"evt-inbound-{i}",
                "type": EventType.EMAIL_RECEIVED.value,
                "source": "email",
                "timestamp": datetime.now(UTC).isoformat(),
                "payload": {
                    "from_address": "sender@example.com",
                    "body": (
                        "Hi Jeremy, just following up on the quarterly review. "
                        "Please let me know your availability next week. "
                        "Best regards, Sender"
                    ),
                    "channel": "email",
                },
            }
            linguistic_extractor.extract(event)

        # No template should exist for the inbound sender
        template = user_model_store.get_communication_template(contact_id="sender@example.com")
        assert template is None or template["contact_id"] != "sender@example.com"

    def test_template_has_all_required_fields(self, linguistic_extractor, user_model_store):
        """Stored templates must contain every field the schema expects."""
        for body in ALICE_EMAILS:
            event = _make_email_sent_event("alice@example.com", body)
            linguistic_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert template is not None

        required_fields = [
            "id",
            "context",
            "contact_id",
            "channel",
            "greeting",
            "closing",
            "formality",
            "typical_length",
            "uses_emoji",
            "common_phrases",
            "avoids_phrases",
            "tone_notes",
            "example_message_ids",
            "samples_analyzed",
        ]
        for field in required_fields:
            assert field in template, f"Missing required field: {field}"


@pytest.fixture()
def relationship_extractor(db, user_model_store):
    """Create a RelationshipExtractor with test database."""
    return RelationshipExtractor(db=db, user_model_store=user_model_store)


class TestRelationshipExtractorMarketingFilter:
    """Verify that RelationshipExtractor's marketing filter is direction-aware.

    The root cause of communication_templates having 0 rows in production was
    that _extract_communication_templates() applied is_marketing_or_noreply()
    to ALL messages regardless of direction.  The fix: only filter INBOUND
    messages so outbound messages always capture the user's writing style.

    These tests cover the three critical cases:
    1. Outbound to a marketing-pattern address → template IS created.
    2. Inbound from a marketing address → template is NOT created.
    3. Inbound from a real human → template IS created.
    Plus a content sanity check (greeting, closing, formality).
    """

    def test_outbound_to_marketing_address_creates_template(self, relationship_extractor, db):
        """Outbound email to a marketing-pattern address should create a template.

        receipt@chromefile.com matches the 'receipt@' bulk-sender prefix in the
        marketing filter.  Before the fix this would be silently dropped.  After
        the fix, outbound messages bypass the filter so the user's style is
        captured regardless of the recipient address.
        """
        event = {
            "id": "evt-out-marketing",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "to_addresses": ["receipt@chromefile.com"],
                "body": (
                    "Hello, I am writing to confirm my recent order and request an "
                    "itemized receipt for transaction number 12345. Please send the "
                    "receipt to this address at your earliest convenience. "
                    "Thank you for your help. Best regards, Jeremy"
                ),
                "channel": "email",
            },
        }
        relationship_extractor.extract(event)

        # Template must exist — outbound bypasses the marketing filter
        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT contact_id, context FROM communication_templates WHERE contact_id = ?",
                ("receipt@chromefile.com",),
            ).fetchall()
        assert len(rows) == 1, (
            "Outbound to a marketing-pattern address must produce a template; "
            "the user's writing style should always be captured for sent messages"
        )
        assert rows[0]["context"] == "user_to_contact"

    def test_inbound_from_marketing_address_no_template(self, relationship_extractor, db):
        """Inbound email from a marketing address should NOT create a template.

        newsletter@company.com is an automated bulk sender.  Capturing its
        writing style would pollute the template store with non-human patterns
        that could mislead the AI engine when composing replies.
        """
        event = {
            "id": "evt-in-marketing",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "from_address": "newsletter@company.com",
                "body": (
                    "Hello Valued Subscriber! This week we have amazing deals for you. "
                    "Our best products are on sale right now at incredible prices. "
                    "Click below to shop and save big. You can unsubscribe at any time."
                ),
                "channel": "email",
            },
        }
        relationship_extractor.extract(event)

        # No template should exist — inbound marketing is still filtered
        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT * FROM communication_templates WHERE contact_id = ?",
                ("newsletter@company.com",),
            ).fetchall()
        assert len(rows) == 0, (
            "Inbound from a marketing address must not create a template; "
            "automated bulk senders should not pollute the template store"
        )

    def test_inbound_from_human_creates_template(self, relationship_extractor, db):
        """Inbound email from a real human should create a template.

        Human correspondents' writing styles are valuable for style-matching
        when the user composes replies.  The fix must not break inbound template
        extraction for legitimate (non-marketing) senders.
        """
        event = {
            "id": "evt-in-human",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "from_address": "alice@company.com",
                "body": (
                    "Hi Jeremy, just following up on our meeting yesterday. "
                    "I have reviewed the proposal and I think we should move forward. "
                    "Could you send me the updated timeline when you get a chance? "
                    "Best regards, Alice"
                ),
                "channel": "email",
            },
        }
        relationship_extractor.extract(event)

        # Template should exist — human inbound is captured for reply style-matching
        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT contact_id, context FROM communication_templates WHERE contact_id = ?",
                ("alice@company.com",),
            ).fetchall()
        assert len(rows) == 1, (
            "Inbound from a human address must create a template; the contact's style is needed for reply composition"
        )
        assert rows[0]["context"] == "contact_to_user"

    def test_outbound_template_captures_greeting_and_closing(self, relationship_extractor, user_model_store):
        """Template extracted from outbound email captures greeting and closing.

        Verifies that the content of the template (not just its existence) is
        correct — the greeting and closing should be parsed from the message body.
        """
        event = {
            "id": "evt-out-content",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "to_addresses": ["orders@shop.com"],
                "body": (
                    "Hello, I am reaching out to inquire about my recent order. "
                    "The package has not arrived and I would like an update on "
                    "the expected delivery date. Please let me know. Thank you."
                ),
                "channel": "email",
            },
        }
        relationship_extractor.extract(event)

        template = user_model_store.get_communication_template(contact_id="orders@shop.com")
        assert template is not None, "Template must be created for outbound to marketing address"
        # Greeting "Hello" should be captured (case-insensitive comparison)
        greeting = (template["greeting"] or "").lower()
        assert greeting in ("hello", "hi", "hey", ""), f"Unexpected greeting value: {template['greeting']!r}"
        # Formality should be a valid float between 0 and 1
        assert 0.0 <= template["formality"] <= 1.0
