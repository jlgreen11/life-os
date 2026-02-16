"""
Tests for marketing email filtering in RelationshipExtractor.

Verifies that marketing emails, no-reply addresses, and automated senders are
filtered out before being tracked in the relationship graph.
"""

import pytest
from datetime import datetime, timezone

from services.signal_extractor.relationship import RelationshipExtractor


class TestMarketingFilter:
    """Test the _is_marketing_or_noreply filter method."""

    def test_noreply_patterns(self):
        """No-reply addresses should be filtered."""
        test_cases = [
            "noreply@company.com",
            "no-reply@service.org",
            "do-not-reply@platform.io",
            "donotreply@system.net",
            "mailer-daemon@mail.example.com",
            "postmaster@email.server.com",
            "auto-reply@automated.io",
            "autoreply@bot.com",
        ]

        for addr in test_cases:
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True, \
                f"Should filter no-reply address: {addr}"

    def test_bulk_sender_localparts(self):
        """Common bulk sender patterns should be filtered."""
        test_cases = [
            "newsletter@company.com",
            "notifications@platform.io",
            "updates@service.org",
            "digest@news.com",
            "marketing@brand.com",
            "promo@deals.io",
            "offers@shopping.com",
            "info@support.org",
            "hello@startup.io",
            "service@payment.com",
            "orders@shop.com",
            "receipts@store.io",
            "shipping@fulfillment.com",
            "rewards@loyalty.com",
        ]

        for addr in test_cases:
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True, \
                f"Should filter bulk sender: {addr}"

    def test_bulk_sender_no_false_positives(self):
        """Personal emails containing bulk keywords should NOT be filtered."""
        # These are legitimate personal addresses that happen to contain
        # words like "email", "reply", "service" but NOT at the start
        test_cases = [
            "john.email@company.com",
            "sarah.reply@startup.io",
            "james.service@consulting.biz",
            "info.specialist@corp.org",  # "info" not at start
        ]

        for addr in test_cases:
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is False, \
                f"Should NOT filter personal address: {addr}"

    def test_embedded_notification_patterns(self):
        """Addresses with notification patterns in the middle should be filtered."""
        test_cases = [
            "HOA-Notifications@community.org",
            "system-alerts@monitoring.io",
            "user-updates@platform.com",
            "team-digest@company.com",
        ]

        for addr in test_cases:
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True, \
                f"Should filter embedded notification: {addr}"

    def test_marketing_domain_patterns(self):
        """Marketing domain patterns should be filtered."""
        test_cases = [
            "promo@news-us.brand.com",
            "deals@email.shopping.com",
            "offers@reply.deals.io",
            "updates@mailing.service.org",
            "info@em.company.com",
            "alert@mg.platform.io",
            "notification@engage.ticketmaster.com",
        ]

        for addr in test_cases:
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True, \
                f"Should filter marketing domain: {addr}"

    def test_marketing_service_providers(self):
        """Third-party email marketing platforms should be filtered."""
        test_cases = [
            "brand@customer.e2ma.net",
            "company@send.sendgrid.net",
            "promo@list.mailchimp.com",
            "updates@email.constantcontact.com",
            "news@marketing.hubspot.com",
        ]

        for addr in test_cases:
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True, \
                f"Should filter marketing service provider: {addr}"

    def test_unsubscribe_detection(self):
        """Emails with unsubscribe links should be filtered."""
        payload_with_unsub = {
            "body_plain": "Great deals! Click here to unsubscribe from this list.",
            "snippet": "Special offer...",
        }

        assert RelationshipExtractor._is_marketing_or_noreply(
            "deals@shop.com", payload_with_unsub
        ) is True

        # Verify unsubscribe detection works in different payload fields
        assert RelationshipExtractor._is_marketing_or_noreply(
            "info@company.com", {"body": "To unsubscribe, click here"}
        ) is True

        assert RelationshipExtractor._is_marketing_or_noreply(
            "updates@service.org", {"snippet": "Unsubscribe at bottom"}
        ) is True

    def test_legitimate_human_contacts(self):
        """Real human email addresses should NOT be filtered."""
        test_cases = [
            "alice@company.com",
            "bob.smith@startup.io",
            "jane.doe@consulting.biz",
            "john+work@gmail.com",
            "sarah_jones@university.edu",
            "michael.brown@freelance.net",
        ]

        for addr in test_cases:
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is False, \
                f"Should NOT filter human contact: {addr}"

    def test_case_insensitivity(self):
        """Filter should work regardless of address case."""
        test_cases = [
            "NoReply@Company.COM",
            "NEWSLETTER@BRAND.IO",
            "Notifications@Service.ORG",
        ]

        for addr in test_cases:
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True, \
                f"Should filter case-insensitive: {addr}"


class TestRelationshipExtractionFiltering:
    """Test that marketing emails are filtered during relationship extraction."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        """Create RelationshipExtractor with test fixtures."""
        return RelationshipExtractor(db, user_model_store)

    def test_marketing_email_not_tracked(self, extractor):
        """Marketing emails should not create relationship signals."""
        marketing_event = {
            "id": "evt_001",
            "type": "email.received",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "from_address": "newsletter@company.com",
                "subject": "Weekly updates",
                "body": "Check out our latest products!",
            },
        }

        signals = extractor.extract(marketing_event)

        # Should return empty list — no signals for marketing senders
        assert signals == []

    def test_human_email_tracked(self, extractor):
        """Human emails should create relationship signals."""
        human_event = {
            "id": "evt_002",
            "type": "email.received",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "from_address": "alice@company.com",
                "subject": "Project update",
                "body": "Here's the latest on our project...",
            },
        }

        signals = extractor.extract(human_event)

        # Should return 1 signal for the human sender
        assert len(signals) == 1
        assert signals[0]["contact_address"] == "alice@company.com"
        assert signals[0]["type"] == "relationship_interaction"

    def test_outbound_marketing_not_tracked(self, extractor):
        """Outbound messages to marketing addresses shouldn't be tracked."""
        # This handles the rare case where the user sends TO a marketing address
        # (e.g., support request to support@company.com)
        outbound_event = {
            "id": "evt_003",
            "type": "email.sent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "to_addresses": ["support@company.com", "alice@company.com"],
                "subject": "Need help",
                "body": "Can you assist with...",
            },
        }

        signals = extractor.extract(outbound_event)

        # Should return only 1 signal (for alice@company.com, not support@)
        assert len(signals) == 1
        assert signals[0]["contact_address"] == "alice@company.com"

    def test_unsubscribe_link_filtering(self, extractor):
        """Emails with unsubscribe links should not be tracked."""
        marketing_event = {
            "id": "evt_004",
            "type": "email.received",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "from_address": "deals@shop.com",
                "subject": "Special offer",
                "body_plain": "Great products! To unsubscribe, click here.",
            },
        }

        signals = extractor.extract(marketing_event)

        # Should return empty list — unsubscribe link detected
        assert signals == []

    def test_relationship_profile_not_polluted(self, extractor, user_model_store):
        """Marketing emails should not pollute the relationships profile."""
        # Process a marketing email
        marketing_event = {
            "id": "evt_005",
            "type": "email.received",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "from_address": "promo@brand.com",
                "subject": "Sale!",
                "body": "Big discounts!",
            },
        }

        extractor.extract(marketing_event)

        # Check that the marketing sender is NOT in the relationships profile
        rel_profile = user_model_store.get_signal_profile("relationships")
        if rel_profile:
            contacts = rel_profile["data"].get("contacts", {})
            assert "promo@brand.com" not in contacts

    def test_communication_template_not_created_for_marketing(self, extractor, db):
        """Marketing emails should not generate communication templates."""
        marketing_event = {
            "id": "evt_006",
            "type": "email.received",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "from_address": "newsletter@company.com",
                "subject": "Weekly digest",
                "body_plain": "Hello! Here are this week's updates. " * 10,  # Long enough for template
                "channel": "email",
            },
        }

        extractor.extract(marketing_event)

        # Check that no template was created for the marketing sender
        with db.get_connection("user_model") as conn:
            templates = conn.execute(
                "SELECT * FROM communication_templates WHERE contact_id = ?",
                ("newsletter@company.com",)
            ).fetchall()

            assert len(templates) == 0


class TestRealWorldMarketingExamples:
    """Test with actual marketing email addresses found in production."""

    def test_real_marketing_addresses(self):
        """Real-world marketing addresses from the Life OS database should be filtered."""
        # These are actual addresses from the production database that were
        # polluting the relationship graph
        real_marketing_addresses = [
            "callofduty@comms.activision.com",
            "rei_email@email.rei.com",
            "hello@attn.us.lg.com",
            "LaMonaRosa@email.sevenrooms.com",
            "noreply@email.ryanlawn.com",
        ]

        for addr in real_marketing_addresses:
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True, \
                f"Real-world marketing address should be filtered: {addr}"
