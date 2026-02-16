"""
Test enhanced marketing email filter patterns.

This test suite validates that the _is_marketing_or_noreply() function correctly
identifies and filters out marketing/automated emails that were previously slipping
through, polluting the relationship tracking and prediction systems.

Context:
    Production data showed 820 "contacts" in the relationship profile when only ~20
    were actual human relationships. Marketing emails like callofduty@comms.activision.com
    were being tracked as relationships, breaking relationship maintenance predictions.

Fix:
    Enhanced the marketing filter with additional domain patterns (@comms., @attn.,
    @txn., etc.) and service provider subdomains (.sailthru.com, .klaviyo.com, etc.)
    to catch these patterns.
"""

import pytest
from services.prediction_engine.engine import PredictionEngine


class TestEnhancedMarketingFilter:
    """Test that enhanced marketing filter catches previously-missed patterns."""

    def test_comms_subdomain_pattern(self):
        """
        Test that @comms. subdomain pattern is filtered.

        Real-world example from production: callofduty@comms.activision.com
        had 284 interactions tracked as a "relationship" when it's clearly
        automated marketing for a video game.
        """
        assert PredictionEngine._is_marketing_or_noreply(
            "callofduty@comms.activision.com", {}
        ) is True

        assert PredictionEngine._is_marketing_or_noreply(
            "updates@comms.github.com", {}
        ) is True

    def test_attn_subdomain_pattern(self):
        """
        Test that @attn. subdomain pattern is filtered.

        Real-world example: hello@attn.us.lg.com appeared in top contacts
        despite being an automated notification platform.
        """
        assert PredictionEngine._is_marketing_or_noreply(
            "hello@attn.us.lg.com", {}
        ) is True

        assert PredictionEngine._is_marketing_or_noreply(
            "notifications@attn.example.com", {}
        ) is True

    def test_communications_subdomain_pattern(self):
        """Test that @communications. subdomain pattern is filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "team@communications.company.com", {}
        ) is True

    def test_transactional_patterns(self):
        """Test that transactional email patterns are filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "order@txn.shopify.com", {}
        ) is True

        assert PredictionEngine._is_marketing_or_noreply(
            "receipt@transactional.stripe.com", {}
        ) is True

    def test_promotional_patterns(self):
        """Test that promotional campaign patterns are filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "sale@deals.nordstrom.com", {}
        ) is True

        assert PredictionEngine._is_marketing_or_noreply(
            "summer@offers.target.com", {}
        ) is True

        assert PredictionEngine._is_marketing_or_noreply(
            "weekly@promo-emails.walmart.com", {}
        ) is True

    def test_campaign_management_patterns(self):
        """Test that campaign management platform patterns are filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "blast@campaigns.mailgun.com", {}
        ) is True

        assert PredictionEngine._is_marketing_or_noreply(
            "monthly@campaign.hubspot.com", {}
        ) is True

    def test_bulk_sender_patterns(self):
        """Test that bulk sender platform patterns are filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "announcement@blast.company.com", {}
        ) is True

        assert PredictionEngine._is_marketing_or_noreply(
            "newsletter@bulk.example.com", {}
        ) is True

    def test_mailing_list_patterns(self):
        """Test that mailing list manager patterns are filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "subscribe@lists.apache.org", {}
        ) is True

        assert PredictionEngine._is_marketing_or_noreply(
            "updates@list.python.org", {}
        ) is True

    def test_new_service_provider_patterns(self):
        """
        Test that newly-added marketing service provider patterns are filtered.

        These are third-party platforms that companies use to send bulk emails.
        Previously, only a few major platforms were caught. Now we catch 15+ more.
        """
        # Sailthru
        assert PredictionEngine._is_marketing_or_noreply(
            "deals@sender.sailthru.com", {}
        ) is True

        # Klaviyo (e-commerce marketing)
        assert PredictionEngine._is_marketing_or_noreply(
            "promo@track.klaviyo.com", {}
        ) is True

        # Customer.io
        assert PredictionEngine._is_marketing_or_noreply(
            "onboarding@track.customer.io", {}
        ) is True

        # Iterable
        assert PredictionEngine._is_marketing_or_noreply(
            "campaign@links.iterable.com", {}
        ) is True

        # Sendinblue/Brevo
        assert PredictionEngine._is_marketing_or_noreply(
            "newsletter@sender.sendinblue.com", {}
        ) is True

        # SparkPost
        assert PredictionEngine._is_marketing_or_noreply(
            "transactional@sender.sparkpostmail.com", {}
        ) is True

        # Intercom
        assert PredictionEngine._is_marketing_or_noreply(
            "notification@intercom-mail.com", {}
        ) is True

    def test_transactional_service_providers(self):
        """Test that transactional email service patterns are filtered."""
        # Postmark
        assert PredictionEngine._is_marketing_or_noreply(
            "receipts@pm-bounces.postmarkapp.com", {}
        ) is True

        # Mandrill
        assert PredictionEngine._is_marketing_or_noreply(
            "order@mandrillapp.com", {}
        ) is True

        # SMTP2GO
        assert PredictionEngine._is_marketing_or_noreply(
            "alert@smtp2go.com", {}
        ) is True

    def test_cloud_service_providers(self):
        """Test that cloud email service patterns are filtered."""
        # Amazon SES
        assert PredictionEngine._is_marketing_or_noreply(
            "billing@notifications.amazonses.com", {}
        ) is True

        # Oracle Responsys
        assert PredictionEngine._is_marketing_or_noreply(
            "promo@sender.responsys.net", {}
        ) is True

        # Salesforce Marketing Cloud (ExactTarget)
        assert PredictionEngine._is_marketing_or_noreply(
            "campaign@sender.exacttarget.com", {}
        ) is True

    def test_existing_patterns_still_work(self):
        """Ensure that previously-working patterns still filter correctly."""
        # @email. pattern (already existed)
        assert PredictionEngine._is_marketing_or_noreply(
            "rei_email@email.rei.com", {}
        ) is True

        assert PredictionEngine._is_marketing_or_noreply(
            "LaMonaRosa@email.sevenrooms.com", {}
        ) is True

        # hello@ pattern (already existed)
        assert PredictionEngine._is_marketing_or_noreply(
            "hello@company.com", {}
        ) is True

        # no-reply pattern (already existed)
        assert PredictionEngine._is_marketing_or_noreply(
            "no-reply@github.com", {}
        ) is True

    def test_legitimate_addresses_not_filtered(self):
        """
        Ensure that legitimate human email addresses are NOT filtered.

        These should pass through to enable relationship maintenance predictions.
        """
        # Personal email addresses
        assert PredictionEngine._is_marketing_or_noreply(
            "john.smith@gmail.com", {}
        ) is False

        assert PredictionEngine._is_marketing_or_noreply(
            "sarah.johnson@company.com", {}
        ) is False

        # Work email addresses
        assert PredictionEngine._is_marketing_or_noreply(
            "alice@startup.io", {}
        ) is False

        assert PredictionEngine._is_marketing_or_noreply(
            "bob.wilson@consulting.com", {}
        ) is False

        # Edge cases - contains marketing keywords but in name not domain
        assert PredictionEngine._is_marketing_or_noreply(
            "marketing.director@company.com", {}  # Person whose title is "marketing"
        ) is False

        assert PredictionEngine._is_marketing_or_noreply(
            "email.person@company.com", {}  # Person whose name happens to be "email"
        ) is False

    def test_filter_prevents_relationship_pollution(self):
        """
        Integration test: verify that the enhanced filter would prevent the
        production issue where 820 "contacts" were tracked when only ~20 were
        actual human relationships.

        This test uses actual addresses from the production relationship profile.
        """
        # These should ALL be filtered (were polluting the relationship profile)
        marketing_addresses = [
            "callofduty@comms.activision.com",
            "rei_email@email.rei.com",
            "hello@attn.us.lg.com",
            "LaMonaRosa@email.sevenrooms.com",
        ]

        for addr in marketing_addresses:
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True, \
                f"{addr} should be filtered as marketing but wasn't"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty string
        assert PredictionEngine._is_marketing_or_noreply("", {}) is False

        # No @ symbol (malformed address)
        assert PredictionEngine._is_marketing_or_noreply("notanemail", {}) is False

        # Multiple @ symbols (malformed)
        assert PredictionEngine._is_marketing_or_noreply("user@@domain.com", {}) is False
