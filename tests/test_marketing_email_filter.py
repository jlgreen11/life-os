"""
Tests for marketing email filtering.

This test suite covers both the shared email classifier (services/email_classifier.py)
and its integration via the prediction engine's _is_marketing_or_noreply method.
The shared classifier is used across the pipeline for early suppression,
prediction filtering, and rules engine fallback.
"""

import pytest

from services.email_classifier import is_marketing_email
from services.prediction_engine.engine import PredictionEngine


class TestMarketingEmailFilter:
    """Test suite for the shared is_marketing_email classifier."""

    def test_noreply_senders_are_filtered(self):
        """No-reply email addresses should be filtered out."""
        test_cases = [
            "noreply@example.com",
            "no-reply@service.com",
            "donotreply@company.com",
            "do-not-reply@business.org",
            "auto-reply@support.com",
            "autoreply@help.com",
            "automated@system.com",
        ]

        for addr in test_cases:
            assert is_marketing_email(addr, {}) is True, \
                f"Expected {addr} to be filtered as noreply"

    def test_mailer_daemon_is_filtered(self):
        """System mailer-daemon and postmaster addresses should be filtered."""
        test_cases = [
            "mailer-daemon@googlemail.com",
            "MAILER-DAEMON@yahoo.com",
            "postmaster@mail.example.com",
            "daemon@server.com",
        ]

        for addr in test_cases:
            assert is_marketing_email(addr, {}) is True, \
                f"Expected {addr} to be filtered as system sender"

    def test_bulk_localpart_patterns_are_filtered(self):
        """Common bulk sender local-parts should be filtered."""
        test_cases = [
            "newsletter@company.com",
            "notifications@service.com",
            "updates@platform.io",
            "digest@news.org",
            "mailer@bulk.com",
            "promo@store.com",
            "marketing@business.com",
            "reply@enterprise.com",
            "email@contact.com",
            "news@media.com",
            "offers@shop.com",
            "deals@ecommerce.com",
        ]

        for addr in test_cases:
            assert is_marketing_email(addr, {}) is True, \
                f"Expected {addr} to be filtered as bulk sender"

    def test_new_bulk_localpart_patterns(self):
        """Newly added bulk sender local-parts should also be filtered."""
        test_cases = [
            "alerts@company.com",
            "announce@service.com",
            "campaign@marketing.com",
            "promotions@store.com",
            "store@brand.com",
            "shop@retailer.com",
            "sales@business.com",
            "team@company.com",
        ]

        for addr in test_cases:
            assert is_marketing_email(addr, {}) is True, \
                f"Expected {addr} to be filtered as bulk sender"

    def test_marketing_domain_patterns_are_filtered(self):
        """Marketing domain patterns should be filtered."""
        test_cases = [
            "boss@news-us.hugoboss.com",
            "D23@email.d23.com",
            "RoyalCaribbean@reply.royalcaribbean.com",
            "sender@newsletters.company.com",
            "promo@marketing.service.com",
            "msg@em.platform.com",
            "alert@mg.notifications.io",
            "info@mail.business.org",
        ]

        for addr in test_cases:
            assert is_marketing_email(addr, {}) is True, \
                f"Expected {addr} to be filtered as marketing domain"

    def test_new_marketing_domain_patterns(self):
        """Newly added marketing domain patterns should be filtered."""
        test_cases = [
            "sender@bounce.company.com",
            "msg@send.service.com",
            "alert@campaign.brand.com",
            "notice@comms.platform.io",
            "info@e.retailer.com",
        ]

        for addr in test_cases:
            assert is_marketing_email(addr, {}) is True, \
                f"Expected {addr} to be filtered as marketing domain"

    def test_common_support_addresses_are_filtered(self):
        """Common support/info addresses are often automated and should be filtered."""
        test_cases = [
            "hello@startup.com",
            "info@business.org",
            "support@platform.io",
            "help@service.com",
        ]

        for addr in test_cases:
            assert is_marketing_email(addr, {}) is True, \
                f"Expected {addr} to be filtered as support address"

    def test_unsubscribe_in_body_is_filtered(self):
        """Emails with 'unsubscribe' in body/snippet should be filtered."""
        test_cases = [
            {"body_plain": "Click here to unsubscribe from our emails"},
            {"snippet": "To unsubscribe, visit..."},
            {"body": "<a href='...'>Unsubscribe</a>"},
            {"body_plain": "UNSUBSCRIBE"},  # Case insensitive
        ]

        for payload in test_cases:
            assert is_marketing_email("sender@example.com", payload) is True, \
                f"Expected email with unsubscribe in {list(payload.keys())} to be filtered"

    def test_expanded_optout_phrases_in_body(self):
        """Expanded opt-out phrases should also trigger filtering."""
        test_cases = [
            {"body_plain": "To opt out of these emails, click here"},
            {"body_plain": "To opt-out of future emails, click here"},
            {"body_plain": "Manage your preferences at this link"},
            {"body_plain": "Update your preferences to stop receiving"},
            {"body_plain": "Email preferences can be changed here"},
            {"body_plain": "If you no longer wish to receive these emails"},
            {"body_plain": "You are on our mailing list because you signed up"},
            {"body_plain": "Stop receiving these emails by clicking here"},
            {"body_plain": "Remove from this list"},
            {"body_plain": "Subscription preferences can be managed here"},
        ]

        for payload in test_cases:
            assert is_marketing_email("sender@example.com", payload) is True, \
                f"Expected email with '{payload['body_plain'][:40]}...' to be filtered"

    def test_marketing_subject_with_unsubscribe_body(self):
        """Promotional subjects combined with unsubscribe in body should be filtered."""
        test_cases = [
            {"subject": "50% OFF everything this weekend!", "body_plain": "Click to unsubscribe"},
            {"subject": "Flash sale - ending soon!", "body_plain": "To opt out, click here"},
            {"subject": "Free shipping on all orders", "body_plain": "Manage preferences"},
            {"subject": "Don't miss our exclusive offer", "body_plain": "Unsubscribe here"},
            {"subject": "New arrivals just for you", "body_plain": "Email preferences"},
            {"subject": "Limited time: 20% off", "body_plain": "Mailing list"},
            {"subject": "Shop now - BOGO deals", "body_plain": "Unsubscribe"},
            {"subject": "Your coupon code inside", "body_plain": "Opt out"},
            {"subject": "Clearance event starts today", "body_plain": "Unsubscribe"},
            {"subject": "Last chance for this discount!", "body_plain": "Unsubscribe"},
        ]

        for payload in test_cases:
            assert is_marketing_email("sender@example.com", payload) is True, \
                f"Expected email with subject '{payload['subject']}' to be filtered"

    def test_marketing_subject_without_unsubscribe_not_filtered(self):
        """Promotional-looking subjects WITHOUT body opt-out should NOT be filtered.

        Subject alone is not enough — we need body confirmation to avoid
        false positives on legitimate emails with marketing-like words.
        """
        test_cases = [
            {"subject": "The sale of our property is complete", "body_plain": "Dear John, the closing went well."},
            {"subject": "Discount rate discussion", "body_plain": "Let's talk about the Fed's rate."},
            {"subject": "Free time this weekend?", "body_plain": "Want to grab coffee?"},
        ]

        for payload in test_cases:
            assert is_marketing_email("sender@example.com", payload) is False, \
                f"Expected email with subject '{payload['subject']}' to NOT be filtered"

    def test_personal_emails_are_not_filtered(self):
        """Personal email addresses should NOT be filtered."""
        test_cases = [
            "john.doe@gmail.com",
            "jane_smith@company.com",
            "alice.b.cooper@university.edu",
            "bob123@protonmail.com",
            "founder@new-company.com",
        ]

        for addr in test_cases:
            assert is_marketing_email(addr, {}) is False, \
                f"Expected {addr} to NOT be filtered (personal email)"

    def test_case_insensitivity(self):
        """Filter should be case-insensitive."""
        test_cases = [
            "NoReply@Example.Com",
            "NEWSLETTER@COMPANY.COM",
            "Marketing@Service.Com",
            "MAILER-DAEMON@SYSTEM.ORG",
        ]

        for addr in test_cases:
            assert is_marketing_email(addr, {}) is True, \
                f"Expected {addr} to be filtered (case insensitive)"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty sender should not crash
        assert is_marketing_email("", {}) is False

        # Missing payload fields should not crash
        assert is_marketing_email("sender@example.com", {}) is False

        # Multiple indicators should still return True
        payload = {"body_plain": "Click to unsubscribe"}
        assert is_marketing_email("newsletter@company.com", payload) is True

    def test_real_world_spam_cases(self):
        """Test against actual spam cases from the production database."""
        # These are the actual unresolved predictions that triggered this fix
        real_spam = [
            "RoyalCaribbean@reply.royalcaribbean.com",
            "mailer-daemon@googlemail.com",
            "D23@email.d23.com",
            "boss@news-us.hugoboss.com",
        ]

        for addr in real_spam:
            assert is_marketing_email(addr, {}) is True, \
                f"Expected {addr} to be filtered (real production spam case)"

    def test_legitimate_emails_with_similar_patterns(self):
        """Ensure we don't over-filter legitimate emails with similar patterns.

        The filter should only match bulk sender patterns at the START of the
        email address (e.g., email@company.com) and not in the middle
        (e.g., john.email@company.com). This prevents false positives while
        still catching actual bulk senders.
        """
        # These should NOT be filtered - patterns appear in middle, not at start
        legitimate = [
            "john.email@company.com",   # has 'email@' but not at start
            "sarah.reply@startup.io",   # has 'reply@' but not at start
            "team.info@company.com",    # has 'info@' but not at start
            "contact.hello@business.org",  # has 'hello@' but not at start
        ]

        for addr in legitimate:
            assert is_marketing_email(addr, {}) is False, \
                f"Expected {addr} to NOT be filtered (pattern not at start)"

        # But these SHOULD be filtered - patterns at the start
        bulk_senders = [
            "email@company.com",
            "reply@startup.io",
            "info@business.org",
            "hello@service.com",
        ]

        for addr in bulk_senders:
            assert is_marketing_email(addr, {}) is True, \
                f"Expected {addr} to be filtered (pattern at start)"


class TestPredictionEngineIntegration:
    """Verify PredictionEngine._is_marketing_or_noreply delegates to shared classifier."""

    def test_prediction_engine_delegates_to_shared_classifier(self):
        """PredictionEngine._is_marketing_or_noreply should delegate to is_marketing_email."""
        # Verify a few representative cases match between the two
        test_cases = [
            ("noreply@example.com", {}, True),
            ("newsletter@company.com", {}, True),
            ("boss@news-us.hugoboss.com", {}, True),
            ("john.doe@gmail.com", {}, False),
            ("sender@example.com", {"body_plain": "Click to unsubscribe"}, True),
            ("sender@example.com", {}, False),
        ]

        for addr, payload, expected in test_cases:
            result = PredictionEngine._is_marketing_or_noreply(addr, payload)
            assert result is expected, \
                f"PredictionEngine._is_marketing_or_noreply({addr}) = {result}, expected {expected}"
