"""
Tests for marketing email filtering in the prediction engine.

This test suite ensures that the prediction engine correctly filters out
marketing, bulk, and automated emails to prevent low-quality prediction spam
and protect the accuracy feedback loop.
"""

import pytest

from services.prediction_engine.engine import PredictionEngine


class TestMarketingEmailFilter:
    """Test suite for the _is_marketing_or_noreply static method."""

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
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True, \
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
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True, \
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
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True, \
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
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True, \
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
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True, \
                f"Expected {addr} to be filtered as support address"

    def test_unsubscribe_in_body_is_filtered(self):
        """Emails with 'unsubscribe' plus bulk phrases in body/snippet should be filtered."""
        test_cases = [
            {"body_plain": "Click here to unsubscribe from our emails"},
            {"snippet": "To unsubscribe from our mailing list, visit..."},
            {"body": "<a href='...'>Unsubscribe</a> | Manage your subscription"},
            {"body_plain": "UNSUBSCRIBE from future emails"},  # Case insensitive
        ]

        for payload in test_cases:
            assert PredictionEngine._is_marketing_or_noreply("sender@example.com", payload) is True, \
                f"Expected email with unsubscribe in {list(payload.keys())} to be filtered"

    def test_personal_emails_are_not_filtered(self):
        """Personal email addresses should NOT be filtered."""
        test_cases = [
            "john.doe@gmail.com",
            "jane_smith@company.com",
            "alice.b.cooper@university.edu",
            "bob123@protonmail.com",
            "team@small-startup.io",
            "founder@new-company.com",
        ]

        for addr in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is False, \
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
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True, \
                f"Expected {addr} to be filtered (case insensitive)"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty sender should not crash
        assert PredictionEngine._is_marketing_or_noreply("", {}) is False

        # Missing payload fields should not crash
        assert PredictionEngine._is_marketing_or_noreply("sender@example.com", {}) is False

        # Multiple indicators should still return True
        payload = {"body_plain": "Click to unsubscribe"}
        assert PredictionEngine._is_marketing_or_noreply("newsletter@company.com", payload) is True

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
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True, \
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
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is False, \
                f"Expected {addr} to NOT be filtered (pattern not at start)"

        # But these SHOULD be filtered - patterns at the start
        bulk_senders = [
            "email@company.com",
            "reply@startup.io",
            "info@business.org",
            "hello@service.com",
        ]

        for addr in bulk_senders:
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True, \
                f"Expected {addr} to be filtered (pattern at start)"
