"""
Tests for enhanced marketing email filter (iteration 152).

PROBLEM:
The marketing filter in iteration 155 missed many common patterns, allowing
300+ marketing emails to be tracked as legitimate relationships. This broke
relationship maintenance predictions which generated 0 predictions despite
319 "eligible" contacts.

MISSED PATTERNS:
- no_reply@ (underscore variant)
- notification@ (singular, not plural)
- @m. (mobile/marketing subdomains like @m.starbucks.com)
- @notification. (subdomains like @notification.capitalone.com)
- @marketing. (direct marketing subdomains)
- @care. (customer care automated systems)
- @mcmap. (marketing campaign managers)
- team@, yourhealth@, smartoption@ (generic automated senders)
- no-reply+ID@ (+ modifier breaks simple substring match)

SOLUTION:
Enhance both _is_marketing_or_noreply() implementations (PredictionEngine and
RelationshipExtractor) to catch all these patterns.

IMPACT:
- Enables relationship maintenance predictions (currently 0 generated)
- Cleans 300+ false-positive contacts from relationships profile
- Improves prediction engine performance (stops processing 300+ fake contacts every 15min)
"""

import pytest

from services.prediction_engine.engine import PredictionEngine
from services.signal_extractor.relationship import RelationshipExtractor


class TestEnhancedMarketingFilter:
    """Test enhanced marketing filter catches all common patterns."""

    # Test data: real addresses from production that were incorrectly tracked as contacts
    MARKETING_ADDRESSES = [
        # No-reply variants with underscores
        "no_reply@mcmap.chase.com",
        "no_reply@example.com",

        # No-reply with + modifiers
        "no-reply+4b0c6802@toast-restaurants.com",
        "noreply+tracking@shopify.com",
        "no_reply+12345@github.com",

        # Automation variants
        "automation@stripe.com",

        # Singular notification (not notifications)
        "notification@paypal.com",

        # Team addresses removed - could be legitimate startups
        # "team@kickstargogo.com", "team@slack.com" - too many false positives

        # Mobile/marketing subdomains (@m.)
        "Starbucks@m.starbucks.com",
        "updates@m.facebook.com",
        "deals@m.kohls.com",

        # Notification subdomains
        "capitalone@notification.capitalone.com",
        "alerts@notifications.bankofamerica.com",

        # Care/support subdomains
        "YourHealth@care.kansashealthsystem.com",
        "account@care.uber.com",

        # Marketing campaign managers
        "no_reply@mcmap.chase.com",
        "offers@mcmap.wellsfargo.com",

        # Prospect/sales management
        "smartoption@soslprospect.salliemae.com",
        "application@soslprospect.discover.com",

        # Personalized automated senders
        "yourhealth@healthsystem.com",
        "youraccount@banking.com",
        "smartoption@lender.com",
        "quickalert@creditcard.com",

        # Generic automated patterns
        "update@app.com",
        "offer@store.com",
        "confirm@booking.com",
        "account@service.com",
        "services@provider.com",

        # Other common marketing domains
        "deals@marketing.target.com",
        "news@campaigns.nytimes.com",
        "promo@blast.retailer.com",
        "updates@lists.newsletter.com",
        "alerts@messages.bank.com",

        # Transaction notification platforms
        "receipt@txn.square.com",
        "order@transactional.shopify.com",

        # Communication platforms
        "updates@communications.company.com",
        "news@comms.activision.com",
    ]

    # Test data: legitimate human addresses that should NOT be filtered
    HUMAN_ADDRESSES = [
        # Personal email providers (must not be blocked)
        "john.doe@gmail.com",
        "jane.smith@hotmail.com",
        "alice@protonmail.com",
        "bob@outlook.com",

        # Work emails (legitimate contacts)
        "sarah.johnson@company.com",
        "michael.team@startup.io",  # "team" is part of surname, not local-part
        "emily.reply@consultancy.com",  # "reply" is part of surname

        # Personal domains
        "hello@personalsite.com",  # Personal blog/portfolio (context-dependent)
        "info@smallbusiness.local",  # Small business owner
    ]

    def test_prediction_engine_filter_catches_all_marketing(self):
        """PredictionEngine._is_marketing_or_noreply() catches all marketing patterns."""
        for addr in self.MARKETING_ADDRESSES:
            result = PredictionEngine._is_marketing_or_noreply(addr, {})
            assert result is True, f"Failed to filter marketing address: {addr}"

    def test_relationship_extractor_filter_catches_all_marketing(self):
        """RelationshipExtractor._is_marketing_or_noreply() catches all marketing patterns."""
        for addr in self.MARKETING_ADDRESSES:
            result = RelationshipExtractor._is_marketing_or_noreply(addr, {})
            assert result is True, f"Failed to filter marketing address: {addr}"

    def test_prediction_engine_filter_allows_humans(self):
        """PredictionEngine._is_marketing_or_noreply() allows legitimate human contacts."""
        for addr in self.HUMAN_ADDRESSES:
            result = PredictionEngine._is_marketing_or_noreply(addr, {})
            # hello@ and info@ are borderline cases - they're filtered by both implementations
            # for safety, so we skip them in this test
            if addr.startswith("hello@") or addr.startswith("info@"):
                continue
            assert result is False, f"Incorrectly filtered human address: {addr}"

    def test_relationship_extractor_filter_allows_humans_strict(self):
        """RelationshipExtractor is more strict than PredictionEngine (keeps @mail. filter)."""
        # RelationshipExtractor filters hello@ and info@ more aggressively
        # This is intentional - building long-term contact profiles should be stricter

        legitimate = [
            "john.doe@gmail.com",
            "jane.smith@hotmail.com",
            "alice@protonmail.com",
            "bob@outlook.com",
            "sarah.johnson@company.com",
            "michael.team@startup.io",
            "emily.reply@consultancy.com",
        ]

        for addr in legitimate:
            result = RelationshipExtractor._is_marketing_or_noreply(addr, {})
            assert result is False, f"Incorrectly filtered human address: {addr}"

    def test_underscore_variants_caught(self):
        """Underscore variants of no-reply are caught (no_reply@, do_not_reply@)."""
        test_cases = [
            "no_reply@domain.com",
            "do_not_reply@service.com",
            "auto_reply@company.com",
        ]

        for addr in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True

    def test_plus_modifier_variants_caught(self):
        """Plus modifiers in no-reply addresses are caught (no-reply+ID@)."""
        test_cases = [
            "no-reply+12345@service.com",
            "noreply+abc@app.com",
            "no_reply+tracking@platform.com",
        ]

        for addr in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True

    def test_singular_plural_consistency(self):
        """Both singular and plural forms are caught (notification/notifications)."""
        singular_plural_pairs = [
            ("notification@", "notifications@"),
            ("update@", "updates@"),
            ("offer@", "offers@"),
            ("alert@", "alerts@"),
            ("service@", "services@"),
        ]

        for singular, plural in singular_plural_pairs:
            addr_singular = f"{singular}domain.com"
            addr_plural = f"{plural}domain.com"

            assert PredictionEngine._is_marketing_or_noreply(addr_singular, {}) is True
            assert PredictionEngine._is_marketing_or_noreply(addr_plural, {}) is True
            assert RelationshipExtractor._is_marketing_or_noreply(addr_singular, {}) is True
            assert RelationshipExtractor._is_marketing_or_noreply(addr_plural, {}) is True

    def test_mobile_marketing_subdomains(self):
        """Mobile/marketing subdomains are caught (@m., @notification., @marketing.)."""
        test_cases = [
            "updates@m.starbucks.com",
            "deals@m.retailer.com",
            "alerts@notification.bank.com",
            "news@notifications.company.com",
            "promo@marketing.brand.com",
        ]

        for addr in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True

    def test_customer_care_subdomains(self):
        """Customer care subdomains are caught (@care.)."""
        test_cases = [
            "alerts@care.healthsystem.com",
            "updates@care.insurance.com",
        ]

        for addr in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True

    def test_campaign_manager_subdomains(self):
        """Marketing campaign manager subdomains are caught (@mcmap., @soslprospect.)."""
        test_cases = [
            "offers@mcmap.chase.com",
            "no_reply@mcmap.wellsfargo.com",
            "application@soslprospect.salliemae.com",
        ]

        for addr in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True

    def test_personalized_automated_senders(self):
        """Personalized automated senders are caught (yourhealth@, youraccount@, smartoption@)."""
        test_cases = [
            "yourhealth@hospital.com",
            "youraccount@bank.com",
            "smartoption@lender.com",
            "quickalert@creditcard.com",
        ]

        for addr in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True

    def test_transaction_notification_platforms(self):
        """Transaction notification platform domains are caught (@txn., @transactional.)."""
        test_cases = [
            "receipt@txn.square.com",
            "order@transactional.shopify.com",
        ]

        for addr in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(addr, {}) is True
            assert RelationshipExtractor._is_marketing_or_noreply(addr, {}) is True

    def test_unsubscribe_link_detection(self):
        """Emails with unsubscribe links are caught (required by law for marketing)."""
        test_cases = [
            {
                "addr": "sender@example.com",
                "payload": {"body": "Check out our deals! Unsubscribe here: http://..."},
                "should_filter": True,
            },
            {
                "addr": "sender@example.com",
                "payload": {"snippet": "Great offers... unsubscribe at bottom"},
                "should_filter": True,
            },
            {
                "addr": "sender@example.com",
                "payload": {"body": "Hey, can we meet tomorrow?"},
                "should_filter": False,
            },
        ]

        for case in test_cases:
            result_pe = PredictionEngine._is_marketing_or_noreply(case["addr"], case["payload"])
            result_re = RelationshipExtractor._is_marketing_or_noreply(case["addr"], case["payload"])

            assert result_pe == case["should_filter"], f"PredictionEngine failed for: {case}"
            assert result_re == case["should_filter"], f"RelationshipExtractor failed for: {case}"

    def test_real_production_addresses(self):
        """Test with actual addresses from production that were incorrectly tracked."""
        # These are real addresses that slipped through the old filter
        production_marketing = [
            # NOTE: team@kickstargogo.com removed - "team@" could be legitimate startups
            # We're being conservative to avoid false positives (blocking real contacts)
            "lafconews@lafco.com",
            "YourHealth@care.kansashealthsystem.com",
            "smartoption@soslprospect.salliemae.com",
            "capitalone@notification.capitalone.com",
            "Starbucks@m.starbucks.com",
            "no_reply@mcmap.chase.com",
            "no-reply+4b0c6802@toast-restaurants.com",
        ]

        for addr in production_marketing:
            result_pe = PredictionEngine._is_marketing_or_noreply(addr, {})
            result_re = RelationshipExtractor._is_marketing_or_noreply(addr, {})

            assert result_pe is True, f"PredictionEngine missed: {addr}"
            assert result_re is True, f"RelationshipExtractor missed: {addr}"
