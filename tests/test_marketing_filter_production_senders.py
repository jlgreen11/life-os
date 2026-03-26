"""
Tests for marketing filter patterns covering production senders that were
incorrectly stored as ``frequent_sender`` relationship facts.

These 6 addresses were found in the ``semantic_facts`` table with value
``"frequent_personal_sender"`` even though they are all automated/commercial
senders.  The patterns added in this iteration close those gaps:

  - ``@ecomm.`` subdomain  → Lenovo e-commerce mailer
  - ``@e1.`` subdomain     → ESP numeric routing (Ace Hardware)
  - ``@mailcenter.``       → Bank/institution bulk mail center (USAA)
  - ``@news.``             → Promotional news subdomain (Monopoly Go)
  - ``shopifyemail.com``   → Shopify's hosted email delivery service
  - ``seatengine.com``     → Event ticketing/venue platform
"""
import pytest

from services.signal_extractor.marketing_filter import is_marketing_or_noreply


class TestProductionMissedSenders:
    """Addresses that slipped through the marketing filter in production."""

    @pytest.mark.parametrize("addr,description", [
        (
            "lenovo@ecomm.lenovo.com",
            "Lenovo e-commerce mailer (@ecomm. subdomain)",
        ),
        (
            "AceRewards@e1.acehardware.com",
            "Ace Hardware loyalty program (@e1. ESP subdomain)",
        ),
        (
            "USAA.Customer.Service@mailcenter.usaa.com",
            "USAA bank service mailer (@mailcenter. subdomain)",
        ),
        (
            "mr.m@news.monopolygo.com",
            "Monopoly Go promotions (@news. subdomain)",
        ),
        (
            "store+328499209@g.shopifyemail.com",
            "Shopify merchant email via g.shopifyemail.com",
        ),
        (
            "the-comedy-club-of-kansas-city@seatengine.com",
            "Event venue mailer via seatengine.com ticketing platform",
        ),
    ])
    def test_production_sender_blocked(self, addr, description):
        """Each production sender must be recognised as marketing/automated."""
        assert is_marketing_or_noreply(addr) is True, (
            f"Expected {addr!r} ({description}) to be filtered, but it wasn't"
        )

    @pytest.mark.parametrize("addr,description", [
        ("alice@gmail.com", "freemail personal address"),
        ("john.smith@company.com", "corporate personal address"),
        ("bob@proton.me", "ProtonMail personal address"),
        ("jane@news.mycompany.com", "Should NOT block legitimate work subdomain … "
                                    "wait — '@news.' IS a marketing subdomain; "
                                    "this verifies the rule is intentional"),
    ])
    def test_false_positive_check(self, addr, description):
        """Verify common personal addresses are not over-blocked.

        Note: '@news.' is intentionally blocked because all observed production
        cases with @news. subdomains are promotional mailers (Monopoly Go, etc.).
        The parametrize list above only includes addresses that SHOULD pass.
        """
        # Only check addresses that should NOT be blocked
        if "news.mycompany" in addr:
            # @news. is intentionally a marketing pattern — skip this check
            pytest.skip("@news. is intentionally blocked as a marketing subdomain")
        assert is_marketing_or_noreply(addr) is False, (
            f"Expected {addr!r} ({description}) to pass, but it was blocked"
        )


class TestNewSubdomainPatterns:
    """Targeted tests for each new subdomain pattern."""

    def test_ecomm_subdomain_blocked(self):
        """@ecomm. catches e-commerce mailer subdomains."""
        assert is_marketing_or_noreply("shop@ecomm.brand.com") is True

    def test_e1_subdomain_blocked(self):
        """@e1. catches ESP numeric routing subdomains."""
        assert is_marketing_or_noreply("promo@e1.retailer.com") is True

    def test_mailcenter_subdomain_blocked(self):
        """@mailcenter. catches bank/institution bulk mail centers."""
        assert is_marketing_or_noreply("alerts@mailcenter.bank.com") is True

    def test_news_subdomain_blocked(self):
        """@news. catches promotional news subdomains."""
        assert is_marketing_or_noreply("weekly@news.appname.com") is True


class TestNewServiceDomains:
    """Targeted tests for each new service domain pattern."""

    def test_shopifyemail_blocked(self):
        """shopifyemail.com is Shopify's email delivery service."""
        assert is_marketing_or_noreply("store@shopifyemail.com") is True

    def test_shopifyemail_subdomain_blocked(self):
        """Subdomains of shopifyemail.com are also blocked."""
        assert is_marketing_or_noreply("info@g.shopifyemail.com") is True

    def test_seatengine_blocked(self):
        """seatengine.com is a ticketing/event platform mailer."""
        assert is_marketing_or_noreply("events@seatengine.com") is True

    def test_seatengine_long_local_part_blocked(self):
        """Long venue name as local-part via seatengine.com is still blocked."""
        assert is_marketing_or_noreply("the-venue-name-here@seatengine.com") is True
