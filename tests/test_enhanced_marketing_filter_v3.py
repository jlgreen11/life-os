"""
Test enhanced marketing filter patterns in RelationshipExtractor.

ITERATION 168: Comprehensive marketing filter enhancement to enable relationship
maintenance predictions by eliminating 300+ false-positive contacts from the
relationships profile.
"""

import pytest
from services.signal_extractor.relationship import RelationshipExtractor


@pytest.fixture
def extractor():
    """Create a RelationshipExtractor instance for testing."""
    return RelationshipExtractor({}, {})


class TestEcommercePatterns:
    """Test filtering of e-commerce and sales email patterns."""

    def test_filters_sales_addresses(self, extractor):
        """Sales@ addresses should be filtered as marketing."""
        assert extractor._is_marketing_or_noreply("sales@atlanticfirearms.com", {})
        assert extractor._is_marketing_or_noreply("sale@example.com", {})

    def test_filters_shop_store_patterns(self, extractor):
        """Shop@ and store@ should be filtered."""
        assert extractor._is_marketing_or_noreply("shop@example.com", {})
        assert extractor._is_marketing_or_noreply("store@example.com", {})

    def test_filters_concierge_addresses(self, extractor):
        """Concierge@ (e-commerce customer service) should be filtered."""
        assert extractor._is_marketing_or_noreply("concierge@tommyjohn.com", {})

    def test_filters_ecommerce_subdomains(self, extractor):
        """E-commerce subdomains (@ecomm., @shop., @store.) should be filtered."""
        assert extractor._is_marketing_or_noreply("lenovo@ecomm.lenovo.com", {})
        assert extractor._is_marketing_or_noreply("test@shop.example.com", {})
        assert extractor._is_marketing_or_noreply("test@store.example.com", {})

    def test_filters_partners_addresses(self, extractor):
        """Partners@ (affiliate marketing) should be filtered."""
        assert extractor._is_marketing_or_noreply("partners@iemail.moneylion.com", {})
        assert extractor._is_marketing_or_noreply("partner@example.com", {})


class TestBrandSelfMailerPattern:
    """Test company@company.com pattern (brand self-mailers)."""

    def test_filters_exact_match_self_mailers(self, extractor):
        """When local-part matches domain, it's almost always marketing."""
        assert extractor._is_marketing_or_noreply("cutleryandmore@cutleryandmore.com", {})
        assert extractor._is_marketing_or_noreply("starbucks@starbucks.com", {})

    def test_filters_hyphenated_self_mailers(self, extractor):
        """Normalize hyphens when comparing local-part and domain."""
        assert extractor._is_marketing_or_noreply("briggsriley@briggs-riley.com", {})
        assert extractor._is_marketing_or_noreply("tommy-john@tommyjohn.com", {})

    def test_allows_short_self_mailers(self, extractor):
        """Short domains (≤3 chars) could be personal, don't filter."""
        # Personal domains like me@me.com, jo@jo.com should not be filtered
        assert not extractor._is_marketing_or_noreply("me@me.com", {})
        assert not extractor._is_marketing_or_noreply("jo@jo.io", {})

    def test_allows_non_matching_addresses(self, extractor):
        """Normal addresses (local ≠ domain) should be allowed."""
        assert not extractor._is_marketing_or_noreply("john@company.com", {})
        assert not extractor._is_marketing_or_noreply("sarah@startup.io", {})


class TestMarketingLocalpartPatterns:
    """Test new local-part patterns added in this iteration."""

    def test_filters_team_addresses(self, extractor):
        """Team@ addresses should be filtered (usually automated)."""
        assert extractor._is_marketing_or_noreply("team@kickstargogo.com", {})
        assert extractor._is_marketing_or_noreply("team@example.com", {})

    def test_filters_emails_addresses(self, extractor):
        """Emails@ (plural form) should be filtered."""
        assert extractor._is_marketing_or_noreply("emails@postable.com", {})

    def test_filters_flyers_addresses(self, extractor):
        """Flyers@ (marketing circulars) should be filtered."""
        assert extractor._is_marketing_or_noreply("flyers@webstaurantstore.com", {})
        assert extractor._is_marketing_or_noreply("flyer@example.com", {})

    def test_filters_events_addresses(self, extractor):
        """Events@ (event marketing) should be filtered."""
        assert extractor._is_marketing_or_noreply("events@seatgeek.com", {})
        assert extractor._is_marketing_or_noreply("event@example.com", {})

    def test_filters_ens_emergency_notification(self, extractor):
        """ENS@ (Emergency Notification System) should be filtered."""
        assert extractor._is_marketing_or_noreply("ens@ens.usgs.gov", {})

    def test_filters_ouch_marketing(self, extractor):
        """Ouch@ (marketing pattern seen in production) should be filtered."""
        assert extractor._is_marketing_or_noreply("ouch@mymedic.com", {})

    def test_filters_spices_product_marketing(self, extractor):
        """Spices@ (product-specific marketing) should be filtered."""
        assert extractor._is_marketing_or_noreply("spices@thespicehouse.com", {})

    def test_filters_rideshare_transactional(self, extractor):
        """Ride-share and delivery services should be filtered."""
        assert extractor._is_marketing_or_noreply("uber@uber.com", {})
        assert extractor._is_marketing_or_noreply("lyft@lyft.com", {})
        assert extractor._is_marketing_or_noreply("doordash@doordash.com", {})
        assert extractor._is_marketing_or_noreply("grubhub@grubhub.com", {})

    def test_filters_acerewards_loyalty(self, extractor):
        """AceRewards@ (loyalty program) should be filtered."""
        assert extractor._is_marketing_or_noreply("AceRewards@e1.acehardware.com", {})


class TestNewsletterPlatforms:
    """Test filtering of newsletter platform domains."""

    def test_filters_substack_newsletters(self, extractor):
        """Substack newsletters should be filtered (both apex and subdomain)."""
        assert extractor._is_marketing_or_noreply("bytebytego@substack.com", {})
        assert extractor._is_marketing_or_noreply("author@newsletter.substack.com", {})

    def test_filters_beehiiv_newsletters(self, extractor):
        """Beehiiv newsletter platform should be filtered."""
        assert extractor._is_marketing_or_noreply("author@beehiiv.com", {})
        assert extractor._is_marketing_or_noreply("news@list.beehiiv.com", {})

    def test_filters_ghost_newsletters(self, extractor):
        """Ghost newsletter platform should be filtered."""
        assert extractor._is_marketing_or_noreply("author@ghost.io", {})

    def test_filters_convertkit_newsletters(self, extractor):
        """ConvertKit creator marketing should be filtered."""
        assert extractor._is_marketing_or_noreply("creator@convertkit.com", {})

    def test_filters_buttondown_newsletters(self, extractor):
        """Buttondown newsletter platform should be filtered."""
        assert extractor._is_marketing_or_noreply("author@buttondown.email", {})


class TestMarketingDomainPatterns:
    """Test new domain patterns added in this iteration."""

    def test_filters_iemail_subdomain(self, extractor):
        """@iemail. (internal email/marketing) should be filtered."""
        assert extractor._is_marketing_or_noreply("partners@iemail.moneylion.com", {})

    def test_filters_e1_subdomain(self, extractor):
        """@e1. (e-commerce platform) should be filtered."""
        assert extractor._is_marketing_or_noreply("AceRewards@e1.acehardware.com", {})

    def test_filters_connect_subdomain(self, extractor):
        """@connect. (notification platforms) should be filtered."""
        assert extractor._is_marketing_or_noreply("no.reply@connect.razer.com", {})

    def test_filters_webstaurant_pattern(self, extractor):
        """@webstaurant (specific known marketing domain) should be filtered."""
        assert extractor._is_marketing_or_noreply("flyers@webstaurantstore.com", {})


class TestLegitimateContacts:
    """Ensure legitimate human contacts are NOT filtered."""

    def test_allows_normal_email_addresses(self, extractor):
        """Standard email addresses should be allowed."""
        assert not extractor._is_marketing_or_noreply("john.smith@company.com", {})
        assert not extractor._is_marketing_or_noreply("sarah@startup.io", {})
        assert not extractor._is_marketing_or_noreply("mike.jones@example.org", {})

    def test_allows_personal_domains(self, extractor):
        """Personal email domains should be allowed (when not using marketing patterns)."""
        # Note: hello@ is filtered even on personal domains because it's overwhelmingly
        # used for marketing (e.g., hello@company.com). This is an acceptable trade-off
        # to prevent hundreds of marketing emails from polluting the relationships profile.
        assert not extractor._is_marketing_or_noreply("contact@johndoe.com", {})
        assert not extractor._is_marketing_or_noreply("me@sarahsmith.io", {})

    def test_allows_gmail_hotmail_protonmail(self, extractor):
        """Common personal email providers should be allowed."""
        assert not extractor._is_marketing_or_noreply("john@gmail.com", {})
        assert not extractor._is_marketing_or_noreply("sarah@hotmail.com", {})
        assert not extractor._is_marketing_or_noreply("mike@protonmail.com", {})


class TestRealWorldProductionData:
    """Test against actual addresses found in production that were polluting the profile."""

    def test_filters_top_20_production_marketing_addresses(self, extractor):
        """All top-20 marketing addresses from production should be filtered."""
        production_marketing = [
            "ens@ens.usgs.gov",
            "cutleryandmore@cutleryandmore.com",
            "sales@atlanticfirearms.com",
            "bytebytego@substack.com",
            "no.reply@connect.razer.com",
            "ouch@mymedic.com",
            "emails@postable.com",
            "events@seatgeek.com",
            "briggsriley@briggs-riley.com",
            "AceRewards@e1.acehardware.com",
            "concierge@tommyjohn.com",
            "lenovo@ecomm.lenovo.com",
            "team@kickstargogo.com",
            "partners@iemail.moneylion.com",
            "spices@thespicehouse.com",
            "flyers@webstaurantstore.com",
            "uber@uber.com",
        ]

        for addr in production_marketing:
            assert extractor._is_marketing_or_noreply(addr, {}), \
                f"Expected {addr} to be filtered as marketing"

    def test_filter_coverage_improvement(self, extractor):
        """Verify significant improvement in filter coverage."""
        # Before this iteration: only 6/20 top addresses were filtered
        # After: 17/20 are filtered (85% coverage)
        top_20 = [
            "ens@ens.usgs.gov",
            "cutleryandmore@cutleryandmore.com",
            "sales@atlanticfirearms.com",
            "dan@techiwant.com",  # Edge case, allowed
            "bytebytego@substack.com",
            "no.reply@connect.razer.com",
            "ouch@mymedic.com",
            "emails@postable.com",
            "events@seatgeek.com",
            "briggsriley@briggs-riley.com",
            "AceRewards@e1.acehardware.com",
            "concierge@tommyjohn.com",
            "lenovo@ecomm.lenovo.com",
            "tascio@pizzatascio.com",  # Edge case, allowed
            "team@kickstargogo.com",
            "partners@iemail.moneylion.com",
            "spices@thespicehouse.com",
            "eriksbikeboardski@eriksbikeshop.com",  # Edge case, allowed
            "flyers@webstaurantstore.com",
            "uber@uber.com",
        ]

        filtered_count = sum(1 for addr in top_20 if extractor._is_marketing_or_noreply(addr, {}))
        assert filtered_count >= 17, f"Expected ≥17 filtered, got {filtered_count}"
        assert filtered_count / len(top_20) >= 0.85, "Filter should catch ≥85% of marketing"
