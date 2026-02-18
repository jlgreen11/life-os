"""
Tests for the shared marketing/automated-sender filter module.

Verifies that services.signal_extractor.marketing_filter.is_marketing_or_noreply()
correctly identifies automated senders and does not produce false positives for
real human email addresses.

This module is the canonical single source of truth for marketing detection —
previously the logic was duplicated (and had diverged) across:

  - services/signal_extractor/relationship.py
  - services/prediction_engine/engine.py
  - services/behavioral_accuracy_tracker/tracker.py

The most critical divergence was that the relationship extractor was missing
the financial/brokerage patterns (Fidelity, Schwab, PayPal, …) and the
retail/hospitality patterns (Customerservice, Reservations, WorldofHyatt, …)
that had been added to the prediction engine in iterations 171 and 178.
This allowed those automated senders to accumulate in the relationships profile
and generate opportunity predictions that could never be fulfilled — a root
cause of the 19% opportunity accuracy rate.
"""
import pytest

from services.signal_extractor.marketing_filter import is_marketing_or_noreply


class TestNoReplyPatterns:
    """Addresses that contain no-reply variants must be blocked."""

    @pytest.mark.parametrize("addr", [
        "noreply@example.com",
        "no-reply@company.com",
        "no_reply@company.com",
        "donotreply@service.com",
        "do-not-reply@service.com",
        "do_not_reply@service.com",
        "mailer-daemon@domain.com",
        "postmaster@domain.com",
        "daemon@domain.com",
        "auto-reply@service.com",
        "autoreply@service.com",
        "automated@system.com",
        "automation@system.com",
        # With + modifier (e.g., no-reply+123@domain.com)
        "no-reply+ABC123@company.com",
        "noreply+token@service.com",
    ])
    def test_noreply_blocked(self, addr):
        """No-reply and automated-system addresses must always return True."""
        assert is_marketing_or_noreply(addr) is True, (
            f"Expected {addr!r} to be filtered as automated, but it wasn't"
        )


class TestBulkSenderLocalParts:
    """Addresses starting with known bulk-sender local-parts must be blocked."""

    @pytest.mark.parametrize("addr", [
        # Core marketing patterns
        "newsletter@company.com",
        "notifications@service.com",
        "notification@service.com",
        "updates@platform.com",
        "marketing@brand.com",
        "reply@brand.com",
        "offers@store.com",
        "alert@bank.com",
        "alerts@bank.com",
        # Transactional
        "orders@shop.com",
        "receipts@store.com",
        "confirmation@retailer.com",
        "shipping@logistics.com",
        "delivery@courier.com",
        "account@provider.com",
        # Financial/brokerage (iteration 171 — previously missing from relationship extractor)
        "fidelity@mail.fidelity.com",
        "fidelity.investments@mail.fidelity.com",
        "schwab@schwab.com",
        "vanguard@vanguard.com",
        "etrade@etrade.com",
        "merrilledge@ml.com",
        "robinhood@robinhood.com",
        "betterment@betterment.com",
        "paypal@paypal.com",
        "venmo@venmo.com",
        "stripe@stripe.com",
        "coinbase@coinbase.com",
        "experian@experian.com",
        "creditkarma@creditkarma.com",
        # Retail/hospitality (iteration 178 — previously missing from relationship extractor)
        "customerservice@nationalcar.com",
        "reservations@nationalcar.com",
        "onlineservice@fedex.com",
        "return@amazon.com",
        "tracking@shipstation.com",
        "transaction@info.samsclub.com",
        "guestservices@boxoffice.axs.com",
        "gaming@nvgaming.nvidia.com",
        "tickets@transactions.axs.com",
        "walgreens@eml.walgreens.com",
        "rei@alerts.rei.com",
        "applecash@insideapple.apple.com",
        "worldofhyatt@loyalty.hyatt.com",
        "disneycruiseline@vacations.disneydestinations.com",
    ])
    def test_bulk_localpart_blocked(self, addr):
        """Addresses starting with known automated local-parts must return True."""
        assert is_marketing_or_noreply(addr) is True, (
            f"Expected {addr!r} to be filtered as bulk/automated, but it wasn't"
        )


class TestEmbeddedNotificationPatterns:
    """Addresses with embedded notification markers in the local-part."""

    @pytest.mark.parametrize("addr", [
        "HOA-Notifications@community.com",
        "user-notifications@service.com",
        "system-alerts@platform.com",
        "price-alert@shop.com",
        "lafconews@domain.com",
        "morningnews@digest.com",
        "no.reply.alerts@chase.com",
        "no.reply@domain.com",
        "do.not.reply@company.com",
        "weekly-digest@newsletter.com",
    ])
    def test_embedded_patterns_blocked(self, addr):
        """Addresses with embedded notification markers must return True."""
        assert is_marketing_or_noreply(addr) is True, (
            f"Expected {addr!r} to be filtered (embedded pattern), but it wasn't"
        )


class TestMarketingDomainSubdomains:
    """Addresses using marketing/transactional subdomain patterns."""

    @pytest.mark.parametrize("addr", [
        # Standard marketing subdomains
        "hello@email.company.com",
        "brand@news-us.company.com",
        "updates@newsletters.service.com",
        "info@comms.activision.com",
        "user@notification.capitalone.com",
        "brand@m.starbucks.com",
        # Retail/hospitality subdomains (iteration 178 — previously missing)
        "brand@alerts.rei.com",
        "account@loyalty.hyatt.com",
        "travel@vacations.disneydestinations.com",
        "receipt@transactions.axs.com",
        "rx@eml.walgreens.com",
        "brand@insideapple.apple.com",
        "payment@card.southwest.com",
        "case@odysseymail.tylertech.cloud",
        "hotel@mc.ihg.com",
        "listing@eg.vrbo.com",
        # American Airlines compound subdomain
        "info@info.email.aa.com",
    ])
    def test_marketing_subdomain_blocked(self, addr):
        """Addresses using marketing subdomains must return True."""
        assert is_marketing_or_noreply(addr) is True, (
            f"Expected {addr!r} to be filtered (marketing subdomain), but it wasn't"
        )


class TestThirdPartyEmailPlatforms:
    """Addresses sent through known third-party email marketing platforms."""

    @pytest.mark.parametrize("addr", [
        "brand@company.sendgrid.net",
        "news@company.mailchimp.com",
        "update@list.hubspot.com",
        "newsletter@company.klaviyo.com",
        "letter@publication.substack.com",
        "bytebytego@substack.com",          # apex domain match
        "author@newsletter.beehiiv.com",
        "vote@broadridge.proxyvote.com",    # proxy voting
        "promo@playatmcd.com",              # McDonald's automated
        "security@facebookmail.com",        # Facebook automated
        "survey@store.smg.com",
    ])
    def test_third_party_platform_blocked(self, addr):
        """Addresses on known ESP domains must return True."""
        assert is_marketing_or_noreply(addr) is True, (
            f"Expected {addr!r} to be filtered (ESP domain), but it wasn't"
        )


class TestBrandSelfMailer:
    """Brand self-mailer pattern: company@company.com."""

    @pytest.mark.parametrize("addr", [
        # These must be >= 4 chars in the local-part to trigger the length guard.
        # Short names like "gap" (3 chars) are exempt to avoid false positives
        # on addresses like gap@gap.com (could be a personal domain).
        "cutleryandmore@cutleryandmore.com",
        "target@target.com",
        "apple@apple.com",
    ])
    def test_brand_self_mailer_blocked(self, addr):
        """Brand self-mailer addresses (local == domain base, len > 3) must return True."""
        assert is_marketing_or_noreply(addr) is True, (
            f"Expected {addr!r} to be filtered (brand self-mailer), but it wasn't"
        )

    def test_short_local_not_filtered(self):
        """Short names (≤ 3 chars) are not filtered by the self-mailer rule to avoid
        false positives like me@me.com or gap@gap.com (plausible personal domains)."""
        # "gap" is 3 chars, so the length guard (> 3) prevents a false positive
        assert is_marketing_or_noreply("gap@gap.com") is False
        assert is_marketing_or_noreply("me@me.com") is False
        assert is_marketing_or_noreply("io@io.io") is False


class TestUnsubscribeLinkInPayload:
    """Payload containing 'unsubscribe' text must trigger the filter."""

    def test_unsubscribe_in_body_blocked(self):
        """An otherwise human-looking address is filtered if body has 'unsubscribe'."""
        payload = {"body": "Click here to unsubscribe from our mailing list."}
        assert is_marketing_or_noreply("sender@company.com", payload) is True

    def test_unsubscribe_in_snippet_blocked(self):
        """Unsubscribe in snippet also triggers the filter."""
        payload = {"snippet": "...to unsubscribe click here..."}
        assert is_marketing_or_noreply("sender@company.com", payload) is True

    def test_no_payload_skips_body_check(self):
        """Without a payload, a genuine human address passes the filter."""
        assert is_marketing_or_noreply("alice@gmail.com") is False
        assert is_marketing_or_noreply("alice@gmail.com", None) is False


class TestHumanAddressesNotBlocked:
    """Real human addresses must NOT be filtered — these are our false-positive tests."""

    @pytest.mark.parametrize("addr", [
        # Personal email providers
        "alice@gmail.com",
        "bob@hotmail.com",
        "carol@protonmail.com",
        "dave@yahoo.com",
        "eve@outlook.com",
        "frank@icloud.com",
        # Work/personal domains
        "john.doe@company.com",
        "jane@startup.io",
        "support.team@mycompany.com",   # Not the "support@" pattern (starts with "support@")
        "sarah@mycorp.org",
        "mike@university.edu",
        # Custom domains
        "contact@smallbiz.com",        # "contact@" is not in the filter (only "contactus@")
        # Edge cases that should NOT be filtered
        "me@me.com",                   # Self-mailer but length ≤ 3
    ])
    def test_human_address_not_blocked(self, addr):
        """Human email addresses must return False (not filtered)."""
        assert is_marketing_or_noreply(addr) is False, (
            f"False positive: {addr!r} was incorrectly filtered as automated"
        )


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string_not_blocked(self):
        """Empty address returns False without crashing."""
        assert is_marketing_or_noreply("") is False

    def test_none_raises_or_returns_false(self):
        """None address is handled gracefully."""
        # The function checks `if not from_addr` so None returns False
        assert is_marketing_or_noreply(None) is False  # type: ignore[arg-type]

    def test_case_insensitive(self):
        """Filter is case-insensitive."""
        assert is_marketing_or_noreply("NOREPLY@EXAMPLE.COM") is True
        assert is_marketing_or_noreply("Newsletter@Company.Com") is True
        assert is_marketing_or_noreply("Fidelity@Mail.Fidelity.Com") is True

    def test_payload_none_does_not_crash(self):
        """None payload is handled without AttributeError."""
        assert is_marketing_or_noreply("alice@gmail.com", None) is False

    def test_payload_empty_dict(self):
        """Empty payload dict works correctly."""
        assert is_marketing_or_noreply("alice@gmail.com", {}) is False
        assert is_marketing_or_noreply("noreply@example.com", {}) is True


class TestConsistencyAcrossCallSites:
    """
    Verify that all three services now produce identical results for addresses
    that previously diverged between the relationship extractor (missing patterns)
    and the prediction engine (had full patterns).

    These are the specific patterns that were MISSING from the relationship
    extractor before this fix, causing automated senders to accumulate in the
    relationships profile.
    """

    @pytest.mark.parametrize("addr", [
        # Financial — were present in engine, missing from relationship extractor
        "fidelity@mail.fidelity.com",
        "schwab@schwab.com",
        "chase.alerts@chase.com",
        "paypal@paypal.com",
        "coinbase@coinbase.com",
        "creditkarma@creditkarma.com",
        # Retail/hospitality — were present in engine, missing from relationship extractor
        "customerservice@nationalcar.com",
        "reservations@hotels.com",
        "worldofhyatt@loyalty.hyatt.com",
        "applecash@insideapple.apple.com",
        "disneycruiseline@vacations.disneydestinations.com",
        # Domain subdomains — were present in engine, missing from relationship extractor
        "account@alerts.rei.com",
        "hotel@loyalty.hyatt.com",
        "ticket@transactions.axs.com",
        "rx@eml.walgreens.com",
    ])
    def test_previously_diverged_patterns_now_consistent(self, addr):
        """
        These addresses were previously accepted by the relationship extractor
        but rejected by the prediction engine.  After this fix, both use the
        shared module and produce identical (True) results.
        """
        from services.signal_extractor.relationship import RelationshipExtractor
        from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker

        shared_result = is_marketing_or_noreply(addr)
        extractor_result = RelationshipExtractor._is_marketing_or_noreply(addr, {})
        tracker_result = BehavioralAccuracyTracker._is_automated_sender(addr)

        assert shared_result is True, (
            f"Shared filter should block {addr!r}"
        )
        assert extractor_result == shared_result, (
            f"RelationshipExtractor diverges from shared filter for {addr!r}"
        )
        assert tracker_result == shared_result, (
            f"BehavioralAccuracyTracker diverges from shared filter for {addr!r}"
        )
