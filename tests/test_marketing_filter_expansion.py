"""
Test suite for expanded marketing email filter patterns.

This test suite validates the enhanced _is_marketing_or_noreply() filter,
which now catches:
- Transactional/automated senders (orders@, auto-confirm@, shipment-tracking@)
- Organizational bulk senders (communications@, development@)
- Loyalty/rewards programs (rewards@, loyalty@)
- Marketing service provider subdomains (@*.e2ma.net, @*.sendgrid.net)

These patterns prevent ~1,975 low-quality predictions (99.7% of unsurfaced
predictions) from polluting the feedback loop.
"""

import pytest
from services.prediction_engine.engine import PredictionEngine


class TestTransactionalSenderFiltering:
    """Test filtering of transactional/automated email senders."""

    def test_orders_sender_filtered(self):
        """Orders@ emails should be filtered (Amazon, Starbucks, etc.)."""
        result = PredictionEngine._is_marketing_or_noreply(
            "orders@starbucks.com",
            {"body_plain": "Your order #12345 is ready for pickup."},
        )
        assert result is True

    def test_order_singular_filtered(self):
        """Order@ variant should also be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "order@example.com",
            {"body_plain": "Order confirmation"},
        )
        assert result is True

    def test_auto_confirm_filtered(self):
        """Auto-confirm@ emails should be filtered (Amazon auto-confirmations)."""
        result = PredictionEngine._is_marketing_or_noreply(
            "auto-confirm@amazon.com",
            {"body_plain": "Your order has been confirmed."},
        )
        assert result is True

    def test_autoconfirm_no_hyphen_filtered(self):
        """Autoconfirm@ variant without hyphen should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "autoconfirm@example.com",
            {"body_plain": "Confirmation"},
        )
        assert result is True

    def test_confirmation_filtered(self):
        """Confirmation@ emails should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "confirmation@hotel.com",
            {"body_plain": "Your reservation is confirmed."},
        )
        assert result is True

    def test_shipment_tracking_filtered(self):
        """Shipment-tracking@ emails should be filtered (Amazon tracking)."""
        result = PredictionEngine._is_marketing_or_noreply(
            "shipment-tracking@amazon.com",
            {"body_plain": "Your package is on the way."},
        )
        assert result is True

    def test_shipping_filtered(self):
        """Shipping@ emails should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "shipping@company.com",
            {"body_plain": "Shipping update"},
        )
        assert result is True

    def test_delivery_filtered(self):
        """Delivery@ emails should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "delivery@restaurant.com",
            {"body_plain": "Your delivery is arriving soon."},
        )
        assert result is True

    def test_receipts_filtered(self):
        """Receipts@ emails should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "receipts@stripe.com",
            {"body_plain": "Payment receipt"},
        )
        assert result is True

    def test_receipt_singular_filtered(self):
        """Receipt@ variant should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "receipt@paypal.com",
            {"body_plain": "Receipt for your payment"},
        )
        assert result is True

    def test_accountservice_filtered(self):
        """AccountService@ emails should be filtered (MLB, etc.)."""
        result = PredictionEngine._is_marketing_or_noreply(
            "AccountService@mlb.com",
            {"body_plain": "Account notification"},
        )
        assert result is True

    def test_account_service_hyphenated_filtered(self):
        """Account-service@ variant with hyphen should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "account-service@example.com",
            {"body_plain": "Service update"},
        )
        assert result is True


class TestOrganizationalBulkSenderFiltering:
    """Test filtering of organizational bulk email senders."""

    def test_communications_filtered(self):
        """Communications@ emails should be filtered (SFZC, etc.)."""
        result = PredictionEngine._is_marketing_or_noreply(
            "communications@sfzc.org",
            {"body_plain": "Monthly newsletter from the community."},
        )
        assert result is True

    def test_development_filtered(self):
        """Development@ emails should be filtered (fundraising departments)."""
        result = PredictionEngine._is_marketing_or_noreply(
            "development@sfzc.org",
            {"body_plain": "Please support our annual fund."},
        )
        assert result is True

    def test_fundraising_filtered(self):
        """Fundraising@ emails should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "fundraising@nonprofit.org",
            {"body_plain": "Help us reach our goal!"},
        )
        assert result is True


class TestLoyaltyProgramFiltering:
    """Test filtering of loyalty and rewards program emails."""

    def test_rewards_filtered(self):
        """Rewards@ emails should be filtered (Sprouts Rewards, etc.)."""
        result = PredictionEngine._is_marketing_or_noreply(
            "rewards@sprouts.com",
            {"body_plain": "You've earned 100 points!"},
        )
        assert result is True

    def test_rewards_compound_name_filtered(self):
        """Compound names like sproutsrewards@ should be filtered via unsubscribe."""
        # While sproutsrewards@ doesn't start with rewards@, these emails
        # typically contain unsubscribe links and are caught by that filter
        result = PredictionEngine._is_marketing_or_noreply(
            "sproutsrewards@rewards.sprouts.com",
            {"body_plain": "You've earned 100 points! Click here to unsubscribe."},
        )
        assert result is True

    def test_loyalty_filtered(self):
        """Loyalty@ emails should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "loyalty@airline.com",
            {"body_plain": "Loyalty program update"},
        )
        assert result is True

    def test_rewards_subdomain_filtered(self):
        """Emails from @rewards.* domains should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "info@rewards.example.com",
            {"body_plain": "Special offer"},
        )
        # This is caught by the local-part pattern (rewards@)
        # But also validates the pattern works with subdomains
        assert result is True


class TestMarketingServiceProviderFiltering:
    """Test filtering of third-party email marketing platform domains."""

    def test_e2ma_net_filtered(self):
        """Emails from @*.e2ma.net should be filtered (Emma marketing platform)."""
        result = PredictionEngine._is_marketing_or_noreply(
            "ToastedOakInfo.concordhotels.com@2sf.e2ma.net",
            {"body_plain": "Special hotel offer"},
        )
        assert result is True

    def test_sendgrid_filtered(self):
        """Emails from @*.sendgrid.net should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "newsletter@em1234.sendgrid.net",
            {"body_plain": "Newsletter content"},
        )
        assert result is True

    def test_mailchimp_filtered(self):
        """Emails from @*.mailchimp.com should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "info@list.mailchimp.com",
            {"body_plain": "Marketing email"},
        )
        assert result is True

    def test_constantcontact_filtered(self):
        """Emails from @*.constantcontact.com should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "newsletter@em.constantcontact.com",
            {"body_plain": "Newsletter"},
        )
        assert result is True

    def test_hubspot_filtered(self):
        """Emails from @*.hubspot.com should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "marketing@info.hubspot.com",
            {"body_plain": "Marketing content"},
        )
        assert result is True

    def test_marketo_filtered(self):
        """Emails from @*.marketo.com should be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "campaigns@mkto.marketo.com",
            {"body_plain": "Campaign email"},
        )
        assert result is True

    def test_pardot_filtered(self):
        """Emails from @*.pardot.com should be filtered (Salesforce Pardot)."""
        result = PredictionEngine._is_marketing_or_noreply(
            "email@go.pardot.com",
            {"body_plain": "Salesforce marketing email"},
        )
        assert result is True

    def test_eloqua_filtered(self):
        """Emails from @*.eloqua.com should be filtered (Oracle Eloqua)."""
        result = PredictionEngine._is_marketing_or_noreply(
            "campaigns@em.eloqua.com",
            {"body_plain": "Oracle marketing email"},
        )
        assert result is True


class TestLegitimateEmailsNotFiltered:
    """Verify that legitimate personal emails are NOT filtered by the new patterns."""

    def test_personal_email_not_filtered(self):
        """Personal emails should pass through."""
        result = PredictionEngine._is_marketing_or_noreply(
            "john.smith@company.com",
            {"body_plain": "Hey, can we schedule a meeting?"},
        )
        assert result is False

    def test_order_in_name_not_filtered(self):
        """Names containing 'order' should not be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "jordan.order@example.com",  # 'order' is in the middle of the local-part
            {"body_plain": "Personal message"},
        )
        # Only catches when 'order@' is at the START of the address
        assert result is False

    def test_reward_in_name_not_filtered(self):
        """Names containing 'reward' should not be filtered."""
        result = PredictionEngine._is_marketing_or_noreply(
            "sarah.reward@company.com",  # 'reward' is in the middle
            {"body_plain": "Personal email"},
        )
        assert result is False

    def test_legitimate_org_email_not_filtered(self):
        """Work emails from non-bulk senders should pass through."""
        result = PredictionEngine._is_marketing_or_noreply(
            "alice@engineering.example.com",
            {"body_plain": "Code review comments"},
        )
        assert result is False

    def test_legitimate_subdomain_not_filtered(self):
        """Emails from legitimate company subdomains should pass through."""
        result = PredictionEngine._is_marketing_or_noreply(
            "bob@sales.company.com",
            {"body_plain": "Follow-up from our conversation"},
        )
        assert result is False


class TestEdgeCases:
    """Test edge cases and boundary conditions for the filter."""

    def test_empty_from_address(self):
        """Empty from address should not crash."""
        result = PredictionEngine._is_marketing_or_noreply(
            "",
            {"body_plain": "Some content"},
        )
        assert result is False

    def test_malformed_email(self):
        """Malformed email addresses should be handled gracefully."""
        result = PredictionEngine._is_marketing_or_noreply(
            "notanemail",
            {"body_plain": "Content"},
        )
        assert result is False

    def test_multiple_at_signs(self):
        """Email with multiple @ signs (malformed) should be handled."""
        result = PredictionEngine._is_marketing_or_noreply(
            "user@@domain.com",
            {"body_plain": "Content"},
        )
        # Should still check patterns against the lowercase version
        assert result is False

    def test_case_insensitive_matching(self):
        """Filter should be case-insensitive."""
        result = PredictionEngine._is_marketing_or_noreply(
            "ORDERS@EXAMPLE.COM",
            {"body_plain": "Order confirmation"},
        )
        assert result is True

    def test_empty_payload(self):
        """Empty payload should not crash."""
        result = PredictionEngine._is_marketing_or_noreply(
            "test@example.com",
            {},
        )
        assert result is False

    def test_unsubscribe_still_works(self):
        """Unsubscribe detection should still work with new patterns."""
        result = PredictionEngine._is_marketing_or_noreply(
            "personal@example.com",  # Not caught by address patterns
            {"body_plain": "Click here to unsubscribe from our newsletter."},
        )
        # Should be caught by the unsubscribe content filter
        assert result is True
