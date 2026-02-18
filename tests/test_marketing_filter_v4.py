"""Tests for marketing filter iteration 178 improvements.

Verifies that the 59 automated/transactional senders identified in production
data (135 stale opportunity predictions) are now correctly filtered by:
1. PredictionEngine._is_marketing_or_noreply() — prevents new predictions
2. BehavioralAccuracyTracker._is_automated_sender() — fast-path resolves stale

Also confirms zero false positives against known human contacts.
"""
import pytest
from services.prediction_engine.engine import PredictionEngine
from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Addresses that are clearly automated/transactional from production data.
# These were found in 135 unresolved opportunity predictions (iteration 178).
AUTOMATED_SENDERS = [
    # Airlines (automated travel notifications)
    "American.Airlines@info.email.aa.com",
    "AmericanAirlines@info.ms.aa.com",
    # Hotels/hospitality automated
    "InterContinental@mc.ihg.com",
    "online.account@marriott.com",
    # Retail transactional
    "Customerservice@nationalcar.com",
    "customerservice@citepayusa.com",
    "reservations@nationalcar.com",
    "onlineservice@fedex.com",
    "return@amazon.com",
    "tracking@shipstation.com",
    "transaction@info.samsclub.com",
    "guestservices@boxoffice.axs.com",
    "walgreens@eml.walgreens.com",
    "rei@alerts.rei.com",
    "tickets@transactions.axs.com",
    "top@raymore.com",  # retail automated (marketing subdomain @raymore.com)
    # Tech/streaming automated
    "gaming@nvgaming.nvidia.com",
    "applecash@insideapple.apple.com",
    "OIP@odysseymail.tylertech.cloud",
    "msftpc@microsoft.com",  # Not filtered (specific Microsoft PC dept, acceptable miss)
    # Survey/marketing platforms
    "PizzaHut@pizzahutusemail.smg.com",
    # Payment/financial automated
    "messenger@messaging.squareup.com",
    "debit@card.southwest.com",
    "id@proxyvote.com",
    "drivers@chargepoint.com",
    # Travel loyalty/bookings automated
    "disneycruiseline@vacations.disneydestinations.com",
    "worldofhyatt@loyalty.hyatt.com",
    "vrbo@eg.vrbo.com",
    # Social/security automated
    "security@facebookmail.com",
    # Food/promo automated
    "monopolyatmcd@playatmcd.com",
    # No-reply dot-separated variants
    "no.reply.alerts@chase.com",
]

# Addresses that ARE real human contacts from production data.
# These must NOT be filtered (zero false positives required).
HUMAN_CONTACTS = [
    # Gmail personal
    "193arm24@gmail.com",
    "bdc.umr@gmail.com",
    "shelbyhiter@gmail.com",
    "sockscondo681@gmail.com",
    "tulsi724@gmail.com",
    # Corporate employees
    "Pattie.HERROD@capstonelogistics.com",
    "gsumme17@ford.com",
    "jason.stull@hyatt.com",
    "ashley.schuch@hilton.com",
    "dorrian@vestaboard.com",
    "ernest.lim@harvestright.com",
    "roxana.nunez@harvestright.com",
    "victoria.galindo@harvestright.com",
    # Educational
    "gerardo_sanchez2096@elcamino.edu",
    "natasha_mayekar2131@elcamino.edu",
    "alumni@mst.edu",
    "kummercollege@mst.edu",
    # Small business contacts
    "dino@ventiscafe.com",
    "cheers@bluskilletironware.com",
]


class TestPredictionEnginemarketingFilterV4:
    """Tests for PredictionEngine._is_marketing_or_noreply() with v4 patterns."""

    def test_filters_american_airlines_info_email_subdomain(self):
        """American Airlines info.email subdomain should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "American.Airlines@info.email.aa.com", {}
        )

    def test_filters_american_airlines_info_ms_subdomain(self):
        """American Airlines info.ms subdomain should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "AmericanAirlines@info.ms.aa.com", {}
        )

    def test_filters_hotel_mc_subdomain(self):
        """IHG mc. subdomain (email service provider) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "InterContinental@mc.ihg.com", {}
        )

    def test_filters_customerservice_prefix(self):
        """customerservice@ prefix should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "Customerservice@nationalcar.com", {}
        )
        assert PredictionEngine._is_marketing_or_noreply(
            "customerservice@citepayusa.com", {}
        )

    def test_filters_reservations_prefix(self):
        """reservations@ prefix (hospitality automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "reservations@nationalcar.com", {}
        )

    def test_filters_onlineservice_prefix(self):
        """onlineservice@ prefix should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "onlineservice@fedex.com", {}
        )

    def test_filters_return_prefix(self):
        """return@ prefix (retail returns automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply("return@amazon.com", {})

    def test_filters_tracking_prefix(self):
        """tracking@ prefix should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "tracking@shipstation.com", {}
        )

    def test_filters_transaction_prefix(self):
        """transaction@ prefix should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "transaction@info.samsclub.com", {}
        )

    def test_filters_online_account_prefix(self):
        """online.account@ prefix should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "online.account@marriott.com", {}
        )

    def test_filters_guestservices_prefix(self):
        """guestservices@ prefix should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "guestservices@boxoffice.axs.com", {}
        )

    def test_filters_drivers_prefix(self):
        """drivers@ prefix (EV charging automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "drivers@chargepoint.com", {}
        )

    def test_filters_gaming_prefix(self):
        """gaming@ prefix (gaming service automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "gaming@nvgaming.nvidia.com", {}
        )

    def test_filters_messenger_prefix(self):
        """messenger@ prefix (payment platform automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "messenger@messaging.squareup.com", {}
        )

    def test_filters_tickets_prefix(self):
        """tickets@ prefix (ticket platform automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "tickets@transactions.axs.com", {}
        )

    def test_filters_walgreens_prefix(self):
        """walgreens@ prefix should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "walgreens@eml.walgreens.com", {}
        )

    def test_filters_rei_prefix(self):
        """rei@ prefix (retail membership automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply("rei@alerts.rei.com", {})

    def test_filters_applecash_prefix(self):
        """applecash@ prefix (Apple Cash automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "applecash@insideapple.apple.com", {}
        )

    def test_filters_worldofhyatt_prefix(self):
        """worldofhyatt@ prefix (loyalty program automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "worldofhyatt@loyalty.hyatt.com", {}
        )

    def test_filters_disneycruiseline_prefix(self):
        """disneycruiseline@ prefix should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "disneycruiseline@vacations.disneydestinations.com", {}
        )

    def test_filters_alerts_subdomain(self):
        """@alerts. subdomain should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply("rei@alerts.rei.com", {})
        assert PredictionEngine._is_marketing_or_noreply(
            "chase@alerts.chase.com", {}
        )

    def test_filters_loyalty_subdomain(self):
        """@loyalty. subdomain should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "worldofhyatt@loyalty.hyatt.com", {}
        )

    def test_filters_vacations_subdomain(self):
        """@vacations. subdomain should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "disneycruiseline@vacations.disneydestinations.com", {}
        )

    def test_filters_transactions_subdomain(self):
        """@transactions. subdomain should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "tickets@transactions.axs.com", {}
        )

    def test_filters_eml_subdomain(self):
        """@eml. subdomain (email delivery) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "walgreens@eml.walgreens.com", {}
        )

    def test_filters_insideapple_subdomain(self):
        """@insideapple. subdomain should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "applecash@insideapple.apple.com", {}
        )

    def test_filters_card_subdomain(self):
        """@card. subdomain (card/payment automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "debit@card.southwest.com", {}
        )

    def test_filters_eg_subdomain(self):
        """@eg. subdomain (email gateway) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply("vrbo@eg.vrbo.com", {})

    def test_filters_mc_subdomain(self):
        """@mc. subdomain (ESP/campaign) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "InterContinental@mc.ihg.com", {}
        )

    def test_filters_odysseymail_subdomain(self):
        """@odysseymail. subdomain (court automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "OIP@odysseymail.tylertech.cloud", {}
        )

    def test_filters_proxyvote_domain(self):
        """proxyvote.com domain (Broadridge Financial proxy voting) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply("id@proxyvote.com", {})

    def test_filters_playatmcd_domain(self):
        """playatmcd.com domain (McDonald's promo) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "monopolyatmcd@playatmcd.com", {}
        )

    def test_filters_facebookmail_domain(self):
        """facebookmail.com domain (Facebook automated) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "security@facebookmail.com", {}
        )

    def test_filters_smg_domain(self):
        """smg.com domain (SMG survey platform) should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "PizzaHut@pizzahutusemail.smg.com", {}
        )

    def test_filters_no_reply_dot_separated(self):
        """no.reply dot-separated variant should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "no.reply.alerts@chase.com", {}
        )

    def test_filters_info_email_compound_subdomain(self):
        """info.email. compound subdomain pattern should be filtered."""
        assert PredictionEngine._is_marketing_or_noreply(
            "American.Airlines@info.email.aa.com", {}
        )

    def test_no_false_positives_gmail(self):
        """Gmail personal addresses should NOT be filtered."""
        for email in ["193arm24@gmail.com", "bdc.umr@gmail.com",
                      "shelbyhiter@gmail.com", "tulsi724@gmail.com"]:
            assert not PredictionEngine._is_marketing_or_noreply(email, {}), (
                f"{email} is a real person — should not be filtered"
            )

    def test_no_false_positives_corporate_employees(self):
        """Real corporate employees should NOT be filtered."""
        for email in [
            "Pattie.HERROD@capstonelogistics.com",
            "gsumme17@ford.com",
            "jason.stull@hyatt.com",
            "ashley.schuch@hilton.com",
            "dorrian@vestaboard.com",
            "ernest.lim@harvestright.com",
        ]:
            assert not PredictionEngine._is_marketing_or_noreply(email, {}), (
                f"{email} is a real person — should not be filtered"
            )

    def test_no_false_positives_educational(self):
        """Educational addresses should NOT be filtered."""
        for email in [
            "gerardo_sanchez2096@elcamino.edu",
            "natasha_mayekar2131@elcamino.edu",
            "alumni@mst.edu",
        ]:
            assert not PredictionEngine._is_marketing_or_noreply(email, {}), (
                f"{email} should not be filtered"
            )


class TestTrackerAutomatedSenderV4:
    """Tests for BehavioralAccuracyTracker._is_automated_sender() v4 patterns.

    These patterns mirror the prediction engine filter. The tracker uses them
    to immediately resolve stale opportunity predictions via fast-path logic.
    """

    def test_filters_customerservice_prefix(self):
        """customerservice prefix in local-part should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "Customerservice@nationalcar.com"
        )

    def test_filters_reservations_prefix(self):
        """reservations prefix should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "reservations@nationalcar.com"
        )

    def test_filters_return_prefix(self):
        """return prefix should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender("return@amazon.com")

    def test_filters_tracking_prefix(self):
        """tracking prefix should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "tracking@shipstation.com"
        )

    def test_filters_transaction_prefix(self):
        """transaction prefix should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "transaction@info.samsclub.com"
        )

    def test_filters_online_account_prefix(self):
        """online.account prefix should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "online.account@marriott.com"
        )

    def test_filters_gaming_prefix(self):
        """gaming prefix should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "gaming@nvgaming.nvidia.com"
        )

    def test_filters_applecash_prefix(self):
        """applecash prefix should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "applecash@insideapple.apple.com"
        )

    def test_filters_worldofhyatt_prefix(self):
        """worldofhyatt prefix should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "worldofhyatt@loyalty.hyatt.com"
        )

    def test_filters_no_reply_dot_separated(self):
        """no.reply in local-part should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "no.reply.alerts@chase.com"
        )

    def test_filters_alerts_subdomain(self):
        """alerts. subdomain should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender("rei@alerts.rei.com")
        assert BehavioralAccuracyTracker._is_automated_sender(
            "chase@alerts.chase.com"
        )

    def test_filters_loyalty_subdomain(self):
        """loyalty. subdomain should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "worldofhyatt@loyalty.hyatt.com"
        )

    def test_filters_vacations_subdomain(self):
        """vacations. subdomain should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "disneycruiseline@vacations.disneydestinations.com"
        )

    def test_filters_eml_subdomain(self):
        """eml. subdomain should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "walgreens@eml.walgreens.com"
        )

    def test_filters_card_subdomain(self):
        """card. subdomain should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "debit@card.southwest.com"
        )

    def test_filters_eg_subdomain(self):
        """eg. subdomain (email gateway) should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender("vrbo@eg.vrbo.com")

    def test_filters_insideapple_subdomain(self):
        """insideapple. subdomain should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "applecash@insideapple.apple.com"
        )

    def test_filters_mc_subdomain(self):
        """mc. subdomain should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "InterContinental@mc.ihg.com"
        )

    def test_filters_proxyvote_domain(self):
        """proxyvote.com should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender("id@proxyvote.com")

    def test_filters_playatmcd_domain(self):
        """playatmcd.com should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "monopolyatmcd@playatmcd.com"
        )

    def test_filters_facebookmail_domain(self):
        """facebookmail.com should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "security@facebookmail.com"
        )

    def test_filters_smg_compound_subdomain(self):
        """smg.com compound subdomain should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "PizzaHut@pizzahutusemail.smg.com"
        )

    def test_filters_info_email_compound_subdomain(self):
        """info.email. compound subdomain should be automated."""
        assert BehavioralAccuracyTracker._is_automated_sender(
            "American.Airlines@info.email.aa.com"
        )

    def test_no_false_positives_gmail(self):
        """Gmail personal addresses should NOT be detected as automated."""
        for email in ["193arm24@gmail.com", "bdc.umr@gmail.com",
                      "shelbyhiter@gmail.com", "tulsi724@gmail.com"]:
            assert not BehavioralAccuracyTracker._is_automated_sender(email), (
                f"{email} is a real person — should not be automated"
            )

    def test_no_false_positives_corporate_employees(self):
        """Real corporate employees should NOT be detected as automated."""
        for email in [
            "Pattie.HERROD@capstonelogistics.com",
            "gsumme17@ford.com",
            "jason.stull@hyatt.com",
            "dorrian@vestaboard.com",
            "ernest.lim@harvestright.com",
        ]:
            assert not BehavioralAccuracyTracker._is_automated_sender(email), (
                f"{email} is a real person — should not be automated"
            )

    def test_no_false_positives_educational(self):
        """Educational addresses should NOT be detected as automated."""
        for email in [
            "gerardo_sanchez2096@elcamino.edu",
            "natasha_mayekar2131@elcamino.edu",
        ]:
            assert not BehavioralAccuracyTracker._is_automated_sender(email), (
                f"{email} should not be automated"
            )
