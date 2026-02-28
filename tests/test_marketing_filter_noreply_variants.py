"""Tests for marketing filter no-reply variant detection and new automated senders.

Iteration 191 extended the shared marketing filter in two ways:

1. **No-reply stem detection**: the previous implementation only caught exact
   prefix patterns like ``noreply@``, ``no-reply@``, ``donotreply@``.  Senders
   that append a suffix to the noreply stem (e.g. ``no-reply-ugc@samsclub.com``,
   ``DoNotReply_US@us.mcdonalds.com``) were missed, causing 74 opportunity
   predictions to be marked inaccurate instead of being resolved as
   ``automated_sender_fast_path``.

   The fix adds a local-part STEM check: extract the local part and test whether
   it starts with any standard noreply stem before the first non-stem character.

2. **New automated-sender patterns**: production data analysis revealed several
   categories of automated senders that weren't detected:
   - ``alerts+HASH@`` — tagged notification senders (PrismIntelligence, etc.)
   - ``stay@``, ``claims@``, ``irrigation@`` — hospitality/government/service
   - ``top@``, ``alumni@``, ``msftpc@`` — retail, educational, corporate automated
   - ``accountservices@`` — plural variant of existing ``accountservice@``
   - ``@customer.``, ``@rewards.`` — CRM/loyalty subdomain patterns
   - ``e-vanguard.com`` — email marketing platform domain
   - ``directpay.irs.gov`` — government payment system domain
"""
import pytest

from services.signal_extractor.marketing_filter import is_marketing_or_noreply


# ---------------------------------------------------------------------------
# 1. No-reply stem variants (the main fix in this iteration)
# ---------------------------------------------------------------------------

class TestNoreplyStems:
    """No-reply senders that use a suffix after the noreply stem."""

    def test_donotreply_with_country_suffix(self):
        """DoNotReply_US@ is a noreply stem variant with _US appended."""
        assert is_marketing_or_noreply("DoNotReply_US@us.mcdonalds.com") is True

    def test_no_reply_with_campaign_suffix(self):
        """no-reply-ugc@ has 'ugc' (user-generated-content) appended to no-reply."""
        assert is_marketing_or_noreply("no-reply-ugc@samsclub.com") is True

    def test_no_reply_with_system_suffix(self):
        """no_reply_msg@ has '_msg' appended to no_reply."""
        assert is_marketing_or_noreply("no_reply_msg@hcahealthcare.com") is True

    def test_noreply_with_service_suffix(self):
        """noreply-service@ has '-service' appended to noreply."""
        assert is_marketing_or_noreply("noreply-service@anker.com") is True

    def test_noreply_with_numeric_suffix(self):
        """noreply123@ — numeric suffix still clearly automated."""
        assert is_marketing_or_noreply("noreply123@example.com") is True

    def test_donotreply_with_numeric_suffix(self):
        """donotreply1@ still caught by stem detection."""
        assert is_marketing_or_noreply("donotreply1@example.com") is True

    def test_no_reply_hyphen_variant_with_id(self):
        """noreply-mfa1@ (multi-factor auth notification) caught by stem."""
        assert is_marketing_or_noreply("noreply-mfa1@aa.com") is True

    # Existing exact-match variants should still work (regression check)
    def test_noreply_exact(self):
        """Exact noreply@ still caught."""
        assert is_marketing_or_noreply("noreply@example.com") is True

    def test_no_reply_exact(self):
        """Exact no-reply@ still caught."""
        assert is_marketing_or_noreply("no-reply@example.com") is True

    def test_donotreply_exact(self):
        """Exact donotreply@ still caught."""
        assert is_marketing_or_noreply("donotreply@example.com") is True


# ---------------------------------------------------------------------------
# 2. alerts+HASH@ tagged notification senders
# ---------------------------------------------------------------------------

class TestAlertsTaggedNotifications:
    """PrismIntelligence-style alerts+hash@ notification senders."""

    def test_alerts_plus_hash_short(self):
        """Short hash variant."""
        assert is_marketing_or_noreply("alerts+Xe6Cu7@prismintelligence.com") is True

    def test_alerts_plus_hash_long(self):
        """Full-length hash variant observed in production data."""
        assert is_marketing_or_noreply(
            "alerts+Xe6Cu7Szh4Os3L1c@prismintelligence.com"
        ) is True

    def test_alerts_plus_hash_another_sender(self):
        """Same pattern from a different domain."""
        assert is_marketing_or_noreply("alerts+1I7jQ0KU2vC-k6EQ@prismintelligence.com") is True

    def test_alerts_at_exact_still_caught(self):
        """Existing exact alerts@ pattern still works (regression check)."""
        assert is_marketing_or_noreply("alerts@example.com") is True


# ---------------------------------------------------------------------------
# 3. Hospitality / government / service automated senders
# ---------------------------------------------------------------------------

class TestHospitalityGovernmentService:
    """stay@, claims@, irrigation@ — transactional automated senders."""

    def test_stay_hotel_booking(self):
        """Hotel booking confirmation system uses stay@ local part."""
        assert is_marketing_or_noreply("stay@hotelvandivort.com") is True

    def test_claims_government_payment(self):
        """Government payment/claims system — never a human correspondent."""
        assert is_marketing_or_noreply("claims@treasurer.mo.gov") is True

    def test_irrigation_service_alert(self):
        """Automated service-alert emails from a lawn/irrigation company."""
        assert is_marketing_or_noreply("irrigation@ryanlawn.com") is True


# ---------------------------------------------------------------------------
# 4. Retail / educational / corporate automated senders
# ---------------------------------------------------------------------------

class TestRetailEducationalCorporate:
    """top@, msftpc@ — brand promos and automated corporate systems."""

    def test_top_retail_promo(self):
        """Retail store 'top picks' promotional mailer."""
        assert is_marketing_or_noreply("top@raymore.com") is True

    def test_alumni_newsletter_filtered(self):
        """alumni@ is a university mailing-list prefix, never a personal address.

        alumni@mst.edu is a bulk alumni mailing list, not an individual's inbox.
        Real educational contacts have personal addresses like john.doe@university.edu
        (which do NOT start with 'alumni@').  The pattern was reinstated in
        iteration 179 after analysis of production data confirmed no false positives
        for human contacts.
        """
        assert is_marketing_or_noreply("alumni@mst.edu") is True
        assert is_marketing_or_noreply("alumni@harvard.edu") is True
        assert is_marketing_or_noreply("alumni@stanford.edu") is True

    def test_msftpc_microsoft_automated(self):
        """Microsoft PC automated system (proactive customer outreach system)."""
        assert is_marketing_or_noreply("msftpc@microsoft.com") is True


# ---------------------------------------------------------------------------
# 5. accountservices@ (plural variant)
# ---------------------------------------------------------------------------

class TestAccountServicesPlural:
    """accountservices@ is the plural variant of the existing accountservice@."""

    def test_accountservices_plural(self):
        """accountservices@ (plural) should be caught, matching accountservice@."""
        assert is_marketing_or_noreply("accountservices@ncl.com") is True

    def test_accountservice_singular_still_works(self):
        """Existing singular accountservice@ not broken (regression check)."""
        assert is_marketing_or_noreply("accountservice@example.com") is True


# ---------------------------------------------------------------------------
# 6. CRM/loyalty subdomain patterns
# ---------------------------------------------------------------------------

class TestSubdomainPatterns:
    """@customer. and @rewards. CRM/loyalty subdomain patterns."""

    def test_customer_subdomain(self):
        """Enterprise Holdings uses @customer.ehi.com for automated mailers."""
        assert is_marketing_or_noreply("national@customer.ehi.com") is True

    def test_rewards_subdomain(self):
        """Sprouts Farmers Market loyalty program uses @rewards.sprouts.com."""
        assert is_marketing_or_noreply("sproutsrewards@rewards.sprouts.com") is True

    def test_rewards_subdomain_different_brand(self):
        """Another hypothetical brand using @rewards. subdomain."""
        assert is_marketing_or_noreply("info@rewards.anystore.com") is True


# ---------------------------------------------------------------------------
# 7. Marketing service platform domains
# ---------------------------------------------------------------------------

class TestMarketingPlatformDomains:
    """e-vanguard.com and directpay.irs.gov specific domain blocks."""

    def test_e_vanguard_marketing_esp(self):
        """e-vanguard.com is an email marketing/CRM platform."""
        assert is_marketing_or_noreply("flagship@eonline.e-vanguard.com") is True

    def test_directpay_irs_gov(self):
        """IRS Direct Pay automated payment notification system."""
        assert is_marketing_or_noreply("mail@directpay.irs.gov") is True


# ---------------------------------------------------------------------------
# 8. False-positive guard: real human addresses must not be blocked
# ---------------------------------------------------------------------------

class TestNoFalsePositives:
    """Real human addresses that must pass through the filter unchanged."""

    def test_gmail_user(self):
        assert is_marketing_or_noreply("alice@gmail.com") is False

    def test_company_employee(self):
        assert is_marketing_or_noreply("bob@company.com") is False

    def test_protonmail_user(self):
        assert is_marketing_or_noreply("david@protonmail.com") is False

    def test_hotmail_user(self):
        assert is_marketing_or_noreply("sarah@hotmail.com") is False

    def test_startup_employee(self):
        assert is_marketing_or_noreply("jane@startup.io") is False

    def test_top_in_middle_of_name(self):
        """'top' appearing mid-word in a human address should not be blocked.
        The local part 'top.chef' does NOT start with 'top@', so it passes.
        """
        assert is_marketing_or_noreply("top.chef@example.com") is False

    def test_alumni_substring_in_name(self):
        """'alumni.member@...' — local part does not start with 'alumni@'."""
        assert is_marketing_or_noreply("alumni.member@company.com") is False

    def test_noreply_name_not_noreply(self):
        """'nora@...' starts with 'nor', not a noreply stem — must pass."""
        assert is_marketing_or_noreply("nora@example.com") is False

    def test_regular_name_with_noreply_in_domain(self):
        """Human sender at a company whose domain happens to have noreply.
        This is unlikely but guards against over-broad domain matching.
        The address alice@noreply-systems.com has 'noreply' in the DOMAIN,
        not the local part — should only match if it's a known marketing pattern.
        """
        # alice@ is a real name; the domain check only blocks specific patterns.
        # This address should pass because the domain is not a marketing platform.
        assert is_marketing_or_noreply("alice@noreply-systems.com") is False
