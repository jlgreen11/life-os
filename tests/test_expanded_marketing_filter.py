"""
Test suite for expanded marketing email filter.

Validates that the prediction engine correctly filters out marketing,
transactional, and bulk notification emails to prevent low-quality
prediction spam.

The Problem:
    Before this fix, emails from service@paypal.com, discover@airbnb.com,
    HOA-Notifications@hoatotalaccess.net, and similar bulk senders were
    generating reminder predictions. This created:
    - 323,144 unsurfaced predictions in 48 hours
    - Pollution of the accuracy feedback loop
    - Database bloat from filtered-but-stored predictions

The Fix:
    Enhanced _is_marketing_or_noreply() to catch:
    - service@ and discover@ sender patterns
    - Embedded notification patterns (-notification@, -alerts@, etc.)
    - Engagement platform domains (@engage., @iluv., @e., @e2.)
"""

import pytest
from services.prediction_engine.engine import PredictionEngine


class TestExpandedMarketingFilter:
    """Test the expanded marketing filter catches all common bulk email patterns."""

    def test_service_senders_filtered(self):
        """Service@ emails from payment processors should be filtered.

        These are transactional emails (payment notifications, receipts) that
        are legitimate but don't require follow-up responses. Filtering them
        prevents reminder spam for routine transactions.
        """
        test_cases = [
            "service@paypal.com",
            "service@stripe.com",
            "service@square.com",
        ]
        for email in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(email, {}), \
                f"{email} should be filtered as service@ sender"

    def test_discover_senders_filtered(self):
        """Discover@ emails are common marketing pattern and should be filtered.

        Used by Airbnb, Spotify, and other platforms for promotional content
        disguised as service emails.
        """
        test_cases = [
            "discover@airbnb.com",
            "discover@spotify.com",
            "discover@lyft.com",
        ]
        for email in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(email, {}), \
                f"{email} should be filtered as discover@ sender"

    def test_alert_notification_senders_filtered(self):
        """Alert@ and notification@ patterns should be filtered.

        These are system-generated alerts, not personal communications.
        """
        test_cases = [
            "alert@security.com",
            "alerts@system.net",
            "notification@platform.io",
            "notifications@app.com",
        ]
        for email in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(email, {}), \
                f"{email} should be filtered as alert/notification sender"

    def test_embedded_notification_patterns_filtered(self):
        """Emails with -notification or -alerts in local-part should be filtered.

        Catches patterns like:
        - HOA-Notifications@hoatotalaccess.net
        - system-alerts@company.com
        - user-notifications@platform.io
        """
        test_cases = [
            "HOA-Notifications@hoatotalaccess.net",
            "security-alerts@bank.com",
            "system-notifications@platform.io",
            "user-updates@service.net",
            "billing-digest@company.com",
        ]
        for email in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(email, {}), \
                f"{email} should be filtered due to embedded notification pattern"

    def test_engagement_platform_domains_filtered(self):
        """Engagement platform domains should be filtered.

        These are email marketing services used by brands:
        - @engage.ticketmaster.com
        - @iluv.southwest.com
        - @e.bathandbodyworks.com
        - @e2.retailer.com
        """
        test_cases = [
            "mammothlive@engage.ticketmaster.com",
            "SouthwestAirlines@iluv.southwest.com",
            "bathandbodyworks@e2.bathandbodyworks.com",
            "support@e.usa.experian.com",
        ]
        for email in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(email, {}), \
                f"{email} should be filtered due to engagement platform domain"

    def test_existing_filters_still_work(self):
        """Verify that existing filter patterns still work after expansion.

        Regression test to ensure we didn't break existing functionality.
        """
        # No-reply patterns
        assert PredictionEngine._is_marketing_or_noreply("noreply@company.com", {})
        assert PredictionEngine._is_marketing_or_noreply("mailer-daemon@googlemail.com", {})

        # Bulk sender patterns
        assert PredictionEngine._is_marketing_or_noreply("newsletter@brand.com", {})
        assert PredictionEngine._is_marketing_or_noreply("marketing@store.net", {})

        # Marketing domain patterns
        assert PredictionEngine._is_marketing_or_noreply("D23@email.d23.com", {})
        assert PredictionEngine._is_marketing_or_noreply("RoyalCaribbean@reply.royalcaribbeanmarketing.com", {})

        # Email service provider patterns
        assert PredictionEngine._is_marketing_or_noreply("sprouts@em.sprouts.com", {})

    def test_legitimate_emails_not_filtered(self):
        """Legitimate personal emails should NOT be filtered.

        Critical negative test: ensure we don't over-filter and miss real
        communications that need follow-up.
        """
        legitimate_emails = [
            "alice@company.com",
            "bob.smith@startup.io",
            "sarah.jones@university.edu",
            "Jeremy.Greenwood@rsmus.com",  # Real person from test data
            "john+work@gmail.com",
            "contact@smallbusiness.net",
        ]
        for email in legitimate_emails:
            assert not PredictionEngine._is_marketing_or_noreply(email, {}), \
                f"{email} should NOT be filtered - it's a legitimate personal email"

    def test_edge_case_similar_but_not_matching(self):
        """Emails with similar but non-matching patterns should NOT be filtered.

        Edge cases to prevent false positives:
        - john.email@company.com (has 'email' but not email@)
        - sarah.reply@startup.io (has 'reply' but not reply@)
        - service.desk@company.net (has 'service' but not service@)
        """
        edge_cases = [
            "john.email@company.com",
            "sarah.reply@startup.io",
            "service.desk@company.net",  # Different from service@
            "discover.team@startup.com",  # Different from discover@
        ]
        for email in edge_cases:
            assert not PredictionEngine._is_marketing_or_noreply(email, {}), \
                f"{email} should NOT be filtered - pattern doesn't match exactly"

    def test_unsubscribe_in_body_plain(self):
        """Emails with unsubscribe in body_plain should be filtered."""
        payload = {
            "body_plain": "Click here to unsubscribe from future emails.",
            "snippet": "Welcome to our newsletter",
        }
        assert PredictionEngine._is_marketing_or_noreply("newsletter@brand.com", payload)

    def test_unsubscribe_in_snippet(self):
        """Emails with unsubscribe in snippet should be filtered."""
        payload = {
            "snippet": "... to unsubscribe click here ...",
            "body_plain": "Some content",
        }
        assert PredictionEngine._is_marketing_or_noreply("updates@service.com", payload)

    def test_unsubscribe_in_body_html(self):
        """Emails with unsubscribe in body HTML should be filtered."""
        payload = {
            "body": "<p>To unsubscribe, <a href='...'>click here</a></p>",
            "snippet": "Some preview",
        }
        assert PredictionEngine._is_marketing_or_noreply("promo@store.net", payload)

    def test_case_insensitive_matching(self):
        """Filter should be case-insensitive for robustness."""
        test_cases = [
            "Service@PayPal.com",
            "DISCOVER@Airbnb.com",
            "HOA-NOTIFICATIONS@HOASystem.net",
            "NoReply@Company.Com",
        ]
        for email in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(email, {}), \
                f"{email} should be filtered (case-insensitive)"

    def test_payload_without_text_fields(self):
        """Filter should handle payloads missing body_plain, snippet, or body."""
        # Empty payload
        assert PredictionEngine._is_marketing_or_noreply("service@paypal.com", {})

        # Payload with None values
        assert PredictionEngine._is_marketing_or_noreply("discover@airbnb.com", {
            "body_plain": None,
            "snippet": None,
            "body": None,
        })

        # Payload with empty strings
        assert PredictionEngine._is_marketing_or_noreply("notifications@app.io", {
            "body_plain": "",
            "snippet": "",
            "body": "",
        })

    def test_real_world_spam_examples(self):
        """Test against real emails from the production database.

        These are actual emails that were creating prediction spam:
        - service@paypal.com: Payment notifications
        - discover@airbnb.com: Terms of service updates
        - HOA-Notifications@hoatotalaccess.net: Community updates
        - RoyalCaribbean@reply.*: Travel promotions

        Note: Some emails like cutleryandmore@cutleryandmore.com and
        ens@ens.usgs.gov are caught by the unsubscribe filter when full
        payload is provided, but this test uses empty payloads to validate
        pattern-based filtering only.
        """
        spam_examples = [
            "service@paypal.com",
            "discover@airbnb.com",
            "HOA-Notifications@hoatotalaccess.net",
            "RoyalCaribbean@reply.royalcaribbeanmarketing.com",
            "D23@email.d23.com",
            "mailer-daemon@googlemail.com",
            "email@dealwiki.net",
            "sprouts@em.sprouts.com",
            "mammothlive@engage.ticketmaster.com",
            "SouthwestAirlines@iluv.southwest.com",
            "bathandbodyworks@e2.bathandbodyworks.com",
            "support@e.usa.experian.com",
        ]
        for email in spam_examples:
            assert PredictionEngine._is_marketing_or_noreply(email, {}), \
                f"{email} from production database should be filtered"

    def test_unsubscribe_catches_remaining_marketing(self):
        """Emails not caught by pattern matching should have unsubscribe links.

        Some marketing emails don't match standard patterns but are caught
        by checking for unsubscribe links in the email body:
        - cutleryandmore@cutleryandmore.com (store promotions)
        - ens@ens.usgs.gov (USGS earthquake alerts - automated service)
        """
        test_cases = [
            ("cutleryandmore@cutleryandmore.com", {
                "body_plain": "Sale on knives! Click here to unsubscribe."
            }),
            ("ens@ens.usgs.gov", {
                "snippet": "Earthquake alert. To unsubscribe from these alerts..."
            }),
        ]
        for email, payload in test_cases:
            assert PredictionEngine._is_marketing_or_noreply(email, payload), \
                f"{email} should be filtered via unsubscribe detection"

    def test_filter_prevents_prediction_creation(self, db, user_model_store):
        """Integration test: verify filter prevents predictions from being created.

        This test simulates the full prediction flow to ensure that filtered
        emails never result in reminder predictions being stored.
        """
        engine = PredictionEngine(db, user_model_store)

        # Insert test events: mix of legitimate and marketing emails
        test_emails = [
            # Marketing emails that should be filtered
            ("service@paypal.com", "Payment received"),
            ("discover@airbnb.com", "New listing in your area"),
            ("HOA-Notifications@hoa.net", "Pool closed for maintenance"),
            # Legitimate email that should generate prediction
            ("alice@company.com", "Can you review the Q4 report?"),
        ]

        import json
        from datetime import datetime, timezone, timedelta

        with db.get_connection("events") as conn:
            for from_addr, subject in test_emails:
                timestamp = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"test-{from_addr}",
                        "email.received",
                        "test",
                        timestamp,
                        1,
                        json.dumps({
                            "message_id": f"<{from_addr}>",
                            "from_address": from_addr,
                            "subject": subject,
                            "body_plain": "Test email body",
                        }),
                        json.dumps({"related_contacts": []}),
                    ),
                )

        # Run prediction engine
        import asyncio
        predictions = asyncio.run(engine.generate_predictions({}))

        # Only the legitimate email should generate a prediction
        # Marketing emails should be filtered out before prediction creation
        with db.get_connection("user_model") as conn:
            reminder_predictions = conn.execute(
                """SELECT supporting_signals FROM predictions
                   WHERE prediction_type = 'reminder'
                   AND created_at > datetime('now', '-1 minute')"""
            ).fetchall()

        # Extract contact emails from predictions
        predicted_contacts = []
        for pred in reminder_predictions:
            signals = json.loads(pred["supporting_signals"]) if pred["supporting_signals"] else {}
            contact = signals.get("contact_email")
            if contact:
                predicted_contacts.append(contact)

        # Marketing emails should NOT have predictions
        assert "service@paypal.com" not in predicted_contacts
        assert "discover@airbnb.com" not in predicted_contacts
        assert "HOA-Notifications@hoa.net" not in predicted_contacts

        # Legitimate email SHOULD have a prediction (but may be filtered by reaction prediction)
        # So we just verify marketing emails are excluded
        assert len([c for c in predicted_contacts if "@paypal.com" in c or "@airbnb.com" in c]) == 0
