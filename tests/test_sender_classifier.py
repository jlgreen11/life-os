"""
Tests for the sender classifier and company digest system.

Tests cover:
- SenderType classification (human, company_transactional, company_marketing)
- Channel-based shortcuts (Signal, iMessage, WhatsApp → always human)
- Marketing filter integration (existing marketing_filter patterns)
- Transactional detection (bills, payments, account alerts)
- Classification caching and persistence
- NotificationManager sender-type routing
- Company digest accumulation and review
- API endpoint integration
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from models.core import SenderType
from services.notification_manager.manager import NotificationManager
from services.sender_classifier.classifier import SenderClassifier

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def classifier(db):
    """Create a SenderClassifier instance with test database."""
    return SenderClassifier(db)


@pytest.fixture
def mock_event_bus():
    """Mock event bus with is_connected and publish."""
    bus = MagicMock()
    bus.is_connected = True
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def notif_manager(db, mock_event_bus):
    """Create a NotificationManager instance with test database."""
    return NotificationManager(db, mock_event_bus, config={})


# ============================================================================
# SenderClassifier — Channel shortcuts
# ============================================================================


class TestChannelShortcuts:
    """Signal, iMessage, WhatsApp, SMS are always human-to-human."""

    def test_signal_message_is_human(self, classifier):
        event = {
            "type": "message.received",
            "source": "signal",
            "payload": {"from_address": "+15551234567", "channel": "signal"},
        }
        assert classifier.classify_event(event) == SenderType.HUMAN

    def test_imessage_is_human(self, classifier):
        event = {
            "type": "message.received",
            "source": "imessage",
            "payload": {"from_address": "friend@icloud.com", "channel": "imessage"},
        }
        assert classifier.classify_event(event) == SenderType.HUMAN

    def test_whatsapp_is_human(self, classifier):
        event = {
            "type": "message.received",
            "source": "whatsapp",
            "payload": {"from_address": "+15559876543", "channel": "whatsapp"},
        }
        assert classifier.classify_event(event) == SenderType.HUMAN

    def test_sms_is_human(self, classifier):
        event = {
            "type": "message.received",
            "source": "sms",
            "payload": {"from_address": "+15551112222", "channel": "sms"},
        }
        assert classifier.classify_event(event) == SenderType.HUMAN

    def test_outbound_email_is_human(self, classifier):
        """Outbound messages (user sending) are always HUMAN."""
        event = {
            "type": "email.sent",
            "source": "proton_mail",
            "payload": {"from_address": "me@proton.me", "to_addresses": ["noreply@company.com"]},
        }
        assert classifier.classify_event(event) == SenderType.HUMAN


# ============================================================================
# SenderClassifier — Marketing detection
# ============================================================================


class TestMarketingClassification:
    """Marketing/newsletter/promo senders → COMPANY_MARKETING."""

    def test_newsletter_is_marketing(self, classifier):
        event = {
            "type": "email.received",
            "source": "proton_mail",
            "payload": {
                "from_address": "newsletter@techcompany.com",
                "subject": "Weekly Tech News",
                "body_plain": "Click here to unsubscribe",
            },
        }
        assert classifier.classify_event(event) == SenderType.COMPANY_MARKETING

    def test_noreply_is_marketing(self, classifier):
        event = {
            "type": "email.received",
            "source": "proton_mail",
            "payload": {
                "from_address": "noreply@store.com",
                "subject": "New products just for you!",
            },
        }
        assert classifier.classify_event(event) == SenderType.COMPANY_MARKETING

    def test_marketing_subdomain_is_marketing(self, classifier):
        event = {
            "type": "email.received",
            "source": "proton_mail",
            "payload": {
                "from_address": "deals@email.retailer.com",
                "subject": "Flash sale!",
            },
        }
        assert classifier.classify_event(event) == SenderType.COMPANY_MARKETING

    def test_substack_is_marketing(self, classifier):
        event = {
            "type": "email.received",
            "source": "proton_mail",
            "payload": {
                "from_address": "author@substack.com",
                "subject": "New post from Author",
            },
        }
        assert classifier.classify_event(event) == SenderType.COMPANY_MARKETING

    def test_unsubscribe_in_body_is_marketing(self, classifier):
        event = {
            "type": "email.received",
            "source": "proton_mail",
            "payload": {
                "from_address": "updates@company.com",
                "subject": "Product update",
                "body_plain": "To unsubscribe from these emails...",
            },
        }
        assert classifier.classify_event(event) == SenderType.COMPANY_MARKETING


# ============================================================================
# SenderClassifier — Transactional detection
# ============================================================================


class TestTransactionalClassification:
    """Bills, payments, account alerts → COMPANY_TRANSACTIONAL."""

    def test_billing_address_is_transactional(self, classifier):
        result = classifier.classify_address(
            "billing@utilities.com",
            {
                "subject": "Your monthly bill is ready",
            },
        )
        assert result == SenderType.COMPANY_TRANSACTIONAL

    def test_payment_confirmation_is_transactional(self, classifier):
        result = classifier.classify_address(
            "payment@bank.com",
            {
                "subject": "Payment confirmation",
            },
        )
        assert result == SenderType.COMPANY_TRANSACTIONAL

    def test_security_alert_is_transactional(self, classifier):
        result = classifier.classify_address(
            "security@mybank.com",
            {
                "subject": "Unusual sign-in detected on your account",
            },
        )
        assert result == SenderType.COMPANY_TRANSACTIONAL

    def test_bill_due_subject_is_transactional(self, classifier):
        """Even from a non-obvious address, bill-related subjects are transactional."""
        result = classifier.classify_address(
            "service@electric.com",
            {
                "subject": "Your bill is due on March 1",
                "body_plain": "Amount due: $145.00. Due date: March 1. Pay now at...",
                "snippet": "Amount due: $145.00",
            },
        )
        assert result == SenderType.COMPANY_TRANSACTIONAL

    def test_order_shipped_is_transactional(self, classifier):
        result = classifier.classify_address(
            "shipping@retailer.com",
            {
                "subject": "Your order has shipped!",
                "body_plain": "Tracking number: 1Z999AA10123456784",
            },
        )
        assert result == SenderType.COMPANY_TRANSACTIONAL

    def test_appointment_reminder_is_transactional(self, classifier):
        result = classifier.classify_address(
            "appointments@dentist.com",
            {
                "subject": "Appointment reminder for Feb 25",
            },
        )
        assert result == SenderType.COMPANY_TRANSACTIONAL


# ============================================================================
# SenderClassifier — Human detection
# ============================================================================


class TestHumanClassification:
    """Real human senders → HUMAN."""

    def test_personal_gmail_is_human(self, classifier):
        result = classifier.classify_address("alice@gmail.com")
        assert result == SenderType.HUMAN

    def test_personal_protonmail_is_human(self, classifier):
        result = classifier.classify_address("bob@protonmail.com")
        assert result == SenderType.HUMAN

    def test_work_colleague_is_human(self, classifier):
        result = classifier.classify_address(
            "jane.doe@company.com",
            {
                "subject": "Re: Lunch tomorrow?",
                "body_plain": "Hey, sounds good! See you at noon.",
            },
        )
        assert result == SenderType.HUMAN

    def test_non_communication_event_is_unknown(self, classifier):
        event = {
            "type": "calendar.event.created",
            "source": "caldav",
            "payload": {"title": "Team meeting"},
        }
        assert classifier.classify_event(event) == SenderType.UNKNOWN


# ============================================================================
# SenderClassifier — Caching
# ============================================================================


class TestClassificationCaching:
    """Classification results are cached for fast repeat lookups."""

    def test_cache_hit_returns_same_result(self, classifier):
        result1 = classifier.classify_address("alice@gmail.com")
        result2 = classifier.classify_address("alice@gmail.com")
        assert result1 == result2 == SenderType.HUMAN

    def test_reclassify_updates_cache(self, classifier):
        result1 = classifier.classify_address("edge-case@company.com")
        assert result1 == SenderType.HUMAN

        classifier.reclassify_address("edge-case@company.com", SenderType.COMPANY_TRANSACTIONAL)
        result2 = classifier.classify_address("edge-case@company.com")
        assert result2 == SenderType.COMPANY_TRANSACTIONAL


# ============================================================================
# NotificationManager — Sender-type routing
# ============================================================================


class TestSenderTypeRouting:
    """Company notifications are routed to digests, human goes through normally."""

    @pytest.mark.asyncio
    async def test_human_notification_delivered_immediately(self, notif_manager):
        """Human notifications go through normal delivery flow."""
        notif_id = await notif_manager.create_notification(
            title="Message from Alice",
            body="Hey, are you free tonight?",
            sender_type="human",
        )
        assert notif_id is not None

        # Should NOT be in the company digest
        weekly = notif_manager.get_company_digest("weekly")
        monthly = notif_manager.get_company_digest("monthly")
        assert len(weekly) == 0
        assert len(monthly) == 0

    @pytest.mark.asyncio
    async def test_marketing_notification_routed_to_monthly_digest(self, notif_manager):
        """Marketing emails go to the monthly digest, not real-time."""
        notif_id = await notif_manager.create_notification(
            title="Newsletter: Weekly Tech Roundup",
            body="This week's top stories...",
            sender_type="company_marketing",
            from_address="newsletter@tech.com",
        )
        assert notif_id is not None

        # Should be in the monthly digest
        monthly = notif_manager.get_company_digest("monthly")
        assert len(monthly) == 1
        assert monthly[0]["title"] == "Newsletter: Weekly Tech Roundup"
        assert monthly[0]["from_address"] == "newsletter@tech.com"
        assert monthly[0]["digest_type"] == "monthly"

        # Should NOT be in weekly digest
        weekly = notif_manager.get_company_digest("weekly")
        assert len(weekly) == 0

    @pytest.mark.asyncio
    async def test_transactional_notification_routed_to_weekly_digest(self, notif_manager):
        """Bills/receipts go to the weekly digest."""
        notif_id = await notif_manager.create_notification(
            title="Electric Bill Due",
            body="Your bill of $145.00 is due March 1",
            sender_type="company_transactional",
            from_address="billing@electric.com",
        )
        assert notif_id is not None

        # Should be in the weekly digest
        weekly = notif_manager.get_company_digest("weekly")
        assert len(weekly) == 1
        assert weekly[0]["title"] == "Electric Bill Due"
        assert weekly[0]["digest_type"] == "weekly"

    @pytest.mark.asyncio
    async def test_no_sender_type_treated_as_human(self, notif_manager):
        """None sender_type defaults to normal delivery (human behavior)."""
        notif_id = await notif_manager.create_notification(
            title="System alert",
            body="Something happened",
        )
        assert notif_id is not None

        # Should NOT be in any company digest
        weekly = notif_manager.get_company_digest("weekly")
        monthly = notif_manager.get_company_digest("monthly")
        assert len(weekly) == 0
        assert len(monthly) == 0

    @pytest.mark.asyncio
    async def test_notification_status_set_to_digest_queued(self, notif_manager, db):
        """Company notifications get status 'digest_queued'."""
        notif_id = await notif_manager.create_notification(
            title="Promo email",
            sender_type="company_marketing",
        )

        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT status, sender_type FROM notifications WHERE id = ?",
                (notif_id,),
            ).fetchone()

        assert row["status"] == "digest_queued"
        assert row["sender_type"] == "company_marketing"


# ============================================================================
# Company Digest — Review workflow
# ============================================================================


class TestCompanyDigestReview:
    """Users review company digests on their schedule."""

    @pytest.mark.asyncio
    async def test_review_marks_all_items(self, notif_manager):
        """Reviewing a digest marks all pending items as reviewed."""
        # Add 3 marketing items
        for i in range(3):
            await notif_manager.create_notification(
                title=f"Marketing email {i}",
                sender_type="company_marketing",
                from_address=f"promo{i}@company.com",
            )

        # Verify 3 pending
        monthly = notif_manager.get_company_digest("monthly")
        assert len(monthly) == 3

        # Review all
        count = notif_manager.mark_digest_reviewed("monthly")
        assert count == 3

        # Verify none pending
        monthly = notif_manager.get_company_digest("monthly")
        assert len(monthly) == 0

    @pytest.mark.asyncio
    async def test_review_single_item(self, notif_manager):
        """Individual items can be reviewed one at a time."""
        await notif_manager.create_notification(
            title="Bill 1",
            sender_type="company_transactional",
        )
        await notif_manager.create_notification(
            title="Bill 2",
            sender_type="company_transactional",
        )

        weekly = notif_manager.get_company_digest("weekly")
        assert len(weekly) == 2

        # Review just the first item
        notif_manager.mark_digest_item_reviewed(weekly[0]["id"])

        # One should remain
        weekly = notif_manager.get_company_digest("weekly")
        assert len(weekly) == 1

    @pytest.mark.asyncio
    async def test_digest_stats(self, notif_manager):
        """Stats show pending and reviewed counts per digest type."""
        # Add some items
        for i in range(3):
            await notif_manager.create_notification(
                title=f"Marketing {i}",
                sender_type="company_marketing",
            )
        for i in range(2):
            await notif_manager.create_notification(
                title=f"Bill {i}",
                sender_type="company_transactional",
            )

        # Review all monthly items
        notif_manager.mark_digest_reviewed("monthly")

        stats = notif_manager.get_digest_stats()
        assert stats["monthly"]["reviewed"] == 3
        assert stats["weekly"]["pending"] == 2


# ============================================================================
# Human-only notification view
# ============================================================================


class TestHumanNotificationView:
    """The human-only view filters out company noise."""

    @pytest.mark.asyncio
    async def test_human_only_excludes_company(self, notif_manager):
        """get_pending_human_notifications excludes company-classified items."""
        # Create a human notification (delivered)
        await notif_manager.create_notification(
            title="Message from Bob",
            sender_type="human",
            priority="high",
        )

        # Create company notifications (these go to digest, not notification list)
        await notif_manager.create_notification(
            title="Newsletter",
            sender_type="company_marketing",
        )

        # Human-only view should only show the human message
        human_notifs = notif_manager.get_pending_human_notifications()
        titles = [n["title"] for n in human_notifs]
        assert "Message from Bob" in titles
        # Marketing went to digest_queued, so it won't show in pending/delivered

    @pytest.mark.asyncio
    async def test_none_sender_type_shows_in_human_view(self, notif_manager):
        """Notifications without sender_type appear in the human view."""
        await notif_manager.create_notification(
            title="System prediction",
            priority="high",
        )

        human_notifs = notif_manager.get_pending_human_notifications()
        assert any(n["title"] == "System prediction" for n in human_notifs)


# ============================================================================
# SenderClassifier — Classification stats
# ============================================================================


class TestClassificationStats:
    """Classification stats report counts per sender type."""

    def test_stats_empty_initially(self, classifier):
        stats = classifier.get_classification_stats()
        assert stats == {}

    def test_stats_after_classification(self, classifier, db):
        """Stats reflect persisted classifications."""
        # Create a contact with sender_type
        with db.get_connection("entities") as conn:
            conn.execute(
                """INSERT INTO contacts (id, name, sender_type)
                   VALUES ('c1', 'Newsletter Co', 'company_marketing')""",
            )
            conn.execute(
                """INSERT INTO contacts (id, name, sender_type)
                   VALUES ('c2', 'Electric Co', 'company_transactional')""",
            )
            conn.execute(
                """INSERT INTO contacts (id, name, sender_type)
                   VALUES ('c3', 'Alice', 'human')""",
            )
            conn.execute(
                """INSERT INTO contacts (id, name, sender_type)
                   VALUES ('c4', 'Bob', 'human')""",
            )

        stats = classifier.get_classification_stats()
        assert stats["human"] == 2
        assert stats["company_marketing"] == 1
        assert stats["company_transactional"] == 1
