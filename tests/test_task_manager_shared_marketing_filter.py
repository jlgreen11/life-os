"""
Tests verifying that TaskManager._is_marketing_email delegates to the shared
is_marketing_or_noreply() filter (iteration 253).

Prior to this change, the task manager maintained a hand-rolled local copy of
the marketing filter that had diverged from the canonical implementation in
services/signal_extractor/marketing_filter.py.  These tests confirm:

  1. The delegation is wired correctly (shared filter is actually called).
  2. TaskManager now catches senders that the old local filter missed but the
     shared filter correctly identifies (financial/brokerage, retail/hospitality,
     embedded notification patterns, ESP platform domains, brand self-mailers).
  3. Personal human emails are still not filtered (no regression).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from services.task_manager.manager import TaskManager
from services.signal_extractor.marketing_filter import is_marketing_or_noreply


class TestSharedFilterDelegation:
    """Verify that _is_marketing_email delegates to is_marketing_or_noreply."""

    def test_delegates_to_shared_filter_positive(self):
        """_is_marketing_email returns True when shared filter returns True."""
        payload = {"from_address": "noreply@example.com"}
        with patch(
            "services.task_manager.manager.is_marketing_or_noreply",
            return_value=True,
        ) as mock_filter:
            result = TaskManager._is_marketing_email(payload)
            mock_filter.assert_called_once_with("noreply@example.com", payload)
            assert result is True

    def test_delegates_to_shared_filter_negative(self):
        """_is_marketing_email returns False when shared filter returns False."""
        payload = {"from_address": "alice@company.com", "body": "Can we meet?"}
        with patch(
            "services.task_manager.manager.is_marketing_or_noreply",
            return_value=False,
        ) as mock_filter:
            result = TaskManager._is_marketing_email(payload)
            mock_filter.assert_called_once_with("alice@company.com", payload)
            assert result is False

    def test_passes_full_payload_for_body_check(self):
        """Full payload is forwarded so the shared filter can check for unsubscribe."""
        payload = {
            "from_address": "alice@startup.io",
            "body": "Important update — click to unsubscribe",
            "snippet": "...",
        }
        with patch(
            "services.task_manager.manager.is_marketing_or_noreply",
            return_value=True,
        ) as mock_filter:
            TaskManager._is_marketing_email(payload)
            # Verify the entire payload is passed (not just the address)
            _, call_kwargs = mock_filter.call_args
            args = mock_filter.call_args[0]
            assert args[1] is payload, "Full payload must be forwarded for body/snippet check"

    def test_handles_missing_from_address(self):
        """Empty from_address is forwarded as empty string, not raising an error."""
        payload = {"body": "Some content"}
        # Should not raise; shared filter gracefully handles empty from_addr
        result = TaskManager._is_marketing_email(payload)
        # Empty from_addr → shared filter returns False (no patterns match "")
        assert result is False


class TestNewPatternsCaughtBySharedFilter:
    """Verify patterns caught by the shared filter that the old local filter missed.

    The old local _is_marketing_email only had:
      - Basic noreply patterns
      - A small set of bulk local-parts
      - A handful of marketing domain patterns

    The shared filter additionally catches financial/brokerage, retail/hospitality,
    embedded notification substrings, ESP platform domains, and brand self-mailers.
    """

    def test_filters_financial_brokerage_senders(self):
        """Financial/brokerage automated senders should be filtered.

        These were added to the shared filter in iteration 171 but were never
        in the task manager's local _is_marketing_email.
        """
        financial_senders = [
            "schwab@notifications.schwab.com",
            "paypal@paypal.com",
            "coinbase@coinbase.com",
            "experian@experian.com",
            "creditkarma@creditkarma.com",
        ]
        for from_addr in financial_senders:
            payload = {"from_address": from_addr}
            assert TaskManager._is_marketing_email(payload) is True, (
                f"Should filter financial sender {from_addr}"
            )

    def test_filters_retail_hospitality_senders(self):
        """Retail/hospitality automated senders should be filtered.

        These were added to the shared filter in iteration 178 but were never
        in the task manager's local _is_marketing_email.
        """
        retail_senders = [
            "customerservice@nationalcar.com",
            "reservations@hotel.com",
            "tracking@shipstation.com",
            "shipping@amazon.com",
            "receipt@store.com",
        ]
        for from_addr in retail_senders:
            payload = {"from_address": from_addr}
            assert TaskManager._is_marketing_email(payload) is True, (
                f"Should filter retail/hospitality sender {from_addr}"
            )

    def test_filters_embedded_notification_patterns(self):
        """Local-parts containing notification substrings should be filtered.

        The old filter only checked startswith; the shared filter also checks
        for embedded patterns like '-notification', 'news', 'no.reply'.
        """
        embedded_patterns = [
            "hoa-notifications@homeowners.org",   # '-notifications' embedded
            "user-alert@service.com",              # '-alert' embedded
            "morningnews@company.com",             # 'news' embedded
            "team-updates@corp.com",               # '-updates' embedded
        ]
        for from_addr in embedded_patterns:
            payload = {"from_address": from_addr}
            assert TaskManager._is_marketing_email(payload) is True, (
                f"Should filter embedded-pattern sender {from_addr}"
            )

    def test_filters_esp_platform_domains(self):
        """Email service provider platform domains should be filtered.

        The shared filter catches @em., @mg., @engage., @iluv., @e., @e2., etc.
        The old filter only had @em., @mg., @mail., @mailing., etc.
        """
        esp_senders = [
            "brand@em.brand.com",           # @em. ESP pattern
            "brand@mg.brand.com",           # @mg. ESP pattern
            "brand@engage.platform.com",    # @engage. ESP pattern
            "brand@e2.retailer.com",        # @e2. ESP pattern
        ]
        for from_addr in esp_senders:
            payload = {"from_address": from_addr}
            assert TaskManager._is_marketing_email(payload) is True, (
                f"Should filter ESP platform sender {from_addr}"
            )

    def test_filters_loyalty_reward_senders(self):
        """Loyalty and reward programme senders should be filtered."""
        loyalty_senders = [
            "loyalty@brand.com",
            "rewards@company.com",
        ]
        for from_addr in loyalty_senders:
            payload = {"from_address": from_addr}
            assert TaskManager._is_marketing_email(payload) is True, (
                f"Should filter loyalty sender {from_addr}"
            )

    def test_still_allows_human_business_contacts(self):
        """Human business contacts that happen to work at financial/retail firms.

        The filter should only block clearly automated sender patterns, not
        every address at a financial or retail company.
        """
        human_contacts = [
            {
                "from_address": "john.smith@schwab.com",
                "body": "Can we discuss your portfolio strategy on Thursday?",
            },
            {
                "from_address": "mary.jones@amazon.com",
                "body": "Following up on the vendor contract, please review section 3.",
            },
            {
                "from_address": "r.martinez@hyatt.com",
                "body": "Your group rate proposal is attached, let me know if you have questions.",
            },
        ]
        for payload in human_contacts:
            assert TaskManager._is_marketing_email(payload) is False, (
                f"Should NOT filter human business contact {payload.get('from_address')}"
            )


class TestProcessEventFilterIntegration:
    """Integration-level tests verifying that process_event skips marketing emails
    by calling _is_marketing_email (which now delegates to the shared filter).

    These tests use a mock AI engine to verify that extract_action_items is never
    called for filtered emails, confirming the filter is wired into the pipeline.
    """

    @pytest.fixture
    def task_manager(self, db):
        """Create a TaskManager with a mock AI engine."""
        from unittest.mock import AsyncMock
        mock_ai = AsyncMock()
        mock_ai.extract_action_items = AsyncMock(return_value=[])
        return TaskManager(db, ai_engine=mock_ai)

    @pytest.mark.asyncio
    async def test_process_event_skips_noreply_email(self, task_manager):
        """process_event should not call AI engine for noreply marketing emails."""
        event = {
            "type": "email.received",
            "payload": {
                "from_address": "noreply@newsletter.com",
                "body": "This week's top offers — click here!",
                "subject": "Weekly Newsletter",
            },
        }
        await task_manager.process_event(event)
        task_manager.ai_engine.extract_action_items.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_event_skips_financial_sender(self, task_manager):
        """process_event should not call AI engine for financial automated senders.

        This catches the gap in the old local filter: financial senders like
        schwab@ are now filtered by the shared filter but were NOT caught by
        the old _is_marketing_email.
        """
        event = {
            "type": "email.received",
            "payload": {
                "from_address": "schwab@notifications.schwab.com",
                "body": "Your trade confirmation: 100 shares of AAPL at $187.50",
                "subject": "Trade Confirmation",
            },
        }
        await task_manager.process_event(event)
        task_manager.ai_engine.extract_action_items.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_event_processes_human_email(self, task_manager):
        """process_event should call AI engine for genuine human emails."""
        event = {
            "type": "email.received",
            "payload": {
                "from_address": "boss@company.com",
                "body": "Please prepare the Q1 budget report for Friday's board meeting.",
                "subject": "Q1 Budget Report",
            },
        }
        await task_manager.process_event(event)
        task_manager.ai_engine.extract_action_items.assert_called_once()
