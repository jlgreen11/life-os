"""
Comprehensive tests for the FinanceConnector (Plaid integration).

Tests cover:
    - Authentication and Plaid client initialization
    - Transaction syncing with cursor-based incremental fetch
    - Large transaction threshold detection and priority elevation
    - Sync cursor persistence across multiple sync cycles
    - Error handling for network failures and invalid tokens
    - Health check validation
    - Read-only enforcement (no execute() support)
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from connectors.finance.connector import FinanceConnector


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


@contextmanager
def mock_plaid_sync_request():
    """Context manager that mocks the TransactionsSyncRequest import.

    The FinanceConnector lazily imports TransactionsSyncRequest inside sync(),
    so we need to mock it at the sys.modules level.
    """
    mock_request_class = MagicMock()
    mock_module = MagicMock()
    mock_module.TransactionsSyncRequest = mock_request_class

    with patch.dict('sys.modules', {'plaid.model.transactions_sync_request': mock_module}):
        yield mock_request_class


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def finance_config() -> dict[str, Any]:
    """Minimal valid FinanceConnector configuration."""
    return {
        "client_id": "test-client-id",
        "secret": "test-secret",
        "access_tokens": ["test-access-token-1", "test-access-token-2"],
        "large_transaction_threshold": 500,
    }


@pytest.fixture
def finance_connector(event_bus, db, finance_config):
    """FinanceConnector instance with mocked dependencies."""
    return FinanceConnector(event_bus, db, finance_config)


@pytest.fixture
def mock_plaid_client():
    """Mock Plaid API client with transactions_sync method."""
    client = MagicMock()
    client.transactions_sync = MagicMock()
    return client


@pytest.fixture
def mock_plaid_transaction():
    """Factory for creating mock Plaid transaction objects."""
    def _create(
        transaction_id: str,
        account_id: str,
        amount: float,
        merchant_name: str | None = None,
        name: str = "Generic Merchant",
        iso_currency_code: str = "USD",
        date_obj: date | None = None,
        pending: bool = False,
        category_primary: str | None = "GENERAL_MERCHANDISE",
    ):
        """Create a mock Plaid transaction with sensible defaults."""
        txn = MagicMock()
        txn.transaction_id = transaction_id
        txn.account_id = account_id
        txn.amount = amount
        txn.merchant_name = merchant_name
        txn.name = name
        txn.iso_currency_code = iso_currency_code
        txn.date = date_obj or date(2026, 2, 15)
        txn.pending = pending

        # Mock personal_finance_category attribute
        if category_primary:
            txn.personal_finance_category = MagicMock()
            txn.personal_finance_category.primary = category_primary
        else:
            txn.personal_finance_category = None

        return txn

    return _create


# ---------------------------------------------------------------------------
# Authentication Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_success(finance_connector, finance_config):
    """Verify that authenticate() initializes the Plaid client successfully."""
    # Mock the lazy import of plaid module
    mock_plaid = MagicMock()
    mock_config = MagicMock()
    mock_plaid.Configuration.return_value = mock_config
    mock_plaid.Environment.Production = "https://production.plaid.com"
    mock_api_client = MagicMock()
    mock_plaid.ApiClient.return_value = mock_api_client
    mock_client = MagicMock()
    mock_plaid.api.plaid_api.PlaidApi.return_value = mock_client

    with patch.dict('sys.modules', {'plaid': mock_plaid, 'plaid.api': mock_plaid.api, 'plaid.model.products': MagicMock()}):
        result = await finance_connector.authenticate()

    assert result is True
    assert finance_connector._client == mock_client
    mock_plaid.Configuration.assert_called_once()
    mock_plaid.ApiClient.assert_called_once_with(mock_config)


@pytest.mark.asyncio
async def test_authenticate_failure_missing_plaid(finance_connector):
    """Verify that authenticate() fails gracefully when plaid SDK is missing."""
    # Simulate plaid module not being installed
    with patch.dict('sys.modules', {'plaid': None}):
        with patch('builtins.__import__', side_effect=ImportError("No module named 'plaid'")):
            result = await finance_connector.authenticate()

    assert result is False
    assert finance_connector._client is None


@pytest.mark.asyncio
async def test_authenticate_failure_invalid_credentials(finance_connector):
    """Verify that authenticate() handles SDK initialization errors."""
    mock_plaid = MagicMock()
    mock_plaid.Configuration.side_effect = ValueError("Invalid client_id")
    mock_plaid.Environment.Production = "https://production.plaid.com"

    with patch.dict('sys.modules', {'plaid': mock_plaid, 'plaid.api': mock_plaid.api, 'plaid.model.products': MagicMock()}):
        result = await finance_connector.authenticate()

    assert result is False
    assert finance_connector._client is None


# ---------------------------------------------------------------------------
# Transaction Sync Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_no_client(finance_connector):
    """Verify that sync() returns 0 when client is not initialized."""
    finance_connector._client = None

    count = await finance_connector.sync()

    assert count == 0


@pytest.mark.asyncio
async def test_sync_initial_fetch_empty_cursor(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify initial sync with empty cursor fetches all recent transactions."""
    finance_connector._client = mock_plaid_client
    # Use a single token to simplify the count assertion
    finance_connector._access_tokens = ["test-token-1"]

    # Initialize connector state row (normally done by start())
    with finance_connector.db.get_connection("state") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO connector_state (connector_id, status, updated_at)
               VALUES (?, ?, ?)""",
            (finance_connector.CONNECTOR_ID, "active", datetime.now(timezone.utc).isoformat())
        )

    # Mock Plaid response with 3 new transactions
    mock_response = MagicMock()
    mock_response.added = [
        mock_plaid_transaction("txn-1", "acct-1", -45.50, merchant_name="Coffee Shop"),
        mock_plaid_transaction("txn-2", "acct-1", -120.00, merchant_name="Grocery Store"),
        mock_plaid_transaction("txn-3", "acct-2", -15.99, name="Online Service"),
    ]
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "cursor-abc123"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request() as mock_request:
        count = await finance_connector.sync()

    assert count == 3
    # Verify cursor was passed as empty string on first sync
    call_args = mock_request.call_args
    assert call_args[1]["cursor"] == ""
    # Verify events were published
    assert event_bus.publish.call_count == 3
    # Verify cursor was persisted in the database
    with finance_connector.db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT sync_cursor FROM connector_state WHERE connector_id = ?",
            (finance_connector.CONNECTOR_ID,)
        ).fetchone()
        assert row["sync_cursor"] == "cursor-abc123"


@pytest.mark.asyncio
async def test_sync_incremental_with_cursor(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify incremental sync uses stored cursor to fetch only new transactions."""
    finance_connector._client = mock_plaid_client
    finance_connector._access_tokens = ["test-token-1"]

    # Initialize connector state row (normally done by start())
    with finance_connector.db.get_connection("state") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO connector_state (connector_id, status, sync_cursor, updated_at)
               VALUES (?, ?, ?, ?)""",
            (finance_connector.CONNECTOR_ID, "active", "cursor-previous", datetime.now(timezone.utc).isoformat())
        )

    # Mock response with 1 new transaction
    mock_response = MagicMock()
    mock_response.added = [
        mock_plaid_transaction("txn-4", "acct-1", -299.99, merchant_name="Electronics Store"),
    ]
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "cursor-new"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request() as mock_request:
        count = await finance_connector.sync()

    assert count == 1
    # Verify previous cursor was passed to Plaid
    call_args = mock_request.call_args
    assert call_args[1]["cursor"] == "cursor-previous"
    # Verify new cursor was persisted in the database
    with finance_connector.db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT sync_cursor FROM connector_state WHERE connector_id = ?",
            (finance_connector.CONNECTOR_ID,)
        ).fetchone()
        assert row["sync_cursor"] == "cursor-new"


@pytest.mark.asyncio
async def test_sync_large_transaction_high_priority(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify transactions above threshold are published with high priority."""
    finance_connector._client = mock_plaid_client
    finance_connector._access_tokens = ["test-token-1"]

    # Transaction of $1500 (above $500 threshold)
    mock_response = MagicMock()
    mock_response.added = [
        mock_plaid_transaction("txn-large", "acct-1", -1500.00, merchant_name="Appliance Store"),
    ]
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "cursor-x"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        count = await finance_connector.sync()

    assert count == 1
    # Verify the event was published with high priority
    call_args = event_bus.publish.call_args
    assert call_args[1]["priority"] == "high"


@pytest.mark.asyncio
async def test_sync_normal_transaction_normal_priority(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify transactions below threshold are published with normal priority."""
    finance_connector._client = mock_plaid_client
    finance_connector._access_tokens = ["test-token-1"]

    # Transaction of $49.99 (below $500 threshold)
    mock_response = MagicMock()
    mock_response.added = [
        mock_plaid_transaction("txn-small", "acct-1", -49.99, merchant_name="Restaurant"),
    ]
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "cursor-y"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        count = await finance_connector.sync()

    assert count == 1
    # Verify the event was published with normal priority
    call_args = event_bus.publish.call_args
    assert call_args[1]["priority"] == "normal"


@pytest.mark.asyncio
async def test_sync_event_payload_structure(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify published event payload contains all expected fields."""
    finance_connector._client = mock_plaid_client

    mock_response = MagicMock()
    mock_response.added = [
        mock_plaid_transaction(
            transaction_id="txn-test",
            account_id="acct-test",
            amount=-99.50,
            merchant_name="Test Merchant",
            name="Test Merchant Inc",
            iso_currency_code="USD",
            date_obj=date(2026, 2, 14),
            pending=False,
            category_primary="FOOD_AND_DRINK",
        ),
    ]
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "cursor-z"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        await finance_connector.sync()

    # Extract the event payload that was published
    call_args = event_bus.publish.call_args
    event_type = call_args[0][0]
    payload = call_args[0][1]

    assert event_type == "finance.transaction.new"
    assert payload["transaction_id"] == "txn-test"
    assert payload["account_id"] == "acct-test"
    assert payload["amount"] == -99.50
    assert payload["currency"] == "USD"
    assert payload["merchant"] == "Test Merchant"
    assert payload["category"] == "FOOD_AND_DRINK"
    assert payload["date"] == "2026-02-14"
    assert payload["is_pending"] is False


@pytest.mark.asyncio
async def test_sync_fallback_to_name_when_no_merchant(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify that raw name is used when merchant_name is missing."""
    finance_connector._client = mock_plaid_client

    mock_response = MagicMock()
    mock_response.added = [
        mock_plaid_transaction(
            "txn-no-merchant", "acct-1", -25.00,
            merchant_name=None,  # No enriched merchant name
            name="Generic Store"
        ),
    ]
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "cursor-fallback"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        await finance_connector.sync()

    call_args = event_bus.publish.call_args
    payload = call_args[0][1]
    assert payload["merchant"] == "Generic Store"


@pytest.mark.asyncio
async def test_sync_missing_category_none(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify that category is None when Plaid doesn't provide one."""
    finance_connector._client = mock_plaid_client

    mock_response = MagicMock()
    mock_response.added = [
        mock_plaid_transaction(
            "txn-no-cat", "acct-1", -10.00,
            merchant_name="Test",
            category_primary=None
        ),
    ]
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "cursor-no-cat"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        await finance_connector.sync()

    call_args = event_bus.publish.call_args
    payload = call_args[0][1]
    assert payload["category"] is None


@pytest.mark.asyncio
async def test_sync_currency_defaults_to_usd(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify that currency defaults to USD when iso_currency_code is missing."""
    finance_connector._client = mock_plaid_client

    # Create transaction without currency code
    txn = mock_plaid_transaction("txn-no-curr", "acct-1", -50.00)
    txn.iso_currency_code = None

    mock_response = MagicMock()
    mock_response.added = [txn]
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "cursor-curr"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        await finance_connector.sync()

    call_args = event_bus.publish.call_args
    payload = call_args[0][1]
    assert payload["currency"] == "USD"


@pytest.mark.asyncio
async def test_sync_positive_amount_credit(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify that positive amounts (credits/refunds) use normal priority."""
    finance_connector._client = mock_plaid_client
    finance_connector._access_tokens = ["test-token-1"]

    # Positive amount = money IN (refund or deposit)
    mock_response = MagicMock()
    mock_response.added = [
        mock_plaid_transaction("txn-refund", "acct-1", 150.00, merchant_name="Refund Inc"),
    ]
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "cursor-credit"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        count = await finance_connector.sync()

    assert count == 1
    call_args = event_bus.publish.call_args
    payload = call_args[0][1]
    # Amount should be preserved as positive
    assert payload["amount"] == 150.00
    # Priority should be normal (abs(150) < 500)
    assert call_args[1]["priority"] == "normal"


@pytest.mark.asyncio
async def test_sync_multiple_access_tokens(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify that sync() processes all configured access tokens."""
    finance_connector._client = mock_plaid_client
    finance_connector._access_tokens = ["token-1", "token-2"]

    # Mock response for each token
    mock_response_1 = MagicMock()
    mock_response_1.added = [
        mock_plaid_transaction("txn-1", "acct-1", -10.00),
    ]
    mock_response_1.modified = []
    mock_response_1.removed = []
    mock_response_1.next_cursor = "cursor-1"

    mock_response_2 = MagicMock()
    mock_response_2.added = [
        mock_plaid_transaction("txn-2", "acct-2", -20.00),
        mock_plaid_transaction("txn-3", "acct-2", -30.00),
    ]
    mock_response_2.modified = []
    mock_response_2.removed = []
    mock_response_2.next_cursor = "cursor-2"

    mock_plaid_client.transactions_sync.side_effect = [mock_response_1, mock_response_2]

    with mock_plaid_sync_request():
        count = await finance_connector.sync()

    # Should have processed 3 total transactions (1 from token-1, 2 from token-2)
    assert count == 3
    assert event_bus.publish.call_count == 3


@pytest.mark.asyncio
async def test_sync_error_one_token_continues_others(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify that errors on one token don't prevent processing other tokens."""
    finance_connector._client = mock_plaid_client
    finance_connector._access_tokens = ["bad-token", "good-token"]

    # First token raises error, second succeeds
    mock_response_good = MagicMock()
    mock_response_good.added = [
        mock_plaid_transaction("txn-good", "acct-1", -50.00),
    ]
    mock_response_good.modified = []
    mock_response_good.removed = []
    mock_response_good.next_cursor = "cursor-good"

    mock_plaid_client.transactions_sync.side_effect = [
        ValueError("Invalid access token"),
        mock_response_good,
    ]

    with mock_plaid_sync_request():
        count = await finance_connector.sync()

    # Should still process the good token's transaction
    assert count == 1
    assert event_bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_sync_cursor_persistence_per_token(
    finance_connector, mock_plaid_client, mock_plaid_transaction
):
    """Verify that sync cursor is updated after processing all tokens."""
    finance_connector._client = mock_plaid_client
    finance_connector._access_tokens = ["token-1"]

    # Initialize connector state row (normally done by start())
    with finance_connector.db.get_connection("state") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO connector_state (connector_id, status, updated_at)
               VALUES (?, ?, ?)""",
            (finance_connector.CONNECTOR_ID, "active", datetime.now(timezone.utc).isoformat())
        )

    mock_response = MagicMock()
    mock_response.added = []
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "new-cursor-value"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        await finance_connector.sync()

    # Verify cursor was persisted in the database
    with finance_connector.db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT sync_cursor FROM connector_state WHERE connector_id = ?",
            (finance_connector.CONNECTOR_ID,)
        ).fetchone()
        assert row["sync_cursor"] == "new-cursor-value"


@pytest.mark.asyncio
async def test_sync_no_new_transactions(
    finance_connector, mock_plaid_client, event_bus
):
    """Verify that sync() handles empty transaction list gracefully."""
    finance_connector._client = mock_plaid_client

    mock_response = MagicMock()
    mock_response.added = []
    mock_response.modified = []
    mock_response.removed = []
    mock_response.next_cursor = "cursor-empty"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        count = await finance_connector.sync()

    assert count == 0
    assert event_bus.publish.call_count == 0


# ---------------------------------------------------------------------------
# Modified Transaction Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_processes_modified_transactions(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify that modified transactions are published with finance.transaction.modified event type.

    Modified transactions occur when e.g. a pending charge clears with a
    different final amount.  They carry the same payload shape as added
    transactions but use a distinct event type so downstream consumers can
    differentiate new vs. updated records.
    """
    finance_connector._client = mock_plaid_client
    finance_connector._access_tokens = ["test-token-1"]

    mock_response = MagicMock()
    mock_response.added = []
    mock_response.modified = [
        # Above threshold — should get high priority
        mock_plaid_transaction("txn-mod-1", "acct-1", -750.00, merchant_name="Updated Merchant"),
        # Below threshold — should get normal priority
        mock_plaid_transaction("txn-mod-2", "acct-1", -25.00, merchant_name="Small Update"),
    ]
    mock_response.removed = []
    mock_response.next_cursor = "cursor-mod"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        count = await finance_connector.sync()

    assert count == 2
    assert event_bus.publish.call_count == 2

    # First call: large modified transaction → high priority
    first_call = event_bus.publish.call_args_list[0]
    assert first_call[0][0] == "finance.transaction.modified"
    assert first_call[0][1]["transaction_id"] == "txn-mod-1"
    assert first_call[0][1]["amount"] == -750.00
    assert first_call[0][1]["merchant"] == "Updated Merchant"
    assert first_call[1]["priority"] == "high"

    # Second call: small modified transaction → normal priority
    second_call = event_bus.publish.call_args_list[1]
    assert second_call[0][0] == "finance.transaction.modified"
    assert second_call[0][1]["transaction_id"] == "txn-mod-2"
    assert second_call[1]["priority"] == "normal"


# ---------------------------------------------------------------------------
# Removed Transaction Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_processes_removed_transactions(
    finance_connector, mock_plaid_client, event_bus
):
    """Verify that removed transactions are published with minimal payload.

    Plaid's removed transaction objects only contain a transaction_id — no
    amount, merchant, or other fields.  The event is published at low
    priority since removals are informational.
    """
    finance_connector._client = mock_plaid_client
    finance_connector._access_tokens = ["test-token-1"]

    # Removed transactions only have transaction_id
    removed_txn = MagicMock()
    removed_txn.transaction_id = "txn-removed-1"

    mock_response = MagicMock()
    mock_response.added = []
    mock_response.modified = []
    mock_response.removed = [removed_txn]
    mock_response.next_cursor = "cursor-rem"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        count = await finance_connector.sync()

    assert count == 1
    assert event_bus.publish.call_count == 1

    call_args = event_bus.publish.call_args
    assert call_args[0][0] == "finance.transaction.removed"
    assert call_args[0][1] == {"transaction_id": "txn-removed-1"}
    assert call_args[1]["priority"] == "low"


@pytest.mark.asyncio
async def test_sync_handles_all_three_lists(
    finance_connector, mock_plaid_client, mock_plaid_transaction, event_bus
):
    """Verify that sync processes added, modified, and removed in one cycle.

    The total count returned should reflect all three lists, and each list
    should produce events with the correct event type.
    """
    finance_connector._client = mock_plaid_client
    finance_connector._access_tokens = ["test-token-1"]

    removed_txn = MagicMock()
    removed_txn.transaction_id = "txn-rem-1"

    mock_response = MagicMock()
    mock_response.added = [
        mock_plaid_transaction("txn-add-1", "acct-1", -30.00, merchant_name="New Store"),
    ]
    mock_response.modified = [
        mock_plaid_transaction("txn-mod-1", "acct-1", -45.00, merchant_name="Updated Store"),
    ]
    mock_response.removed = [removed_txn]
    mock_response.next_cursor = "cursor-all"
    mock_plaid_client.transactions_sync.return_value = mock_response

    with mock_plaid_sync_request():
        count = await finance_connector.sync()

    assert count == 3
    assert event_bus.publish.call_count == 3

    # Verify event types published in order: added, modified, removed
    event_types = [call[0][0] for call in event_bus.publish.call_args_list]
    assert event_types == [
        "finance.transaction.new",
        "finance.transaction.modified",
        "finance.transaction.removed",
    ]


# ---------------------------------------------------------------------------
# Execute Tests (Read-Only Enforcement)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_raises_on_any_action(finance_connector):
    """Verify that execute() raises ValueError for any action (read-only)."""
    with pytest.raises(ValueError, match="read-only"):
        await finance_connector.execute("send_payment", {"amount": 100})

    with pytest.raises(ValueError, match="read-only"):
        await finance_connector.execute("transfer", {})


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_ok_when_configured(finance_connector):
    """Verify health_check() returns 'ok' when client and tokens are present."""
    finance_connector._client = MagicMock()
    finance_connector._access_tokens = ["token-1", "token-2"]

    result = await finance_connector.health_check()

    assert result["status"] == "ok"
    assert result["connector"] == "finance"
    assert result["accounts"] == 2


@pytest.mark.asyncio
async def test_health_check_error_no_client(finance_connector):
    """Verify health_check() returns 'error' when client is missing."""
    finance_connector._client = None
    finance_connector._access_tokens = ["token-1"]

    result = await finance_connector.health_check()

    assert result["status"] == "error"
    assert "Not configured" in result["details"]


@pytest.mark.asyncio
async def test_health_check_error_no_tokens(finance_connector):
    """Verify health_check() returns 'error' when tokens are missing."""
    finance_connector._client = MagicMock()
    finance_connector._access_tokens = []

    result = await finance_connector.health_check()

    assert result["status"] == "error"
    assert "Not configured" in result["details"]


@pytest.mark.asyncio
async def test_health_check_error_neither(finance_connector):
    """Verify health_check() returns 'error' when both client and tokens are missing."""
    finance_connector._client = None
    finance_connector._access_tokens = []

    result = await finance_connector.health_check()

    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


def test_constructor_defaults(event_bus, db):
    """Verify FinanceConnector uses sensible defaults when config is minimal."""
    minimal_config = {
        "client_id": "test-id",
        "secret": "test-secret",
    }
    connector = FinanceConnector(event_bus, db, minimal_config)

    assert connector._access_tokens == []
    assert connector._large_threshold == 500  # Default threshold


def test_constructor_custom_threshold(event_bus, db):
    """Verify custom large_transaction_threshold is respected."""
    config = {
        "client_id": "test-id",
        "secret": "test-secret",
        "large_transaction_threshold": 1000,
    }
    connector = FinanceConnector(event_bus, db, config)

    assert connector._large_threshold == 1000


def test_connector_metadata():
    """Verify connector class metadata is correct."""
    assert FinanceConnector.CONNECTOR_ID == "finance"
    assert FinanceConnector.DISPLAY_NAME == "Finance (Plaid)"
    assert FinanceConnector.SYNC_INTERVAL_SECONDS == 3600  # Hourly
