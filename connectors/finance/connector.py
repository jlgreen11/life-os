"""
Life OS — Finance Connector (Plaid / Teller)

Connects to bank accounts and credit cards through Plaid or Teller API
for transaction monitoring, subscription detection, and spending analysis.

Configuration:
    connectors:
      finance:
        provider: "plaid"          # or "teller"
        client_id: "your-client-id"
        secret: "your-secret"
        access_tokens:             # One per linked account
          - "access-token-1"
        sync_interval: 3600        # Hourly
        large_transaction_threshold: 500
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class FinanceConnector(BaseConnector):
    """Read-only connector that pulls bank / credit-card transactions via Plaid.

    Integration pattern:
        1. The user completes Plaid Link in a separate onboarding flow, which
           yields one *access token* per linked financial institution.
        2. This connector stores those tokens and calls Plaid's
           ``transactions/sync`` endpoint on each sync cycle.
        3. ``transactions/sync`` is cursor-based: we persist a server-side
           cursor so each call returns only *new* transactions since the last
           sync, keeping bandwidth and processing costs low.
        4. Each transaction is normalised into a Life OS event payload and
           published to the event bus.  Large transactions (above a
           configurable threshold) are flagged with ``priority="high"`` so the
           agent can surface spending anomalies.
    """

    CONNECTOR_ID = "finance"
    DISPLAY_NAME = "Finance (Plaid)"
    # Financial data changes infrequently; hourly polling is sufficient.
    SYNC_INTERVAL_SECONDS = 3600  # Hourly

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        # PlaidApi client instance; initialised during authenticate().
        self._client = None
        # One Plaid access token per linked bank / institution.
        self._access_tokens = config.get("access_tokens", [])
        # Dollar amount above which a transaction is considered "large" and
        # gets elevated priority for anomaly alerting.
        self._large_threshold = config.get("large_transaction_threshold", 500)

    async def authenticate(self) -> bool:
        """Initialise the Plaid API client.

        Plaid authentication uses a ``client_id`` + ``secret`` pair (server-side
        credentials).  The ``plaid`` library is imported lazily so the rest of
        Life OS can run without it when finance is not configured.

        Note: no network call is made here -- we only construct the client
        object.  The first real validation happens on the initial ``sync()``.
        """
        try:
            # Lazy import: plaid SDK is an optional dependency.
            import plaid
            from plaid.api import plaid_api
            from plaid.model.products import Products

            # Build the SDK configuration pointing at the Production environment.
            # For sandbox / development testing, swap ``plaid.Environment.Production``
            # with ``plaid.Environment.Sandbox``.
            configuration = plaid.Configuration(
                host=plaid.Environment.Production,
                api_key={
                    "clientId": self.config["client_id"],
                    "secret": self.config["secret"],
                },
            )
            api_client = plaid.ApiClient(configuration)
            self._client = plaid_api.PlaidApi(api_client)
            return True
        except Exception as e:
            logger.error("Auth failed: %s", e)
            return False

    def _normalize_transaction(self, txn) -> dict[str, Any]:
        """Build a normalised event payload from a Plaid transaction object.

        Used for both newly added and modified transactions, which share the
        same field set in Plaid's ``transactions/sync`` response.
        """
        return {
            "transaction_id": txn.transaction_id,
            "account_id": txn.account_id,
            # Raw signed amount (negative = money out).
            "amount": txn.amount,
            "currency": txn.iso_currency_code or "USD",
            # Prefer the enriched merchant name; fall back to raw name.
            "merchant": txn.merchant_name or txn.name,
            # Plaid's personal-finance category (e.g., "FOOD_AND_DRINK").
            "category": txn.personal_finance_category.primary if txn.personal_finance_category else None,
            "date": txn.date.isoformat(),
            "is_pending": txn.pending,
        }

    async def sync(self) -> int:
        """Fetch new transactions using Plaid's cursor-based sync endpoint.

        Plaid ``transactions/sync`` works like a changelog:
            - On the first call the cursor is empty, so Plaid returns the full
              initial set of recent transactions.
            - On subsequent calls we pass the cursor from the previous
              response, and Plaid returns only *added*, *modified*, and
              *removed* transactions since that point.

        This approach is more efficient than date-range polling and avoids
        duplicates.  All three transaction lists are processed:
            - **added**: new transactions → ``finance.transaction.new``
            - **modified**: updated transactions (e.g. pending→cleared with a
              different final amount) → ``finance.transaction.modified``
            - **removed**: reversed/refunded transactions →
              ``finance.transaction.removed``

        Returns the number of transaction events published.
        """
        if not self._client:
            return 0

        count = 0
        # Retrieve the opaque cursor saved from the last successful sync.
        cursor = self.get_sync_cursor()

        for token in self._access_tokens:
            try:
                from plaid.model.transactions_sync_request import TransactionsSyncRequest

                # Build the sync request.  An empty cursor triggers a full
                # initial fetch; a non-empty cursor triggers incremental sync.
                request = TransactionsSyncRequest(
                    access_token=token,
                    cursor=cursor or "",
                )
                response = self._client.transactions_sync(request)

                # ---- New Transactions ----
                for txn in response.added:
                    payload = self._normalize_transaction(txn)
                    # Transactions above the configured dollar threshold are
                    # promoted to "high" priority for anomaly alerting.
                    amount = abs(txn.amount)
                    priority = "high" if amount >= self._large_threshold else "normal"
                    await self.publish_event(
                        "finance.transaction.new", payload, priority=priority,
                    )
                    count += 1

                # ---- Modified Transactions ----
                # Pending transactions that clear with a different final
                # amount, or any other server-side corrections.
                for txn in response.modified:
                    payload = self._normalize_transaction(txn)
                    amount = abs(txn.amount)
                    priority = "high" if amount >= self._large_threshold else "normal"
                    await self.publish_event(
                        "finance.transaction.modified", payload, priority=priority,
                    )
                    count += 1

                # ---- Removed Transactions ----
                # Reversed or refunded transactions.  Plaid only provides
                # the transaction_id for removals (no amount/merchant/etc.).
                for txn in response.removed:
                    payload = {
                        "transaction_id": txn.transaction_id,
                    }
                    await self.publish_event(
                        "finance.transaction.removed", payload, priority="low",
                    )
                    count += 1

                # ---- Cursor Persistence for Incremental Sync ----
                # Persist the server-provided cursor so the next sync cycle
                # picks up only new activity.
                if response.next_cursor:
                    self.set_sync_cursor(response.next_cursor)

            except Exception as e:
                logger.error("Sync error for token: %s", e)

        return count

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """No write actions are supported -- financial data is read-only."""
        raise ValueError("Finance connector is read-only")

    async def health_check(self) -> dict[str, Any]:
        """Return status based on whether the client and tokens are present.

        A full health check would call Plaid's ``/item/get`` endpoint for each
        token; for now we only verify local configuration.
        """
        if self._client and self._access_tokens:
            return {"status": "ok", "connector": self.CONNECTOR_ID,
                    "accounts": len(self._access_tokens)}
        return {"status": "error", "details": "Not configured"}
