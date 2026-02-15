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

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


class FinanceConnector(BaseConnector):

    CONNECTOR_ID = "finance"
    DISPLAY_NAME = "Finance (Plaid)"
    SYNC_INTERVAL_SECONDS = 3600  # Hourly

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        self._client = None
        self._access_tokens = config.get("access_tokens", [])
        self._large_threshold = config.get("large_transaction_threshold", 500)

    async def authenticate(self) -> bool:
        try:
            import plaid
            from plaid.api import plaid_api
            from plaid.model.products import Products

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
            print(f"[finance] Auth failed: {e}")
            return False

    async def sync(self) -> int:
        if not self._client:
            return 0

        count = 0
        cursor = self.get_sync_cursor()

        for token in self._access_tokens:
            try:
                from plaid.model.transactions_sync_request import TransactionsSyncRequest

                request = TransactionsSyncRequest(
                    access_token=token,
                    cursor=cursor or "",
                )
                response = self._client.transactions_sync(request)

                for txn in response.added:
                    amount = abs(txn.amount)

                    payload = {
                        "transaction_id": txn.transaction_id,
                        "account_id": txn.account_id,
                        "amount": txn.amount,
                        "currency": txn.iso_currency_code or "USD",
                        "merchant": txn.merchant_name or txn.name,
                        "category": txn.personal_finance_category.primary if txn.personal_finance_category else None,
                        "date": txn.date.isoformat(),
                        "is_pending": txn.pending,
                    }

                    # Determine priority based on amount
                    priority = "normal"
                    if amount >= self._large_threshold:
                        priority = "high"

                    await self.publish_event(
                        "finance.transaction.new", payload, priority=priority,
                    )
                    count += 1

                # Store cursor for incremental sync
                if response.next_cursor:
                    self.set_sync_cursor(response.next_cursor)

            except Exception as e:
                print(f"[finance] Sync error for token: {e}")

        return count

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("Finance connector is read-only")

    async def health_check(self) -> dict[str, Any]:
        if self._client and self._access_tokens:
            return {"status": "ok", "connector": self.CONNECTOR_ID,
                    "accounts": len(self._access_tokens)}
        return {"status": "error", "details": "Not configured"}
