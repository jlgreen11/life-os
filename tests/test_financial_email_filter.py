"""
Tests for financial/brokerage automated email filtering.

CONTEXT (iteration 171):
    The follow-up needs prediction was generating spam for automated financial
    emails like "Fidelity.Investments@mail.fidelity.com" sending trade confirmations.
    These transactional emails require no response and pollute the prediction feed.

IMPROVEMENT:
    Extended the _is_marketing_or_noreply filter with comprehensive financial/
    brokerage sender patterns to catch:
    - Brokerage firms (Fidelity, Schwab, Vanguard, E*TRADE, etc.)
    - Banks (Chase, Bank of America, Citi, Wells Fargo, etc.)
    - Payment processors (PayPal, Venmo, Stripe, Square)
    - Crypto exchanges (Coinbase, Binance, Kraken)
    - Credit bureaus (Experian, Equifax, TransUnion, Credit Karma)

EXPECTED IMPACT:
    - Eliminates prediction spam for trade confirmations, account alerts, etc.
    - Improves prediction quality by filtering out zero-value automated emails
    - Protects accuracy feedback loop from financial transaction noise
"""

from datetime import datetime, timedelta, timezone

from models.core import ConfidenceGate
from services.prediction_engine.engine import PredictionEngine


def test_fidelity_investments_filtered(db, user_model_store):
    """Fidelity.Investments@mail.fidelity.com should be filtered as automated financial sender."""
    engine = PredictionEngine(db, user_model_store)

    # Test the exact address from the diagnostic
    assert engine._is_marketing_or_noreply("Fidelity.Investments@mail.fidelity.com", {})
    assert engine._is_marketing_or_noreply("fidelity.investments@mail.fidelity.com", {})  # Case insensitive
    assert engine._is_marketing_or_noreply("FIDELITY.INVESTMENTS@MAIL.FIDELITY.COM", {})


def test_brokerage_firms_filtered(db, user_model_store):
    """All major brokerage firms should be filtered."""
    engine = PredictionEngine(db, user_model_store)

    brokerage_addresses = [
        "fidelity@fidelity.com",
        "schwab@schwab.com",
        "vanguard@vanguard.com",
        "etrade@etrade.com",
        "td.ameritrade@tdameritrade.com",
        "merrilledge@ml.com",
        "robinhood@robinhood.com",
        "wealthfront@wealthfront.com",
        "betterment@betterment.com",
    ]

    for addr in brokerage_addresses:
        assert engine._is_marketing_or_noreply(addr, {}), f"{addr} should be filtered"


def test_bank_alerts_filtered(db, user_model_store):
    """Bank alert systems should be filtered."""
    engine = PredictionEngine(db, user_model_store)

    bank_alert_addresses = [
        "chase.alerts@chase.com",
        "bankofamerica.alerts@bankofamerica.com",
        "citi.alerts@citi.com",
        "wellsfargo.alerts@wellsfargo.com",
        "usbank.alerts@usbank.com",
        "pnc.alerts@pnc.com",
    ]

    for addr in bank_alert_addresses:
        assert engine._is_marketing_or_noreply(addr, {}), f"{addr} should be filtered"


def test_payment_processors_filtered(db, user_model_store):
    """Payment processors like PayPal, Venmo, Stripe should be filtered."""
    engine = PredictionEngine(db, user_model_store)

    payment_addresses = [
        "paypal@paypal.com",
        "venmo@venmo.com",
        "stripe@stripe.com",
        "square@squareup.com",
    ]

    for addr in payment_addresses:
        assert engine._is_marketing_or_noreply(addr, {}), f"{addr} should be filtered"


def test_crypto_exchanges_filtered(db, user_model_store):
    """Cryptocurrency exchange automated emails should be filtered."""
    engine = PredictionEngine(db, user_model_store)

    crypto_addresses = [
        "coinbase@coinbase.com",
        "binance@binance.com",
        "kraken@kraken.com",
    ]

    for addr in crypto_addresses:
        assert engine._is_marketing_or_noreply(addr, {}), f"{addr} should be filtered"


def test_credit_bureaus_filtered(db, user_model_store):
    """Credit bureau and credit monitoring services should be filtered."""
    engine = PredictionEngine(db, user_model_store)

    credit_addresses = [
        "experian@experian.com",
        "equifax@equifax.com",
        "transunion@transunion.com",
        "creditkarma@creditkarma.com",
    ]

    for addr in credit_addresses:
        assert engine._is_marketing_or_noreply(addr, {}), f"{addr} should be filtered"


def test_personal_financial_contacts_not_filtered(db, user_model_store):
    """Personal emails with financial terms should NOT be filtered."""
    engine = PredictionEngine(db, user_model_store)

    # These should pass through - actual humans
    personal_addresses = [
        "john.fidelity@gmail.com",  # Surname
        "sarah.schwab@company.com",  # Surname
        "mike@myfidelity.biz",       # Personal domain containing "fidelity"
        "contact@verifidelity.com",  # Company with "fidelity" in name
        "jane.chase@example.com",    # Surname
    ]

    for addr in personal_addresses:
        assert not engine._is_marketing_or_noreply(addr, {}), f"{addr} should NOT be filtered (is human)"


async def test_follow_up_predictions_skip_financial_emails(db, user_model_store, event_store):
    """Follow-up needs should not generate predictions for financial automated emails."""
    engine = PredictionEngine(db, user_model_store)

    # Create unreplied financial email events
    now = datetime.now(timezone.utc)
    hours_ago_5 = now - timedelta(hours=5)

    financial_senders = [
        "Fidelity.Investments@mail.fidelity.com",
        "schwab@schwab.com",
        "chase.alerts@chase.com",
        "paypal@paypal.com",
    ]

    for sender in financial_senders:
        event_store.store_event({
            "id": f"email-{sender}-{int(hours_ago_5.timestamp())}",
            "type": "email.received",
            "source": "google",
            "timestamp": hours_ago_5.isoformat(),
            "priority": "normal",
            "payload": {
                "message_id": f"msg-{sender}",
                "from_address": sender,
                "subject": "Your trade confirmation is available",
                "snippet": "Your order has been executed",
            },
            "metadata": {"related_contacts": []},
        })

    # Generate predictions
    predictions = await engine._check_follow_up_needs({})

    # None of the financial emails should generate predictions
    financial_pred_count = sum(
        1 for p in predictions
        if any(sender in p.description for sender in financial_senders)
    )

    assert financial_pred_count == 0, "Financial automated emails should not generate follow-up predictions"


async def test_follow_up_predictions_allow_personal_finance_contacts(db, user_model_store, event_store):
    """Follow-up needs should still generate predictions for personal contacts in finance industry."""
    engine = PredictionEngine(db, user_model_store)

    # Create unreplied email from a personal contact who works in finance
    now = datetime.now(timezone.utc)
    hours_ago_5 = now - timedelta(hours=5)

    event_store.store_event({
        "id": "email-personal-finance-contact",
        "type": "email.received",
        "source": "google",
        "timestamp": hours_ago_5.isoformat(),
        "priority": "normal",
        "payload": {
            "message_id": "msg-personal-finance",
            "from_address": "john.advisor@financialfirm.com",
            "subject": "Following up on our meeting",
            "snippet": "Hey, wanted to circle back on the portfolio discussion",
        },
        "metadata": {"related_contacts": []},
    })

    # Generate predictions
    predictions = await engine._check_follow_up_needs({})

    # Should generate a prediction for the personal contact
    # Description now uses resolved contact name (email prefix as fallback)
    personal_pred = [
        p for p in predictions
        if "john.advisor" in p.description
    ]

    assert len(personal_pred) == 1, "Personal finance contacts should generate predictions"
    assert personal_pred[0].prediction_type == "reminder"
    assert personal_pred[0].confidence >= 0.3  # Should meet minimum threshold


async def test_fidelity_trade_confirmation_specific_case(db, user_model_store, event_store):
    """
    Regression test for the exact case from diagnostic output:
    "Unreplied message from Fidelity.Investments@mail.fidelity.com: 'Your trade confirmation is available'"
    """
    engine = PredictionEngine(db, user_model_store)

    # Recreate the exact scenario from the diagnostic
    now = datetime.now(timezone.utc)
    hours_ago_6 = now - timedelta(hours=6)
    hours_ago_15 = now - timedelta(hours=15)

    # Store multiple trade confirmation emails (as seen in diagnostic)
    for i, hours_delta in enumerate([6, 6, 15]):
        timestamp = now - timedelta(hours=hours_delta)
        event_store.store_event({
            "id": f"email-fidelity-{i}",
            "type": "email.received",
            "source": "google",
            "timestamp": timestamp.isoformat(),
            "priority": "normal",
            "payload": {
                "message_id": f"msg-fidelity-{i}",
                "from_address": "Fidelity.Investments@mail.fidelity.com",
                "subject": "Your trade confirmation is available",
                "snippet": "Your order for 100 shares has been executed",
            },
            "metadata": {"related_contacts": []},
        })

    # Generate predictions
    predictions = await engine._check_follow_up_needs({})

    # NONE of the Fidelity emails should generate predictions
    fidelity_preds = [
        p for p in predictions
        if "Fidelity.Investments@mail.fidelity.com" in p.description
    ]

    assert len(fidelity_preds) == 0, "Fidelity trade confirmations should be completely filtered out"
