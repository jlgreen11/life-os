"""
Tests for connector health monitoring predictions.

Verifies that the PredictionEngine surfaces risk predictions when enabled
connectors are stuck in an error state, helping users discover broken
integrations (e.g., expired OAuth tokens, authentication failures) before
they silently degrade the system.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from models.core import ConfidenceGate
from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _insert_connector_state(db, connector_id, *, status="active", enabled=1,
                            error_count=0, last_error=None, last_sync=None,
                            updated_at=None):
    """Insert a row into the connector_state table for testing."""
    if updated_at is None:
        updated_at = datetime.now(timezone.utc).isoformat()
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO connector_state
               (connector_id, status, enabled, last_sync, error_count, last_error, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (connector_id, status, enabled, last_sync, error_count, last_error, updated_at),
        )


def _insert_existing_prediction(db, connector_id, *, days_ago=0):
    """Insert a deduplication marker — an existing connector health prediction."""
    created_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, supporting_signals, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "risk",
                f"{connector_id} connector has been failing",
                0.95,
                "default",
                "this_week",
                json.dumps({
                    "prediction_source": "connector_health",
                    "connector_id": connector_id,
                    "error_count": 5,
                    "last_error": "Auth failed",
                }),
                created_at,
            ),
        )


# -------------------------------------------------------------------------
# Core behavior tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_broken_connector_produces_risk_prediction(db, user_model_store):
    """An enabled connector with error status and error_count >= 3 should produce a risk prediction."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")
    last_sync = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()

    _insert_connector_state(
        db, "google",
        status="error", enabled=1, error_count=10,
        last_error="Authentication failed", last_sync=last_sync,
    )

    predictions = await engine._check_connector_health({})

    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.prediction_type == "risk"
    assert pred.confidence == 0.95
    assert pred.confidence_gate == ConfidenceGate.DEFAULT
    assert "google" in pred.description
    assert "Authentication failed" in pred.description
    assert "5 day" in pred.description
    assert pred.supporting_signals["connector_id"] == "google"
    assert pred.supporting_signals["error_count"] == 10
    assert pred.supporting_signals["prediction_source"] == "connector_health"
    assert pred.suggested_action is not None
    assert "google" in pred.suggested_action


@pytest.mark.asyncio
async def test_disabled_connector_is_ignored(db, user_model_store):
    """Disabled connectors should not produce predictions even if in error state."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    _insert_connector_state(
        db, "proton_mail",
        status="error", enabled=0, error_count=50,
        last_error="Connection refused",
    )

    predictions = await engine._check_connector_health({})
    assert predictions == []


@pytest.mark.asyncio
async def test_low_error_count_is_ignored(db, user_model_store):
    """Connectors with error_count < 3 (transient errors) should not be flagged."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    _insert_connector_state(
        db, "signal",
        status="error", enabled=1, error_count=2,
        last_error="Timeout",
    )

    predictions = await engine._check_connector_health({})
    assert predictions == []


@pytest.mark.asyncio
async def test_healthy_connector_produces_no_prediction(db, user_model_store):
    """Active, healthy connectors should not generate any predictions."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    _insert_connector_state(
        db, "caldav",
        status="active", enabled=1, error_count=0,
        last_sync=datetime.now(timezone.utc).isoformat(),
    )

    predictions = await engine._check_connector_health({})
    assert predictions == []


@pytest.mark.asyncio
async def test_deduplication_prevents_re_alerting(db, user_model_store):
    """Should not re-alert for a connector that was already predicted in the last 7 days."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    _insert_connector_state(
        db, "google",
        status="error", enabled=1, error_count=10,
        last_error="Auth failed",
        last_sync=(datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
    )

    # Insert an existing prediction for this connector from 3 days ago
    _insert_existing_prediction(db, "google", days_ago=3)

    predictions = await engine._check_connector_health({})
    assert predictions == []


@pytest.mark.asyncio
async def test_old_prediction_allows_re_alerting(db, user_model_store):
    """After 7+ days, a new prediction should be generated for the same connector."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    _insert_connector_state(
        db, "google",
        status="error", enabled=1, error_count=15,
        last_error="Token expired",
        last_sync=(datetime.now(timezone.utc) - timedelta(days=14)).isoformat(),
    )

    # Insert an existing prediction from 8 days ago (older than 7-day window)
    _insert_existing_prediction(db, "google", days_ago=8)

    predictions = await engine._check_connector_health({})
    assert len(predictions) == 1
    assert predictions[0].supporting_signals["connector_id"] == "google"


@pytest.mark.asyncio
async def test_multiple_broken_connectors(db, user_model_store):
    """Multiple broken connectors should each produce their own prediction."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    for cid in ["google", "proton_mail", "signal"]:
        _insert_connector_state(
            db, cid,
            status="error", enabled=1, error_count=5,
            last_error=f"{cid} auth failed",
            last_sync=(datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
        )

    predictions = await engine._check_connector_health({})
    assert len(predictions) == 3
    connector_ids = {p.supporting_signals["connector_id"] for p in predictions}
    assert connector_ids == {"google", "proton_mail", "signal"}


@pytest.mark.asyncio
async def test_connector_health_with_null_last_sync(db, user_model_store):
    """Connectors with NULL last_sync should use updated_at for staleness calculation."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    updated_at = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    _insert_connector_state(
        db, "imessage",
        status="error", enabled=1, error_count=4,
        last_error="Database locked", last_sync=None,
        updated_at=updated_at,
    )

    predictions = await engine._check_connector_health({})
    assert len(predictions) == 1
    assert "2 day" in predictions[0].description


@pytest.mark.asyncio
async def test_db_error_is_caught_gracefully(db, user_model_store):
    """DB errors in connector_state query should be caught (fail-open), returning empty list."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Drop the connector_state table to force a DB error
    with db.get_connection("state") as conn:
        conn.execute("DROP TABLE connector_state")

    # Should not raise — fail-open returns empty list
    predictions = await engine._check_connector_health({})
    assert predictions == []


@pytest.mark.asyncio
async def test_connector_health_wired_into_generate_predictions(db, event_store, user_model_store):
    """Connector health check should run as part of generate_predictions when time-based trigger fires."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    _insert_connector_state(
        db, "google",
        status="error", enabled=1, error_count=10,
        last_error="Auth expired",
        last_sync=(datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
    )

    # Force time-based trigger by setting last run to long ago
    engine._last_time_based_run = datetime.now(timezone.utc) - timedelta(hours=1)

    # Add a dummy event so the engine doesn't skip entirely
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "system.heartbeat",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {},
        "metadata": {},
    })

    predictions = await engine.generate_predictions({})

    # Find our connector health prediction in the results
    connector_preds = [
        p for p in predictions
        if p.supporting_signals.get("prediction_source") == "connector_health"
    ]
    assert len(connector_preds) == 1
    assert connector_preds[0].supporting_signals["connector_id"] == "google"
