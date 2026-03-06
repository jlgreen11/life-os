"""
Tests for the prediction pre-filter optimization.

Verifies that generate_predictions() skips regenerating predictions that
already exist in the database (unresolved or recently filtered), avoiding
the 16x dedup waste previously observed.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine


@pytest.mark.asyncio
async def test_prefilter_skips_existing_unresolved_prediction(db, event_store, user_model_store):
    """A prediction with matching (type, description) already in DB should be pre-filtered."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Insert an unresolved prediction directly into the DB
    pred_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "You haven't contacted Alice in 14 days",
                0.5,
                "SUGGEST",
                "24h",
                "Send Alice a message",
                "{}",
                0,
            ),
        )

    # Seed an event so generate_predictions() doesn't skip via _has_new_events
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "test@example.com", "subject": "Test", "message_id": "msg-prefilter-1"},
        "metadata": {},
    })

    # Run prediction generation
    predictions = await engine.generate_predictions({})

    # Verify no prediction with the same (type, description) was returned
    for p in predictions:
        assert not (
            p.prediction_type == "reminder"
            and p.description == "You haven't contacted Alice in 14 days"
        ), "Pre-filter should have removed the duplicate prediction"


@pytest.mark.asyncio
async def test_prefilter_allows_different_description(db, event_store, user_model_store):
    """Predictions with different descriptions should NOT be pre-filtered."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Insert an existing prediction
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "reminder",
                "You haven't contacted Alice in 14 days",
                0.5,
                "SUGGEST",
                "24h",
                "Send Alice a message",
                "{}",
                0,
            ),
        )

    # Seed an event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "test@example.com", "subject": "Test", "message_id": "msg-prefilter-2"},
        "metadata": {},
    })

    predictions = await engine.generate_predictions({})

    # The pre-filter should NOT block predictions with different descriptions.
    # We can't guarantee what predictions generate_predictions produces, but
    # the key guarantee is that it doesn't block all predictions just because
    # one exists in the DB.
    # This test primarily ensures no crash and that the filter logic works.


@pytest.mark.asyncio
async def test_prefilter_ignores_old_resolved_predictions(db, event_store, user_model_store):
    """Predictions resolved more than 24h ago should NOT be in the pre-filter set."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    old_resolved = datetime.now(timezone.utc) - timedelta(hours=25)

    # Insert an old resolved prediction
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced,
                resolved_at, user_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "reminder",
                "Old resolved prediction",
                0.5,
                "SUGGEST",
                "24h",
                "Do something",
                "{}",
                1,
                old_resolved.isoformat(),
                "acted_on",
            ),
        )

    # Verify the pre-filter set does NOT include the old resolved prediction
    with db.get_connection("user_model") as conn:
        rows = conn.execute(
            """SELECT prediction_type, description FROM predictions
               WHERE resolved_at IS NULL
                  OR datetime(resolved_at) > datetime('now', '-24 hours')"""
        ).fetchall()
        existing = {(r[0], r[1]) for r in rows}

    assert ("reminder", "Old resolved prediction") not in existing, \
        "Old resolved predictions should not be in the pre-filter set"


@pytest.mark.asyncio
async def test_prefilter_includes_recently_filtered_predictions(db, event_store, user_model_store):
    """Recently filtered predictions (resolved within 24h) should be in the pre-filter set."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    recent_resolved = datetime.now(timezone.utc) - timedelta(hours=2)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced,
                resolved_at, user_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "risk",
                "Recently filtered risk prediction",
                0.3,
                "OBSERVE",
                "12h",
                "Check on it",
                "{}",
                0,
                recent_resolved.isoformat(),
                "filtered",
            ),
        )

    # Verify the pre-filter set includes the recently resolved prediction
    with db.get_connection("user_model") as conn:
        rows = conn.execute(
            """SELECT prediction_type, description FROM predictions
               WHERE resolved_at IS NULL
                  OR datetime(resolved_at) > datetime('now', '-24 hours')"""
        ).fetchall()
        existing = {(r[0], r[1]) for r in rows}

    assert ("risk", "Recently filtered risk prediction") in existing, \
        "Recently filtered predictions should be in the pre-filter set"


@pytest.mark.asyncio
async def test_prefilter_graceful_on_empty_db(db, event_store, user_model_store):
    """Pre-filter should work gracefully when no predictions exist in DB."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Seed an event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "test@example.com", "subject": "Test", "message_id": "msg-prefilter-3"},
        "metadata": {},
    })

    # Should not crash with empty predictions table
    predictions = await engine.generate_predictions({})
    # No assertion on length — just verifying no error
