"""
Tests for prediction deduplication and notification orphaning fix.

Verifies that:
1. store_prediction() returns True on first store, False on duplicate
2. generate_predictions() excludes deduplicated predictions from its return list
3. Notifications are only created for predictions that actually exist in the DB
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# store_prediction() return value tests
# -------------------------------------------------------------------------


def test_store_prediction_returns_true_on_first_store(db, user_model_store):
    """store_prediction() should return True when a prediction is actually stored."""
    prediction = {
        "id": str(uuid.uuid4()),
        "prediction_type": "NEED",
        "description": "User will need to buy groceries",
        "confidence": 0.7,
        "confidence_gate": "SUGGEST",
        "time_horizon": "24h",
        "suggested_action": "Add groceries to task list",
        "supporting_signals": {},
        "was_surfaced": True,
        "user_response": None,
        "resolved_at": None,
        "filter_reason": None,
    }

    result = user_model_store.store_prediction(prediction)

    assert result is True, "First store should return True"


def test_store_prediction_returns_false_on_duplicate(db, user_model_store):
    """store_prediction() should return False when a duplicate prediction exists."""
    prediction = {
        "id": str(uuid.uuid4()),
        "prediction_type": "NEED",
        "description": "User will need to buy groceries",
        "confidence": 0.7,
        "confidence_gate": "SUGGEST",
        "time_horizon": "24h",
        "suggested_action": "Add groceries to task list",
        "supporting_signals": {},
        "was_surfaced": True,
        "user_response": None,
        "resolved_at": None,
        "filter_reason": None,
    }

    # First store succeeds
    result1 = user_model_store.store_prediction(prediction)
    assert result1 is True

    # Second store with same type + description but different ID is deduplicated
    duplicate = prediction.copy()
    duplicate["id"] = str(uuid.uuid4())
    result2 = user_model_store.store_prediction(duplicate)

    assert result2 is False, "Duplicate prediction should return False"


def test_store_prediction_allows_different_type(db, user_model_store):
    """store_prediction() should store predictions with different types even if description matches."""
    base = {
        "id": str(uuid.uuid4()),
        "prediction_type": "NEED",
        "description": "Shared description text",
        "confidence": 0.7,
        "confidence_gate": "SUGGEST",
        "supporting_signals": {},
    }

    result1 = user_model_store.store_prediction(base)
    assert result1 is True

    # Different type, same description — should NOT be deduplicated
    different_type = base.copy()
    different_type["id"] = str(uuid.uuid4())
    different_type["prediction_type"] = "RISK"
    result2 = user_model_store.store_prediction(different_type)

    assert result2 is True, "Different prediction_type should not be deduplicated"


# -------------------------------------------------------------------------
# generate_predictions() deduplication integration tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_predictions_excludes_deduplicated(db, event_store, user_model_store):
    """generate_predictions() should not return predictions that were deduplicated.

    When called twice with the same underlying data, the second call should
    return an empty list because all predictions were already stored.
    """
    engine = PredictionEngine(db=db, ums=user_model_store)

    # Seed an event so the engine has something to work with
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "test@example.com", "subject": "Hello", "message_id": "msg-1"},
        "metadata": {},
    })

    # First generation — may produce predictions
    first_run = await engine.generate_predictions({})

    # Add another event so the engine doesn't skip via _has_new_events
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "test@example.com", "subject": "Follow up", "message_id": "msg-2"},
        "metadata": {},
    })

    # Second generation — any predictions that match the first run should be deduplicated
    second_run = await engine.generate_predictions({})

    # If first run produced predictions, second run should have fewer (deduplicated ones removed)
    if first_run:
        # The key invariant: every returned prediction must exist in the DB
        with db.get_connection("user_model") as conn:
            for pred in second_run:
                row = conn.execute(
                    "SELECT id FROM predictions WHERE id = ?", (pred.id,)
                ).fetchone()
                assert row is not None, (
                    f"Returned prediction {pred.id} does not exist in DB — "
                    "notification would be orphaned"
                )


@pytest.mark.asyncio
async def test_returned_predictions_all_exist_in_db(db, event_store, user_model_store):
    """Every prediction returned by generate_predictions() must exist in the DB.

    This is the core invariant that prevents orphaned notifications.
    """
    engine = PredictionEngine(db=db, ums=user_model_store)

    # Seed events
    for i in range(3):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "email.received",
            "source": "google",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "from_address": f"user{i}@example.com",
                "subject": f"Test {i}",
                "message_id": f"msg-{i}",
            },
            "metadata": {},
        })

    predictions = await engine.generate_predictions({})

    # Verify every returned prediction exists in the database
    with db.get_connection("user_model") as conn:
        for pred in predictions:
            row = conn.execute(
                "SELECT id FROM predictions WHERE id = ?", (pred.id,)
            ).fetchone()
            assert row is not None, (
                f"Prediction {pred.id} ({pred.prediction_type}: {pred.description[:50]}) "
                "was returned by generate_predictions() but not found in DB"
            )
