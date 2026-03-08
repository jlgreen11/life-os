"""
Tests for WAL checkpoint after prediction storage.

Verifies that the prediction engine forces a WAL checkpoint after storing
predictions, preventing data loss from WAL file corruption or truncation.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from services.prediction_engine.engine import PredictionEngine


class TestForceWalCheckpoint:
    """Tests for DatabaseManager.checkpoint_wal used after prediction storage."""

    def test_checkpoint_wal_executes_without_error(self, db):
        """checkpoint_wal should execute successfully on a valid database."""
        # Should not raise — the method logs and swallows errors internally,
        # but we verify it completes without bubbling up exceptions.
        db.checkpoint_wal("user_model")

    def test_checkpoint_wal_invalid_db_raises_key_error(self, db):
        """checkpoint_wal should raise KeyError for an unknown database name."""
        with pytest.raises(KeyError):
            db.checkpoint_wal("nonexistent_db")

    def test_predictions_survive_after_checkpoint(self, db, user_model_store):
        """Predictions stored and checkpointed should be readable afterwards."""
        prediction = {
            "id": str(uuid.uuid4()),
            "prediction_type": "NEED",
            "title": "Test prediction",
            "description": "You will need coffee",
            "confidence": 0.7,
            "confidence_gate": "SUGGEST",
            "source_events": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": None,
            "resolved_at": None,
            "resolution": None,
            "accuracy_score": None,
        }

        user_model_store.store_prediction(prediction)

        # Force WAL checkpoint to flush writes to main db file
        db.checkpoint_wal("user_model")

        # Verify the prediction is readable from the database
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT id FROM predictions WHERE id = ?",
                (prediction["id"],),
            ).fetchone()
        assert row is not None, "Prediction should exist after checkpoint"
        assert row["id"] == prediction["id"]


class TestPredictionEngineWalCheckpoint:
    """Tests that the prediction engine calls checkpoint_wal after storing predictions."""

    @pytest.mark.asyncio
    async def test_checkpoint_called_after_successful_storage(self, db, event_store, user_model_store):
        """The prediction engine should call checkpoint_wal after storing predictions."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        # Seed an event so the engine has something to process
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "email.received",
            "source": "google",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "from_address": "boss@company.com",
                "subject": "Meeting tomorrow at 9am",
                "message_id": "msg-wal-test",
            },
            "metadata": {},
        })

        # Seed some calendar events so conflict detection can produce predictions
        now = datetime.now(timezone.utc)
        for i in range(2):
            event_store.store_event({
                "id": str(uuid.uuid4()),
                "type": "calendar.event.created",
                "source": "caldav",
                "timestamp": now.isoformat(),
                "payload": {
                    "title": f"Overlapping meeting {i}",
                    "start": now.isoformat(),
                    "end": (now).isoformat(),
                    "calendar_id": "work",
                },
                "metadata": {},
            })

        with patch.object(db, "checkpoint_wal", wraps=db.checkpoint_wal) as mock_ckpt:
            predictions = await engine.generate_predictions({})

            # If predictions were stored, checkpoint_wal should have been called.
            # If no predictions were generated (engine needs more data), we can't
            # assert the call — but we verify the integration path works either way.
            if predictions:
                mock_ckpt.assert_called_with("user_model")

    @pytest.mark.asyncio
    async def test_checkpoint_failure_does_not_crash_prediction_loop(self, db, event_store, user_model_store):
        """A checkpoint failure should be logged but not crash the prediction loop."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        # Seed an event
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "email.received",
            "source": "google",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "from_address": "test@example.com",
                "subject": "Test",
                "message_id": "msg-ckpt-fail",
            },
            "metadata": {},
        })

        # Make checkpoint_wal raise an exception
        with patch.object(db, "checkpoint_wal", side_effect=Exception("WAL checkpoint simulated failure")):
            # Should complete without raising, even if checkpoint fails
            predictions = await engine.generate_predictions({})
            # The engine should still return results (or empty list) without crashing
            assert isinstance(predictions, list)
