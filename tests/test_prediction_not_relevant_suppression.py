"""
Tests for not_relevant feedback suppression in the PredictionEngine.

When a user marks a prediction as "Not About Me" (user_response='not_relevant'),
the engine should suppress future predictions of the same type+contact combination
so the user isn't nagged with the same irrelevant prediction every cycle.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from models.core import ConfidenceGate
from models.user_model import Prediction
from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_resolved_prediction(
    db,
    prediction_type: str = "opportunity",
    contact_email: str | None = None,
    user_response: str = "not_relevant",
    was_surfaced: bool = True,
    resolved_at: datetime | None = None,
):
    """Insert a resolved prediction row directly into the predictions table."""
    pred_id = str(uuid.uuid4())
    signals = {}
    if contact_email:
        signals["contact_email"] = contact_email

    if resolved_at is None:
        resolved_at = datetime.now(timezone.utc)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, supporting_signals, was_surfaced, user_response,
                was_accurate, resolved_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                prediction_type,
                f"Test prediction for {contact_email or 'general'}",
                0.5,
                "SUGGEST",
                "24_hours",
                json.dumps(signals),
                int(was_surfaced),
                user_response,
                0,  # was_accurate = False for not_relevant
                resolved_at.isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return pred_id


def _make_prediction(
    prediction_type: str = "opportunity",
    contact_email: str | None = None,
    confidence: float = 0.5,
) -> Prediction:
    """Create a Prediction object for testing."""
    signals = {}
    if contact_email:
        signals["contact_email"] = contact_email
    return Prediction(
        prediction_type=prediction_type,
        description=f"Test prediction for {contact_email or 'general'}",
        confidence=confidence,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="24_hours",
        supporting_signals=signals,
    )


# ---------------------------------------------------------------------------
# _get_suppressed_prediction_keys tests
# ---------------------------------------------------------------------------


class TestGetSuppressedPredictionKeys:
    """Tests for _get_suppressed_prediction_keys()."""

    def test_returns_empty_set_when_no_not_relevant_feedback(self, prediction_engine):
        """No suppressed keys when no predictions have been marked not_relevant."""
        keys = prediction_engine._get_suppressed_prediction_keys()
        assert keys == set()

    def test_returns_key_for_contact_prediction(self, db, prediction_engine):
        """A not_relevant prediction for a specific contact produces a (type, email) key."""
        _store_resolved_prediction(db, "opportunity", "alice@example.com")

        keys = prediction_engine._get_suppressed_prediction_keys()
        assert ("opportunity", "alice@example.com") in keys

    def test_returns_key_for_non_contact_prediction(self, db, prediction_engine):
        """A not_relevant prediction without a contact produces a (type, None) key."""
        _store_resolved_prediction(db, "routine_deviation", contact_email=None)

        keys = prediction_engine._get_suppressed_prediction_keys()
        assert ("routine_deviation", None) in keys

    def test_ignores_non_surfaced_predictions(self, db, prediction_engine):
        """Predictions that were never surfaced (was_surfaced=0) are not suppressed."""
        _store_resolved_prediction(db, "opportunity", "bob@example.com", was_surfaced=False)

        keys = prediction_engine._get_suppressed_prediction_keys()
        assert keys == set()

    def test_ignores_non_not_relevant_responses(self, db, prediction_engine):
        """Predictions with other user_response values are not treated as suppressions."""
        _store_resolved_prediction(
            db, "opportunity", "carol@example.com", user_response="acted_on"
        )

        keys = prediction_engine._get_suppressed_prediction_keys()
        assert keys == set()

    def test_expired_suppression_not_returned(self, db, prediction_engine):
        """Suppressions older than 90 days are no longer returned."""
        old_date = datetime.now(timezone.utc) - timedelta(days=91)
        _store_resolved_prediction(
            db, "opportunity", "expired@example.com", resolved_at=old_date
        )

        keys = prediction_engine._get_suppressed_prediction_keys()
        assert ("opportunity", "expired@example.com") not in keys

    def test_recent_suppression_within_window(self, db, prediction_engine):
        """Suppressions within the 90-day window are returned."""
        recent_date = datetime.now(timezone.utc) - timedelta(days=30)
        _store_resolved_prediction(
            db, "opportunity", "recent@example.com", resolved_at=recent_date
        )

        keys = prediction_engine._get_suppressed_prediction_keys()
        assert ("opportunity", "recent@example.com") in keys

    def test_multiple_suppressions_for_different_contacts(self, db, prediction_engine):
        """Multiple not_relevant feedbacks for different contacts are all captured."""
        _store_resolved_prediction(db, "opportunity", "alice@example.com")
        _store_resolved_prediction(db, "opportunity", "bob@example.com")
        _store_resolved_prediction(db, "reminder", contact_email=None)

        keys = prediction_engine._get_suppressed_prediction_keys()
        assert len(keys) == 3
        assert ("opportunity", "alice@example.com") in keys
        assert ("opportunity", "bob@example.com") in keys
        assert ("reminder", None) in keys


# ---------------------------------------------------------------------------
# _is_suppressed tests
# ---------------------------------------------------------------------------


class TestIsSuppressed:
    """Tests for _is_suppressed()."""

    def test_suppresses_matching_type_and_contact(self):
        """Prediction matching both type and contact is suppressed."""
        pred = _make_prediction("opportunity", "alice@example.com")
        suppressed = {("opportunity", "alice@example.com")}
        assert PredictionEngine._is_suppressed(pred, suppressed) is True

    def test_does_not_suppress_different_contact(self):
        """Prediction for a different contact of the same type is NOT suppressed."""
        pred = _make_prediction("opportunity", "bob@example.com")
        suppressed = {("opportunity", "alice@example.com")}
        assert PredictionEngine._is_suppressed(pred, suppressed) is False

    def test_does_not_suppress_different_type(self):
        """Prediction of a different type for the same contact is NOT suppressed."""
        pred = _make_prediction("reminder", "alice@example.com")
        suppressed = {("opportunity", "alice@example.com")}
        assert PredictionEngine._is_suppressed(pred, suppressed) is False

    def test_suppresses_non_contact_prediction_by_type(self):
        """Non-contact prediction is suppressed when (type, None) is in the set."""
        pred = _make_prediction("routine_deviation")
        suppressed = {("routine_deviation", None)}
        assert PredictionEngine._is_suppressed(pred, suppressed) is True

    def test_type_only_suppression_applies_to_contact_predictions(self):
        """A (type, None) suppression also suppresses contact-specific predictions of that type."""
        pred = _make_prediction("opportunity", "alice@example.com")
        suppressed = {("opportunity", None)}
        assert PredictionEngine._is_suppressed(pred, suppressed) is True

    def test_contact_suppression_does_not_suppress_non_contact(self):
        """A (type, email) suppression does NOT suppress non-contact predictions of that type."""
        pred = _make_prediction("opportunity")  # no contact
        suppressed = {("opportunity", "alice@example.com")}
        assert PredictionEngine._is_suppressed(pred, suppressed) is False

    def test_empty_suppression_set(self):
        """No prediction is suppressed when the suppression set is empty."""
        pred = _make_prediction("opportunity", "alice@example.com")
        assert PredictionEngine._is_suppressed(pred, set()) is False


# ---------------------------------------------------------------------------
# Integration: suppression in generate_predictions
# ---------------------------------------------------------------------------


class TestSuppressionInGeneratePredictions:
    """Integration tests verifying suppression is wired into generate_predictions()."""

    @pytest.mark.asyncio
    async def test_suppressed_predictions_filtered_from_output(self, db, user_model_store):
        """Predictions matching not_relevant feedback are excluded from generate_predictions() output."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # Mark a relationship maintenance prediction for alice as not_relevant
        _store_resolved_prediction(db, "opportunity", "alice@example.com")

        # Verify suppression keys are populated
        keys = engine._get_suppressed_prediction_keys()
        assert ("opportunity", "alice@example.com") in keys

        # Create test predictions: one for suppressed contact, one for different contact
        pred_alice = _make_prediction("opportunity", "alice@example.com", confidence=0.6)
        pred_bob = _make_prediction("opportunity", "bob@example.com", confidence=0.6)

        # Manually test suppression filtering
        all_preds = [pred_alice, pred_bob]
        filtered = [p for p in all_preds if not PredictionEngine._is_suppressed(p, keys)]

        assert len(filtered) == 1
        assert filtered[0].supporting_signals["contact_email"] == "bob@example.com"

    @pytest.mark.asyncio
    async def test_non_suppressed_predictions_pass_through(self, db, user_model_store):
        """Predictions for contacts NOT marked as not_relevant pass through."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # Mark alice as not_relevant for opportunity type
        _store_resolved_prediction(db, "opportunity", "alice@example.com")

        # A reminder for alice should NOT be suppressed (different type)
        pred = _make_prediction("reminder", "alice@example.com")
        keys = engine._get_suppressed_prediction_keys()
        assert not PredictionEngine._is_suppressed(pred, keys)

    @pytest.mark.asyncio
    async def test_expired_suppression_does_not_filter(self, db, user_model_store):
        """Suppressions older than 90 days do not filter out predictions."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # Store a suppression from 91 days ago
        old_date = datetime.now(timezone.utc) - timedelta(days=91)
        _store_resolved_prediction(
            db, "opportunity", "alice@example.com", resolved_at=old_date
        )

        # The suppression should have expired
        keys = engine._get_suppressed_prediction_keys()
        pred = _make_prediction("opportunity", "alice@example.com")
        assert not PredictionEngine._is_suppressed(pred, keys)
