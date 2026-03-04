"""
Tests for semantic fact mutation endpoints: confirm and degraded-DB paths.

Covers POST /api/user-model/facts/{key}/confirm (fact confirmation) and
the degraded-DB error paths for both PATCH (correct) and POST (confirm).

Correction happy-path tests live in tests/web/test_fact_correction.py;
this file adds the confirmation endpoint coverage and degraded-DB
resilience checks that were previously missing.

Uses a real DatabaseManager + UserModelStore (from conftest fixtures) so
that actual SQL state changes are verified end-to-end.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_life_os(db, user_model_store, event_bus):
    """Create a LifeOS-like object with real DB/stores and mocked services."""
    life_os = Mock()

    # Real database and stores
    life_os.db = db
    life_os.user_model_store = user_model_store

    # Mock event bus (from conftest — functional mock with publish tracking)
    life_os.event_bus = event_bus

    # Mock feedback collector with a trackable _store_feedback
    life_os.feedback_collector = Mock()
    life_os.feedback_collector._store_feedback = AsyncMock()

    # Mock signal extractor (needed for other user-model routes)
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})

    return life_os


@pytest.fixture()
def client(mock_life_os):
    """Create a FastAPI TestClient wired to the real-DB LifeOS mock."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


def _seed_fact(user_model_store, key="test_fact", category="preference",
               value="blue", confidence=0.8):
    """Insert a semantic fact via the store for test setup."""
    user_model_store.update_semantic_fact(
        key=key, category=category, value=value, confidence=confidence
    )


# ---------------------------------------------------------------------------
# POST /api/user-model/facts/{key}/confirm — Fact Confirmation
# ---------------------------------------------------------------------------

class TestConfirmFact:
    """Tests for the POST /api/user-model/facts/{key}/confirm endpoint."""

    def test_confirm_fact_bumps_confidence(self, client, user_model_store):
        """POST confirm increases confidence by +0.05 (0.5 -> 0.55)."""
        _seed_fact(user_model_store, key="confirm_bump", confidence=0.5)

        response = client.post("/api/user-model/facts/confirm_bump/confirm")

        assert response.status_code == 200
        data = response.json()
        assert data["old_confidence"] == 0.5
        assert data["new_confidence"] == 0.55
        assert data["status"] == "confirmed"

        # Verify DB state directly
        fact = user_model_store.get_semantic_fact("confirm_bump")
        assert fact["confidence"] == 0.55
        assert fact["times_confirmed"] >= 1

    def test_confirm_fact_caps_at_one(self, client, user_model_store):
        """POST confirm with confidence=0.98 caps at 1.0, never exceeds."""
        _seed_fact(user_model_store, key="cap_fact", confidence=0.98)

        response = client.post("/api/user-model/facts/cap_fact/confirm")

        assert response.status_code == 200
        data = response.json()
        assert data["new_confidence"] == 1.0

        fact = user_model_store.get_semantic_fact("cap_fact")
        assert fact["confidence"] == 1.0

    def test_confirm_fact_not_found(self, client):
        """POST confirm on a nonexistent key returns 404."""
        response = client.post("/api/user-model/facts/nonexistent_key/confirm")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_confirm_fact_with_reason(self, client, user_model_store):
        """POST confirm with a reason body succeeds and logs the reason."""
        _seed_fact(user_model_store, key="reason_fact", confidence=0.6)

        response = client.post(
            "/api/user-model/facts/reason_fact/confirm",
            json={"reason": "verified by user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "confirmed"
        assert data["new_confidence"] == 0.65

    def test_confirm_fact_increments_times_confirmed(self, client, user_model_store):
        """Multiple confirmations increment times_confirmed each time."""
        _seed_fact(user_model_store, key="multi_confirm", confidence=0.5)

        # First confirmation
        client.post("/api/user-model/facts/multi_confirm/confirm")
        fact = user_model_store.get_semantic_fact("multi_confirm")
        assert fact["times_confirmed"] >= 1

        # Second confirmation
        client.post("/api/user-model/facts/multi_confirm/confirm")
        fact = user_model_store.get_semantic_fact("multi_confirm")
        assert fact["times_confirmed"] >= 2
        assert fact["confidence"] == 0.6  # 0.5 + 0.05 + 0.05

    def test_confirm_fact_logs_to_feedback_collector(self, client, mock_life_os, user_model_store):
        """POST confirm logs the confirmation to the feedback collector."""
        _seed_fact(user_model_store, key="fb_confirm", confidence=0.7)

        client.post(
            "/api/user-model/facts/fb_confirm/confirm",
            json={"reason": "looks right"},
        )

        mock_life_os.feedback_collector._store_feedback.assert_called_once()
        call_args = mock_life_os.feedback_collector._store_feedback.call_args[0][0]
        assert call_args["action_type"] == "semantic_fact"
        assert call_args["feedback_type"] == "confirmed"
        assert call_args["context"]["fact_key"] == "fb_confirm"
        assert call_args["context"]["reason"] == "looks right"

    def test_confirm_fact_degraded_db(self, client, mock_life_os, user_model_store):
        """POST confirm returns 503 when user_model.db is degraded."""
        _seed_fact(user_model_store, key="degraded_confirm", confidence=0.5)

        # Simulate database corruption flag
        mock_life_os.db.user_model_degraded = True

        response = client.post("/api/user-model/facts/degraded_confirm/confirm")

        assert response.status_code == 503
        data = response.json()
        assert "temporarily unavailable" in data["error"].lower()


# ---------------------------------------------------------------------------
# PATCH /api/user-model/facts/{key} — Degraded DB path
# ---------------------------------------------------------------------------

class TestCorrectFactDegradedDB:
    """Test the degraded-DB error path for PATCH /api/user-model/facts/{key}."""

    def test_correct_fact_degraded_db(self, client, mock_life_os, user_model_store):
        """PATCH returns 503 when user_model.db is degraded."""
        _seed_fact(user_model_store, key="degraded_correct", confidence=0.8)

        # Simulate database corruption flag
        mock_life_os.db.user_model_degraded = True

        response = client.patch(
            "/api/user-model/facts/degraded_correct",
            json={"reason": "should fail"},
        )

        assert response.status_code == 503
        data = response.json()
        assert "temporarily unavailable" in data["error"].lower()
