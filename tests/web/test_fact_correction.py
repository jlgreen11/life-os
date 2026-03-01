"""
Tests for semantic fact correction and deletion API endpoints.

Covers PATCH /api/user-model/facts/{key} (fact correction) and
DELETE /api/user-model/facts/{key} (fact deletion) — the primary way
users correct AI-inferred facts about themselves.

These tests use a real DatabaseManager + UserModelStore (from conftest
fixtures) so they can verify actual DB state changes made by the raw SQL
in the PATCH/DELETE endpoints. External services (event_bus,
feedback_collector) are mocked.
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_life_os(db, user_model_store, event_bus):
    """Create a LifeOS-like object with real DB/stores and mocked services.

    The PATCH and DELETE fact endpoints use ``life_os.db.get_connection()``
    directly (raw SQL), so we need a real DatabaseManager. The
    UserModelStore is real too, for seeding facts and verifying state.
    External services are mocked to avoid side effects.
    """
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
    """Helper to insert a semantic fact via the store for test setup."""
    user_model_store.update_semantic_fact(
        key=key, category=category, value=value, confidence=confidence
    )


# ---------------------------------------------------------------------------
# PATCH /api/user-model/facts/{key} — Fact Correction
# ---------------------------------------------------------------------------

class TestCorrectFact:
    """Tests for the PATCH /api/user-model/facts/{key} endpoint."""

    def test_correct_fact_reduces_confidence(self, client, user_model_store):
        """PATCH reduces confidence by 0.30 (0.8 → 0.5)."""
        _seed_fact(user_model_store, key="fav_color", confidence=0.8)

        response = client.patch(
            "/api/user-model/facts/fav_color",
            json={"reason": "wrong color"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["old_confidence"] == 0.8
        assert data["new_confidence"] == 0.5

        # Verify DB state directly
        fact = user_model_store.get_semantic_fact("fav_color")
        assert fact["confidence"] == 0.5

    def test_correct_fact_confidence_floor_at_zero(self, client, user_model_store):
        """PATCH with low confidence floors at 0.0 (never negative)."""
        _seed_fact(user_model_store, key="low_fact", confidence=0.1)

        response = client.patch(
            "/api/user-model/facts/low_fact",
            json={"reason": "incorrect"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["new_confidence"] == 0.0

        fact = user_model_store.get_semantic_fact("low_fact")
        assert fact["confidence"] == 0.0

    def test_correct_fact_sets_is_user_corrected(self, client, user_model_store):
        """PATCH sets is_user_corrected = 1 in the database."""
        _seed_fact(user_model_store, key="corrected_fact", confidence=0.7)

        response = client.patch(
            "/api/user-model/facts/corrected_fact",
            json={"reason": "not accurate"},
        )

        assert response.status_code == 200
        fact = user_model_store.get_semantic_fact("corrected_fact")
        assert fact["is_user_corrected"] == 1

    def test_correct_fact_with_corrected_value(self, client, user_model_store):
        """PATCH with corrected_value updates the fact value."""
        _seed_fact(user_model_store, key="fav_food", value="pizza", confidence=0.8)

        response = client.patch(
            "/api/user-model/facts/fav_food",
            json={"corrected_value": "sushi", "reason": "I prefer sushi"},
        )

        assert response.status_code == 200
        fact = user_model_store.get_semantic_fact("fav_food")
        assert fact["value"] == "sushi"

    def test_correct_fact_without_corrected_value(self, client, user_model_store):
        """PATCH without corrected_value keeps original value but reduces confidence."""
        _seed_fact(user_model_store, key="fav_sport", value="tennis", confidence=0.8)

        response = client.patch(
            "/api/user-model/facts/fav_sport",
            json={"reason": "not sure about this"},
        )

        assert response.status_code == 200
        fact = user_model_store.get_semantic_fact("fav_sport")
        assert fact["value"] == "tennis"  # value unchanged
        assert fact["confidence"] == 0.5  # confidence reduced

    def test_correct_fact_returns_404_for_missing_key(self, client):
        """PATCH returns 404 when the fact key doesn't exist."""
        response = client.patch(
            "/api/user-model/facts/nonexistent_key",
            json={"reason": "does not exist"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_correct_fact_response_includes_old_and_new_confidence(self, client, user_model_store):
        """PATCH response body includes old_confidence and new_confidence."""
        _seed_fact(user_model_store, key="resp_fact", confidence=0.9)

        response = client.patch(
            "/api/user-model/facts/resp_fact",
            json={"reason": "checking response"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "old_confidence" in data
        assert "new_confidence" in data
        assert data["status"] == "corrected"
        assert "fact" in data

    def test_correct_fact_logs_to_feedback_collector(self, client, mock_life_os, user_model_store):
        """PATCH logs the correction to the feedback collector."""
        _seed_fact(user_model_store, key="fb_fact", confidence=0.7)

        client.patch(
            "/api/user-model/facts/fb_fact",
            json={"reason": "wrong info"},
        )

        mock_life_os.feedback_collector._store_feedback.assert_called_once()
        call_args = mock_life_os.feedback_collector._store_feedback.call_args[0][0]
        assert call_args["action_type"] == "semantic_fact"
        assert call_args["feedback_type"] == "corrected"
        assert call_args["context"]["fact_key"] == "fb_fact"

    def test_correct_fact_with_not_about_me_reason(self, client, user_model_store):
        """PATCH with 'Not about me' reason (mimics the UI notAboutMeFact flow)."""
        _seed_fact(user_model_store, key="wrong_person_fact", confidence=0.6)

        response = client.patch(
            "/api/user-model/facts/wrong_person_fact",
            json={"reason": "Not about me — this fact refers to someone else"},
        )

        assert response.status_code == 200
        fact = user_model_store.get_semantic_fact("wrong_person_fact")
        assert fact["is_user_corrected"] == 1
        assert fact["confidence"] == 0.3

    def test_inference_skips_user_corrected_fact(self, client, user_model_store):
        """After PATCH sets is_user_corrected=1, inference cannot overwrite the fact.

        This is a critical invariant: the update_semantic_fact() method skips
        facts where is_user_corrected=1, ensuring user corrections persist.
        """
        _seed_fact(user_model_store, key="guarded_fact", value="correct_value", confidence=0.8)

        # User corrects the fact
        response = client.patch(
            "/api/user-model/facts/guarded_fact",
            json={"corrected_value": "user_value", "reason": "I know better"},
        )
        assert response.status_code == 200

        # Attempt inference update — should be silently skipped
        user_model_store.update_semantic_fact(
            key="guarded_fact",
            category="preference",
            value="inferred_value",
            confidence=0.9,
        )

        # Verify the user correction is preserved
        fact = user_model_store.get_semantic_fact("guarded_fact")
        assert fact["value"] == "user_value"
        assert fact["is_user_corrected"] == 1


# ---------------------------------------------------------------------------
# DELETE /api/user-model/facts/{key} — Fact Deletion
# ---------------------------------------------------------------------------

class TestDeleteFact:
    """Tests for the DELETE /api/user-model/facts/{key} endpoint."""

    def test_delete_fact_removes_from_db(self, client, user_model_store):
        """DELETE removes the fact from the database."""
        _seed_fact(user_model_store, key="to_delete", confidence=0.7)

        # Verify fact exists first
        assert user_model_store.get_semantic_fact("to_delete") is not None

        response = client.delete("/api/user-model/facts/to_delete")
        assert response.status_code == 200

        # Verify fact is gone
        assert user_model_store.get_semantic_fact("to_delete") is None

    def test_delete_fact_nonexistent_key(self, client):
        """DELETE on a nonexistent key returns 200 (current behavior — silent success).

        Note: This contrasts with PATCH which returns 404 for missing keys.
        This test documents the current behavior.
        """
        response = client.delete("/api/user-model/facts/does_not_exist")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    def test_delete_fact_response_status(self, client, user_model_store):
        """DELETE response body is {"status": "deleted"}."""
        _seed_fact(user_model_store, key="del_resp", confidence=0.5)

        response = client.delete("/api/user-model/facts/del_resp")
        assert response.status_code == 200
        assert response.json() == {"status": "deleted"}


# ---------------------------------------------------------------------------
# Integration — Combined Operations
# ---------------------------------------------------------------------------

class TestFactIntegration:
    """Integration tests combining correction and deletion operations."""

    def test_correct_then_delete_fact(self, client, user_model_store):
        """PATCH a fact, then DELETE it — both operations succeed."""
        _seed_fact(user_model_store, key="combo_fact", confidence=0.8)

        # Correct it first
        patch_response = client.patch(
            "/api/user-model/facts/combo_fact",
            json={"corrected_value": "new_value", "reason": "wrong"},
        )
        assert patch_response.status_code == 200

        # Now delete it
        delete_response = client.delete("/api/user-model/facts/combo_fact")
        assert delete_response.status_code == 200

        # Verify it's truly gone
        assert user_model_store.get_semantic_fact("combo_fact") is None

    def test_get_facts_after_correction_shows_updated_confidence(self, client, user_model_store):
        """After PATCH, GET /api/user-model/facts reflects the reduced confidence."""
        _seed_fact(user_model_store, key="visible_fact", confidence=0.8)

        # Correct the fact (reduces confidence 0.8 → 0.5)
        client.patch(
            "/api/user-model/facts/visible_fact",
            json={"reason": "inaccurate"},
        )

        # GET facts — the corrected fact should appear with reduced confidence
        response = client.get("/api/user-model/facts")
        assert response.status_code == 200
        facts = response.json()["facts"]

        corrected = [f for f in facts if f["key"] == "visible_fact"]
        assert len(corrected) == 1
        assert corrected[0]["confidence"] == 0.5
