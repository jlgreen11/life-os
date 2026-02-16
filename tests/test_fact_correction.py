"""
Life OS — Tests for Semantic Fact Correction Flow

Tests the user correction flow for semantic facts, which closes the feedback
loop by allowing the system to learn from negative signals (corrections) in
addition to positive signals (confirmations).

The correction flow:
    1. User identifies an incorrect semantic fact via UI
    2. PATCH /api/user-model/facts/{key} is called with optional corrected value
    3. is_user_corrected flag is set, confidence is reduced by 0.30
    4. Correction is logged to feedback_log for analytics
    5. Telemetry event is published for monitoring

Coverage:
    - Basic correction (flag + confidence reduction)
    - Correction with replacement value
    - Correction with reason/notes
    - 404 handling for non-existent facts
    - Confidence floor (never below 0.0)
    - Integration with feedback collector
    - Telemetry event publishing
"""

import json
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from pytest import approx

from main import LifeOS
from storage.database import DatabaseManager, UserModelStore
from web.app import create_web_app


@pytest.fixture
def client(db: DatabaseManager, user_model_store: UserModelStore):
    """Create a FastAPI test client with a minimal LifeOS instance."""
    # Create a minimal LifeOS instance for testing
    # We only need the components required for the correction endpoint
    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = user_model_store
    life_os.event_bus = None  # No event bus needed for basic tests
    life_os.feedback_collector = None  # We'll test feedback integration separately

    app = create_web_app(life_os)
    return TestClient(app)


class TestFactCorrectionEndpoint:
    """Test the PATCH /api/user-model/facts/{key} endpoint."""

    def test_correct_fact_basic(self, client, user_model_store: UserModelStore):
        """Test basic fact correction: sets flag and reduces confidence."""
        # Arrange: Create a semantic fact with moderate confidence
        user_model_store.update_semantic_fact(
            key="test_preference_email_style",
            category="communication_preference",
            value="formal",
            confidence=0.70,
        )

        # Act: Correct the fact without providing a replacement value
        response = client.patch(
            "/api/user-model/facts/test_preference_email_style",
            json={},  # No corrected value, just mark it wrong
        )

        # Assert: Returns 200 with updated fact
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "corrected"
        assert result["old_confidence"] == 0.70
        assert result["new_confidence"] == 0.40  # 0.70 - 0.30 = 0.40

        # Assert: Database record is updated correctly
        fact = result["fact"]
        assert fact["key"] == "test_preference_email_style"
        assert fact["is_user_corrected"] == 1
        assert fact["confidence"] == 0.40
        # Original value should be unchanged (we didn't provide a corrected value)
        assert json.loads(fact["value"]) == "formal"

    def test_correct_fact_with_replacement_value(self, client, user_model_store: UserModelStore):
        """Test correcting a fact and providing a replacement value."""
        # Arrange: Create a fact with wrong value
        user_model_store.update_semantic_fact(
            key="test_preference_notification_hour",
            category="notification_preference",
            value=9,  # Wrong: user prefers 8 AM
            confidence=0.65,
        )

        # Act: Correct the fact and provide the right value
        response = client.patch(
            "/api/user-model/facts/test_preference_notification_hour",
            json={"corrected_value": 8},
        )

        # Assert: Value is replaced
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "corrected"
        assert result["new_confidence"] == 0.35  # 0.65 - 0.30 = 0.35

        fact = result["fact"]
        assert json.loads(fact["value"]) == 8  # Updated to corrected value
        assert fact["is_user_corrected"] == 1

    def test_correct_fact_with_reason(self, client, user_model_store: UserModelStore):
        """Test providing a reason/explanation for the correction."""
        # Arrange
        user_model_store.update_semantic_fact(
            key="test_relationship_priority_contact",
            category="relationship",
            value="john@example.com",
            confidence=0.80,
        )

        # Act: Include a reason for the correction
        response = client.patch(
            "/api/user-model/facts/test_relationship_priority_contact",
            json={
                "corrected_value": "jane@example.com",
                "reason": "John is no longer my manager, Jane is.",
            },
        )

        # Assert: Correction succeeds
        assert response.status_code == 200
        result = response.json()
        assert result["new_confidence"] == 0.50  # 0.80 - 0.30

        # The reason is logged but not stored in the fact itself
        # (it goes to feedback_log via feedback_collector in the real system)

    def test_correct_fact_confidence_floor(self, client, user_model_store: UserModelStore):
        """Test that confidence never goes below 0.0 after correction."""
        # Arrange: Create a low-confidence fact
        user_model_store.update_semantic_fact(
            key="test_low_confidence_fact",
            category="inferred",
            value="some_value",
            confidence=0.15,  # Already low
        )

        # Act: Correct it (would go to -0.15 without floor)
        response = client.patch(
            "/api/user-model/facts/test_low_confidence_fact",
            json={},
        )

        # Assert: Confidence is clamped to 0.0
        assert response.status_code == 200
        result = response.json()
        assert result["old_confidence"] == 0.15
        assert result["new_confidence"] == 0.0  # Floor applied
        assert result["fact"]["confidence"] == 0.0

    def test_correct_nonexistent_fact_returns_404(self, client):
        """Test that correcting a non-existent fact returns 404."""
        # Act: Try to correct a fact that doesn't exist
        response = client.patch(
            "/api/user-model/facts/nonexistent_fact_key",
            json={},
        )

        # Assert: 404 error
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_correct_fact_updates_last_confirmed(self, client, user_model_store: UserModelStore):
        """Test that correction updates the last_confirmed timestamp."""
        # Arrange
        user_model_store.update_semantic_fact(
            key="test_timestamp_update",
            category="test",
            value="value",
            confidence=0.60,
        )

        # Record the original timestamp
        with user_model_store.db.get_connection("user_model") as conn:
            original = conn.execute(
                "SELECT last_confirmed FROM semantic_facts WHERE key = ?",
                ("test_timestamp_update",),
            ).fetchone()
            original_timestamp = original["last_confirmed"]

        # Act: Correct the fact
        response = client.patch(
            "/api/user-model/facts/test_timestamp_update",
            json={},
        )

        # Assert: Timestamp is updated
        assert response.status_code == 200
        fact = response.json()["fact"]
        new_timestamp = fact["last_confirmed"]
        assert new_timestamp != original_timestamp
        assert new_timestamp > original_timestamp  # Later timestamp

    def test_correct_fact_preserves_other_fields(self, client, user_model_store: UserModelStore):
        """Test that correction doesn't affect fields like category, key, first_observed."""
        # Arrange: Create a fact with rich metadata
        user_model_store.update_semantic_fact(
            key="test_preserve_fields",
            category="expertise",
            value="python",
            confidence=0.75,
            episode_id="episode_123",
        )

        # Record original state
        with user_model_store.db.get_connection("user_model") as conn:
            original = conn.execute(
                "SELECT * FROM semantic_facts WHERE key = ?",
                ("test_preserve_fields",),
            ).fetchone()

        # Act: Correct with a new value
        response = client.patch(
            "/api/user-model/facts/test_preserve_fields",
            json={"corrected_value": "javascript"},
        )

        # Assert: Only value, confidence, is_user_corrected, last_confirmed changed
        assert response.status_code == 200
        fact = response.json()["fact"]
        assert fact["key"] == original["key"]
        assert fact["category"] == original["category"]
        assert fact["first_observed"] == original["first_observed"]
        assert fact["source_episodes"] == original["source_episodes"]
        assert fact["times_confirmed"] == original["times_confirmed"]

        # These fields should have changed
        assert json.loads(fact["value"]) != json.loads(original["value"])
        assert fact["confidence"] != original["confidence"]
        assert fact["is_user_corrected"] == 1
        assert fact["last_confirmed"] != original["last_confirmed"]

    def test_correct_already_corrected_fact(self, client, user_model_store: UserModelStore):
        """Test that correcting an already-corrected fact works (e.g., second correction)."""
        # Arrange: Create and correct a fact once
        user_model_store.update_semantic_fact(
            key="test_double_correction",
            category="test",
            value="original",
            confidence=0.80,
        )

        client.patch(
            "/api/user-model/facts/test_double_correction",
            json={"corrected_value": "first_correction"},
        )

        # Act: Correct it again
        response = client.patch(
            "/api/user-model/facts/test_double_correction",
            json={"corrected_value": "second_correction"},
        )

        # Assert: Second correction works
        assert response.status_code == 200
        result = response.json()
        # First correction: 0.80 - 0.30 = 0.50
        # Second correction: 0.50 - 0.30 = 0.20
        assert result["new_confidence"] == 0.20
        assert json.loads(result["fact"]["value"]) == "second_correction"


class TestFactCorrectionIntegration:
    """Test integration of fact correction with feedback collector and telemetry."""

    def test_correction_logs_to_feedback_collector(self, db: DatabaseManager, user_model_store: UserModelStore):
        """Test that corrections are logged to feedback_log for analytics."""
        from services.feedback_collector.collector import FeedbackCollector

        # Arrange: Create LifeOS with feedback collector
        life_os = LifeOS.__new__(LifeOS)
        life_os.db = db
        life_os.user_model_store = user_model_store
        life_os.event_bus = None
        life_os.feedback_collector = FeedbackCollector(db, user_model_store, event_bus=None)

        app = create_web_app(life_os)
        client = TestClient(app)

        # Create a fact
        user_model_store.update_semantic_fact(
            key="test_feedback_logging",
            category="test",
            value="value",
            confidence=0.70,
        )

        # Act: Correct the fact with a reason
        response = client.patch(
            "/api/user-model/facts/test_feedback_logging",
            json={"reason": "This was wrong because..."},
        )

        # Assert: Correction succeeds
        assert response.status_code == 200

        # Assert: Feedback was logged
        with db.get_connection("preferences") as conn:
            feedback_entries = conn.execute(
                """SELECT * FROM feedback_log
                   WHERE action_type = 'semantic_fact'
                   AND feedback_type = 'corrected'
                   ORDER BY timestamp DESC
                   LIMIT 1"""
            ).fetchall()

            assert len(feedback_entries) == 1
            feedback = feedback_entries[0]
            assert feedback["action_id"] == "fact_correction_test_feedback_logging"
            assert feedback["notes"] == "This was wrong because..."

            context = json.loads(feedback["context"])
            assert context["fact_key"] == "test_feedback_logging"
            assert context["old_confidence"] == 0.70
            assert context["new_confidence"] == 0.40
            assert context["reason"] == "This was wrong because..."

    def test_correction_publishes_telemetry_event(self, db: DatabaseManager, user_model_store: UserModelStore):
        """Test that corrections publish telemetry events to the event bus."""
        from services.event_bus.bus import EventBus
        from unittest.mock import AsyncMock

        # Arrange: Create LifeOS with a mocked event bus
        life_os = LifeOS.__new__(LifeOS)
        life_os.db = db
        life_os.user_model_store = user_model_store
        life_os.feedback_collector = None

        # Mock the event bus
        mock_bus = AsyncMock()
        mock_bus.is_connected = True
        mock_bus.publish = AsyncMock()
        life_os.event_bus = mock_bus

        app = create_web_app(life_os)
        client = TestClient(app)

        # Create a fact
        user_model_store.update_semantic_fact(
            key="test_telemetry",
            category="test",
            value="value",
            confidence=0.85,
        )

        # Act: Correct the fact
        response = client.patch(
            "/api/user-model/facts/test_telemetry",
            json={},
        )

        # Assert: Correction succeeds
        assert response.status_code == 200

        # Assert: Telemetry event was published
        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == "usermodel.fact.corrected"
        payload = call_args[0][1]
        assert payload["key"] == "test_telemetry"
        assert payload["old_confidence"] == 0.85
        assert payload["new_confidence"] == 0.55  # 0.85 - 0.30
        assert payload["category"] == "test"

    def test_get_corrected_facts(self, client, user_model_store: UserModelStore):
        """Test that corrected facts are visible in GET /api/user-model/facts."""
        # Arrange: Create and correct some facts
        user_model_store.update_semantic_fact(
            key="test_visible_correction_1",
            category="test",
            value="v1",
            confidence=0.60,
        )
        user_model_store.update_semantic_fact(
            key="test_visible_correction_2",
            category="test",
            value="v2",
            confidence=0.70,
        )

        client.patch("/api/user-model/facts/test_visible_correction_1", json={})
        client.patch("/api/user-model/facts/test_visible_correction_2", json={})

        # Act: Fetch all facts
        response = client.get("/api/user-model/facts")

        # Assert: Corrected facts are included with is_user_corrected flag
        assert response.status_code == 200
        facts = response.json()["facts"]

        corrected_facts = [f for f in facts if f["is_user_corrected"] == 1]
        assert len(corrected_facts) >= 2

        # Check the corrected facts have reduced confidence
        for fact in corrected_facts:
            if fact["key"] == "test_visible_correction_1":
                assert fact["confidence"] == 0.30  # 0.60 - 0.30
            if fact["key"] == "test_visible_correction_2":
                assert fact["confidence"] == 0.40  # 0.70 - 0.30


class TestFactCorrectionEdgeCases:
    """Test edge cases and error conditions."""

    def test_correct_fact_with_complex_value(self, client, user_model_store: UserModelStore):
        """Test correcting a fact that has a complex JSON value."""
        # Arrange: Create a fact with nested object value
        complex_value = {
            "preferences": ["coffee", "tea"],
            "times": {"morning": 8, "evening": 18},
        }
        user_model_store.update_semantic_fact(
            key="test_complex_value",
            category="preferences",
            value=complex_value,
            confidence=0.75,
        )

        # Act: Correct with a new complex value
        new_value = {
            "preferences": ["coffee"],  # Removed tea
            "times": {"morning": 9, "evening": 17},  # Adjusted times
        }
        response = client.patch(
            "/api/user-model/facts/test_complex_value",
            json={"corrected_value": new_value},
        )

        # Assert: Complex value is stored correctly
        assert response.status_code == 200
        fact = response.json()["fact"]
        stored_value = json.loads(fact["value"])
        assert stored_value == new_value

    def test_correct_fact_with_null_value(self, client, user_model_store: UserModelStore):
        """Test that correcting without providing a value preserves original value."""
        # Arrange
        user_model_store.update_semantic_fact(
            key="test_null_correction",
            category="test",
            value="some_value",
            confidence=0.60,
        )

        # Act: Correct without providing a replacement value
        # (In Pydantic, corrected_value=None means "not provided")
        response = client.patch(
            "/api/user-model/facts/test_null_correction",
            json={},  # Don't provide corrected_value
        )

        # Assert: Original value is preserved
        assert response.status_code == 200
        fact = response.json()["fact"]
        # Value should be unchanged
        assert json.loads(fact["value"]) == "some_value"
        # But confidence and correction flag should be updated
        assert fact["is_user_corrected"] == 1
        assert fact["confidence"] == 0.30  # 0.60 - 0.30

    def test_multiple_corrections_reach_zero_confidence(self, client, user_model_store: UserModelStore):
        """Test that multiple corrections eventually drive confidence to zero."""
        # Arrange: Create a fact with 0.90 confidence
        user_model_store.update_semantic_fact(
            key="test_multi_correction",
            category="test",
            value="value",
            confidence=0.90,
        )

        # Act: Correct it 4 times
        # 1st: 0.90 - 0.30 = 0.60
        # 2nd: 0.60 - 0.30 = 0.30
        # 3rd: 0.30 - 0.30 = 0.00
        # 4th: 0.00 - 0.30 = 0.00 (floor)
        for i in range(4):
            response = client.patch(
                "/api/user-model/facts/test_multi_correction",
                json={},
            )
            assert response.status_code == 200

        # Assert: Final confidence is 0.0
        final_response = client.get("/api/user-model/facts")
        facts = final_response.json()["facts"]
        target_fact = next(f for f in facts if f["key"] == "test_multi_correction")
        assert target_fact["confidence"] == 0.0
