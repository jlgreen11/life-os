"""
Test prediction storage deduplication in UserModelStore.

Before iteration 147, the prediction engine generated 340K+ duplicate
predictions because there was no deduplication logic at the STORAGE layer.
The same "reminder" predictions were stored repeatedly every 15 minutes for
the same unreplied emails, causing massive database bloat and event bus spam.

This test suite verifies that:
1. Duplicate unresolved predictions are skipped (not stored)
2. Resolved predictions can be regenerated (dedup only applies to unresolved)
3. Different predictions are always stored (dedup is type + description specific)

Note: This is different from test_prediction_deduplication.py (iteration 62),
which tested deduplication at the GENERATION layer (checking before calling
generate). This tests deduplication at the STORAGE layer (inside store_prediction).
"""

from datetime import UTC, datetime, timedelta


class TestPredictionStorageDeduplication:
    """Test suite for prediction storage deduplication logic."""

    def test_identical_unresolved_predictions_are_deduplicated(self, user_model_store, db):
        """First prediction is stored, duplicate is skipped."""
        ums = user_model_store

        prediction1 = {
            "id": "pred-1",
            "prediction_type": "reminder",
            "description": "Unreplied message from alice@example.com",
            "confidence": 0.85,
            "confidence_gate": "DEFAULT",
            "time_horizon": "24_hours",
            "supporting_signals": {"contact_id": "alice"},
            "was_surfaced": True,
            "resolved_at": None,  # Unresolved
        }

        prediction2 = {
            "id": "pred-2",  # Different ID
            "prediction_type": "reminder",  # Same type
            "description": "Unreplied message from alice@example.com",  # Same description
            "confidence": 0.90,  # Different confidence (shouldn't matter)
            "confidence_gate": "AUTONOMOUS",  # Different gate (shouldn't matter)
            "time_horizon": "24_hours",  # Same horizon (part of dedup key)
            "supporting_signals": {"contact_id": "alice", "extra": "data"},
            "was_surfaced": False,
            "resolved_at": None,  # Unresolved
        }

        # Store first prediction
        ums.store_prediction(prediction1)

        # Attempt to store duplicate
        ums.store_prediction(prediction2)

        # Verify: only one prediction stored
        with ums.db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            assert count == 1, "Duplicate prediction should not be stored"

            # Verify the first prediction was kept (not the duplicate)
            stored = conn.execute("SELECT id FROM predictions").fetchone()
            assert stored["id"] == "pred-1", "Original prediction should be kept"

    def test_resolved_predictions_can_be_regenerated(self, user_model_store, db):
        """
        If a prediction is resolved > 24h ago, the same prediction can be generated again.

        UPDATED: Iteration 149 added 24h deduplication window for filtered predictions.
        This prevents the same "confidence:0.25" prediction from being stored 42 times
        in 2 hours (the bug that this iteration fixes).
        """
        ums = user_model_store

        # Resolved prediction from 25+ hours ago (outside deduplication window)
        old_resolved_time = (datetime.now(UTC) - timedelta(hours=25)).isoformat()

        prediction1 = {
            "id": "pred-1",
            "prediction_type": "reminder",
            "description": "Unreplied message from bob@example.com",
            "confidence": 0.80,
            "confidence_gate": "SUGGEST",
            "time_horizon": "24_hours",
            "supporting_signals": {"contact_id": "bob"},
            "was_surfaced": True,
            "user_response": "dismissed",
            "resolved_at": old_resolved_time,  # RESOLVED 25h ago
        }

        prediction2 = {
            "id": "pred-2",
            "prediction_type": "reminder",
            "description": "Unreplied message from bob@example.com",  # Same description
            "confidence": 0.85,
            "confidence_gate": "DEFAULT",
            "time_horizon": "12_hours",
            "supporting_signals": {"contact_id": "bob"},
            "was_surfaced": False,
            "resolved_at": None,  # Unresolved
        }

        # Store old resolved prediction
        ums.store_prediction(prediction1)

        # Store new prediction with same description (should succeed - outside 24h window)
        ums.store_prediction(prediction2)

        # Verify: both predictions stored
        with ums.db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            assert count == 2, "Resolved predictions > 24h old should allow regeneration"

    def test_different_types_are_not_duplicates(self, user_model_store, db):
        """Same description but different type → both stored."""
        ums = user_model_store

        prediction1 = {
            "id": "pred-1",
            "prediction_type": "reminder",
            "description": "Follow up on project proposal",
            "confidence": 0.75,
            "confidence_gate": "SUGGEST",
            "time_horizon": "24_hours",
            "supporting_signals": {},
            "was_surfaced": True,
        }

        prediction2 = {
            "id": "pred-2",
            "prediction_type": "need",  # Different type
            "description": "Follow up on project proposal",  # Same description
            "confidence": 0.80,
            "confidence_gate": "DEFAULT",
            "time_horizon": "2_hours",
            "supporting_signals": {},
            "was_surfaced": False,
        }

        ums.store_prediction(prediction1)
        ums.store_prediction(prediction2)

        with ums.db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            assert count == 2, "Different types should both be stored"

    def test_different_descriptions_are_not_duplicates(self, user_model_store, db):
        """Same type but different description → both stored."""
        ums = user_model_store

        prediction1 = {
            "id": "pred-1",
            "prediction_type": "reminder",
            "description": "Unreplied message from alice@example.com",
            "confidence": 0.85,
            "confidence_gate": "DEFAULT",
            "time_horizon": "24_hours",
            "supporting_signals": {},
            "was_surfaced": True,
        }

        prediction2 = {
            "id": "pred-2",
            "prediction_type": "reminder",  # Same type
            "description": "Unreplied message from bob@example.com",  # Different description
            "confidence": 0.85,
            "confidence_gate": "DEFAULT",
            "time_horizon": "24_hours",
            "supporting_signals": {},
            "was_surfaced": True,
        }

        ums.store_prediction(prediction1)
        ums.store_prediction(prediction2)

        with ums.db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            assert count == 2, "Different descriptions should both be stored"

    def test_multiple_duplicates_skipped(self, user_model_store, db):
        """If the same prediction is generated 10 times, only the first is stored."""
        ums = user_model_store

        base_prediction = {
            "prediction_type": "reminder",
            "description": "Weekly team meeting in 1 hour",
            "confidence": 0.90,
            "confidence_gate": "AUTONOMOUS",
            "time_horizon": "1_hour",
            "supporting_signals": {"calendar_event_id": "evt-123"},
            "was_surfaced": True,
        }

        # Generate 10 identical predictions (only IDs differ)
        for i in range(10):
            prediction = base_prediction.copy()
            prediction["id"] = f"pred-{i}"
            ums.store_prediction(prediction)

        # Verify: only one stored
        with ums.db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            assert count == 1, "All duplicates should be skipped"

            stored = conn.execute("SELECT id FROM predictions").fetchone()
            assert stored["id"] == "pred-0", "First prediction should be kept"

    def test_deduplication_with_filtered_predictions(self, user_model_store, db):
        """
        Filtered predictions within 24h are deduplicated, but allow regeneration after 24h.

        UPDATED: Iteration 149 fixes the bug where filtered predictions (resolved_at set)
        were not being deduplicated, causing 42 duplicates of the same "confidence:0.25"
        prediction to be stored in 2 hours. The 24h window prevents this spam while still
        allowing regeneration when conditions truly change (after 24h).
        """
        ums = user_model_store

        # Filtered prediction from 25+ hours ago (outside deduplication window)
        old_filtered_time = (datetime.now(UTC) - timedelta(hours=25)).isoformat()

        prediction1 = {
            "id": "pred-1",
            "prediction_type": "reminder",
            "description": "Calendar conflict detected",
            "confidence": 0.25,  # Low confidence
            "confidence_gate": "OBSERVE",
            "time_horizon": "2_hours",
            "supporting_signals": {},
            "was_surfaced": False,
            "user_response": "filtered",
            "filter_reason": "confidence:0.25 (threshold:0.3)",
            "resolved_at": old_filtered_time,  # Filtered 25h ago
        }

        prediction2 = {
            "id": "pred-2",
            "prediction_type": "reminder",
            "description": "Calendar conflict detected",  # Same
            "confidence": 0.30,  # Now above threshold
            "confidence_gate": "SUGGEST",
            "time_horizon": "2_hours",
            "supporting_signals": {},
            "was_surfaced": True,
            "resolved_at": None,  # New unresolved prediction
        }

        # Store old filtered prediction
        ums.store_prediction(prediction1)

        # Attempt to store new prediction (should succeed - outside 24h window)
        ums.store_prediction(prediction2)

        with ums.db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            assert count == 2, "Filtered predictions > 24h old should allow regeneration"

    def test_deduplication_production_scenario(self, user_model_store, db):
        """Simulate production: 340K reminders for same emails generated over time."""
        ums = user_model_store

        # Simulate 3 unreplied emails
        emails = [
            {"from": "alice@example.com", "subject": "Project update"},
            {"from": "bob@example.com", "subject": "Meeting notes"},
            {"from": "carol@example.com", "subject": "Question about report"},
        ]

        # Prediction engine runs every 15 minutes for 24 hours = 96 cycles
        # For each cycle, it generates a reminder for each unreplied email
        # Without deduplication: 3 emails × 96 cycles = 288 predictions stored
        # With deduplication: 3 emails × 1 (first cycle only) = 3 predictions stored

        for cycle in range(96):
            for idx, email in enumerate(emails):
                prediction = {
                    "id": f"pred-cycle{cycle}-email{idx}",
                    "prediction_type": "reminder",
                    "description": f"Unreplied message from {email['from']}: \"{email['subject']}\"",
                    "confidence": 0.80,
                    "confidence_gate": "DEFAULT",
                    "time_horizon": "24_hours",
                    "supporting_signals": {"from": email["from"]},
                    "was_surfaced": True,
                }
                ums.store_prediction(prediction)

        # Verify: only 3 predictions stored (one per email, NOT one per cycle)
        with ums.db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            assert count == 3, f"Expected 3 predictions (one per email), got {count}"

            # Verify predictions are from first cycle
            stored = conn.execute("SELECT id FROM predictions ORDER BY id").fetchall()
            assert stored[0]["id"] == "pred-cycle0-email0"
            assert stored[1]["id"] == "pred-cycle0-email1"
            assert stored[2]["id"] == "pred-cycle0-email2"
