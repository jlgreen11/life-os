"""
Test that prediction deduplication includes time_horizon in the dedup key.

Previously, dedup used only (prediction_type, description), which caused
legitimately different predictions with the same type and description but
different time_horizon values to be suppressed. For example, a calendar
conflict prediction for the same meeting at different time windows would
be silently dropped.

The fix adds time_horizon to the dedup key in both:
1. store_prediction() in storage/user_model_store.py
2. The pre-filter in services/prediction_engine/engine.py
"""

from dataclasses import dataclass


class TestStoreDeduplicationWithTimeHorizon:
    """Tests for time_horizon-aware deduplication in store_prediction()."""

    def test_different_time_horizon_both_stored(self, user_model_store):
        """Two predictions with same type+description but different time_horizon should both be stored."""
        ums = user_model_store

        pred_base = {
            "prediction_type": "conflict",
            "description": "Calendar conflict: Team standup overlaps with 1:1",
            "confidence": 0.75,
            "confidence_gate": "SUGGEST",
            "supporting_signals": {},
            "was_surfaced": True,
        }

        pred1 = {**pred_base, "id": "pred-th-1", "time_horizon": "2_hours"}
        pred2 = {**pred_base, "id": "pred-th-2", "time_horizon": "24_hours"}

        result1 = ums.store_prediction(pred1)
        result2 = ums.store_prediction(pred2)

        assert result1 is True, "First prediction should be stored"
        assert result2 is True, "Second prediction with different time_horizon should also be stored"

        with ums.db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction_type = 'conflict'").fetchone()[0]
            assert count == 2, "Both predictions should exist in the database"

    def test_same_time_horizon_deduped(self, user_model_store):
        """Two predictions with same type+description+time_horizon should be deduped."""
        ums = user_model_store

        pred_base = {
            "prediction_type": "reminder",
            "description": "Follow up with Alice about project update",
            "confidence": 0.80,
            "confidence_gate": "DEFAULT",
            "time_horizon": "24_hours",
            "supporting_signals": {},
            "was_surfaced": True,
        }

        pred1 = {**pred_base, "id": "pred-same-1"}
        pred2 = {**pred_base, "id": "pred-same-2"}

        result1 = ums.store_prediction(pred1)
        result2 = ums.store_prediction(pred2)

        assert result1 is True
        assert result2 is False, "Duplicate prediction with same time_horizon should be deduped"

        with ums.db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction_type = 'reminder'").fetchone()[0]
            assert count == 1

    def test_null_time_horizon_deduped_correctly(self, user_model_store):
        """Two predictions with NULL time_horizon and same type+description should be deduped."""
        ums = user_model_store

        pred_base = {
            "prediction_type": "risk",
            "description": "Spending exceeds budget threshold",
            "confidence": 0.65,
            "confidence_gate": "SUGGEST",
            "supporting_signals": {},
            "was_surfaced": False,
        }

        pred1 = {**pred_base, "id": "pred-null-1"}
        pred2 = {**pred_base, "id": "pred-null-2"}

        result1 = ums.store_prediction(pred1)
        result2 = ums.store_prediction(pred2)

        assert result1 is True
        assert result2 is False, "Duplicate with NULL time_horizon should still be deduped"

    def test_null_vs_non_null_time_horizon_not_deduped(self, user_model_store):
        """A prediction with NULL time_horizon and one with a value should not be deduped."""
        ums = user_model_store

        pred_base = {
            "prediction_type": "opportunity",
            "description": "Good time to reach out to Bob",
            "confidence": 0.70,
            "confidence_gate": "SUGGEST",
            "supporting_signals": {},
            "was_surfaced": True,
        }

        pred1 = {**pred_base, "id": "pred-nv-1"}
        pred2 = {**pred_base, "id": "pred-nv-2", "time_horizon": "this_week"}

        result1 = ums.store_prediction(pred1)
        result2 = ums.store_prediction(pred2)

        assert result1 is True
        assert result2 is True, "NULL vs non-NULL time_horizon should not be treated as duplicates"


class TestPreFilterWithTimeHorizon:
    """Tests for the pre-filter set construction including time_horizon."""

    def test_pre_filter_set_includes_time_horizon(self, user_model_store):
        """Verify that a prediction with a new time_horizon passes the pre-filter."""
        ums = user_model_store

        ums.store_prediction({
            "id": "pred-pf-1",
            "prediction_type": "conflict",
            "description": "Meeting overlap detected",
            "confidence": 0.80,
            "confidence_gate": "DEFAULT",
            "time_horizon": "2_hours",
            "supporting_signals": {},
            "was_surfaced": True,
        })

        with ums.db.get_connection("user_model") as conn:
            rows = conn.execute(
                """SELECT prediction_type, description, time_horizon FROM predictions
                   WHERE resolved_at IS NULL
                      OR datetime(resolved_at) > datetime('now', '-24 hours')"""
            ).fetchall()
            existing = {(r[0], r[1], r[2]) for r in rows}

        assert ("conflict", "Meeting overlap detected", "2_hours") in existing

        @dataclass
        class FakePrediction:
            prediction_type: str
            description: str
            time_horizon: str | None

        same_horizon = FakePrediction("conflict", "Meeting overlap detected", "2_hours")
        diff_horizon = FakePrediction("conflict", "Meeting overlap detected", "24_hours")

        assert (same_horizon.prediction_type, same_horizon.description, same_horizon.time_horizon) in existing
        assert (diff_horizon.prediction_type, diff_horizon.description, diff_horizon.time_horizon) not in existing
