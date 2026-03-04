"""
Tests for Semantic Fact Inferrer observability improvements.

Verifies that the SemanticFactInferrer logs skip/processed messages at INFO
level (visible at default log config) and returns status dicts from each
inference method so callers can inspect which profiles were processed.
"""

import logging

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


# -----------------------------------------------------------------------
# Status dict return values
# -----------------------------------------------------------------------


class TestInferenceStatusDicts:
    """Each infer_from_* method returns a status dict with type, processed, reason."""

    def test_linguistic_skip_returns_status(self, user_model_store):
        """Linguistic method returns skip status when profile has no samples."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_linguistic_profile()

        assert result is not None
        assert result["type"] == "linguistic"
        assert result["processed"] is False
        assert "insufficient samples" in result["reason"]

    def test_linguistic_processed_returns_status(self, user_model_store):
        """Linguistic method returns processed status when profile has data."""
        user_model_store.update_signal_profile("linguistic", {
            "averages": {"formality": 0.5},
        })
        _set_samples(user_model_store, "linguistic", 5)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_linguistic_profile()

        assert result is not None
        assert result["type"] == "linguistic"
        assert result["processed"] is True
        assert result["reason"] is None

    def test_relationship_skip_returns_status(self, user_model_store):
        """Relationship method returns skip status when profile has < 10 samples."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result is not None
        assert result["type"] == "relationship"
        assert result["processed"] is False
        assert "insufficient samples" in result["reason"]

    def test_topic_skip_returns_status(self, user_model_store):
        """Topic method returns skip status when profile has < 30 samples."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        assert result is not None
        assert result["type"] == "topic"
        assert result["processed"] is False
        assert "insufficient samples" in result["reason"]

    def test_cadence_skip_returns_status(self, user_model_store):
        """Cadence method returns skip status when profile has < 50 samples."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_cadence_profile()

        assert result is not None
        assert result["type"] == "cadence"
        assert result["processed"] is False
        assert "insufficient samples" in result["reason"]

    def test_mood_skip_returns_status(self, user_model_store):
        """Mood method returns skip status when profile has < 5 samples."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_mood_profile()

        assert result is not None
        assert result["type"] == "mood"
        assert result["processed"] is False
        assert "insufficient samples" in result["reason"]

    def test_temporal_skip_returns_status(self, user_model_store):
        """Temporal method returns skip status when profile has < 50 samples."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_temporal_profile()

        assert result is not None
        assert result["type"] == "temporal"
        assert result["processed"] is False
        assert "insufficient samples" in result["reason"]

    def test_spatial_skip_returns_status(self, user_model_store):
        """Spatial method returns skip status when profile has < 10 samples."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_spatial_profile()

        assert result is not None
        assert result["type"] == "spatial"
        assert result["processed"] is False
        assert "insufficient samples" in result["reason"]

    def test_decision_skip_returns_status(self, user_model_store):
        """Decision method returns skip status when profile has < 20 samples."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_decision_profile()

        assert result is not None
        assert result["type"] == "decision"
        assert result["processed"] is False
        assert "insufficient samples" in result["reason"]


# -----------------------------------------------------------------------
# Log output at INFO level (caplog tests)
# -----------------------------------------------------------------------


class TestSkipLogsAtInfoLevel:
    """Skip messages should be visible at the default INFO log level."""

    def test_all_profiles_empty_logs_skips_at_info(self, user_model_store, caplog):
        """When all profiles are empty, each skip is logged at INFO, not DEBUG."""
        inferrer = SemanticFactInferrer(user_model_store)

        with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.run_all_inference()

        # All 9 profile types should appear in skip messages at INFO level
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        skip_messages = [m for m in info_messages if "skipping inference" in m.lower()]

        # At minimum, all 9 methods should log a skip (profiles are empty)
        assert len(skip_messages) >= 9, (
            f"Expected at least 9 skip messages at INFO level, got {len(skip_messages)}: {skip_messages}"
        )

    def test_skip_messages_not_at_debug(self, user_model_store, caplog):
        """Ensure skip messages are NOT at DEBUG level (they were upgraded to INFO)."""
        inferrer = SemanticFactInferrer(user_model_store)

        with caplog.at_level(logging.DEBUG, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.run_all_inference()

        debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        debug_skip_messages = [m for m in debug_messages if "skipping inference" in m.lower()]

        assert len(debug_skip_messages) == 0, (
            f"Found skip messages at DEBUG level (should be INFO): {debug_skip_messages}"
        )

    def test_processed_profile_logs_at_info(self, user_model_store, caplog):
        """When a profile has data, the 'Inferred semantic facts' message is at INFO."""
        user_model_store.update_signal_profile("linguistic", {
            "averages": {"formality": 0.2},
        })
        _set_samples(user_model_store, "linguistic", 5)

        inferrer = SemanticFactInferrer(user_model_store)

        with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.infer_from_linguistic_profile()

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        inferred_messages = [m for m in info_messages if "Inferred semantic facts from linguistic" in m]

        assert len(inferred_messages) == 1, (
            f"Expected 1 'Inferred semantic facts from linguistic' message, got: {info_messages}"
        )


# -----------------------------------------------------------------------
# Inference summary
# -----------------------------------------------------------------------


class TestInferenceSummary:
    """The summary log line shows which profiles were processed vs skipped."""

    def test_all_skipped_summary(self, user_model_store, caplog):
        """When all profiles are empty, summary shows all skipped."""
        inferrer = SemanticFactInferrer(user_model_store)

        with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.run_all_inference()

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        summary_messages = [m for m in info_messages if "inference cycle complete" in m]

        assert len(summary_messages) == 1, f"Expected 1 summary message, got: {summary_messages}"
        summary = summary_messages[0]

        # All 8 types should appear in the skipped portion
        assert "processed: none" in summary
        for profile_type in ["linguistic", "relationship", "topic", "cadence", "mood", "temporal", "spatial", "decision"]:
            assert profile_type in summary, f"Expected '{profile_type}' in summary: {summary}"

    def test_mixed_processed_and_skipped_summary(self, user_model_store, caplog):
        """When some profiles have data, summary shows both processed and skipped."""
        # Set up linguistic with data (threshold is 1 sample)
        user_model_store.update_signal_profile("linguistic", {
            "averages": {"formality": 0.5},
        })
        _set_samples(user_model_store, "linguistic", 5)

        inferrer = SemanticFactInferrer(user_model_store)

        with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.run_all_inference()

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        summary_messages = [m for m in info_messages if "inference cycle complete" in m]

        assert len(summary_messages) == 1
        summary = summary_messages[0]

        # linguistic should be in the processed portion
        assert "linguistic" in summary
        # Other profiles should be in the skipped portion
        assert "topic" in summary
        assert "cadence" in summary

    def test_run_all_inference_returns_none(self, user_model_store):
        """run_all_inference() still returns None (backwards-compatible)."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.run_all_inference()
        assert result is None
