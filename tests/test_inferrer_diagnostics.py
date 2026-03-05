"""Tests for SemanticFactInferrer.get_diagnostics() method."""

import json

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


@pytest.fixture()
def inferrer(user_model_store):
    """A SemanticFactInferrer wired to the temporary UserModelStore."""
    return SemanticFactInferrer(user_model_store)


class TestGetDiagnosticsNoData:
    """Diagnostics when no signal profiles exist."""

    def test_all_profiles_unavailable(self, inferrer):
        """With no profiles stored, all should report unavailable."""
        diag = inferrer.get_diagnostics()
        for ptype, info in diag["profile_availability"].items():
            assert info["available"] is False
            assert info["samples"] == 0

    def test_health_no_data(self, inferrer):
        """Health should be 'no_data' when no profiles exist."""
        diag = inferrer.get_diagnostics()
        assert diag["health"] == "no_data"

    def test_last_cycle_none_before_inference(self, inferrer):
        """Before any inference run, last_cycle should be None."""
        diag = inferrer.get_diagnostics()
        assert diag["last_cycle"] is None

    def test_last_inference_time_none_before_inference(self, inferrer):
        """Before any inference run, last_inference_time should be None."""
        diag = inferrer.get_diagnostics()
        assert diag["last_inference_time"] is None

    def test_total_facts_zero(self, inferrer):
        """With no facts written, total_facts should be 0."""
        diag = inferrer.get_diagnostics()
        assert diag["total_facts"] == 0


class TestGetDiagnosticsWithProfile:
    """Diagnostics when one or more signal profiles exist."""

    def test_profile_shows_available(self, inferrer, user_model_store):
        """A stored profile should appear as available with sample count."""
        user_model_store.update_signal_profile("relationships", {
            "contacts": {"alice": {"response_time_avg": 300}},
        })
        diag = inferrer.get_diagnostics()
        rel = diag["profile_availability"]["relationships"]
        assert rel["available"] is True
        assert rel["samples"] >= 1

    def test_health_degraded_with_profiles_no_facts(self, inferrer, user_model_store):
        """Health should be 'degraded' when profiles exist but no facts written."""
        user_model_store.update_signal_profile("relationships", {
            "contacts": {},
            "samples_count": 10,
        })
        diag = inferrer.get_diagnostics()
        assert diag["health"] == "degraded"

    def test_health_ok_with_facts(self, inferrer, user_model_store):
        """Health should be 'ok' when profiles exist and facts have been written."""
        user_model_store.update_signal_profile("relationships", {
            "contacts": {},
            "samples_count": 10,
        })
        # Write a fact directly
        user_model_store.update_semantic_fact(
            key="test_fact",
            category="preferences",
            value="test_value",
            confidence=0.5,
        )
        diag = inferrer.get_diagnostics()
        assert diag["health"] == "ok"
        assert diag["total_facts"] >= 1


class TestRunAllInferenceCachesResults:
    """Verify that run_all_inference() populates diagnostic cache."""

    def test_caches_last_inference_time(self, inferrer):
        """After run_all_inference(), last_inference_time should be set."""
        inferrer.run_all_inference()
        diag = inferrer.get_diagnostics()
        assert diag["last_inference_time"] is not None

    def test_caches_last_inference_results(self, inferrer):
        """After run_all_inference(), _last_inference_results should be populated."""
        inferrer.run_all_inference()
        assert len(inferrer._last_inference_results) == 9  # 9 profile types

    def test_last_cycle_shows_skipped_profiles(self, inferrer):
        """With no profiles, all should be skipped in last_cycle."""
        inferrer.run_all_inference()
        diag = inferrer.get_diagnostics()
        assert diag["last_cycle"] is not None
        assert len(diag["last_cycle"]["skipped"]) > 0
        # Each skipped entry has type and reason
        for entry in diag["last_cycle"]["skipped"]:
            assert "type" in entry
            assert "reason" in entry

    def test_last_cycle_processed_with_data(self, inferrer, user_model_store):
        """With a profile present, its type should appear in processed list."""
        # Store a linguistic profile with enough data to be processed
        user_model_store.update_signal_profile("linguistic", {
            "avg_formality": 0.3,
            "avg_sentence_length": 12.0,
            "vocabulary_richness": 0.6,
            "avg_message_length": 150.0,
            "emoji_frequency": 0.05,
            "exclamation_frequency": 0.1,
            "question_frequency": 0.2,
            "contraction_frequency": 0.4,
            "samples_count": 30,
        })
        inferrer.run_all_inference()
        diag = inferrer.get_diagnostics()
        assert "linguistic" in diag["last_cycle"]["processed"]


class TestDiagnosticsStructure:
    """Verify the overall structure of the diagnostics response."""

    def test_has_all_required_keys(self, inferrer):
        """Diagnostics should contain all expected top-level keys."""
        diag = inferrer.get_diagnostics()
        assert "last_inference_time" in diag
        assert "total_facts_written_last_cycle" in diag
        assert "profile_availability" in diag
        assert "last_cycle" in diag
        assert "total_facts" in diag
        assert "health" in diag

    def test_profile_availability_has_all_types(self, inferrer):
        """All 9 profile types should appear in profile_availability."""
        diag = inferrer.get_diagnostics()
        expected = {
            "linguistic", "linguistic_inbound", "relationships", "topics",
            "cadence", "mood_signals", "temporal", "spatial", "decision",
        }
        assert set(diag["profile_availability"].keys()) == expected
