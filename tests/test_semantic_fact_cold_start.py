"""
Tests for SemanticFactInferrer cold-start diagnostics and zero-fact output fix.

The system was observed to have 0 semantic facts despite having 220K+ relationship
profile samples.  These tests verify that:

  1. run_all_inference() produces at least one fact when the relationship profile
     has 220K+ samples with inbound-only contacts (cold-start path).

  2. run_all_inference() populates _last_inference_results with per-method
     facts_written counts so the diagnostic endpoint shows meaningful data.

  3. update_semantic_fact() writes facts correctly even when episode_id is None
     (no episodes in DB yet).

  4. The pre-run profile diagnostic log includes sample counts for all 9 profiles.

Root causes fixed by this PR:
  - _infer_from_inbound_only_contacts() returned "facts_stored" key but
    _log_inference_summary() summed "facts_written" — mismatch caused 0 to be
    reported even when facts were actually written.
  - run_all_inference() had no pre-run logging of profile sample counts, making
    cold-start debugging impossible without adding ad-hoc print statements.
  - Inference methods that don't explicitly return facts_written now get an
    accurate count via the COUNT(*) delta in run_all_inference().
"""

import logging

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_samples(ums, profile_type: str, count: int) -> None:
    """Manually set samples_count for a signal profile row."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


def _build_inbound_contacts(count: int, base_inbound: int = 10) -> dict:
    """Build a contacts dict with inbound-only entries (outbound_count=0).

    Contacts are named person{i}@org{i}.example.com so they span multiple
    distinct email domains, which enables domain-breadth aggregate facts.
    """
    contacts = {}
    for i in range(count):
        contacts[f"person{i}@org{i}.example.com"] = {
            "interaction_count": base_inbound + i,
            "inbound_count": base_inbound + i,
            "outbound_count": 0,
            "channels_used": ["email"],
        }
    return contacts


# ---------------------------------------------------------------------------
# Cold-start: relationship profile with inbound-only data
# ---------------------------------------------------------------------------

class TestColdStartRelationship:
    """run_all_inference() must produce facts when only inbound relationship data exists."""

    def test_220k_inbound_relationship_produces_facts(self, user_model_store):
        """220K+ relationship samples with inbound-only contacts should yield >= 1 fact.

        This directly reproduces the production scenario: 220,351 relationship
        samples exist but 0 semantic facts were in the DB.  The inferrer should
        produce at least a communication_volume_category fact via the inbound-only
        fallback path.
        """
        # Build a realistic profile: 300 inbound-only contacts spanning many domains
        contacts = _build_inbound_contacts(300, base_inbound=50)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 220351)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        # At minimum, communication_volume_category should be written
        all_facts = user_model_store.get_semantic_facts()
        assert len(all_facts) >= 1, (
            "Expected at least 1 semantic fact from 220K relationship samples, "
            f"got 0.  last_inference_results={inferrer._last_inference_results}"
        )

        # Verify the communication volume fact exists
        volume_fact = next(
            (f for f in all_facts if f["key"] == "communication_volume_category"), None
        )
        assert volume_fact is not None, (
            "Expected communication_volume_category fact from 300-contact inbound profile"
        )
        assert volume_fact["value"] == "high_volume_email"  # 300 contacts > 50 threshold

    def test_inbound_with_diverse_domains_produces_domain_breadth_fact(self, user_model_store):
        """Inbound-only contacts spanning 10+ domains should produce a domain breadth fact.

        _infer_aggregate_relationship_facts() is now called even in the inbound-only
        primary path so that domain diversity facts are captured regardless of whether
        the user has sent any emails.
        """
        # 20 contacts each from a distinct domain → 20 distinct domains
        contacts = _build_inbound_contacts(20, base_inbound=10)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 5000)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        all_facts = user_model_store.get_semantic_facts()
        domain_fact = next(
            (f for f in all_facts if f["key"] == "contact_network_breadth"), None
        )
        assert domain_fact is not None, (
            "Expected contact_network_breadth fact from 20-domain inbound-only profile"
        )
        assert domain_fact["value"] == "diverse_multi_domain_network"

    def test_inbound_only_facts_written_with_no_episodes(self, user_model_store):
        """Facts should be persisted even when the episodes table is empty (episode_id=None).

        When the system cold-starts, no episodes exist yet.  _get_recent_episodes()
        returns [] and episode_id=None.  update_semantic_fact() must still INSERT
        the fact with source_episodes='[]'.
        """
        # Verify no episodes exist (empty DB)
        with user_model_store.db.get_connection("user_model") as conn:
            episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert episode_count == 0, "Precondition: episodes table must be empty"

        contacts = _build_inbound_contacts(5, base_inbound=20)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 1000)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True
        assert result.get("facts_written", 0) >= 1

        # Verify the fact was actually persisted
        all_facts = user_model_store.get_semantic_facts()
        assert len(all_facts) >= 1

        # Fact should have empty source_episodes (no episodes to link)
        volume_fact = next(
            (f for f in all_facts if f["key"] == "communication_volume_category"), None
        )
        assert volume_fact is not None
        assert volume_fact["source_episodes"] == []  # episode_id was None → empty list


# ---------------------------------------------------------------------------
# Per-method facts_written in _last_inference_results
# ---------------------------------------------------------------------------

class TestDiagnosticsPerMethodResults:
    """run_all_inference() must report per-method facts_written in _last_inference_results."""

    def test_last_inference_results_has_facts_written_for_each_method(self, user_model_store):
        """Every result dict in _last_inference_results must include a facts_written key.

        The orchestrator and diagnostics endpoint rely on this field to show
        "why are there 0 facts?" at a per-method level.  Without it, all methods
        appear to produce 0 facts even when they actually wrote some.
        """
        # Set up a relationship profile so at least one method produces facts
        contacts = _build_inbound_contacts(10, base_inbound=15)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 5000)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        assert inferrer._last_inference_results, "Expected non-empty _last_inference_results"

        # Every method result must have facts_written (even if 0)
        for r in inferrer._last_inference_results:
            assert "facts_written" in r, (
                f"Method '{r.get('type', '?')}' result missing facts_written key: {r}"
            )
            assert isinstance(r["facts_written"], int), (
                f"facts_written must be int for method '{r.get('type', '?')}', got {type(r['facts_written'])}"
            )
            assert r["facts_written"] >= 0, (
                f"facts_written must be non-negative for '{r.get('type', '?')}'"
            )

    def test_total_facts_written_equals_sum_of_per_method(self, user_model_store):
        """_total_facts_written_last_cycle should equal sum of per-method facts_written."""
        contacts = _build_inbound_contacts(10, base_inbound=20)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 5000)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        per_method_total = sum(
            r.get("facts_written", 0) for r in inferrer._last_inference_results
        )
        assert inferrer._total_facts_written_last_cycle == per_method_total, (
            f"_total_facts_written_last_cycle ({inferrer._total_facts_written_last_cycle}) "
            f"!= sum of per-method facts_written ({per_method_total})"
        )

    def test_relationship_method_result_uses_facts_written_key(self, user_model_store):
        """infer_from_relationship_profile() must return facts_written (not facts_stored).

        Pre-fix: _infer_from_inbound_only_contacts() returned the key 'facts_stored'
        but _log_inference_summary() summed 'facts_written'.  This caused 0 to be
        reported as total_facts_written even when facts were actually in the DB.
        """
        contacts = _build_inbound_contacts(5, base_inbound=10)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 1000)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert "facts_written" in result, (
            f"infer_from_relationship_profile() must return 'facts_written' key, got: {result.keys()}"
        )
        assert "facts_stored" not in result, (
            "Old 'facts_stored' key still present — should have been renamed to 'facts_written'"
        )
        assert result["facts_written"] >= 1  # At minimum: communication_volume_category


# ---------------------------------------------------------------------------
# Pre-run profile diagnostic logging
# ---------------------------------------------------------------------------

class TestColdStartDiagnosticLogging:
    """run_all_inference() must log profile sample counts before running methods."""

    def test_pre_run_log_includes_profile_sample_counts(self, user_model_store, caplog):
        """run_all_inference() should emit a log line with per-profile sample counts.

        The log line format is:
          "SemanticFactInferrer cold-start diagnostics — profile sample counts: ..."

        This is the first diagnostic an operator should check when semantic facts
        appear empty despite signal data existing.
        """
        # Set a nonzero count for the relationship profile so the log is informative
        contacts = _build_inbound_contacts(3, base_inbound=5)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 220351)

        inferrer = SemanticFactInferrer(user_model_store)

        with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.run_all_inference()

        # Verify the cold-start diagnostic log line was emitted
        cold_start_lines = [
            r.message for r in caplog.records
            if "cold-start diagnostics" in r.message
        ]
        assert len(cold_start_lines) >= 1, (
            "Expected a 'cold-start diagnostics' log line but none found. "
            f"Log messages: {[r.message for r in caplog.records]}"
        )

        # The log line must include sample counts for key profiles
        diag_line = cold_start_lines[0]
        assert "relationship=" in diag_line, (
            f"Expected 'relationship=' in diagnostic log: {diag_line}"
        )
        assert "220351" in diag_line, (
            f"Expected relationship sample count '220351' in diagnostic log: {diag_line}"
        )
        # All 9 profile types should appear in the log
        for profile_name in ["linguistic", "relationship", "topic", "cadence", "mood",
                              "temporal", "spatial", "decision"]:
            assert profile_name in diag_line, (
                f"Expected '{profile_name}' in diagnostic log: {diag_line}"
            )

    def test_per_method_log_in_inference_summary(self, user_model_store, caplog):
        """_log_inference_summary() should include per-method breakdown in its log line.

        Example: "per_method: [linguistic=0, relationship=3, topic=skipped, ...]"

        This allows operators to identify which methods ran and which skipped.
        """
        contacts = _build_inbound_contacts(5, base_inbound=10)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 1000)

        inferrer = SemanticFactInferrer(user_model_store)

        with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.run_all_inference()

        # Verify the per-method summary log line exists
        summary_lines = [
            r.message for r in caplog.records
            if "per_method" in r.message
        ]
        assert len(summary_lines) >= 1, (
            "Expected a log line with 'per_method' breakdown but none found."
        )
        summary_line = summary_lines[0]
        # relationship should appear in the per-method breakdown
        assert "relationship" in summary_line
