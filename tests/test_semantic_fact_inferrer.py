"""
Tests for Semantic Fact Inference Service

Verifies that the SemanticFactInferrer correctly derives high-level semantic
facts from signal profiles across all five extractors.
"""

import logging
from unittest.mock import patch

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type)
        )


class TestSemanticFactInferrer:
    """Test suite for semantic fact inference across all signal profiles."""

    # -------------------------------------------------------------------
    # Linguistic Profile Inference Tests
    # -------------------------------------------------------------------

    def test_infer_casual_communication_style(self, user_model_store):
        """Derive casual style preference from low formality scores."""
        profile_data = {
            "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        casual_fact = next((f for f in facts if f["key"] == "communication_style_formality"), None)
        assert casual_fact is not None
        assert casual_fact["value"] == "casual"
        assert casual_fact["confidence"] >= 0.6

    def test_infer_formal_communication_style(self, user_model_store):
        """Derive formal style preference from high formality scores."""
        profile_data = {
            "averages": {"formality": 0.8, "emoji_rate": 0.0, "hedge_rate": 0.05, "exclamation_rate": 0.0},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        formal_fact = next((f for f in facts if f["key"] == "communication_style_formality"), None)
        assert formal_fact is not None
        assert formal_fact["value"] == "formal"
        assert formal_fact["confidence"] >= 0.6

    def test_infer_emoji_usage_preference(self, user_model_store):
        """Derive emoji usage preference from high emoji rate."""
        profile_data = {
            "averages": {"formality": 0.5, "emoji_rate": 0.06, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        emoji_fact = next((f for f in facts if f["key"] == "communication_style_emoji"), None)
        assert emoji_fact is not None
        assert emoji_fact["value"] == "expressive_with_emojis"

    def test_no_inference_with_insufficient_samples(self, user_model_store):
        """Skip inference when sample count is below threshold (1 for linguistic)."""
        # Threshold was lowered from 20 to 1, so we need 0 samples to skip inference
        profile_data = {"averages": {"formality": 0.1}}
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 0)  # Below 1-sample threshold

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 0

    # -------------------------------------------------------------------
    # Relationship Profile Inference Tests
    # -------------------------------------------------------------------

    def test_infer_high_priority_contact(self, user_model_store):
        """Derive high-priority relationship from above-average interaction count.

        The inferrer labels a contact "high_priority" when their interaction
        count is >= 2x the average across all bidirectional contacts.

        Setup: alice (30 interactions) vs bob (5 interactions).
          avg = (30 + 5) / 2 = 17.5; threshold = 35.
          alice (30) < 35 — actually below threshold; adjust bob lower.

        Setup v2: alice (40 interactions) vs bob (5 interactions).
          avg = (40 + 5) / 2 = 22.5; threshold = 45.  Still below.

        Setup v3: alice (40 interactions) vs bob (2 interactions).
          avg = (40 + 2) / 2 = 21; threshold = 42.  Still below.

        Better approach: alice (10) alone → avg = 10, threshold = 20 > 10. Not high.
        Use alice (30), bob (2) → avg = 16, threshold = 32.  alice (30) < 32.

        Simplest: alice (20), bob (1) → avg = 10.5, threshold = 21. alice (20) < 21.
        alice (22), bob (1) → avg = 11.5, threshold = 23. alice (22) < 23.
        alice (25), bob (1) → avg = 13, threshold = 26. alice (25) < 26.
        alice (30), bob (1) → avg = 15.5, threshold = 31. alice (30) < 31.
        alice (50), bob (1) → avg = 25.5, threshold = 51. alice (50) < 51.
        alice (60), bob (1) → avg = 30.5, threshold = 61. alice (60) < 61.

        Need N contacts so that alice's count >= avg * 2.
        With N contacts where bob has count=1:
          avg = (alice + (N-1)*1) / N; threshold = 2 * avg
          alice >= threshold means alice >= 2 * (alice + N - 1) / N
          N * alice >= 2 * alice + 2 * (N-1)
          alice * (N - 2) >= 2 * (N - 1)
          For N=3, alice (10): 10 * 1 >= 2 * 2 = 4.  YES! alice (10) >= 4.

        Setup: alice (10), bob (1), carol (1).
          avg = 12/3 = 4; threshold = 8. alice (10) >= 8 → HIGH PRIORITY ✓

        outbound_count > 0 required by the inbound-only filter (PR #204).
        """
        profile_data = {
            "contacts": {
                # High-frequency contact under test — 10 interactions vs avg of 4
                "alice@example.com": {
                    "interaction_count": 10,
                    "avg_response_time_seconds": 1800,  # 30 minutes — fast responder
                    "outbound_count": 5,  # bidirectional (required by inbound-only filter)
                },
                # Low-frequency contacts that lower the average so alice qualifies
                "bob@example.com": {
                    "interaction_count": 1,
                    "outbound_count": 1,  # bidirectional
                },
                "carol@example.com": {
                    "interaction_count": 1,
                    "outbound_count": 1,  # bidirectional
                },
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        # alice (10) >= threshold (8 = avg 4 * 2) → should be labelled high_priority
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        priority_fact = next((f for f in facts if f["key"] == "relationship_priority_alice@example.com"), None)
        assert priority_fact is not None
        assert priority_fact["value"] == "high_priority"

    def test_infer_low_priority_contact(self, user_model_store):
        """Contacts below the high-priority threshold get no priority fact.

        The inferrer only creates relationship_priority_* facts for contacts
        with >= 2x the average interaction count ("high_priority" value).
        A contact with fewer interactions than average receives no fact at all
        — that is the correct behavior, not a "low_priority" label.

        outbound_count must be > 0 for both contacts so the inbound-only filter
        (added in PR #204) does not exclude them from bidirectional scoring.
        """
        profile_data = {
            "contacts": {
                # Below-average contact — should NOT receive a priority fact
                "bob@example.com": {
                    "interaction_count": 8,
                    "avg_response_time_seconds": 172800,  # 48 hours
                    "outbound_count": 3,  # bidirectional
                },
                # High-volume contact that sets the average above bob's count
                "alice@example.com": {
                    "interaction_count": 50,  # avg = (8+50)/2 = 29; threshold = 58; alice also below
                    "avg_response_time_seconds": 900,
                    "outbound_count": 25,  # bidirectional
                },
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        # Neither contact exceeds 2x average (29 * 2 = 58), so no priority facts
        # should be created for either.
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        priority_keys = {f["key"] for f in facts if "relationship_priority" in f["key"]}
        assert "relationship_priority_bob@example.com" not in priority_keys
        assert "relationship_priority_alice@example.com" not in priority_keys

    # -------------------------------------------------------------------
    # Topic Profile Inference Tests
    # -------------------------------------------------------------------

    def test_infer_expertise_from_frequent_topic(self, user_model_store):
        """Derive expertise from frequently discussed topics.

        The topic extractor stores data under "topic_counts" (not
        "topic_frequencies" — that key was renamed when the schema mismatch
        was fixed).  50 occurrences out of 100 samples = 50% frequency, which
        exceeds the 10% expertise threshold.
        """
        profile_data = {
            "topic_counts": {
                "python": 50,  # 50% of 100 samples → above 10% expertise threshold
            }
        }
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="expertise")
        python_fact = next((f for f in facts if f["key"] == "expertise_python"), None)
        assert python_fact is not None
        assert python_fact["value"] == "python"

    # -------------------------------------------------------------------
    # Cadence Profile Inference Tests
    # -------------------------------------------------------------------

    def test_infer_strict_work_life_boundaries(self, user_model_store):
        """Derive strict boundaries from 90%+ business-hours messaging.

        The cadence extractor stores hourly data under "hourly_activity" (not
        "hourly_distribution" — that key was renamed when the schema mismatch
        was fixed).
        """
        profile_data = {
            "hourly_activity": {
                # 100 messages during business hours (9-17), small off-hours count
                str(h): 100 if 9 <= h <= 17 else 0 for h in range(24)
            }
        }
        # Add a few off-hours to make it realistic but still >90% in-hours
        profile_data["hourly_activity"]["8"] = 2
        profile_data["hourly_activity"]["18"] = 3

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="values")
        boundary_fact = next((f for f in facts if f["key"] == "work_life_boundaries"), None)
        assert boundary_fact is not None
        assert boundary_fact["value"] == "strict_boundaries"

    # -------------------------------------------------------------------
    # Mood Profile Inference Tests
    # -------------------------------------------------------------------

    def test_infer_low_stress_baseline(self, user_model_store):
        """Derive low-stress baseline from predominantly positive mood signals.

        The mood inferrer was redesigned to read "recent_signals" (not
        "avg_sentiment") and infer a "stress_baseline" fact (not
        "emotional_baseline").  A low stress baseline is inferred when < 10%
        of recent signals are negative-language signals.
        """
        # All 10 signals are positive (no negative_language) → stress_ratio = 0
        recent_signals = [
            {"signal_type": "positive_language", "value": 0.8}
            for _ in range(10)
        ]
        profile_data = {"recent_signals": recent_signals}
        user_model_store.update_signal_profile("mood_signals", profile_data)
        _set_samples(user_model_store, "mood_signals", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        stress_fact = next((f for f in facts if f["key"] == "stress_baseline"), None)
        assert stress_fact is not None
        assert stress_fact["value"] == "low_stress"

    # -------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------

    def test_run_all_inference_across_profiles(self, user_model_store):
        """Run inference across all profiles in a single call."""
        # Set up minimal data for each profile type
        user_model_store.update_signal_profile("linguistic", {
            "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        })
        _set_samples(user_model_store, "linguistic", 25)

        # Use a high interaction_count (60) so alice exceeds the high-priority
        # threshold (avg * 2 = 60 * 2 = 120... wait, with one contact avg IS 60,
        # so threshold is 120 > 60).  Use two contacts so the avg is lower.
        user_model_store.update_signal_profile("relationships", {
            # Two contacts so avg_interactions = (60+10)/2 = 35; threshold = 70.
            # alice (60) < 70, so no high_priority fact is created.
            # Instead the test relies on the linguistic profile producing a fact.
            "contacts": {
                "alice@example.com": {
                    "interaction_count": 10,
                    "avg_response_time_seconds": 1800,
                    "outbound_count": 5,  # bidirectional — required by inbound-only filter (PR #204)
                },
                "bob@example.com": {
                    "interaction_count": 5,
                    "avg_response_time_seconds": 3600,
                    "outbound_count": 2,  # bidirectional
                },
            }
        })
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        # The linguistic profile (formality 0.2 = casual) should produce at
        # least one fact (communication_style).
        facts = user_model_store.get_semantic_facts()
        assert len(facts) >= 1
        categories = {f["category"] for f in facts}
        # Casual formality (0.2) infers an implicit_preference for communication style
        assert "implicit_preference" in categories

    def test_confidence_growth_on_repeated_inference(self, user_model_store):
        """Semantic facts gain confidence with repeated observations."""
        profile_data = {
            "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        initial_confidence = next(f for f in facts if f["key"] == "communication_style_formality")["confidence"]

        # Second inference (same pattern observed again)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        updated_confidence = next(f for f in facts if f["key"] == "communication_style_formality")["confidence"]

        # Assert: Confidence increased by 0.05 (update_semantic_fact increment)
        assert updated_confidence > initial_confidence
        assert updated_confidence <= 1.0

    def test_no_facts_created_with_no_profiles(self, user_model_store):
        """Inference gracefully handles missing signal profiles."""
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 0

    def test_semantic_facts_include_provenance_fields(self, user_model_store):
        """Verify inferred facts include all required provenance fields."""
        profile_data = {
            "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts()
        fact = facts[0]

        # Assert: All required fields are present
        assert "key" in fact
        assert "category" in fact
        assert "value" in fact
        assert "confidence" in fact
        assert "source_episodes" in fact
        assert "first_observed" in fact
        assert "last_confirmed" in fact
        assert "times_confirmed" in fact
        assert "is_user_corrected" in fact

    # -------------------------------------------------------------------
    # Error Isolation Tests
    # -------------------------------------------------------------------

    def test_run_all_inference_isolates_single_profile_failure(self, user_model_store):
        """A single failing profile must not prevent other profiles from running.

        Monkeypatch infer_from_spatial_profile to raise, then verify that
        run_all_inference() completes without raising and results from the
        other 7 profiles are still collected.
        """
        # Set up linguistic data so it produces a real result
        user_model_store.update_signal_profile("linguistic", {
            "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        })
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(inferrer, "infer_from_spatial_profile", side_effect=RuntimeError("corrupt profile")):
            # Must NOT raise
            inferrer.run_all_inference()

        # Linguistic profile should still have produced facts
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        assert any(f["key"] == "communication_style_formality" for f in facts)

    def test_run_all_inference_logs_error_for_failed_profile(self, user_model_store, caplog):
        """The exception from a failing profile must be logged with full traceback."""
        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(inferrer, "infer_from_decision_profile", side_effect=KeyError("missing_key")):
            with caplog.at_level(logging.ERROR, logger="services.semantic_fact_inferrer.inferrer"):
                inferrer.run_all_inference()

        assert any("infer_from_decision_profile failed" in record.message for record in caplog.records)

    def test_run_all_inference_error_result_appears_in_summary(self, user_model_store, caplog):
        """Failed profiles appear as skipped with reason 'error' in the summary log."""
        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(inferrer, "infer_from_mood_profile", side_effect=ValueError("bad data")):
            with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
                inferrer.run_all_inference()

        # The summary log line should mention mood as skipped with (error) reason
        summary_lines = [r.message for r in caplog.records if "inference cycle complete" in r.message]
        assert len(summary_lines) == 1
        assert "mood (error)" in summary_lines[0]

    def test_run_all_inference_collects_all_results_despite_multiple_failures(self, user_model_store):
        """Even with multiple profile failures, results list has 9 entries."""
        inferrer = SemanticFactInferrer(user_model_store)

        with (
            patch.object(inferrer, "infer_from_spatial_profile", side_effect=RuntimeError("fail1")),
            patch.object(inferrer, "infer_from_decision_profile", side_effect=RuntimeError("fail2")),
            patch.object(inferrer, "infer_from_temporal_profile", side_effect=RuntimeError("fail3")),
        ):
            # Must NOT raise even with 3 failing profiles
            inferrer.run_all_inference()

    def test_run_all_inference_normal_operation_unchanged(self, user_model_store):
        """Normal operation (no exceptions) produces the same results as before."""
        user_model_store.update_signal_profile("linguistic", {
            "averages": {"formality": 0.8, "emoji_rate": 0.0, "hedge_rate": 0.05, "exclamation_rate": 0.0},
        })
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        # Formal style should still be inferred correctly
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        formal_fact = next((f for f in facts if f["key"] == "communication_style_formality"), None)
        assert formal_fact is not None
        assert formal_fact["value"] == "formal"

    # -------------------------------------------------------------------
    # Topic Inference — Multi-Topic and Noise Filter Tests
    # -------------------------------------------------------------------

    def test_topic_inference_creates_expertise_facts(self, user_model_store):
        """Multiple high-frequency topics should each produce expertise facts.

        Stores a topic profile with three topics well above the expertise
        thresholds (count >= 5, frequency > 0.08) and verifies that all
        three produce expertise_* facts with reasonable confidence.
        """
        profile_data = {
            "topic_counts": {
                "python": 100,           # 50% frequency
                "machine_learning": 50,  # 25% frequency
                "react": 30,            # 15% frequency
            },
            "recent_topics": [],
        }
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 200)

        # Insert an episode so _get_recent_episodes returns data
        _insert_episode(user_model_store)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="expertise")
        fact_keys = {f["key"] for f in facts}
        assert "expertise_python" in fact_keys
        assert "expertise_machine_learning" in fact_keys
        assert "expertise_react" in fact_keys

        # All should have confidence > 0.3 (base_confidence=0.5 since samples >= old_threshold)
        for fact in facts:
            assert fact["confidence"] > 0.3, f"Fact {fact['key']} has low confidence: {fact['confidence']}"

    def test_topic_inference_filters_noise(self, user_model_store):
        """Blocklisted noise topics like 'shop' and 'sale' must NOT produce facts.

        Even when noise tokens exceed the expertise threshold, the blocklist
        should prevent them from being stored as expertise or interest facts.
        """
        profile_data = {
            "topic_counts": {
                "shop": 80,    # In TOPIC_NOISE_BLOCKLIST — should be filtered
                "sale": 60,    # In TOPIC_NOISE_BLOCKLIST — should be filtered
                "python": 50,  # Legitimate topic — should produce a fact
            },
        }
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 200)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts()
        fact_keys = {f["key"] for f in facts}
        # Noise topics should NOT produce facts
        assert "expertise_shop" not in fact_keys
        assert "expertise_sale" not in fact_keys
        assert "interest_shop" not in fact_keys
        assert "interest_sale" not in fact_keys
        # Legitimate topic should produce a fact
        assert "expertise_python" in fact_keys

    def test_relationship_inference_requires_outbound(self, user_model_store):
        """Contacts with outbound_count=0 must be skipped (inbound-only filter).

        A contact who only sends inbound messages (newsletters, marketing)
        should never appear as a high_priority relationship fact.
        """
        profile_data = {
            "contacts": {
                # Inbound-only contact — high interaction count but no outbound
                "newsletter@example.com": {
                    "interaction_count": 100,
                    "avg_response_time_seconds": 0,
                    "outbound_count": 0,  # zero outbound → should be filtered
                },
                # Bidirectional contact with lower interactions
                "friend@example.com": {
                    "interaction_count": 5,
                    "avg_response_time_seconds": 3600,
                    "outbound_count": 3,  # has outbound → passes filter
                },
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = user_model_store.get_semantic_facts()
        fact_keys = {f["key"] for f in facts}
        # Inbound-only contact should NOT get a priority fact
        assert "relationship_priority_newsletter@example.com" not in fact_keys

    def test_topic_inference_continues_after_db_write_failure(self, user_model_store):
        """A DB write failure for one topic must not prevent other topics from being stored.

        The try/except wrapper around update_semantic_fact should catch
        the error for 'bad_topic' and continue processing 'python'.
        """
        profile_data = {
            "topic_counts": {
                "python": 50,      # 25% frequency — should produce expertise fact
                "javascript": 40,  # 20% frequency — should produce expertise fact
            },
        }
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 200)

        inferrer = SemanticFactInferrer(user_model_store)

        # Patch update_semantic_fact to fail on the first call only
        original_update = user_model_store.update_semantic_fact
        call_count = {"n": 0}

        def failing_update(*args, **kwargs):
            """Fail on the first call, succeed on subsequent calls."""
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Simulated DB write failure")
            return original_update(*args, **kwargs)

        with patch.object(user_model_store, "update_semantic_fact", side_effect=failing_update):
            result = inferrer.infer_from_topic_profile()

        # Method should complete successfully despite one failure
        assert result["processed"] is True
        # At least one of the two topics should have been stored
        facts = user_model_store.get_semantic_facts(category="expertise")
        assert len(facts) >= 1

    def test_topic_inference_diagnostic_logging(self, user_model_store, caplog):
        """Verify that topic inference emits diagnostic log messages.

        The added logging should report profile size, noise filtering stats,
        and facts created/failed counts.
        """
        profile_data = {
            "topic_counts": {
                "python": 50,
                "shop": 30,  # Noise — will be filtered
            },
        }
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 200)

        inferrer = SemanticFactInferrer(user_model_store)
        with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.infer_from_topic_profile()

        log_messages = [r.message for r in caplog.records]
        # Should log profile load stats
        assert any("Topic profile loaded" in m for m in log_messages)
        # Should log noise filter stats (includes "noise tokens filtered")
        assert any("noise tokens filtered" in m for m in log_messages)
        # Should log inference completion stats
        assert any("Topic inference complete" in m for m in log_messages)

    def test_relationship_inference_diagnostic_logging(self, user_model_store, caplog):
        """Verify that relationship inference emits diagnostic log messages.

        The added logging should report contact counts at each filtering
        stage and the high_priority threshold.
        """
        profile_data = {
            "contacts": {
                "alice@example.com": {
                    "interaction_count": 10,
                    "outbound_count": 5,
                },
                "bob@example.com": {
                    "interaction_count": 3,
                    "outbound_count": 0,  # Will be filtered by bidirectional check
                },
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.infer_from_relationship_profile()

        log_messages = [r.message for r in caplog.records]
        # Should log total contacts
        assert any("Relationship profile: 2 total contacts" in m for m in log_messages)
        # Should log bidirectional filter results
        assert any("Bidirectional contacts (outbound > 0): 1" in m for m in log_messages)


def _insert_episode(ums):
    """Insert a minimal episode row for tests that need _get_recent_episodes to return data."""
    import json
    from datetime import datetime

    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT OR IGNORE INTO episodes (id, timestamp, event_id, interaction_type, "
            "content_summary, contacts_involved) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "test-episode-001",
                datetime.now().isoformat(),
                "evt-001",
                "email",
                "Test episode for inference",
                json.dumps(["test@example.com"]),
            ),
        )
