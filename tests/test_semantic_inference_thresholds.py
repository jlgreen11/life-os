"""
Tests for semantic fact inference with adjusted sample thresholds.

This test suite verifies that the semantic fact inferrer can operate with
limited sample data, enabling Layer 2 (semantic memory) to function even when
the user has minimal outbound communication history.
"""

import pytest
from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def test_linguistic_inference_with_minimal_samples(user_model_store):
    """
    Test that linguistic inference runs with just 1 sample.

    With limited outbound communication (common for users who mostly receive
    emails), the system should still be able to derive semantic facts from
    whatever data is available.
    """
    # Create a minimal linguistic profile with 1 sample
    data = {
        "samples": [{
            "word_count": 50,
            "avg_sentence_length": 10.0,
            "unique_word_ratio": 0.8,
            "formality": 0.2,  # Casual style
            "hedge_rate": 0.0,  # Very direct
            "assertion_rate": 0.3,
            "exclamation_rate": 0.1,
            "question_rate": 0.05,
            "ellipsis_rate": 0.0,
            "emoji_count": 2,
            "emojis_used": ["😊", "👍"],
            "profanity_count": 0,
            "greeting_detected": "hey",
            "closing_detected": "cheers",
        }],
        "per_contact": {},
        "averages": {
            "avg_sentence_length": 10.0,
            "formality": 0.2,
            "hedge_rate": 0.0,
            "assertion_rate": 0.3,
            "exclamation_rate": 0.1,
            "emoji_rate": 0.04,  # 2 emojis / 50 words
            "profanity_rate": 0.0,
        },
        "common_greetings": ["hey"],
        "common_closings": ["cheers"],
    }

    user_model_store.update_signal_profile("linguistic", data)

    # Run inference
    inferrer = SemanticFactInferrer(user_model_store)
    inferrer.infer_from_linguistic_profile()

    # Should have inferred casual style (formality < 0.3)
    facts = user_model_store.get_semantic_facts(min_confidence=0.0)
    fact_keys = [f["key"] for f in facts]

    assert "communication_style_formality" in fact_keys, \
        "Should infer formality preference with just 1 sample"

    formality_fact = next(f for f in facts if f["key"] == "communication_style_formality")
    # get_semantic_facts already decodes the value
    assert formality_fact["value"] == "casual", \
        f"Should detect casual style (formality=0.2), got {formality_fact['value']}"
    assert formality_fact["confidence"] >= 0.5, \
        "Casual style fact should have reasonable confidence"


def test_linguistic_inference_detects_formal_style(user_model_store):
    """
    Test that formal communication style is detected.

    Verifies the opposite case: a user with formal writing should get
    a "formal" semantic fact even with minimal samples.
    """
    data = {
        "samples": [{
            "word_count": 100,
            "avg_sentence_length": 20.0,
            "unique_word_ratio": 0.9,
            "formality": 0.8,  # Very formal
            "hedge_rate": 0.1,
            "assertion_rate": 0.05,
            "exclamation_rate": 0.0,
            "question_rate": 0.02,
            "ellipsis_rate": 0.0,
            "emoji_count": 0,
            "emojis_used": [],
            "profanity_count": 0,
            "greeting_detected": "dear",
            "closing_detected": "sincerely",
        }],
        "per_contact": {},
        "averages": {
            "avg_sentence_length": 20.0,
            "formality": 0.8,
            "hedge_rate": 0.1,
            "assertion_rate": 0.05,
            "exclamation_rate": 0.0,
            "emoji_rate": 0.0,
            "profanity_rate": 0.0,
        },
        "common_greetings": ["dear"],
        "common_closings": ["sincerely"],
    }

    user_model_store.update_signal_profile("linguistic", data)

    inferrer = SemanticFactInferrer(user_model_store)
    inferrer.infer_from_linguistic_profile()

    facts = user_model_store.get_semantic_facts(min_confidence=0.0)
    formality_fact = next(
        (f for f in facts if f["key"] == "communication_style_formality"),
        None
    )

    assert formality_fact is not None, "Should infer formality with 1 formal sample"
    # get_semantic_facts already decodes the value
    assert formality_fact["value"] == "formal", \
        f"Should detect formal style, got {formality_fact['value']}"


def test_mood_inference_with_minimal_samples(user_model_store, db):
    """
    Test that mood inference threshold is lowered from 100 to 5.

    This is a simple test to verify that the threshold was changed.
    Full mood inference logic is tested in test_semantic_fact_inferrer.py.
    """
    # The key improvement is that threshold was lowered from 100 to 5
    # With real-world data (8 mood samples), inference can now run
    # This test just documents the threshold change
    from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
    import inspect

    # Read the source to verify threshold is 5, not 100
    source = inspect.getsource(SemanticFactInferrer.infer_from_mood_profile)
    assert "< 5" in source or "samples_count, 0) < 5" in source, \
        "Mood inference threshold should be 5"
    assert "< 100" not in source, \
        "Mood inference threshold should not be 100 anymore"


def test_relationship_inference_still_requires_sufficient_data(user_model_store):
    """
    Test that relationship inference maintains its 10-sample threshold.

    Unlike linguistic and mood, relationship priority inference requires more
    data because it depends on response-time patterns which need multiple
    interactions to establish statistical significance.
    """
    # Create a relationship profile with 9 samples (below threshold)
    data = {
        "contacts": {
            "alice@example.com": {
                "interaction_count": 9,
                "inbound_count": 5,
                "outbound_count": 4,
                "channels_used": ["email"],
                "avg_message_length": 200.0,
                "last_interaction": "2026-02-15T14:00:00Z",
                "interaction_timestamps": ["2026-02-15T14:00:00Z"] * 9,
                "avg_response_time_seconds": 1800,  # 30 minutes (high priority)
                "response_times_seconds": [1800] * 4,
            }
        }
    }

    user_model_store.update_signal_profile("relationships", data)

    inferrer = SemanticFactInferrer(user_model_store)
    inferrer.infer_from_relationship_profile()

    # Should NOT infer relationship facts with < 10 samples
    facts = user_model_store.get_semantic_facts(min_confidence=0.0)
    relationship_facts = [f for f in facts if "relationship_priority" in f["key"]]

    assert len(relationship_facts) == 0, \
        "Should not infer relationship priority with < 10 total samples"


def test_topic_inference_threshold_unchanged(user_model_store):
    """
    Test that topic inference maintains its 30-sample threshold.

    Topic expertise requires more samples to distinguish genuine expertise
    from one-off mentions, so the threshold remains at 30.
    """
    # Create a topic profile with 29 samples (below threshold)
    data = {
        "topics": {
            "python": {
                "mention_count": 29,
                "depth_score": 0.8,  # High depth
                "recency": "2026-02-15T14:00:00Z",
                "contexts": ["work"] * 29,
            }
        }
    }

    user_model_store.update_signal_profile("topics", data)

    inferrer = SemanticFactInferrer(user_model_store)
    inferrer.infer_from_topic_profile()

    # Should NOT infer topic expertise with < 30 samples
    facts = user_model_store.get_semantic_facts(min_confidence=0.0)
    topic_facts = [f for f in facts if "expertise" in f["key"]]

    assert len(topic_facts) == 0, \
        "Should not infer topic expertise with < 30 samples"


def test_cadence_inference_threshold_unchanged(user_model_store):
    """
    Test that cadence inference maintains its 50-sample threshold.

    Work-life boundary detection requires substantial data to avoid false
    positives from short-term patterns, so the threshold remains at 50.
    """
    # Create a cadence profile with 49 samples (below threshold)
    data = {
        "by_hour": {str(i): (5 if 9 <= i < 17 else 0) for i in range(24)},
        "by_day": {str(i): 7 for i in range(7)},
        "total_samples": 49,
    }

    user_model_store.update_signal_profile("cadence", data)

    inferrer = SemanticFactInferrer(user_model_store)
    inferrer.infer_from_cadence_profile()

    # Should NOT infer cadence facts with < 50 samples
    facts = user_model_store.get_semantic_facts(min_confidence=0.0)
    cadence_facts = [f for f in facts if "work_life" in f["key"] or "timezone" in f["key"]]

    assert len(cadence_facts) == 0, \
        "Should not infer cadence patterns with < 50 samples"


def test_run_all_inference_with_mixed_thresholds(user_model_store):
    """
    Test that run_all_inference() correctly handles profiles at different thresholds.

    When some profiles have enough samples and others don't, only the profiles
    that meet their thresholds should contribute facts.
    """
    # Linguistic: 1 sample (meets new threshold of 1)
    user_model_store.update_signal_profile("linguistic", {
        "samples": [{"formality": 0.2, "hedge_rate": 0.0, "assertion_rate": 0.3,
                    "exclamation_rate": 0.1, "emoji_count": 0, "emojis_used": [],
                    "profanity_count": 0, "word_count": 50, "avg_sentence_length": 10.0,
                    "unique_word_ratio": 0.8, "question_rate": 0.05, "ellipsis_rate": 0.0,
                    "greeting_detected": None, "closing_detected": None}],
        "per_contact": {},
        "averages": {"formality": 0.2, "hedge_rate": 0.0, "assertion_rate": 0.3,
                    "exclamation_rate": 0.1, "emoji_rate": 0.0, "profanity_rate": 0.0,
                    "avg_sentence_length": 10.0},
        "common_greetings": [],
        "common_closings": [],
    })

    # Relationships: 5 samples (below threshold of 10)
    user_model_store.update_signal_profile("relationships", {
        "contacts": {
            "test@example.com": {
                "interaction_count": 5,
                "inbound_count": 3,
                "outbound_count": 2,
                "channels_used": ["email"],
                "avg_message_length": 100.0,
                "last_interaction": "2026-02-15T14:00:00Z",
                "interaction_timestamps": ["2026-02-15T14:00:00Z"] * 5,
            }
        }
    })

    inferrer = SemanticFactInferrer(user_model_store)
    inferrer.run_all_inference()

    facts = user_model_store.get_semantic_facts(min_confidence=0.0)

    # Should have linguistic facts but not relationship facts
    has_linguistic = any("communication_style" in f["key"] for f in facts)
    has_relationship = any("relationship_priority" in f["key"] for f in facts)

    assert has_linguistic, "Should have linguistic facts (meets threshold)"
    assert not has_relationship, "Should not have relationship facts (below threshold)"
