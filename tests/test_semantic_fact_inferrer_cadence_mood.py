"""
Tests for Semantic Fact Inference — Cadence & Mood Profiles

Comprehensive unit tests for the two inference methods that derive
work-life boundaries, peak communication patterns, quiet hours, domain
response priorities, stress baselines, and incoming pressure exposure
from cadence and mood signal profiles.
"""

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


# ===================================================================
# Cadence Profile Inference Tests
# ===================================================================


class TestCadenceInsufficientSamples:
    """Verify cadence inference is skipped when samples are too few."""

    def test_insufficient_samples_returns_not_processed(self, user_model_store):
        """With <25 samples, cadence inference returns processed=False."""
        profile_data = {
            "hourly_activity": {str(h): 5 for h in range(24)},
        }
        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 20)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_cadence_profile()

        assert result["processed"] is False
        assert result["reason"] == "insufficient samples (<25)"

    def test_insufficient_samples_stores_no_facts(self, user_model_store):
        """With <25 samples, no semantic facts are created."""
        profile_data = {
            "hourly_activity": {str(h): 10 for h in range(24)},
        }
        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 24)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 0

    def test_no_cadence_profile_returns_not_processed(self, user_model_store):
        """With no cadence profile at all, inference returns processed=False."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_cadence_profile()

        assert result["processed"] is False
        assert result["reason"] == "insufficient samples (<25)"


class TestCadenceWorkLifeBoundaries:
    """Verify work-life boundary inference from hourly activity patterns."""

    def test_strict_boundaries_above_90_percent(self, user_model_store):
        """When >90% of messages are in business hours (9-17), infer strict_boundaries."""
        # 100 messages per business hour (9-17 = 9 hours), small off-hours
        hourly = {str(h): 0 for h in range(24)}
        for h in range(9, 18):  # 9 through 17 inclusive
            hourly[str(h)] = 100
        # Add tiny off-hours to make ratio realistic but still >90%
        hourly["7"] = 2
        hourly["19"] = 3

        user_model_store.update_signal_profile("cadence", {"hourly_activity": hourly})
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_cadence_profile()

        assert result["processed"] is True
        facts = user_model_store.get_semantic_facts(category="values")
        boundary_fact = next((f for f in facts if f["key"] == "work_life_boundaries"), None)

        assert boundary_fact is not None
        assert boundary_fact["value"] == "strict_boundaries"
        assert boundary_fact["confidence"] > 0

    def test_flexible_boundaries_below_30_percent(self, user_model_store):
        """When <30% of messages are in business hours, infer flexible_boundaries."""
        # Spread messages mostly outside business hours
        hourly = {str(h): 0 for h in range(24)}
        # 20 messages during business hours
        hourly["10"] = 10
        hourly["14"] = 10
        # 80 messages outside business hours
        for h in [0, 1, 2, 3, 20, 21, 22, 23]:
            hourly[str(h)] = 10

        user_model_store.update_signal_profile("cadence", {"hourly_activity": hourly})
        _set_samples(user_model_store, "cadence", 60)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_cadence_profile()

        assert result["processed"] is True
        facts = user_model_store.get_semantic_facts(category="values")
        boundary_fact = next((f for f in facts if f["key"] == "work_life_boundaries"), None)

        assert boundary_fact is not None
        assert boundary_fact["value"] == "flexible_boundaries"
        assert boundary_fact["confidence"] > 0

    def test_no_boundary_fact_for_mixed_schedule(self, user_model_store):
        """When business-hours ratio is between 30-90%, no boundary fact is stored."""
        # Even spread across all hours → ~37.5% business hours
        hourly = {str(h): 10 for h in range(24)}

        user_model_store.update_signal_profile("cadence", {"hourly_activity": hourly})
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="values")
        boundary_fact = next((f for f in facts if f["key"] == "work_life_boundaries"), None)
        assert boundary_fact is None


class TestCadencePeakCommunicationHour:
    """Verify peak communication hour inference from hourly activity spikes."""

    def test_peak_hour_stored_when_above_20_percent(self, user_model_store):
        """An hour with >20% of all messages should be stored as peak_communication_hour."""
        hourly = {str(h): 0 for h in range(24)}
        # 50 messages at hour 10, only 10 elsewhere = 60 total → 50/60 ≈ 83%
        hourly["10"] = 50
        hourly["14"] = 5
        hourly["16"] = 5

        user_model_store.update_signal_profile("cadence", {"hourly_activity": hourly})
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        peak_fact = next((f for f in facts if f["key"] == "peak_communication_hour"), None)

        assert peak_fact is not None
        assert peak_fact["value"] == 10  # Stored as integer
        assert peak_fact["confidence"] > 0.5

    def test_no_peak_hour_when_evenly_distributed(self, user_model_store):
        """With evenly distributed messages, no single hour exceeds 20%."""
        # 10 messages per hour across 10 hours → each is 10% = below 20%
        hourly = {str(h): 0 for h in range(24)}
        for h in range(9, 19):
            hourly[str(h)] = 10

        user_model_store.update_signal_profile("cadence", {"hourly_activity": hourly})
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        peak_fact = next((f for f in facts if f["key"] == "peak_communication_hour"), None)
        assert peak_fact is None


class TestCadencePeakHoursList:
    """Verify peak_hours derived metric from cadence profile."""

    def test_peak_hours_stored_as_fact(self, user_model_store):
        """peak_hours list with 2+ entries gets stored as peak_communication_hours fact."""
        profile_data = {
            "hourly_activity": {str(h): 0 for h in range(24)},
            "peak_hours": [9, 10, 11, 14],
        }

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 60)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        peak_hours_fact = next((f for f in facts if f["key"] == "peak_communication_hours"), None)

        assert peak_hours_fact is not None
        assert peak_hours_fact["value"] == [9, 10, 11, 14]
        assert peak_hours_fact["confidence"] > 0.5

    def test_single_peak_hour_not_stored(self, user_model_store):
        """A peak_hours list with only 1 entry is too noisy to store."""
        profile_data = {
            "hourly_activity": {str(h): 0 for h in range(24)},
            "peak_hours": [10],
        }

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        peak_hours_fact = next((f for f in facts if f["key"] == "peak_communication_hours"), None)
        assert peak_hours_fact is None

    def test_empty_peak_hours_not_stored(self, user_model_store):
        """An empty peak_hours list creates no fact."""
        profile_data = {
            "hourly_activity": {str(h): 0 for h in range(24)},
            "peak_hours": [],
        }

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        peak_hours_fact = next((f for f in facts if f["key"] == "peak_communication_hours"), None)
        assert peak_hours_fact is None


class TestCadenceQuietHours:
    """Verify quiet hours inference from quiet_hours_observed data."""

    def test_quiet_window_stored_for_3_plus_hour_span(self, user_model_store):
        """A quiet window spanning >=3 hours should be stored as observed_quiet_window."""
        profile_data = {
            "hourly_activity": {str(h): 0 for h in range(24)},
            "quiet_hours_observed": [(22, 6)],  # 8-hour window crossing midnight
        }

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        quiet_fact = next((f for f in facts if f["key"] == "observed_quiet_window"), None)

        assert quiet_fact is not None
        assert quiet_fact["value"] == "22:00-06:00 UTC"
        assert quiet_fact["confidence"] > 0.5

    def test_short_quiet_window_not_stored(self, user_model_store):
        """A quiet window shorter than 3 hours should not be stored."""
        profile_data = {
            "hourly_activity": {str(h): 0 for h in range(24)},
            "quiet_hours_observed": [(2, 4)],  # Only 2-hour window
        }

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        quiet_fact = next((f for f in facts if f["key"] == "observed_quiet_window"), None)
        assert quiet_fact is None

    def test_empty_quiet_hours_no_fact(self, user_model_store):
        """No quiet windows in data means no fact stored."""
        profile_data = {
            "hourly_activity": {str(h): 0 for h in range(24)},
            "quiet_hours_observed": [],
        }

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        quiet_fact = next((f for f in facts if f["key"] == "observed_quiet_window"), None)
        assert quiet_fact is None


class TestCadenceDomainResponseTimes:
    """Verify domain response time inference from avg_response_time_by_domain."""

    def test_fast_domain_gets_high_priority_fact(self, user_model_store):
        """A domain with <300s avg response time gets stored as high_priority_domain."""
        profile_data = {
            "hourly_activity": {str(h): 0 for h in range(24)},
            "avg_response_time_by_domain": {
                "company.com": 180,    # 3 minutes — fast
                "partner.org": 600,    # 10 minutes — moderate
            },
        }

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        fast_fact = next((f for f in facts if f["key"] == "high_priority_domain_company.com"), None)

        assert fast_fact is not None
        assert "avg_reply=3min" in fast_fact["value"]
        assert fast_fact["confidence"] > 0.7

    def test_slow_domain_above_3600s_not_stored(self, user_model_store):
        """Domains with >3600s avg response should NOT get a high_priority_domain fact."""
        profile_data = {
            "hourly_activity": {str(h): 0 for h in range(24)},
            "avg_response_time_by_domain": {
                "newsletter.com": 7200,   # 2 hours — slow
                "spam-domain.io": 14400,  # 4 hours — very slow
            },
        }

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        domain_facts = [f for f in facts if "high_priority_domain" in f["key"]]
        assert len(domain_facts) == 0

    def test_max_three_fast_domains_stored(self, user_model_store):
        """At most 3 domain response facts are created, even with more fast domains."""
        profile_data = {
            "hourly_activity": {str(h): 0 for h in range(24)},
            "avg_response_time_by_domain": {
                "a.com": 60,
                "b.com": 120,
                "c.com": 180,
                "d.com": 240,
                "e.com": 300,
            },
        }

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        domain_facts = [f for f in facts if "high_priority_domain" in f["key"]]
        assert len(domain_facts) == 3

    def test_single_domain_not_stored(self, user_model_store):
        """With only 1 domain, no response time facts are created (requires >=2)."""
        profile_data = {
            "hourly_activity": {str(h): 0 for h in range(24)},
            "avg_response_time_by_domain": {
                "solo.com": 120,
            },
        }

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        domain_facts = [f for f in facts if "high_priority_domain" in f["key"]]
        assert len(domain_facts) == 0


class TestCadenceEarlyInferenceConfidence:
    """Verify early inference confidence scaling for cadence profiles."""

    def test_30_samples_has_lower_confidence_than_100(self, user_model_store):
        """Samples=30 (between 25 and 50 thresholds) should yield lower confidence than samples=100."""
        hourly_strict = {str(h): 0 for h in range(24)}
        for h in range(9, 18):
            hourly_strict[str(h)] = 100
        hourly_strict["7"] = 2

        # First: 30 samples (early inference)
        user_model_store.update_signal_profile("cadence", {"hourly_activity": hourly_strict})
        _set_samples(user_model_store, "cadence", 30)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts_early = user_model_store.get_semantic_facts(category="values")
        early_fact = next((f for f in facts_early if f["key"] == "work_life_boundaries"), None)
        assert early_fact is not None
        early_confidence = early_fact["confidence"]

        # Now create a fresh store with 100 samples
        # We need a fresh user_model_store to avoid fact confidence accumulation
        # Instead, we just verify the early confidence scaling math directly
        inferrer2 = SemanticFactInferrer(user_model_store)
        conf_30 = inferrer2._early_inference_confidence(30, old_threshold=50)
        conf_100 = inferrer2._early_inference_confidence(100, old_threshold=50)

        assert conf_30 < conf_100, (
            f"Confidence at 30 samples ({conf_30}) should be less than at 100 samples ({conf_100})"
        )

    def test_exactly_25_samples_has_lowest_cadence_confidence(self, user_model_store):
        """At exactly 25 samples (minimum threshold), confidence is at its lowest."""
        inferrer = SemanticFactInferrer(user_model_store)
        conf_25 = inferrer._early_inference_confidence(25, old_threshold=50)
        conf_50 = inferrer._early_inference_confidence(50, old_threshold=50)

        assert conf_25 < conf_50
        assert conf_25 >= 0.3  # Lower bound from _early_inference_confidence


class TestCadenceFactProvenance:
    """Verify that cadence facts are stored with correct keys, categories, and metadata."""

    def test_boundary_fact_has_correct_category(self, user_model_store):
        """Work-life boundary facts should use category='values'."""
        hourly = {str(h): 0 for h in range(24)}
        for h in range(9, 18):
            hourly[str(h)] = 100

        user_model_store.update_signal_profile("cadence", {"hourly_activity": hourly})
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="values")
        boundary_fact = next((f for f in facts if f["key"] == "work_life_boundaries"), None)
        assert boundary_fact is not None
        assert boundary_fact["category"] == "values"

    def test_peak_hour_fact_has_correct_category(self, user_model_store):
        """Peak communication hour facts should use category='implicit_preference'."""
        hourly = {str(h): 0 for h in range(24)}
        hourly["10"] = 50
        hourly["14"] = 5

        user_model_store.update_signal_profile("cadence", {"hourly_activity": hourly})
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        peak_fact = next((f for f in facts if f["key"] == "peak_communication_hour"), None)
        assert peak_fact is not None
        assert peak_fact["category"] == "implicit_preference"

    def test_cadence_return_value_on_success(self, user_model_store):
        """Successful cadence inference returns processed=True."""
        profile_data = {"hourly_activity": {str(h): 0 for h in range(24)}}
        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_cadence_profile()

        assert result["type"] == "cadence"
        assert result["processed"] is True
        assert result["reason"] is None


class TestCadenceStringHourKeys:
    """Verify that hourly_activity keys are correctly handled as strings."""

    def test_string_hour_keys_parsed_correctly(self, user_model_store):
        """The inferrer must handle string keys ('0'-'23') in hourly_activity.

        The cadence extractor stores hourly data with STRING keys, and
        the inferrer converts them with int(hour) at line 697.
        """
        # Use explicit string keys to verify parsing
        hourly = {}
        for h in range(24):
            hourly[str(h)] = 100 if 9 <= h <= 17 else 1

        user_model_store.update_signal_profile("cadence", {"hourly_activity": hourly})
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        # Should not raise KeyError or TypeError from int(hour) conversion
        result = inferrer.infer_from_cadence_profile()
        assert result["processed"] is True


# ===================================================================
# Mood Profile Inference Tests
# ===================================================================


class TestMoodInsufficientSamples:
    """Verify mood inference is skipped when samples are too few."""

    def test_insufficient_samples_returns_not_processed(self, user_model_store):
        """With <3 samples, mood inference returns processed=False."""
        recent_signals = [{"signal_type": "negative_language", "value": 0.8}]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 2)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_mood_profile()

        assert result["processed"] is False
        assert result["reason"] == "insufficient samples (<3)"

    def test_insufficient_samples_stores_no_facts(self, user_model_store):
        """With <3 samples, no mood facts are created."""
        recent_signals = [{"signal_type": "negative_language", "value": 0.9}]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 1)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 0

    def test_no_mood_profile_returns_not_processed(self, user_model_store):
        """With no mood profile at all, inference returns processed=False."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_mood_profile()

        assert result["processed"] is False
        assert result["reason"] == "insufficient samples (<3)"


class TestMoodEmptySignals:
    """Verify handling of empty recent_signals."""

    def test_empty_recent_signals_returns_processed_true(self, user_model_store):
        """With enough samples but empty recent_signals, returns processed=True."""
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": []})
        _set_samples(user_model_store, "mood_signals", 10)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_mood_profile()

        assert result["processed"] is True
        assert result["reason"] is None

    def test_empty_recent_signals_stores_no_facts(self, user_model_store):
        """With empty recent_signals, no mood facts should be stored."""
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": []})
        _set_samples(user_model_store, "mood_signals", 10)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 0


class TestMoodHighStress:
    """Verify high stress baseline inference from negative language signals."""

    def test_high_stress_above_30_percent(self, user_model_store):
        """When >30% of signals are negative_language, infer high_stress baseline."""
        # 4 out of 10 signals = 40% negative → high stress
        recent_signals = (
            [{"signal_type": "negative_language", "value": 0.8} for _ in range(4)]
            + [{"signal_type": "positive_language", "value": 0.7} for _ in range(6)]
        )
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        stress_fact = next((f for f in facts if f["key"] == "stress_baseline"), None)

        assert stress_fact is not None
        assert stress_fact["value"] == "high_stress"

    def test_incoming_negative_language_counts_as_stress(self, user_model_store):
        """Both negative_language and incoming_negative_language count toward stress ratio."""
        # Mix of outgoing and incoming negative — total = 4/10 = 40% > 30%
        recent_signals = (
            [{"signal_type": "negative_language", "value": 0.8} for _ in range(2)]
            + [{"signal_type": "incoming_negative_language", "value": 0.7} for _ in range(2)]
            + [{"signal_type": "positive_language", "value": 0.6} for _ in range(6)]
        )
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        stress_fact = next((f for f in facts if f["key"] == "stress_baseline"), None)

        assert stress_fact is not None
        assert stress_fact["value"] == "high_stress"


class TestMoodLowStress:
    """Verify low stress baseline inference from predominantly positive signals."""

    def test_low_stress_below_10_percent(self, user_model_store):
        """When <10% of signals are negative_language, infer low_stress baseline."""
        # 0 out of 10 signals are negative → 0% < 10%
        recent_signals = [
            {"signal_type": "positive_language", "value": 0.8} for _ in range(10)
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        stress_fact = next((f for f in facts if f["key"] == "stress_baseline"), None)

        assert stress_fact is not None
        assert stress_fact["value"] == "low_stress"

    def test_no_stress_fact_for_moderate_stress(self, user_model_store):
        """When stress ratio is between 10-30%, no stress_baseline fact is stored."""
        # 2 out of 10 = 20% negative → between 10% and 30%
        recent_signals = (
            [{"signal_type": "negative_language", "value": 0.8} for _ in range(2)]
            + [{"signal_type": "positive_language", "value": 0.7} for _ in range(8)]
        )
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        stress_fact = next((f for f in facts if f["key"] == "stress_baseline"), None)
        assert stress_fact is None


class TestMoodIncomingPressure:
    """Verify incoming pressure exposure inference."""

    def test_high_pressure_above_20_percent(self, user_model_store):
        """When >20% of signals are incoming_pressure, infer high_pressure_environment."""
        # 3 out of 10 = 30% incoming pressure
        recent_signals = (
            [{"signal_type": "incoming_pressure", "value": 0.9} for _ in range(3)]
            + [{"signal_type": "positive_language", "value": 0.6} for _ in range(7)]
        )
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        pressure_fact = next((f for f in facts if f["key"] == "incoming_pressure_exposure"), None)

        assert pressure_fact is not None
        assert pressure_fact["value"] == "high_pressure_environment"

    def test_no_pressure_fact_below_20_percent(self, user_model_store):
        """When incoming_pressure signals are <=20%, no pressure fact is stored."""
        # 1 out of 10 = 10% incoming pressure → below 20%
        recent_signals = (
            [{"signal_type": "incoming_pressure", "value": 0.9}]
            + [{"signal_type": "positive_language", "value": 0.6} for _ in range(9)]
        )
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        pressure_fact = next((f for f in facts if f["key"] == "incoming_pressure_exposure"), None)
        assert pressure_fact is None


class TestMoodConfidenceCapping:
    """Verify that mood facts have confidence capped correctly."""

    def test_high_stress_confidence_capped_at_075(self, user_model_store):
        """High stress facts should have confidence capped at min(0.75, ...)."""
        # All signals are negative to maximize stress_ratio
        recent_signals = [{"signal_type": "negative_language", "value": 0.9} for _ in range(10)]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 200)  # High samples for max base confidence

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        stress_fact = next((f for f in facts if f["key"] == "stress_baseline"), None)

        assert stress_fact is not None
        # Confidence formula: min(0.75, base_confidence + stress_ratio)
        # With 200 samples (>> old threshold 5), base_confidence = 0.5
        # stress_ratio = 1.0, so base_confidence + stress_ratio = 1.5
        # min(0.75, 1.5) = 0.75
        assert stress_fact["confidence"] <= 0.75

    def test_low_stress_confidence_capped_at_080(self, user_model_store):
        """Low stress facts should have confidence capped at min(0.8, ...)."""
        recent_signals = [{"signal_type": "positive_language", "value": 0.9} for _ in range(10)]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 200)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        stress_fact = next((f for f in facts if f["key"] == "stress_baseline"), None)

        assert stress_fact is not None
        # Confidence formula: min(0.8, base_confidence + (0.1 - stress_ratio) * 3)
        # base_confidence = 0.5, stress_ratio = 0.0
        # 0.5 + 0.1 * 3 = 0.8 → min(0.8, 0.8) = 0.8
        assert stress_fact["confidence"] <= 0.8

    def test_pressure_confidence_capped_at_080(self, user_model_store):
        """Pressure facts should have confidence capped at min(0.8, ...)."""
        # All signals are incoming_pressure to maximize confidence
        recent_signals = [{"signal_type": "incoming_pressure", "value": 0.9} for _ in range(10)]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 200)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        pressure_fact = next((f for f in facts if f["key"] == "incoming_pressure_exposure"), None)

        assert pressure_fact is not None
        # Confidence formula: min(0.8, base_confidence + pressure_ratio * 2)
        # base_confidence = 0.5, pressure_ratio = 1.0
        # 0.5 + 1.0 * 2 = 2.5 → min(0.8, 2.5) = 0.8
        assert pressure_fact["confidence"] <= 0.8


class TestMoodEarlyInferenceConfidence:
    """Verify early inference confidence scaling for mood profiles."""

    def test_3_samples_has_lower_confidence_than_10(self, user_model_store):
        """Samples=3 (minimum threshold) should yield lower confidence than samples=10."""
        inferrer = SemanticFactInferrer(user_model_store)

        conf_3 = inferrer._early_inference_confidence(3, old_threshold=5)
        conf_10 = inferrer._early_inference_confidence(10, old_threshold=5)

        assert conf_3 < conf_10, (
            f"Confidence at 3 samples ({conf_3}) should be less than at 10 samples ({conf_10})"
        )

    def test_4_samples_between_3_and_5(self, user_model_store):
        """Samples=4 (between new=3 and old=5 thresholds) scales linearly."""
        inferrer = SemanticFactInferrer(user_model_store)

        conf_3 = inferrer._early_inference_confidence(3, old_threshold=5)
        conf_4 = inferrer._early_inference_confidence(4, old_threshold=5)
        conf_5 = inferrer._early_inference_confidence(5, old_threshold=5)

        assert conf_3 < conf_4 < conf_5

    def test_early_mood_inference_produces_lower_stress_confidence(self, user_model_store):
        """With only 3 mood samples, inferred stress fact should have lower confidence."""
        recent_signals = [{"signal_type": "positive_language", "value": 0.8} for _ in range(10)]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 3)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        stress_fact = next((f for f in facts if f["key"] == "stress_baseline"), None)

        assert stress_fact is not None
        assert stress_fact["value"] == "low_stress"
        # With 3 samples and old_threshold=5, base_confidence should be:
        # 0.3 + (0.5 - 0.3) * (3/5) = 0.3 + 0.12 = 0.42
        # Much lower than the full 0.5 base confidence
        assert stress_fact["confidence"] < 0.8


class TestMoodFactProvenance:
    """Verify mood facts are stored with correct metadata."""

    def test_stress_fact_has_correct_category(self, user_model_store):
        """Stress baseline facts should use category='implicit_preference'."""
        recent_signals = [{"signal_type": "positive_language", "value": 0.8} for _ in range(10)]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        stress_fact = next((f for f in facts if f["key"] == "stress_baseline"), None)
        assert stress_fact is not None
        assert stress_fact["category"] == "implicit_preference"

    def test_pressure_fact_has_correct_key(self, user_model_store):
        """Incoming pressure fact uses key='incoming_pressure_exposure'."""
        recent_signals = (
            [{"signal_type": "incoming_pressure", "value": 0.9} for _ in range(5)]
            + [{"signal_type": "positive_language", "value": 0.6} for _ in range(5)]
        )
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        pressure_fact = next((f for f in facts if f["key"] == "incoming_pressure_exposure"), None)
        assert pressure_fact is not None

    def test_mood_return_value_on_success(self, user_model_store):
        """Successful mood inference returns type='mood' and processed=True."""
        recent_signals = [{"signal_type": "positive_language", "value": 0.8}]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 10)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_mood_profile()

        assert result["type"] == "mood"
        assert result["processed"] is True
        assert result["reason"] is None


class TestCadenceMoodCombined:
    """Integration tests combining cadence and mood inference."""

    def test_both_profiles_produce_independent_facts(self, user_model_store):
        """Cadence and mood inferences produce facts that don't interfere with each other."""
        # Set up cadence profile
        hourly = {str(h): 0 for h in range(24)}
        for h in range(9, 18):
            hourly[str(h)] = 100
        user_model_store.update_signal_profile("cadence", {"hourly_activity": hourly})
        _set_samples(user_model_store, "cadence", 50)

        # Set up mood profile
        recent_signals = [{"signal_type": "positive_language", "value": 0.8} for _ in range(10)]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": recent_signals})
        _set_samples(user_model_store, "mood_signals", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        cadence_result = inferrer.infer_from_cadence_profile()
        mood_result = inferrer.infer_from_mood_profile()

        assert cadence_result["processed"] is True
        assert mood_result["processed"] is True

        # Both should produce facts
        all_facts = user_model_store.get_semantic_facts()
        fact_keys = {f["key"] for f in all_facts}

        assert "work_life_boundaries" in fact_keys
        assert "stress_baseline" in fact_keys
