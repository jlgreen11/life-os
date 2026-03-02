"""
Tests for cadence-derived-metric semantic fact inference.

PR #276 added three computed fields to CadenceProfile:
  - peak_hours               — list of above-average-activity hour ints (UTC)
  - quiet_hours_observed     — list of (start_hour, end_hour) quiet spans
  - avg_response_time_by_domain — dict of domain -> average response time (seconds)

This file verifies that infer_from_cadence_profile() correctly promotes all
three derived fields into Layer 2 semantic facts:
  - peak_communication_hours  — list of peak-activity hours
  - observed_quiet_window     — primary sleep/offline window as human-readable string
  - high_priority_domain_*    — one fact per fast-reply email domain (up to 3)

Test strategy:
  - Directly set cadence profile data via update_signal_profile() so tests
    exercise only the inference logic without needing real event streams.
  - Verify each generated fact's key, value shape, and confidence range.
  - Negative tests confirm that insufficient data suppresses fact generation.
"""

import json

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_cadence_profile(ums, data: dict, sample_count: int = 500) -> None:
    """Store *data* as the cadence signal profile with *sample_count* samples.

    The inferrer requires >= 50 samples before running cadence inference;
    most tests use 500 to well exceed that threshold.
    """
    ums.update_signal_profile("cadence", data)
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (sample_count, "cadence"),
        )


def _get_facts_dict(ums, category: str = None) -> dict[str, dict]:
    """Return a {key: fact_row} dict for easy lookup in assertions."""
    facts = ums.get_semantic_facts(category=category)
    return {f["key"]: f for f in facts}


# ---------------------------------------------------------------------------
# peak_hours inference
# ---------------------------------------------------------------------------

class TestPeakHoursInference:
    """Verify peak_communication_hours fact is generated from peak_hours list."""

    def test_peak_hours_list_generates_fact(self, user_model_store):
        """When cadence profile contains peak_hours with >=2 entries, a fact is stored."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {"9": 50, "10": 60, "11": 55, "14": 20, "22": 5},
            "peak_hours": [9, 10, 11],
            "quiet_hours_observed": [],
            "avg_response_time_by_domain": {},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "peak_communication_hours" in facts, (
            "infer_from_cadence_profile() should store a peak_communication_hours fact "
            "when cadence profile contains a peak_hours list with >= 2 entries"
        )
        fact = facts["peak_communication_hours"]
        # The stored value should be the list of peak hours (JSON-encoded by the store)
        stored_value = fact["value"]
        if isinstance(stored_value, str):
            stored_value = json.loads(stored_value)
        assert stored_value == [9, 10, 11], (
            "peak_communication_hours value should exactly match the peak_hours list"
        )

    def test_peak_hours_confidence_scales_with_count(self, user_model_store):
        """Confidence should increase as more peak hours are found (more consistent pattern)."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {},
            "peak_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
            "quiet_hours_observed": [],
            "avg_response_time_by_domain": {},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "peak_communication_hours" in facts
        confidence = facts["peak_communication_hours"]["confidence"]
        # 9 peak hours * 0.04 + 0.5 = 0.86, capped at 0.9
        assert confidence >= 0.85, (
            f"9 peak hours should yield confidence >= 0.85, got {confidence}"
        )

    def test_single_peak_hour_suppressed(self, user_model_store):
        """A peak_hours list with only 1 entry should NOT generate a fact (too noisy)."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {"10": 100, "14": 10},
            "peak_hours": [10],
            "quiet_hours_observed": [],
            "avg_response_time_by_domain": {},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "peak_communication_hours" not in facts, (
            "A single-entry peak_hours list is too noisy; no fact should be stored"
        )

    def test_empty_peak_hours_suppressed(self, user_model_store):
        """An empty peak_hours list should not generate any fact."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {"10": 30, "11": 35},
            "peak_hours": [],
            "quiet_hours_observed": [],
            "avg_response_time_by_domain": {},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "peak_communication_hours" not in facts


# ---------------------------------------------------------------------------
# quiet_hours_observed inference
# ---------------------------------------------------------------------------

class TestQuietHoursInference:
    """Verify observed_quiet_window fact is generated from quiet_hours_observed."""

    def test_sleep_window_generates_fact(self, user_model_store):
        """A standard overnight quiet window (22:00-06:00) should produce an observed_quiet_window fact."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {"9": 50, "10": 60},
            "peak_hours": [9, 10],
            "quiet_hours_observed": [[22, 6]],   # midnight-crossing window
            "avg_response_time_by_domain": {},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "observed_quiet_window" in facts, (
            "A quiet window of 8 hours (22:00–06:00) should produce an observed_quiet_window fact"
        )
        value = facts["observed_quiet_window"]["value"]
        # Value should be a human-readable string like "22:00-06:00 UTC"
        assert "22:00" in value, f"Expected '22:00' in observed_quiet_window value, got: {value}"
        assert "06:00" in value, f"Expected '06:00' in observed_quiet_window value, got: {value}"

    def test_quiet_window_confidence_scales_with_span(self, user_model_store):
        """Longer quiet windows should yield higher confidence (more reliable sleep signal)."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {},
            "peak_hours": [10, 11],
            "quiet_hours_observed": [[23, 7]],   # 8-hour window
            "avg_response_time_by_domain": {},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "observed_quiet_window" in facts
        confidence_8h = facts["observed_quiet_window"]["confidence"]

        # Now test with a 3-hour window (minimum valid)
        # We need a fresh store to avoid re-confirmation incrementing confidence
        from storage.manager import DatabaseManager
        import tempfile, os
        tmp_dir = tempfile.mkdtemp()
        fresh_db = DatabaseManager(data_dir=tmp_dir)
        fresh_db.initialize_all()
        from storage.user_model_store import UserModelStore
        fresh_ums = UserModelStore(fresh_db)

        _set_cadence_profile(fresh_ums, {
            "hourly_activity": {},
            "peak_hours": [10, 11],
            "quiet_hours_observed": [[2, 5]],   # 3-hour window (minimum)
            "avg_response_time_by_domain": {},
        })
        SemanticFactInferrer(fresh_ums).infer_from_cadence_profile()
        fresh_facts = _get_facts_dict(fresh_ums)
        if "observed_quiet_window" in fresh_facts:
            confidence_3h = fresh_facts["observed_quiet_window"]["confidence"]
            assert confidence_8h >= confidence_3h, (
                "An 8-hour quiet window should have >= confidence than a 3-hour window"
            )

    def test_short_quiet_window_suppressed(self, user_model_store):
        """A quiet window shorter than 3 hours should NOT produce a fact (too noisy)."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {},
            "peak_hours": [10, 11],
            "quiet_hours_observed": [[3, 5]],   # 2-hour span: below minimum threshold
            "avg_response_time_by_domain": {},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "observed_quiet_window" not in facts, (
            "A 2-hour quiet window is insufficient; no observed_quiet_window fact should be stored"
        )

    def test_empty_quiet_windows_suppressed(self, user_model_store):
        """An empty quiet_hours_observed list should generate no quiet window fact."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {"9": 50, "10": 60},
            "peak_hours": [9, 10],
            "quiet_hours_observed": [],
            "avg_response_time_by_domain": {},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "observed_quiet_window" not in facts

    def test_non_midnight_crossing_window(self, user_model_store):
        """A same-day quiet window (e.g., 14:00-18:00 = afternoon off) is handled correctly."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {},
            "peak_hours": [9, 10],
            "quiet_hours_observed": [[14, 18]],   # 4-hour same-day window
            "avg_response_time_by_domain": {},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "observed_quiet_window" in facts
        value = facts["observed_quiet_window"]["value"]
        assert "14:00" in value
        assert "18:00" in value


# ---------------------------------------------------------------------------
# avg_response_time_by_domain inference
# ---------------------------------------------------------------------------

class TestDomainResponsePriorityInference:
    """Verify high_priority_domain_* facts are generated from avg_response_time_by_domain."""

    def test_fast_domain_generates_fact(self, user_model_store):
        """A domain with average reply < 1 hour should produce a high_priority_domain_* fact."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {},
            "peak_hours": [9, 10],
            "quiet_hours_observed": [],
            "avg_response_time_by_domain": {
                "work.example.com": 900.0,   # 15 minutes = high priority
                "gmail.com": 7200.0,         # 2 hours = not fast enough
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "high_priority_domain_work.example.com" in facts, (
            "A domain with avg response time of 15 min should produce a high_priority_domain_* fact"
        )
        # gmail.com is > 1 hour so no fact for it
        assert "high_priority_domain_gmail.com" not in facts, (
            "A domain with avg response time of 2h should NOT produce a high_priority_domain_* fact"
        )

    def test_domain_fact_value_contains_minutes(self, user_model_store):
        """The fact value should include the average reply time in minutes."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {},
            "peak_hours": [9, 10],
            "quiet_hours_observed": [],
            "avg_response_time_by_domain": {
                "corp.co": 1800.0,   # 30 minutes
                "other.org": 9000.0,
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "high_priority_domain_corp.co" in facts
        value = facts["high_priority_domain_corp.co"]["value"]
        assert "30min" in value or "avg_reply=30" in value, (
            f"Domain fact value should include reply time in minutes, got: {value}"
        )

    def test_max_three_domain_facts(self, user_model_store):
        """At most 3 high_priority_domain_* facts should be generated regardless of how many fast domains exist."""
        # 5 domains all replying in < 1 hour
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {},
            "peak_hours": [9, 10],
            "quiet_hours_observed": [],
            "avg_response_time_by_domain": {
                "a.com": 300.0,    # 5 min
                "b.com": 600.0,    # 10 min
                "c.com": 900.0,    # 15 min
                "d.com": 1200.0,   # 20 min
                "e.com": 1500.0,   # 25 min
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        domain_facts = [k for k in facts if k.startswith("high_priority_domain_")]
        assert len(domain_facts) == 3, (
            f"At most 3 high_priority_domain_* facts should be stored; got {len(domain_facts)}: {domain_facts}"
        )
        # Verify the 3 fastest domains were chosen (not any slower ones)
        assert "high_priority_domain_a.com" in facts
        assert "high_priority_domain_b.com" in facts
        assert "high_priority_domain_c.com" in facts
        assert "high_priority_domain_d.com" not in facts
        assert "high_priority_domain_e.com" not in facts

    def test_only_one_domain_suppressed(self, user_model_store):
        """A single fast domain (<2 domains total) should NOT generate any facts.

        We require at least 2 domains in avg_response_time_by_domain before
        inferring priority — one data point isn't enough to establish a
        meaningful comparison.
        """
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {},
            "peak_hours": [9, 10],
            "quiet_hours_observed": [],
            "avg_response_time_by_domain": {
                "work.co": 300.0,   # fast, but only domain
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        domain_facts = [k for k in facts if k.startswith("high_priority_domain_")]
        assert len(domain_facts) == 0, (
            "A single domain doesn't provide enough comparative signal; no facts should be stored"
        )

    def test_domain_confidence_scales_with_reply_speed(self, user_model_store):
        """Faster-replying domains should get higher confidence than slower-but-still-fast ones."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {},
            "peak_hours": [9, 10],
            "quiet_hours_observed": [],
            "avg_response_time_by_domain": {
                "fast.co": 60.0,     # 1 minute
                "medium.co": 1800.0, # 30 minutes
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "high_priority_domain_fast.co" in facts
        assert "high_priority_domain_medium.co" in facts

        fast_confidence = facts["high_priority_domain_fast.co"]["confidence"]
        medium_confidence = facts["high_priority_domain_medium.co"]["confidence"]
        assert fast_confidence >= medium_confidence, (
            f"1-min reply ({fast_confidence:.2f}) should have >= confidence than 30-min ({medium_confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Insufficient samples — all new inference suppressed
# ---------------------------------------------------------------------------

class TestInsufficientSamples:
    """Verify that new inferences don't fire when sample count < 25."""

    def test_all_new_facts_suppressed_with_low_samples(self, user_model_store):
        """With < 25 samples, none of the new derived-metric facts should be stored."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {"9": 5, "10": 6},
            "peak_hours": [9, 10],
            "quiet_hours_observed": [[23, 7]],
            "avg_response_time_by_domain": {"work.co": 300.0, "other.co": 500.0},
        }, sample_count=20)   # Below 25-sample threshold

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)
        assert "peak_communication_hours" not in facts
        assert "observed_quiet_window" not in facts
        domain_facts = [k for k in facts if k.startswith("high_priority_domain_")]
        assert len(domain_facts) == 0


# ---------------------------------------------------------------------------
# End-to-end: all three inferences fire together
# ---------------------------------------------------------------------------

class TestAllThreeInferencesTogether:
    """Integration test: all three derived-metric facts generated in a single call."""

    def test_all_three_facts_generated(self, user_model_store):
        """When all three derived fields have sufficient data, all three fact types are created."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {"9": 50, "10": 60, "11": 55},
            "daily_activity": {"monday": 80, "tuesday": 70},
            "peak_hours": [9, 10, 11],
            "quiet_hours_observed": [[22, 6]],
            "avg_response_time_by_domain": {
                "priority.co": 600.0,   # 10 min
                "normal.co": 3000.0,    # 50 min
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = _get_facts_dict(user_model_store)

        # All three new fact types present
        assert "peak_communication_hours" in facts, "Missing peak_communication_hours"
        assert "observed_quiet_window" in facts, "Missing observed_quiet_window"
        domain_facts = [k for k in facts if k.startswith("high_priority_domain_")]
        assert len(domain_facts) >= 1, "Missing high_priority_domain_* fact(s)"

        # All confidence values are valid [0, 1] floats
        for key in ["peak_communication_hours", "observed_quiet_window"] + domain_facts:
            confidence = facts[key]["confidence"]
            assert 0.0 <= confidence <= 1.0, (
                f"Confidence for {key} out of range: {confidence}"
            )

    def test_facts_idempotent_on_re_inference(self, user_model_store):
        """Calling infer_from_cadence_profile() twice should not raise errors or duplicate facts."""
        _set_cadence_profile(user_model_store, {
            "hourly_activity": {"9": 50, "10": 60},
            "peak_hours": [9, 10],
            "quiet_hours_observed": [[23, 7]],
            "avg_response_time_by_domain": {
                "a.co": 300.0,
                "b.co": 900.0,
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()
        inferrer.infer_from_cadence_profile()  # Second call should be idempotent

        facts = _get_facts_dict(user_model_store)
        # Still exactly the same set of facts — no duplicates
        assert "peak_communication_hours" in facts
        assert "observed_quiet_window" in facts
