"""
Life OS — Stale signal profile detection tests.

Verifies that _is_profile_stale() correctly identifies profiles with empty or
meaningless data, and that check_and_rebuild_missing_profiles() triggers a
rebuild for stale profiles (not just completely missing rows).
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from services.signal_extractor.pipeline import (
    SignalExtractorPipeline,
    _is_profile_stale,
)
from storage.event_store import EventStore


def _store_test_event(db, event_id, event_type, source, payload, metadata=None, timestamp=None):
    """Helper to insert an event into events.db."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    es = EventStore(db)
    es.store_event({
        "id": event_id,
        "type": event_type,
        "source": source,
        "timestamp": timestamp,
        "priority": "normal",
        "payload": payload,
        "metadata": metadata or {},
    })


# ---------------------------------------------------------------------------
# Unit tests for _is_profile_stale()
# ---------------------------------------------------------------------------


class TestIsProfileStale:
    """Unit tests for the _is_profile_stale helper function."""

    def test_empty_data_zero_samples(self):
        """Profile with data={} and samples_count=0 is stale."""
        profile = {"data": {}, "samples_count": 0}
        assert _is_profile_stale(profile) is True

    def test_none_data(self):
        """Profile with data=None is stale."""
        profile = {"data": None, "samples_count": 3}
        assert _is_profile_stale(profile) is True

    def test_empty_nested_dicts_low_samples(self):
        """Profile with data={'averages': {}} and low samples is stale."""
        profile = {"data": {"averages": {}}, "samples_count": 2}
        assert _is_profile_stale(profile) is True

    def test_empty_nested_dicts_high_samples(self):
        """Profile with data={'averages': {}} is stale even with enough samples — no real data."""
        profile = {"data": {"averages": {}}, "samples_count": 10}
        assert _is_profile_stale(profile) is True

    def test_metadata_only_keys(self):
        """Profile with only metadata keys (updated_at) is stale."""
        profile = {"data": {"updated_at": "2026-01-01T00:00:00Z"}, "samples_count": 10}
        assert _is_profile_stale(profile) is True

    def test_real_data_sufficient_samples(self):
        """Profile with real signal data and enough samples is NOT stale."""
        profile = {
            "data": {"averages": {"formality": 0.5, "vocabulary_richness": 0.7}},
            "samples_count": 10,
        }
        assert _is_profile_stale(profile) is False

    def test_real_data_low_samples(self):
        """Profile with real data but fewer than 5 samples is stale."""
        profile = {
            "data": {"averages": {"formality": 0.5}},
            "samples_count": 4,
        }
        assert _is_profile_stale(profile) is True

    def test_real_data_exactly_five_samples(self):
        """Profile with exactly 5 samples and real data is NOT stale."""
        profile = {
            "data": {"averages": {"formality": 0.5}},
            "samples_count": 5,
        }
        assert _is_profile_stale(profile) is False

    def test_missing_samples_count_key(self):
        """Profile without samples_count key defaults to 0 (stale)."""
        profile = {"data": {"averages": {"formality": 0.5}}}
        assert _is_profile_stale(profile) is True

    def test_mixed_empty_and_real_values(self):
        """Profile where some data keys have values and others are empty is NOT stale (if samples >= 5)."""
        profile = {
            "data": {"averages": {"formality": 0.5}, "patterns": {}},
            "samples_count": 10,
        }
        # At least one key ("averages") has a non-empty dict, so not all are empty.
        assert _is_profile_stale(profile) is False


# ---------------------------------------------------------------------------
# Integration tests for check_and_rebuild_missing_profiles with stale profiles
# ---------------------------------------------------------------------------


class TestStaleProfileRebuild:
    """Tests that check_and_rebuild_missing_profiles detects and rebuilds stale profiles."""

    def test_stale_profile_triggers_rebuild(self, db, user_model_store):
        """A profile row that exists but has empty data triggers rebuild."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # Pre-populate ALL profiles — but make 'linguistic' stale (empty data).
        expected = [
            "linguistic", "linguistic_inbound", "cadence", "mood_signals",
            "relationships", "topics", "temporal", "spatial", "decision",
        ]
        for profile_type in expected:
            user_model_store.update_signal_profile(profile_type, {"averages": {"formality": 0.5}})

        # Now overwrite 'linguistic' with empty data.  We need to set it to
        # empty data AND ensure samples_count is low.  Since update_signal_profile
        # increments samples_count, we directly write to the DB.
        import json

        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE signal_profiles SET data = ?, samples_count = 0 WHERE profile_type = ?",
                (json.dumps({}), "linguistic"),
            )

        # Store events so rebuild can proceed.
        for i in range(3):
            _store_test_event(db, f"stale-{i}", "email.received", "proton_mail", {
                "subject": f"Test {i}",
                "body": f"Message content {i} about work projects.",
                "from": "alice@company.com",
                "to": ["user@example.com"],
            }, timestamp=f"2026-03-01T{10 + i}:00:00+00:00")

        result = pipeline.check_and_rebuild_missing_profiles()

        # 'linguistic' should have been detected as needing rebuild.
        assert "linguistic" in result["missing_before"]
        # The rebuild should have been attempted.
        assert result["skipped"] is False

    def test_all_profiles_present_but_stale_triggers_rebuild(self, db, user_model_store):
        """When all profile rows exist but some are stale, rebuild is triggered (not skipped)."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        import json

        expected = [
            "linguistic", "linguistic_inbound", "cadence", "mood_signals",
            "relationships", "topics", "temporal", "spatial", "decision",
        ]
        # Create all profiles with minimal data.
        for pt in expected:
            user_model_store.update_signal_profile(pt, {"placeholder": True})

        # Make all stale by setting samples_count=1 and empty data.
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE signal_profiles SET data = ?, samples_count = 1",
                (json.dumps({}),),
            )

        # Store an event so the rebuild path is entered.
        _store_test_event(db, "e1", "email.received", "test", {
            "subject": "Hi", "body": "Hello there", "from": "a@b.com", "to": ["u@x.com"],
        })

        result = pipeline.check_and_rebuild_missing_profiles()

        # Previously this would have been skipped because rows exist.
        # Now it should detect them as stale and attempt rebuild.
        assert len(result["missing_before"]) > 0
        assert result["skipped"] is False

    def test_healthy_profiles_not_treated_as_stale(self, db, user_model_store):
        """Profiles with real data and sufficient samples are not flagged."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        import json

        expected = [
            "linguistic", "linguistic_inbound", "cadence", "mood_signals",
            "relationships", "topics", "temporal", "spatial", "decision",
        ]
        for pt in expected:
            user_model_store.update_signal_profile(pt, {"averages": {"formality": 0.5}})

        # Set all to have sufficient samples.
        with db.get_connection("user_model") as conn:
            conn.execute("UPDATE signal_profiles SET samples_count = 10")

        result = pipeline.check_and_rebuild_missing_profiles()

        # All profiles healthy — should skip rebuild.
        assert result["missing_before"] == []
        assert result["skipped"] is True

    def test_diagnostics_reports_stale_status(self, db, user_model_store):
        """The get_diagnostics method reports stale profiles distinctly."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        import json

        expected = [
            "linguistic", "linguistic_inbound", "cadence", "mood_signals",
            "relationships", "topics", "temporal", "spatial", "decision",
        ]
        for pt in expected:
            user_model_store.update_signal_profile(pt, {"averages": {"formality": 0.5}})

        # Make all samples sufficient.
        with db.get_connection("user_model") as conn:
            conn.execute("UPDATE signal_profiles SET samples_count = 10")

        # Now make 'cadence' stale.
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE signal_profiles SET data = ?, samples_count = 2 WHERE profile_type = ?",
                (json.dumps({}), "cadence"),
            )

        # Store an event so event type queries don't fail.
        _store_test_event(db, "e1", "email.received", "test", {
            "subject": "Hi", "body": "Hello", "from": "a@b.com", "to": ["u@x.com"],
        })

        diag = pipeline.get_diagnostics()

        assert diag["profiles"]["cadence"]["status"] == "stale"
        assert diag["profiles"]["linguistic"]["status"] == "ok"
        # Stale profiles count towards missing.
        assert diag["profiles_missing"] >= 1
