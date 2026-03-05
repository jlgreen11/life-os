"""
Tests for the profile health diagnostics added to SignalExtractorPipeline.

Verifies:
- get_profile_health() returns correct status for missing, stale, and ok profiles
- rebuild_profiles_from_events() returns extractor_error_counts in its result
- Error detail list is capped at 20 entries to avoid flooding logs
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from services.signal_extractor.pipeline import (
    SignalExtractorPipeline,
    _is_profile_stale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_profile(db, profile_type: str, data: dict, samples_count: int):
    """Insert a signal profile with a specific sample count directly via SQL."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (profile_type, json.dumps(data), samples_count),
        )


def _insert_event(db, event_type: str, payload: dict | None = None, ts: str | None = None):
    """Insert a synthetic event into events.db and return its id."""
    event_id = str(uuid.uuid4())
    timestamp = ts or datetime.now(timezone.utc).isoformat()
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                event_type,
                "test",
                timestamp,
                "normal",
                json.dumps(payload or {}),
                json.dumps({}),
            ),
        )
    return event_id


# ---------------------------------------------------------------------------
# Tests for get_profile_health()
# ---------------------------------------------------------------------------

class TestGetProfileHealth:
    """Verify get_profile_health() correctly classifies each profile."""

    def test_all_missing_when_no_profiles_exist(self, db, user_model_store):
        """With a fresh database, all 9 profiles should report 'missing'."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        health = pipeline.get_profile_health()

        assert len(health) == 9
        for ptype, info in health.items():
            assert info["status"] == "missing", f"{ptype} should be missing"
            assert info["samples"] == 0
            assert info["data_keys"] == []

    def test_ok_profile_with_sufficient_data(self, db, user_model_store):
        """A profile with data and >= 5 samples should report 'ok'."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # Manually insert a well-populated profile via raw SQL.
        _insert_profile(db, "relationships",
                        {"contacts": {"alice": 10}, "response_times": {"alice": 300}},
                        samples_count=100)

        health = pipeline.get_profile_health()

        assert health["relationships"]["status"] == "ok"
        assert health["relationships"]["samples"] == 100
        assert len(health["relationships"]["data_keys"]) > 0
        # Other profiles should still be missing.
        assert health["linguistic"]["status"] == "missing"

    def test_stale_profile_with_few_samples(self, db, user_model_store):
        """A profile with < 5 samples should report 'stale'."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # Insert a profile with too few samples to be useful.
        _insert_profile(db, "cadence", {"hourly_counts": {"10": 2}}, samples_count=3)

        health = pipeline.get_profile_health()

        assert health["cadence"]["status"] == "stale"
        assert health["cadence"]["samples"] == 3

    def test_stale_profile_with_empty_data(self, db, user_model_store):
        """A profile with empty data dict should report 'stale'."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        _insert_profile(db, "topics", {}, samples_count=50)

        health = pipeline.get_profile_health()

        assert health["topics"]["status"] == "stale"

    def test_caches_result_on_instance(self, db, user_model_store):
        """get_profile_health() should cache results on _last_profile_health."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        assert pipeline._last_profile_health is None

        health = pipeline.get_profile_health()
        assert pipeline._last_profile_health is health

    def test_data_keys_limited_to_five(self, db, user_model_store):
        """data_keys should contain at most 5 keys even if profile has more."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        _insert_profile(db, "linguistic", {f"key_{i}": i for i in range(10)}, samples_count=20)

        health = pipeline.get_profile_health()

        assert health["linguistic"]["status"] == "ok"
        assert len(health["linguistic"]["data_keys"]) == 5


# ---------------------------------------------------------------------------
# Tests for extractor_error_counts in rebuild
# ---------------------------------------------------------------------------

class TestRebuildErrorCounts:
    """Verify rebuild_profiles_from_events() tracks errors per extractor."""

    def test_extractor_error_counts_in_result(self, db, user_model_store):
        """Result dict should include extractor_error_counts key."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # Insert a valid event so the rebuild actually runs.
        _insert_event(db, "email.received", {"subject": "test", "from": "a@b.com"})

        result = pipeline.rebuild_profiles_from_events(event_limit=10)

        assert "extractor_error_counts" in result
        assert isinstance(result["extractor_error_counts"], dict)

    def test_error_counts_accumulate_per_extractor(self, db, user_model_store):
        """When an extractor raises on every event, its error count should match event count."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # Insert multiple events.
        for i in range(5):
            _insert_event(db, "email.sent", {"subject": f"test {i}", "body": f"body {i}"})

        # Make the first extractor always fail.
        original_extract = pipeline.extractors[0].extract

        def failing_extract(event):
            raise ValueError("forced test error")

        pipeline.extractors[0].extract = failing_extract

        result = pipeline.rebuild_profiles_from_events(event_limit=100)

        extractor_name = type(pipeline.extractors[0]).__name__
        assert extractor_name in result["extractor_error_counts"]
        assert result["extractor_error_counts"][extractor_name] == 5

    def test_error_detail_capped_at_20(self, db, user_model_store):
        """The detailed errors list should never exceed 20 entries."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # Insert 30 events that will trigger the failing extractor.
        for i in range(30):
            _insert_event(db, "email.sent", {"subject": f"test {i}", "body": f"body {i}"})

        # Make ALL extractors fail to generate lots of errors.
        for ext in pipeline.extractors:
            ext.extract = lambda event: (_ for _ in ()).throw(ValueError("forced"))

        result = pipeline.rebuild_profiles_from_events(event_limit=100)

        # Detailed errors capped at 20, but total error count should be higher.
        assert len(result["errors"]) <= 20
        total_errors = sum(result["extractor_error_counts"].values())
        assert total_errors > 20

    def test_rebuild_result_cached_on_instance(self, db, user_model_store):
        """rebuild_profiles_from_events() should cache its result on _last_rebuild_result."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        assert pipeline._last_rebuild_result is None

        _insert_event(db, "email.received", {"subject": "test"})
        result = pipeline.rebuild_profiles_from_events(event_limit=10)

        assert pipeline._last_rebuild_result is result


# ---------------------------------------------------------------------------
# Tests for _is_profile_stale helper
# ---------------------------------------------------------------------------

class TestIsProfileStale:
    """Edge cases for _is_profile_stale used by get_profile_health."""

    def test_none_data_is_stale(self):
        """Profile with None data is stale."""
        assert _is_profile_stale({"data": None, "samples_count": 100}) is True

    def test_metadata_only_keys_is_stale(self):
        """Profile whose data has only metadata keys is stale."""
        assert _is_profile_stale({
            "data": {"updated_at": "2026-01-01", "created_at": "2025-01-01"},
            "samples_count": 100,
        }) is True

    def test_below_min_samples_is_stale(self):
        """Profile with real data but < 5 samples is stale."""
        assert _is_profile_stale({
            "data": {"real_key": {"value": 1}},
            "samples_count": 4,
        }) is True

    def test_healthy_profile_is_not_stale(self):
        """Profile with real data and >= 5 samples is not stale."""
        assert _is_profile_stale({
            "data": {"real_key": {"value": 1}},
            "samples_count": 10,
        }) is False
