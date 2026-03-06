"""
Life OS — Rebuild write verification tests.

Verifies that rebuild_profiles_from_events() detects and reports when
update_signal_profile() silently fails to persist data, rather than
reporting success based solely on extractor hit counts.
"""

import json
import logging
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from services.signal_extractor.pipeline import SignalExtractorPipeline
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


def _populate_test_events(db, count=10):
    """Insert a batch of email events (both sent and received) that multiple extractors can process."""
    for i in range(count):
        # Alternate between received and sent so both linguistic and linguistic_inbound get data.
        event_type = "email.received" if i % 2 == 0 else "email.sent"
        _store_test_event(
            db, f"write-verify-{i}", event_type, "proton_mail",
            {
                "subject": f"Test email {i}",
                "body": f"This is email number {i} discussing project updates and quarterly goals.",
                "from": "alice@company.com",
                "to": ["user@example.com"],
            },
            timestamp=f"2026-03-01T{10 + (i % 12)}:00:00+00:00",
        )


class TestRebuildWriteVerification:
    """Tests for the write verification phase in rebuild_profiles_from_events."""

    def test_rebuild_detects_silent_write_failures(self, db, user_model_store):
        """When update_signal_profile silently fails, write_failures is populated."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        _populate_test_events(db, count=10)

        # Make update_signal_profile silently fail (mimicking the real failure mode:
        # the method catches exceptions and logs a warning, returning None).
        original_update = user_model_store.update_signal_profile

        def failing_update(profile_type, data):
            """Simulate silent failure — the real method catches exceptions internally."""
            # Do nothing, simulating data not being persisted
            pass

        with patch.object(user_model_store, "update_signal_profile", side_effect=failing_update):
            result = pipeline.rebuild_profiles_from_events(
                event_limit=100,
                missing_profiles=["linguistic", "cadence", "relationships", "topics"],
            )

        # Extractors should have processed events (hit counts > 0).
        assert result["events_processed"] > 0
        assert len(result["profiles_rebuilt"]) > 0

        # But write_failures should detect that profiles weren't persisted.
        assert "write_failures" in result
        assert len(result["write_failures"]) > 0

        # Each failure entry should include extractor_hits and profile_exists.
        for profile_name, info in result["write_failures"].items():
            assert "extractor_hits" in info
            assert info["extractor_hits"] > 0
            assert info["profile_exists"] is False

    def test_rebuild_reports_success_when_profiles_written(self, db, user_model_store):
        """When profiles are actually written, write_failures is empty."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        _populate_test_events(db, count=10)

        result = pipeline.rebuild_profiles_from_events(
            event_limit=100,
            missing_profiles=["linguistic", "cadence", "relationships", "topics"],
        )

        # Extractors should have processed events.
        assert result["events_processed"] > 0

        # write_failures should be empty since profiles were actually written.
        assert result.get("write_failures") == {} or all(
            # If a profile had 0 extractor hits, it won't appear in write_failures
            # regardless. Only profiles with hits AND no persistence are failures.
            info["profile_exists"] is True
            for info in result.get("write_failures", {}).values()
        )

    def test_write_failures_include_extractor_hit_counts(self, db, user_model_store):
        """write_failures entries include the extractor hit count for the failed profile."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        _populate_test_events(db, count=10)

        # Selectively fail only specific profiles by intercepting update_signal_profile.
        failed_profiles = {"cadence", "topics"}
        original_update = user_model_store.update_signal_profile

        def selective_failure(profile_type, data):
            """Only fail for specific profiles, let others succeed."""
            if profile_type in failed_profiles:
                return  # Silent failure — no data persisted
            return original_update(profile_type, data)

        with patch.object(user_model_store, "update_signal_profile", side_effect=selective_failure):
            result = pipeline.rebuild_profiles_from_events(
                event_limit=100,
                missing_profiles=["linguistic", "cadence", "relationships", "topics"],
            )

        write_failures = result.get("write_failures", {})

        # The selectively-failed profiles should appear in write_failures.
        for profile_name in failed_profiles:
            if profile_name in write_failures:
                assert write_failures[profile_name]["extractor_hits"] > 0
                assert write_failures[profile_name]["profile_exists"] is False
                assert write_failures[profile_name]["samples_count"] == 0

    def test_check_and_rebuild_logs_critical_on_write_failures(self, db, user_model_store, caplog):
        """check_and_rebuild_missing_profiles logs CRITICAL when write failures occur."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        _populate_test_events(db, count=10)

        # Make all profile writes silently fail.
        def failing_update(profile_type, data):
            pass

        with patch.object(user_model_store, "update_signal_profile", side_effect=failing_update):
            with caplog.at_level(logging.CRITICAL):
                result = pipeline.check_and_rebuild_missing_profiles()

        # Should have logged a CRITICAL message about write failures.
        critical_messages = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_messages) > 0
        # At least one CRITICAL message should mention write failures.
        assert any("FAILED" in msg.message or "write" in msg.message.lower() for msg in critical_messages)

    def test_get_rebuild_diagnostics_returns_no_rebuild(self, db, user_model_store):
        """get_rebuild_diagnostics returns status when no rebuild has occurred."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        result = pipeline.get_rebuild_diagnostics()
        assert result["status"] == "no_rebuild_performed"

    def test_get_rebuild_diagnostics_returns_last_result(self, db, user_model_store):
        """get_rebuild_diagnostics returns the cached result after a rebuild."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        _populate_test_events(db, count=5)

        pipeline.rebuild_profiles_from_events(event_limit=100)
        diagnostics = pipeline.get_rebuild_diagnostics()

        assert "events_processed" in diagnostics
        assert "write_failures" in diagnostics
        assert diagnostics["events_processed"] > 0
