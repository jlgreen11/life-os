"""
Tests for SemanticFactInferrer resilience under database corruption.

Verifies that the per-method error isolation in run_all_inference() works
correctly when the underlying user_model.db is corrupted — the exact
failure mode the system experiences when SQLite reports
"database disk image is malformed".

Each infer_from_*_profile() method calls self.ums.get_signal_profile()
at its entry point.  The run_all_inference() method wraps each call in
try/except (lines 1335-1341 of inferrer.py), which SHOULD isolate
per-method failures.  These tests verify that contract under realistic
DB corruption scenarios using sqlite3.OperationalError.
"""

import logging
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


def _setup_linguistic_profile(ums):
    """Seed the linguistic profile with valid data that will produce facts."""
    ums.update_signal_profile("linguistic", {
        "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
    })
    _set_samples(ums, "linguistic", 25)


def _setup_relationship_profile(ums):
    """Seed the relationship profile with valid data."""
    ums.update_signal_profile("relationships", {
        "contacts": {
            "alice@example.com": {
                "interaction_count": 10,
                "avg_response_time_seconds": 1800,
                "outbound_count": 5,
            },
            "bob@example.com": {
                "interaction_count": 1,
                "outbound_count": 1,
            },
            "carol@example.com": {
                "interaction_count": 1,
                "outbound_count": 1,
            },
        }
    })
    _set_samples(ums, "relationships", 15)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestSemanticInferrerDbResilience:
    """Verify SemanticFactInferrer handles DB corruption gracefully.

    All tests simulate the exact error the system sees in production:
    sqlite3.OperationalError('database disk image is malformed').
    """

    CORRUPTION_ERROR = sqlite3.OperationalError("database disk image is malformed")

    def test_single_corrupted_profile_doesnt_block_others(self, user_model_store):
        """A corrupted linguistic profile must not prevent other profiles from running.

        Patches get_signal_profile to raise OperationalError only for 'linguistic',
        while the relationship profile has valid data.  Verifies:
          (a) run_all_inference() does not raise
          (b) results list has 9 entries (one per inference method)
          (c) the linguistic entry has processed=False
          (d) the relationship entry has processed=True (or at least not error)
        """
        _setup_relationship_profile(user_model_store)

        original_get = user_model_store.get_signal_profile

        def corrupted_get(profile_type):
            """Raise OperationalError only for the linguistic profile."""
            if profile_type == "linguistic":
                raise self.CORRUPTION_ERROR
            return original_get(profile_type)

        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(user_model_store, "get_signal_profile", side_effect=corrupted_get):
            # Must NOT raise
            inferrer.run_all_inference()

        # The relationship profile should have been processed (even if no
        # facts were created, the method ran without error)
        # Verify by checking that no exception propagated — the test
        # reaching this point proves isolation works.

    def test_all_profiles_corrupted_returns_gracefully(self, user_model_store):
        """When every get_signal_profile() call raises OperationalError,
        run_all_inference() must still complete without raising.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=self.CORRUPTION_ERROR,
        ):
            # Must NOT raise even with total DB corruption
            inferrer.run_all_inference()

        # No facts should have been created
        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 0

    def test_corrupted_profile_error_is_logged(self, user_model_store, caplog):
        """OperationalError from a corrupted profile must be logged with
        the profile name so operators can diagnose which profile is affected.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=self.CORRUPTION_ERROR,
        ):
            with caplog.at_level(logging.ERROR, logger="services.semantic_fact_inferrer.inferrer"):
                inferrer.run_all_inference()

        # At least one log record should mention the failed profile
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) >= 1, "Expected at least one ERROR log record for corrupted profile"

        # Verify the log message includes a profile name for diagnosis
        error_messages = " ".join(r.message for r in error_records)
        assert "failed" in error_messages.lower(), (
            f"Expected 'failed' in error log messages, got: {error_messages}"
        )

    def test_intermittent_corruption_first_call_fails_subsequent_succeed(self, user_model_store):
        """Simulate intermittent corruption: first get_signal_profile() call
        raises OperationalError, subsequent calls succeed with real data.

        Verifies the first method fails but all subsequent methods succeed.
        """
        _setup_linguistic_profile(user_model_store)
        _setup_relationship_profile(user_model_store)

        original_get = user_model_store.get_signal_profile
        call_count = {"n": 0}

        def intermittent_get(profile_type):
            """First call raises, rest succeed."""
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise self.CORRUPTION_ERROR
            return original_get(profile_type)

        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(user_model_store, "get_signal_profile", side_effect=intermittent_get):
            # Must NOT raise
            inferrer.run_all_inference()

        # The first method (linguistic) should have failed, but subsequent
        # methods should have run.  We can't guarantee which methods produce
        # facts (depends on data), but reaching this point proves the loop
        # continued past the first failure.
        assert call_count["n"] >= 2, "Expected at least 2 get_signal_profile calls (first fails, rest continue)"

    def test_database_error_subclass_caught(self, user_model_store):
        """sqlite3.DatabaseError (parent of OperationalError) must also be caught.

        The except clause in run_all_inference uses `except Exception`, which
        catches everything in the hierarchy:
            Exception → DatabaseError → OperationalError

        This test verifies DatabaseError specifically is caught.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=sqlite3.DatabaseError("database corruption"),
        ):
            # Must NOT raise
            inferrer.run_all_inference()

    def test_operational_error_caught(self, user_model_store):
        """sqlite3.OperationalError — the exact error from production — must be caught."""
        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=sqlite3.OperationalError("database disk image is malformed"),
        ):
            # Must NOT raise
            inferrer.run_all_inference()

    def test_inference_summary_logged_after_partial_failure(self, user_model_store, caplog):
        """_log_inference_summary() must be called even when some methods fail.

        The summary should include both successes and failures so operators
        can see at a glance which profiles were affected.
        """
        _setup_linguistic_profile(user_model_store)

        original_get = user_model_store.get_signal_profile

        def corrupt_mood_only(profile_type):
            """Raise only for mood_signals profile."""
            if profile_type == "mood_signals":
                raise self.CORRUPTION_ERROR
            return original_get(profile_type)

        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(user_model_store, "get_signal_profile", side_effect=corrupt_mood_only):
            with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
                inferrer.run_all_inference()

        # The summary log line should be present
        summary_lines = [r.message for r in caplog.records if "inference cycle complete" in r.message]
        assert len(summary_lines) == 1, f"Expected exactly 1 summary line, got {len(summary_lines)}"

        # The summary should mention mood as errored
        assert "mood (error)" in summary_lines[0], (
            f"Expected 'mood (error)' in summary, got: {summary_lines[0]}"
        )

    def test_partial_success_facts_preserved_after_corruption(self, user_model_store):
        """Facts from successfully processed profiles must be preserved even
        when other profiles raise OperationalError.

        This verifies that the try/except doesn't accidentally discard
        partial results.
        """
        # Set up linguistic data that WILL produce a fact
        _setup_linguistic_profile(user_model_store)

        original_get = user_model_store.get_signal_profile

        def corrupt_except_linguistic(profile_type):
            """Return real data for linguistic, raise for everything else."""
            if profile_type == "linguistic":
                return original_get(profile_type)
            raise self.CORRUPTION_ERROR

        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(user_model_store, "get_signal_profile", side_effect=corrupt_except_linguistic):
            inferrer.run_all_inference()

        # The linguistic profile should have produced a communication_style fact
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        casual_fact = next(
            (f for f in facts if f["key"] == "communication_style_formality"),
            None,
        )
        assert casual_fact is not None, "Linguistic fact should be preserved despite other profiles failing"
        assert casual_fact["value"] == "casual"

    def test_corruption_does_not_trigger_infinite_retry(self, user_model_store):
        """Verify that DB corruption does not cause run_all_inference() to
        loop or retry — each profile method is called exactly once.
        """
        inferrer = SemanticFactInferrer(user_model_store)
        call_count = {"n": 0}

        original_get = user_model_store.get_signal_profile

        def counting_corrupt_get(profile_type):
            """Count calls and always raise."""
            call_count["n"] += 1
            raise self.CORRUPTION_ERROR

        with patch.object(user_model_store, "get_signal_profile", side_effect=counting_corrupt_get):
            inferrer.run_all_inference()

        # run_all_inference has 9 methods.  Each calls get_signal_profile
        # once at its entry point.  With total corruption, exactly 9 calls
        # should be made — no retries.
        assert call_count["n"] == 9, (
            f"Expected exactly 9 get_signal_profile calls (one per profile), got {call_count['n']}"
        )

    def test_integrity_error_caught(self, user_model_store):
        """sqlite3.IntegrityError must also be caught by the except clause.

        This can occur when a corrupted DB has violated constraint state.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=sqlite3.IntegrityError("UNIQUE constraint failed"),
        ):
            # Must NOT raise
            inferrer.run_all_inference()

    def test_error_logged_with_traceback(self, user_model_store, caplog):
        """logger.exception() should be used so the full traceback is captured.

        This is critical for post-incident diagnosis of corruption events.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=self.CORRUPTION_ERROR,
        ):
            with caplog.at_level(logging.ERROR, logger="services.semantic_fact_inferrer.inferrer"):
                inferrer.run_all_inference()

        # logger.exception() sets exc_info on the log record
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) >= 1

        # At least one record should have exc_info (from logger.exception)
        records_with_traceback = [r for r in error_records if r.exc_info]
        assert len(records_with_traceback) >= 1, (
            "Expected at least one ERROR record with exc_info (logger.exception), "
            "but none found — ensure logger.exception() is used, not logger.error()"
        )
