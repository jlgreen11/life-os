"""
Tests for SignalExtractorPipeline.periodic_health_check() and
the linguistic false-positive annotation in get_profile_health().

Covers:
- All profiles present → status='healthy'
- Missing profiles → triggers rebuild on first detection
- Same profiles missing on consecutive checks → no redundant rebuild
- force_rebuild=True → always rebuilds regardless of prior state
- linguistic missing with linguistic_inbound present → annotated as expected
- _last_health_check_missing state management across checks
- Persistent missing with high event counts → retry after backoff
- Persistent missing with low event counts → no retry (stays degraded)
- Retry counter increments correctly across consecutive degraded checks

Follows patterns from tests/test_signal_extractor_pipeline.py:
- Real DatabaseManager/UserModelStore via db/user_model_store fixtures
- No mocking of the storage layer
- Mock only the rebuild helper and event count query to keep tests focused
"""

import pytest

from services.signal_extractor.pipeline import (
    SignalExtractorPipeline,
    _RETRY_EVERY_N_CHECKS,
    _RETRY_EVENT_COUNT_THRESHOLD,
)


class TestPeriodicHealthCheck:
    """Test suite for SignalExtractorPipeline.periodic_health_check()."""

    @pytest.fixture
    def pipeline(self, db, user_model_store):
        """Create a SignalExtractorPipeline backed by test databases."""
        return SignalExtractorPipeline(db, user_model_store)

    @pytest.fixture
    def populated_pipeline(self, db, user_model_store):
        """Pipeline with all 9 expected signal profiles populated above the stale threshold.

        Calls update_signal_profile 6 times per profile so samples_count == 6,
        which exceeds the _MIN_SAMPLES threshold of 5 used by _is_profile_stale().
        The data dict includes a non-metadata key with a scalar value so the
        stale check's meaningful-keys and empty-container guards both pass.
        """
        expected_profiles = [
            "linguistic", "linguistic_inbound", "cadence", "mood_signals",
            "relationships", "topics", "temporal", "spatial", "decision",
        ]
        for ptype in expected_profiles:
            for _ in range(6):
                user_model_store.update_signal_profile(ptype, {"signal_key": "populated"})

        return SignalExtractorPipeline(db, user_model_store)

    # --- Healthy status ---

    def test_healthy_when_all_profiles_present(self, populated_pipeline):
        """When all profiles have sufficient data, periodic_health_check returns status='healthy'."""
        result = populated_pipeline.periodic_health_check()

        assert result["status"] == "healthy"
        assert result["missing"] == []

    def test_healthy_clears_last_missing_state(self, populated_pipeline):
        """A healthy check resets _last_health_check_missing to an empty list."""
        populated_pipeline._last_health_check_missing = ["temporal", "decision"]

        result = populated_pipeline.periodic_health_check()

        assert result["status"] == "healthy"
        assert populated_pipeline._last_health_check_missing == []

    def test_healthy_does_not_trigger_rebuild(self, populated_pipeline):
        """When all profiles are healthy, check_and_rebuild_missing_profiles is never called."""
        rebuild_called = []
        populated_pipeline.check_and_rebuild_missing_profiles = lambda: rebuild_called.append(1) or {}

        populated_pipeline.periodic_health_check()

        assert len(rebuild_called) == 0

    # --- Rebuild on first detection ---

    def test_first_missing_detection_triggers_rebuild(self, pipeline):
        """On a fresh pipeline (no prior check), any missing profiles trigger a rebuild."""
        rebuild_called = []

        def mock_rebuild():
            rebuild_called.append(True)
            return {"missing_before": [], "rebuilt": [], "skipped": False}

        pipeline.check_and_rebuild_missing_profiles = mock_rebuild

        # A fresh DB has no signal profiles — all will be missing.
        result = pipeline.periodic_health_check()

        assert result["status"] == "rebuilt"
        assert len(rebuild_called) == 1
        # The missing list is non-empty (fresh DB has no profiles).
        assert len(result["missing"]) > 0

    def test_rebuild_result_included_in_response(self, pipeline):
        """When a rebuild is triggered, its result dict is attached to the response."""
        expected_result = {"missing_before": ["temporal"], "rebuilt": ["temporal"], "skipped": False}
        pipeline.check_and_rebuild_missing_profiles = lambda: expected_result

        result = pipeline.periodic_health_check()

        assert result["status"] == "rebuilt"
        assert result["rebuild_result"] == expected_result

    def test_missing_list_populated_in_rebuilt_response(self, pipeline):
        """The 'missing' key in a 'rebuilt' response lists the affected profiles."""
        pipeline.check_and_rebuild_missing_profiles = lambda: {"missing_before": [], "rebuilt": [], "skipped": False}

        result = pipeline.periodic_health_check()

        assert "missing" in result
        assert isinstance(result["missing"], list)
        assert len(result["missing"]) > 0

    # --- No redundant rebuild ---

    def test_no_redundant_rebuild_for_same_missing_profiles(self, pipeline):
        """When the same profiles remain missing on consecutive checks, rebuild fires only once."""
        rebuild_call_count = []

        def mock_rebuild():
            rebuild_call_count.append(1)
            return {"missing_before": [], "rebuilt": [], "skipped": False}

        pipeline.check_and_rebuild_missing_profiles = mock_rebuild

        # First check: all profiles newly missing → triggers rebuild.
        result1 = pipeline.periodic_health_check()
        assert result1["status"] == "rebuilt"
        assert len(rebuild_call_count) == 1

        # Second check: same profiles still missing → no rebuild.
        result2 = pipeline.periodic_health_check()
        assert result2["status"] == "degraded"
        assert len(rebuild_call_count) == 1  # No additional rebuild.

    def test_degraded_status_includes_explanatory_note(self, pipeline):
        """status='degraded' includes a human-readable note explaining why no rebuild ran."""
        pipeline.check_and_rebuild_missing_profiles = lambda: {"missing_before": [], "rebuilt": [], "skipped": False}

        pipeline.periodic_health_check()  # First: triggers rebuild.
        result = pipeline.periodic_health_check()  # Second: degraded.

        assert result["status"] == "degraded"
        assert "note" in result
        assert "same profiles missing" in result["note"]

    def test_degraded_status_lists_missing_profiles(self, pipeline):
        """status='degraded' response includes the full list of still-missing profiles."""
        pipeline.check_and_rebuild_missing_profiles = lambda: {"missing_before": [], "rebuilt": [], "skipped": False}

        pipeline.periodic_health_check()  # First check.
        result = pipeline.periodic_health_check()  # Second check.

        assert "missing" in result
        assert len(result["missing"]) > 0

    # --- force_rebuild ---

    def test_force_rebuild_triggers_rebuild_when_same_missing(self, pipeline):
        """force_rebuild=True triggers rebuild even when the missing set hasn't changed."""
        rebuild_call_count = []

        def mock_rebuild():
            rebuild_call_count.append(1)
            return {"missing_before": [], "rebuilt": [], "skipped": False}

        pipeline.check_and_rebuild_missing_profiles = mock_rebuild

        # First check with force_rebuild: triggers rebuild (also newly missing).
        result1 = pipeline.periodic_health_check(force_rebuild=True)
        assert result1["status"] == "rebuilt"
        assert len(rebuild_call_count) == 1

        # Second check with force_rebuild: same profiles missing, but force → rebuild again.
        result2 = pipeline.periodic_health_check(force_rebuild=True)
        assert result2["status"] == "rebuilt"
        assert len(rebuild_call_count) == 2

    def test_force_rebuild_with_no_missing_profiles_still_healthy(self, populated_pipeline):
        """force_rebuild=True when all profiles are healthy returns status='healthy' (no rebuild)."""
        rebuild_called = []
        populated_pipeline.check_and_rebuild_missing_profiles = lambda: rebuild_called.append(1) or {}

        result = populated_pipeline.periodic_health_check(force_rebuild=True)

        # No missing profiles → healthy; force_rebuild has nothing to act on.
        assert result["status"] == "healthy"
        assert len(rebuild_called) == 0

    # --- State management ---

    def test_last_missing_state_updated_after_rebuild(self, pipeline):
        """After a rebuild, _last_health_check_missing reflects the current missing list."""
        pipeline.check_and_rebuild_missing_profiles = lambda: {"missing_before": [], "rebuilt": [], "skipped": False}

        result = pipeline.periodic_health_check()

        assert result["status"] == "rebuilt"
        # State must match what was returned.
        assert set(pipeline._last_health_check_missing) == set(result["missing"])

    def test_last_missing_state_preserved_after_degraded(self, pipeline):
        """_last_health_check_missing remains the same list across consecutive degraded checks."""
        pipeline.check_and_rebuild_missing_profiles = lambda: {"missing_before": [], "rebuilt": [], "skipped": False}

        pipeline.periodic_health_check()  # First: rebuilt.
        missing_after_first = list(pipeline._last_health_check_missing)

        pipeline.periodic_health_check()  # Second: degraded.
        missing_after_second = list(pipeline._last_health_check_missing)

        assert missing_after_first == missing_after_second

    def test_initial_last_health_check_missing_is_empty(self, pipeline):
        """A freshly constructed pipeline has _last_health_check_missing == []."""
        assert pipeline._last_health_check_missing == []

    def test_newly_missing_subset_triggers_rebuild(self, pipeline):
        """Rebuild fires when ONLY SOME profiles are newly missing (not all were missing before)."""
        rebuild_call_count = []

        def mock_rebuild():
            rebuild_call_count.append(1)
            return {"missing_before": [], "rebuilt": [], "skipped": False}

        pipeline.check_and_rebuild_missing_profiles = mock_rebuild

        # Simulate a state where some profiles were already known missing.
        # Put a subset of the actual missing profiles into the prior state
        # so that the remaining ones appear as "newly missing".
        # We can only do this by pre-setting _last_health_check_missing
        # to a partial list — the get_profile_health() call will return
        # ALL profiles as missing (fresh DB), so any profile NOT in the
        # prior list counts as newly missing.
        pipeline._last_health_check_missing = ["linguistic"]  # Only linguistic known missing.

        result = pipeline.periodic_health_check()

        # Other profiles are newly missing → rebuild must fire.
        assert result["status"] == "rebuilt"
        assert len(rebuild_call_count) == 1


class TestGetProfileHealthLinguisticAnnotation:
    """Test the linguistic false-positive annotation added by get_profile_health()."""

    @pytest.fixture
    def pipeline(self, db, user_model_store):
        """Create a SignalExtractorPipeline backed by test databases."""
        return SignalExtractorPipeline(db, user_model_store)

    def test_linguistic_annotated_when_missing_but_inbound_healthy(self, pipeline, user_model_store):
        """When 'linguistic' (outbound) is absent but 'linguistic_inbound' has healthy data,
        a 'note' key is added to the linguistic health entry explaining the expected behavior."""
        # Populate linguistic_inbound with 6 samples to exceed the stale threshold.
        for _ in range(6):
            user_model_store.update_signal_profile("linguistic_inbound", {
                "avg_sentence_length": 12.5,
                "vocabulary_richness": 0.65,
            })
        # Leave 'linguistic' (outbound) absent.

        health = pipeline.get_profile_health()

        assert health["linguistic_inbound"]["status"] == "ok"
        assert health["linguistic"]["status"] == "missing"
        # A note must be present indicating this is expected, not a failure.
        assert "note" in health["linguistic"]
        note = health["linguistic"]["note"].lower()
        assert "inbound" in note or "outbound" in note

    def test_linguistic_not_annotated_when_inbound_also_missing(self, pipeline):
        """When both 'linguistic' and 'linguistic_inbound' are missing, no note is added.

        An annotation would be misleading here — the missing outbound profile is NOT
        explained by having inbound data when there is no inbound data either.
        """
        # Leave both profiles absent (fresh DB).
        health = pipeline.get_profile_health()

        assert health["linguistic"]["status"] == "missing"
        assert "note" not in health["linguistic"]

    def test_linguistic_ok_has_no_annotation(self, pipeline, user_model_store):
        """When 'linguistic' itself has healthy data, no note is added."""
        for _ in range(6):
            user_model_store.update_signal_profile("linguistic", {
                "avg_sentence_length": 14.0,
                "vocabulary_richness": 0.75,
            })

        health = pipeline.get_profile_health()

        assert health["linguistic"]["status"] == "ok"
        assert "note" not in health["linguistic"]

    def test_linguistic_stale_not_annotated_as_expected(self, pipeline, user_model_store):
        """When 'linguistic' is stale (few samples) and inbound is healthy, no note is added.

        The annotation applies only to the 'missing' state (profile row absent from DB),
        not to stale rows that are present but under-sampled.
        """
        # Write only 1 sample — stale threshold requires >= 5.
        user_model_store.update_signal_profile("linguistic", {"avg_sentence_length": 12.0})
        for _ in range(6):
            user_model_store.update_signal_profile("linguistic_inbound", {
                "avg_sentence_length": 12.5,
                "vocabulary_richness": 0.65,
            })

        health = pipeline.get_profile_health()

        # linguistic should be stale, not missing.
        assert health["linguistic"]["status"] == "stale"
        # No note on stale — annotation is only for status='missing'.
        assert "note" not in health["linguistic"]

    def test_annotation_note_references_inbound_health(self, pipeline, user_model_store):
        """The note text explicitly mentions inbound data being healthy."""
        for _ in range(6):
            user_model_store.update_signal_profile("linguistic_inbound", {
                "vocabulary_richness": 0.7,
            })

        health = pipeline.get_profile_health()
        note = health["linguistic"].get("note", "")

        # Note must reference the inbound profile being ok to help operators understand.
        assert "inbound" in note.lower()


class TestPeriodicHealthCheckPersistentRetry:
    """Tests for the persistent missing profile retry-with-backoff logic.

    When the same profiles remain missing across multiple consecutive health
    checks (because the first rebuild did not fix them), periodic_health_check()
    should eventually retry the rebuild — but only for profiles that have enough
    qualifying events to make a retry worthwhile.

    The test strategy:
    - Use a fresh pipeline (empty DB) so all profiles are missing from the start.
    - Mock check_and_rebuild_missing_profiles to avoid expensive event replay.
    - Mock _count_qualifying_events to control event-count thresholds.
    - Pre-seed _rebuild_retry_count to simulate N previous degraded checks so
      the "second check" in the test can reach the backoff threshold without
      running N-1 actual health checks.
    """

    @pytest.fixture
    def pipeline(self, db, user_model_store):
        """Create a SignalExtractorPipeline backed by test databases."""
        return SignalExtractorPipeline(db, user_model_store)

    def _install_mock_rebuild(self, pipeline, call_log):
        """Replace check_and_rebuild_missing_profiles with a no-op that records calls."""
        def mock_rebuild():
            call_log.append(True)
            return {"missing_before": [], "rebuilt": [], "skipped": False}
        pipeline.check_and_rebuild_missing_profiles = mock_rebuild

    # --- Initial first-check behavior (regression guard) ---

    def test_first_check_triggers_rebuild(self, pipeline):
        """On the very first health check, any missing profiles trigger a rebuild.

        This is a regression guard: the persistent retry logic must not
        interfere with the original first-detection rebuild behaviour.
        """
        rebuild_called = []
        self._install_mock_rebuild(pipeline, rebuild_called)

        result = pipeline.periodic_health_check()

        assert result["status"] == "rebuilt"
        assert len(rebuild_called) == 1

    # --- High event count → retry after backoff ---

    def test_persistent_missing_high_event_count_retries_at_backoff_threshold(self, pipeline):
        """A persistently missing profile with >_RETRY_EVENT_COUNT_THRESHOLD events
        triggers a rebuild when its retry counter reaches a multiple of
        _RETRY_EVERY_N_CHECKS.

        Setup: run a first check (rebuild fires, leaving profiles still missing),
        then pre-set the retry counter so the immediately following check lands
        on the backoff threshold.  Mock event count to be well above the threshold.
        """
        rebuild_calls = []
        self._install_mock_rebuild(pipeline, rebuild_calls)

        # First check: all newly missing → rebuild fires.
        result1 = pipeline.periodic_health_check()
        assert result1["status"] == "rebuilt"
        assert len(rebuild_calls) == 1

        # Simulate (_RETRY_EVERY_N_CHECKS - 1) degraded checks having already
        # elapsed by pre-seeding the retry counter.  The next increment will
        # bring each profile's count to exactly _RETRY_EVERY_N_CHECKS, which
        # is a multiple of _RETRY_EVERY_N_CHECKS → retry fires.
        for profile_name in pipeline._last_health_check_missing:
            pipeline._rebuild_retry_count[profile_name] = _RETRY_EVERY_N_CHECKS - 1

        # Override event count to report a high number (well above threshold).
        pipeline._count_qualifying_events = lambda p: _RETRY_EVENT_COUNT_THRESHOLD + 500

        # Second check: same profiles still missing, event count is high, backoff
        # threshold reached → rebuild should fire again.
        result2 = pipeline.periodic_health_check()

        assert result2["status"] == "rebuilt", (
            f"Expected 'rebuilt' but got {result2['status']!r}. "
            "Persistent missing profile with high event count should retry at backoff threshold."
        )
        assert len(rebuild_calls) == 2

    def test_persistent_missing_high_event_count_rebuild_result_in_response(self, pipeline):
        """When a persistent retry fires, the rebuild result is included in the response."""
        rebuild_calls = []
        expected_result = {"missing_before": ["temporal"], "rebuilt": [], "skipped": False}
        pipeline.check_and_rebuild_missing_profiles = lambda: rebuild_calls.append(True) or expected_result

        pipeline.periodic_health_check()  # First: newly missing → rebuild.

        for profile_name in pipeline._last_health_check_missing:
            pipeline._rebuild_retry_count[profile_name] = _RETRY_EVERY_N_CHECKS - 1
        pipeline._count_qualifying_events = lambda p: _RETRY_EVENT_COUNT_THRESHOLD + 500

        result2 = pipeline.periodic_health_check()

        assert result2["status"] == "rebuilt"
        assert result2["rebuild_result"] == expected_result

    # --- Low event count → no retry ---

    def test_persistent_missing_low_event_count_stays_degraded(self, pipeline):
        """A persistently missing profile with <=_RETRY_EVENT_COUNT_THRESHOLD events
        does NOT trigger a rebuild, even when the backoff counter is at threshold.

        This is the key guard that prevents wasted rebuilds for profiles like
        'linguistic' on inbound-only systems that have only a handful of qualifying
        (outbound) events.
        """
        rebuild_calls = []
        self._install_mock_rebuild(pipeline, rebuild_calls)

        # First check: newly missing → rebuild fires.
        pipeline.periodic_health_check()
        assert len(rebuild_calls) == 1

        # Bring counter to backoff threshold.
        for profile_name in pipeline._last_health_check_missing:
            pipeline._rebuild_retry_count[profile_name] = _RETRY_EVERY_N_CHECKS - 1

        # Override event count to report a LOW number (at or below threshold).
        pipeline._count_qualifying_events = lambda p: _RETRY_EVENT_COUNT_THRESHOLD

        # Second check: same profiles still missing, event count is low → no retry.
        result2 = pipeline.periodic_health_check()

        assert result2["status"] == "degraded", (
            f"Expected 'degraded' but got {result2['status']!r}. "
            "Persistent missing profile with low event count should NOT retry."
        )
        assert len(rebuild_calls) == 1  # No additional rebuild.

    def test_persistent_missing_zero_event_count_stays_degraded(self, pipeline):
        """A persistently missing profile with 0 qualifying events stays degraded.

        Covers the inbound-only linguistic case (0 sent messages ever).
        """
        rebuild_calls = []
        self._install_mock_rebuild(pipeline, rebuild_calls)

        pipeline.periodic_health_check()

        for profile_name in pipeline._last_health_check_missing:
            pipeline._rebuild_retry_count[profile_name] = _RETRY_EVERY_N_CHECKS - 1
        pipeline._count_qualifying_events = lambda p: 0

        result = pipeline.periodic_health_check()

        assert result["status"] == "degraded"
        assert len(rebuild_calls) == 1

    # --- Counter increment behaviour ---

    def test_retry_counter_increments_each_degraded_check(self, pipeline):
        """The retry counter increments on every check where the profile is
        persistently missing and NOT in the retry-fire window.

        Verifies that the counter accumulates correctly so that after exactly
        _RETRY_EVERY_N_CHECKS degraded checks the retry will fire.
        """
        rebuild_calls = []
        self._install_mock_rebuild(pipeline, rebuild_calls)

        # First check: newly missing → rebuild fires.
        pipeline.periodic_health_check()

        # Override to low event count so no retries fire (counter increments only).
        pipeline._count_qualifying_events = lambda p: 0

        # Run (_RETRY_EVERY_N_CHECKS - 1) more checks; all should be degraded.
        for expected_count in range(1, _RETRY_EVERY_N_CHECKS):
            result = pipeline.periodic_health_check()
            assert result["status"] == "degraded"
            # Each profile's counter should equal expected_count.
            for profile_name in result["missing"]:
                assert pipeline._rebuild_retry_count.get(profile_name, 0) == expected_count

        assert len(rebuild_calls) == 1  # Only the initial newly-missing rebuild.

    def test_retry_does_not_fire_below_backoff_threshold(self, pipeline):
        """The retry must not fire until the counter is a multiple of _RETRY_EVERY_N_CHECKS.

        Runs exactly (_RETRY_EVERY_N_CHECKS - 1) degraded checks with high event
        count and verifies that rebuild is still suppressed on all of them.
        """
        rebuild_calls = []
        self._install_mock_rebuild(pipeline, rebuild_calls)

        # First check: newly missing → rebuild.
        pipeline.periodic_health_check()

        # High event count but counter not yet at threshold.
        pipeline._count_qualifying_events = lambda p: _RETRY_EVENT_COUNT_THRESHOLD + 500

        for _ in range(_RETRY_EVERY_N_CHECKS - 1):
            result = pipeline.periodic_health_check()
            assert result["status"] == "degraded", (
                "Rebuild should NOT fire until counter reaches a multiple of _RETRY_EVERY_N_CHECKS"
            )

        assert len(rebuild_calls) == 1  # Only the initial rebuild.

    def test_retry_fires_exactly_at_backoff_threshold(self, pipeline):
        """After exactly _RETRY_EVERY_N_CHECKS degraded checks with high event count,
        the rebuild fires on the next check.

        This confirms the boundary condition: the Nth degraded check fires,
        not the (N-1)th.
        """
        rebuild_calls = []
        self._install_mock_rebuild(pipeline, rebuild_calls)

        # First check: rebuild fires.
        pipeline.periodic_health_check()

        # Run (_RETRY_EVERY_N_CHECKS - 1) silent checks (low event count).
        pipeline._count_qualifying_events = lambda p: 0
        for _ in range(_RETRY_EVERY_N_CHECKS - 1):
            pipeline.periodic_health_check()

        # Switch to high event count.  The next check is the Nth degraded check
        # and must fire the retry.
        pipeline._count_qualifying_events = lambda p: _RETRY_EVENT_COUNT_THRESHOLD + 500
        result = pipeline.periodic_health_check()

        assert result["status"] == "rebuilt"
        assert len(rebuild_calls) == 2

    # --- Degraded note preserved ---

    def test_degraded_note_present_when_retry_not_fired(self, pipeline):
        """When a persistently missing profile doesn't trigger a retry, the
        explanatory 'note' key is still present in the degraded response."""
        rebuild_calls = []
        self._install_mock_rebuild(pipeline, rebuild_calls)

        pipeline.periodic_health_check()  # First: newly missing.

        pipeline._count_qualifying_events = lambda p: 0  # Low event count.
        result = pipeline.periodic_health_check()

        assert result["status"] == "degraded"
        assert "note" in result
        assert "same profiles missing" in result["note"]
