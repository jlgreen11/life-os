"""
Tests for _verify_and_retry_backfills() in LifeOS startup.

Verifies that after all 8 backfill methods run, the verification step:
  1. Detects which signal profiles are populated vs empty
  2. Retries backfill methods for missing profiles
  3. Logs correct summary messages with populated/missing counts
  4. Never crashes startup (fail-open) even if verification itself errors
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from main import LifeOS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_PROFILE_TYPES = [
    "relationships",
    "temporal",
    "topics",
    "linguistic",
    "linguistic_inbound",
    "cadence",
    "mood_signals",
    "spatial",
    "decision",
]


def _make_life_os_with_mocks(profile_data: dict[str, dict | None] | None = None):
    """Build a minimal LifeOS-like object with mocked stores and backfill methods.

    Args:
        profile_data: Maps profile_type → mock return value for get_signal_profile().
                      Use None to simulate a missing/empty profile.
    """
    if profile_data is None:
        profile_data = {}

    life_os = LifeOS.__new__(LifeOS)

    # Mock user_model_store.get_signal_profile
    mock_store = MagicMock()

    def _get_signal_profile(profile_type):
        return profile_data.get(profile_type)

    mock_store.get_signal_profile = MagicMock(side_effect=_get_signal_profile)
    life_os.user_model_store = mock_store

    # Mock db.get_connection (used to refresh the connection pool before retry)
    mock_db = MagicMock()
    life_os.db = mock_db

    # Mock all 8 backfill methods as AsyncMock
    life_os._backfill_relationship_profile_if_needed = AsyncMock()
    life_os._backfill_temporal_profile_if_needed = AsyncMock()
    life_os._backfill_topic_profile_if_needed = AsyncMock()
    life_os._backfill_linguistic_profile_if_needed = AsyncMock()
    life_os._backfill_cadence_profile_if_needed = AsyncMock()
    life_os._backfill_mood_signals_profile_if_needed = AsyncMock()
    life_os._backfill_spatial_profile_if_needed = AsyncMock()
    life_os._backfill_decision_profile_if_needed = AsyncMock()

    return life_os


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_all_profiles_populated():
    """When all 8 profiles have data, no retries should be triggered."""
    profile_data = {pt: {"profile_type": pt, "data": {}, "samples_count": 50} for pt in EXPECTED_PROFILE_TYPES}
    life_os = _make_life_os_with_mocks(profile_data)

    await life_os._verify_and_retry_backfills()

    # No backfill method should have been called (all profiles are populated)
    assert not life_os._backfill_relationship_profile_if_needed.called
    assert not life_os._backfill_temporal_profile_if_needed.called
    assert not life_os._backfill_topic_profile_if_needed.called
    assert not life_os._backfill_linguistic_profile_if_needed.called
    assert not life_os._backfill_cadence_profile_if_needed.called
    assert not life_os._backfill_mood_signals_profile_if_needed.called
    assert not life_os._backfill_spatial_profile_if_needed.called
    assert not life_os._backfill_decision_profile_if_needed.called


@pytest.mark.asyncio
async def test_verify_retries_missing_profiles():
    """When 2 profiles are missing, their backfill methods should be retried."""
    # Populate 6 profiles, leave spatial and decision empty
    profile_data = {}
    for pt in EXPECTED_PROFILE_TYPES:
        if pt not in ("spatial", "decision"):
            profile_data[pt] = {"profile_type": pt, "data": {}, "samples_count": 50}
        # spatial and decision will return None (missing)

    life_os = _make_life_os_with_mocks(profile_data)

    await life_os._verify_and_retry_backfills()

    # Only the missing profiles should trigger a retry
    assert life_os._backfill_spatial_profile_if_needed.called, "spatial should be retried"
    assert life_os._backfill_decision_profile_if_needed.called, "decision should be retried"

    # The already-populated profiles should NOT be retried
    assert not life_os._backfill_relationship_profile_if_needed.called
    assert not life_os._backfill_temporal_profile_if_needed.called
    assert not life_os._backfill_topic_profile_if_needed.called
    assert not life_os._backfill_linguistic_profile_if_needed.called
    assert not life_os._backfill_cadence_profile_if_needed.called
    assert not life_os._backfill_mood_signals_profile_if_needed.called


@pytest.mark.asyncio
async def test_verify_logs_final_status(caplog):
    """Verify the summary log message includes correct populated/missing counts."""
    # 5 populated, 4 missing (cadence, mood_signals, spatial, linguistic_inbound)
    profile_data = {}
    for pt in EXPECTED_PROFILE_TYPES:
        if pt not in ("cadence", "mood_signals", "spatial", "linguistic_inbound"):
            profile_data[pt] = {"profile_type": pt, "data": {}, "samples_count": 25}

    life_os = _make_life_os_with_mocks(profile_data)

    with caplog.at_level(logging.INFO, logger="main"):
        await life_os._verify_and_retry_backfills()

    # Check for the initial status log
    status_messages = [r.message for r in caplog.records if "Signal profile status" in r.message]
    assert len(status_messages) >= 1, "Should log initial profile status"
    assert "5/9" in status_messages[0], f"Expected 5/9 in status message, got: {status_messages[0]}"

    # Check that retry messages were logged for the missing profiles
    retry_messages = [r.message for r in caplog.records if "Retrying backfill" in r.message]
    assert len(retry_messages) == 4, f"Expected 4 retry messages, got {len(retry_messages)}"


@pytest.mark.asyncio
async def test_verify_failopen_on_error():
    """Verification errors must never crash startup (fail-open behavior)."""
    life_os = LifeOS.__new__(LifeOS)

    # Set up user_model_store to raise an exception on any call
    mock_store = MagicMock()
    mock_store.get_signal_profile = MagicMock(side_effect=RuntimeError("DB connection pool exhausted"))
    life_os.user_model_store = mock_store
    life_os.db = MagicMock()

    # Should complete without raising — the exception is caught internally
    await life_os._verify_and_retry_backfills()


@pytest.mark.asyncio
async def test_verify_retry_failure_does_not_crash():
    """A retry that itself raises an exception should not crash verification."""
    # All profiles missing
    life_os = _make_life_os_with_mocks(profile_data={})

    # Make one backfill method raise during retry
    life_os._backfill_spatial_profile_if_needed = AsyncMock(side_effect=RuntimeError("Script not found"))

    # Should complete without raising — individual retry failures are caught
    await life_os._verify_and_retry_backfills()

    # Other backfill retries should still have been attempted
    assert life_os._backfill_relationship_profile_if_needed.called
    assert life_os._backfill_temporal_profile_if_needed.called


@pytest.mark.asyncio
async def test_verify_zero_samples_treated_as_missing():
    """Profiles with samples_count=0 should be treated as missing and retried."""
    profile_data = {pt: {"profile_type": pt, "data": {}, "samples_count": 50} for pt in EXPECTED_PROFILE_TYPES}
    # Override two profiles to have 0 samples (populated but empty)
    profile_data["linguistic"] = {"profile_type": "linguistic", "data": {}, "samples_count": 0}
    profile_data["cadence"] = {"profile_type": "cadence", "data": {}, "samples_count": 0}

    life_os = _make_life_os_with_mocks(profile_data)

    await life_os._verify_and_retry_backfills()

    # Zero-sample profiles should trigger retries
    assert life_os._backfill_linguistic_profile_if_needed.called
    assert life_os._backfill_cadence_profile_if_needed.called

    # Non-zero profiles should not be retried
    assert not life_os._backfill_relationship_profile_if_needed.called


@pytest.mark.asyncio
async def test_verify_logs_warning_for_still_missing(caplog):
    """After retry, still-empty profiles should produce a WARNING log."""
    # All profiles missing, and they'll stay missing because the mocked
    # backfill methods don't actually populate the store
    life_os = _make_life_os_with_mocks(profile_data={})

    with caplog.at_level(logging.WARNING, logger="main"):
        await life_os._verify_and_retry_backfills()

    warning_messages = [
        r.message for r in caplog.records if r.levelno >= logging.WARNING and "still empty after retry" in r.message
    ]
    assert len(warning_messages) >= 1, "Should warn about profiles still empty after retry"
