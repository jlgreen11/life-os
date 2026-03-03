"""
Tests for InsightEngine time-based correlator caching.

The InsightEngine runs 14+ correlators on every call to generate_insights(),
each executing SQL queries.  A time-based cache skips re-running correlators
when the last successful run was within a configurable TTL, avoiding redundant
computation on repeated API calls (e.g. dashboard refreshes).

This test suite validates:
- Cache prevents duplicate correlator runs within the TTL window
- Correlators run again after the TTL expires
- cache_ttl_seconds=0 disables caching (every call runs correlators)
- Backwards-compatible constructor (no new required params)
"""

import time
from unittest.mock import patch

import pytest

from services.insight_engine.engine import InsightEngine


# =============================================================================
# Cache TTL — correlators skipped within window
# =============================================================================


@pytest.mark.asyncio
async def test_generate_insights_skips_correlators_within_ttl(db, user_model_store):
    """Second call within TTL should return immediately without running correlators."""
    engine = InsightEngine(db, user_model_store, cache_ttl_seconds=60.0)

    with patch.object(engine, "_place_frequency_insights", return_value=[]) as mock_corr:
        # First call — correlators run
        await engine.generate_insights()
        assert mock_corr.call_count == 1

        # Second call within TTL — correlators should NOT run
        result = await engine.generate_insights()
        assert mock_corr.call_count == 1  # still 1
        assert result == []


# =============================================================================
# Cache TTL — correlators run after expiry
# =============================================================================


@pytest.mark.asyncio
async def test_generate_insights_runs_correlators_after_ttl_expires(db, user_model_store):
    """Correlators should run again once the TTL has elapsed."""
    engine = InsightEngine(db, user_model_store, cache_ttl_seconds=0.1)

    with patch.object(engine, "_place_frequency_insights", return_value=[]) as mock_corr:
        # First call
        await engine.generate_insights()
        assert mock_corr.call_count == 1

        # Wait past the TTL
        time.sleep(0.15)

        # Second call — correlators should run again
        await engine.generate_insights()
        assert mock_corr.call_count == 2


# =============================================================================
# Cache disabled (TTL = 0)
# =============================================================================


@pytest.mark.asyncio
async def test_cache_disabled_when_ttl_zero(db, user_model_store):
    """Setting cache_ttl_seconds=0 should disable caching — every call runs correlators."""
    engine = InsightEngine(db, user_model_store, cache_ttl_seconds=0)

    with patch.object(engine, "_place_frequency_insights", return_value=[]) as mock_corr:
        await engine.generate_insights()
        await engine.generate_insights()
        await engine.generate_insights()
        assert mock_corr.call_count == 3


# =============================================================================
# Backwards compatibility
# =============================================================================


@pytest.mark.asyncio
async def test_backwards_compatible_constructor(db, user_model_store):
    """InsightEngine(db=..., ums=...) should still work without cache_ttl_seconds."""
    engine = InsightEngine(db, user_model_store)
    assert engine._insight_cache_ttl == 300.0
    assert engine._last_insight_run == 0.0


@pytest.mark.asyncio
async def test_backwards_compatible_constructor_with_kwargs(db, user_model_store):
    """Existing keyword-argument patterns should continue to work."""
    engine = InsightEngine(db=db, ums=user_model_store, timezone="America/New_York")
    assert engine._insight_cache_ttl == 300.0


# =============================================================================
# Cache reset
# =============================================================================


@pytest.mark.asyncio
async def test_manual_cache_reset_forces_rerun(db, user_model_store):
    """Resetting _last_insight_run to 0.0 should force a fresh correlator run."""
    engine = InsightEngine(db, user_model_store, cache_ttl_seconds=600.0)

    with patch.object(engine, "_place_frequency_insights", return_value=[]) as mock_corr:
        await engine.generate_insights()
        assert mock_corr.call_count == 1

        # Force cache reset
        engine._last_insight_run = 0.0

        await engine.generate_insights()
        assert mock_corr.call_count == 2


# =============================================================================
# First call always runs (cache cold)
# =============================================================================


@pytest.mark.asyncio
async def test_first_call_always_runs_correlators(db, user_model_store):
    """The very first call should always execute correlators regardless of TTL."""
    engine = InsightEngine(db, user_model_store, cache_ttl_seconds=9999.0)

    with patch.object(engine, "_place_frequency_insights", return_value=[]) as mock_corr:
        await engine.generate_insights()
        assert mock_corr.call_count == 1
