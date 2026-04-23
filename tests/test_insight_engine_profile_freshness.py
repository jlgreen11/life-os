"""
Tests for signal profile freshness checking in InsightEngine.get_data_sufficiency_report().

The report now checks not only whether a profile has enough samples, but also
whether the data is fresh.  A profile that was last updated 10+ days ago should
report ``freshness='very_stale'`` and the status should be downgraded from
``'ready'`` to ``'stale_data'`` — even if sample counts are above the minimum.

Freshness bands:
  - 'fresh'      — updated within the last 24 hours
  - 'stale'      — updated 1–7 days ago
  - 'very_stale' — updated more than 7 days ago
  - 'unknown'    — updated_at missing or unparseable

Status downgrade rule:
  - profile with samples >= min_required AND freshness == 'very_stale' → 'stale_data'
  - profile with samples >= min_required AND freshness != 'very_stale'  → 'ready'
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.engine import InsightEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_signal_profile(db, profile_type: str, samples_count: int, updated_at: datetime) -> None:
    """Write a signal_profiles row with an explicit updated_at timestamp.

    Uses a direct INSERT rather than UserModelStore.update_signal_profile()
    so that the updated_at can be set to an arbitrary point in the past for
    testing staleness.

    Args:
        db: DatabaseManager instance with access to user_model.db.
        profile_type: E.g. ``'linguistic'``, ``'relationships'``.
        samples_count: Value to store in the samples_count column.
        updated_at: The desired last-updated timestamp (tz-aware UTC).
    """
    updated_at_str = updated_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles
               (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (profile_type, json.dumps({}), samples_count, updated_at_str),
        )


# ---------------------------------------------------------------------------
# Freshness field tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_profile_returns_fresh_freshness(db, user_model_store):
    """A profile updated 1 hour ago should report freshness='fresh'."""
    now = datetime.now(timezone.utc)
    _insert_signal_profile(db, "linguistic", samples_count=20, updated_at=now - timedelta(hours=1))

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    entry = report["_communication_style_insights"]
    assert entry["freshness"] == "fresh", f"Expected 'fresh', got {entry['freshness']!r}"
    assert entry["last_updated"] is not None
    assert entry["age_hours"] is not None
    assert entry["age_hours"] < 24.0


@pytest.mark.asyncio
async def test_stale_profile_returns_stale_freshness(db, user_model_store):
    """A profile updated 3 days ago should report freshness='stale'."""
    now = datetime.now(timezone.utc)
    _insert_signal_profile(db, "linguistic", samples_count=20, updated_at=now - timedelta(days=3))

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    entry = report["_communication_style_insights"]
    assert entry["freshness"] == "stale", f"Expected 'stale', got {entry['freshness']!r}"
    assert entry["age_hours"] is not None
    # 3 days = 72 hours; allow a small margin for test execution time
    assert 71.0 < entry["age_hours"] < 73.0


@pytest.mark.asyncio
async def test_very_stale_profile_returns_very_stale_freshness(db, user_model_store):
    """A profile updated 10 days ago should report freshness='very_stale'."""
    now = datetime.now(timezone.utc)
    _insert_signal_profile(db, "linguistic", samples_count=20, updated_at=now - timedelta(days=10))

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    entry = report["_communication_style_insights"]
    assert entry["freshness"] == "very_stale", f"Expected 'very_stale', got {entry['freshness']!r}"
    assert entry["age_hours"] is not None
    assert entry["age_hours"] > 7 * 24  # over 7 days in hours


# ---------------------------------------------------------------------------
# Status downgrade tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_very_stale_profile_with_enough_samples_is_stale_data(db, user_model_store):
    """A profile with sufficient samples but very stale data should be 'stale_data', not 'ready'."""
    now = datetime.now(timezone.utc)
    # linguistic requires 10 samples; insert 20 but 10 days stale
    _insert_signal_profile(db, "linguistic", samples_count=20, updated_at=now - timedelta(days=10))

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    entry = report["_communication_style_insights"]
    assert entry["status"] == "stale_data", (
        f"Expected status='stale_data' for very stale profile with enough samples, "
        f"got {entry['status']!r}"
    )
    assert entry["freshness"] == "very_stale"


@pytest.mark.asyncio
async def test_fresh_profile_with_enough_samples_is_ready(db, user_model_store):
    """A profile with sufficient samples and fresh data should be 'ready'."""
    now = datetime.now(timezone.utc)
    _insert_signal_profile(db, "linguistic", samples_count=20, updated_at=now - timedelta(hours=2))

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    entry = report["_communication_style_insights"]
    assert entry["status"] == "ready", f"Expected status='ready', got {entry['status']!r}"
    assert entry["freshness"] == "fresh"


@pytest.mark.asyncio
async def test_stale_but_not_very_stale_profile_remains_ready(db, user_model_store):
    """A profile updated 3 days ago (stale but not very_stale) should still be 'ready' if samples suffice."""
    now = datetime.now(timezone.utc)
    _insert_signal_profile(db, "linguistic", samples_count=20, updated_at=now - timedelta(days=3))

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    entry = report["_communication_style_insights"]
    # 'stale' (1-7 days) does NOT trigger downgrade — only 'very_stale' does
    assert entry["status"] == "ready", f"Expected status='ready' for stale (not very_stale), got {entry['status']!r}"
    assert entry["freshness"] == "stale"


@pytest.mark.asyncio
async def test_very_stale_but_insufficient_samples_is_partial(db, user_model_store):
    """A very stale profile with fewer samples than the minimum should be 'partial', not 'stale_data'."""
    now = datetime.now(timezone.utc)
    # linguistic requires 10; use only 5
    _insert_signal_profile(db, "linguistic", samples_count=5, updated_at=now - timedelta(days=10))

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    entry = report["_communication_style_insights"]
    # The status downgrade to 'stale_data' only applies when samples >= min_required.
    # Here samples < min_required → 'partial'.
    assert entry["status"] == "partial", f"Expected 'partial', got {entry['status']!r}"
    assert entry["freshness"] == "very_stale"


# ---------------------------------------------------------------------------
# Freshness summary tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_freshness_summary_counts_fresh(db, user_model_store):
    """freshness_summary.fresh should count profiles updated within 24 hours."""
    now = datetime.now(timezone.utc)
    _insert_signal_profile(db, "linguistic", samples_count=20, updated_at=now - timedelta(hours=1))

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    summary = report["freshness_summary"]
    assert summary["fresh"] >= 1, f"Expected at least 1 fresh, got {summary}"


@pytest.mark.asyncio
async def test_freshness_summary_counts_very_stale(db, user_model_store):
    """freshness_summary.very_stale should count profiles not updated in > 7 days."""
    now = datetime.now(timezone.utc)
    _insert_signal_profile(db, "linguistic", samples_count=20, updated_at=now - timedelta(days=15))
    _insert_signal_profile(db, "relationships", samples_count=15, updated_at=now - timedelta(days=8))

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    summary = report["freshness_summary"]
    assert summary["very_stale"] >= 2, (
        f"Expected at least 2 very_stale profiles, got {summary}"
    )


@pytest.mark.asyncio
async def test_freshness_summary_counts_unknown_when_no_profile(db, user_model_store):
    """Profiles with no data in DB should count as 'unknown' in the freshness summary."""
    # No profiles inserted — all will be missing from the DB
    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    summary = report["freshness_summary"]
    total = summary["fresh"] + summary["stale"] + summary["very_stale"] + summary["unknown"]
    # 9 profile-based correlators are checked
    assert total == 9, f"Expected 9 total profile freshness entries, got {total}: {summary}"
    # With no profiles in DB, all should be 'unknown'
    assert summary["unknown"] == 9, f"Expected all 9 to be 'unknown', got {summary}"


@pytest.mark.asyncio
async def test_freshness_summary_mixed_freshness(db, user_model_store):
    """freshness_summary should correctly count a mix of fresh, stale, and very_stale profiles."""
    now = datetime.now(timezone.utc)
    _insert_signal_profile(db, "linguistic", samples_count=20, updated_at=now - timedelta(hours=2))   # fresh
    _insert_signal_profile(db, "relationships", samples_count=15, updated_at=now - timedelta(days=3))  # stale
    _insert_signal_profile(db, "topics", samples_count=25, updated_at=now - timedelta(days=10))        # very_stale

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    summary = report["freshness_summary"]
    assert summary["fresh"] >= 1, f"Expected >= 1 fresh, got {summary}"
    assert summary["stale"] >= 1, f"Expected >= 1 stale, got {summary}"
    assert summary["very_stale"] >= 1, f"Expected >= 1 very_stale, got {summary}"


# ---------------------------------------------------------------------------
# Edge case: missing or malformed updated_at
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_updated_at_reports_unknown_freshness(db, user_model_store):
    """A profile with a non-parseable updated_at should report freshness='unknown'."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles
               (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            ("linguistic", json.dumps({}), 20, "not-a-date"),
        )

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    entry = report["_communication_style_insights"]
    assert entry["freshness"] == "unknown", f"Expected 'unknown', got {entry['freshness']!r}"
    assert entry["age_hours"] is None


@pytest.mark.asyncio
async def test_last_updated_field_is_present(db, user_model_store):
    """Each profile-based correlator entry must include last_updated and age_hours fields."""
    now = datetime.now(timezone.utc)
    _insert_signal_profile(db, "linguistic", samples_count=20, updated_at=now - timedelta(hours=5))

    engine = InsightEngine(db, user_model_store)
    report = await engine.get_data_sufficiency_report()

    entry = report["_communication_style_insights"]
    assert "last_updated" in entry, "Entry missing 'last_updated' key"
    assert "age_hours" in entry, "Entry missing 'age_hours' key"
    assert "freshness" in entry, "Entry missing 'freshness' key"
    assert entry["last_updated"] is not None
    assert isinstance(entry["age_hours"], float)
