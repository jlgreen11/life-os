"""Test that _fallback_follow_up_actions handles mixed tz-aware/naive timestamps.

Regression test for TypeError: can't compare offset-naive and offset-aware
datetimes — triggered when episode timestamps have inconsistent timezone
formatting (some with +00:00 suffix, some without).
"""

import pytest

from services.routine_detector.detector import RoutineDetector


@pytest.fixture
def detector(db):
    from storage.user_model_store import UserModelStore

    ums = UserModelStore(db)
    return RoutineDetector(db, ums, timezone="UTC")


def test_mixed_tz_naive_and_aware_timestamps(detector):
    """Mixing offset-naive and offset-aware timestamps should not raise TypeError."""
    # Trigger timestamps: offset-aware (with +00:00)
    trigger_timestamps = [
        "2026-02-18T10:00:00+00:00",
        "2026-02-19T10:00:00+00:00",
        "2026-02-20T10:00:00+00:00",
    ]

    # Fallback rows: offset-naive (no timezone suffix)
    all_fallback_rows = [
        ("email_received", "2026-02-18T10:00:00"),
        ("calendar_blocked", "2026-02-18T10:30:00"),
        ("email_received", "2026-02-19T10:00:00"),
        ("calendar_blocked", "2026-02-19T10:30:00"),
        ("email_received", "2026-02-20T10:00:00"),
        ("calendar_blocked", "2026-02-20T10:30:00"),
    ]

    # Should not raise TypeError
    results = detector._fallback_follow_up_actions(
        trigger_timestamps, "email_received", all_fallback_rows
    )

    # calendar_blocked follows email_received on 3 days
    assert len(results) >= 1
    follow_types = [r[0] for r in results]
    assert "calendar_blocked" in follow_types


def test_all_naive_timestamps(detector):
    """All offset-naive timestamps should work fine."""
    trigger_timestamps = [
        "2026-02-18T10:00:00",
        "2026-02-19T10:00:00",
        "2026-02-20T10:00:00",
    ]
    all_fallback_rows = [
        ("email_received", "2026-02-18T10:00:00"),
        ("meeting_scheduled", "2026-02-18T11:00:00"),
        ("email_received", "2026-02-19T10:00:00"),
        ("meeting_scheduled", "2026-02-19T11:00:00"),
        ("email_received", "2026-02-20T10:00:00"),
        ("meeting_scheduled", "2026-02-20T11:00:00"),
    ]

    results = detector._fallback_follow_up_actions(
        trigger_timestamps, "email_received", all_fallback_rows
    )
    assert len(results) >= 1


def test_all_aware_timestamps(detector):
    """All offset-aware timestamps should work fine."""
    trigger_timestamps = [
        "2026-02-18T10:00:00+00:00",
        "2026-02-19T10:00:00+00:00",
        "2026-02-20T10:00:00+00:00",
    ]
    all_fallback_rows = [
        ("email_received", "2026-02-18T10:00:00+00:00"),
        ("meeting_scheduled", "2026-02-18T11:00:00+00:00"),
        ("email_received", "2026-02-19T10:00:00+00:00"),
        ("meeting_scheduled", "2026-02-19T11:00:00+00:00"),
        ("email_received", "2026-02-20T10:00:00+00:00"),
        ("meeting_scheduled", "2026-02-20T11:00:00+00:00"),
    ]

    results = detector._fallback_follow_up_actions(
        trigger_timestamps, "email_received", all_fallback_rows
    )
    assert len(results) >= 1


def test_reverse_mixed_tz(detector):
    """Naive triggers + aware fallback rows should not raise TypeError."""
    trigger_timestamps = [
        "2026-02-18T10:00:00",
        "2026-02-19T10:00:00",
        "2026-02-20T10:00:00",
    ]
    all_fallback_rows = [
        ("email_received", "2026-02-18T10:00:00+00:00"),
        ("calendar_blocked", "2026-02-18T10:30:00+00:00"),
        ("email_received", "2026-02-19T10:00:00+00:00"),
        ("calendar_blocked", "2026-02-19T10:30:00+00:00"),
        ("email_received", "2026-02-20T10:00:00+00:00"),
        ("calendar_blocked", "2026-02-20T10:30:00+00:00"),
    ]

    results = detector._fallback_follow_up_actions(
        trigger_timestamps, "email_received", all_fallback_rows
    )
    assert len(results) >= 1
