"""Tests for :class:`producers.temporal.TemporalProducer`.

Covers the four behaviors called out in the Week 4 task body:

- **History gate** — no Moment if the temporal profile has fewer than
  :data:`producers.temporal.MIN_PROFILE_DAYS` (14) days of behavioral
  history, even when the user is currently inside a focus window.
- **Focus-window trigger** — fires when ``now`` falls inside a known
  high-focus window from the chronotype profile.
- **Calendar-gap trigger** — fires when ``now`` falls inside a calendar
  gap whose remaining duration is at least
  :data:`producers.temporal.MIN_GAP_SECONDS` (60 min).
- **Trigger filter** — events whose ``type`` is not in
  :data:`producers.temporal.TRIGGER_EVENT_TYPES` short-circuit to ``[]``
  without touching the DB.

Plus surrounding correctness:

- Per ``(day, local-hour)`` dedup: two pulses in the same hour collapse
  to one row via the moments-table UNIQUE constraint.
- Microcopy follows the format the task body specifies
  ("You have {X} min free. Historical focus pattern at this hour:
  {description}.").
- Action is :class:`ActionKind.SCHEDULE_BLOCK` with ``duration_minutes``
  and ``label`` in ``params``.
- Confidence scales with ``data_days``, capped at 0.9.
- Gap-with-evidence uses real evidence ids; focus-only uses a synthetic
  ``temporal:focus:HH-HH:YYYY-MM-DD`` id; gap-with-empty-ids falls back
  to a synthetic id rather than emitting evidence-less Moments.
- Malformed profile JSON, non-object profile, missing fields, and
  non-list focus_windows all silently skip rather than raising.

All tests use stdlib only (no freezegun) — clocks are injected via
``now_fn`` per the eng-review §1c convention.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator

import pytest

from core.moment.types import ActionKind, InsightType
from producers.temporal import (
    DEFAULT_EXPIRY_SECONDS,
    MIN_GAP_SECONDS,
    MIN_PROFILE_DAYS,
    TEMPORAL_PRODUCER_KEY,
    TRIGGER_EVENT_TYPES,
    TemporalProducer,
)
from storage import schema
from storage.repos.moments import MomentRepository

REF_NOW = 1_777_204_800  # 2026-04-26T12:00:00Z (UTC hour=12, date=2026-04-26)
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        c.execute(stmt)
    c.commit()
    yield c
    c.close()


@pytest.fixture
def producer(conn: sqlite3.Connection) -> TemporalProducer:
    """A producer pinned to ``REF_NOW`` and a deterministic id stream."""
    counter = {"n": 0}

    def _id() -> str:
        counter["n"] += 1
        return f"moment-{counter['n']:04d}"

    return TemporalProducer(conn, now_fn=lambda: REF_NOW, id_fn=_id)


def _insert_profile(
    conn: sqlite3.Connection,
    *,
    key: str = "self",
    data_days: int = 30,
    tz_offset_hours: int = 0,
    focus_windows: list[dict] | None = None,
    current_calendar_gaps: list[dict] | None = None,
    extra: dict | None = None,
) -> dict:
    """Insert a temporal profile.

    Defaults: 30 days of history, UTC, a single 11:00-13:00 focus window
    that covers REF_NOW (UTC hour 12), no calendar gaps. Returns the dict
    that was written.
    """
    profile: dict = {
        "data_days": data_days,
        "tz_offset_hours": tz_offset_hours,
        "focus_windows": (
            focus_windows
            if focus_windows is not None
            else [{"start_hour": 11, "end_hour": 13, "description": "deep work block"}]
        ),
        "current_calendar_gaps": current_calendar_gaps if current_calendar_gaps is not None else [],
    }
    if extra:
        profile.update(extra)
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (TEMPORAL_PRODUCER_KEY, key, json.dumps(profile)),
    )
    conn.commit()
    return profile


def _trigger_event(event_type: str = "time.tick", id_: str = "tick-1") -> dict:
    return {"id": id_, "type": event_type, "timestamp": REF_NOW, "source": "test"}


# ---------------------------------------------------------------------------
# Trigger filter
# ---------------------------------------------------------------------------


def test_non_trigger_event_returns_empty_without_db_hit(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """An event the producer doesn't care about must short-circuit to []."""
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": "email.received"}))
    assert out == []


def test_observe_handles_time_tick_type(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe(_trigger_event("time.tick")))
    assert len(out) == 1


def test_observe_handles_calendar_event_deleted_type(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe(_trigger_event("calendar.event.deleted")))
    assert len(out) == 1


def test_observe_handles_calendar_event_created_type(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe(_trigger_event("calendar.event.created")))
    assert len(out) == 1


def test_observe_handles_calendar_event_updated_type(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe(_trigger_event("calendar.event.updated")))
    assert len(out) == 1


def test_trigger_event_type_set_matches_spec() -> None:
    assert TRIGGER_EVENT_TYPES == {
        "time.tick",
        "calendar.event.created",
        "calendar.event.updated",
        "calendar.event.deleted",
    }


# ---------------------------------------------------------------------------
# History gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("days", [0, 1, 7, 13])
def test_no_moment_when_data_days_below_min(conn: sqlite3.Connection, producer: TemporalProducer, days: int) -> None:
    """data_days < 14 must never produce a Moment, even inside a focus window."""
    assert days < MIN_PROFILE_DAYS
    _insert_profile(conn, data_days=days)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_moment_emitted_at_min_data_days_boundary(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """data_days == 14 (boundary) is enough; producer fires."""
    _insert_profile(conn, data_days=MIN_PROFILE_DAYS)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Focus-window trigger
# ---------------------------------------------------------------------------


def test_no_moment_outside_focus_window(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """REF_NOW is hour 12 UTC; a 09:00-11:00 window does not cover it."""
    _insert_profile(
        conn,
        focus_windows=[{"start_hour": 9, "end_hour": 11, "description": "morning"}],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_moment_emitted_inside_focus_window(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(
        conn,
        focus_windows=[{"start_hour": 11, "end_hour": 13, "description": "deep work block"}],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1
    assert "deep work block" in out[0].insight


def test_focus_window_end_hour_is_exclusive(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """A window of [11, 12) does not cover hour 12 (REF_NOW)."""
    _insert_profile(
        conn,
        focus_windows=[{"start_hour": 11, "end_hour": 12, "description": "morning"}],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_tz_offset_shifts_focus_window_match(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """tz_offset_hours=-5 shifts UTC-12 to local-7; a 7:00-9:00 window covers it."""
    _insert_profile(
        conn,
        tz_offset_hours=-5,
        focus_windows=[{"start_hour": 7, "end_hour": 9, "description": "early morning"}],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1


def test_first_matching_focus_window_wins(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """Two overlapping windows: profile order decides which description shows."""
    _insert_profile(
        conn,
        focus_windows=[
            {"start_hour": 12, "end_hour": 13, "description": "lunch focus"},
            {"start_hour": 11, "end_hour": 14, "description": "midday block"},
        ],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert "lunch focus" in out[0].insight


# ---------------------------------------------------------------------------
# Calendar-gap trigger
# ---------------------------------------------------------------------------


def test_moment_emitted_for_open_gap_outside_focus_window(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """Gap-only (no focus window match) still fires with neutral description."""
    gap_end = REF_NOW + 90 * 60  # 90 min remaining
    _insert_profile(
        conn,
        focus_windows=[],  # no chronotype match at hour 12
        current_calendar_gaps=[{"start_ts": REF_NOW - 300, "end_ts": gap_end, "evidence_ids": ["evt-cal-1"]}],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1
    assert "open calendar block" in out[0].insight
    assert "90 min free" in out[0].insight


def test_no_moment_for_gap_below_min_remaining(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """Gap with <60 min remaining does not fire, even though now is in it."""
    gap_end = REF_NOW + (MIN_GAP_SECONDS - 1)
    _insert_profile(
        conn,
        focus_windows=[],
        current_calendar_gaps=[{"start_ts": REF_NOW - 600, "end_ts": gap_end, "evidence_ids": ["evt-1"]}],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_gap_at_exactly_min_remaining_fires(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """Boundary: remaining == 60 min should fire (≥ semantics)."""
    gap_end = REF_NOW + MIN_GAP_SECONDS
    _insert_profile(
        conn,
        focus_windows=[],
        current_calendar_gaps=[{"start_ts": REF_NOW - 600, "end_ts": gap_end, "evidence_ids": ["evt-1"]}],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1


def test_no_moment_for_gap_in_future(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """Gap that hasn't started yet doesn't fire."""
    _insert_profile(
        conn,
        focus_windows=[],
        current_calendar_gaps=[
            {
                "start_ts": REF_NOW + 3600,
                "end_ts": REF_NOW + 7200,
                "evidence_ids": ["evt-1"],
            }
        ],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_no_moment_for_gap_already_ended(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(
        conn,
        focus_windows=[],
        current_calendar_gaps=[
            {
                "start_ts": REF_NOW - 7200,
                "end_ts": REF_NOW - 60,
                "evidence_ids": ["evt-1"],
            }
        ],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_gap_evidence_ids_used_when_present(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(
        conn,
        focus_windows=[{"start_hour": 11, "end_hour": 13, "description": "deep work"}],
        current_calendar_gaps=[
            {
                "start_ts": REF_NOW - 60,
                "end_ts": REF_NOW + 5400,
                "evidence_ids": ["evt-cal-a", "evt-cal-b"],
            }
        ],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].evidence == ["evt-cal-a", "evt-cal-b"]


def test_gap_evidence_uses_at_most_three(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(
        conn,
        focus_windows=[],
        current_calendar_gaps=[
            {
                "start_ts": REF_NOW - 60,
                "end_ts": REF_NOW + 5400,
                "evidence_ids": ["e1", "e2", "e3", "e4", "e5"],
            }
        ],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].evidence == ["e1", "e2", "e3"]


def test_focus_window_borrows_description_for_gap(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """When focus window AND gap both fire, description comes from window."""
    _insert_profile(
        conn,
        focus_windows=[{"start_hour": 11, "end_hour": 13, "description": "deep work block"}],
        current_calendar_gaps=[
            {
                "start_ts": REF_NOW - 60,
                "end_ts": REF_NOW + 5400,  # 90 min remaining
                "evidence_ids": ["evt-cal-1"],
            }
        ],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert "deep work block" in out[0].insight
    assert "90 min free" in out[0].insight  # gap-derived duration
    assert out[0].evidence == ["evt-cal-1"]  # gap evidence wins


# ---------------------------------------------------------------------------
# Synthetic evidence fallback
# ---------------------------------------------------------------------------


def test_focus_only_uses_synthetic_evidence(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(
        conn,
        focus_windows=[{"start_hour": 11, "end_hour": 13, "description": "deep work"}],
        current_calendar_gaps=[],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    # REF_NOW = 2026-04-26T12:00Z, window 11-13
    assert out[0].evidence == ["temporal:focus:11-13:2026-04-26"]


def test_gap_with_empty_evidence_ids_falls_back_to_synthetic(
    conn: sqlite3.Connection, producer: TemporalProducer
) -> None:
    """A gap with no evidence_ids must still produce non-empty evidence."""
    _insert_profile(
        conn,
        focus_windows=[],
        current_calendar_gaps=[{"start_ts": REF_NOW - 60, "end_ts": REF_NOW + 5400, "evidence_ids": []}],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out[0].evidence) == 1
    assert out[0].evidence[0].startswith("temporal:gap:")


# ---------------------------------------------------------------------------
# Microcopy + payload shape
# ---------------------------------------------------------------------------


def test_insight_microcopy_format(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """Insight matches the spec template literally."""
    _insert_profile(
        conn,
        focus_windows=[{"start_hour": 11, "end_hour": 13, "description": "deep work block"}],
    )
    insight = asyncio.run(producer.observe(_trigger_event()))[0].insight
    # Default focus-only path: window covers 11-13, current is hour 12, 60 min remaining.
    assert insight == "You have 60 min free. Historical focus pattern at this hour: deep work block."


def test_proposed_action_is_schedule_block(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(
        conn,
        focus_windows=[{"start_hour": 11, "end_hour": 13, "description": "deep work"}],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    action = out[0].proposed_action
    assert action.kind is ActionKind.SCHEDULE_BLOCK
    assert action.params == {"duration_minutes": 60, "label": "deep work"}


def test_moment_carries_source_insight_type_temporal(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_trigger_event()))[0]
    assert moment.source_insight_type is InsightType.TEMPORAL


def test_moment_expires_at_default_72_hours(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_trigger_event()))[0]
    assert moment.expires_at - moment.created_at == DEFAULT_EXPIRY_SECONDS


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_scales_with_data_days(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """confidence = min(0.9, 0.4 + data_days/60)."""
    _insert_profile(conn, data_days=14)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].confidence == pytest.approx(0.4 + 14 / 60.0)


def test_confidence_capped_at_0_9(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    _insert_profile(conn, data_days=365)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].confidence == 0.9


# ---------------------------------------------------------------------------
# Dedup via evidence_hash (per day, hour)
# ---------------------------------------------------------------------------


def test_dedup_within_same_hour_collapses_to_one_row(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """Two pulses in the same (day, hour) yield the same evidence_hash."""
    _insert_profile(conn)
    first = asyncio.run(producer.observe(_trigger_event(id_="t1")))
    second = asyncio.run(producer.observe(_trigger_event(id_="t2")))
    assert len(first) == 1 and len(second) == 1
    assert first[0].evidence_hash == second[0].evidence_hash

    repo = MomentRepository(conn, now_fn=lambda: REF_NOW)
    id_first = repo.create(first[0])
    id_second = repo.create(second[0])
    assert id_first == id_second  # idempotent: collision returns existing id


def test_dedup_distinct_hash_across_different_keys(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """Two profile keys produce distinct hashes even with same hour."""
    _insert_profile(conn, key="self")
    _insert_profile(conn, key="weekend")
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 2
    assert out[0].evidence_hash != out[1].evidence_hash


def test_dedup_distinct_hash_across_hours(conn: sqlite3.Connection) -> None:
    """A pulse one hour later yields a distinct evidence_hash."""
    counter = {"n": 0}

    def _id() -> str:
        counter["n"] += 1
        return f"m-{counter['n']:04d}"

    _insert_profile(
        conn,
        focus_windows=[{"start_hour": 11, "end_hour": 14, "description": "focus"}],
    )
    p1 = TemporalProducer(conn, now_fn=lambda: REF_NOW, id_fn=_id)
    p2 = TemporalProducer(conn, now_fn=lambda: REF_NOW + SECONDS_PER_HOUR, id_fn=_id)
    out_a = asyncio.run(p1.observe(_trigger_event()))
    out_b = asyncio.run(p2.observe(_trigger_event()))
    assert len(out_a) == 1 and len(out_b) == 1
    assert out_a[0].evidence_hash != out_b[0].evidence_hash


# ---------------------------------------------------------------------------
# Malformed input fails open
# ---------------------------------------------------------------------------


def test_malformed_profile_json_is_skipped(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (TEMPORAL_PRODUCER_KEY, "broken", "{not valid json"),
    )
    _insert_profile(conn, key="self")
    conn.commit()
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1  # the broken row is silently dropped


def test_non_object_profile_is_skipped(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (TEMPORAL_PRODUCER_KEY, "list-not-obj", json.dumps([1, 2, 3])),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_profile_missing_data_days_is_skipped(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (TEMPORAL_PRODUCER_KEY, "incomplete", json.dumps({"focus_windows": []})),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_focus_windows_non_list_is_skipped(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """Malformed focus_windows: producer skips them but still considers gaps."""
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            TEMPORAL_PRODUCER_KEY,
            "broken-windows",
            json.dumps({"data_days": 30, "focus_windows": "not a list"}),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_focus_window_with_missing_fields_is_skipped(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """A bad window in a list does not crash; valid windows still match."""
    _insert_profile(
        conn,
        focus_windows=[
            {"description": "missing hours"},  # bad
            {"start_hour": 11, "end_hour": 13, "description": "good window"},
        ],
    )
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1
    assert "good window" in out[0].insight


def test_profile_for_other_producer_is_ignored(conn: sqlite3.Connection, producer: TemporalProducer) -> None:
    """Only ``producer='temporal'`` rows are read; other producers' rows are inert."""
    _insert_profile(conn)
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            "cadence",
            "alice",
            json.dumps({"expected_cadence_days": 1.0, "count": 100}),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1
    assert out[0].source_insight_type is InsightType.TEMPORAL


# ---------------------------------------------------------------------------
# Registry side-effect
# ---------------------------------------------------------------------------


def test_temporal_producer_registers_under_insight_type() -> None:
    """Importing the module must populate the global PRODUCERS map."""
    from core.moment.producer import PRODUCERS

    assert PRODUCERS.get(InsightType.TEMPORAL) is TemporalProducer
