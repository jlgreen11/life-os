"""Tests for :class:`producers.routine.RoutineProducer`.

Covers the behaviors called out in the Week 5 task body:

- **Fires at detected routine times only** — profile specifies
  ``(weekday, hour)``; outside that bucket the producer returns ``[]``.
- **Evidence includes last 3 occurrences** — the Moment's ``evidence``
  field is the first three elements of ``profile.last_occurrences``
  (newest-first per the detector's contract).

Plus surrounding correctness:

- Trigger filter: only ``time.tick`` events touch the DB.
- History gate: ``len(last_occurrences) < MIN_OCCURRENCES`` (3) means
  "not yet a routine" and the producer does not fire.
- Daily routines: ``weekday`` missing / ``None`` matches any day.
- Weekly routines: mismatched weekday does not fire.
- tz-offset shifts the local-now into the configured timezone.
- Microcopy format matches the spec template literally.
- Action is :class:`ActionKind.SET_REMINDER` with
  ``{"routine_key", "description"}`` in ``params``.
- Confidence prefers ``consistency`` when provided; falls back to a
  count-based curve; both cap at 0.9.
- Dedup: two ticks in the same local hour collapse to one row via the
  moments-table UNIQUE constraint.
- Malformed JSON, non-object profiles, and bad field types silently
  skip rather than raising.

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
from producers.routine import (
    DEFAULT_EXPIRY_SECONDS,
    MIN_OCCURRENCES,
    ROUTINE_PRODUCER_KEY,
    TRIGGER_EVENT_TYPES,
    RoutineProducer,
)
from storage import schema
from storage.repos.moments import MomentRepository

# 2026-04-26T12:00:00Z — Sunday (weekday == 6 in Python's Mon=0 convention).
REF_NOW = 1_777_204_800
REF_WEEKDAY_SUN = 6
REF_HOUR_UTC = 12
REF_DATE = "2026-04-26"
SECONDS_PER_HOUR = 3600


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
def producer(conn: sqlite3.Connection) -> RoutineProducer:
    """A producer pinned to ``REF_NOW`` and a deterministic id stream."""
    counter = {"n": 0}

    def _id() -> str:
        counter["n"] += 1
        return f"moment-{counter['n']:04d}"

    return RoutineProducer(conn, now_fn=lambda: REF_NOW, id_fn=_id)


def _insert_profile(
    conn: sqlite3.Connection,
    *,
    key: str = "plan_week",
    description: str = "plan your week",
    weekday: int | None = REF_WEEKDAY_SUN,
    hour: int = REF_HOUR_UTC,
    tz_offset_hours: int = 0,
    last_occurrences: list[str] | None = None,
    consistency: float | None = None,
    extra: dict | None = None,
) -> dict:
    """Insert a routine profile.

    Defaults: Sunday 12:00 UTC (== REF_NOW), three recent
    occurrences (satisfies the history gate), no consistency field.
    """
    profile: dict = {
        "description": description,
        "hour": hour,
        "tz_offset_hours": tz_offset_hours,
        "last_occurrences": (last_occurrences if last_occurrences is not None else ["evt-1", "evt-2", "evt-3"]),
    }
    if weekday is not None:
        profile["weekday"] = weekday
    if consistency is not None:
        profile["consistency"] = consistency
    if extra:
        profile.update(extra)
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (ROUTINE_PRODUCER_KEY, key, json.dumps(profile)),
    )
    conn.commit()
    return profile


def _trigger_event(event_type: str = "time.tick", id_: str = "tick-1") -> dict:
    return {"id": id_, "type": event_type, "timestamp": REF_NOW, "source": "test"}


# ---------------------------------------------------------------------------
# Trigger filter
# ---------------------------------------------------------------------------


def test_non_trigger_event_returns_empty(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """An event the producer doesn't care about must short-circuit to []."""
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": "email.received"}))
    assert out == []


def test_observe_handles_time_tick_type(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe(_trigger_event("time.tick")))
    assert len(out) == 1


def test_trigger_event_type_set_matches_spec() -> None:
    assert TRIGGER_EVENT_TYPES == {"time.tick"}


# ---------------------------------------------------------------------------
# History gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2])
def test_no_moment_when_occurrences_below_min(conn: sqlite3.Connection, producer: RoutineProducer, n: int) -> None:
    """<3 occurrences means the detector is not yet confident; do not fire."""
    assert n < MIN_OCCURRENCES
    _insert_profile(conn, last_occurrences=[f"evt-{i}" for i in range(n)])
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_moment_emitted_at_min_occurrences_boundary(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """Exactly 3 occurrences is enough; producer fires."""
    _insert_profile(conn, last_occurrences=[f"evt-{i}" for i in range(MIN_OCCURRENCES)])
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Routine-time matching
# ---------------------------------------------------------------------------


def test_moment_emitted_on_matching_weekday_and_hour(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    _insert_profile(conn, weekday=REF_WEEKDAY_SUN, hour=REF_HOUR_UTC)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1


def test_no_moment_when_hour_mismatched(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """REF_NOW hour is 12; a routine scheduled at 09:00 does not fire."""
    _insert_profile(conn, hour=9)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_no_moment_when_weekday_mismatched(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """REF_NOW weekday is Sunday (6); a Monday (0) routine does not fire."""
    _insert_profile(conn, weekday=0)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_weekday_missing_means_daily(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """No ``weekday`` key means "daily" — fires any day at the matching hour."""
    _insert_profile(conn, weekday=None, hour=REF_HOUR_UTC)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1


def test_weekday_explicit_none_means_daily(conn: sqlite3.Connection) -> None:
    """Explicit ``"weekday": null`` in JSON also means daily."""
    counter = {"n": 0}

    def _id() -> str:
        counter["n"] += 1
        return f"m-{counter['n']}"

    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            ROUTINE_PRODUCER_KEY,
            "daily",
            json.dumps(
                {
                    "description": "stand up",
                    "weekday": None,
                    "hour": REF_HOUR_UTC,
                    "last_occurrences": ["a", "b", "c"],
                }
            ),
        ),
    )
    conn.commit()
    p = RoutineProducer(conn, now_fn=lambda: REF_NOW, id_fn=_id)
    out = asyncio.run(p.observe(_trigger_event()))
    assert len(out) == 1


def test_tz_offset_shifts_local_hour(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """tz_offset_hours=-5 shifts UTC 12:00 to local 07:00; Sunday is unchanged."""
    _insert_profile(conn, tz_offset_hours=-5, hour=7, weekday=REF_WEEKDAY_SUN)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1


def test_tz_offset_can_shift_weekday(conn: sqlite3.Connection) -> None:
    """tz_offset_hours=+14 pushes Sunday 12:00 UTC to Monday 02:00 local."""
    counter = {"n": 0}

    def _id() -> str:
        counter["n"] += 1
        return f"m-{counter['n']}"

    # weekday=0 (Monday), hour=2, tz +14 → should fire
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            ROUTINE_PRODUCER_KEY,
            "early_bird",
            json.dumps(
                {
                    "description": "review inbox",
                    "weekday": 0,  # Monday
                    "hour": 2,
                    "tz_offset_hours": 14,
                    "last_occurrences": ["a", "b", "c"],
                }
            ),
        ),
    )
    conn.commit()
    p = RoutineProducer(conn, now_fn=lambda: REF_NOW, id_fn=_id)
    out = asyncio.run(p.observe(_trigger_event()))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Evidence & microcopy
# ---------------------------------------------------------------------------


def test_evidence_is_first_three_occurrences(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    _insert_profile(conn, last_occurrences=["e1", "e2", "e3", "e4", "e5"])
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].evidence == ["e1", "e2", "e3"]


def test_insight_microcopy_matches_spec(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    _insert_profile(conn, description="plan your week")
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].insight == "You usually plan your week. Want to start now?"


def test_proposed_action_is_set_reminder(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    _insert_profile(conn, key="plan_week", description="plan your week")
    out = asyncio.run(producer.observe(_trigger_event()))
    action = out[0].proposed_action
    assert action.kind is ActionKind.SET_REMINDER
    assert action.params == {"routine_key": "plan_week", "description": "plan your week"}


def test_moment_carries_source_insight_type_routine(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_trigger_event()))[0]
    assert moment.source_insight_type is InsightType.ROUTINE


def test_moment_expires_at_default_72_hours(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_trigger_event()))[0]
    assert moment.expires_at - moment.created_at == DEFAULT_EXPIRY_SECONDS


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_scales_with_consistency(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """When ``consistency`` is set: confidence = min(0.9, 0.4 + 0.5*c)."""
    _insert_profile(conn, consistency=0.6)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].confidence == pytest.approx(0.4 + 0.5 * 0.6)


def test_confidence_capped_at_0_9_by_consistency(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    _insert_profile(conn, consistency=1.0)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].confidence == 0.9


def test_confidence_falls_back_to_count_curve(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """No consistency → confidence = min(0.9, 0.4 + min(n, 20)/40)."""
    _insert_profile(conn, last_occurrences=[f"e{i}" for i in range(3)])
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].confidence == pytest.approx(0.4 + 3 / 40.0)


def test_confidence_count_curve_caps_at_0_9(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    _insert_profile(conn, last_occurrences=[f"e{i}" for i in range(100)])
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].confidence == 0.9


def test_consistency_out_of_range_is_clamped(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """A misbehaving detector writing c=1.5 is clamped, not crashed."""
    _insert_profile(conn, consistency=1.5)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out[0].confidence == 0.9


# ---------------------------------------------------------------------------
# Dedup via evidence_hash (per routine, day, hour)
# ---------------------------------------------------------------------------


def test_dedup_within_same_hour_collapses_to_one_row(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """Two ticks in the same (day, hour) yield the same evidence_hash."""
    _insert_profile(conn)
    first = asyncio.run(producer.observe(_trigger_event(id_="t1")))
    second = asyncio.run(producer.observe(_trigger_event(id_="t2")))
    assert len(first) == 1 and len(second) == 1
    assert first[0].evidence_hash == second[0].evidence_hash

    repo = MomentRepository(conn, now_fn=lambda: REF_NOW)
    id_first = repo.create(first[0])
    id_second = repo.create(second[0])
    assert id_first == id_second  # idempotent: collision returns existing id


def test_dedup_distinct_hash_across_routines(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """Two routine keys produce distinct hashes even at the same hour."""
    _insert_profile(conn, key="plan_week")
    _insert_profile(conn, key="review_inbox")
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 2
    assert out[0].evidence_hash != out[1].evidence_hash


# ---------------------------------------------------------------------------
# Malformed input fails open
# ---------------------------------------------------------------------------


def test_malformed_profile_json_is_skipped(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (ROUTINE_PRODUCER_KEY, "broken", "{not valid json"),
    )
    _insert_profile(conn, key="plan_week")
    conn.commit()
    out = asyncio.run(producer.observe(_trigger_event()))
    assert len(out) == 1  # the broken row is silently dropped


def test_non_object_profile_is_skipped(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (ROUTINE_PRODUCER_KEY, "list-not-obj", json.dumps([1, 2, 3])),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_profile_missing_description_is_skipped(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            ROUTINE_PRODUCER_KEY,
            "no-desc",
            json.dumps(
                {
                    "hour": REF_HOUR_UTC,
                    "weekday": REF_WEEKDAY_SUN,
                    "last_occurrences": ["a", "b", "c"],
                }
            ),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_profile_missing_hour_is_skipped(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            ROUTINE_PRODUCER_KEY,
            "no-hour",
            json.dumps(
                {
                    "description": "plan week",
                    "weekday": REF_WEEKDAY_SUN,
                    "last_occurrences": ["a", "b", "c"],
                }
            ),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_hour_out_of_range_is_skipped(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """hour=25 is invalid; do not crash, do not fire."""
    _insert_profile(conn, hour=25)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_weekday_out_of_range_is_skipped(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """weekday=9 is invalid; do not crash, do not fire."""
    _insert_profile(conn, weekday=9)
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_last_occurrences_non_list_is_skipped(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            ROUTINE_PRODUCER_KEY,
            "bad-occ",
            json.dumps(
                {
                    "description": "plan week",
                    "hour": REF_HOUR_UTC,
                    "weekday": REF_WEEKDAY_SUN,
                    "last_occurrences": "not a list",
                }
            ),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_trigger_event()))
    assert out == []


def test_profile_for_other_producer_is_ignored(conn: sqlite3.Connection, producer: RoutineProducer) -> None:
    """Only ``producer='routine'`` rows are read; other rows are inert."""
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
    assert out[0].source_insight_type is InsightType.ROUTINE


# ---------------------------------------------------------------------------
# Registry side-effect
# ---------------------------------------------------------------------------


def test_routine_producer_registers_under_insight_type() -> None:
    """Importing the module must populate the global PRODUCERS map."""
    from core.moment.producer import PRODUCERS

    assert PRODUCERS.get(InsightType.ROUTINE) is RoutineProducer


def test_unused_ref_date_constant_is_the_current_anchor_day() -> None:
    """Guard against someone changing REF_NOW without updating REF_DATE."""
    from datetime import UTC, datetime

    assert datetime.fromtimestamp(REF_NOW, tz=UTC).strftime("%Y-%m-%d") == REF_DATE
