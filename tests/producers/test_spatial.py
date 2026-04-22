"""Tests for :class:`producers.spatial.SpatialProducer`.

Covers the four behaviors called out in the Week 5 task body:

- **Arrival trigger** — fires on ``context.location.updated`` events
  whose payload carries ``arrival: <place>`` for a place the spatial
  signal profile knows about, emitting the "You're at {Place}. Last
  time here you worked on {topic}." microcopy.
- **Departure trigger** — fires on ``departure: <place>`` events using
  ``payload.duration_minutes`` against the profile's
  ``avg_duration_minutes`` to emit "You've been at {Place} {X} min, avg
  {Y}.".
- **Action kind** — every Moment uses :class:`ActionKind.NOTE_OBSERVATION`
  (read-only in Phase 1).
- **Dedup within same location visit** — two arrivals to the same place
  on the same UTC day collapse to one row via the
  ``UNIQUE (source_insight_type, evidence_hash)`` constraint.

Plus surrounding correctness:

- Trigger filter — non-``context.location.updated`` events short-circuit
  without touching the DB.
- Min visit count — places with fewer than
  :data:`producers.spatial.MIN_VISIT_COUNT` (3) historical visits do not
  fire.
- Arrival without ``last_topic`` skips silently (nothing to recall).
- Departure without a numeric duration in payload or profile skips.
- Malformed profile JSON, non-object profile, and missing fields fail
  open.
- Confidence scales with ``visit_count``, capped at 0.9.

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
from producers.spatial import (
    ARRIVAL,
    DEFAULT_EXPIRY_SECONDS,
    DEPARTURE,
    MIN_VISIT_COUNT,
    SPATIAL_PRODUCER_KEY,
    TRIGGER_EVENT_TYPE,
    SpatialProducer,
)
from storage import schema
from storage.repos.moments import MomentRepository

REF_NOW = 1_777_204_800  # 2026-04-26T12:00:00Z
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
def producer(conn: sqlite3.Connection) -> SpatialProducer:
    """A producer pinned to ``REF_NOW`` and a deterministic id stream."""
    counter = {"n": 0}

    def _id() -> str:
        counter["n"] += 1
        return f"moment-{counter['n']:04d}"

    return SpatialProducer(conn, now_fn=lambda: REF_NOW, id_fn=_id)


def _insert_profile(
    conn: sqlite3.Connection,
    *,
    key: str = "Office",
    visit_count: int = 40,
    avg_duration_minutes: int | None = 180,
    last_topic: str | None = "API design review",
    last_event_ids: list[str] | None = None,
    place_name: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Insert a spatial profile.

    Defaults: a well-worn ``Office`` profile with 40 visits, 180-min
    average, a last_topic, and three historical event ids — the happy
    path for both arrival and departure microcopy. Pass ``None`` to
    drop a field from the profile.
    """
    profile: dict = {
        "visit_count": visit_count,
    }
    if avg_duration_minutes is not None:
        profile["avg_duration_minutes"] = avg_duration_minutes
    if last_topic is not None:
        profile["last_topic"] = last_topic
    profile["last_event_ids"] = (
        last_event_ids if last_event_ids is not None else ["evt-loc-1", "evt-loc-2", "evt-loc-3"]
    )
    if place_name is not None:
        profile["place_name"] = place_name
    if extra:
        profile.update(extra)
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (SPATIAL_PRODUCER_KEY, key, json.dumps(profile)),
    )
    conn.commit()
    return profile


def _arrival_event(place: str = "Office", id_: str = "loc-1") -> dict:
    return {
        "id": id_,
        "type": TRIGGER_EVENT_TYPE,
        "timestamp": REF_NOW,
        "source": "ios",
        "payload": {"arrival": place},
    }


def _departure_event(
    place: str = "Office",
    duration_minutes: int = 210,
    id_: str = "loc-2",
) -> dict:
    return {
        "id": id_,
        "type": TRIGGER_EVENT_TYPE,
        "timestamp": REF_NOW,
        "source": "ios",
        "payload": {"departure": place, "duration_minutes": duration_minutes},
    }


# ---------------------------------------------------------------------------
# Trigger filter
# ---------------------------------------------------------------------------


def test_non_trigger_event_returns_empty(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": "email.received", "payload": {"arrival": "Office"}}))
    assert out == []


def test_missing_payload_returns_empty(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": TRIGGER_EVENT_TYPE}))
    assert out == []


def test_payload_without_arrival_or_departure_returns_empty(
    conn: sqlite3.Connection, producer: SpatialProducer
) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": TRIGGER_EVENT_TYPE, "payload": {"other": "Office"}}))
    assert out == []


def test_non_dict_payload_returns_empty(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": TRIGGER_EVENT_TYPE, "payload": "nope"}))
    assert out == []


# ---------------------------------------------------------------------------
# Arrival happy path
# ---------------------------------------------------------------------------


def test_arrival_emits_expected_microcopy(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, last_topic="API design review")
    out = asyncio.run(producer.observe(_arrival_event()))
    assert len(out) == 1
    assert out[0].insight == "You're at Office. Last time here you worked on API design review."


def test_arrival_action_is_note_observation(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_arrival_event()))[0]
    assert moment.proposed_action.kind is ActionKind.NOTE_OBSERVATION
    assert moment.proposed_action.params == {"place": "Office", "kind": ARRIVAL}


def test_arrival_uses_display_name_from_profile(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    """``profile.place_name`` overrides the key for display."""
    _insert_profile(conn, key="office-main", place_name="The Office", last_topic="planning")
    out = asyncio.run(producer.observe({"id": "x", "type": TRIGGER_EVENT_TYPE, "payload": {"arrival": "office-main"}}))
    assert len(out) == 1
    assert "The Office" in out[0].insight


def test_arrival_without_last_topic_is_skipped(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, last_topic=None)
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out == []


def test_arrival_with_blank_last_topic_is_skipped(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, last_topic="   ")
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out == []


def test_arrival_strips_last_topic(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, last_topic="  deep work block  ")
    out = asyncio.run(producer.observe(_arrival_event()))
    assert "deep work block." in out[0].insight


# ---------------------------------------------------------------------------
# Departure happy path
# ---------------------------------------------------------------------------


def test_departure_emits_expected_microcopy(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, avg_duration_minutes=180)
    out = asyncio.run(producer.observe(_departure_event(duration_minutes=210)))
    assert len(out) == 1
    assert out[0].insight == "You've been at Office 210 min, avg 180."


def test_departure_action_carries_durations(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, avg_duration_minutes=180)
    out = asyncio.run(producer.observe(_departure_event(duration_minutes=210)))
    params = out[0].proposed_action.params
    assert params["place"] == "Office"
    assert params["kind"] == DEPARTURE
    assert params["duration_minutes"] == 210
    assert params["avg_duration_minutes"] == 180


def test_departure_without_duration_in_payload_is_skipped(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, avg_duration_minutes=180)
    out = asyncio.run(producer.observe({"id": "x", "type": TRIGGER_EVENT_TYPE, "payload": {"departure": "Office"}}))
    assert out == []


def test_departure_with_zero_duration_is_skipped(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    """A 0-minute visit is almost certainly GPS jitter, not a real visit."""
    _insert_profile(conn, avg_duration_minutes=180)
    out = asyncio.run(producer.observe(_departure_event(duration_minutes=0)))
    assert out == []


def test_departure_without_avg_in_profile_is_skipped(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, avg_duration_minutes=None)
    out = asyncio.run(producer.observe(_departure_event()))
    assert out == []


# ---------------------------------------------------------------------------
# Gating: visit count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2])
def test_no_moment_when_visit_count_below_min(conn: sqlite3.Connection, producer: SpatialProducer, count: int) -> None:
    assert count < MIN_VISIT_COUNT
    _insert_profile(conn, visit_count=count)
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out == []


def test_moment_at_min_visit_count_boundary(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, visit_count=MIN_VISIT_COUNT)
    out = asyncio.run(producer.observe(_arrival_event()))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Unknown place: no profile → no Moment
# ---------------------------------------------------------------------------


def test_unknown_place_yields_no_moment(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, key="Office")
    out = asyncio.run(producer.observe(_arrival_event(place="GymZ")))
    assert out == []


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_evidence_uses_profile_event_ids(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, last_event_ids=["e1", "e2", "e3", "e4"])
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out[0].evidence == ["e1", "e2", "e3"]


def test_evidence_falls_back_to_synthetic_when_profile_event_ids_empty(
    conn: sqlite3.Connection, producer: SpatialProducer
) -> None:
    _insert_profile(conn, last_event_ids=[])
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out[0].evidence == ["spatial:arrival:Office:2026-04-26"]


def test_evidence_falls_back_when_event_ids_missing(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    """A profile without last_event_ids still produces a Moment with synthetic evidence."""
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            SPATIAL_PRODUCER_KEY,
            "Home",
            json.dumps(
                {
                    "visit_count": 10,
                    "avg_duration_minutes": 600,
                    "last_topic": "reading",
                }
            ),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_arrival_event(place="Home")))
    assert out[0].evidence == ["spatial:arrival:Home:2026-04-26"]


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------


def test_source_insight_type_is_spatial(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_arrival_event()))[0]
    assert moment.source_insight_type is InsightType.SPATIAL


def test_moment_expires_at_default_72_hours(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_arrival_event()))[0]
    assert moment.expires_at - moment.created_at == DEFAULT_EXPIRY_SECONDS


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_scales_with_visit_count(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, visit_count=MIN_VISIT_COUNT)
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out[0].confidence == pytest.approx(0.3 + MIN_VISIT_COUNT / 50.0)


def test_confidence_capped_at_0_9(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    _insert_profile(conn, visit_count=9999)
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out[0].confidence == 0.9


# ---------------------------------------------------------------------------
# Dedup within same visit
# ---------------------------------------------------------------------------


def test_same_day_arrivals_collapse_in_repository(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    """Two arrivals to same place, same day → one row via UNIQUE constraint."""
    _insert_profile(conn)
    first = asyncio.run(producer.observe(_arrival_event(id_="a1")))
    second = asyncio.run(producer.observe(_arrival_event(id_="a2")))
    assert first[0].evidence_hash == second[0].evidence_hash

    repo = MomentRepository(conn, now_fn=lambda: REF_NOW)
    id_first = repo.create(first[0])
    id_second = repo.create(second[0])
    assert id_first == id_second  # UNIQUE collision returns existing id


def test_arrival_and_departure_same_day_are_distinct(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    """Arrival and departure on the same day yield distinct hashes (different kinds)."""
    _insert_profile(conn)
    arr = asyncio.run(producer.observe(_arrival_event()))
    dep = asyncio.run(producer.observe(_departure_event()))
    assert len(arr) == 1 and len(dep) == 1
    assert arr[0].evidence_hash != dep[0].evidence_hash


def test_arrivals_on_different_days_produce_distinct_hashes(
    conn: sqlite3.Connection,
) -> None:
    """An arrival the next UTC day yields a fresh hash — new day, new visit."""
    _insert_profile(conn)

    counter = {"n": 0}

    def _id() -> str:
        counter["n"] += 1
        return f"moment-{counter['n']:04d}"

    p1 = SpatialProducer(conn, now_fn=lambda: REF_NOW, id_fn=_id)
    p2 = SpatialProducer(conn, now_fn=lambda: REF_NOW + SECONDS_PER_DAY, id_fn=_id)
    out_a = asyncio.run(p1.observe(_arrival_event(id_="a1")))
    out_b = asyncio.run(p2.observe(_arrival_event(id_="a2")))
    assert len(out_a) == 1 and len(out_b) == 1
    assert out_a[0].evidence_hash != out_b[0].evidence_hash


# ---------------------------------------------------------------------------
# Malformed input fails open
# ---------------------------------------------------------------------------


def test_malformed_profile_json_is_skipped(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (SPATIAL_PRODUCER_KEY, "Office", "{not valid json"),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out == []


def test_non_object_profile_is_skipped(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (SPATIAL_PRODUCER_KEY, "Office", json.dumps([1, 2, 3])),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out == []


def test_profile_missing_visit_count_is_skipped(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            SPATIAL_PRODUCER_KEY,
            "Office",
            json.dumps({"last_topic": "x", "avg_duration_minutes": 10}),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out == []


def test_profile_for_other_producer_is_ignored(conn: sqlite3.Connection, producer: SpatialProducer) -> None:
    """A cadence row with key=Office must not leak into spatial lookups."""
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            "cadence",
            "Office",
            json.dumps({"expected_cadence_days": 1.0, "count": 100}),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_arrival_event()))
    assert out == []


# ---------------------------------------------------------------------------
# Registry side-effect
# ---------------------------------------------------------------------------


def test_spatial_producer_registers_under_insight_type() -> None:
    """Importing the module must populate the global PRODUCERS map."""
    from core.moment.producer import PRODUCERS

    assert PRODUCERS.get(InsightType.SPATIAL) is SpatialProducer
