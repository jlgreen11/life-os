"""Tests for :class:`producers.cadence.CadenceProducer`.

Covers the four behaviors called out in the Week 4 task body:

- **History gate** — no Moment if a profile has fewer than
  :data:`producers.cadence.MIN_HISTORY_COUNT` historical observations,
  even if it is otherwise eligible.
- **Drift threshold** — fires precisely once
  ``days_since_last_inbound > expected_cadence_days * 1.3``; does not
  fire at exactly the boundary or below.
- **Dedup via evidence_hash** — same profile observed twice produces
  the same ``evidence_hash``, and the
  ``UNIQUE (source_insight_type, evidence_hash)`` constraint on the
  ``moments`` table collapses the second insert to the first id.
- **Trigger filter** — events whose ``type`` is not in
  :data:`producers.cadence.INBOUND_EVENT_TYPES` short-circuit to ``[]``
  without touching the DB.

Plus surrounding correctness:

- Confidence scaling matches ``min(0.9, days_since / expected / 2)``.
- ``proposed_action`` is :class:`ActionKind.NUDGE` with ``contact_id``
  and ``channel`` in ``params``.
- Insight microcopy follows the format the task body specifies
  ("{N} days since you've heard from {Name}. Usual cadence {X} days.").
- Malformed profile JSON, non-object profile, missing fields, and
  empty ``last_inbound_event_ids`` all silently skip rather than
  raising.

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
from producers.cadence import (
    CADENCE_DRIFT_FACTOR,
    CADENCE_PRODUCER_KEY,
    DEFAULT_EXPIRY_SECONDS,
    INBOUND_EVENT_TYPES,
    MIN_HISTORY_COUNT,
    CadenceProducer,
)
from storage import schema
from storage.repos.moments import MomentRepository

REF_NOW = 1_777_204_800  # 2026-04-22T12:00:00Z, matches sibling test fixtures
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
def producer(conn: sqlite3.Connection) -> CadenceProducer:
    """A producer pinned to ``REF_NOW`` and a deterministic id stream."""
    counter = {"n": 0}

    def _id() -> str:
        counter["n"] += 1
        return f"moment-{counter['n']:04d}"

    return CadenceProducer(conn, now_fn=lambda: REF_NOW, id_fn=_id)


def _insert_profile(
    conn: sqlite3.Connection,
    contact_id: str,
    *,
    expected_days: float = 7.0,
    count: int = 12,
    days_since: float = 12.0,
    event_ids: list[str] | None = None,
    contact_name: str = "Alice",
    channel: str = "email",
    extra: dict | None = None,
) -> dict:
    """Insert a cadence profile that says ``contact_id`` is ``days_since`` days late.

    Returns the dict that was written so tests can assert against the
    same structure the producer reads.
    """
    last_inbound_ts = REF_NOW - int(days_since * SECONDS_PER_DAY)
    profile: dict = {
        "expected_cadence_days": expected_days,
        "count": count,
        "last_inbound_ts": last_inbound_ts,
        "last_inbound_event_ids": event_ids
        if event_ids is not None
        else [f"evt-{contact_id}-1", f"evt-{contact_id}-2", f"evt-{contact_id}-3"],
        "contact_name": contact_name,
        "channel": channel,
    }
    if extra:
        profile.update(extra)
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (CADENCE_PRODUCER_KEY, contact_id, json.dumps(profile)),
    )
    conn.commit()
    return profile


def _inbound_event(event_type: str = "email.received", id_: str = "trigger-1") -> dict:
    return {"id": id_, "type": event_type, "timestamp": REF_NOW, "source": "test"}


# ---------------------------------------------------------------------------
# Trigger filter
# ---------------------------------------------------------------------------


def test_non_inbound_event_returns_empty_without_db_hit(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """An event the producer doesn't care about must short-circuit to []."""
    _insert_profile(conn, "alice@example.com")
    out = asyncio.run(producer.observe({"id": "x", "type": "calendar.created"}))
    assert out == []


def test_observe_handles_email_received_type(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    _insert_profile(conn, "alice@example.com", days_since=12.0)
    out = asyncio.run(producer.observe(_inbound_event("email.received")))
    assert len(out) == 1


def test_observe_handles_message_received_type(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    _insert_profile(conn, "+15555550100", days_since=12.0, channel="sms")
    out = asyncio.run(producer.observe(_inbound_event("message.received")))
    assert len(out) == 1


def test_inbound_event_type_set_matches_spec() -> None:
    """Sanity: the trigger set is exactly the two types the task body names."""
    assert INBOUND_EVENT_TYPES == {"email.received", "message.received"}


# ---------------------------------------------------------------------------
# History gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_no_moment_when_history_below_min(conn: sqlite3.Connection, producer: CadenceProducer, count: int) -> None:
    """count < 5 must never produce a Moment, regardless of drift."""
    assert count < MIN_HISTORY_COUNT
    _insert_profile(conn, "alice@example.com", count=count, days_since=30.0)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_moment_emitted_at_min_history_with_drift(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """count == 5 (boundary) is enough history; with drift the producer fires."""
    _insert_profile(conn, "alice@example.com", count=MIN_HISTORY_COUNT, days_since=12.0)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Drift threshold
# ---------------------------------------------------------------------------


def test_no_moment_at_or_below_drift_boundary(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """At exactly expected * 1.3 the producer does not fire (strict >)."""
    expected = 7.0
    boundary_days = expected * CADENCE_DRIFT_FACTOR
    _insert_profile(
        conn,
        "alice@example.com",
        expected_days=expected,
        days_since=boundary_days,  # not strictly greater
    )
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_moment_emitted_just_past_drift_boundary(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """A nudge over the boundary fires."""
    expected = 7.0
    over_days = expected * CADENCE_DRIFT_FACTOR + 0.01
    _insert_profile(
        conn,
        "alice@example.com",
        expected_days=expected,
        days_since=over_days,
    )
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 1


def test_no_moment_when_within_normal_cadence(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """A contact heard from yesterday should not fire even with rich history."""
    _insert_profile(conn, "alice@example.com", expected_days=7.0, days_since=1.0)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


# ---------------------------------------------------------------------------
# Dedup via evidence_hash
# ---------------------------------------------------------------------------


def test_dedup_via_evidence_hash_collapses_to_one_row(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """Two observations of the same profile yield the same evidence_hash;
    persisting both lands one row in ``moments``."""
    _insert_profile(
        conn,
        "alice@example.com",
        expected_days=7.0,
        days_since=12.0,
        event_ids=["evt-a", "evt-b", "evt-c"],
    )
    first = asyncio.run(producer.observe(_inbound_event(id_="trigger-1")))
    second = asyncio.run(producer.observe(_inbound_event(id_="trigger-2")))
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].evidence_hash == second[0].evidence_hash

    repo = MomentRepository(conn, now_fn=lambda: REF_NOW)
    id_first = repo.create(first[0])
    id_second = repo.create(second[0])
    assert id_first == id_second  # idempotent: collision returns existing id


def test_dedup_resets_when_evidence_ids_change(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """A profile refresh that swaps the last-3 ids yields a new hash."""
    _insert_profile(
        conn,
        "alice@example.com",
        days_since=12.0,
        event_ids=["evt-a", "evt-b", "evt-c"],
    )
    first = asyncio.run(producer.observe(_inbound_event()))[0]

    # Simulate the signal extractor refreshing the rolling window.
    conn.execute(
        "UPDATE signal_profiles SET profile=? WHERE producer=? AND key=?",
        (
            json.dumps(
                {
                    "expected_cadence_days": 7.0,
                    "count": 13,
                    "last_inbound_ts": REF_NOW - 12 * SECONDS_PER_DAY,
                    "last_inbound_event_ids": ["evt-d", "evt-e", "evt-f"],
                    "contact_name": "Alice",
                    "channel": "email",
                }
            ),
            CADENCE_PRODUCER_KEY,
            "alice@example.com",
        ),
    )
    conn.commit()
    second = asyncio.run(producer.observe(_inbound_event()))[0]
    assert first.evidence_hash != second.evidence_hash


def test_evidence_hash_is_order_independent_for_same_set(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """The producer must inherit Producer.evidence_hash's ordering invariance."""
    _insert_profile(
        conn,
        "alice@example.com",
        days_since=12.0,
        event_ids=["evt-c", "evt-a", "evt-b"],
    )
    permuted_first = asyncio.run(producer.observe(_inbound_event()))[0]

    conn.execute(
        "UPDATE signal_profiles SET profile=? WHERE producer=? AND key=?",
        (
            json.dumps(
                {
                    "expected_cadence_days": 7.0,
                    "count": 13,
                    "last_inbound_ts": REF_NOW - 12 * SECONDS_PER_DAY,
                    "last_inbound_event_ids": ["evt-a", "evt-b", "evt-c"],
                    "contact_name": "Alice",
                    "channel": "email",
                }
            ),
            CADENCE_PRODUCER_KEY,
            "alice@example.com",
        ),
    )
    conn.commit()
    sorted_second = asyncio.run(producer.observe(_inbound_event()))[0]
    assert permuted_first.evidence_hash == sorted_second.evidence_hash


# ---------------------------------------------------------------------------
# Confidence + payload shape
# ---------------------------------------------------------------------------


def test_confidence_scales_with_drift(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """Confidence = min(0.9, days_since / expected / 2)."""
    _insert_profile(conn, "alice@example.com", expected_days=7.0, days_since=10.0)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out[0].confidence == pytest.approx(10.0 / 7.0 / 2.0)


def test_confidence_capped_at_0_9(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """The cap protects against runaway confidence on long-dormant contacts."""
    _insert_profile(conn, "alice@example.com", expected_days=7.0, days_since=200.0)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out[0].confidence == 0.9


def test_proposed_action_is_nudge_with_contact_and_channel(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    _insert_profile(
        conn,
        "alice@example.com",
        days_since=12.0,
        channel="email",
    )
    out = asyncio.run(producer.observe(_inbound_event()))
    action = out[0].proposed_action
    assert action.kind is ActionKind.NUDGE
    assert action.params == {"contact_id": "alice@example.com", "channel": "email"}


def test_insight_microcopy_includes_days_and_name(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    _insert_profile(
        conn,
        "alice@example.com",
        expected_days=7.0,
        days_since=12.0,
        contact_name="Alice",
    )
    insight = asyncio.run(producer.observe(_inbound_event()))[0].insight
    assert "12 days since you've heard from Alice" in insight
    assert "Usual cadence 7 days" in insight


def test_moment_carries_source_insight_type_cadence(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    _insert_profile(conn, "alice@example.com", days_since=12.0)
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.source_insight_type is InsightType.CADENCE


def test_moment_expires_at_default_72_hours(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    _insert_profile(conn, "alice@example.com", days_since=12.0)
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.expires_at - moment.created_at == DEFAULT_EXPIRY_SECONDS


def test_moment_evidence_uses_last_three_inbound_ids(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    _insert_profile(
        conn,
        "alice@example.com",
        days_since=12.0,
        event_ids=["evt-a", "evt-b", "evt-c", "evt-d", "evt-e"],
    )
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    # Producer uses the most-recent 3 (first 3 in the list, which the
    # signal extractor maintains in newest-first order).
    assert moment.evidence == ["evt-a", "evt-b", "evt-c"]


# ---------------------------------------------------------------------------
# Multi-profile + filtering
# ---------------------------------------------------------------------------


def test_only_drifted_contacts_emit_moments(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """One overdue contact emits one Moment; the on-cadence one is silent."""
    _insert_profile(conn, "alice@example.com", expected_days=7.0, days_since=12.0)
    _insert_profile(conn, "bob@example.com", expected_days=7.0, days_since=2.0)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 1
    assert out[0].proposed_action.params["contact_id"] == "alice@example.com"


def test_multiple_drifted_contacts_emit_one_moment_each(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    _insert_profile(conn, "alice@example.com", expected_days=7.0, days_since=12.0)
    _insert_profile(conn, "carol@example.com", expected_days=14.0, days_since=30.0)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 2
    contact_ids = {m.proposed_action.params["contact_id"] for m in out}
    assert contact_ids == {"alice@example.com", "carol@example.com"}


# ---------------------------------------------------------------------------
# Malformed input fails open
# ---------------------------------------------------------------------------


def test_malformed_profile_json_is_skipped(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (CADENCE_PRODUCER_KEY, "broken", "{not valid json"),
    )
    _insert_profile(conn, "alice@example.com", days_since=12.0)
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    # alice still fires; broken row is silently dropped.
    assert len(out) == 1
    assert out[0].proposed_action.params["contact_id"] == "alice@example.com"


def test_non_object_profile_is_skipped(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (CADENCE_PRODUCER_KEY, "list-not-obj", json.dumps([1, 2, 3])),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_profile_missing_required_fields_is_skipped(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (CADENCE_PRODUCER_KEY, "incomplete", json.dumps({"count": 7})),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_profile_with_empty_event_ids_is_skipped(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """No evidence ⇒ producer cannot honor evidence-backed contract → skip."""
    _insert_profile(conn, "alice@example.com", days_since=12.0, event_ids=[])
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_profile_with_zero_expected_cadence_is_skipped(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """A profile that says 'cadence is 0 days' is nonsense; don't divide by 0."""
    _insert_profile(conn, "alice@example.com", expected_days=0.0, days_since=12.0)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_profile_for_other_producer_is_ignored(conn: sqlite3.Connection, producer: CadenceProducer) -> None:
    """Only ``producer='cadence'`` rows are read; other producers' rows are inert."""
    _insert_profile(conn, "alice@example.com", days_since=12.0)
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            "relationship",  # different producer namespace
            "alice@example.com",
            json.dumps({"expected_cadence_days": 1.0, "count": 100}),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 1
    # Came from the cadence row, not the relationship row.
    assert out[0].source_insight_type is InsightType.CADENCE


# ---------------------------------------------------------------------------
# Registry side-effect
# ---------------------------------------------------------------------------


def test_cadence_producer_registers_under_insight_type() -> None:
    """Importing the module must populate the global PRODUCERS map."""
    from core.moment.producer import PRODUCERS

    assert PRODUCERS.get(InsightType.CADENCE) is CadenceProducer
