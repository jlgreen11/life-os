"""Tests for :class:`producers.relationship.RelationshipProducer`.

Covers the three behaviors called out in the Week 4 task body:

- **Interaction gate** — no Moment if ``total_interactions < 20``,
  even if the ratio collapse is dramatic.
- **Real drop fires** — a contact with ``previous_ratio > 0.5`` and
  ``current_ratio < 0.3`` over at least 20 interactions emits a
  Moment with ``ActionKind.NUDGE``.
- **Idempotent per week** — two observations of the same profile in
  the same ISO week share an ``evidence_hash`` and the
  ``UNIQUE (source_insight_type, evidence_hash)`` constraint on
  ``moments`` collapses the second insert to the first id; advancing
  to a new ISO week yields a fresh hash.

Plus surrounding correctness:

- Strict boundary behavior: ``previous_ratio == 0.5`` does not fire
  (task says "previously > 0.5"); ``current_ratio == 0.3`` does not
  fire ("drops below 0.3").
- Confidence = ``min(0.9, (previous - current) / previous)``.
- ``proposed_action`` is :class:`ActionKind.NUDGE` with ``contact_id``
  and ``channel`` in ``params``.
- Insight microcopy follows the format the task body specifies
  ("You've been replying less to {Name}. Outbound dropped {X}%.").
- Trigger-filter parity with cadence: non-inbound events short-circuit
  to ``[]`` without touching the DB.
- Malformed profile JSON, non-object profile, missing fields, empty
  ``last_event_ids``, and non-list ``last_event_ids`` all silently
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
from producers.relationship import (
    CURRENT_RATIO_MAX,
    DEFAULT_EXPIRY_SECONDS,
    INBOUND_EVENT_TYPES,
    MIN_INTERACTION_COUNT,
    PREVIOUS_RATIO_MIN,
    RELATIONSHIP_PRODUCER_KEY,
    RelationshipProducer,
)
from storage import schema
from storage.repos.moments import MomentRepository

# 2026-04-22T12:00:00Z → ISO week 2026-W17. Matches sibling cadence test fixture.
REF_NOW = 1_777_204_800
SECONDS_PER_WEEK = 7 * 86400


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
def producer(conn: sqlite3.Connection) -> RelationshipProducer:
    """A producer pinned to ``REF_NOW`` with a deterministic id stream."""
    counter = {"n": 0}

    def _id() -> str:
        counter["n"] += 1
        return f"moment-{counter['n']:04d}"

    return RelationshipProducer(conn, now_fn=lambda: REF_NOW, id_fn=_id)


def _insert_profile(
    conn: sqlite3.Connection,
    contact_id: str,
    *,
    previous_ratio: float = 0.7,
    current_ratio: float = 0.2,
    total_interactions: int = 30,
    event_ids: list[str] | None = None,
    contact_name: str = "Alice",
    channel: str = "email",
    extra: dict | None = None,
) -> dict:
    profile: dict = {
        "previous_ratio": previous_ratio,
        "current_ratio": current_ratio,
        "total_interactions": total_interactions,
        "last_event_ids": event_ids
        if event_ids is not None
        else [f"evt-{contact_id}-1", f"evt-{contact_id}-2", f"evt-{contact_id}-3"],
        "contact_name": contact_name,
        "channel": channel,
    }
    if extra:
        profile.update(extra)
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (RELATIONSHIP_PRODUCER_KEY, contact_id, json.dumps(profile)),
    )
    conn.commit()
    return profile


def _inbound_event(event_type: str = "email.received", id_: str = "trigger-1") -> dict:
    return {"id": id_, "type": event_type, "timestamp": REF_NOW, "source": "test"}


# ---------------------------------------------------------------------------
# Trigger filter
# ---------------------------------------------------------------------------


def test_non_inbound_event_returns_empty(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    _insert_profile(conn, "alice@example.com")
    out = asyncio.run(producer.observe({"id": "x", "type": "calendar.created"}))
    assert out == []


def test_observe_handles_email_received(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    _insert_profile(conn, "alice@example.com")
    out = asyncio.run(producer.observe(_inbound_event("email.received")))
    assert len(out) == 1


def test_observe_handles_message_received(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    _insert_profile(conn, "+15555550100", channel="sms")
    out = asyncio.run(producer.observe(_inbound_event("message.received")))
    assert len(out) == 1


def test_inbound_event_type_set_matches_spec() -> None:
    assert INBOUND_EVENT_TYPES == {"email.received", "message.received"}


# ---------------------------------------------------------------------------
# Interaction gate (task body: "0 Moment if <20 interactions")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 5, 10, 19])
def test_no_moment_when_interactions_below_min(
    conn: sqlite3.Connection, producer: RelationshipProducer, count: int
) -> None:
    assert count < MIN_INTERACTION_COUNT
    _insert_profile(conn, "alice@example.com", total_interactions=count)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_moment_fires_at_min_interactions(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    """count == 20 (boundary) is enough interactions for the producer to fire."""
    _insert_profile(conn, "alice@example.com", total_interactions=MIN_INTERACTION_COUNT)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Ratio boundaries (task body: "drops below 0.3 (previously > 0.5)")
# ---------------------------------------------------------------------------


def test_no_moment_when_previous_ratio_at_baseline(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    """previous_ratio == 0.5 is NOT "previously > 0.5" (strict). Do not fire."""
    _insert_profile(conn, "alice@example.com", previous_ratio=PREVIOUS_RATIO_MIN, current_ratio=0.1)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_no_moment_when_previous_ratio_below_baseline(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    _insert_profile(conn, "alice@example.com", previous_ratio=0.4, current_ratio=0.1)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_no_moment_when_current_ratio_at_threshold(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    """current_ratio == 0.3 is NOT "drops below 0.3" (strict). Do not fire."""
    _insert_profile(conn, "alice@example.com", previous_ratio=0.8, current_ratio=CURRENT_RATIO_MAX)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_no_moment_when_current_ratio_above_threshold(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    _insert_profile(conn, "alice@example.com", previous_ratio=0.8, current_ratio=0.5)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_moment_fires_just_past_both_boundaries(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    _insert_profile(
        conn,
        "alice@example.com",
        previous_ratio=PREVIOUS_RATIO_MIN + 0.01,
        current_ratio=CURRENT_RATIO_MAX - 0.01,
    )
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Idempotency per week
# ---------------------------------------------------------------------------


def test_same_week_same_evidence_hash(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    """Two observations in the same ISO week yield the same evidence_hash."""
    _insert_profile(conn, "alice@example.com")
    first = asyncio.run(producer.observe(_inbound_event(id_="trigger-1")))
    second = asyncio.run(producer.observe(_inbound_event(id_="trigger-2")))
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].evidence_hash == second[0].evidence_hash


def test_same_week_dedups_via_unique_constraint(conn: sqlite3.Connection) -> None:
    """Two producer firings → one row in moments (the dedup the task asks for)."""
    _insert_profile(conn, "alice@example.com")
    prod = RelationshipProducer(conn, now_fn=lambda: REF_NOW, id_fn=lambda: "mid-1")
    m1 = asyncio.run(prod.observe(_inbound_event()))[0]
    prod2 = RelationshipProducer(conn, now_fn=lambda: REF_NOW, id_fn=lambda: "mid-2")
    m2 = asyncio.run(prod2.observe(_inbound_event()))[0]
    repo = MomentRepository(conn, now_fn=lambda: REF_NOW)
    id_first = repo.create(m1)
    id_second = repo.create(m2)
    assert id_first == id_second  # collision returns existing id


def test_new_week_produces_new_hash(conn: sqlite3.Connection) -> None:
    """Advance 7 days (next ISO week) → fresh hash for same contact+profile."""
    _insert_profile(conn, "alice@example.com")

    prod_wk17 = RelationshipProducer(conn, now_fn=lambda: REF_NOW, id_fn=lambda: "m-wk17")
    prod_wk18 = RelationshipProducer(conn, now_fn=lambda: REF_NOW + SECONDS_PER_WEEK, id_fn=lambda: "m-wk18")
    wk17 = asyncio.run(prod_wk17.observe(_inbound_event()))[0]
    wk18 = asyncio.run(prod_wk18.observe(_inbound_event()))[0]
    assert wk17.evidence_hash != wk18.evidence_hash


def test_different_contacts_same_week_have_different_hashes(
    conn: sqlite3.Connection, producer: RelationshipProducer
) -> None:
    _insert_profile(conn, "alice@example.com")
    _insert_profile(conn, "bob@example.com", contact_name="Bob")
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 2
    hashes = {m.evidence_hash for m in out}
    assert len(hashes) == 2  # distinct per-contact


# ---------------------------------------------------------------------------
# Confidence + payload shape
# ---------------------------------------------------------------------------


def test_confidence_scales_with_drop(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    """Confidence = min(0.9, (previous - current) / previous)."""
    _insert_profile(conn, "alice@example.com", previous_ratio=0.6, current_ratio=0.2)
    out = asyncio.run(producer.observe(_inbound_event()))
    # (0.6 - 0.2) / 0.6 = 0.6666...
    assert out[0].confidence == pytest.approx((0.6 - 0.2) / 0.6)


def test_confidence_capped_at_0_9(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    """A near-total collapse saturates at the 0.9 ceiling."""
    _insert_profile(conn, "alice@example.com", previous_ratio=0.95, current_ratio=0.02)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out[0].confidence == 0.9


def test_proposed_action_is_nudge_with_contact_and_channel(
    conn: sqlite3.Connection, producer: RelationshipProducer
) -> None:
    _insert_profile(conn, "alice@example.com", channel="email")
    out = asyncio.run(producer.observe(_inbound_event()))
    action = out[0].proposed_action
    assert action.kind is ActionKind.NUDGE
    assert action.params == {"contact_id": "alice@example.com", "channel": "email"}


def test_insight_microcopy_contains_name_and_drop_percent(
    conn: sqlite3.Connection, producer: RelationshipProducer
) -> None:
    _insert_profile(
        conn,
        "alice@example.com",
        previous_ratio=0.5001,  # strictly > baseline
        current_ratio=0.1,
        contact_name="Alice",
    )
    # (0.5001 - 0.1) / 0.5001 = 0.8001... → 80%
    insight = asyncio.run(producer.observe(_inbound_event()))[0].insight
    assert "You've been replying less to Alice" in insight
    assert "Outbound dropped 80%" in insight


def test_moment_carries_source_insight_type_relationship(
    conn: sqlite3.Connection, producer: RelationshipProducer
) -> None:
    _insert_profile(conn, "alice@example.com")
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.source_insight_type is InsightType.RELATIONSHIP


def test_moment_expires_at_default_72_hours(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    _insert_profile(conn, "alice@example.com")
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.expires_at - moment.created_at == DEFAULT_EXPIRY_SECONDS


def test_moment_evidence_carries_last_three_event_ids(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    _insert_profile(
        conn,
        "alice@example.com",
        event_ids=["evt-a", "evt-b", "evt-c", "evt-d", "evt-e"],
    )
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.evidence == ["evt-a", "evt-b", "evt-c"]


# ---------------------------------------------------------------------------
# Multi-profile + filtering
# ---------------------------------------------------------------------------


def test_only_drifted_contacts_emit_moments(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    """Healthy-ratio contact is silent; drifted contact emits one Moment."""
    _insert_profile(conn, "alice@example.com", previous_ratio=0.7, current_ratio=0.2)
    _insert_profile(conn, "bob@example.com", previous_ratio=0.6, current_ratio=0.55)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 1
    assert out[0].proposed_action.params["contact_id"] == "alice@example.com"


def test_multiple_drifted_contacts_emit_one_moment_each(
    conn: sqlite3.Connection, producer: RelationshipProducer
) -> None:
    _insert_profile(conn, "alice@example.com", previous_ratio=0.7, current_ratio=0.2)
    _insert_profile(conn, "carol@example.com", previous_ratio=0.8, current_ratio=0.1, contact_name="Carol")
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 2
    contacts = {m.proposed_action.params["contact_id"] for m in out}
    assert contacts == {"alice@example.com", "carol@example.com"}


# ---------------------------------------------------------------------------
# Malformed input fails open
# ---------------------------------------------------------------------------


def test_malformed_profile_json_is_skipped(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (RELATIONSHIP_PRODUCER_KEY, "broken", "{not valid json"),
    )
    _insert_profile(conn, "alice@example.com")
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 1
    assert out[0].proposed_action.params["contact_id"] == "alice@example.com"


def test_non_object_profile_is_skipped(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (RELATIONSHIP_PRODUCER_KEY, "list-not-obj", json.dumps([1, 2, 3])),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_profile_missing_required_fields_is_skipped(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (RELATIONSHIP_PRODUCER_KEY, "incomplete", json.dumps({"previous_ratio": 0.7})),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_profile_with_empty_event_ids_is_skipped(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    _insert_profile(conn, "alice@example.com", event_ids=[])
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_profile_with_non_list_event_ids_is_skipped(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    _insert_profile(conn, "alice@example.com", extra={"last_event_ids": "not-a-list"})
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_profile_for_other_producer_is_ignored(conn: sqlite3.Connection, producer: RelationshipProducer) -> None:
    """Only ``producer='relationship'`` rows are read; cadence rows are inert."""
    _insert_profile(conn, "alice@example.com")
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            "cadence",
            "bob@example.com",
            json.dumps({"expected_cadence_days": 1.0, "count": 100}),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 1
    assert out[0].source_insight_type is InsightType.RELATIONSHIP
    assert out[0].proposed_action.params["contact_id"] == "alice@example.com"


# ---------------------------------------------------------------------------
# Registry side-effect
# ---------------------------------------------------------------------------


def test_relationship_producer_registers_under_insight_type() -> None:
    """Importing the module must populate the global PRODUCERS map."""
    from core.moment.producer import PRODUCERS

    assert PRODUCERS.get(InsightType.RELATIONSHIP) is RelationshipProducer
