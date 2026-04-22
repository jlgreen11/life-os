"""Tests for :class:`producers.comm_template.CommTemplateProducer`.

Covers the two behaviors called out in the Week 5 task body:

- **Producer returns stub for known contact** — fires on inbound
  ``email.received`` / ``message.received`` events whose
  ``payload.contact_id`` matches a row in ``signal_profiles`` keyed
  under ``producer='comm_template'``, emitting a Moment whose
  proposed action carries the deterministic ``"Hi {name},"`` draft.
- **Empty for unknown** — an inbound from a contact with no
  comm-template profile silently returns ``[]``.

Plus surrounding correctness:

- Trigger filter — non-inbound event types short-circuit without
  touching the DB.
- Payload gating — missing/non-dict payload, missing/empty
  ``contact_id``, and missing/empty event ``id`` all return ``[]``.
- Stub semantics — the draft is exactly ``"Hi {contact_name},"``;
  ``contact_name`` falls back to the contact id when the profile
  field is absent.
- Action shape — :class:`ActionKind.DRAFT_MESSAGE` with
  ``contact_id``, ``channel``, ``draft``, and
  ``in_reply_to_event_id`` on the params.
- Confidence — fixed at :data:`STUB_CONFIDENCE` (0.5) until Week 7
  AI engine integration replaces it.
- Evidence — uses ``profile.last_event_ids`` if present (top 3),
  else the trigger event id.
- Idempotency per inbound — the same (event_id, contact_id) pair
  collapses via the ``UNIQUE (source_insight_type, evidence_hash)``
  constraint; a new event_id for the same contact produces a fresh
  hash.
- Malformed input fails open — invalid JSON, non-object profile,
  blank contact_name resolution.

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
from producers.comm_template import (
    COMM_TEMPLATE_PRODUCER_KEY,
    DEFAULT_EXPIRY_SECONDS,
    STUB_CONFIDENCE,
    CommTemplateProducer,
)
from storage import schema
from storage.repos.moments import MomentRepository

REF_NOW = 1_777_204_800  # 2026-04-26T12:00:00Z


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
def producer(conn: sqlite3.Connection) -> CommTemplateProducer:
    """A producer pinned to ``REF_NOW`` and a deterministic id stream."""
    counter = {"n": 0}

    def _id() -> str:
        counter["n"] += 1
        return f"moment-{counter['n']:04d}"

    return CommTemplateProducer(conn, now_fn=lambda: REF_NOW, id_fn=_id)


def _insert_profile(
    conn: sqlite3.Connection,
    *,
    key: str = "contact-alice",
    contact_name: str | None = "Alice",
    channel: str | None = "email",
    template_style: str | None = "casual",
    last_event_ids: list[str] | None = None,
) -> dict:
    """Insert a comm-template profile.

    Defaults: an Alice profile with channel/style/three event ids —
    the happy path. Pass ``None`` to drop a field.
    """
    profile: dict = {}
    if contact_name is not None:
        profile["contact_name"] = contact_name
    if channel is not None:
        profile["channel"] = channel
    if template_style is not None:
        profile["template_style"] = template_style
    profile["last_event_ids"] = (
        last_event_ids if last_event_ids is not None else ["evt-prior-1", "evt-prior-2", "evt-prior-3"]
    )
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (COMM_TEMPLATE_PRODUCER_KEY, key, json.dumps(profile)),
    )
    conn.commit()
    return profile


def _inbound_event(
    contact_id: str = "contact-alice",
    event_id: str = "evt-inbound-1",
    type_: str = "email.received",
) -> dict:
    return {
        "id": event_id,
        "type": type_,
        "timestamp": REF_NOW,
        "source": "proton_mail",
        "payload": {"contact_id": contact_id, "subject": "hi", "body": "ping"},
    }


# ---------------------------------------------------------------------------
# Trigger filter
# ---------------------------------------------------------------------------


def test_non_trigger_event_returns_empty(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe(_inbound_event(type_="context.location.updated")))
    assert out == []


def test_message_received_also_triggers(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe(_inbound_event(type_="message.received")))
    assert len(out) == 1


def test_missing_payload_returns_empty(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": "email.received"}))
    assert out == []


def test_non_dict_payload_returns_empty(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": "email.received", "payload": "nope"}))
    assert out == []


def test_missing_contact_id_returns_empty(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": "email.received", "payload": {"subject": "hi"}}))
    assert out == []


def test_blank_contact_id_returns_empty(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": "email.received", "payload": {"contact_id": ""}}))
    assert out == []


def test_non_string_contact_id_returns_empty(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"id": "x", "type": "email.received", "payload": {"contact_id": 42}}))
    assert out == []


def test_missing_event_id_returns_empty(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe({"type": "email.received", "payload": {"contact_id": "contact-alice"}}))
    assert out == []


# ---------------------------------------------------------------------------
# Known contact: stub draft happy path
# ---------------------------------------------------------------------------


def test_known_contact_emits_one_moment(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    out = asyncio.run(producer.observe(_inbound_event()))
    assert len(out) == 1


def test_stub_draft_uses_contact_name(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn, contact_name="Alice")
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.proposed_action.params["draft"] == "Hi Alice,"


def test_stub_draft_falls_back_to_contact_id_when_name_missing(
    conn: sqlite3.Connection, producer: CommTemplateProducer
) -> None:
    _insert_profile(conn, contact_name=None)
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    # contact_id is "contact-alice" — used as the greeting fallback.
    assert moment.proposed_action.params["draft"] == "Hi contact-alice,"


def test_microcopy_uses_contact_name(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn, contact_name="Alice")
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.insight == "Reply to Alice? Draft ready."


# ---------------------------------------------------------------------------
# Unknown contact: empty
# ---------------------------------------------------------------------------


def test_unknown_contact_returns_empty(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn, key="contact-alice")
    out = asyncio.run(producer.observe(_inbound_event(contact_id="contact-unknown")))
    assert out == []


def test_no_profiles_at_all_returns_empty(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


# ---------------------------------------------------------------------------
# Action shape
# ---------------------------------------------------------------------------


def test_action_kind_is_draft_message(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.proposed_action.kind is ActionKind.DRAFT_MESSAGE


def test_action_params_carry_contact_channel_draft_and_reply_pointer(
    conn: sqlite3.Connection, producer: CommTemplateProducer
) -> None:
    _insert_profile(conn, channel="imessage")
    moment = asyncio.run(producer.observe(_inbound_event(event_id="evt-inbound-99")))[0]
    params = moment.proposed_action.params
    assert params["contact_id"] == "contact-alice"
    assert params["channel"] == "imessage"
    assert params["draft"] == "Hi Alice,"
    assert params["in_reply_to_event_id"] == "evt-inbound-99"


def test_action_params_channel_defaults_to_empty_string(
    conn: sqlite3.Connection, producer: CommTemplateProducer
) -> None:
    _insert_profile(conn, channel=None)
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.proposed_action.params["channel"] == ""


# ---------------------------------------------------------------------------
# Confidence + envelope fields
# ---------------------------------------------------------------------------


def test_confidence_is_fixed_stub(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.confidence == STUB_CONFIDENCE == 0.5


def test_source_insight_type_is_comm_template(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.source_insight_type is InsightType.COMM_TEMPLATE


def test_moment_expires_at_default_72_hours(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn)
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.expires_at - moment.created_at == DEFAULT_EXPIRY_SECONDS


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_evidence_uses_profile_event_ids(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn, last_event_ids=["e1", "e2", "e3", "e4"])
    moment = asyncio.run(producer.observe(_inbound_event()))[0]
    assert moment.evidence == ["e1", "e2", "e3"]


def test_evidence_falls_back_to_trigger_event_id(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    _insert_profile(conn, last_event_ids=[])
    moment = asyncio.run(producer.observe(_inbound_event(event_id="evt-trigger-only")))[0]
    assert moment.evidence == ["evt-trigger-only"]


def test_evidence_falls_back_when_field_missing(conn: sqlite3.Connection) -> None:
    """A profile without ``last_event_ids`` still yields a Moment."""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        c.execute(stmt)
    c.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            COMM_TEMPLATE_PRODUCER_KEY,
            "contact-alice",
            json.dumps({"contact_name": "Alice", "channel": "email"}),
        ),
    )
    c.commit()
    p = CommTemplateProducer(c, now_fn=lambda: REF_NOW, id_fn=lambda: "moment-x")
    moment = asyncio.run(p.observe(_inbound_event(event_id="evt-only")))[0]
    assert moment.evidence == ["evt-only"]
    c.close()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_same_event_and_contact_collapse_in_repository(
    conn: sqlite3.Connection, producer: CommTemplateProducer
) -> None:
    """Same inbound replayed for same contact → one row via UNIQUE constraint."""
    _insert_profile(conn)
    first = asyncio.run(producer.observe(_inbound_event(event_id="evt-1")))
    second = asyncio.run(producer.observe(_inbound_event(event_id="evt-1")))
    assert first[0].evidence_hash == second[0].evidence_hash

    repo = MomentRepository(conn, now_fn=lambda: REF_NOW)
    id_first = repo.create(first[0])
    id_second = repo.create(second[0])
    assert id_first == id_second  # UNIQUE collision returns existing id


def test_distinct_inbound_events_produce_distinct_hashes(
    conn: sqlite3.Connection, producer: CommTemplateProducer
) -> None:
    _insert_profile(conn)
    a = asyncio.run(producer.observe(_inbound_event(event_id="evt-1")))[0]
    b = asyncio.run(producer.observe(_inbound_event(event_id="evt-2")))[0]
    assert a.evidence_hash != b.evidence_hash


def test_distinct_contacts_same_event_id_produce_distinct_hashes(
    conn: sqlite3.Connection, producer: CommTemplateProducer
) -> None:
    """A theoretical event id collision across contacts must not alias."""
    _insert_profile(conn, key="contact-alice", contact_name="Alice")
    _insert_profile(conn, key="contact-bob", contact_name="Bob")
    a = asyncio.run(producer.observe(_inbound_event(contact_id="contact-alice", event_id="evt-shared")))[0]
    b = asyncio.run(producer.observe(_inbound_event(contact_id="contact-bob", event_id="evt-shared")))[0]
    assert a.evidence_hash != b.evidence_hash


# ---------------------------------------------------------------------------
# Malformed input fails open
# ---------------------------------------------------------------------------


def test_malformed_profile_json_is_skipped(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (COMM_TEMPLATE_PRODUCER_KEY, "contact-alice", "{not valid json"),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_non_object_profile_is_skipped(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (COMM_TEMPLATE_PRODUCER_KEY, "contact-alice", json.dumps([1, 2, 3])),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


def test_blank_contact_name_with_blank_id_fallback_skips(
    conn: sqlite3.Connection, producer: CommTemplateProducer
) -> None:
    """Whitespace-only ``contact_name`` and an unusable id fallback skip."""
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            COMM_TEMPLATE_PRODUCER_KEY,
            "   ",  # key strips to empty
            json.dumps({"contact_name": "   ", "channel": "email"}),
        ),
    )
    conn.commit()
    out = asyncio.run(
        producer.observe(
            {
                "id": "evt-1",
                "type": "email.received",
                "payload": {"contact_id": "   "},  # blank ASCII space passes the empty-string gate? no — non-empty
            }
        )
    )
    # contact_id="   " is non-empty so it passes the trigger gate, but
    # no profile keyed exactly "   " (whitespace) is read for that
    # exact id either; so the unknown-contact path returns [].
    # If a future change keys profiles by trimmed strings this test
    # will need updating — that is the intended trip-wire.
    assert out == []


def test_profile_for_other_producer_is_ignored(conn: sqlite3.Connection, producer: CommTemplateProducer) -> None:
    """A cadence row keyed by the same contact must not leak into comm_template."""
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
        (
            "cadence",
            "contact-alice",
            json.dumps({"expected_cadence_days": 1.0, "count": 100}),
        ),
    )
    conn.commit()
    out = asyncio.run(producer.observe(_inbound_event()))
    assert out == []


# ---------------------------------------------------------------------------
# Registry side-effect
# ---------------------------------------------------------------------------


def test_comm_template_producer_registers_under_insight_type() -> None:
    """Importing the module must populate the global PRODUCERS map."""
    from core.moment.producer import PRODUCERS

    assert PRODUCERS.get(InsightType.COMM_TEMPLATE) is CommTemplateProducer
