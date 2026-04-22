"""Tests for :class:`ai.context.ContextAssembler`.

Focus per the Week 7 task body:

- ``assemble_briefing_context`` returns a dict whose keys match the
  11 contracted :data:`ai.context.BRIEFING_SECTIONS`.
- Empty state → every list section is ``[]`` and ``preferences`` is
  ``{}``; nothing is ``None`` and no keys are missing.
- Populated state → each section surfaces the rows it's responsible
  for, without pulling in stray data from other sections.
- Fail-open — a single-section failure (dropped table) degrades only
  that section; the other ten still return.

The fixture drives a real in-memory SQLite connection seeded from
``storage.schema.get_all_ddl()`` so the tests are schema-authoritative:
a column-name drift in the DDL fails here, not silently in production.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, date, datetime

import pytest

from ai.context import BRIEFING_SECTIONS, ContextAssembler
from storage import schema

REF_DAY = date(2026, 4, 22)
REF_NOW = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
REF_DAY_START_TS = int(datetime(2026, 4, 22, 0, 0, tzinfo=UTC).timestamp())
REF_NOW_TS = int(REF_NOW.timestamp())


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """Fresh in-memory SQLite with the full v2 schema applied."""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        c.execute(stmt)
    c.commit()
    return c


@pytest.fixture()
def assembler(conn: sqlite3.Connection) -> ContextAssembler:
    return ContextAssembler(conn, now_fn=lambda: REF_NOW)


def _insert_event(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    timestamp: int,
    payload: dict,
    source: str = "test",
) -> str:
    """Insert an event row and return its id."""
    eid = f"e-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, 'normal', ?)",
        (eid, event_type, source, timestamp, json.dumps(payload)),
    )
    return eid


def _insert_moment(
    conn: sqlite3.Connection,
    *,
    state: str = "suggested",
    insight_type: str = "cadence",
    insight: str = "test insight",
    confidence: float = 0.8,
    scheduled_for: int | None = None,
    context_trigger: str | None = None,
    action_kind: str = "draft_message",
    transitions: list[tuple[str, int]] | None = None,
) -> str:
    """Insert a Moment row (plus its history) and return its id."""
    mid = f"m-{uuid.uuid4().hex[:8]}"
    evidence_hash = uuid.uuid4().hex
    created_at = REF_DAY_START_TS
    expires_at = created_at + 72 * 3600
    conn.execute(
        """
        INSERT INTO moments (
            id, created_at, scheduled_for, expires_at, context_trigger,
            insight, evidence, evidence_hash, proposed_action, state,
            snooze_until, confidence, feedback_weight, source_insight_type,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            mid,
            created_at,
            scheduled_for,
            expires_at,
            context_trigger,
            insight,
            json.dumps([]),
            evidence_hash,
            json.dumps({"kind": action_kind, "params": {}}),
            state,
            None,
            confidence,
            1.0,
            insight_type,
            created_at,
        ),
    )
    conn.execute(
        "INSERT INTO moment_state_history (moment_id, from_state, to_state, ts, annotation) "
        "VALUES (?, NULL, 'suggested', ?, 'create')",
        (mid, created_at),
    )
    for from_state, to_state, ts in [(h[0], h[1], h[2]) for h in (transitions or [])]:
        conn.execute(
            "INSERT INTO moment_state_history (moment_id, from_state, to_state, ts, annotation) "
            "VALUES (?, ?, ?, ?, NULL)",
            (mid, from_state, to_state, ts),
        )
    return mid


# ---------------------------------------------------------------------------
# dict shape + empty-state
# ---------------------------------------------------------------------------


def test_returns_all_11_keys_on_empty_db(assembler: ContextAssembler) -> None:
    """Every contracted key is present and defaulted — no None, no KeyError."""
    out = assembler.assemble_briefing_context("user-1", REF_DAY)
    assert set(out.keys()) == set(BRIEFING_SECTIONS)
    assert len(BRIEFING_SECTIONS) == 11


def test_empty_state_returns_empty_containers(assembler: ContextAssembler) -> None:
    """List sections → ``[]``; preferences → ``{}``; never None."""
    out = assembler.assemble_briefing_context("user-1", REF_DAY)
    list_keys = [k for k in BRIEFING_SECTIONS if k != "preferences"]
    for k in list_keys:
        assert out[k] == [], f"{k!r} should be [] when empty, got {out[k]!r}"
    assert out["preferences"] == {}
    for v in out.values():
        assert v is not None


# ---------------------------------------------------------------------------
# populated sections
# ---------------------------------------------------------------------------


def test_calendar_surfaces_upcoming_events(conn, assembler) -> None:
    eid = _insert_event(
        conn,
        event_type="calendar.event.created",
        timestamp=REF_DAY_START_TS + 3600,
        payload={
            "title": "Standup",
            "start_time": "2026-04-22T09:00:00Z",
            "end_time": "2026-04-22T09:30:00Z",
            "location": "Zoom",
        },
    )
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert len(out["calendar"]) == 1
    assert out["calendar"][0]["id"] == eid
    assert out["calendar"][0]["title"] == "Standup"
    assert out["calendar"][0]["is_all_day"] is False


def test_calendar_filters_non_calendar_events(conn, assembler) -> None:
    _insert_event(
        conn,
        event_type="email.received",
        timestamp=REF_DAY_START_TS + 3600,
        payload={"subject": "Hi"},
    )
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert out["calendar"] == []


def test_moments_surface_suggested_only(conn, assembler) -> None:
    sug = _insert_moment(conn, state="suggested", insight="suggested one")
    _insert_moment(conn, state="dismissed", insight="dismissed one")
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert len(out["moments"]) == 1
    m = out["moments"][0]
    assert m["id"] == sug
    assert m["insight"] == "suggested one"
    assert m["source_insight_type"] == "cadence"
    assert m["action_kind"] == "draft_message"


def test_moments_hides_legacy_task_rows(conn, assembler) -> None:
    _insert_moment(conn, state="suggested", insight_type="legacy_task", insight="legacy")
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert out["moments"] == []


def test_unread_messages_only_from_last_12h(conn, assembler) -> None:
    _insert_event(
        conn,
        event_type="email.received",
        timestamp=REF_NOW_TS - 3600,
        payload={"from_address": "a@x", "subject": "recent"},
    )
    _insert_event(
        conn,
        event_type="email.received",
        timestamp=REF_NOW_TS - 13 * 3600,
        payload={"from_address": "a@x", "subject": "stale"},
    )
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert [m["subject"] for m in out["unread_messages"]] == ["recent"]


def test_completions_read_from_history_not_updated_at(conn, assembler) -> None:
    mid = _insert_moment(
        conn,
        state="done",
        insight="finished",
        transitions=[
            ("suggested", "accepted", REF_DAY_START_TS + 3600),
            ("accepted", "done", REF_DAY_START_TS + 7200),
        ],
    )
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert len(out["completions"]) == 1
    assert out["completions"][0]["id"] == mid


def test_completions_ignores_other_days(conn, assembler) -> None:
    yesterday = REF_DAY_START_TS - 3600
    _insert_moment(
        conn,
        state="done",
        insight="yesterday",
        transitions=[
            ("suggested", "accepted", yesterday),
            ("accepted", "done", yesterday),
        ],
    )
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert out["completions"] == []


def test_predictions_include_scheduled_and_context_triggered(conn, assembler) -> None:
    # scheduled Moment in range
    _insert_moment(
        conn,
        state="suggested",
        insight="sched",
        scheduled_for=REF_NOW_TS + 3600,
    )
    # context-triggered Moment (no scheduled_for but always eligible)
    _insert_moment(
        conn,
        state="suggested",
        insight="ctx",
        context_trigger="arrive:home",
    )
    # plain suggested with neither — should NOT appear
    _insert_moment(conn, state="suggested", insight="plain")
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    insights = sorted(p["insight"] for p in out["predictions"])
    assert insights == ["ctx", "sched"]


def test_episodes_surface_recent_events(conn, assembler) -> None:
    eid = _insert_event(
        conn,
        event_type="message.received",
        timestamp=REF_NOW_TS - 1800,
        payload={"from_address": "b@y", "subject": "hey"},
    )
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert any(e["id"] == eid for e in out["episodes"])


def test_episodes_drops_rows_older_than_24h(conn, assembler) -> None:
    _insert_event(
        conn,
        event_type="message.received",
        timestamp=REF_NOW_TS - 25 * 3600,
        payload={"subject": "too old"},
    )
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert out["episodes"] == []


def test_facts_only_confirmed(conn, assembler) -> None:
    for status in ("pending", "confirmed", "denied"):
        conn.execute(
            "INSERT INTO semantic_facts (id, subject, predicate, object, confidence, status) "
            "VALUES (?, 'user', 'likes', ?, 0.9, ?)",
            (f"f-{status}", status, status),
        )
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert [f["object"] for f in out["facts"]] == ["confirmed"]


def test_insights_emit_rows_with_decisions(conn, assembler) -> None:
    conn.execute("INSERT INTO feedback_weights (insight_type, weight, decision_count) VALUES ('cadence', 0.75, 5)")
    conn.execute("INSERT INTO feedback_weights (insight_type, weight, decision_count) VALUES ('temporal', 1.0, 0)")
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert len(out["insights"]) == 1
    assert out["insights"][0]["insight_type"] == "cadence"
    assert out["insights"][0]["decision_count"] == 5


def test_routines_and_habits_split_by_producer(conn, assembler) -> None:
    for producer, key, profile in [
        ("routine", "morning", {"steps": ["email", "coffee"]}),
        ("cadence", "alice@x", {"reply_hours": 4.2}),
        ("temporal", "chronotype", {"peak_hour": 10}),
    ]:
        conn.execute(
            "INSERT INTO signal_profiles (producer, key, profile) VALUES (?, ?, ?)",
            (producer, key, json.dumps(profile)),
        )
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert len(out["routines"]) == 1
    assert out["routines"][0]["producer"] == "routine"
    # habits aggregates cadence + temporal
    assert len(out["habits"]) == 2
    producers = {h["producer"] for h in out["habits"]}
    assert producers == {"cadence", "temporal"}


def test_routines_skips_malformed_json(conn, assembler) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES ('routine', 'bad', ?)",
        ("not-json",),
    )
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile) VALUES ('routine', 'good', ?)",
        (json.dumps({"ok": True}),),
    )
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert [r["key"] for r in out["routines"]] == ["good"]


def test_preferences_dict_excludes_encrypted(conn, assembler) -> None:
    conn.execute("INSERT INTO preferences (key, value, encrypted) VALUES ('theme', 'dark', 0)")
    conn.execute("INSERT INTO preferences (key, value, encrypted) VALUES ('cred', 'ciphertext', 1)")
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert out["preferences"] == {"theme": "dark"}


# ---------------------------------------------------------------------------
# fail-open
# ---------------------------------------------------------------------------


def test_single_section_failure_does_not_kill_others(conn, assembler) -> None:
    """Dropping a single table must only blank that one section."""
    conn.execute("INSERT INTO preferences (key, value, encrypted) VALUES ('tone', 'warm', 0)")
    conn.execute("DROP TABLE semantic_facts")
    conn.commit()
    out = assembler.assemble_briefing_context("u", REF_DAY)
    assert out["facts"] == []  # degraded
    assert out["preferences"] == {"tone": "warm"}  # unaffected
    assert set(out.keys()) == set(BRIEFING_SECTIONS)
