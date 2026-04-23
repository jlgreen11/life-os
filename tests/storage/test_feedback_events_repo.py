"""Tests for `storage/repos/feedback_events.py`.

Covers:

- Happy-path append + hydrate round-trip.
- Injected ``now_fn`` stamps ``created_at`` deterministically.
- Append inside an externally-owned transaction (``conn=``).
- CHECK and PRIMARY KEY violations on invalid inputs.
- ``count`` / ``count_by_action_and_type`` / ``list_by_action`` / ``recent``
  semantics and ordering.
"""

from __future__ import annotations

import sqlite3

import pytest

# Bootstrap the core.moment package first to avoid a pre-existing circular
# import between `storage.repos.moments` and `core.moment.engine` that
# surfaces only when `storage.repos.__init__` runs before `core.moment`.
import core.moment.state  # noqa: F401
from storage import schema
from storage.repos.feedback_events import FeedbackEvent, FeedbackEventsRepository


@pytest.fixture
def fresh_db(tmp_path):
    """A fresh file-backed SQLite DB with the v2 schema applied, FKs on."""
    db_path = tmp_path / "lifeos.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        conn.execute(stmt)
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def repo(fresh_db):
    """Repo backed by ``fresh_db`` with a frozen ``now_fn`` for determinism."""
    return FeedbackEventsRepository(fresh_db, now_fn=lambda: 1_700_000_000)


def test_append_then_recent_round_trip(repo):
    repo.append(
        id="fb-1",
        ts=1_699_000_000,
        action_id="act-1",
        action_type="notification",
        feedback_type="acted_on",
        response_latency_seconds=1.25,
        context='{"domain":"email"}',
        notes="quick ack",
    )

    rows = repo.recent(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, FeedbackEvent)
    assert row.id == "fb-1"
    assert row.ts == 1_699_000_000
    assert row.action_id == "act-1"
    assert row.action_type == "notification"
    assert row.feedback_type == "acted_on"
    assert row.response_latency_seconds == 1.25
    assert row.context == '{"domain":"email"}'
    assert row.notes == "quick ack"
    assert row.source == "v2"
    assert row.created_at == 1_700_000_000  # frozen now_fn


def test_append_defaults_to_v2_source(repo):
    repo.append(id="fb-a", ts=1_699_000_000)
    repo.append(id="fb-b", ts=1_699_000_001, source="v1_migration")

    by_source = {r.id: r.source for r in repo.recent(limit=10)}
    assert by_source == {"fb-a": "v2", "fb-b": "v1_migration"}


def test_append_rejects_invalid_source(repo):
    with pytest.raises(sqlite3.IntegrityError):
        repo.append(id="fb-bad", ts=1, source="not-a-source")


def test_append_is_rejected_on_duplicate_id(repo):
    repo.append(id="fb-dup", ts=1_699_000_000)
    with pytest.raises(sqlite3.IntegrityError):
        repo.append(id="fb-dup", ts=1_699_000_001)


def test_append_with_external_connection_honors_caller_rollback(fresh_db):
    """``conn=`` piggybacks on the caller's transaction — rollback undoes it."""
    repo = FeedbackEventsRepository(fresh_db, now_fn=lambda: 1_700_000_000)

    # Caller owns the transaction; repo must not COMMIT on its own, so a
    # caller-side ROLLBACK undoes the INSERT.
    fresh_db.execute("BEGIN IMMEDIATE")
    repo.append(id="fb-ext", ts=1_699_000_000, conn=fresh_db)
    fresh_db.execute("ROLLBACK")

    assert repo.count() == 0

    # And a COMMIT makes it stick.
    fresh_db.execute("BEGIN IMMEDIATE")
    repo.append(id="fb-ext", ts=1_699_000_000, conn=fresh_db)
    fresh_db.execute("COMMIT")
    assert repo.count() == 1


def test_count_and_count_by_action_and_type(repo):
    repo.append(id="fb-a", ts=1, action_id="act-1", feedback_type="dismissed")
    repo.append(id="fb-b", ts=2, action_id="act-1", feedback_type="dismissed")
    repo.append(id="fb-c", ts=3, action_id="act-1", feedback_type="acted_on")
    repo.append(id="fb-d", ts=4, action_id="act-2", feedback_type="dismissed")

    assert repo.count() == 4
    assert repo.count_by_action_and_type("act-1", "dismissed") == 2
    assert repo.count_by_action_and_type("act-1", "acted_on") == 1
    assert repo.count_by_action_and_type("act-2", "dismissed") == 1
    assert repo.count_by_action_and_type("act-unknown", "dismissed") == 0


def test_list_by_action_orders_desc_ts_then_asc_id(repo):
    # Two rows share ts=5 so the id-ASC tiebreaker is exercised.
    repo.append(id="fb-older", ts=1, action_id="act-1")
    repo.append(id="fb-b-tie", ts=5, action_id="act-1")
    repo.append(id="fb-a-tie", ts=5, action_id="act-1")
    repo.append(id="fb-other", ts=9, action_id="act-2")  # filtered out

    rows = repo.list_by_action("act-1")
    assert [r.id for r in rows] == ["fb-a-tie", "fb-b-tie", "fb-older"]

    # Limit honored.
    assert [r.id for r in repo.list_by_action("act-1", limit=1)] == ["fb-a-tie"]
    assert repo.list_by_action("act-1", limit=0) == []


def test_list_by_action_rejects_negative_limit(repo):
    with pytest.raises(ValueError):
        repo.list_by_action("act-1", limit=-1)


def test_recent_rejects_negative_limit(repo):
    with pytest.raises(ValueError):
        repo.recent(limit=-1)


def test_recent_respects_limit_and_ordering(repo):
    for i in range(5):
        repo.append(id=f"fb-{i}", ts=100 + i)

    ids_desc = [r.id for r in repo.recent(limit=3)]
    assert ids_desc == ["fb-4", "fb-3", "fb-2"]
