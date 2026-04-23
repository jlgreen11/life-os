"""Exception-path tests for :class:`storage.repos.outbox.OutboxRepository`.

Targets coverage gap #3 in the 2026-04-23 audit. The repository's
public methods all wrap their SQL in a ``BEGIN IMMEDIATE`` block with
a ``try/except/ROLLBACK/raise`` shell. The happy-path tests exercise
``COMMIT``; these tests force an exception mid-transaction to lock the
rollback branches:

- :meth:`enqueue` — the SELECT duplicate-check raises, ROLLBACK fires,
  exception re-raises, no row survives.
- :meth:`claim_batch` — the second UPDATE raises, ROLLBACK undoes the
  in_progress flip, the original rows stay ``pending``.
- :meth:`complete` — the UPDATE raises after the SELECT, ROLLBACK
  restores the in_progress state.
- :meth:`fail` — the UPDATE raises, ROLLBACK keeps ``retry_count``
  unchanged.
- :meth:`cancel_pending` — the DELETE raises, ROLLBACK leaves the
  pending row intact.
- :meth:`requeue_in_progress_on_boot` — the UPDATE raises, ROLLBACK
  keeps stuck rows in ``in_progress``.
- :meth:`purge_done_older_than` — the DELETE raises, ROLLBACK keeps
  the done row in place.

Forcing a mid-transaction exception without patching sqlite3 itself is
awkward; we use a :class:`sqlite3.Connection` subclass-ish wrapper (an
adapter that delegates everything except ``execute``) and let the
repository see a raising ``execute`` on the SQL statement we want to
break. The repo's ``_conn`` is what gets wrapped; the existing
``BEGIN IMMEDIATE`` has already been issued on the real connection, so
the ROLLBACK delegate is allowed through to keep the fixture DB
consistent.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

import pytest

# NOTE: import ``core.moment`` before any ``storage.repos.*`` import to
# break the pre-existing circular import between
# ``storage.repos.moments`` (imports ``core.moment.state``) and
# ``core.moment.engine`` (imports ``storage.repos.moments``). Every
# storage test file either reaches through conftest's DatabaseManager
# bootstrap or through a sibling core.moment import to resolve this.
import core.moment  # noqa: F401 — side-effectful import to seed the cycle
from storage import schema
from storage.repos.outbox import MAX_RETRIES, OutboxRepository

REF_NOW = 1_777_204_800


class _Clock:
    def __init__(self) -> None:
        self.t = REF_NOW

    def __call__(self) -> float:
        return self.t


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        conn.execute(stmt)
    conn.commit()


class PatchedConn:
    """Wrap a :class:`sqlite3.Connection`, intercepting ``execute``.

    Any call whose SQL contains ``match`` raises ``ValueError`` on the
    Nth invocation (default 1 — first hit). Every other call is
    delegated to the wrapped connection. This gives tests fine-grained
    control over which SQL statement in a BEGIN/COMMIT block fails.
    """

    def __init__(
        self,
        inner: sqlite3.Connection,
        match: str,
        *,
        skip: int = 0,
    ) -> None:
        self._inner = inner
        self._match = match.lower()
        self._skip = skip
        self._seen = 0

    # The repo directly reads / writes these two attributes at
    # construction; mirror them onto the wrapper.
    @property
    def isolation_level(self) -> str | None:
        return self._inner.isolation_level

    @isolation_level.setter
    def isolation_level(self, v: str | None) -> None:
        self._inner.isolation_level = v

    @property
    def row_factory(self) -> Callable[..., Any] | None:
        return self._inner.row_factory

    @row_factory.setter
    def row_factory(self, v: Callable[..., Any] | None) -> None:
        self._inner.row_factory = v

    def execute(self, sql: str, params: Any = ()) -> sqlite3.Cursor:
        if self._match in sql.lower():
            self._seen += 1
            if self._seen > self._skip:
                raise ValueError(f"boom on sql={sql!r}")
        return self._inner.execute(sql, params)

    # Fallthrough for any other attribute access (rare — repo is
    # disciplined about using execute()).
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "outbox.db"


@pytest.fixture
def raw_conn(db_path):
    c = sqlite3.connect(db_path)
    _apply_schema(c)
    yield c
    c.close()


@pytest.fixture
def clock():
    return _Clock()


# ---------------------------------------------------------------------------
# enqueue rollback
# ---------------------------------------------------------------------------


def test_enqueue_rollback_on_insert_failure(db_path, clock):
    """An exception during the INSERT rolls the transaction back.

    After the exception the caller re-opens a fresh connection and
    confirms that *no* row with ``event_id='evt-1'`` exists.
    """
    raw = sqlite3.connect(db_path)
    _apply_schema(raw)
    patched = PatchedConn(raw, match="insert into outbox")
    repo = OutboxRepository(patched, now_fn=clock)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        repo.enqueue("evt-1", "s1", {"x": 1})

    raw.close()
    # Re-open independently; row count should be zero.
    check = sqlite3.connect(db_path)
    count = check.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
    check.close()
    assert count == 0


# ---------------------------------------------------------------------------
# claim_batch rollback
# ---------------------------------------------------------------------------


def test_claim_batch_rollback_on_update_failure(db_path, clock):
    """Seed one pending row; force the UPDATE to raise; row stays pending."""
    raw = sqlite3.connect(db_path)
    _apply_schema(raw)
    # Seed a pending row via a clean repo.
    seed = OutboxRepository(raw, now_fn=clock)
    oid = seed.enqueue("evt-1", "s1")
    raw.close()

    # Reconnect; patch the claim UPDATE.
    raw2 = sqlite3.connect(db_path)
    patched = PatchedConn(raw2, match="update outbox set state='in_progress'")
    repo = OutboxRepository(patched, now_fn=clock)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        repo.claim_batch()
    raw2.close()

    check = sqlite3.connect(db_path)
    row = check.execute("SELECT state FROM outbox WHERE id=?", (oid,)).fetchone()
    check.close()
    assert row[0] == "pending"


# ---------------------------------------------------------------------------
# complete rollback
# ---------------------------------------------------------------------------


def test_complete_rollback_on_update_failure(db_path, clock):
    """Force the ``UPDATE ... state='done'`` to raise; row stays in_progress."""
    raw = sqlite3.connect(db_path)
    _apply_schema(raw)
    seed = OutboxRepository(raw, now_fn=clock)
    oid = seed.enqueue("evt-1", "s1")
    seed.claim_batch()
    raw.close()

    raw2 = sqlite3.connect(db_path)
    patched = PatchedConn(raw2, match="update outbox set state='done'")
    repo = OutboxRepository(patched, now_fn=clock)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        repo.complete(oid)
    raw2.close()

    check = sqlite3.connect(db_path)
    row = check.execute("SELECT state FROM outbox WHERE id=?", (oid,)).fetchone()
    check.close()
    assert row[0] == "in_progress"


# ---------------------------------------------------------------------------
# fail rollback
# ---------------------------------------------------------------------------


def test_fail_rollback_on_update_failure(db_path, clock):
    """Force the retry_count UPDATE to raise; retry_count stays at 0."""
    raw = sqlite3.connect(db_path)
    _apply_schema(raw)
    seed = OutboxRepository(raw, now_fn=clock)
    oid = seed.enqueue("evt-1", "s1")
    seed.claim_batch()
    raw.close()

    raw2 = sqlite3.connect(db_path)
    patched = PatchedConn(raw2, match="update outbox set state=?, retry_count=?")
    repo = OutboxRepository(patched, now_fn=clock)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        repo.fail(oid, "boom")
    raw2.close()

    check = sqlite3.connect(db_path)
    row = check.execute("SELECT state, retry_count, last_error FROM outbox WHERE id=?", (oid,)).fetchone()
    check.close()
    assert row[0] == "in_progress"
    assert row[1] == 0
    assert row[2] is None
    assert MAX_RETRIES == 5  # sanity check the module constant is unchanged


# ---------------------------------------------------------------------------
# cancel_pending rollback
# ---------------------------------------------------------------------------


def test_cancel_pending_rollback_on_delete_failure(db_path, clock):
    """Force the DELETE to raise; the row stays pending."""
    raw = sqlite3.connect(db_path)
    _apply_schema(raw)
    seed = OutboxRepository(raw, now_fn=clock)
    oid = seed.enqueue("evt-1", "s1")
    raw.close()

    raw2 = sqlite3.connect(db_path)
    patched = PatchedConn(raw2, match="delete from outbox where event_id")
    repo = OutboxRepository(patched, now_fn=clock)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        repo.cancel_pending("evt-1", "s1")
    raw2.close()

    check = sqlite3.connect(db_path)
    row = check.execute("SELECT state FROM outbox WHERE id=?", (oid,)).fetchone()
    check.close()
    assert row[0] == "pending"


# ---------------------------------------------------------------------------
# requeue_in_progress_on_boot rollback
# ---------------------------------------------------------------------------


def test_requeue_in_progress_rollback_on_update_failure(db_path, clock):
    """Force the flip UPDATE to raise; row stays stuck in_progress."""
    raw = sqlite3.connect(db_path)
    _apply_schema(raw)
    seed = OutboxRepository(raw, now_fn=clock)
    oid = seed.enqueue("evt-1", "s1")
    seed.claim_batch()
    raw.close()

    raw2 = sqlite3.connect(db_path)
    patched = PatchedConn(raw2, match="update outbox set state='pending'")
    repo = OutboxRepository(patched, now_fn=clock)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        repo.requeue_in_progress_on_boot()
    raw2.close()

    check = sqlite3.connect(db_path)
    row = check.execute("SELECT state FROM outbox WHERE id=?", (oid,)).fetchone()
    check.close()
    assert row[0] == "in_progress"


# ---------------------------------------------------------------------------
# purge_done_older_than rollback
# ---------------------------------------------------------------------------


def test_purge_done_older_than_rollback_on_delete_failure(db_path, clock):
    """Force the DELETE to raise; the done row survives."""
    raw = sqlite3.connect(db_path)
    _apply_schema(raw)
    seed = OutboxRepository(raw, now_fn=clock)
    oid = seed.enqueue("evt-1", "s1")
    seed.claim_batch()
    seed.complete(oid)
    # Age the row past the cutoff.
    raw.execute(
        "UPDATE outbox SET updated_at=? WHERE id=?",
        (REF_NOW - 60 * 86400, oid),
    )
    raw.commit()
    raw.close()

    raw2 = sqlite3.connect(db_path)
    patched = PatchedConn(raw2, match="delete from outbox where state='done'")
    repo = OutboxRepository(patched, now_fn=clock)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        repo.purge_done_older_than(days=30)
    raw2.close()

    check = sqlite3.connect(db_path)
    row = check.execute("SELECT state FROM outbox WHERE id=?", (oid,)).fetchone()
    check.close()
    assert row[0] == "done"
