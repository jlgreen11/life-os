"""Tests for ``scripts/daily_integrity_check.py``.

Exercises the three exit-code paths (ok / integrity_failed / script_error)
plus outbox-row insertion on failure. The script is loaded via
``importlib`` to match the existing ``tests/scripts/`` convention.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from core.integrity import IntegrityReport
from storage import schema


def _load_module():
    """Load the daily integrity script as an importable module."""
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "daily_integrity_check.py"
    spec = importlib.util.spec_from_file_location("daily_integrity_check", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["daily_integrity_check"] = mod
    spec.loader.exec_module(mod)
    return mod


dic = _load_module()


@pytest.fixture
def fresh_db(tmp_path):
    """A fresh v2-schema SQLite file."""
    db_path = tmp_path / "lifeos.db"
    conn = sqlite3.connect(db_path)
    try:
        for stmt in schema.get_all_ddl():
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_run_on_healthy_db_returns_exit_ok(fresh_db, capsys):
    exit_code = dic.run(str(fresh_db))
    assert exit_code == dic.EXIT_OK
    out = capsys.readouterr().out
    assert "ok:" in out
    assert str(fresh_db) in out


def test_run_on_missing_db_returns_exit_script_error(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.db"
    exit_code = dic.run(str(missing))
    assert exit_code == dic.EXIT_SCRIPT_ERROR
    err = capsys.readouterr().err
    assert "not found" in err


def test_run_on_healthy_db_does_not_enqueue(fresh_db):
    dic.run(str(fresh_db))
    conn = sqlite3.connect(fresh_db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_run_on_integrity_failure_enqueues_alert(fresh_db, capsys, monkeypatch):
    """Failed report → exit 1 + one outbox row with the expected payload.

    We monkey-patch :func:`check_sqlite_integrity` to return a fixed
    failure report rather than synthesising byte-level corruption here:
    the integrity check is covered exhaustively in
    :mod:`tests.core.test_integrity`, and realistic corruption severe
    enough for SQLite to notice tends to also prevent subsequent INSERT
    operations — which would route to ``EXIT_SCRIPT_ERROR`` instead and
    defeat the point of this test (that a *detected* failure is alerted
    via the outbox).
    """
    fake_report = IntegrityReport(
        ok=False,
        errors=[
            "row 17 missing from index sqlite_autoindex_events_1",
            "wrong # of entries in index idx_events_timestamp",
        ],
    )
    monkeypatch.setattr(dic, "check_sqlite_integrity", lambda path: fake_report)

    exit_code = dic.run(str(fresh_db))
    assert exit_code == dic.EXIT_INTEGRITY_FAILED

    out = capsys.readouterr().out
    assert "fail:" in out
    assert "2 error(s)" in out

    conn = sqlite3.connect(fresh_db)
    try:
        rows = conn.execute("SELECT id, event_id, subject, payload, state FROM outbox").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    outbox_id, event_id, subject, payload_raw, state = rows[0]
    assert subject == dic.ALERT_SUBJECT
    assert state == "pending"
    assert event_id.startswith("integrity-")
    assert outbox_id  # non-empty uuid
    payload = json.loads(payload_raw)
    assert payload["kind"] == "integrity_failure"
    assert payload["db_path"] == str(fresh_db)
    assert payload["errors"] == fake_report.errors
    assert isinstance(payload["detected_at"], int)


def test_enqueue_alert_inserts_one_row(fresh_db):
    """Direct call path: ``enqueue_alert`` inserts a row with the passed errors."""
    report = IntegrityReport(ok=False, errors=["row 17 missing", "page 2 bad"])
    outbox_id = dic.enqueue_alert(str(fresh_db), report, now=1700001234)

    conn = sqlite3.connect(fresh_db)
    try:
        row = conn.execute(
            "SELECT id, event_id, subject, payload FROM outbox WHERE id=?",
            (outbox_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    _, event_id, subject, payload_raw = row
    assert subject == dic.ALERT_SUBJECT
    assert event_id.startswith("integrity-1700001234-")
    payload = json.loads(payload_raw)
    assert payload["errors"] == ["row 17 missing", "page 2 bad"]
    assert payload["detected_at"] == 1700001234


def test_run_on_db_without_outbox_table_returns_script_error(tmp_path, capsys, monkeypatch):
    """If integrity fails but outbox INSERT raises, exit EXIT_SCRIPT_ERROR.

    Scenario: integrity_check reports errors (forced here), but the DB
    is so incomplete (no outbox table) that enqueue raises
    :class:`sqlite3.OperationalError`. We prefer the wrapper to page
    the operator out-of-band over silently swallowing.
    """
    db_path = tmp_path / "no_outbox.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(schema.CREATE_EVENTS_SQL)
        conn.commit()
    finally:
        conn.close()

    fake_report = IntegrityReport(ok=False, errors=["row 1 missing"])
    monkeypatch.setattr(dic, "check_sqlite_integrity", lambda path: fake_report)

    exit_code = dic.run(str(db_path))
    assert exit_code == dic.EXIT_SCRIPT_ERROR
    err = capsys.readouterr().err
    assert "outbox" in err


def test_main_parses_db_flag(fresh_db):
    """``main`` honours ``--db`` and routes to :func:`run`."""
    exit_code = dic.main(["--db", str(fresh_db)])
    assert exit_code == dic.EXIT_OK
