"""Targeted tests for :func:`api.routes.settings._load_preferences`.

Targets coverage gap #8 in the 2026-04-23 audit — the for-loop body
at ``api/routes/settings.py`` lines 156-173 that coerces rows from
the ``preferences`` table into the render payload. Existing tests
exercise the happy path via the FastAPI :class:`TestClient`; the edge
branches (NULL value, malformed JSON, bad float, non-string key, empty
life_os.db, schema-missing table, connection on another thread) were
unhit before these tests.

We call ``_load_preferences`` directly with a minimal
:class:`types.SimpleNamespace` request so no threading machinery is
involved — the existing :class:`TestClient`-backed test in
``test_routes_settings.py`` hits an unrelated pre-existing threading
failure; this file avoids it by going under the web layer.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from api.routes.settings import _PREFERENCE_DEFAULTS, _load_preferences
from storage import schema


def _make_request(conn: sqlite3.Connection | None) -> SimpleNamespace:
    """Build a request whose ``app.state.life_os.db`` points at ``conn``."""
    life_os = SimpleNamespace(db=conn) if conn is not None else None
    app = SimpleNamespace(state=SimpleNamespace(life_os=life_os))
    return SimpleNamespace(app=app)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        c.execute(stmt)
    c.commit()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# happy path — every defaults key loaded from the preferences table
# ---------------------------------------------------------------------------


def test_load_preferences_returns_defaults_when_no_rows(conn):
    """Fresh DB with the schema applied but no rows → defaults."""
    req = _make_request(conn)
    prefs = _load_preferences(req)
    assert prefs == _PREFERENCE_DEFAULTS


def test_load_preferences_reads_persisted_strings_and_floats(conn):
    """JSON-encoded values round-trip through the coercion logic."""
    # quiet_hours_start / quiet_hours_end — strings
    # autonomy_level / proactivity — floats (JSON-encoded numbers)
    rows = [
        ("quiet_hours_start", json.dumps("21:30")),
        ("quiet_hours_end", json.dumps("06:45")),
        ("autonomy_level", json.dumps(0.82)),
        ("proactivity", json.dumps(0.17)),
    ]
    for key, value in rows:
        conn.execute(
            "INSERT INTO preferences (key, value, encrypted, updated_at) VALUES (?, ?, 0, strftime('%s','now'))",
            (key, value),
        )
    conn.commit()

    prefs = _load_preferences(_make_request(conn))
    assert prefs == {
        "quiet_hours_start": "21:30",
        "quiet_hours_end": "06:45",
        "autonomy_level": pytest.approx(0.82),
        "proactivity": pytest.approx(0.17),
    }


# ---------------------------------------------------------------------------
# defensive branches
# ---------------------------------------------------------------------------


def test_load_preferences_falls_back_to_raw_on_malformed_json(conn):
    """Hand-edited rows with non-JSON strings should pass through as-is
    for the string-typed keys (not blank the field)."""
    conn.execute(
        "INSERT INTO preferences (key, value, encrypted, updated_at) "
        "VALUES ('quiet_hours_start', 'not-valid-json', 0, strftime('%s','now'))"
    )
    conn.commit()
    prefs = _load_preferences(_make_request(conn))
    # json.loads fails → raw string used. The key is a string-type,
    # so it lands as the raw value.
    assert prefs["quiet_hours_start"] == "not-valid-json"


def test_load_preferences_falls_back_on_invalid_float(conn):
    """A float-typed key with a non-numeric value must fall back to
    the default rather than raising."""
    conn.execute(
        "INSERT INTO preferences (key, value, encrypted, updated_at) "
        "VALUES ('autonomy_level', '\"not-a-number\"', 0, strftime('%s','now'))"
    )
    conn.commit()
    prefs = _load_preferences(_make_request(conn))
    assert prefs["autonomy_level"] == _PREFERENCE_DEFAULTS["autonomy_level"]


def test_load_preferences_stringifies_non_string_string_keys(conn):
    """A string-typed key whose stored JSON decodes to a non-string
    (e.g. an int) must be coerced via ``str()`` — defense against a
    client sending the wrong shape."""
    conn.execute(
        "INSERT INTO preferences (key, value, encrypted, updated_at) "
        "VALUES ('quiet_hours_end', '42', 0, strftime('%s','now'))"
    )
    conn.commit()
    prefs = _load_preferences(_make_request(conn))
    assert prefs["quiet_hours_end"] == "42"


# ---------------------------------------------------------------------------
# fail-soft fallbacks
# ---------------------------------------------------------------------------


def test_load_preferences_without_life_os_returns_defaults():
    """No ``app.state.life_os`` → defaults, no exception."""
    app = SimpleNamespace(state=SimpleNamespace())
    req = SimpleNamespace(app=app)
    assert _load_preferences(req) == _PREFERENCE_DEFAULTS


def test_load_preferences_with_none_db_returns_defaults():
    """``life_os.db is None`` → defaults (partial-wiring path)."""
    life_os = SimpleNamespace(db=None)
    app = SimpleNamespace(state=SimpleNamespace(life_os=life_os))
    req = SimpleNamespace(app=app)
    assert _load_preferences(req) == _PREFERENCE_DEFAULTS


def test_load_preferences_without_preferences_table_returns_defaults(tmp_path):
    """A DB without the preferences table (fresh install) → defaults."""
    db_path = tmp_path / "no-prefs.db"
    conn = sqlite3.connect(db_path)
    # Intentionally do NOT apply the schema.
    try:
        prefs = _load_preferences(_make_request(conn))
        assert prefs == _PREFERENCE_DEFAULTS
    finally:
        conn.close()
