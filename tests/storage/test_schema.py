"""Tests for `storage/schema.py` — the v2 consolidated DDL source.

Verifies that every CREATE statement executes on a fresh SQLite instance,
that the expected 14 tables + all named indexes materialize, and that the
declared foreign keys actually block invalid writes when `foreign_keys=ON`.
"""

from __future__ import annotations

import sqlite3

import pytest

from storage import schema


@pytest.fixture
def fresh_db(tmp_path):
    """A fresh file-backed SQLite DB with the v2 schema applied, FKs on."""
    db_path = tmp_path / "lifeos.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        for stmt in schema.get_all_ddl():
            conn.execute(stmt)
        conn.commit()
        yield conn
    finally:
        conn.close()


def test_schema_version_constant():
    """SCHEMA_VERSION is a positive integer starting at 1."""
    assert isinstance(schema.SCHEMA_VERSION, int)
    assert schema.SCHEMA_VERSION == 1


def test_get_all_ddl_executes_on_empty_sqlite(tmp_path):
    """Every DDL string executes cleanly on an empty SQLite database."""
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)
    try:
        for stmt in schema.get_all_ddl():
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()


def test_get_all_ddl_is_idempotent_per_run(tmp_path):
    """Re-running the DDL on the same DB raises — schema.py is create-once.

    A migration runner is responsible for skipping if tables exist; schema.py
    itself is pure CREATE-only and must fail loudly on a non-empty DB.
    """
    db_path = tmp_path / "again.db"
    conn = sqlite3.connect(db_path)
    try:
        for stmt in schema.get_all_ddl():
            conn.execute(stmt)
        conn.commit()

        with pytest.raises(sqlite3.OperationalError):
            for stmt in schema.get_all_ddl():
                conn.execute(stmt)
    finally:
        conn.close()


def test_all_14_tables_exist(fresh_db):
    """The schema materializes all 14 expected tables."""
    expected = {
        "events",
        "event_tags",
        "entities",
        "moments",
        "moment_state_history",
        "outbox",
        "feedback_weights",
        "signal_profiles",
        "connector_state",
        "preferences",
        "rules",
        "semantic_facts",
        "feedback_events",
        "schema_version",
    }
    assert len(expected) == 14

    rows = fresh_db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    actual = {row[0] for row in rows}

    assert actual == expected
    assert set(schema.get_table_names()) == expected


def test_feedback_events_source_check_rejects_unknown(fresh_db):
    """feedback_events.source CHECK rejects values outside {v1_migration, v2}."""
    with pytest.raises(sqlite3.IntegrityError):
        fresh_db.execute(
            "INSERT INTO feedback_events (id, ts, source) VALUES (?, ?, ?)",
            ("fb-bad", 1_700_000_000, "not-a-source"),
        )

    # Each allowed source works.
    for i, source in enumerate(("v1_migration", "v2")):
        fresh_db.execute(
            "INSERT INTO feedback_events (id, ts, source) VALUES (?, ?, ?)",
            (f"fb-{i}", 1_700_000_000 + i, source),
        )
    fresh_db.commit()


def test_feedback_events_primary_key_blocks_duplicates(fresh_db):
    """feedback_events.id is the primary key; duplicate inserts raise."""
    fresh_db.execute(
        "INSERT INTO feedback_events (id, ts) VALUES (?, ?)",
        ("fb-dup", 1_700_000_000),
    )
    fresh_db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        fresh_db.execute(
            "INSERT INTO feedback_events (id, ts) VALUES (?, ?)",
            ("fb-dup", 1_700_000_001),
        )


def test_all_named_indexes_exist(fresh_db):
    """Every index declared by schema.py actually lands in the DB."""
    rows = fresh_db.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'").fetchall()
    actual = {row[0] for row in rows}

    for idx_name in schema.get_index_names():
        assert idx_name in actual, f"Missing index: {idx_name}"


def test_moments_unique_on_source_and_evidence_hash(fresh_db):
    """Producer idempotency: two moments with same (type, hash) are rejected."""
    fresh_db.execute(
        "INSERT INTO moments (id, expires_at, insight, evidence, evidence_hash, "
        "proposed_action, state, source_insight_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "m1",
            9999999999,
            "ping Mike",
            "[]",
            "hash-a",
            "{}",
            "suggested",
            "cadence",
        ),
    )
    fresh_db.commit()

    with pytest.raises(sqlite3.IntegrityError):
        fresh_db.execute(
            "INSERT INTO moments (id, expires_at, insight, evidence, "
            "evidence_hash, proposed_action, state, source_insight_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "m2",
                9999999999,
                "ping Mike (dup)",
                "[]",
                "hash-a",
                "{}",
                "suggested",
                "cadence",
            ),
        )


def test_outbox_unique_on_event_id_and_subject(fresh_db):
    """Outbox enqueue idempotency: (event_id, subject) is unique."""
    fresh_db.execute(
        "INSERT INTO outbox (id, event_id, subject, payload) VALUES (?, ?, ?, ?)",
        ("o1", "evt-1", "moment.accepted", "{}"),
    )
    fresh_db.commit()

    with pytest.raises(sqlite3.IntegrityError):
        fresh_db.execute(
            "INSERT INTO outbox (id, event_id, subject, payload) VALUES (?, ?, ?, ?)",
            ("o2", "evt-1", "moment.accepted", "{}"),
        )


def test_moment_state_check_rejects_unknown_state(fresh_db):
    """The CHECK constraint on moments.state blocks invalid states."""
    with pytest.raises(sqlite3.IntegrityError):
        fresh_db.execute(
            "INSERT INTO moments (id, expires_at, insight, evidence, "
            "evidence_hash, proposed_action, state, source_insight_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "bad",
                9999999999,
                "x",
                "[]",
                "hash-bad",
                "{}",
                "inventing-a-state",
                "cadence",
            ),
        )


def test_moment_state_history_fk_enforced(fresh_db):
    """Inserting history for a non-existent moment_id fails with FKs on."""
    with pytest.raises(sqlite3.IntegrityError):
        fresh_db.execute(
            "INSERT INTO moment_state_history (moment_id, from_state, to_state) VALUES (?, ?, ?)",
            ("no-such-moment", None, "suggested"),
        )


def test_event_tags_fk_cascade_delete(fresh_db):
    """Deleting an event cascades to its tags (FK ON DELETE CASCADE)."""
    fresh_db.execute(
        "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
        ("e1", "email", "proton", 1_700_000_000, "{}"),
    )
    fresh_db.execute(
        "INSERT INTO event_tags (event_id, tag, value) VALUES (?, ?, ?)",
        ("e1", "topic", "work"),
    )
    fresh_db.commit()

    assert fresh_db.execute("SELECT COUNT(*) FROM event_tags WHERE event_id='e1'").fetchone()[0] == 1

    fresh_db.execute("DELETE FROM events WHERE id='e1'")
    fresh_db.commit()

    assert fresh_db.execute("SELECT COUNT(*) FROM event_tags WHERE event_id='e1'").fetchone()[0] == 0


def test_entities_kind_check(fresh_db):
    """entities.kind CHECK rejects unknown kinds."""
    with pytest.raises(sqlite3.IntegrityError):
        fresh_db.execute(
            "INSERT INTO entities (id, kind, name) VALUES (?, ?, ?)",
            ("ent-bad", "not-a-kind", "x"),
        )

    # Sanity: each allowed kind works.
    for i, kind in enumerate(("contact", "place", "subscription", "topic")):
        fresh_db.execute(
            "INSERT INTO entities (id, kind, name) VALUES (?, ?, ?)",
            (f"ent-{i}", kind, f"name-{i}"),
        )
    fresh_db.commit()


def test_outbox_state_check(fresh_db):
    """outbox.state CHECK rejects unknown states."""
    with pytest.raises(sqlite3.IntegrityError):
        fresh_db.execute(
            "INSERT INTO outbox (id, event_id, subject, payload, state) VALUES (?, ?, ?, ?, ?)",
            ("o-bad", "e-x", "s", "{}", "not-a-state"),
        )


def test_schema_version_row_can_be_inserted(fresh_db):
    """The schema_version table accepts the active SCHEMA_VERSION."""
    fresh_db.execute(
        "INSERT INTO schema_version (version) VALUES (?)",
        (schema.SCHEMA_VERSION,),
    )
    fresh_db.commit()

    row = fresh_db.execute("SELECT version FROM schema_version").fetchone()
    assert row[0] == schema.SCHEMA_VERSION


def test_ddl_ordering_tables_before_indexes():
    """get_all_ddl() lists every table before any index (FK-safe order)."""
    ddl = schema.get_all_ddl()
    kinds = ["TABLE" if "CREATE TABLE" in stmt else "INDEX" for stmt in ddl]
    last_table = max(i for i, k in enumerate(kinds) if k == "TABLE")
    first_index = min(i for i, k in enumerate(kinds) if k == "INDEX")
    assert last_table < first_index
