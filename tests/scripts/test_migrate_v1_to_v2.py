"""Tests for ``scripts/migrate_v1_to_v2.py`` (dry-run v1→v2 migration).

Builds synthetic v1 SQLite databases in a ``tmp_path`` directory using v1 DDL
that matches ``storage/manager.py`` (schema frozen on v2-rewrite branch).
Round-trips through the migrator and asserts per-table row-count invariants
plus the dropped-profile guard required by the CEO plan.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts import migrate_v1_to_v2 as migrate
from storage import schema as v2_schema


# ---------------------------------------------------------------------------
# Fixture builders: minimal v1 DDL, just enough to exercise the migrator.
# ---------------------------------------------------------------------------
def _build_events_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE events (
                id              TEXT PRIMARY KEY,
                type            TEXT NOT NULL,
                source          TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                priority        TEXT NOT NULL DEFAULT 'normal',
                payload         TEXT NOT NULL DEFAULT '{}',
                metadata        TEXT NOT NULL DEFAULT '{}',
                embedding_id    TEXT,
                created_at      TEXT NOT NULL DEFAULT '2026-04-01T00:00:00.000Z'
            );
            """
        )
        rows = [
            (
                f"evt-{i:03d}",
                "email.received",
                "proton_mail",
                f"2026-04-{(i % 28) + 1:02d}T12:00:00.000Z",
                "normal",
                json.dumps({"subject": f"hello {i}"}),
                "{}",
                None,
                "2026-04-01T00:00:00.000Z",
            )
            for i in range(10)
        ]
        conn.executemany(
            """
            INSERT INTO events
                (id, type, source, timestamp, priority, payload, metadata, embedding_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _build_entities_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE contacts (
                id                      TEXT PRIMARY KEY,
                name                    TEXT NOT NULL,
                aliases                 TEXT DEFAULT '[]',
                emails                  TEXT DEFAULT '[]',
                phones                  TEXT DEFAULT '[]',
                channels                TEXT DEFAULT '{}',
                relationship            TEXT,
                domains                 TEXT DEFAULT '[]',
                is_priority             INTEGER DEFAULT 0,
                preferred_channel       TEXT,
                always_surface          INTEGER DEFAULT 0,
                typical_response_time   REAL,
                communication_style     TEXT,
                last_contact            TEXT,
                contact_frequency_days  REAL,
                notes                   TEXT DEFAULT '[]',
                created_at              TEXT,
                updated_at              TEXT
            );
            CREATE TABLE places (
                id                      TEXT PRIMARY KEY,
                name                    TEXT NOT NULL,
                latitude                REAL,
                longitude               REAL,
                address                 TEXT,
                wifi_ssid               TEXT,
                place_type              TEXT,
                domain                  TEXT,
                visit_count             INTEGER DEFAULT 0,
                avg_duration_minutes    REAL,
                associated_behaviors    TEXT DEFAULT '{}',
                created_at              TEXT,
                updated_at              TEXT
            );
            CREATE TABLE subscriptions (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                amount          REAL NOT NULL,
                currency        TEXT DEFAULT 'USD',
                frequency       TEXT DEFAULT 'monthly',
                last_charge     TEXT,
                next_charge     TEXT,
                category        TEXT,
                last_used       TEXT,
                usage_frequency TEXT,
                cancel_url      TEXT,
                notes           TEXT,
                created_at      TEXT,
                updated_at      TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO contacts (id, name, aliases, emails, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "c-1",
                "Alice",
                json.dumps(["Ally"]),
                json.dumps(["alice@example.com"]),
                "2026-01-01T00:00:00.000Z",
                "2026-04-01T00:00:00.000Z",
            ),
        )
        conn.execute(
            "INSERT INTO places (id, name, latitude, longitude, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("p-1", "Home", 37.77, -122.41, "2026-01-01T00:00:00.000Z", "2026-04-01T00:00:00.000Z"),
        )
        conn.execute(
            "INSERT INTO subscriptions (id, name, amount, currency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("s-1", "Netflix", 15.99, "USD", "2026-01-01T00:00:00.000Z", "2026-04-01T00:00:00.000Z"),
        )


def _build_state_db(path: Path, *, task_count: int = 2) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE tasks (
                id                  TEXT PRIMARY KEY,
                title               TEXT NOT NULL,
                description         TEXT,
                source              TEXT,
                source_event_id     TEXT,
                source_context      TEXT,
                domain              TEXT,
                priority            TEXT,
                tags                TEXT,
                due_date            TEXT,
                reminder_at         TEXT,
                estimated_minutes   INTEGER,
                related_contacts    TEXT DEFAULT '[]',
                related_files       TEXT,
                related_events      TEXT,
                depends_on          TEXT,
                status              TEXT DEFAULT 'pending',
                completed_at        TEXT,
                created_at          TEXT,
                updated_at          TEXT
            );
            """
        )
        for i in range(task_count):
            conn.execute(
                """
                INSERT INTO tasks (id, title, description, source, priority, related_contacts, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"task-{i:03d}",
                    f"Task {i}",
                    "desc",
                    "ai",
                    "normal",
                    json.dumps([]),
                    "2026-04-10T08:00:00.000Z",
                ),
            )


def _build_user_model_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE signal_profiles (
                profile_type        TEXT PRIMARY KEY,
                data                TEXT NOT NULL DEFAULT '{}',
                samples_count       INTEGER DEFAULT 0,
                updated_at          TEXT
            );
            """
        )
        # Two kept + one dropped + one unknown.
        rows = [
            ("cadence", json.dumps({"per_contact": {"c-1": {"avg_days": 3}}}), 42, "2026-04-15T00:00:00.000Z"),
            ("relationship", json.dumps({"pairs": []}), 12, "2026-04-15T00:00:00.000Z"),
            ("mood", json.dumps({"valence": 0.1}), 99, "2026-04-15T00:00:00.000Z"),  # DROP
            ("unknown_legacy", json.dumps({}), 0, "2026-04-15T00:00:00.000Z"),  # DROP
        ]
        conn.executemany(
            "INSERT INTO signal_profiles (profile_type, data, samples_count, updated_at) VALUES (?, ?, ?, ?)",
            rows,
        )


def _build_preferences_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE user_preferences (
                key             TEXT PRIMARY KEY,
                value           TEXT NOT NULL,
                set_by          TEXT,
                updated_at      TEXT
            );
            CREATE TABLE feedback_log (
                id              TEXT PRIMARY KEY,
                timestamp       TEXT,
                action_id       TEXT,
                action_type     TEXT,
                feedback_type   TEXT,
                response_latency_seconds REAL,
                context         TEXT,
                mood_at_time    TEXT,
                notes           TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO user_preferences (key, value, set_by, updated_at) VALUES (?, ?, ?, ?)",
            ("verbosity", "brief", "onboarding", "2026-03-01T00:00:00.000Z"),
        )
        conn.execute(
            "INSERT INTO user_preferences (key, value, set_by, updated_at) VALUES (?, ?, ?, ?)",
            ("tone", "concise", "user", "2026-03-01T00:00:00.000Z"),
        )
        # 3 historical notification feedback rows (will be skipped).
        for i in range(3):
            conn.execute(
                "INSERT INTO feedback_log (id, timestamp, action_id, action_type, feedback_type) VALUES (?, ?, ?, ?, ?)",
                (f"fb-{i}", "2026-03-10T00:00:00.000Z", "act-1", "notification", "acted_on"),
            )


@pytest.fixture()
def v1_sample_dir(tmp_path: Path) -> Path:
    _build_events_db(tmp_path / "events.db")
    _build_entities_db(tmp_path / "entities.db")
    _build_state_db(tmp_path / "state.db")
    _build_user_model_db(tmp_path / "user_model.db")
    _build_preferences_db(tmp_path / "preferences.db")
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_migration_row_counts(v1_sample_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "dryrun.db"
    report = migrate.run_migration(v1_sample_dir, out)

    # Row-count invariants per table.
    assert report.events.source == 10
    assert report.events.translated == 10
    assert report.events.dropped == 0

    # 1 contact + 1 place + 1 subscription = 3 entities
    assert report.entities.source == 3
    assert report.entities.translated == 3

    assert report.moments_from_tasks.source == 2
    assert report.moments_from_tasks.translated == 2

    # 2 kept (cadence, relationship), 2 dropped (mood, unknown_legacy)
    assert report.signal_profiles.source == 4
    assert report.signal_profiles.translated == 2
    assert report.signal_profiles.dropped == 2

    assert report.preferences.source == 2
    assert report.preferences.translated == 2

    # Notification feedback is skipped; count matches source rows.
    assert report.notification_feedback_skipped == 3

    # No invariant violations.
    assert not [n for n in report.notes if n.startswith("INVARIANT:")]


def test_dropped_profile_types_never_appear_in_output(v1_sample_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "dryrun.db"
    migrate.run_migration(v1_sample_dir, out)

    with sqlite3.connect(out) as conn:
        producers = {row[0] for row in conn.execute("SELECT DISTINCT producer FROM signal_profiles").fetchall()}

    for dropped in ("mood", "decision", "expertise", "values", "unknown_legacy"):
        assert dropped not in producers, f"dropped profile type leaked: {dropped}"


def test_output_schema_matches_v2(v1_sample_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "dryrun.db"
    migrate.run_migration(v1_sample_dir, out)

    with sqlite3.connect(out) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }

    # Every table declared in storage/schema.py must exist in the output.
    for name in v2_schema.get_table_names():
        assert name in tables, f"missing v2 table: {name}"


def test_legacy_task_becomes_moment_with_history(v1_sample_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "dryrun.db"
    migrate.run_migration(v1_sample_dir, out)

    with sqlite3.connect(out) as conn:
        moments = conn.execute(
            "SELECT id, source_insight_type, state, evidence FROM moments WHERE id LIKE 'task-%'"
        ).fetchall()
        history = conn.execute("SELECT moment_id, to_state, annotation FROM moment_state_history").fetchall()

    assert len(moments) == 2
    for mid, sit, state, evidence in moments:
        assert sit == "legacy_task"
        assert state == "suggested"
        assert evidence == "[]"

    hist_by_id = {row[0]: row for row in history}
    assert set(hist_by_id) == {m[0] for m in moments}
    for row in history:
        assert row[1] == "suggested"
        assert row[2] == "legacy_task_migration"


def test_refuses_to_overwrite_existing_target(v1_sample_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "already.db"
    out.write_bytes(b"")
    with pytest.raises(FileExistsError):
        migrate.run_migration(v1_sample_dir, out)


def test_events_timestamp_coerced_to_unix_integer(v1_sample_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "dryrun.db"
    migrate.run_migration(v1_sample_dir, out)

    with sqlite3.connect(out) as conn:
        rows = conn.execute("SELECT id, timestamp FROM events LIMIT 3").fetchall()
    for _, ts in rows:
        assert isinstance(ts, int)
        assert ts > 1_700_000_000  # plausibly post-2023


def test_missing_source_dbs_are_noted(tmp_path: Path) -> None:
    # Create only preferences.db; the other four are missing.
    _build_preferences_db(tmp_path / "preferences.db")
    out = tmp_path / "dryrun.db"
    report = migrate.run_migration(tmp_path, out)

    assert set(report.missing_source_dbs) == {
        "events.db",
        "entities.db",
        "state.db",
        "user_model.db",
    }
    assert report.preferences.translated == 2
    assert report.events.translated == 0
    assert report.entities.translated == 0
