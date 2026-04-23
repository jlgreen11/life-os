"""Build scaled v1 SQLite sample databases for migration scale testing.

Mirrors the v1 DDL used by ``scripts/migrate_v1_to_v2.py`` for the five
source databases (``events.db``, ``entities.db``, ``state.db``,
``user_model.db``, ``preferences.db``). Rows are deterministic
(index-derived) so the test is reproducible and the row-count invariants
encoded in ``SampleCounts`` are exact.

Defaults mirror the NEXT_TASKS scale spec:

- 10,000 events
- 500 entities (400 contacts + 50 places + 50 subscriptions)
- 200 signal-profile rows (6 kept types + 4 dropped + 190 unknown)
- 50 legacy tasks → migrated to ``Moment``
- 10 preferences + 100 notification-feedback rows (skipped by the migrator)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EVENTS: int = 10_000
DEFAULT_CONTACTS: int = 400
DEFAULT_PLACES: int = 50
DEFAULT_SUBSCRIPTIONS: int = 50
DEFAULT_TASKS: int = 50
DEFAULT_SIGNAL_PROFILES: int = 200
DEFAULT_PREFERENCES: int = 10
DEFAULT_FEEDBACK_LOG: int = 100

# Types known to the migrator; counts here drive invariant assertions.
_KEPT_PROFILE_TYPES: tuple[str, ...] = (
    "cadence",
    "relationship",
    "temporal",
    "spatial",
    "comm_template",
    "routine",
)
_DROPPED_PROFILE_TYPES: tuple[str, ...] = ("mood", "decision", "expertise", "values")


@dataclass(frozen=True)
class SampleCounts:
    """Row counts written into the scaled fixture, used for invariant asserts."""

    events: int
    contacts: int
    places: int
    subscriptions: int
    tasks: int
    signal_profiles: int
    preferences: int
    feedback_log: int

    @property
    def total_entities(self) -> int:
        return self.contacts + self.places + self.subscriptions

    @property
    def kept_signal_profiles(self) -> int:
        # Only the six canonical kept types survive migration.
        return len(_KEPT_PROFILE_TYPES)

    @property
    def dropped_signal_profiles(self) -> int:
        return self.signal_profiles - self.kept_signal_profiles


def _build_events_db(path: Path, n: int) -> None:
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
        sources = ("proton_mail", "imessage", "caldav", "ios_context")
        types = (
            "email.received",
            "message.received",
            "calendar.event",
            "location.changed",
        )
        rows = (
            (
                f"evt-{i:07d}",
                types[i % len(types)],
                sources[i % len(sources)],
                f"2026-{((i // 2500) % 12) + 1:02d}-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00.000Z",
                "normal",
                json.dumps({"seq": i, "subject": f"event {i}"}),
                "{}",
                None,
                "2026-04-01T00:00:00.000Z",
            )
            for i in range(n)
        )
        conn.executemany(
            "INSERT INTO events "
            "(id, type, source, timestamp, priority, payload, metadata, embedding_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def _build_entities_db(path: Path, contacts: int, places: int, subscriptions: int) -> None:
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
        now = "2026-04-01T00:00:00.000Z"
        conn.executemany(
            "INSERT INTO contacts (id, name, aliases, emails, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                (
                    f"c-{i:05d}",
                    f"Contact {i}",
                    json.dumps([f"alias-{i}"]),
                    json.dumps([f"contact{i}@example.com"]),
                    now,
                    now,
                )
                for i in range(contacts)
            ),
        )
        conn.executemany(
            "INSERT INTO places (id, name, latitude, longitude, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                (
                    f"p-{i:05d}",
                    f"Place {i}",
                    37.0 + i * 0.01,
                    -122.0 - i * 0.01,
                    now,
                    now,
                )
                for i in range(places)
            ),
        )
        conn.executemany(
            "INSERT INTO subscriptions (id, name, amount, currency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ((f"s-{i:05d}", f"Sub {i}", 9.99 + i, "USD", now, now) for i in range(subscriptions)),
        )


def _build_state_db(path: Path, n: int) -> None:
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
        now = "2026-04-10T08:00:00.000Z"
        conn.executemany(
            "INSERT INTO tasks "
            "(id, title, description, source, priority, related_contacts, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ((f"task-{i:05d}", f"Task {i}", "auto", "ai", "normal", "[]", now) for i in range(n)),
        )


def _build_user_model_db(path: Path, n: int) -> None:
    """Seed ``n`` signal_profile rows.

    Guarantees the 6 kept types are always present (so the migrator translates
    exactly that many). Any remaining slots are filled with unknown-type rows
    that the migrator drops.
    """
    if n < len(_KEPT_PROFILE_TYPES):
        raise ValueError(
            f"signal_profiles count must be >= {len(_KEPT_PROFILE_TYPES)} to seed every kept type; got {n}"
        )
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE signal_profiles (
                profile_type    TEXT PRIMARY KEY,
                data            TEXT NOT NULL DEFAULT '{}',
                samples_count   INTEGER DEFAULT 0,
                updated_at      TEXT
            );
            """
        )
        now = "2026-04-15T00:00:00.000Z"
        rows: list[tuple[str, str, int, str]] = [(pt, "{}", 10, now) for pt in _KEPT_PROFILE_TYPES]
        dropped_canonical = _DROPPED_PROFILE_TYPES[: max(0, n - len(_KEPT_PROFILE_TYPES))]
        rows += [(pt, "{}", 10, now) for pt in dropped_canonical]
        remaining = n - len(rows)
        rows += [(f"unknown-legacy-{i:04d}", "{}", 0, now) for i in range(remaining)]
        conn.executemany(
            "INSERT INTO signal_profiles (profile_type, data, samples_count, updated_at) VALUES (?, ?, ?, ?)",
            rows,
        )


def _build_preferences_db(path: Path, prefs: int, feedback: int) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE user_preferences (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                set_by      TEXT,
                updated_at  TEXT
            );
            CREATE TABLE feedback_log (
                id                       TEXT PRIMARY KEY,
                timestamp                TEXT,
                action_id                TEXT,
                action_type              TEXT,
                feedback_type            TEXT,
                response_latency_seconds REAL,
                context                  TEXT,
                mood_at_time             TEXT,
                notes                    TEXT
            );
            """
        )
        now = "2026-03-01T00:00:00.000Z"
        conn.executemany(
            "INSERT INTO user_preferences (key, value, set_by, updated_at) VALUES (?, ?, ?, ?)",
            ((f"pref-{i:03d}", f"val-{i}", "onboarding", now) for i in range(prefs)),
        )
        conn.executemany(
            "INSERT INTO feedback_log (id, timestamp, action_id, action_type, feedback_type) VALUES (?, ?, ?, ?, ?)",
            ((f"fb-{i:05d}", now, "act-1", "notification", "acted_on") for i in range(feedback)),
        )


def build_scaled_v1_sample(
    out_dir: Path,
    *,
    events: int = DEFAULT_EVENTS,
    contacts: int = DEFAULT_CONTACTS,
    places: int = DEFAULT_PLACES,
    subscriptions: int = DEFAULT_SUBSCRIPTIONS,
    tasks: int = DEFAULT_TASKS,
    signal_profiles: int = DEFAULT_SIGNAL_PROFILES,
    preferences: int = DEFAULT_PREFERENCES,
    feedback_log: int = DEFAULT_FEEDBACK_LOG,
) -> SampleCounts:
    """Build all five v1 SQLite files under ``out_dir`` sized for scale testing.

    Defaults mirror the NEXT_TASKS spec: 10K events, 500 entities
    (400 contacts + 50 places + 50 subscriptions), 200 signal_profiles.
    The returned :class:`SampleCounts` documents exactly what was written so
    the caller can assert per-table row-count invariants against it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _build_events_db(out_dir / "events.db", events)
    _build_entities_db(out_dir / "entities.db", contacts, places, subscriptions)
    _build_state_db(out_dir / "state.db", tasks)
    _build_user_model_db(out_dir / "user_model.db", signal_profiles)
    _build_preferences_db(out_dir / "preferences.db", preferences, feedback_log)
    return SampleCounts(
        events=events,
        contacts=contacts,
        places=places,
        subscriptions=subscriptions,
        tasks=tasks,
        signal_profiles=signal_profiles,
        preferences=preferences,
        feedback_log=feedback_log,
    )
