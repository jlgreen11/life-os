"""
Life OS — Database Manager

Manages all SQLite database connections and schema migrations.

Database Architecture:
    events.db       — Immutable event log (append-only)
    entities.db     — People, places, subscriptions, relationships
    state.db        — Current state, tasks, active contexts
    user_model.db   — The user model and all memory layers
    preferences.db  — User preferences and rules
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


class DatabaseManager:
    """Manages all SQLite database connections and schema migrations."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._databases: dict[str, str] = {
            "events": str(self.data_dir / "events.db"),
            "entities": str(self.data_dir / "entities.db"),
            "state": str(self.data_dir / "state.db"),
            "user_model": str(self.data_dir / "user_model.db"),
            "preferences": str(self.data_dir / "preferences.db"),
        }

    def initialize_all(self):
        """Create all databases and run schema migrations."""
        self._init_events_db()
        self._init_entities_db()
        self._init_state_db()
        self._init_user_model_db()
        self._init_preferences_db()

    @contextmanager
    def get_connection(self, db_name: str) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with WAL mode and foreign keys enabled."""
        conn = sqlite3.connect(self._databases[db_name])
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # events.db — The immutable event log
    # -----------------------------------------------------------------------

    def _init_events_db(self):
        with self.get_connection("events") as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id              TEXT PRIMARY KEY,
                    type            TEXT NOT NULL,
                    source          TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    priority        TEXT NOT NULL DEFAULT 'normal',
                    payload         TEXT NOT NULL DEFAULT '{}',
                    metadata        TEXT NOT NULL DEFAULT '{}',
                    embedding_id    TEXT,
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
                CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_priority ON events(priority);

                -- Processed events tracking (which services have seen this event)
                CREATE TABLE IF NOT EXISTS event_processing_log (
                    event_id        TEXT NOT NULL,
                    service         TEXT NOT NULL,
                    processed_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    result          TEXT,
                    PRIMARY KEY (event_id, service)
                );
            """)

    # -----------------------------------------------------------------------
    # entities.db — People, places, things
    # -----------------------------------------------------------------------

    def _init_entities_db(self):
        with self.get_connection("entities") as conn:
            conn.executescript("""
                -- Contacts (people in the user's life)
                CREATE TABLE IF NOT EXISTS contacts (
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
                    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_contacts_name ON contacts(name);
                CREATE INDEX IF NOT EXISTS idx_contacts_priority ON contacts(is_priority);

                -- Contact identifiers (for quick lookup by email/phone/handle)
                CREATE TABLE IF NOT EXISTS contact_identifiers (
                    identifier      TEXT NOT NULL,
                    identifier_type TEXT NOT NULL,
                    contact_id      TEXT NOT NULL,
                    PRIMARY KEY (identifier, identifier_type),
                    FOREIGN KEY (contact_id) REFERENCES contacts(id)
                );

                CREATE INDEX IF NOT EXISTS idx_identifiers_contact ON contact_identifiers(contact_id);

                -- Places
                CREATE TABLE IF NOT EXISTS places (
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
                    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                -- Subscriptions
                CREATE TABLE IF NOT EXISTS subscriptions (
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
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                -- Relationships between entities
                CREATE TABLE IF NOT EXISTS entity_relationships (
                    id              TEXT PRIMARY KEY,
                    entity_a_type   TEXT NOT NULL,
                    entity_a_id     TEXT NOT NULL,
                    relationship    TEXT NOT NULL,
                    entity_b_type   TEXT NOT NULL,
                    entity_b_id     TEXT NOT NULL,
                    weight          REAL DEFAULT 1.0,
                    metadata        TEXT DEFAULT '{}',
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_rel_a ON entity_relationships(entity_a_type, entity_a_id);
                CREATE INDEX IF NOT EXISTS idx_rel_b ON entity_relationships(entity_b_type, entity_b_id);
            """)

    # -----------------------------------------------------------------------
    # state.db — Current state of the world
    # -----------------------------------------------------------------------

    def _init_state_db(self):
        with self.get_connection("state") as conn:
            conn.executescript("""
                -- Tasks
                CREATE TABLE IF NOT EXISTS tasks (
                    id                  TEXT PRIMARY KEY,
                    title               TEXT NOT NULL,
                    description         TEXT,
                    source              TEXT DEFAULT 'user',
                    source_event_id     TEXT,
                    source_context      TEXT,
                    domain              TEXT,
                    priority            TEXT DEFAULT 'normal',
                    tags                TEXT DEFAULT '[]',
                    due_date            TEXT,
                    reminder_at         TEXT,
                    estimated_minutes   INTEGER,
                    related_contacts    TEXT DEFAULT '[]',
                    related_files       TEXT DEFAULT '[]',
                    related_events      TEXT DEFAULT '[]',
                    depends_on          TEXT DEFAULT '[]',
                    status              TEXT DEFAULT 'pending',
                    completed_at        TEXT,
                    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_date);
                CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);

                -- Notifications
                CREATE TABLE IF NOT EXISTS notifications (
                    id              TEXT PRIMARY KEY,
                    title           TEXT NOT NULL,
                    body            TEXT,
                    priority        TEXT DEFAULT 'normal',
                    source_event_id TEXT,
                    domain          TEXT,
                    status          TEXT DEFAULT 'pending',
                    delivered_at    TEXT,
                    read_at         TEXT,
                    acted_on_at     TEXT,
                    dismissed_at    TEXT,
                    action_url      TEXT,
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_notif_status ON notifications(status);

                -- Connector state (sync cursors, auth tokens, health)
                CREATE TABLE IF NOT EXISTS connector_state (
                    connector_id    TEXT PRIMARY KEY,
                    status          TEXT DEFAULT 'inactive',
                    last_sync       TEXT,
                    sync_cursor     TEXT,
                    error_count     INTEGER DEFAULT 0,
                    last_error      TEXT,
                    config          TEXT DEFAULT '{}',
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                -- Key-value store for miscellaneous state
                CREATE TABLE IF NOT EXISTS kv_store (
                    key             TEXT PRIMARY KEY,
                    value           TEXT NOT NULL,
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
            """)

    # -----------------------------------------------------------------------
    # user_model.db — The user model and all memory layers
    # -----------------------------------------------------------------------

    def _init_user_model_db(self):
        with self.get_connection("user_model") as conn:
            conn.executescript("""
                -- Episodic memory (Layer 1)
                CREATE TABLE IF NOT EXISTS episodes (
                    id                  TEXT PRIMARY KEY,
                    timestamp           TEXT NOT NULL,
                    event_id            TEXT NOT NULL,
                    location            TEXT,
                    inferred_mood       TEXT,
                    active_domain       TEXT,
                    energy_level        REAL,
                    interaction_type    TEXT NOT NULL,
                    content_summary     TEXT NOT NULL,
                    content_full        TEXT,
                    contacts_involved   TEXT DEFAULT '[]',
                    topics              TEXT DEFAULT '[]',
                    entities            TEXT DEFAULT '[]',
                    outcome             TEXT,
                    user_satisfaction   REAL,
                    embedding_id        TEXT,
                    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
                CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(interaction_type);
                CREATE INDEX IF NOT EXISTS idx_episodes_domain ON episodes(active_domain);

                -- Semantic memory (Layer 2) — facts and preferences
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    key                 TEXT PRIMARY KEY,
                    category            TEXT NOT NULL,
                    value               TEXT NOT NULL,
                    confidence          REAL DEFAULT 0.5,
                    source_episodes     TEXT DEFAULT '[]',
                    first_observed      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    last_confirmed      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    times_confirmed     INTEGER DEFAULT 1,
                    is_user_corrected   INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_semantic_category ON semantic_facts(category);
                CREATE INDEX IF NOT EXISTS idx_semantic_confidence ON semantic_facts(confidence);

                -- Procedural memory (Layer 3) — routines and workflows
                CREATE TABLE IF NOT EXISTS routines (
                    name                TEXT PRIMARY KEY,
                    trigger_condition   TEXT NOT NULL,
                    steps               TEXT NOT NULL DEFAULT '[]',
                    typical_duration    REAL DEFAULT 30.0,
                    consistency_score   REAL DEFAULT 0.5,
                    times_observed      INTEGER DEFAULT 0,
                    variations          TEXT DEFAULT '[]',
                    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE TABLE IF NOT EXISTS communication_templates (
                    id                  TEXT PRIMARY KEY,
                    context             TEXT NOT NULL,
                    contact_id          TEXT,
                    channel             TEXT,
                    greeting            TEXT,
                    closing             TEXT,
                    formality           REAL DEFAULT 0.5,
                    typical_length      REAL DEFAULT 50.0,
                    uses_emoji          INTEGER DEFAULT 0,
                    common_phrases      TEXT DEFAULT '[]',
                    avoids_phrases      TEXT DEFAULT '[]',
                    tone_notes          TEXT DEFAULT '[]',
                    example_message_ids TEXT DEFAULT '[]',
                    samples_analyzed    INTEGER DEFAULT 0,
                    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                -- Signal profiles (stored as JSON blobs, updated incrementally)
                CREATE TABLE IF NOT EXISTS signal_profiles (
                    profile_type        TEXT PRIMARY KEY,
                    data                TEXT NOT NULL DEFAULT '{}',
                    samples_count       INTEGER DEFAULT 0,
                    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                -- Mood history
                CREATE TABLE IF NOT EXISTS mood_history (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           TEXT NOT NULL,
                    energy_level        REAL,
                    stress_level        REAL,
                    social_battery      REAL,
                    cognitive_load      REAL,
                    emotional_valence   REAL,
                    confidence          REAL,
                    contributing_signals TEXT DEFAULT '[]',
                    trend               TEXT DEFAULT 'stable'
                );

                CREATE INDEX IF NOT EXISTS idx_mood_timestamp ON mood_history(timestamp);

                -- Predictions log (for learning)
                CREATE TABLE IF NOT EXISTS predictions (
                    id                  TEXT PRIMARY KEY,
                    prediction_type     TEXT NOT NULL,
                    description         TEXT NOT NULL,
                    confidence          REAL NOT NULL,
                    confidence_gate     TEXT NOT NULL,
                    time_horizon        TEXT,
                    suggested_action    TEXT,
                    supporting_signals  TEXT DEFAULT '[]',
                    was_surfaced        INTEGER DEFAULT 0,
                    user_response       TEXT,
                    was_accurate        INTEGER,
                    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    resolved_at         TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type);
                CREATE INDEX IF NOT EXISTS idx_predictions_accuracy ON predictions(was_accurate);
            """)

    # -----------------------------------------------------------------------
    # preferences.db — User preferences and automation rules
    # -----------------------------------------------------------------------

    def _init_preferences_db(self):
        with self.get_connection("preferences") as conn:
            conn.executescript("""
                -- User preferences (the onboarding output + manual updates)
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key             TEXT PRIMARY KEY,
                    value           TEXT NOT NULL,
                    set_by          TEXT DEFAULT 'onboarding',
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                -- Automation rules
                CREATE TABLE IF NOT EXISTS rules (
                    id              TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    description     TEXT,
                    trigger_event   TEXT NOT NULL,
                    conditions      TEXT NOT NULL DEFAULT '[]',
                    actions         TEXT NOT NULL DEFAULT '[]',
                    is_active       INTEGER DEFAULT 1,
                    times_triggered INTEGER DEFAULT 0,
                    last_triggered  TEXT,
                    created_by      TEXT DEFAULT 'user',
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_rules_trigger ON rules(trigger_event);
                CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(is_active);

                -- Feedback log (for the learning loop)
                CREATE TABLE IF NOT EXISTS feedback_log (
                    id              TEXT PRIMARY KEY,
                    timestamp       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    action_id       TEXT NOT NULL,
                    action_type     TEXT NOT NULL,
                    feedback_type   TEXT NOT NULL,
                    response_latency_seconds REAL,
                    context         TEXT DEFAULT '{}',
                    mood_at_time    TEXT DEFAULT '{}',
                    notes           TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback_log(feedback_type);
                CREATE INDEX IF NOT EXISTS idx_feedback_action ON feedback_log(action_type);

                -- Vault definitions
                CREATE TABLE IF NOT EXISTS vaults (
                    name            TEXT PRIMARY KEY,
                    auth_method     TEXT DEFAULT 'biometric',
                    excluded_from   TEXT DEFAULT '["search", "briefing", "unified_inbox"]',
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
            """)
