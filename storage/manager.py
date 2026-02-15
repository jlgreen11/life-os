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

        # --- 5-Database Architecture ---
        # Data is split across five separate SQLite files by domain concern:
        #   events       — Append-only event log; the single source of truth.
        #   entities     — People, places, subscriptions, relationships (reference data).
        #   state        — Mutable current state: tasks, notifications, connector cursors.
        #   user_model   — Learned user profile: episodes, facts, routines, mood, predictions.
        #   preferences  — Explicit user settings, automation rules, feedback, vaults.
        #
        # Keeping them separate provides:
        #   1. Independent backup/restore — e.g. wipe state without losing the event history.
        #   2. Reduced write contention — high-throughput event writes don't block task reads.
        #   3. Clear ownership boundaries — each store module touches only its own DB.
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

        # WAL (Write-Ahead Logging) mode allows concurrent readers while a writer
        # is active, which is critical because the event bus may write events at
        # the same time the web layer is reading tasks or notifications.
        conn.execute("PRAGMA journal_mode=WAL")

        # Foreign keys are disabled by default in SQLite; enable them so that
        # ON DELETE / REFERENCES constraints in the schema are actually enforced.
        conn.execute("PRAGMA foreign_keys=ON")

        # sqlite3.Row lets us access columns by name (dict-like), so callers can
        # write ``row["id"]`` instead of relying on positional indexing.
        conn.row_factory = sqlite3.Row

        # Context-manager pattern: yield the connection to the caller, then
        # commit on success.  If the caller raises an exception, roll back all
        # changes made during the ``with`` block so the DB stays consistent.
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
        """Create the events database schema.

        The events table is the immutable, append-only log at the heart of Life OS.
        Every piece of incoming data (emails, messages, calendar changes, etc.) is
        recorded here as a typed event before any downstream processing occurs.

        Design decisions:
        - ``id`` is a caller-supplied TEXT (UUID), not AUTOINCREMENT, so that
          connectors can generate deterministic IDs for deduplication.
        - ``payload`` and ``metadata`` are stored as JSON TEXT blobs, giving each
          event type full flexibility in its schema without table-per-type overhead.
        - ``embedding_id`` links to the vector store for semantic search over events.
        - Indexes on type, source, timestamp, and priority support the most common
          query patterns (filter-by-type, recent-events, priority triage).
        - ``event_processing_log`` tracks which downstream services (e.g. signal
          extractor, notification manager) have already processed a given event,
          enabling exactly-once processing via the composite PRIMARY KEY.
        """
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
        """Create the entities database schema.

        Stores reference/master data for the people, places, and services in
        the user's life.  These are "slowly changing" records (unlike the fast
        append-only event log), so UPDATE operations are expected.

        Design decisions:
        - ``contacts`` uses JSON arrays (aliases, emails, phones, notes) for
          multi-value fields to avoid many-to-many join tables for simple lists.
        - ``contact_identifiers`` provides a fast reverse-lookup table so an
          incoming email/phone/handle can be resolved to a contact in O(1).
          Its composite PRIMARY KEY (identifier, identifier_type) prevents
          duplicates while allowing the same string for different types.
        - ``entity_relationships`` is a generic graph-edge table connecting any
          entity to any other entity with a typed relationship and weight,
          supporting queries like "who is related to this project?".
        - Indexes target the primary query paths: name search, priority contacts,
          and relationship traversal from either side.
        """
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
        """Create the state database schema.

        Holds all *mutable* current-state data: tasks, notifications, connector
        sync cursors, and a general-purpose key-value store.

        Design decisions:
        - ``tasks`` carries rich context (source_event_id, source_context,
          related_contacts, related_files, depends_on) so the AI engine can
          explain *why* a task exists and prioritize it intelligently.
        - ``notifications`` track a full lifecycle (pending -> delivered ->
          read -> acted_on / dismissed) with separate timestamp columns for
          each transition, enabling response-latency analytics.
        - ``connector_state`` persists sync cursors and error counts so that
          connectors can resume from where they left off after a restart and
          implement exponential back-off on repeated failures.
        - ``kv_store`` is a simple key-value table for miscellaneous state
          that does not justify its own table (e.g. "last_briefing_time").
        """
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
                    enabled         INTEGER DEFAULT 0,
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
        """Create the user-model database schema.

        Implements a three-layer cognitive memory architecture:

        Layer 1 — Episodic memory (``episodes``):
            Individual interaction records with context (location, mood, domain).
            Indexed by timestamp, interaction_type, and domain for temporal and
            categorical recall.

        Layer 2 — Semantic memory (``semantic_facts``):
            Distilled facts and preferences with a confidence score that grows
            each time the fact is re-confirmed (+0.05 per confirmation, capped
            at 1.0).  ``is_user_corrected`` flags override inferred values.

        Layer 3 — Procedural memory (``routines``, ``communication_templates``):
            Learned behavioral patterns — daily routines with consistency scores
            and per-contact/channel communication style templates.

        Supporting tables:
        - ``signal_profiles``  — JSON blobs for linguistic/cadence/behavioral
          profiles, updated incrementally (samples_count tracks how many data
          points have been incorporated).
        - ``mood_history``     — Time-series of inferred mood dimensions.
        - ``predictions``      — Logged predictions with later accuracy tracking,
          enabling the system to learn from its own forecast quality.
        """
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

                -- Insights (cross-signal discoveries from InsightEngine)
                CREATE TABLE IF NOT EXISTS insights (
                    id                  TEXT PRIMARY KEY,
                    type                TEXT NOT NULL,
                    summary             TEXT NOT NULL,
                    confidence          REAL NOT NULL,
                    evidence            TEXT DEFAULT '[]',
                    category            TEXT DEFAULT '',
                    entity              TEXT,
                    staleness_ttl_hours INTEGER DEFAULT 168,
                    dedup_key           TEXT,
                    feedback            TEXT,
                    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(type);
                CREATE INDEX IF NOT EXISTS idx_insights_dedup ON insights(dedup_key);
                CREATE INDEX IF NOT EXISTS idx_insights_created ON insights(created_at);
            """)

    # -----------------------------------------------------------------------
    # preferences.db — User preferences and automation rules
    # -----------------------------------------------------------------------

    def _init_preferences_db(self):
        """Create the preferences database schema.

        Stores explicit user intent — settings, automation rules, feedback, and
        vault definitions — as opposed to inferred knowledge in user_model.db.

        Design decisions:
        - ``user_preferences`` uses a key-value layout so new preference
          types can be added without schema changes.  ``set_by`` tracks whether
          a value came from onboarding, the API, or a rule action.
        - ``rules`` implements an event-driven automation engine: each rule
          specifies a trigger_event type, a list of JSON conditions, and a list
          of JSON actions.  ``times_triggered`` + ``last_triggered`` support
          analytics and rate-limiting.
        - ``feedback_log`` captures explicit user feedback (thumbs up/down,
          corrections) with response latency and mood context, providing the
          training signal for the learning loop.
        - ``vaults`` define sensitive-data containers that are excluded from
          search, briefing, and inbox by default, ensuring privacy-by-design.
        """
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
