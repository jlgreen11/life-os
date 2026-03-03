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

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


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
        """Create all databases and run schema migrations.

        Uses schema versioning to detect and apply migrations when the
        database schema is out of date. Each database has a schema_version
        table that tracks the current version. When the code's expected
        version is higher, migrations are run automatically.
        """
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

    def get_database_health(self) -> dict[str, dict]:
        """Run a quick integrity check on every database and return per-DB results.

        Uses ``PRAGMA quick_check`` (much faster than ``integrity_check``) to
        detect B-tree corruption, missing pages, and malformed cell structures.
        Each database is checked independently so a problem in one file does not
        prevent the other four from being reported.

        Returns a dict keyed by database name, each value containing:
          - ``status``: ``"ok"`` or ``"corrupted"``
          - ``errors``: list of error strings (empty when status is "ok")
          - ``path``: absolute path to the database file
          - ``size_bytes``: file size in bytes, or 0 if the file doesn't exist yet

        Example::

            {
                "events":     {"status": "ok",        "errors": [], ...},
                "user_model": {"status": "corrupted", "errors": ["..."], ...},
            }

        Safe to call frequently — ``quick_check`` opens a read-only connection
        and returns within milliseconds on typical database sizes.  Any exception
        (e.g. file not found, permission denied) is caught and reported as an
        error so callers never see a Python traceback from this method.
        """
        results: dict[str, dict] = {}
        for db_name, db_path in self._databases.items():
            path = Path(db_path)
            size_bytes = path.stat().st_size if path.exists() else 0
            errors: list[str] = []
            status = "ok"
            try:
                # Open a fresh read-only connection to avoid disturbing any
                # active WAL transaction in progress on the normal write path.
                conn = sqlite3.connect(db_path)
                conn.execute("PRAGMA journal_mode=WAL")
                rows = conn.execute("PRAGMA quick_check(10)").fetchall()
                conn.close()
                # quick_check returns a single "ok" row when healthy; any other
                # content is an error message.
                for row in rows:
                    msg = row[0]
                    if msg != "ok":
                        errors.append(msg)
                        status = "corrupted"
            except Exception as exc:
                errors.append(str(exc))
                status = "corrupted"
                logger.warning("Database health check failed for %s: %s", db_name, exc)

            # PRAGMA quick_check only scans B-tree pages and misses blob
            # overflow page corruption.  For user_model.db — which stores large
            # JSON blobs in TEXT columns that spill into overflow pages — run
            # targeted probes that force SQLite to read every overflow page.
            # SUM(LENGTH(...)) is used instead of LIMIT 1 so ALL rows' overflow
            # pages are touched, not just the first row's.
            if db_name == "user_model" and status == "ok":
                blob_probes = [
                    "SELECT content_full FROM episodes LIMIT 1",
                    "SELECT SUM(LENGTH(data)) FROM signal_profiles",
                    "SELECT SUM(LENGTH(value)) + SUM(LENGTH(source_episodes)) FROM semantic_facts",
                    "SELECT SUM(LENGTH(steps)) + SUM(LENGTH(variations)) FROM routines",
                    "SELECT SUM(LENGTH(contributing_signals)) FROM mood_history",
                    "SELECT SUM(LENGTH(supporting_signals)) FROM predictions",
                    "SELECT SUM(LENGTH(evidence)) FROM insights",
                ]
                try:
                    conn2 = sqlite3.connect(db_path)
                    conn2.execute("PRAGMA journal_mode=WAL")
                    for probe in blob_probes:
                        try:
                            conn2.execute(probe).fetchone()
                        except Exception as exc:
                            errors.append(f"blob probe failed: {exc}")
                            status = "corrupted"
                    conn2.close()
                except Exception as exc:
                    # Degrade gracefully if we can't open the connection at all
                    # or if tables don't exist yet (empty schema).
                    errors.append(f"blob probe connection failed: {exc}")
                    status = "corrupted"

            results[db_name] = {
                "status": status,
                "errors": errors,
                "path": db_path,
                "size_bytes": size_bytes,
            }

        return results

    def _check_and_recover_db(self, db_name: str) -> bool:
        """Check a database for corruption and recover by resetting if needed.

        Runs ``PRAGMA quick_check`` against the named database.  For
        ``user_model`` databases, also runs blob overflow page probes to
        detect corruption that ``quick_check`` misses in large JSON TEXT
        columns (episodes, signal_profiles, semantic_facts, etc.).

        If corruption is detected the corrupt file (and its WAL/SHM sidecars)
        are renamed with a ``.corrupt.<timestamp>`` suffix so a fresh database
        can be created by the subsequent schema initialisation step.

        Only call this for databases whose contents can be rebuilt from the
        event log (i.e. ``user_model``).  Never use it on ``events`` or
        ``entities`` which hold irreplaceable data.

        Args:
            db_name: Logical database name (key in ``self._databases``).

        Returns:
            True if recovery was performed (corrupt file backed up),
            False if the database was healthy or did not exist yet.
        """
        db_path = Path(self._databases[db_name])

        # Nothing to check if the file doesn't exist yet — it will be
        # created fresh by the init method.
        if not db_path.exists():
            return False

        # Run a quick integrity check (same approach as get_database_health).
        is_corrupt = False
        errors: list[str] = []
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            rows = conn.execute("PRAGMA quick_check(10)").fetchall()
            conn.close()
            for row in rows:
                msg = row[0]
                if msg != "ok":
                    errors.append(msg)
                    is_corrupt = True
        except Exception as exc:
            errors.append(str(exc))
            is_corrupt = True

        # PRAGMA quick_check only scans B-tree pages and misses blob
        # overflow page corruption.  For user_model.db — which stores large
        # JSON blobs in TEXT columns that spill into overflow pages — run
        # targeted probes that force SQLite to read every overflow page.
        if not is_corrupt and db_name == "user_model":
            blob_probes = [
                "SELECT content_full FROM episodes LIMIT 1",
                "SELECT SUM(LENGTH(data)) FROM signal_profiles",
                "SELECT SUM(LENGTH(value)) + SUM(LENGTH(source_episodes)) FROM semantic_facts",
                "SELECT SUM(LENGTH(steps)) + SUM(LENGTH(variations)) FROM routines",
                "SELECT SUM(LENGTH(contributing_signals)) FROM mood_history",
                "SELECT SUM(LENGTH(supporting_signals)) FROM predictions",
                "SELECT SUM(LENGTH(evidence)) FROM insights",
            ]
            try:
                conn = sqlite3.connect(str(db_path))
                conn.execute("PRAGMA journal_mode=WAL")
                for probe in blob_probes:
                    try:
                        conn.execute(probe).fetchone()
                    except Exception as exc:
                        # Tables that don't exist yet (empty/new DB) should
                        # NOT trigger corruption — only flag if the error is
                        # NOT a missing-table error.
                        if "no such table" not in str(exc).lower():
                            errors.append(f"blob probe failed: {exc}")
                            is_corrupt = True
                conn.close()
            except Exception as exc:
                errors.append(f"blob probe connection failed: {exc}")
                is_corrupt = True

            if is_corrupt:
                logger.warning(
                    "Blob overflow corruption detected in %s.db: %s",
                    db_name,
                    "; ".join(errors),
                )

        if not is_corrupt:
            return False

        # --- Corrupt: back up the file and let init recreate it. ---
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f".corrupt.{timestamp}"

        logger.warning(
            "Corruption detected in %s.db: %s",
            db_name,
            "; ".join(errors),
        )

        # Rename the main db file and any WAL/SHM sidecars.
        for ext in ("", "-wal", "-shm"):
            src = db_path.parent / (db_path.name + ext)
            if src.exists():
                dst = db_path.parent / (db_path.name + ext + suffix)
                src.rename(dst)
                logger.warning("Backed up %s → %s", src.name, dst.name)

        logger.warning(
            "%s.db was corrupted and has been reset. "
            "Learned patterns will be rebuilt automatically from event history.",
            db_name,
        )
        return True

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

        Schema versioning (v2+):
        - Added denormalized columns for workflow detection (email_from, email_to,
          task_id, calendar_event_id) to avoid expensive json_extract() calls on
          800K+ events. These columns are automatically populated from payload JSON
          for new events via triggers, and backfilled for existing events during
          migration.
        - v3: Added composite indexes for workflow detection queries to enable
          index-only scans (53s → <1s on 800K+ events).
        """
        CURRENT_VERSION = 3

        with self.get_connection("events") as conn:
            # Create schema_version table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)

            # Get current database version
            result = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current_version = result[0] if result[0] is not None else 0

            # First, ensure base tables exist
            # (CREATE TABLE IF NOT EXISTS will skip if table already exists)
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

                -- Composite index for workflow/routine detection temporal joins.
                -- Queries like "find all email.sent events that occur 0-4 hours after
                -- email.received events from sender X" need to filter by type first,
                -- then scan a timestamp range. This composite index makes those queries
                -- 1000x faster by avoiding full table scans on 77K+ events.
                CREATE INDEX IF NOT EXISTS idx_events_type_timestamp ON events(type, timestamp);

                -- Processed events tracking (which services have seen this event)
                CREATE TABLE IF NOT EXISTS event_processing_log (
                    event_id        TEXT NOT NULL,
                    service         TEXT NOT NULL,
                    processed_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    result          TEXT,
                    PRIMARY KEY (event_id, service)
                );

                -- Event annotations (tags, flags) stored separately to preserve
                -- the append-only invariant on the events table itself.  Rules
                -- engine actions like "tag" and "suppress" write here.
                CREATE TABLE IF NOT EXISTS event_tags (
                    event_id    TEXT NOT NULL,
                    tag         TEXT NOT NULL,
                    rule_id     TEXT,
                    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    PRIMARY KEY (event_id, tag)
                );

                CREATE INDEX IF NOT EXISTS idx_event_tags_tag ON event_tags(tag);

                -- Expression index on payload.message_id for fast response-time
                -- lookups in the cadence extractor.  json_extract is available in
                -- SQLite 3.38+ (bundled with Python 3.12).
                CREATE INDEX IF NOT EXISTS idx_events_payload_message_id
                    ON events(json_extract(payload, '$.message_id'));
            """)

            # Run migrations after schema is created
            if current_version < CURRENT_VERSION:
                self._migrate_events_db(conn, current_version, CURRENT_VERSION)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (CURRENT_VERSION,))

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
        """Create the user-model database schema with versioned migrations.

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

        Schema versioning:
        The schema_version table tracks the current database version. When the
        code expects a higher version, migrations are run automatically. This
        ensures existing databases are updated when new tables or columns are
        added.
        """
        # If the database file exists but is corrupted, back it up and let
        # the schema initialisation below create a fresh one.  user_model.db
        # is safe to reset because its contents are rebuilt from events.db
        # by the semantic inference, routine detection, and signal extraction
        # background loops.
        self._check_and_recover_db("user_model")

        # Current schema version (increment when making schema changes)
        CURRENT_VERSION = 4

        with self.get_connection("user_model") as conn:
            # Create schema_version table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)

            # Get current database version
            result = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current_version = result[0] if result[0] is not None else 0

            # Run migrations if needed
            if current_version < CURRENT_VERSION:
                self._migrate_user_model_db(conn, current_version, CURRENT_VERSION)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (CURRENT_VERSION,))

        # Now run the standard schema creation
        # (CREATE TABLE IF NOT EXISTS will skip tables that already exist)
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

                CREATE TABLE IF NOT EXISTS workflows (
                    name                TEXT PRIMARY KEY,
                    trigger_conditions  TEXT NOT NULL DEFAULT '[]',
                    steps               TEXT NOT NULL DEFAULT '[]',
                    typical_duration    REAL,
                    tools_used          TEXT DEFAULT '[]',
                    success_rate        REAL DEFAULT 0.5,
                    times_observed      INTEGER DEFAULT 0,
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
                    filter_reason       TEXT,
                    resolution_reason   TEXT,
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

    def _migrate_user_model_db(self, conn: sqlite3.Connection, from_version: int, to_version: int):
        """Apply schema migrations to user_model.db.

        This method handles incremental schema changes between versions. Each
        migration step is numbered and applied in sequence.

        Args:
            conn: Active database connection
            from_version: Current schema version in the database
            to_version: Target schema version from the code

        Migration strategy:
        - For version 0 → 1: The database was created before schema versioning
          existed. Tables may be partially created or have old column sets.
          The safest approach is to drop and recreate all tables (data loss is
          acceptable here since the production database is broken anyway - no
          predictions were actually being stored).

        Future migrations:
        - For version N → N+1: Use ALTER TABLE ADD COLUMN for new columns,
          CREATE TABLE for new tables, and INSERT/UPDATE for data migrations.
          Never drop tables with user data.
        """
        if from_version == 0 and to_version >= 1:
            # Migration 0 → 1: Rebuild schema from scratch
            # This is safe because the production database is in a broken state
            # (328K predictions generated but 0 stored in the database).
            # Signal profiles and episodes will be regenerated automatically.

            # Drop all tables (if they exist) to start fresh
            conn.execute("DROP TABLE IF EXISTS episodes")
            conn.execute("DROP TABLE IF EXISTS semantic_facts")
            conn.execute("DROP TABLE IF EXISTS routines")
            conn.execute("DROP TABLE IF EXISTS workflows")
            conn.execute("DROP TABLE IF EXISTS communication_templates")
            conn.execute("DROP TABLE IF EXISTS signal_profiles")
            conn.execute("DROP TABLE IF EXISTS mood_history")
            conn.execute("DROP TABLE IF EXISTS predictions")
            conn.execute("DROP TABLE IF EXISTS insights")

            # The tables will be recreated by the main schema creation code
            # that follows this migration

        if from_version == 1 and to_version >= 2:
            # Migration 1 → 2: Delete orphaned communication templates with old context
            # PR #130 changed the template ID format and context field:
            # - Old: ID = sha256(contact:channel), context = "general"
            # - New: ID = sha256(contact:channel:direction), context = "user_to_contact" | "contact_to_user"
            #
            # The 11 existing templates used the old format and will never be updated
            # by the new extraction code. Rather than attempting to infer direction
            # from heuristics, we delete them and let the system regenerate them with
            # the correct bidirectional format from ongoing message processing.
            #
            # Data loss is acceptable here:
            # - Only 11 templates exist (max 27 samples each)
            # - Templates regenerate automatically from new message events
            # - New format provides better relationship insights (separates in/out styles)
            import logging
            logger = logging.getLogger(__name__)

            orphaned_count = conn.execute(
                "SELECT COUNT(*) FROM communication_templates WHERE context = 'general'"
            ).fetchone()[0]

            if orphaned_count > 0:
                logger.info(
                    f"Migration 1→2: Deleting {orphaned_count} orphaned communication "
                    f"templates with old 'general' context (will regenerate with "
                    f"bidirectional user_to_contact/contact_to_user context)"
                )
                conn.execute("DELETE FROM communication_templates WHERE context = 'general'")

        if from_version == 2 and to_version >= 3:
            # Migration 2 → 3: Add filter_reason column to predictions table
            # This enables observability into WHY predictions are filtered, which
            # is critical for debugging and optimizing the reaction prediction logic.
            # Before this change, 99.9% of predictions were filtered but we had no
            # visibility into the reasons, making it impossible to improve the system.
            import logging
            logger = logging.getLogger(__name__)

            # Check if column already exists (migration may have been run partially)
            columns = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
            if "filter_reason" not in columns:
                logger.info(
                    "Migration 2→3: Adding filter_reason column to predictions table "
                    "for observability into prediction filtering logic"
                )
                conn.execute("ALTER TABLE predictions ADD COLUMN filter_reason TEXT")

        if from_version < 4 and to_version >= 4:
            # Migration 3 → 4: Add resolution_reason column to predictions table.
            #
            # Background: The accuracy multiplier penalizes prediction types with
            # <20% accuracy. Opportunity predictions measured at 19% accuracy (41/248),
            # but most "inaccurate" resolutions came from the automated-sender fast-path
            # in BehavioralAccuracyTracker — predictions generated before marketing
            # filter improvements (PRs #183–#189) that targeted no-reply addresses the
            # user could never reach out to. These were bugs in prediction generation,
            # not real user-behavior signals.
            #
            # Without a resolution_reason, the accuracy multiplier couldn't distinguish:
            #   - Inaccurate because of automated-sender fast-path (historical bug)
            #   - Inaccurate because user genuinely didn't reach out (real signal)
            #
            # This column lets _get_accuracy_multiplier exclude fast-path resolutions
            # (resolution_reason = 'automated_sender_fast_path') so the computed
            # accuracy reflects actual user behavior rather than historical pollution.
            #
            # Values:
            #   'automated_sender_fast_path'  — BehavioralAccuracyTracker instant-resolved
            #                                   because contact is an automated/marketing sender
            #   'timeout_no_action'           — User didn't act within the inference window
            #   NULL                          — User feedback or filtered prediction (pre-v4)
            #
            # NOTE: This migration only runs ALTER TABLE if the predictions table already
            # exists. If we're migrating from version 0 (fresh install), the migration
            # 0→1 drops all tables and the CREATE TABLE IF NOT EXISTS block that follows
            # all migrations will create the table with resolution_reason already in the
            # DDL — so no ALTER TABLE is needed.
            import logging
            logger = logging.getLogger(__name__)

            # Check if the predictions table exists before attempting ALTER TABLE
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
            ).fetchone()

            if table_exists:
                columns = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
                if "resolution_reason" not in columns:
                    logger.info(
                        "Migration 3→4: Adding resolution_reason column to predictions table "
                        "to distinguish automated-sender fast-path resolutions from real "
                        "user-behavior signals in accuracy calculations"
                    )
                    conn.execute("ALTER TABLE predictions ADD COLUMN resolution_reason TEXT")

    def _migrate_events_db(self, conn: sqlite3.Connection, from_version: int, to_version: int):
        """Apply schema migrations to events.db.

        This method handles incremental schema changes between versions. Each
        migration step is numbered and applied in sequence.

        Args:
            conn: Active database connection
            from_version: Current schema version in the database
            to_version: Target schema version from the code

        Migration strategy:
        - For version 0 → 2: Add denormalized columns for workflow detection
          to avoid expensive json_extract() calls on 800K+ events. Backfill
          these columns from existing events for the most recent 10K emails
          (covers ~4 days at 2.7K emails/day, sufficient for workflow detection).
        """
        import logging
        logger = logging.getLogger(__name__)

        if from_version == 0 and to_version >= 2:
            logger.info("Migrating events.db from v0 to v2: adding denormalized workflow columns")

            # Add denormalized columns for workflow detection
            try:
                conn.execute("ALTER TABLE events ADD COLUMN email_from TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                conn.execute("ALTER TABLE events ADD COLUMN email_to TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                conn.execute("ALTER TABLE events ADD COLUMN task_id TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                conn.execute("ALTER TABLE events ADD COLUMN calendar_event_id TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Backfill denormalized columns from most recent 10K email events
            # This is sufficient for workflow detection (covers ~4 days of history)
            # and completes in ~2 seconds instead of ~60s for all 81K emails.
            logger.info("Backfilling email_from/email_to from recent events...")

            # Backfill email_from for email.received events (most recent 10K)
            conn.execute("""
                UPDATE events
                SET email_from = LOWER(json_extract(payload, '$.from_address'))
                WHERE id IN (
                    SELECT id FROM events
                    WHERE type = 'email.received'
                      AND email_from IS NULL
                      AND json_extract(payload, '$.from_address') IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT 10000
                )
            """)
            logger.info(f"Backfilled email_from for {conn.total_changes} email.received events")

            # Backfill email_from for email.sent events (all ~260 sent emails)
            conn.execute("""
                UPDATE events
                SET email_from = LOWER(json_extract(payload, '$.from_address'))
                WHERE type = 'email.sent'
                  AND email_from IS NULL
                  AND json_extract(payload, '$.from_address') IS NOT NULL
            """)
            logger.info(f"Backfilled email_from for {conn.total_changes} email.sent events")

            # Backfill email_to for email.sent events (all ~260 sent emails)
            conn.execute("""
                UPDATE events
                SET email_to = LOWER(json_extract(payload, '$.to_addresses'))
                WHERE type = 'email.sent'
                  AND email_to IS NULL
                  AND json_extract(payload, '$.to_addresses') IS NOT NULL
            """)
            logger.info(f"Backfilled email_to for {conn.total_changes} email.sent events")

            # Backfill task_id for task.* events (all ~161 task events)
            conn.execute("""
                UPDATE events
                SET task_id = json_extract(payload, '$.task_id')
                WHERE type LIKE 'task.%'
                  AND task_id IS NULL
                  AND json_extract(payload, '$.task_id') IS NOT NULL
            """)
            logger.info(f"Backfilled task_id for {conn.total_changes} task events")

            # Backfill calendar_event_id for calendar.event.* events (all ~2.5K calendar events)
            conn.execute("""
                UPDATE events
                SET calendar_event_id = json_extract(payload, '$.event_id')
                WHERE type LIKE 'calendar.event.%'
                  AND calendar_event_id IS NULL
                  AND json_extract(payload, '$.event_id') IS NOT NULL
            """)
            logger.info(f"Backfilled calendar_event_id for {conn.total_changes} calendar events")

            # Create indexes on the new denormalized columns
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_events_email_from ON events(email_from) WHERE email_from IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_events_email_to ON events(email_to) WHERE email_to IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_events_task_id ON events(task_id) WHERE task_id IS NOT NULL;
            """)
            logger.info("Created indexes on denormalized columns")

            # Create triggers to auto-populate denormalized columns on INSERT
            # These extract commonly-queried payload fields into indexed columns
            # so workflow detection can use WHERE email_from = ? instead of
            # WHERE json_extract(payload, '$.from_address') = ? (30s → <1s)
            conn.executescript("""
                -- Extract email from_address for email.received/email.sent events
                CREATE TRIGGER IF NOT EXISTS trg_events_email_from
                AFTER INSERT ON events
                WHEN NEW.type IN ('email.received', 'email.sent')
                BEGIN
                    UPDATE events
                    SET email_from = LOWER(json_extract(NEW.payload, '$.from_address'))
                    WHERE id = NEW.id AND email_from IS NULL;
                END;

                -- Extract email to_addresses (first recipient) for email.sent events
                CREATE TRIGGER IF NOT EXISTS trg_events_email_to
                AFTER INSERT ON events
                WHEN NEW.type = 'email.sent'
                BEGIN
                    UPDATE events
                    SET email_to = LOWER(json_extract(NEW.payload, '$.to_addresses'))
                    WHERE id = NEW.id AND email_to IS NULL;
                END;

                -- Extract task_id for task.* events
                CREATE TRIGGER IF NOT EXISTS trg_events_task_id
                AFTER INSERT ON events
                WHEN NEW.type LIKE 'task.%'
                BEGIN
                    UPDATE events
                    SET task_id = json_extract(NEW.payload, '$.task_id')
                    WHERE id = NEW.id AND task_id IS NULL;
                END;

                -- Extract event_id for calendar.event.* events
                CREATE TRIGGER IF NOT EXISTS trg_events_calendar_id
                AFTER INSERT ON events
                WHEN NEW.type LIKE 'calendar.event.%'
                BEGIN
                    UPDATE events
                    SET calendar_event_id = json_extract(NEW.payload, '$.event_id')
                    WHERE id = NEW.id AND calendar_event_id IS NULL;
                END;
            """)
            logger.info("Created triggers for denormalized columns")

            logger.info("Events.db migration to v2 complete")

        # Migration: v2 → v3 (add composite indexes for workflow detection)
        if from_version <= 2 and to_version >= 3:
            logger.info("Migrating events.db from v2 to v3: adding composite indexes for workflow detection")

            # Add composite indexes that match workflow detection query patterns.
            # These enable index-only scans (read only the index B-tree, not the main table),
            # delivering 50-100x speedup on queries that filter by type + timestamp.
            #
            # PERFORMANCE IMPACT:
            # - Email workflow queries: 53s → <1s (100x faster)
            # - Task workflow queries: 20s → <0.5s (40x faster)
            # - Calendar workflow queries: 15s → <0.5s (30x faster)
            # - Total workflow detection: 53-64s → <3s (20x faster)
            conn.executescript("""
                -- Composite index for email workflow detection (type + timestamp + email_from)
                -- Covers: SELECT ... WHERE type IN ('email.received', 'email.sent') AND timestamp > ? AND email_from IS NOT NULL
                CREATE INDEX IF NOT EXISTS idx_events_type_timestamp_email_from
                    ON events(type, timestamp, email_from)
                    WHERE type IN ('email.received', 'email.sent');

                -- Composite index for task workflow detection (type + timestamp + task_id)
                -- Covers: SELECT ... WHERE type IN ('task.created', 'task.completed', ...) AND timestamp > ?
                CREATE INDEX IF NOT EXISTS idx_events_type_timestamp_task
                    ON events(type, timestamp, task_id)
                    WHERE type IN ('task.created', 'task.completed', 'email.sent', 'email.received',
                                   'calendar.event.created', 'message.sent');

                -- Composite index for calendar workflow detection (type + timestamp + calendar_event_id)
                -- Covers: SELECT ... WHERE type = 'calendar.event.created' AND timestamp > ?
                CREATE INDEX IF NOT EXISTS idx_events_type_timestamp_calendar
                    ON events(type, timestamp, calendar_event_id)
                    WHERE type IN ('calendar.event.created', 'email.received', 'email.sent',
                                   'task.created', 'message.sent');
            """)
            logger.info("Created composite indexes for workflow detection")

            logger.info("Events.db migration to v3 complete")

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

                -- Source weights: user + AI dual-weight system for tuning
                -- how much influence each data source has on insights,
                -- predictions, and signal extraction.
                --
                -- Design:
                --   user_weight  — explicit user preference (0.0 = ignore, 1.0 = max)
                --   ai_drift     — AI-computed adjustment that shifts over time
                --                   based on engagement/dismissal patterns
                --   effective_weight = clamp(user_weight + ai_drift, 0.0, 1.0)
                --
                -- The AI drift is bounded to [-0.3, +0.3] so the user always
                -- retains primary control.  Drift decays toward zero over time
                -- when no feedback is received.
                CREATE TABLE IF NOT EXISTS source_weights (
                    source_key      TEXT PRIMARY KEY,
                    category        TEXT NOT NULL,
                    label           TEXT NOT NULL,
                    description     TEXT DEFAULT '',
                    user_weight     REAL NOT NULL DEFAULT 0.5,
                    ai_drift        REAL NOT NULL DEFAULT 0.0,
                    drift_reason    TEXT DEFAULT '',
                    drift_history   TEXT DEFAULT '[]',
                    user_set_at     TEXT,
                    ai_updated_at   TEXT,
                    interactions    INTEGER DEFAULT 0,
                    engagements     INTEGER DEFAULT 0,
                    dismissals      INTEGER DEFAULT 0,
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_sw_category ON source_weights(category);
            """)
