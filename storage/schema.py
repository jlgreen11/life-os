"""v2 schema DDL as code.

Single authoritative source for the consolidated SQLite schema used by the v2
rewrite. All 13 tables (12 domain tables + `schema_version` marker) live in
one `lifeos.db` file; LanceDB stays as a separate index directory.

Reference: `docs/plans/2026-04-21-v2-rewrite-plan.md` § "13-table schema".
CEO plan: `~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md`.

Every DDL here is expressed as a module-level `CREATE_<NAME>_SQL` constant.
`get_all_ddl()` returns the ordered list (tables first, then indexes) so that
`DatabaseManager.initialize()` can execute them sequentially on a fresh DB.

Invariants enforced at this layer:
- FKs ON (set by caller via `PRAGMA foreign_keys=ON`).
- `events` is append-only (enforced at repo layer, not DB).
- Moment uniqueness is `(source_insight_type, evidence_hash)` — producers are
  expected to be idempotent.
- `moment_state_history` is append-only and written only by
  `MomentRepository.transition(...)` inside the same transaction as the
  state update.
"""

from __future__ import annotations

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# 1. events — immutable append-only event log.
# ---------------------------------------------------------------------------
CREATE_EVENTS_SQL = """
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    priority TEXT NOT NULL DEFAULT 'normal',
    payload TEXT NOT NULL,
    metadata TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""

CREATE_EVENTS_TIMESTAMP_INDEX_SQL = "CREATE INDEX idx_events_timestamp ON events(timestamp)"
CREATE_EVENTS_SOURCE_TIMESTAMP_INDEX_SQL = "CREATE INDEX idx_events_source_timestamp ON events(source, timestamp)"


# ---------------------------------------------------------------------------
# 2. event_tags — N-to-N tags on events (source classification, topic,
#    priority, etc.).
# ---------------------------------------------------------------------------
CREATE_EVENT_TAGS_SQL = """
CREATE TABLE event_tags (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    value TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    PRIMARY KEY (event_id, tag, value)
)
"""

CREATE_EVENT_TAGS_EVENT_ID_INDEX_SQL = "CREATE INDEX idx_event_tags_event_id ON event_tags(event_id)"
CREATE_EVENT_TAGS_TAG_INDEX_SQL = "CREATE INDEX idx_event_tags_tag ON event_tags(tag)"


# ---------------------------------------------------------------------------
# 3. entities — contacts, places, subscriptions, topics. `kind` disambiguates.
# ---------------------------------------------------------------------------
CREATE_ENTITIES_SQL = """
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('contact', 'place', 'subscription', 'topic')),
    name TEXT NOT NULL,
    aliases TEXT,
    attributes TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""

CREATE_ENTITIES_KIND_NAME_INDEX_SQL = "CREATE INDEX idx_entities_kind_name ON entities(kind, name)"


# ---------------------------------------------------------------------------
# 4. moments — first-class Moment primitive.
# ---------------------------------------------------------------------------
CREATE_MOMENTS_SQL = """
CREATE TABLE moments (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    scheduled_for INTEGER,
    expires_at INTEGER NOT NULL,
    context_trigger TEXT,
    insight TEXT NOT NULL,
    evidence TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    proposed_action TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'suggested', 'accepted', 'dismissed', 'snoozed', 'done', 'expired'
    )),
    snooze_until INTEGER,
    confidence REAL NOT NULL DEFAULT 0.0,
    feedback_weight REAL NOT NULL DEFAULT 1.0,
    source_insight_type TEXT NOT NULL CHECK (source_insight_type IN (
        'cadence', 'relationship', 'temporal', 'spatial',
        'comm_template', 'routine', 'legacy_task'
    )),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    UNIQUE (source_insight_type, evidence_hash)
)
"""

CREATE_MOMENTS_STATE_SCHEDULED_INDEX_SQL = "CREATE INDEX idx_moments_state_scheduled ON moments(state, scheduled_for)"


# ---------------------------------------------------------------------------
# 5. moment_state_history — append-only audit log of state transitions.
# ---------------------------------------------------------------------------
CREATE_MOMENT_STATE_HISTORY_SQL = """
CREATE TABLE moment_state_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    moment_id TEXT NOT NULL REFERENCES moments(id) ON DELETE CASCADE,
    from_state TEXT,
    to_state TEXT NOT NULL,
    ts INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    annotation TEXT
)
"""

CREATE_MOMENT_STATE_HISTORY_MOMENT_TS_INDEX_SQL = (
    "CREATE INDEX idx_moment_state_history_moment_ts ON moment_state_history(moment_id, ts)"
)


# ---------------------------------------------------------------------------
# 6. outbox — transactional outbox for side-effecting events.
#    ``not_before`` is the "earliest-dispatchable" epoch-second stamp used by
#    the 3 s Undo grace window: rows stay un-claimable until
#    ``not_before <= now()``. NULL means "claim-eligible immediately" — the
#    default for every producer-side enqueue. See design note
#    docs/plans/2026-04-22-undo-grace.md § "Decision 2".
# ---------------------------------------------------------------------------
CREATE_OUTBOX_SQL = """
CREATE TABLE outbox (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    payload TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN (
        'pending', 'in_progress', 'done', 'failed', 'dead'
    )),
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    claimed_at INTEGER,
    not_before INTEGER,
    UNIQUE (event_id, subject)
)
"""

# The claim query filters on (state='pending' AND (not_before IS NULL OR
# not_before <= ?)) ORDER BY created_at — this covering index spans all three
# columns so the hot path stays index-only. Replaces the original
# idx_outbox_state_created (design note § "Decision 2 → Index update").
CREATE_OUTBOX_STATE_NOTBEFORE_CREATED_INDEX_SQL = (
    "CREATE INDEX idx_outbox_state_notbefore_created ON outbox(state, not_before, created_at)"
)


# ---------------------------------------------------------------------------
# 7. feedback_weights — EWMA per-insight-type accept rate.
#    Exponentially-weighted moving average, alpha=0.1, half-life about 7
#    decisions (per CEO plan).
# ---------------------------------------------------------------------------
CREATE_FEEDBACK_WEIGHTS_SQL = """
CREATE TABLE feedback_weights (
    insight_type TEXT PRIMARY KEY,
    weight REAL NOT NULL DEFAULT 1.0,
    decision_count INTEGER NOT NULL DEFAULT 0,
    last_updated INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""


# ---------------------------------------------------------------------------
# 8. signal_profiles — per-producer rolling profiles. `key` is producer-scoped
#    (e.g. contact_id for cadence, place name for spatial).
# ---------------------------------------------------------------------------
CREATE_SIGNAL_PROFILES_SQL = """
CREATE TABLE signal_profiles (
    producer TEXT NOT NULL,
    key TEXT NOT NULL,
    profile TEXT NOT NULL,
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    PRIMARY KEY (producer, key)
)
"""

CREATE_SIGNAL_PROFILES_PRODUCER_KEY_INDEX_SQL = (
    "CREATE INDEX idx_signal_profiles_producer_key ON signal_profiles(producer, key)"
)


# ---------------------------------------------------------------------------
# 9. connector_state — last-sync timestamps, cursors, health per connector.
# ---------------------------------------------------------------------------
CREATE_CONNECTOR_STATE_SQL = """
CREATE TABLE connector_state (
    connector_id TEXT PRIMARY KEY,
    last_sync_at INTEGER,
    cursor TEXT,
    health_status TEXT NOT NULL DEFAULT 'unknown' CHECK (health_status IN (
        'healthy', 'degraded', 'unhealthy', 'unknown'
    )),
    last_error TEXT,
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""


# ---------------------------------------------------------------------------
# 10. preferences — user settings (quiet hours, autonomy, verbosity, tone,
#     Fernet-encrypted creds). Encrypted values set `encrypted=1`.
# ---------------------------------------------------------------------------
CREATE_PREFERENCES_SQL = """
CREATE TABLE preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    encrypted INTEGER NOT NULL DEFAULT 0 CHECK (encrypted IN (0, 1)),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""


# ---------------------------------------------------------------------------
# 11. rules — deterministic automation rules. Engine runs in Phase 1; UI hidden.
# ---------------------------------------------------------------------------
CREATE_RULES_SQL = """
CREATE TABLE rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    trigger TEXT NOT NULL,
    condition TEXT NOT NULL,
    action TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    priority INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""


# ---------------------------------------------------------------------------
# 12. semantic_facts — confirm/deny-gated high-level facts. Never auto-surfaced.
# ---------------------------------------------------------------------------
CREATE_SEMANTIC_FACTS_SQL = """
CREATE TABLE semantic_facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'confirmed', 'denied'
    )),
    evidence TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""


# ---------------------------------------------------------------------------
# 13. schema_version — single-row migration marker.
# ---------------------------------------------------------------------------
CREATE_SCHEMA_VERSION_SQL = """
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""


TABLE_DDL: tuple[tuple[str, str], ...] = (
    ("events", CREATE_EVENTS_SQL),
    ("event_tags", CREATE_EVENT_TAGS_SQL),
    ("entities", CREATE_ENTITIES_SQL),
    ("moments", CREATE_MOMENTS_SQL),
    ("moment_state_history", CREATE_MOMENT_STATE_HISTORY_SQL),
    ("outbox", CREATE_OUTBOX_SQL),
    ("feedback_weights", CREATE_FEEDBACK_WEIGHTS_SQL),
    ("signal_profiles", CREATE_SIGNAL_PROFILES_SQL),
    ("connector_state", CREATE_CONNECTOR_STATE_SQL),
    ("preferences", CREATE_PREFERENCES_SQL),
    ("rules", CREATE_RULES_SQL),
    ("semantic_facts", CREATE_SEMANTIC_FACTS_SQL),
    ("schema_version", CREATE_SCHEMA_VERSION_SQL),
)


INDEX_DDL: tuple[tuple[str, str], ...] = (
    ("idx_events_timestamp", CREATE_EVENTS_TIMESTAMP_INDEX_SQL),
    ("idx_events_source_timestamp", CREATE_EVENTS_SOURCE_TIMESTAMP_INDEX_SQL),
    ("idx_event_tags_event_id", CREATE_EVENT_TAGS_EVENT_ID_INDEX_SQL),
    ("idx_event_tags_tag", CREATE_EVENT_TAGS_TAG_INDEX_SQL),
    ("idx_entities_kind_name", CREATE_ENTITIES_KIND_NAME_INDEX_SQL),
    ("idx_moments_state_scheduled", CREATE_MOMENTS_STATE_SCHEDULED_INDEX_SQL),
    (
        "idx_moment_state_history_moment_ts",
        CREATE_MOMENT_STATE_HISTORY_MOMENT_TS_INDEX_SQL,
    ),
    (
        "idx_outbox_state_notbefore_created",
        CREATE_OUTBOX_STATE_NOTBEFORE_CREATED_INDEX_SQL,
    ),
    (
        "idx_signal_profiles_producer_key",
        CREATE_SIGNAL_PROFILES_PRODUCER_KEY_INDEX_SQL,
    ),
)


def get_all_ddl() -> list[str]:
    """Return every CREATE statement in the order it must be executed.

    Tables first (FK dependencies honored by ordering — `event_tags` after
    `events`, `moment_state_history` after `moments`), then all named indexes.
    """
    return [ddl for _, ddl in TABLE_DDL] + [ddl for _, ddl in INDEX_DDL]


def get_table_names() -> list[str]:
    """Return the 13 table names in declaration order."""
    return [name for name, _ in TABLE_DDL]


def get_index_names() -> list[str]:
    """Return every named index declared by the schema."""
    return [name for name, _ in INDEX_DDL]
