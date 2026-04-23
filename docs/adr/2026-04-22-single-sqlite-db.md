# ADR 005: Consolidate 5 SQLite DBs into one `lifeos.db`

- **Status:** Accepted (backfilled)
- **Authored:** 2026-04-22
- **Decision date:** 2026-04-21
- **Author:** autonomous v2-rewrite agent (iter 16)

## Context

v1 persistence was split across five SQLite files plus one LanceDB
directory:

| v1 file             | Owner service(s)                            |
|---------------------|---------------------------------------------|
| `events.db`         | connectors, event_store                     |
| `entities.db`       | entity_extractor, relationship_service      |
| `predictions.db`    | prediction_engine, notification_manager     |
| `preferences.db`    | settings, feedback_collector (feedback_log) |
| `context.db`        | iOS context service, spatial producer       |
| `lifeos.lance/`     | vector index for semantic search            |

The split happened organically — each service added the DB it needed
— with two consequences:

1. **No cross-table joins.** Answering "show me every Moment sourced
   from events tagged with contact X" required application-level joins
   across two SQLite connections, with all the O(N) fetch + merge
   waste that implies.
2. **Five schemas to migrate.** Any schema change touching more than
   one domain required coordinated migrations across files, with no
   atomic guarantee between them. The v1 migration script for
   adding `feedback_log.response_latency_seconds` landed in two
   separate commits that briefly left the schemas inconsistent.
3. **Five backup / restore seams.** The v1 snapshot procedure
   (runbook in `docs/archive/v1/`) had to checkpoint five WAL files in
   sequence; a failure partway left an inconsistent snapshot.
4. **Five `PRAGMA` configurations to keep in sync.** WAL, synchronous,
   journal mode — each file's setup drifted over time.

The CEO plan's single-user scope and the observed v1 row counts
(~200 MB total across the five files after 6 months) make the split
unjustified on any performance grounds.

## Consequences (pre-decision sizing)

The eng plan budget (§ "13-table schema") sized the combined `lifeos.db`
at ≤100 MB for steady-state Phase 1, based on the v1 row volumes minus
the soft-insight data being dropped under ADR 003. SQLite is
comfortable well past this; the largest known well-behaved SQLite DBs
are in the multi-TB range. 100 MB is not a constraint.

## Decision

**v2 persists to a single SQLite file `data/lifeos.db` holding all 13
tables (now 14 after ADR 001). LanceDB stays as a separate directory
`data/lifeos.lance/` — not consolidated into SQLite.**

Concretely:

- `storage/schema.py` defines every table and every index in one
  place. `SCHEMA_VERSION` is a single integer in one
  `schema_version` table.
- Connection is `sqlite3.connect("data/lifeos.db")` with
  `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON`
  applied once at startup.
- The migration script (`scripts/migrate_v1_to_v2.py`) reads all
  five v1 files, translates, and writes a single v2 file.
- Backup / restore is a single `cp data/lifeos.db* backup/` (with WAL
  checkpoint first) plus the LanceDB directory tree.
- `PRAGMA integrity_check` runs periodically (Week-2 WAL integrity
  loop called out in the eng plan) on one file.

**LanceDB is *not* consolidated** because:

- SQLite does not ship a native vector type; forcing vectors into
  BLOB would give up all Lance's ANN query affordances.
- LanceDB's storage format is columnar and mmap-friendly in a way
  SQLite's row pages are not; the performance gap for vector search
  is large.
- The migration surface is smaller if we keep the vector store
  independent — a future vector-store swap (e.g., to `sqlite-vss` if
  it matures) is a local change, not a schema reshape.

## Consequences

### Positive

- **One transaction boundary.** "Write moment + append state history +
  enqueue outbox" is atomic within the same file. This is the
  foundation ADR 004 relies on.
- **Cross-table joins.** "Every Moment surfaced for events from
  contact X" is one SQL query across `moments` / `events` /
  `event_tags` / `entities` in a single connection. v1 could not
  write this query.
- **Single snapshot file.** Backups are one checkpointed WAL file +
  the LanceDB directory. The cutover runbook has one DB file to
  stop/swap/restart per rollback.
- **Fewer PRAGMA / config drift surfaces.** One file → one set of
  durability settings.
- **Schema evolution is atomic.** A single migration file executes
  in a single transaction; inconsistent intermediate states are
  impossible.

### Negative

- **All tables share one WAL.** A long write on any table contends
  for the same WAL as every other table. For v1's traffic this is
  not a concern (contention was never observed in v1 per-file); for
  future high-ingest workloads (e.g., dense iOS-context stream)
  it may surface. Mitigation is partitioning at read time, not
  files.
- **Single-file blast radius.** Corruption of `lifeos.db` loses
  everything except vectors. v1 could in principle recover four of
  five domains if one DB corrupted. In practice v1 had no
  corruption incidents; WAL + synchronous=NORMAL + integrity_check
  scans are a reasonable mitigation.
- **Schema file grows.** `storage/schema.py` holds 14 tables + their
  indexes in one module. Careful organization (section comments,
  `CREATE TABLE IF NOT EXISTS` grouped by domain) keeps this
  navigable; a test asserts every table name in `__all__`.
- **LanceDB stays separate.** "One DB file" is slightly
  aspirational; the real answer is "one SQLite file + one LanceDB
  tree". Operational docs must be clear about both.

## Alternatives considered

1. **Keep 5 files, make them consistent via coordinated PRAGMAs.**
   *Rejected* — does not address cross-table joins or atomic
   state+outbox writes (ADR 004's foundation). Strictly less than
   consolidation.

2. **Consolidate everything including vectors into SQLite via a
   BLOB column + app-side ANN index.**
   *Rejected* — ceding LanceDB's ANN performance for a "one file"
   bullet point is a bad trade. The vector search latency matters
   for the semantic-fact confirm/deny flow; we have measured
   LanceDB is 10-50x faster than BLOB + numpy scans at realistic
   sizes.

3. **Move to PostgreSQL.**
   *Rejected* — adds an external process (see ADR 004 on broker
   ops burden), needs a user, needs restart orchestration. SQLite
   fits the single-user deployment model.

## Follow-up

- Integrity check + alert: eng plan Week-2 task (`PRAGMA integrity_check`
  run on a schedule; failures enqueue an outbox alert). Tracked in
  NEXT_TASKS for when observability work resumes.
- Backup rotation: not part of v2 scope; operator responsibility per
  cutover runbook.
- Phase 2 family-share: if we ever need per-user isolation at the
  DB layer, the answer is one DB file per user (same schema, same
  code path), not re-splintering within a file.
