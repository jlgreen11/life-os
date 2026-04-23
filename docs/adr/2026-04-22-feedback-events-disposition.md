# ADR 001: v1 `feedback_log` disposition in v2

- **Status:** Accepted
- **Date:** 2026-04-22
- **Author:** autonomous v2-rewrite agent (iteration 15)
- **Supersedes:** the "skip `feedback_log` at migration time" placeholder in
  `scripts/migrate_v1_to_v2.py` noted at
  `scripts/migrate_v1_to_v2.py:24-27` and `NEXT_TASKS.md` § Category B.

## Context

The v1 production schema includes a `feedback_log` table (in
`preferences.db`) that captures every interaction with a notification /
predicted action:

```sql
CREATE TABLE feedback_log (
    id                       TEXT PRIMARY KEY,
    timestamp                TEXT,            -- ISO-8601 UTC
    action_id                TEXT,            -- v1 notification / prediction id
    action_type              TEXT,            -- 'notification' | 'draft' | 'explicit' | 'fact_correction' | …
    feedback_type            TEXT,            -- 'acted_on' | 'dismissed' | 'expired' | 'corrected' | …
    response_latency_seconds REAL,            -- how long the user took to react
    context                  TEXT,            -- JSON blob (domain, prediction scores, etc.)
    mood_at_time             TEXT,            -- v1 mood inference snapshot
    notes                    TEXT
);
```

Roughly 30+ v1 tests exercise it, and live services (`feedback_collector`,
`prediction_engine`, `notification_manager`) both write and read rows —
the v1 reaction-prediction pipeline uses `feedback_log` to suppress
domains the user has repeatedly dismissed.

The v2 migration script (`scripts/migrate_v1_to_v2.py`) currently
**skips** these rows: it reads the source table only to COUNT them, logs
a warning, and writes nothing to the v2 database. The placeholder was
intentional — at migration-script authoring time the v2 schema did not
define a target and the engineering team had not decided whether
historical feedback should survive the cutover.

## Decision

**Add a `feedback_events` table to the v2 schema and migrate every v1
`feedback_log` row into it.**

- **Table:** `feedback_events` (plural to match v2's append-only event
  log convention — cf. `events`, `moment_state_history`).
- **Shape:** verbatim of the v1 columns *minus* `mood_at_time` (killed
  per CEO plan § "Killed from v1: mood inference"), plus a `source`
  column tagging the row's origin (`v1_migration` vs `v2`) and a
  `created_at` audit stamp.
- **Timestamp:** stored as INTEGER unix-seconds (v2 convention),
  converted from v1 ISO-8601 by the migration. NULLs in v1 fall back to
  `created_at`.
- **Repository:** append-only. No UPDATE, no DELETE (except future
  retention purge by age, mirroring `outbox.purge_done_older_than`).
- **Migration:** `migrate_notification_feedback` replaces
  `skip_notification_feedback`. The migration report surfaces
  `notification_feedback: TableCounts(source, translated, dropped)`
  instead of the previous scalar `notification_feedback_skipped`.

### Why this shape

| Concern | Resolution |
|---|---|
| **Preserves reaction-prediction training signal.** The v2 producers will eventually learn from "user ignored every `draft` suggestion for domain X" — the raw history is irreplaceable. | Keep the columns v1 code actually read (`action_id`, `action_type`, `feedback_type`, `context`). |
| **Drops mood.** CEO plan explicitly kills mood inference. Carrying `mood_at_time` forward would re-seed a dead primitive. | Omit the column. Fail-loud if a caller references it (it is simply not in the schema). |
| **Append-only.** Like `events` and `moment_state_history`, feedback is a log of user decisions. A row corrected after the fact is a *new* row, not an UPDATE. | No UPDATE / DELETE paths in the repo. |
| **Source tagging.** Future forensic work ("did this signal come from legacy v1 or native v2 producers?") needs a fast filter. | `source` CHECK-constrained to `v1_migration` | `v2`; index covers it implicitly via table scans — the table is small by design. |
| **Index.** The common queries are "how often was `action_id=X` `dismissed`?" (suppression) and "N most recent events". | Composite index on `(action_id, feedback_type)` for suppression; ordering by `ts` is well-served by the natural key scan until volume justifies more. |

### Non-goals

- Not re-implementing v1 `feedback_collector` semantics. The v2
  acceptance path (`MomentEngine`, feedback-weight EWMA in
  `feedback_weights`) remains the primary feedback signal. v2 may
  eventually write structured rows to `feedback_events` on accept /
  dismiss / undo; that is out of scope for this ADR.
- Not porting v1 `mood_at_time`.

## Consequences

### Positive

- Zero data loss at cutover; all historical reaction signal survives.
- Unblocks a later analytics task that mines the log for
  never-surfaced v1 dismissals.
- Matches v2's "append-only event tables" convention across `events`,
  `moment_state_history`, and now `feedback_events`.

### Negative

- Schema grows from 13 → 14 tables. Updated `tests/storage/test_schema.py`
  expectations. Updated `SCHEMA_VERSION` bookkeeping deferred (still
  version 1; v2 has not shipped, so the pre-cutover schema is
  redefined rather than migrated).
- Migration output size grows by roughly (v1 `feedback_log` row count)
  rows. At ~tens of thousands of rows on the live Mac Mini, footprint
  stays well below the 100 MB plan-budget for `lifeos.db`.
- Adds one more v1-only table that the migration must translate —
  keeps the migration script's surface area finite and reviewable.

## Alternatives considered

1. **Keep skipping (status quo).**
   *Rejected* — silently drops months of labeled training data the user
   spent effort producing. Fail-loud per v2 principle: if data is
   valuable, migrate it; if it's not, mention explicitly in this ADR
   why (see `mood_at_time`).

2. **Archive `preferences.db.feedback_log` verbatim to
   `data/archive/feedback_log_v1.db`.**
   *Rejected* — accessing a v1-shape sidecar DB from v2 code
   re-introduces the 5-DB splintering that v2 consolidated away.
   If we want the data, we want it indexed alongside every other v2
   signal in `lifeos.db`.

3. **Port to `events` as an event `type='feedback'`.**
   *Rejected* — `events` is the connector-sourced signal stream;
   mixing first-party user-decision records with that stream would
   conflate "what happened in the world" with "what the user told us
   about a prediction". Different lifecycles, different retention
   policies.

## Follow-up

- A future task should write to `feedback_events` from the v2 accept /
  dismiss / undo path so v2-native decisions accrue in the same table
  as v1-migrated ones.
- Retention policy (TTL / purge) is not defined; defer until volume
  justifies it. The `outbox.purge_done_older_than` precedent is available
  if we want mirror semantics.
