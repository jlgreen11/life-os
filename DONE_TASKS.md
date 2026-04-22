# DONE_TASKS — Life OS v2 Rewrite

> Append-only log (most recent first). Autonomous agent prepends here after
> moving an item out of NEXT_TASKS.md.
>
> Format:
> ```
> - [x] <task title> — SHA `<short>` · <YYYY-MM-DD HH:MM> · <one-line outcome>
> ```

---

<!-- Agent will prepend below this line -->

- [x] **Scaffold `core/moment/` package** — SHA `241a9dc` · 2026-04-22 · added `core/__init__.py`, `core/moment/__init__.py`, and `core/moment/types.py` with MomentState/InsightType/ActionKind as `StrEnum` (values match the `moments` table CHECK constraints in `storage/schema.py`), plus `Action`, `ContextTrigger`, `StateHistoryEntry`, and `Moment` dataclasses. Every enum and dataclass carries a docstring citing the CEO plan. No business logic: state transitions and persistence are deferred to the next two tasks. 12 tests in `tests/core/moment/test_types.py` cover enum membership (values + names), `Moment` defaults, mutable-default isolation across instances, StateHistoryEntry defaults, full `dataclasses.asdict → json.dumps → json.loads` round-trip over every field (enums serialize to strings because StrEnum is a str subclass), and `core.moment` package-level re-exports. `ruff check` + `ruff format` clean.

- [x] **Write `scripts/migrate_v1_to_v2.py` (dry-run only)** — SHA `b17cefd` · 2026-04-22 · added `scripts/migrate_v1_to_v2.py` that reads v1 DBs with `mode=ro` and writes a fresh `./data/lifeos_v2_dryrun.db` built from `storage/schema.py`. Translations: events (ISO→unix ts coercion, drop `embedding_id`), entities (contacts/places/subscriptions → `entities.kind`), state.db tasks → synthetic `legacy_task` Moments + `moment_state_history` rows, signal_profiles (DROP mood/decision/expertise/values + unknown types; KEEP cadence/relationship/temporal/spatial/comm_template/routine), preferences passthrough. Every decision logged; row-count invariants verified post-write; `MigrationReport` returned to caller. Notification feedback SKIPPED — v2 schema has no `feedback_events` table (NOTE left above the task slot in NEXT_TASKS). 7 tests in `tests/scripts/test_migrate_v1_to_v2.py` synthesize v1 fixtures (10 events, 3 entities, 2 tasks, 4 profiles, 2 prefs, 3 feedback rows) and assert: per-table row counts, no dropped profile types leak into output, full v2 schema materialized, legacy Moments have `suggested` state + history row, timestamps coerced to INTEGER unix, overwrite guard, missing-source-DB handling. `ruff check` + `ruff format` clean.

- [x] **Finalize schema DDL as code** — SHA `b8f2abb` · 2026-04-22 · added `storage/schema.py` with all 13 CREATE TABLE DDLs (events, event_tags, entities, moments, moment_state_history, outbox, feedback_weights, signal_profiles, connector_state, preferences, rules, semantic_facts, schema_version) + 9 named indexes matching the v2 plan § "13-table schema". `SCHEMA_VERSION = 1`. `get_all_ddl()` returns tables-then-indexes in FK-safe order; `get_table_names()` + `get_index_names()` introspection helpers. 14 tests in `tests/storage/test_schema.py` cover: DDL executes on fresh SQLite, re-run raises (create-once), all 13 tables materialize, all 9 named indexes materialize, moments UNIQUE(source_insight_type, evidence_hash), outbox UNIQUE(event_id, subject), moment state CHECK, moment_state_history FK, event_tags FK cascade, entities kind CHECK, outbox state CHECK, schema_version insert, DDL ordering. `ruff check` + `ruff format` clean.

- [x] **Ollama latency baseline** — SHA `19a325d` · 2026-04-21 · added `scripts/measure_ollama_budget.py` (5 ops × N=10, p50/p95/p99 + token counts, embedding-based semantic search, graceful skip when Ollama unreachable) + stub `docs/plans/2026-04-22-ollama-baseline.md` (marked NOT RUN; real measurement deferred to Mac Mini where mistral + nomic-embed-text are installed). 15 smoke tests in `tests/scripts/test_measure_ollama_budget.py` cover pure helpers. `ruff check` clean.

- [x] **Promote engineering plan into the repo** — SHA `2caca68` · 2026-04-21 · wrote `docs/plans/2026-04-21-v2-rewrite-plan.md` (week-by-week sequence, 14-endpoint API contract + iOS compat shim, 13-table inventory w/ indexes + invariants, outbox pattern spec, Moment state-transition table). Links back to CEO plan.

- [x] **Profile the 5 existing SQLite DBs** — SHA `1f52f12` · 2026-04-21 · added `scripts/profile_v1_dbs.py` (read-only, `mode=ro`, FK graph, top-5, events by source) + stub `MIGRATION_PROFILE.md` (no local v1 DBs on this machine; operator runs on Mac Mini).
