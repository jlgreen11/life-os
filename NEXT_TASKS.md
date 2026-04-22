# NEXT_TASKS — Life OS v2 Rewrite

> Ordered queue. Autonomous agent takes the **top** unchecked `- [ ]` item.
> When done, move it to `DONE_TASKS.md` (prepend, most recent first) with
> commit SHA + date.
>
> One task per iteration. If a task is too big for one iteration's $10
> budget, split it into sub-items before starting.
>
> Reference: CEO plan at
> `~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md`

---

## Week 0 — prework (safe for agent)

_All Week 0 prework tasks complete. See DONE_TASKS.md._

## Week 1 — Ollama measurement + schema DDL + migration dry-run

- [ ] **Ollama latency baseline.** Write `scripts/measure_ollama_budget.py` that runs N=10 iterations of each of: briefing_synthesis (feed a 12-context-block synthesis prompt), task_extraction (feed 5 realistic email bodies), priority_classification (same), draft_reply (feed 3 contact profiles), semantic_search (embed a query + top-5 nearest neighbors). Report p50/p95/p99 latency + token counts per operation. Emit `docs/plans/2026-04-22-ollama-baseline.md`. If ollama is not running locally (`ollama list` returns error), emit the script but skip the measurement; leave a NOTE in NEXT_TASKS that baseline must be measured on the Mac Mini.

- [ ] **Finalize schema DDL as code.** Create `storage/schema.py` with the 13 table DDLs from the eng review: events, event_tags, entities, moments, moment_state_history, outbox, feedback_weights, signal_profiles, connector_state, preferences, rules, semantic_facts. Each table as a module-level string constant `CREATE_<NAME>_SQL`. A `SCHEMA_VERSION = 1` constant. A `get_all_ddl()` function returning the ordered list. Include all indexes. Write unit tests in `tests/storage/test_schema.py` that create an in-memory SQLite, execute every DDL, verify no errors, verify all expected tables/indexes exist, verify FKs are enforced. Use pytest + fixture `tmp_path`. Run `ruff format` before commit.

- [ ] **Write migrate_v1_to_v2.py (dry-run only, no production writes).** Create `scripts/migrate_v1_to_v2.py` that reads v1 DBs read-only (if they exist at `./data/`) and WRITES to a NEW empty `./data/lifeos.db` matching the v2 schema from `storage/schema.py`. Translations: v1 events → events (pass through); v1 entities.db contacts/places/subscriptions → entities rows (kind=contact/place/topic); v1 state.db tasks → synthetic Moments with source_insight_type='legacy_task'; v1 user_model.db signal_profiles → signal_profiles (translate profile_type name mapping); v1 preferences.db → preferences. Log every translation decision to stdout. Verify: row counts in v2 match expected transformations (events 1:1, entities 1:1, etc.). This is a DRY RUN — output goes to `./data/lifeos_v2_dryrun.db`, never touches live DBs. If no local v1 data exists, the script should run against synthetic fixtures in `tests/fixtures/v1_sample/` (create these too — tiny in-memory sample with 10 events, 3 entities, 2 profiles). Tests: round-trip on the fixtures. Run: `python -m pytest tests/scripts/test_migrate.py -v`.

## Week 2 — core/moment engine scaffolding

- [ ] **Scaffold core/moment/ package.** Create `core/__init__.py`, `core/moment/__init__.py`, `core/moment/types.py` (enums: MomentState, InsightType, ActionKind; dataclasses: Action, ContextTrigger, Moment). Every enum + dataclass has a docstring citing the CEO plan. No business logic yet — just typed definitions. Unit tests in `tests/core/moment/test_types.py` verifying each enum has the expected members.

- [ ] **State machine with legal-transition enforcement.** Add `core/moment/state.py` defining `_LEGAL_TRANSITIONS: dict[MomentState|None, set[MomentState]]` per the CEO plan state-transition table, plus `IllegalTransition(ValueError)` exception, plus `validate_transition(from_state, to_state) -> None` (raises IllegalTransition on failure). Tests in `tests/core/moment/test_state.py`: (1) table-driven test covering every legal combination (succeeds), (2) property-based test using `hypothesis` covering every (from, to) pair not in the legal set — must raise. Install hypothesis if not already a dependency; if install is denied, write the table-driven tests without hypothesis and leave a NOTE.

- [ ] **Moment repository (SQLite-backed).** Add `storage/repos/__init__.py` and `storage/repos/moments.py` with `MomentRepository` class that owns moments + moment_state_history tables. Methods: `create(moment) -> moment_id` (insert; unique(source_insight_type, evidence_hash) handled), `get(id) -> Moment`, `transition(id, new_state, annotation=None) -> Moment` (calls validate_transition, updates state, appends history row — all in one transaction), `list_pending(limit=20) -> list[Moment]`, `list_scheduled(horizon_seconds=86400, limit=10) -> list[Moment]`, `list_done_today(limit=10) -> list[Moment]`. Tests in `tests/storage/repos/test_moments.py`: create, get, transition happy path, transition illegal raises, idempotency on duplicate create (same evidence_hash + type), list queries with fixture data.

## Week 3 — outbox + scheduler

- [ ] **Outbox table + repository.** Add `storage/repos/outbox.py` with `OutboxRepository`. Methods: `enqueue(event_id, subject) -> outbox_id` (idempotent on event_id+subject pair), `claim_batch(limit=10) -> list[OutboxEntry]` (transitions pending→in_progress atomically), `complete(outbox_id)`, `fail(outbox_id, error_msg)` (increments retry_count; transitions to 'dead' if count >= 5), `requeue_in_progress_on_boot()` (for crash recovery — transitions any state='in_progress' rows back to 'pending'). Tests: enqueue idempotency, claim atomicity with concurrent callers (use threading), retry/dead logic, boot recovery.

- [ ] **Scheduler: wall-clock firing loop.** Add `core/moment/scheduler.py` with `Scheduler` class. `run_forever()` asyncio coroutine: every tick, `SELECT id FROM moments WHERE state IN ('suggested','snoozed') AND scheduled_for <= ? ORDER BY scheduled_for LIMIT 50`, then for each: fire (log + transition to 'suggested' if it was 'snoozed', or enqueue to outbox for delivery). Boot recovery: on `Scheduler.__init__`, scan for past-due Moments and fire them immediately. Tests with clock freeze via `freezegun` (install if missing; fall back to manual time injection if denied).

## When blocked / unclear

If a task above mentions something you can't verify (e.g., local `data/` not present, ollama not running, hypothesis not installed), don't guess. Leave a `<!-- NOTE: <reason> -->` comment above the task, commit the note, move on to the next unchecked task. Don't check the task off.

## Guardrails reminder (your system prompt enforces these, but as a reminder)

- Never `git checkout master`, never `git push`, never run `gh pr ...`
- Never write to `data/` (user's production DBs)
- Never edit `config/settings.yaml`
- Everything stays on `v2-rewrite` branch, committed locally only
- One task per iteration; if budget runs out, `V2_AUTONOMOUS_ITERATION_PARTIAL`
