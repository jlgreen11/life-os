# Life OS v2 Rewrite — Engineering Plan

> Condensed engineering view of the CEO plan at
> `~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md`.
> This file exists to keep the execution contract in the repo. The CEO plan
> remains authoritative for vision, scope decisions, and acceptance KPIs; this
> file is authoritative for the engineering interfaces called out below.
>
> Related in-repo anchors:
> - `DESIGN.md` — tokens, 4-tab IA, Moment card states
> - `MIGRATION_PROFILE.md` — v1 DB profile output (populated on the Mac Mini)
> - `NEXT_TASKS.md` / `DONE_TASKS.md` — ordered execution queue

---

## Week-by-week sequence (p90 = 16 weeks; hard cap 18)

| Week | Focus | Key outputs |
|------|-------|-------------|
| 0 | Prework | Triage 12 open improvement PRs · pause CI agent · profile v1 DBs → `MIGRATION_PROFILE.md` · back up v1 data + LanceDB · branch `v2-rewrite` · promote eng plan (this file) · freeze v1 schema |
| 1 | Measurement + schema DDL + migration dry-run | `scripts/measure_ollama_budget.py` + `docs/plans/2026-04-22-ollama-baseline.md` · `storage/schema.py` (13 DDLs + indexes + `SCHEMA_VERSION=1`) · `scripts/migrate_v1_to_v2.py` (dry-run only) + synthetic v1 fixtures |
| 2 | Core Moment engine scaffolding | `core/moment/types.py` · `core/moment/state.py` (state machine w/ legal-transition enforcement) · `storage/repos/moments.py` (MomentRepository) · WAL integrity-check loop (gap called out in eng review) |
| 3 | Outbox + scheduler | `storage/repos/outbox.py` · `core/moment/scheduler.py` (asyncio wall-clock firing + boot recovery) |
| 4 | Insight producers (batch 1) | cadence + relationship refactored as Moment producers; golden-dataset replay harness stub |
| 5 | Insight producers (batch 2) | temporal + spatial + comm-template + routine producers |
| 6 | API (FastAPI) | 14 REST + 1 WS endpoints (see contract table); iOS compat shim (8 preserve + 4 stub) |
| 7 | Web UI shell | HTMX + Tailwind + modular Jinja; delete `web/template.py`; Now tab wired to `/api/now` |
| 8 | Web UI — You / People / Settings | Remaining tabs; HTMX swaps; `/ws` Moment push |
| 9 | Briefing + draft-reply AI flows | Ollama budgets honored; PII-shielded cloud fallback |
| 10 | Observability + migration rehearsal | `/health` · `/metrics` · `lifeos-report` CLI · 3 migration dry-runs (row diff, FK integrity, vector integrity) |
| 11 | Golden-dataset regression pass | Replay 30 days of v1 events through v2; assert Moment parity ±10% |
| 12 | Migration cutover | Stop v1 · snapshot v1 DBs + LanceDB · run migration · bring up v2 · verify /health · end-to-end Moment fire |
| 13 | Acceptance loop instrumentation | Per-type accept-rate dashboard; weekly digest on You tab + optional daily email |
| 14 | Recalibration window | Review observed accept-rate distributions; adjust 60/40/20 thresholds before any kill/demote fires |
| 15 | Phase 1 exit criterion checkpoint | ≥3 insight types at ≥60% for 2 consecutive weeks w/ ≥20 Moments/type/week |
| 16 | Buffer + polish | axe-core CI smoke test; error-budget reporting |

**Exit criterion (single authoritative wording):** at least 3 insight types at ≥60% accept rate for 2 consecutive weeks, with ≥20 Moments per type per week.

**Hard cap:** 18 weeks. If acceptance KPIs have not converged by then, pivot — do not extend.

---

## 13-table schema (SQLite, single `lifeos.db` file + separate `lifeos.lance/`)

Full DDL lives in `storage/schema.py` (Week 1 task). Table inventory:

| # | Table | Purpose |
|---|-------|---------|
| 1 | `events` | Immutable event log (append-only). All data from connectors lands here. |
| 2 | `event_tags` | N-to-N tags on events (source classification, topic, priority). |
| 3 | `entities` | Contacts, places, subscriptions, topics. `kind` enum disambiguates. |
| 4 | `moments` | First-class Moment primitive. One row per proposed action. |
| 5 | `moment_state_history` | Audit log of every state transition (from, to, ts, annotation). |
| 6 | `outbox` | Transactional outbox for side-effecting events (send_message, etc.). |
| 7 | `feedback_weights` | EWMA per-insight-type accept rate (α=0.1, half-life ≈ 7 decisions). |
| 8 | `signal_profiles` | Per-producer rolling profiles (cadence, relationship, temporal, spatial, comm-template, routine). |
| 9 | `connector_state` | Last-sync timestamps, cursors, health status per connector. |
| 10 | `preferences` | User settings (quiet hours, autonomy, verbosity, tone, Fernet-encrypted creds). |
| 11 | `rules` | Deterministic automation rules (engine runs; UI hidden in Phase 1). |
| 12 | `semantic_facts` | Confirm/deny-gated high-level facts. Never auto-surfaced. |
| 13 | `schema_version` | Single-row migration marker; holds `SCHEMA_VERSION` constant. |

**Indexes** (minimum set, finalized in `storage/schema.py`):
- `events(timestamp)`, `events(source, timestamp)`
- `event_tags(event_id)`, `event_tags(tag)`
- `entities(kind, name)`
- `moments(state, scheduled_for)`, `moments(source_insight_type, evidence_hash)` UNIQUE
- `moment_state_history(moment_id, ts)`
- `outbox(state, created_at)`, `outbox(event_id, subject)` UNIQUE
- `signal_profiles(producer, key)`
- `connector_state(connector_id)` UNIQUE

**Invariants (enforced at DB + repo layer):**
- FKs ON. WAL mode. `synchronous=NORMAL`.
- `events` is append-only — no UPDATE / DELETE at repo layer.
- Moment uniqueness: `(source_insight_type, evidence_hash)` — idempotent producers.
- `moment_state_history` rows are only appended via `MomentRepository.transition(...)` which validates legality in the same SQLite transaction.

**LanceDB** stays as a separate index directory `./data/lifeos.lance/` — not consolidated into SQLite.

---

## 14-endpoint API contract (+1 WebSocket)

All endpoints are JSON. Authentication is Tailscale-only in Phase 1 (single user). The 14 REST endpoints are locked by count; the iOS compat shim (8 preserve + 4 stub) is in addition. Final verb/path binding is finalized when `api/` lands in Week 6; the contract table below captures the committed shape.

### Primary web app (14 REST + 1 WS)

| # | Method | Path | Purpose |
|---|--------|------|---------|
| 1 | GET | `/api/now` | Now tab payload: pending · scheduled · done-today |
| 2 | GET | `/api/you` | Mirror payload: when-at-your-best, portrait, routines, drifting |
| 3 | GET | `/api/people` | People index (contact list with last-contact + cadence deviation) |
| 4 | GET | `/api/people/{contact_id}` | Per-contact dossier |
| 5 | POST | `/api/moments/{id}/accept` | Transition suggested → accepted |
| 6 | POST | `/api/moments/{id}/dismiss` | Transition suggested → dismissed (terminal) |
| 7 | POST | `/api/moments/{id}/snooze` | Transition suggested → snoozed (requires `snooze_until`) |
| 8 | POST | `/api/moments/{id}/edit` | Update `proposed_action.params` (e.g. draft body) in place |
| 9 | GET | `/api/health` | Connector status · DB last-write · scheduler heartbeat · producer-per-type activity · pending count |
| 10 | GET | `/api/metrics` | Prometheus-format counters |
| 11 | GET | `/api/connectors` | List connectors (4 active + 8 dormant) with state |
| 12 | POST | `/api/connectors/{id}/test` | Dry-run sync; returns structured error or success |
| 13 | GET/PUT | `/api/preferences` | Read + update user preferences |
| 14 | GET | `/api/weekly-report` | Acceptance-rate digest for the You tab + CLI `lifeos-report` |
| WS | WS | `/ws` | Per-tab channel Moment state push (HTMX `hx-ws`) |

### iOS compat shim (8 preserve + 4 stub) — enumerated from `ios/LifeOS/Services/APIClient.swift`

| Method | Path | Status |
|--------|------|--------|
| POST | `/api/context/event` | PRESERVE (spatial signal lifeblood) |
| POST | `/api/context/batch` | PRESERVE |
| GET | `/api/context/summary` | PRESERVE |
| GET | `/api/status` | PRESERVE |
| GET | `/api/briefing` | PRESERVE (returns latest briefing Moment text) |
| POST | `/api/feedback` | PRESERVE (rewired through Moment feedback) |
| GET/PUT | `/api/preferences` | PRESERVE (shared with web contract row 13) |
| GET | `/api/tasks` | PRESERVE (returns Moments shaped as v1 tasks during iOS freeze) |
| (stub) | `/api/predictions` | STUB (empty list; iOS expects field, Phase 2 rebuild removes) |
| (stub) | `/api/notifications` | STUB (empty list) |
| (stub) | `/api/user-model` | STUB (schema-compatible empty payload) |
| (stub) | `/api/insights` | STUB |

---

## Outbox pattern spec

Transactional outbox for side-effecting events (send_message, create_calendar_entry, etc.). Ensures "insert Moment state-change + publish effect" is atomic without distributed-transaction complexity.

**Shape (captured in DDL under `outbox`):**

```
outbox {
  id: uuid PK
  event_id: uuid         # logical key for idempotency
  subject: text          # e.g. "moment.accepted", "send_message.v1"
  payload: json
  state: enum(pending, in_progress, done, failed, dead)
  retry_count: int default 0
  last_error: text null
  created_at: ts
  updated_at: ts
  claimed_at: ts null
}
UNIQUE(event_id, subject)        -- enqueue idempotency
INDEX(state, created_at)         -- claim_batch ordering
```

**Pattern properties:**
- **Enqueue is idempotent** on `(event_id, subject)`. Duplicate enqueues are a no-op.
- **Claim is atomic.** `claim_batch(limit=10)` selects `state='pending' ORDER BY created_at LIMIT ? FOR UPDATE` equivalent (SQLite BEGIN IMMEDIATE), then UPDATEs to `in_progress` + stamps `claimed_at`. Two callers cannot claim the same row.
- **Complete / fail.** Successful side-effect → `state='done'`. Failure increments `retry_count`, sets `last_error`, re-queues as `pending`. When `retry_count >= 5`, transitions to `dead` (dead-letter; surfaced via `/health`).
- **Boot recovery.** On service start, `requeue_in_progress_on_boot()` flips any `state='in_progress'` rows back to `pending` — covers the case where the process died mid-delivery.
- **Retention.** Rows in `state IN ('done', 'dead')` older than **30 days** are garbage-collected by a daily job.
- **Durability settings.** SQLite `journal_mode=WAL`, `synchronous=NORMAL`. Outbox writes are inside the same DB file; a single transaction covers both the Moment state transition and the outbox enqueue.

**Non-goals (Phase 1):** distributed queues (Kafka/NATS), cross-process claim protocols. Single-writer single-reader per worker is sufficient for a single-user system.

---

## Moment primitive — state-transition table

Copied from the CEO plan; authoritative source for `core/moment/state.py::_LEGAL_TRANSITIONS`. Transitions not in this table **must raise `IllegalTransition`**.

| From         | → suggested    | → accepted | → dismissed | → snoozed | → done | → expired |
|--------------|:--------------:|:----------:|:-----------:|:---------:|:------:|:---------:|
| *(create)*   | ✓              |            |             |           |        |           |
| suggested    |                | ✓          | ✓           | ✓         |        | ✓         |
| accepted     |                |            |             |           | ✓      |           |
| dismissed    |                |            |             |           |        |           |
| snoozed      | ✓ (re-fire)    |            |             |           |        | ✓         |
| done         |                |            |             |           |        |           |
| expired      |                |            |             |           |        |           |

**Terminal states:** `dismissed`, `done`, `expired` — no outbound transitions.

**Snooze semantics:** `snoozed` re-enters `suggested` at `snooze_until`, subject to `expires_at` still being in the future. UI should refuse snooze targets past `expires_at`; server-side violation is logged and coerced to `expired`.

**TTL:** `expires_at` default = `created_at + 72h` unless the producer overrides. Tunable per insight_type via preferences.

**Boot recovery annotation:** Any Moment touched by the boot-recovery scan receives a `boot_recovery` entry in `moment_state_history` alongside the normal transition row.

---

## Producers retained (evidence-backed only)

`cadence`, `relationship`, `temporal`, `spatial`, `comm_template`, `routine`. Killed: mood, decision, expertise, values (see CEO plan § Killed Insights). Killed code archived under `deprecated/` for reference only; not imported by v2 runtime.

## Connectors

**Phase 1 active (4):** Proton Mail · iMessage · CalDAV · iOS context.
**Phase 1 dormant (8):** Signal · Gmail · Google Calendar · Google Contacts · Plaid · Home Assistant · browser-YouTube · browser-Reddit · browser-WhatsApp · browser-generic. (Disabled via config; code retained.)

## Non-negotiables

- Ollama budgets: `task_extraction` ≤ 2s foreground, `briefing` ≤ 20s async; baseline measurement is Week 1 task.
- Golden-dataset regression harness: 30-day v1 event replay; assert Moment parity ±10%.
- Error budget: ≤1 silent failure per week (Moment created but never surfaced).
- WAL integrity detection (`PRAGMA integrity_check` + alert via outbox on failure) — Week 2 addition flagged by eng review.
- Mood state tracked internally **never** exposed externally.
