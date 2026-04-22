# NEXT_TASKS — Life OS v2 Rewrite

> Ordered queue. Autonomous agent takes the **top** unchecked `- [ ]` item.
> When done, move it to `DONE_TASKS.md` (prepend, most recent first) with
> commit SHA + date.
>
> **Task sizing:** each item fits ≤ 1 iteration ($10 opus budget, ~15–25 min
> of Claude work). If a task runs out of budget, agent prints
> `V2_AUTONOMOUS_ITERATION_PARTIAL: <reason>` and leaves it checked out.
>
> **Source of truth:** CEO plan at
> `~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md`.
> Design tokens/IA in `DESIGN.md`. Engineering plan in `docs/plans/2026-04-21-v2-rewrite-plan.md`.
>
> **Agent rules:**
> - Never checkout master. Never push (orchestrator handles it). No PRs. No merges.
> - Never write to `data/` (production DBs) or edit `config/settings.yaml`.
> - Working tree clean at iteration end (agent commits everything it touches).
> - One task per iteration unless the task body says otherwise.
> - If blocked (missing data/dep, unresolved design), leave `<!-- NOTE: reason -->`
>   above the task, commit the note, move on. Don't guess architectural decisions.

---

## Week 0 — prework (complete, see DONE_TASKS.md)

## Week 1 — migration dry-run

<!-- NOTE (2026-04-22, follow-up to SHA of this commit): v2 schema has no `feedback_events` table (only `feedback_weights` EWMA). The migration script skips v1 `feedback_log` rows with a logged warning and surfaces the count in `MigrationReport.notification_feedback_skipped`. Before cutover, either (a) add `feedback_events` to `storage/schema.py` or (b) decide legacy feedback is intentionally not carried forward. Tracked as an open design decision on the engineering plan. -->

## Week 2 — Moment primitive (complete, see DONE_TASKS.md)

## Week 3 — outbox + scheduler + WAL integrity

- [ ] **Transactional outbox (`storage/repos/outbox.py`).** Per eng review §1b:
  - `enqueue(event_id, subject, *, conn=None) -> int`: idempotent on (event_id, subject); callable inside existing transaction for atomicity
  - `claim_batch(limit=10) -> list[OutboxEntry]`: atomic pending→in_progress via BEGIN IMMEDIATE
  - `complete(outbox_id)`: in_progress → done, set delivered_at
  - `fail(outbox_id, error_msg)`: retry_count++, transitions to 'dead' if >= 5
  - `requeue_in_progress_on_boot() -> int`: any 'in_progress' rows → 'pending' (crash recovery)
  - `purge_done_older_than(days=30) -> int`: bulk delete (daily maintenance)
  Tests: enqueue idempotency, concurrent claim_batch via threading (exactly one claim per row), retry→dead progression, boot recovery, retention purge.

- [ ] **Scheduler wall-clock firing loop (`core/moment/scheduler.py`).** `class Scheduler(moment_repo, outbox_repo, bus)`:
  - `async run_forever(tick_seconds=30)`: each tick, SELECT suggested+snoozed Moments with scheduled_for <= NOW; for each: if snoozed transition→suggested with annotation='scheduler_fire'; enqueue notification event to outbox; log fire latency
  - `async boot_recovery()`: past-due Moments — fire if expires_at > now (annotation='boot_recovery'), else transition → expired
  - `_matches_trigger(moment, event) -> bool`: parse ContextTrigger grammar; return bool
  Tests: past-due fires via boot_recovery; snoozed past wake-time transitions; expired is terminal; fire latency recorded. Use `time.monotonic()` injection (stdlib only; no freezegun).

## Week 4 — Producer base + 3 producers

- [ ] **Producer base class (`core/moment/producer.py`).**
  - `class Producer(ABC)` with `insight_type: InsightType` class attr
  - `async def observe(event: Event) -> list[Moment]`: 0–N candidate Moments per event
  - `@staticmethod def evidence_hash(event_ids: list[str]) -> str`: stable sorted hash for idempotency
  - Registry: `PRODUCERS: dict[InsightType, type[Producer]]` + `@register` decorator
  Tests: abstract non-instantiable, registry decorator works, evidence_hash deterministic across orderings.

- [ ] **Cadence producer (`producers/cadence.py`).** InsightType.CADENCE:
  - Observes `email.received` and `message.received`
  - Reads cadence profile from signal_profiles (scope_key=contact_id)
  - If `days_since_last_inbound > expected_cadence * 1.3` AND `count >= 5` historical → emit Moment:
    - insight: "{N} days since you've heard from {Name}. Usual cadence {X} days."
    - proposed_action: ActionKind.NUDGE, params {contact_id, channel}
    - confidence: `min(0.9, days_since / expected_cadence / 2)`
    - evidence: last 3 inbound events
    - Microcopy: "cadence dropped {N}% below {X}-day norm"
  Tests: no Moment with <5 history, fires at 1.3× expected, dedup via evidence_hash.

- [ ] **Relationship producer (`producers/relationship.py`).** InsightType.RELATIONSHIP:
  - Tracks reciprocity drift: if outbound/inbound ratio for a contact drops below 0.3 over last 4 weeks (previously > 0.5) → Moment
  - Insight: "You've been replying less to {Name}. Outbound dropped {X}%."
  - Action: ActionKind.NUDGE or ActionKind.DRAFT_MESSAGE
  Tests: 0 Moment if <20 interactions, fires on real drop, idempotent per week.

- [ ] **Temporal producer (`producers/temporal.py`).** InsightType.TEMPORAL:
  - Uses detected chronotype (from temporal signal profile) + time + calendar state
  - Emits Moment when: (a) user enters a historical high-focus window, OR (b) calendar has ≥ 60 min gap opening
  - Insight: "You have {X} min free. Historical focus pattern at this hour: {description}."
  Tests: no Moment if temporal profile <2 weeks of data, fires in known focus windows.

## Week 5 — 3 more producers

- [ ] **Spatial producer (`producers/spatial.py`).** InsightType.SPATIAL:
  - Observes `context.location.updated` events from iOS compat
  - Fires on arrival/departure to places in spatial signal profile
  - Insight examples: "You're at {Place}. Last time here you worked on {topic}." / "You've been at {Place} {X} min, avg {Y}."
  - Action: ActionKind.NOTE_OBSERVATION (read-only in Phase 1)
  Tests with fixture iOS context events; dedup within same location visit.

- [ ] **Comm-template producer (`producers/comm_template.py`).** InsightType.COMM_TEMPLATE:
  - Observes inbound email/message; reads per-contact comm template from signal_profiles
  - Week 5 scope: scaffold + stub draft generation (deterministic "Hi {name},"); mark `<!-- NOTE: AI engine integration deferred to Week 7 -->`
  - After Week 7's AI engine lands, this producer wires draft_reply() in
  Tests: producer returns stub for known contact, empty for unknown.

- [ ] **Routine producer (`producers/routine.py`).** InsightType.ROUTINE:
  - Observes event stream; triggers at routine-violation points (e.g., Sunday 5pm if user historically plans week — routine detector writes these into signal_profiles.routine)
  - Insight: "You usually {routine description}. Want to start now?"
  - Action: ActionKind.SET_REMINDER or ActionKind.NOTE_OBSERVATION
  Tests: fires at detected routine times only; evidence includes last 3 occurrences.

## Week 6 — feedback + wiring

- [ ] **Feedback-weight EWMA (`core/moment/feedback_weight.py`).** Per eng review §1d:
  - `class FeedbackWeightStore` (owns feedback_weights table)
  - `update(insight_type, moment_state)`: signal = 1.0 ACCEPTED / 0.0 DISMISSED / 0.5 SNOOZED / no-update EXPIRED|DONE; alpha=0.1; `w_new = alpha*signal + (1-alpha)*w_old`; sample_count++
  - `get(insight_type) -> (weight, sample_count)` — defaults (0.5, 0)
  - `get_threshold_for(insight_type) -> float`: `base (0.6) + (1.0 - weight)` — higher bar when user rejects a lot
  Tests: happy path, update sequence converges predictably, unknown type returns defaults.

- [ ] **MomentEngine wiring (`core/moment/engine.py`).**
  - `class MomentEngine(producers, moment_repo, feedback_weight_store)`
  - `async def on_event(event)`: for each producer → observe(event); filter by confidence >= threshold_for(insight_type); create via moment_repo
  Integration test: fixture 30-day event stream → verify Moments per producer land, dedup via evidence_hash.

## Week 7 — AI engine extraction

- [ ] **Extract AI engine (`ai/engine.py`).** Pull Ollama + Claude + routing out of v1's `services/ai_engine/engine.py` into fresh module:
  - `class AIEngine(ollama_url, ollama_model, cloud_api_key=None, use_cloud=False)`
  - `async briefing_synthesis(context) -> str`
  - `async task_extraction(event) -> list[dict]`
  - `async priority_classification(event) -> str`
  - `async draft_reply(contact_id, recent_messages, user_style) -> str`
  - `async semantic_search(query, k=5) -> list[SearchResult]`
  - Each method honors per-operation latency budget from CEO plan §1e; on exceed raises `AIBudgetExceeded`
  Tests: mock Ollama responses per method, verify `AIBudgetExceeded` on injected delay.

- [ ] **PII shield (`ai/pii.py`).** Port v1 `services/ai_engine/pii.py`:
  - `class PIIShield` with `redact(text) -> (redacted, mapping)` and `restore(text, mapping) -> text`
  - Redacts: email, phone, names (from entities table), addresses
  - Mapping is one-time; discarded after restore
  Tests: round-trip on samples; no PII leaks in redacted output.

- [ ] **Context assembly (`ai/context.py`).** Port v1 `services/ai_engine/context.py`:
  - `assemble_briefing_context(user_id, date) -> dict` with 11 sections: calendar, moments (was tasks), unread messages, completions, predictions, episodes, facts, insights, routines, habits, preferences
  - REMOVE all references to mood/decision/expertise/values (dropped in v2)
  Tests: returns dict with expected keys, empty-state returns empty lists (not None).

## Week 8 — API surface

- [ ] **API skeleton (`api/app.py` + `api/schemas.py`).** FastAPI factory + Pydantic request/response schemas:
  - `MomentOut`, `MomentListOut`, `MomentActionIn`
  - `YouOut`, `PeopleListOut`, `ContactDossierOut`
  - `ConnectorOut`, `ConnectorConfigIn`
  - `HealthOut`, `MetricsOut`
  No route logic yet. Tests: schemas round-trip, validation rejects malformed input.

- [ ] **`api/routes/now.py` (Now tab + moment actions).** 5 REST endpoints:
  - `GET /api/now` → `{pending: [], scheduled: [], done: []}` limits 20/10/10
  - `POST /api/moments/{id}/accept` → SUGGESTED → ACCEPTED, triggers action (stub in Week 11)
  - `POST /api/moments/{id}/dismiss` → DISMISSED + feedback weight update
  - `POST /api/moments/{id}/snooze` body {until_iso} → SNOOZED + weight update (0.5)
  - `POST /api/moments/{id}/edit` body {action_params} → updates proposed_action.params (stays SUGGESTED)
  Tests with TestClient: schema round-trips; 409 on invalid transitions; 404 on missing.

- [ ] **`api/routes/you.py` + `api/routes/people.py`.**
  - `GET /api/you` → self-portrait: when_at_best, how_you_write, your_routines, drifting
  - `GET /api/people` → paginated + search (q); YOU first entry; NEEDS_ATTENTION + ACTIVE_THIS_WEEK sections
  - `GET /api/people/{contact_id}` → full dossier: comm_template, cadence sparkline, recent_topics, predicted_next
  Tests: empty-state returns empty lists (never None); large-set pagination.

- [ ] **`api/routes/settings.py` + `api/routes/health.py`.**
  - `GET /api/connectors`: list + status + last_sync + last_error
  - `PATCH /api/connectors/{id}`: update config (Fernet creds never returned raw)
  - `POST /api/connectors/{id}/test`: dry-run sync
  - `GET /api/health`: deep-health multi-key (connectors, DB last-write, scheduler heartbeat, producer activity, pending count)
  - `GET /metrics`: Prometheus text + daily jsonl dump to `./data/metrics/metrics-YYYYMMDD.jsonl` for `lifeos-report` CLI
  Tests: Fernet creds never serialize; health returns multi-key response; /metrics parses with prometheus-client.

- [ ] **iOS compat shim (`api/routes/context.py` + legacy proxies).** Per eng review §1g. Preserve for the existing iOS app:
  - `POST /api/context/event` + `/batch` + `GET /api/context/summary` (context pipeline)
  - `GET /api/status` (smoke)
  - WebSocket `/ws` (Moment push)
  Proxy to v2:
  - `GET /api/briefing` → wraps briefing_synthesis + returns v1-shape
  - `POST /api/feedback` → writes to feedback_events
  - `POST /api/preferences` → writes to preferences
  Stub 501:
  - `POST /api/command`, `GET /api/notifications`, `GET/POST /api/tasks`, `POST /api/search`
  Tests: compat endpoints with realistic iOS payloads; stubs return 501 with helpful body.

## Week 9 — Web: base + Now tab

- [ ] **Web base template + design tokens (`web/templates/base.html`, `web/static/tokens.css`).** Implement DESIGN.md:
  - `tokens.css` — all CSS custom properties from DESIGN.md (color, typography, spacing, elevation, radius, motion)
  - `base.html` — HTMX + `hx-ws` extension, Tailwind CDN, SF font stack, Lucide icons helper, top nav Now/You/People/⚙ with active-tab underline, header with date/time
  - A11y: `<nav aria-label>`, `<main>`, skip-link, `prefers-reduced-motion` respected
  Render smoke test: template renders with fixture data, no Jinja errors, tokens load.

- [ ] **Now tab + Moment card primitive (`web/templates/now.html` + partials).**
  - Page sections: NOW (2–3 cards) / UP NEXT (compact list) / DONE TODAY (collapsed default)
  - `partials/moment_card.html`: insight (22pt display), evidence link (HTMX reveal), draft block (recessed bg), action buttons (one primary filled + 2–3 ghost)
  - "Why am I seeing this" microcopy under evidence link
  - Evidence reveal: `GET /api/moments/{id}/evidence` returns partial HTML (event excerpts)
  Tests: card renders with all states (default, draft-pending, expanded-evidence); manual a11y checklist from DESIGN.md.

- [ ] **HTMX wiring: accept / dismiss / snooze.** Each action button:
  - `<button hx-post="/api/moments/{id}/accept" hx-swap="outerHTML" hx-target="closest .moment-card">`
  - On success: card fades + server returns next pending Moment partial; swap replaces it
  - Snooze duration picker: chip row popover `[1h][3h][Tonight][Tomorrow][3d][Custom]`
  Tests: each action POSTs correctly; swap target updates; Undo toast appears 3s (vanilla JS, no framework).

## Week 10 — Web: You, People, Settings

- [ ] **You tab (`web/templates/you.html`).** Per DESIGN.md:
  - Header: "Observed {N} months · {M} interactions · confidence {X}%"
  - Sections: WHEN YOU'RE AT YOUR BEST (top 3 patterns), HOW YOU WRITE (3–6 per-audience summaries), YOUR ROUTINES (detected + "No routine detected yet" per empty pattern), DRIFTING (contact list with days-since vs usual)
  - NO mood bars, NO progress bars, NO pie charts
  Tests: empty-state per section <20 samples; rendering with fixture matches DESIGN.md wireframe.

- [ ] **People tab (`web/templates/people.html`) + per-contact dossier.**
  - List: YOU first; NEEDS ATTENTION; ACTIVE THIS WEEK
  - Right-aligned monospace stats
  - Search: HTMX debounced `hx-get="/api/people?q=..."` `hx-trigger="keyup changed delay:200ms"`
  - Per-contact (`contact_dossier.html`): matches DESIGN.md wireframe
  - NO avatars, plain text only
  Tests: search filters correctly; dossier renders known contact; empty state for new.

- [ ] **Settings tab (`web/templates/settings.html`).**
  - Connector list: status dot + last sync + enabled checkbox
  - Edit form opens in detail pane (not inline expansion)
  - Per-connector test button: dry-run sync + inline success/failure
  - Preferences: quiet hours, autonomy level, proactivity slider (all persist to preferences table)
  Tests: Fernet creds round-trip (no plaintext in response); preferences persist; test button state.

## Week 11 — Real-time + full flows

- [ ] **WebSocket push (`api/routes/websocket.py` + base wiring).**
  - Server broadcasts Moment state changes to connected clients
  - Client: `hx-ws="connect:/ws"` on base; server pushes partial HTML; HTMX swaps
  - On new pending Moment: insert at top of NOW with subtle accent bar for 3s (client animation)
  - On done: update DONE TODAY
  - Reconnect: exponential backoff; "Reconnecting..." pill in header on drop
  Tests: TestClient WebSocket integration; drop triggers banner; state change pushes partial.

- [ ] **E2E action flows + Undo toast.**
  - Inline draft editing: clicking draft → textarea; Esc cancels, Cmd+Enter commits via POST /api/moments/{id}/accept with edited action_params
  - Snooze popover: chip row; selection POSTs + closes
  - Undo toast: bottom-right, 3s, POST /api/moments/{id}/transition back to SUGGESTED if clicked in time
  - Deferred send: POST accept returns 202 + toast; actual action dispatched to outbox after 3s grace
  E2E tests with Playwright if installed; Selenium fallback; if neither: leave NOTE. 5 critical paths from DESIGN.md test plan.

## Week 12 — regression + cutover rehearsal

- [ ] **Golden-dataset regression harness (`tests/regression/test_golden_30day.py`).**
  - Requires: v1 snapshot (events.db + user_model.db) at fixed commit; path documented
  - Replays 30 days through v2 pipeline (all producers + scheduler + outbox)
  - Asserts: (a) Moment count within ±10% of v1 prediction count same window, (b) every high-signal v1 prediction has v2 Moment equivalent (thematic match via topic overlap), (c) no dedup violations, (d) Ollama latencies within CEO plan §1e budgets
  - Output: `docs/regression-runs/{date}.md` with pass/fail per assertion
  - If no v1 snapshot: skip with clear message; leave NOTE that harness needs a real snapshot.

- [ ] **Cutover rehearsal (full dry-run end-to-end).**
  - Extends `scripts/migrate_v1_to_v2.py` against full v1 backup (expected at `./data/backup-YYYYMMDD/`)
  - Verify: 3 dry-run checks from CEO plan data-migration section (row-count diff, FK integrity, vector-store integrity)
  - Emit `docs/cutover-rehearsals/{date}.md`: pass/fail + runtime + disk used
  - If no backup: NOTE saying "rehearsal requires local v1 backup; run on Mac Mini"
  - DOES NOT perform actual cutover (human operator only). Flag in summary.

## Week 13+ — HUMAN ONLY (agent should skip with NOTE)

- [ ] **MIGRATION CUTOVER — HUMAN.** Supervised v1→v2 production cutover. Not agent work. Agent: leave `<!-- NOTE: human-only — skip -->` and move on.

- [ ] **PHASE 1 ACCEPTANCE LOOP — HUMAN.** 2–4 week dogfood period; KPI recalibration at week 2 per CEO plan. Not agent work. Skip with NOTE.

- [ ] **APPLE DEV ENROLLMENT — HUMAN.** Phase 2 Week 0 prework. Not Phase 1. Skip with NOTE.

---

## When blocked / unclear

If a task mentions something you can't verify (local `data/` not present, ollama not running, hypothesis not installed, etc.), don't guess. Leave a `<!-- NOTE: <reason> -->` above the task, commit the note, move on to the next unchecked task. Don't check the task off.

## Guardrails reminder

- Never `git checkout master`, never `git push` (orchestrator does it), never `gh pr ...`
- Never write to `data/` (production DBs); never edit `config/settings.yaml`
- Everything on `v2-rewrite` branch, committed locally only
- One task per iteration; if budget runs out, `V2_AUTONOMOUS_ITERATION_PARTIAL`
- No destructive git, no force-push, no `rm`, no `launchctl`/`sudo`/`docker`/`brew`/`pip install`
