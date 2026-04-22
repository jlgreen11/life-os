# NEXT_TASKS — Life OS (Phase 2 runway)

> Ordered queue. Autonomous agent takes the **top** unchecked `- [ ]` item.
> When done, move it to `DONE_TASKS.md` (prepend, most recent first).
>
> **Phase 1 status:** agent-addressable work complete (43 tasks shipped,
> 30,844 LOC, 123 files). Only human-only Phase 1 tasks remain (cutover,
> dogfood, Apple Dev enrollment).
>
> **This queue runs Phase 2 groundwork + Phase 1 polish + cutover prep —
> all agent-safe (no Apple Dev account required).**
>
> **Task sizing:** each item fits ≤ 1 iteration ($10 opus budget, ~15 min).
>
> **Source of truth:**
> - `~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md`
> - `DESIGN.md` (design tokens + IA)
> - `docs/plans/2026-04-21-v2-rewrite-plan.md` (engineering plan)
>
> **Agent rules:**
> - Branch: v2-rewrite only. Never checkout master. Never push. No PRs, no merges.
> - Never write to `data/` (production DBs) or edit `config/settings.yaml`.
> - Working tree clean at iteration end.
> - If blocked, leave `<!-- NOTE: reason -->` above the task, commit the note, move on.

---

## Category A — iOS foundation (agent-safe; no Apple Dev needed)

These build the iOS app scaffold. Code compiles on any Mac; no device runtime needed. When Apple Dev enrolls post-trip, you plug APNs/widgets/TestFlight on top.

<!-- NOTE (2026-04-22): Design-token task landed in commit 43887cc (as part of the docs-refresh sweep). Swift files exist at ios/LifeOS/DesignSystem/{Color+Tokens,Font+Tokens,Spacing,Elevation,Radius}.swift. Follow-up: add DesignSystemTests.swift verifying constants match DESIGN.md exactly — can be a separate small task if desired. -->

<!-- DONE (see DONE_TASKS.md): Port DESIGN.md tokens → Swift. 5 Swift files in ios/LifeOS/DesignSystem/, committed in 43887cc. -->

<!-- DONE (see DONE_TASKS.md): Regenerate APIClient.swift against v2 endpoints (with APITypes.swift + APIClientTests.swift using URLProtocol mock). -->

<!-- NOTE (2026-04-22): APIClient regen removed `sendCommand` / `getNotifications` / `getTasks` / `createTask` / `search` as specified. The legacy iOS views under `ios/LifeOS/Views/{Dashboard,Chat,Components}/` and `ios/LifeOS/App/AppState.swift` still reference these methods and WILL NOT COMPILE until the next task (Views restructure) lands. The Python test suite is unaffected; iOS is a scaffold-in-progress. -->

<!-- DONE (see DONE_TASKS.md): WebSocketManager.swift v2 rewrite — typed `WebSocketEvent` envelope (moment.created / moment.state_changed / connector.status_changed + forward-compat .unknown bucket), `WebSocketTransport` injection seam, exponential-backoff reconnect (capped 60s), 30s heartbeat ping. -->

<!-- NOTE (2026-04-22): WebSocketManager rewrite changed the public contract from `WebSocketMessage` to `WebSocketEvent` and `onMessage` to `onEvent`. `AppState.swift` still references the old shape via `webSocket?.onMessage = ...` + `handleWebSocketMessage(message: WebSocketMessage)` and WILL NOT COMPILE until the Views-restructure task below lands. The legacy `WebSocketMessage` struct in `Models/Models.swift` is now unreferenced by the manager but kept until that task wholesale-deletes the v1 view layer. -->

<!-- DONE (see DONE_TASKS.md): Restructure `ios/LifeOS/Views/` around 4-tab IA. Stub `NowTabView`/`YouTabView`/`PeopleTabView`/`SettingsTabView` + new `RootTabView` with 4 tabs (tray.full / person.crop.circle / person.2 / gear). Legacy v1 views (Dashboard/Chat/Context/Components/Settings/SettingsView), `ContentView.swift`, AppState.sendCommand/getNotifications wiring, and the `WebSocketMessage` legacy struct deleted. `RootTabViewTests.swift` covers tab order, titles, icons. -->

<!-- DONE (see DONE_TASKS.md): `NowTabView.swift` + `MomentCardView.swift` + `Previews/MockData.swift` + `NowTabViewTests.swift`. Full NOW / UP NEXT / DONE TODAY (collapsed default) sections, 22pt insight, evidence sheet, draft block (Radius.lg, draft tints), primary-action + ghost Edit/Snooze + dismiss buttons. 6 test classes / 26 tests covering MockData shape, section splitting, up-next prefix, primary labels (every ActionKind), draft body predicate, evidence copy + accessibility label. -->

<!-- DONE (see DONE_TASKS.md): `YouTabView.swift` + 4-section self-portrait + 22 tests. Header line ("Observed N months · M interactions"), four sections (WHEN YOU'RE AT YOUR BEST / HOW YOU WRITE / YOUR ROUTINES / DRIFTING) locked via `YouSection.allCases`, plain-text rows on bgRaised tiles (no mood bars, no progress bars, no charts), calm empty-state copy per DESIGN.md. `MockData.selfPortrait` + `MockData.emptySelfPortrait` fixtures. -->


- [ ] **`PeopleTabView.swift` + `ContactDossierView.swift`.** Per DESIGN.md:
  - `PeopleTabView` — search field, YOU first entry, NEEDS ATTENTION + ACTIVE THIS WEEK sections
  - Right-aligned monospace stats
  - Tap row → navigate to `ContactDossierView`
  - `ContactDossierView` — communication style, recent topics, cadence sparkline (SwiftUI Path), predicted next, [Start a message] primary
  NO avatars; plain text only.
  Tests: preview renders; navigation works.

- [ ] **`SettingsTabView.swift`.** Per DESIGN.md:
  - Connector list (status dot + last sync + enabled toggle)
  - Tap connector → `ConnectorEditView` (detail pane with form)
  - Preferences section (quiet hours, autonomy slider, proactivity)
  - No raw credentials shown
  Tests: preview renders; form validation.

- [ ] **ViewModels + XCTest unit tests.** Create `ios/LifeOS/ViewModels/`:
  - `NowViewModel` (fetches `/api/now`, holds `@Published var feed: MomentFeed`)
  - `YouViewModel`, `PeopleViewModel`, `SettingsViewModel`
  - Each VM uses `APIClient` via constructor injection; testable with mock client
  - XCTest: load happy path, error path, action dispatches (`accept()` / `dismiss()` / `snooze()`) update state correctly

## Category B — Phase 1 polish (agent-safe)

- [ ] **Test coverage audit.** Run `python -m pytest --cov=core --cov=storage --cov=producers --cov=ai --cov=api --cov=web --cov-report=term-missing --cov-report=html` and emit `docs/coverage-report-{date}.md` summarizing: overall %, per-module %, top 10 uncovered branches by importance (prioritize core/moment, storage/repos, outbox, scheduler, producers). Install `pytest-cov` if missing; if install denied, leave NOTE.

- [ ] **Fix top 5 coverage gaps.** Read the coverage report from prior task; write targeted tests for the 5 highest-priority gaps. Each gap → 1-3 new tests. Commit test file(s) + coverage-report-after-{date}.md showing delta.

- [ ] **Ruff + mypy cleanup.** Run `ruff check core/ storage/ producers/ ai/ api/ web/ scripts/ --fix` and `ruff format .`; install mypy if missing and run `mypy --strict core/ storage/repos/ producers/`. Fix auto-fixable; for mypy errors fix top 10; leave remaining in `docs/mypy-gaps.md`.

- [ ] **Public API docstrings.** Audit `core/moment/engine.py`, `scheduler.py`, `state.py`, `storage/repos/moments.py`, `outbox.py`, `ai/engine.py`, `api/routes/now.py`, `api/routes/you.py`. Every public class, method, function gets a docstring with: one-line summary, Args, Returns, Raises, and short example for non-trivial APIs. Google style, consistent across files.

- [ ] **Scheduler performance profile.** Write `scripts/profile_scheduler.py` that seeds 10,000 synthetic Moments (mix of scheduled + snoozed, time-distributed across next 24h), runs 1 hour of simulated ticks (monotonic clock injection), measures p50/p95/p99 tick latency + throughput. Emit `docs/perf-scheduler-{date}.md`. Acceptance: p99 tick latency < 500ms at 10K-row fleet.

- [ ] **Migration rehearsal at scale.** Extend `tests/fixtures/v1_sample/` with a larger fixture (10K events, 500 entities, 200 signal profile rows). Add `tests/scripts/test_migrate_v1_to_v2_scale.py` that runs the migration and asserts: (a) no memory spike > 512MB, (b) completes in < 2 min on this fixture, (c) all row-count invariants hold. If psutil unavailable, skip memory check with NOTE.

- [ ] **Resolve `feedback_events` table design question.** Currently the migration script skips v1 `feedback_log` rows. Write `docs/adr/2026-04-22-feedback-events-disposition.md` as a proper ADR (Context / Decision / Consequences / Alternatives). Decision: **add `feedback_events` table** (preserves legacy feedback for future analysis). Implement: (a) add DDL to `storage/schema.py`, (b) add `storage/repos/feedback_events.py` with `FeedbackEventsRepository`, (c) update `migrate_v1_to_v2.py` to write feedback rows, (d) tests.

- [ ] **ADR index + backfill.** Create `docs/adr/README.md` (index) and backfill 6 ADRs for major v2 decisions: (1) Moment primitive as first-class entity, (2) kill soft-insight services, (3) drop NATS for asyncio + outbox, (4) consolidate 5 SQLite DBs → 1, (5) HTMX + Tailwind + Jinja over SPA, (6) web-first Phase 1, iOS Phase 2. Format: Context / Decision / Consequences / Status.

## Category C — Cutover preparation (agent-safe)

- [ ] **Cutover runbook (`docs/cutover-runbook.md`).** Step-by-step operator guide for the Phase 1 v1→v2 cutover. Content:
  - Pre-flight (v1 backup + disk-space + health snapshot)
  - Stop v1 services
  - Run migration script (exact command + expected runtime + expected output)
  - Bring up v2 (exact commands)
  - Verify (`/api/health` expected JSON, sample Moment fires end-to-end, iOS compat shim responds)
  - 24-hour watch window (what to monitor; alert thresholds)
  - Rollback trigger criteria + procedure
  Include timing estimates (total ≤ 30 min RTO target per CEO plan).

- [ ] **Cutover monitor script (`scripts/cutover_monitor.py`).** Continuous loop polling `/api/health` every 10s, logs structured state, alerts (log.error + writes `data/cutover-alerts.jsonl`) on: connector offline > 5 min, DB write lag > 30s, scheduler heartbeat missing > 2 min, pending Moment count growing without accept/dismiss activity. Exit codes: 0 healthy for N min (configurable), 1 alert fired. Tests: simulate each alert scenario with mocks.

- [ ] **Cutover rollback script (`scripts/cutover_rollback.py`).** Automates "restore v1 snapshot" path from CEO plan. Inputs: snapshot directory path, target v1 service name. Actions: stop v2, restore v1 DBs from snapshot, restore LanceDB, restart v1 services, verify v1 `/api/status`. Hard-coded RTO target ≤ 30 min. Tests: dry-run mode that logs but doesn't execute; mocked FS.

- [ ] **v1/v2 data diff tool (`scripts/v1_v2_diff.py`).** Post-migration sanity check. Compares: row counts per table, spot-checks (10 random events, match v1 vs v2 by ID + content), FK integrity (every evidence.event_id resolves). Emits `docs/cutover-diffs/{date}.md` with pass/fail per check. Agent-runnable against synthetic fixtures.

- [ ] **Dry-run cutover CI harness (`tests/integration/test_cutover_dryrun.py`).** End-to-end integration test: fresh v1 fixture → run migrate_v1_to_v2.py → bring up v2 (FastAPI TestClient) → hit `/api/health` → create a test Moment → verify scheduler transitions it → shut down v2 → restore v1 snapshot via rollback script → verify v1 comes back. Acceptance: completes in < 2 min in CI; every assertion passes.

## Human-only (Phase 1 cutover-side and Phase 2 real-device work)

- [ ] **MIGRATION CUTOVER — HUMAN OPERATOR.** Live production v1→v2 cutover. Runbook in `docs/cutover-runbook.md`. Agent: skip with NOTE.
- [ ] **PHASE 1 DOGFOOD ACCEPTANCE LOOP — HUMAN.** 2-4 week dogfood. Agent: skip with NOTE.
- [ ] **APPLE DEV PROGRAM ENROLLMENT — HUMAN.** Prerequisite for Phase 2 APNs/TestFlight/widgets. Agent: skip with NOTE.

---

## When blocked / unclear

If a task mentions something you can't verify (Xcode not installed, `psutil` denied, coverage.py missing), don't guess. Leave `<!-- NOTE: reason -->` above the task, commit the note, move on.

## Guardrails

- Never `git checkout master`, never `git push` (orchestrator handles), never `gh pr ...`
- Never write to `data/` (production DBs); never edit `config/settings.yaml`
- Everything on `v2-rewrite` branch, committed locally only
- One task per iteration; budget $10; `V2_AUTONOMOUS_ITERATION_PARTIAL` if exceeded
- No destructive git, no force-push, no `rm`, no `launchctl`/`sudo`/`docker`/`brew`/`pip install`
