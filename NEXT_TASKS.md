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

<!-- DONE (see DONE_TASKS.md): `PeopleTabView.swift` + `ContactDossierView.swift` + `CadenceSparkline` + 43 tests. People tab: `.searchable`-driven query, pinned YOU identity row, NEEDS ATTENTION + ACTIVE THIS WEEK sections with right-aligned SF-Mono stats ("9d ago" + "+4d"), `NavigationLink(value:)` push to per-contact dossier. Dossier: last-contact sentence, commTemplate/fallback, bullet topics, SwiftUI-Path cadence sparkline (no chart libs), predicted-next sentence, single `[Start a message]` primary. Pure static helpers for filter/lastSeenLabel/cadenceLabel/sparklinePoints so XCTest can lock every copy string and the sparkline math against a frozen `MockData.anchorDate`. -->

<!-- DONE (see DONE_TASKS.md): `SettingsTabView.swift` + `ConnectorEditView.swift` + `Preferences` + 55 tests. Settings tab: CONNECTORS tile list (status dot + display name + "Ready · 3m ago" / "Paused" / "Error · <detail>" statusLine + enabled toggle + chevron) pushing to per-connector detail pane on tap; PREFERENCES card with two HH:MM quiet-hours `TextField`s (with `isValidTime` warning) and autonomy/proactivity `Slider`s showing percent labels. `ConnectorEditView`: header card + enable toggle + CONFIGURATION `TextField`s + CREDENTIALS `SecureField`s with "••• saved" placeholders (raw creds never rendered) + disabled-by-default `[Save changes]` gated by pure `canSave`. New `Preferences` Codable struct mirrors `_PREFERENCE_DEFAULTS` in `api/routes/settings.py`. Four connector fixtures exercise every status-dot branch (ready/syncing/paused/error). Pure static helpers (`statusDotColor`/`displayName`/`statusLine`/`lastSyncLabel`/`isValidTime`/`clampUnit`/`percentLabel` + `ConnectorEditView.{defaultConfigKeys,defaultSecretKeys,labelForKey,isValidDraftValue,headerSubtitle,hasChanges,allEditedValuesValid,canSave}`) locked by XCTest without rendering. -->

<!-- DONE (see DONE_TASKS.md): ViewModels + XCTest unit tests. `ios/LifeOS/ViewModels/` houses the new `APIClientProtocol` (12-method surface — every method the four tab VMs need; `APIClient` actor gets a free conformance via extension) plus `NowViewModel` / `YouViewModel` / `PeopleViewModel` / `SettingsViewModel`. Each VM is `@MainActor`, takes the protocol via constructor injection, and exposes flat `@Published` `isLoading`/`errorMessage` slots alongside its tab-specific payload. NowViewModel ships a static `reconcile(feed:updated:)` that locks the (state → bucket) truth table so action dispatchers (`accept`/`dismiss`/`snooze`/`undo`/edit) correctly re-bucket the returned Moment. SettingsViewModel ships a static `applying(key:value:to:)` so preference upserts patch the local `Preferences` slot deterministically. `ios/LifeOSTests/ViewModelsTests.swift` adds a `MockAPIClient` (Result-typed slots per method, callLog) plus 33 tests across 5 `@MainActor` test classes covering load happy + error paths, action dispatch + reconcile truth table, query-string forwarding, dossier load, connector upsert/append, and preference patching. -->


## Category B — Phase 1 polish (agent-safe)

<!-- NOTE (2026-04-22): `pytest-cov` is not installed in the active Python env (pytest 8.4.2 is present; `python3 -c "import pytest_cov"` → ModuleNotFoundError). `Bash(pip install:*)` is in the deny list in `.claude/settings.json`, so the agent cannot install it. Unblocking options: (a) human runs `python -m pip install pytest-cov`, or (b) add `pytest-cov` to a requirements file + install, then re-queue this task. Task left `- [ ]`. -->

- [ ] **Test coverage audit.** Run `python -m pytest --cov=core --cov=storage --cov=producers --cov=ai --cov=api --cov=web --cov-report=term-missing --cov-report=html` and emit `docs/coverage-report-{date}.md` summarizing: overall %, per-module %, top 10 uncovered branches by importance (prioritize core/moment, storage/repos, outbox, scheduler, producers). Install `pytest-cov` if missing; if install denied, leave NOTE.

<!-- NOTE (2026-04-22 iter11): Depends on the blocked coverage-audit task above. Until pytest-cov + a working Python env are available, there is no coverage report to read. Keep `- [ ]`. -->

- [ ] **Fix top 5 coverage gaps.** Read the coverage report from prior task; write targeted tests for the 5 highest-priority gaps. Each gap → 1-3 new tests. Commit test file(s) + coverage-report-after-{date}.md showing delta.

<!-- NOTE (2026-04-22 iter11): Tried this task. `python3 -m ruff check core/ storage/ producers/ ai/ api/ web/ scripts/ --fix` succeeds (260/319 errors auto-fixed, 59 remain) and `ruff format` reformats 32 files cleanly. BUT cannot verify the auto-fixes are safe because no available Python environment has the project deps installed: system `python3` is 3.9 (`from datetime import UTC` import error in storage/event_store.py); `python3.12` is present but lacks `fastapi` and other runtime deps; `.venv/` referenced by scripts/run-v2-autonomous.sh does not exist; `pip install` is in the deny list. Pre-existing test failures also exist at HEAD: `tests/storage/test_moments_repo.py::test_transition_dismissed_is_terminal` (DID NOT RAISE IllegalTransition) and a circular-import error in `storage/repos/moments.py` that masks `tests/storage/test_outbox_repo.py`. Per "broken tests stop everything" rule, reverted all auto-fixes. Unblock options: (a) human creates `.venv/` with `requirements.txt` installed (`python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`), (b) human triages the two pre-existing test failures, (c) then re-queue this task. mypy half is independently blocked (no module, pip install denied). -->

- [ ] **Ruff + mypy cleanup.** Run `ruff check core/ storage/ producers/ ai/ api/ web/ scripts/ --fix` and `ruff format .`; install mypy if missing and run `mypy --strict core/ storage/repos/ producers/`. Fix auto-fixable; for mypy errors fix top 10; leave remaining in `docs/mypy-gaps.md`.

<!-- DONE (see DONE_TASKS.md): Public API docstrings audit — 8 target files reviewed. Only structural gap found was `OutboxRepository.__init__` (no docstring while parallel `MomentRepository.__init__` had one); filled. Every other public class / method / function already carries docstring with summary + args + returns + raises + example for non-trivial APIs. Style consistency: two dialects coexist (reST sections in ai/engine + storage/repos; narrative prose in core/moment + api/routes); both cover the acceptance criteria and a mass-rewrite toward one style would lose per-file readability. Audit doc: docs/docstrings-audit-2026-04-22.md. -->

<!-- DONE (see DONE_TASKS.md): Scheduler performance profile — `scripts/profile_scheduler.py` + `tests/scripts/test_profile_scheduler.py` (23 tests) + `docs/perf-scheduler-2026-04-23.md`. 10K-row fleet, 120 ticks × 30s = 1 simulated hour, WAL + synchronous=NORMAL tempfile DB, batch_limit=1000. Real-world run: p50=10.91ms, p95=26.67ms, **p99=28.51ms** (vs 500ms budget), max=29.75ms, 19,564 total fires across the window (163 fires/tick mean, 5.43 fires/simulated-second). PASS — p99 is ~18× under budget. -->

<!-- DONE (see DONE_TASKS.md): Migration rehearsal at scale — `tests/fixtures/v1_sample/` package (builder.py + SampleCounts) + `tests/scripts/test_migrate_v1_to_v2_scale.py` (4 tests, module-scoped single-run fixture). 10K events / 500 entities (400 contacts + 50 places + 50 subscriptions) / 200 signal_profiles (6 kept canonical + 4 dropped canonical + 190 unknown-legacy, exercising drop path at scale). Budget checks: <120s wall-clock, <512MB peak RSS delta (stdlib `resource.getrusage`, macOS-bytes / Linux-KB shim — psutil-skip path in spec not needed since `resource` is always available). Real run on M-series Mac: 0.07s / 8.6MB. -->

<!-- DONE (see DONE_TASKS.md): Resolve feedback_events table design question — ADR at docs/adr/2026-04-22-feedback-events-disposition.md, DDL added to storage/schema.py (13→14 tables), storage/repos/feedback_events.py + FeedbackEventsRepository + 10 tests, migrate_v1_to_v2.py writes v1 feedback_log → feedback_events tagged source='v1_migration', existing migration tests updated (v1/scale/cutover_rehearsal). -->


<!-- DONE (see DONE_TASKS.md): ADR index + backfill. `docs/adr/README.md` index + 6 backfilled ADRs (002 Moment primitive, 003 kill soft-insights, 004 asyncio+outbox over NATS, 005 single SQLite, 006 HTMX+Tailwind+Jinja, 007 web-first/iOS Phase 2). Context / Decision / Consequences / Alternatives / Follow-up on each. -->

## Category C — Cutover preparation (agent-safe)

<!-- DONE (see DONE_TASKS.md): Cutover runbook — docs/cutover-runbook.md + tests/docs/test_cutover_runbook.py (pure-stdlib structural lock on the runbook). Sections 0-8 cover pre-flight (backup + disk + health snapshot + secrets key note), stop v1 (launchd + docker compose), run migration (exact command + expected output + hard-fail checklist + FK check), bring up v2 (NATS teardown + python -m life_os + log tail), verify (/api/health expected JSON, /api/now sample Moment, /api/context/event iOS shim, row-count diff), 24-hour watch window (alert-threshold table + checkpoint schedule), rollback (trigger criteria + scripted + manual procedure + post-rollback actions). At-a-glance timeline sums to ~20 min wall-clock inside the 30-min RTO. 11 tests lock structure (required sections, order, RTO-30 language, verification endpoints, alert thresholds, rollback criteria+procedure, migration command flags, python -m life_os canonical start, balanced code fences, referenced scripts present-or-queued, timing sums under RTO). -->

<!-- DONE (see DONE_TASKS.md): Cutover monitor script — `scripts/cutover_monitor.py` (~460 lines, stdlib only) + `tests/scripts/test_cutover_monitor.py` (22 tests). Polls `/api/health` at configurable cadence (default 10s), tracks state across polls, alerts on ok:false / connector offline >5m / db_last_write_ts lag >30s / scheduler_heartbeat_ts missing or stale >2m / pending_moments growing without accept/dismiss >15m / HTTP error. Pure `evaluate(state, health, now_ts, config)` function keeps the scrape→alert computation testable at virtual clock without sleeping. Alerts append-only to JSONL + log.error; same-kind suppression prevents flood during a sustained outage. Exit 0 on healthy_for_minutes; exit 1 on first alert. -->

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
