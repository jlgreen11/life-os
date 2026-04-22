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

## Week 3 — outbox + scheduler + WAL integrity (complete, see DONE_TASKS.md)

## Week 4 — Producer base + 3 producers (complete, see DONE_TASKS.md)

## Week 5 — 3 more producers (complete, see DONE_TASKS.md)

## Week 6 — feedback + wiring (complete, see DONE_TASKS.md)

## Week 7 — AI engine extraction (complete, see DONE_TASKS.md)

## Week 8 — API surface

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
