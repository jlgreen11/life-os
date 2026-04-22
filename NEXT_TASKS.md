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

## Week 9 — Web: base + Now tab (complete, see DONE_TASKS.md)

## Week 10 — Web: You, People, Settings (complete, see DONE_TASKS.md)

## Week 11 — Real-time + full flows

<!-- NOTE (2026-04-22, iteration 21): the original "E2E action flows + Undo
toast" task was a $10+ iteration and contained an unresolved design decision
the CEO/eng plan doesn't cover. Specifically:

  1. The state machine in `core/moment/state.py` has no edge from
     ACCEPTED → SUGGESTED or DISMISSED → SUGGESTED (DISMISSED is
     terminal; ACCEPTED can only go to DONE). The original task body
     asked for "POST /api/moments/{id}/transition back to SUGGESTED" —
     not implementable without either (a) adding undo edges to
     `_LEGAL_TRANSITIONS` (CEO plan signoff needed; affects audit log
     semantics) or (b) modeling deferred dispatch as a new in-flight
     state (e.g. `ACCEPTED_PENDING_DISPATCH`) so undo means cancelling
     the outbox enqueue, not reversing state.

  2. The outbox spec in docs/plans/2026-04-21-v2-rewrite-plan.md §
     "Outbox pattern spec" defines `state IN (pending, in_progress,
     done, failed, dead)` — there is no "delayed / not-before" column
     for grace-period dispatch. A 3 s outbox-grace pattern needs either
     a `not_before TIMESTAMP` column on outbox or a separate scheduling
     mechanism (cf. moment scheduler). Neither is in the eng plan.

Splitting the original task into the well-defined slices below; the two
slices that depend on the undo design land last and carry their own
NOTE flagging the blocker. -->

- [ ] **Deferred outbox dispatch (3 s grace).**
  Per design note at `docs/plans/2026-04-22-undo-grace.md` § "Decision 2".
  Add `not_before INTEGER NULL` column to `outbox`; `enqueue(...,
  not_before=)` kwarg; `claim_batch` filters rows whose `not_before >
  now()`; `cancel_pending(event_id, subject)` repo method. `accept`
  enqueues with `not_before = now() + 3`. Undo calls `cancel_pending`
  inside the same txn as the state reversal. Replace
  `idx_outbox_state_created` with `idx_outbox_state_notbefore_created`
  per design note.
  Tests: claim_batch respects not_before; undo within grace cancels;
  undo after claim is a 410; boot recovery test per design note
  § "Decision 3".

- [ ] **Now-tab E2E tests.**
  Five DESIGN.md "test plan" critical paths: (1) accept moves card
  out, (2) dismiss moves card out, (3) snooze chip sets snooze_until +
  removes from pending, (4) WS push appends new card, (5) WS push
  removes accepted card via OOB. Use Playwright if importable (try/except
  at module top); Selenium fallback; otherwise mark with
  `pytest.skip("playwright/selenium not installed")` and emit a warning
  so the orchestrator surfaces it. Tests live in `tests/e2e/`.

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
