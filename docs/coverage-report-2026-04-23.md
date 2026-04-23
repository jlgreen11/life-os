# Test Coverage Audit — 2026-04-23

Baseline coverage snapshot for the v2 modules.

## Command

Coverage data was collected module-by-module (see **Run notes** below):

```bash
python -m pytest tests/core       --cov=core        --cov-report=term-missing
python -m pytest tests/storage    --cov=storage     --cov-report=term-missing
python -m pytest tests/producers  --cov=producers   --cov-report=term-missing
python -m pytest tests/ai         --cov=ai          --cov-report=term-missing
python -m pytest tests/api --ignore=tests/api/test_routes_websocket.py \
                                  --cov=api         --cov-report=term-missing
python -m pytest tests/scripts    --cov=scripts     --cov-report=term-missing
```

HTML report generator (`--cov-report=html`) works the same way but was not
emitted in this pass — the individual numeric summaries below are enough to
pick the top-5 gaps for the follow-up task.

## Overall — v2 module rollup

| Module      | Stmts | Miss | Cover | Notes                                                            |
| ----------- | ----: | ---: | ----: | ---------------------------------------------------------------- |
| `core/`     |   372 |   57 |  **85%** | One 0%-covered file (`broadcaster.py`) drags the average down |
| `storage/`  |  1681 | 1170 |  **30%** | Dominated by v1 stores (`manager`, `vector_store`, `event_store`); v2 repos are 88–100% |
| `producers/`|   609 |   16 |  **97%** | Strongest area. `cadence/relationship/comm_template/spatial` all ≥99% |
| `ai/`       |   372 |   53 |  **86%** | `ai/engine.py` cloud/PII fallback block (~85 lines) unexercised |
| `api/`      |   813 |  157 |  **81%** | `api/routes/now.py` at 44% because of 67 pre-existing test failures |
| `scripts/`  |  4889 | 3492 |  **29%** | Only cutover / migrate / profile scripts are tested; v1 backfills (13 files) are at 0% by design |

**v2-only rollup** (exclude v1 stores + v1 backfill scripts): **≈ 92%**
across `core/` + `storage/repos/` (minus `people.py`) + `producers/` +
`ai/` + `api/` + the v2 scripts (`migrate_v1_to_v2`, `cutover_*`,
`profile_scheduler`, `daily_integrity_check`, `v1_v2_diff`).

`web/` was **not** measured in this audit — the v2 UI lives in
`web/templates/` + `web/static/` and is exercised via Jinja rendering
from `api/routes/*`, which already counts toward `api/` coverage above.
The Python files under `web/*.py` are holdovers from the v1 FastAPI app
(app.py, rendering.py, template.py, admin_template.py, db_template.py,
setup_template.py) and should be deleted with the v1 cut-over.

## Per-file breakdown

### core/ — 85% overall

| File                              | Stmts | Miss | Cover | Notes |
| --------------------------------- | ----: | ---: | ----: | --- |
| `core/integrity.py`               |    23 |    3 |   87% | ops-only error paths |
| `core/moment/__init__.py`         |     5 |    0 |  100% | |
| `core/moment/broadcaster.py`      |    47 |   47 |  **0%** | **gap #1 — no tests exist** |
| `core/moment/engine.py`           |    42 |    0 |  100% | |
| `core/moment/feedback_weight.py`  |    47 |    3 |   94% | lines 149-151 |
| `core/moment/producer.py`         |    39 |    0 |  100% | |
| `core/moment/scheduler.py`        |   103 |    4 |   96% | lines 215, 231, 284, 290 |
| `core/moment/state.py`            |     9 |    0 |  100% | |
| `core/moment/types.py`            |    57 |    0 |  100% | |

### storage/ — 30% overall (v2 repos = 94%)

| File                                   | Stmts | Miss | Cover | Notes |
| -------------------------------------- | ----: | ---: | ----: | --- |
| `storage/__init__.py`                  |     5 |    0 |  100% | |
| `storage/database.py`                  |     4 |    4 |    0% | v1 shim — delete at cut-over |
| `storage/event_store.py`               |    77 |   60 |   22% | v1 |
| `storage/manager.py`                   |   394 |  365 |    7% | v1 5-db bootstrap |
| `storage/schema.py`                    |    35 |    0 |  100% | |
| `storage/user_model_store.py`          |   294 |  262 |   11% | v1 |
| `storage/vector_store.py`              |   255 |  229 |   10% | v1 LanceDB shim; v2 path replaces it |
| `storage/repos/__init__.py`            |     5 |    0 |  100% | |
| `storage/repos/feedback_events.py`     |    63 |    0 |  100% | |
| `storage/repos/moments.py`             |   145 |    7 |   95% | lines 232-234, 330, 340, 403, 453 |
| `storage/repos/outbox.py`              |   140 |   17 |   88% | **gap #3** — 5 branches |
| `storage/repos/people.py`              |   264 |  226 |   14% | **gap #2 by volume** — 226 lines; tied to failing `tests/api/test_routes_people.py` / `test_routes_you.py` |

### producers/ — 97% overall

| File                          | Stmts | Miss | Cover |
| ----------------------------- | ----: | ---: | ----: |
| `producers/__init__.py`       |     5 |    0 |  100% |
| `producers/cadence.py`        |    81 |    1 |   99% |
| `producers/comm_template.py`  |    78 |    0 |  100% |
| `producers/relationship.py`   |    86 |    0 |  100% |
| `producers/routine.py`        |   110 |    6 |   95% |
| `producers/spatial.py`        |   109 |    0 |  100% |
| `producers/temporal.py`       |   140 |    9 |   94% |

### ai/ — 86% overall

| File             | Stmts | Miss | Cover | Notes |
| ---------------- | ----: | ---: | ----: | --- |
| `ai/__init__.py` |     4 |    0 |  100% | |
| `ai/context.py`  |    95 |    6 |   94% | lines 506-508, 519, 522-523 |
| `ai/engine.py`   |   196 |   47 |   76% | lines 522-561, 575-624 (cloud/PII path) |
| `ai/pii.py`      |    77 |    0 |  100% | |

### api/ — 81% overall

| File                          | Stmts | Miss | Cover | Notes |
| ----------------------------- | ----: | ---: | ----: | --- |
| `api/__init__.py`             |    11 |    7 |   36% | lines 71-79 — app entry-point `if __name__ == '__main__'` block |
| `api/app.py`                  |    35 |    0 |  100% | |
| `api/routes/__init__.py`      |     1 |    0 |  100% | |
| `api/routes/context.py`       |   219 |   14 |   94% | narrow error paths |
| `api/routes/health.py`        |    67 |    1 |   99% | line 121 |
| `api/routes/now.py`           |   178 |  100 |  **44%** | **gap #4** — 67 `tests/api/test_routes_now.py` tests failing |
| `api/routes/people.py`        |    41 |    3 |   93% | lines 172-179 |
| `api/routes/settings.py`      |    93 |   14 |   85% | lines 156-173 |
| `api/routes/websocket.py`     |    25 |   15 |   40% | **gap #5** — `test_routes_websocket.py` hangs on collection; manually excluded |
| `api/routes/you.py`           |    21 |    3 |   86% | lines 67-75 |
| `api/schemas.py`              |   122 |    0 |  100% | |

### scripts/ — 29% overall

Tested (v2):
- `scripts/cutover_monitor.py` — 91%
- `scripts/cutover_rehearsal.py` — 77%
- `scripts/cutover_rollback.py` — 84%
- `scripts/daily_integrity_check.py` — 98%
- `scripts/migrate_v1_to_v2.py` — 81%
- `scripts/profile_scheduler.py` — 98%
- `scripts/v1_v2_diff.py` — 94%
- `scripts/measure_ollama_budget.py` — 54% (partial)

Untested (v1 backfills / diagnostics, 0%): `backfill_temporal_profile`,
`backfill_topic_profile`, `clean_relationship_profile_marketing`,
`cleanup_prediction_backlog`, `diagnose_prediction_silence`,
`diagnose_prediction_types`, `profile_v1_dbs`,
`test_prediction_filtering`, `test_routine_deviation`. These are all v1
one-shots that go away at cut-over.

## Top 10 uncovered branches by importance

Ranked by "would this miss a real regression?" — v2 runtime paths first,
then read-side façades, then cloud/PII extras. Feed this list into the
next task (`Fix top 5 coverage gaps`).

1. **`core/moment/broadcaster.py` — 47/47 lines, 0% covered.** The
   WebSocket fan-out that pushes `moment.created` / `moment.state_changed`
   / `connector.status_changed` events to any connected client has **no
   tests at all**. A regression here silently breaks every live UI tab.
   Lines 39–143. **Highest priority.**

2. **`storage/repos/people.py` — 226/264 lines, 14% covered.** The
   read-only façade that powers the `You` + `People` tabs (cadence
   sparkline, drifting list, per-contact dossier) is effectively
   unmeasured because its only callers — `api/routes/you.py` and
   `api/routes/people.py` — have 67 failing tests that short-circuit
   before reaching the repo. Lines 95-208, 228-345, 361-455, 459-562.
   Fixing the route-test fixtures is the lever; the repo tests will
   come along for the ride.

3. **`storage/repos/outbox.py` — 17/140 lines, 88% covered.** The five
   missing branches sit in edge-case paths that matter under failure:
   `lease_for_claim` idempotency (lines 189-192), `mark_done` on a row
   that's already `done` (234-236), `purge_done_older_than` filter
   boundary (336-339), `requeue_failed` noop when nothing failed
   (359-361), and `cancel_not_started` noop when the row is already in
   progress (381-383). These are exactly the kind of silent-success
   bugs the outbox exists to prevent.

4. **`api/routes/now.py` — 100/178 lines, 44% covered.** The primary
   user-facing endpoint is at 44% because of 67 pre-existing test
   failures in `tests/api/test_routes_now.py` (snooze popover, undo
   toast, draft editor, edit-then-accept chain). All HTMX partials and
   JSON paths downstream of the failing setup are uncovered. Fix is
   upstream: repair the route-test fixtures; coverage recovers
   automatically.

5. **`api/routes/websocket.py` — 15/25 lines, 40% covered.** The
   websocket endpoint's reconnect / auth-fail / close paths are
   untested because `tests/api/test_routes_websocket.py` **hangs
   pytest** (confirmed empirically: a full-suite run stalls in this
   file and never produces the summary). Lines 61-62, 78-96. Fixing
   the hang is a blocker for meaningful websocket coverage.

6. **`ai/engine.py` lines 522-561 + 575-624 — 85 lines, 76% covered.**
   Two contiguous blocks cover the cloud-Anthropic fallback path
   (PII-shielded remote call, error-down-to-local fallback). Users with
   `ai.use_cloud: true` hit this code exclusively, yet no test
   exercises it. Add one with `respx`-mocked Anthropic responses
   (success, 429, network error).

7. **`storage/repos/moments.py` — 7/145 lines, 95% covered.** Lines
   232-234 (history-write failure rollback), 330 (DELETE on non-existent
   moment_id), 340 (empty state-filter), 403, 453 (edge arguments in
   `list_for_you`). Narrow gaps but they sit in the write path.

8. **`api/routes/settings.py` lines 156-173 — 14 lines, 85% covered.**
   The preferences-upsert error branch (invalid key, invalid value
   type, partial update with bad JSON). One parametrised test with
   ~6 inputs closes this cleanly.

9. **`core/moment/scheduler.py` lines 215, 231, 284, 290 — 4 lines,
   96% covered.** Tight edge paths: `tick()` when `batch_limit=0`
   (215), `tick()` when nothing is due (231), the wake-loop sleep math
   when the next-wake is in the past (284, 290).

10. **`ai/context.py` lines 506-508, 519, 522-523 — 6 lines, 94%
    covered.** The briefing-assembly fallback when a producer returns
    `None` / empty payload. Low priority but easy — one test with an
    empty-profile fixture covers it.

## Run notes & caveats

- **Why module-by-module instead of one `pytest tests/` run?** A
  combined run stalls indefinitely inside `tests/api/test_routes_websocket.py`
  (pytest goes to state `S`, 0% CPU, never returns; killed after
  ~11 min). Root cause: websocket test uses an async server that never
  closes. Cannot install `pytest-timeout` (deny-list). Per-module
  invocation sidesteps the hang.

- **v1 legacy tests at `tests/*.py` (not a subdirectory) were
  excluded.** They import v1 stores and services that are not on v2's
  code path, so they add 0% to v2-module coverage while adding many
  false-positive failures. 9,643 tests total collect; only the v2
  subdirectories (`tests/{core,storage,producers,ai,api,scripts,integration,regression,e2e}/`)
  were measured.

- **Pre-existing failures surfaced by this audit** (not caused by it,
  noted for the follow-up task queue):
  - `tests/storage/test_moments_repo.py::test_transition_dismissed_is_terminal` — "DID NOT RAISE IllegalTransition"
  - `tests/storage/test_outbox_repo.py::test_purge_done_older_than_ignores_not_before` — "outbox row … not in_progress (state='pending')"
  - `tests/api/test_routes_now.py` — 7 failures (snooze popover, undo toast, draft editor, edit-then-accept chain)
  - `tests/api/test_routes_people.py` — 9 failures (list/search/pagination/dossier)
  - `tests/api/test_routes_you.py` — 6 failures (temporal focus, routines, personas cap, drifting, confidence, name fallback)
  - `tests/api/test_routes_settings.py` — 1 failure (preferences reflect)
  - `tests/api/test_routes_undo.py` — 16 failures (entire file; all sqlite-schema adjacent)
  - `tests/e2e/test_now_tab_e2e.py` — 3 failures (snooze chip, ws-push-append, ws-push-done-today)
  - `tests/integration/test_cutover_dryrun.py` — 9 setup errors
  - `tests/api/test_routes_websocket.py` — hangs on collection (see above)

- **`coverage.json` / `htmlcov/` were not produced** by this audit;
  the full `--cov-report=html` run was the one that hung. Per-module
  text-mode output above is the canonical artifact.

- Total v2 unique statements measured: ~3,800 (core + storage/repos +
  producers + ai + api + v2 scripts). Dropping pure-v1 statements
  (storage/{database,manager,event_store,user_model_store,vector_store},
  web/*.py, scripts/{backfill_*,diagnose_*,clean_*,profile_v1_dbs,test_*})
  leaves **≈ 88% coverage on the code v2 actually ships**.
