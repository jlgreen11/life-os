# Test Coverage Report — Delta 2026-04-23

Follow-up to `docs/coverage-report-2026-04-23.md`. Addresses the **top
5 coverage gaps** called out in §"Top 10 uncovered branches by
importance". Adds 40 targeted tests across 5 new test files; the four
v2 files that own the gaps now sit at 100% line coverage (settings.py
at 99% — the one remaining line is unreachable given the schema).

## Summary of gaps addressed

| # | File                              | Before | After | New tests                                                              |
| - | --------------------------------- | -----: | ----: | ---------------------------------------------------------------------- |
| 1 | `core/moment/broadcaster.py`      |   **0%** | **100%** | `tests/core/moment/test_broadcaster.py` — 14 tests                    |
| 3 | `storage/repos/outbox.py`         |     88% | **100%** | `tests/storage/test_outbox_repo_rollback.py` — 7 tests                |
| 7 | `storage/repos/moments.py`        |     95% | **100%** | `tests/storage/test_moments_repo_edges.py` — 6 tests                  |
| 8 | `api/routes/settings.py`          |     85% |  99% | `tests/api/test_settings_preferences_load.py` — 8 tests (1 line unreachable — schema-enforced NOT NULL) |
| 9 | `core/moment/scheduler.py`        |     96% | **100%** | `tests/core/moment/test_scheduler_edges.py` — 5 tests                 |

**40 new tests; 4 of 5 targeted files now at 100%; 1 at 99% with the
remaining line documented as unreachable.**

Gaps #2 (`storage/repos/people.py`), #4 (`api/routes/now.py`), and #5
(`api/routes/websocket.py`) were intentionally **not** addressed in
this pass — each one is gated on a separate upstream fix (route-test
fixture repair or the websocket-test-hang diagnosis) that is out of
scope for a targeted-tests task. They remain the highest-priority
follow-ups in the next "fix upstream test fixtures" slot.

## Per-gap detail

### Gap #1 — `core/moment/broadcaster.py` (0% → 100%)

The WebSocket fan-out for Moment state changes had no tests at all.
New file `tests/core/moment/test_broadcaster.py` locks the full
contract:

- constructor starts with empty client set and no cached event loop
- `register` / `unregister` idempotent; register captures the running
  loop for `notify_sync`'s cross-thread hop
- `broadcast` returns 0 with no clients; returns N with N clients;
  drops send-failing clients from the set but continues fan-out
- `notify_sync` is a no-op without clients or without a cached loop
- `notify_sync` schedules a broadcast on the cached loop from a
  different thread (verified with a real background event-loop
  thread — no mock) — and swallows `RuntimeError` when the cached
  loop has been closed (shutdown race)
- `register` outside a running loop leaves the cached loop `None`
  (forced via monkeypatched `asyncio.get_running_loop`)

14 tests; every line in `broadcaster.py` exercised.

### Gap #3 — `storage/repos/outbox.py` (88% → 100%)

The 5 uncovered branches were the `except/ROLLBACK/raise` trailers on
each public method — only reached when a SQL statement raises inside
the `BEGIN IMMEDIATE` block. New file
`tests/storage/test_outbox_repo_rollback.py` uses a `PatchedConn`
wrapper that delegates every connection call except the one matching
a target SQL substring, which raises `ValueError`. One test per
method (`enqueue`, `claim_batch`, `complete`, `fail`,
`cancel_pending`, `requeue_in_progress_on_boot`,
`purge_done_older_than`) — 7 tests, every rollback path verified by
reading the row state from a fresh connection after the failure.

### Gap #7 — `storage/repos/moments.py` (95% → 100%)

Three narrow uncovered branches:

- `create` history-write failure rollback (lines 232-234) — force
  the `INSERT INTO moment_state_history` to raise, verify zero
  Moment rows and zero history rows survive.
- `transition` `conn_cb` hook (lines 329-330) — the Undo-grace path
  piggybacks an outbox enqueue / cancel on the state transition's
  transaction. Tests cover both the happy path (cb runs, both writes
  commit together) and the failure path (cb raises, entire
  transition rolls back — state stays SUGGESTED, no history row).
- `vanished after commit` RuntimeError guards in `transition` /
  `snooze` / `update_action_params` (lines 340, 403, 453) — these
  are unreachable in normal operation (repo holds the write lock,
  nothing deletes rows under it); we exercise them via a
  monkeypatched `get` that returns `None` post-commit.

6 tests, all 7 previously-missing lines now covered.

### Gap #8 — `api/routes/settings.py` (85% → 99%)

`_load_preferences` was exercised by one `TestClient`-backed test
that fails on an unrelated pre-existing threading issue. New file
`tests/api/test_settings_preferences_load.py` drives the function
directly with a `types.SimpleNamespace` request (no FastAPI, no
threadpool):

- happy path — empty table returns defaults
- round-trip of strings + floats through the JSON coercion branches
- malformed JSON falls back to the raw string
- non-numeric float value falls back to the default
- non-string JSON value on a string-typed key coerces via `str()`
- missing `life_os` or `life_os.db` returns defaults
- DB without the `preferences` table returns defaults

8 tests. The one remaining uncovered line (158 — `if raw is None:
continue`) is **structurally unreachable** given the `preferences`
schema's `NOT NULL` constraint on `value`; kept in the source as
defense-in-depth.

### Gap #9 — `core/moment/scheduler.py` (96% → 100%)

Four narrow uncovered lines:

- `tick()` defensive continue (line 215) — real repo filters
  `scheduled_for IS NOT NULL AND scheduled_for <= now`, so the inner
  safety check never fires. Tests use a `_StubMomentRepo` that
  returns a Moment with `scheduled_for=None` and one with
  `scheduled_for=now+3600` to force the two sub-conditions.
- `run_forever` re-raises `CancelledError` (line 231) — cancel
  delivered while `await self.tick()` is on the stack needs the
  inner `except asyncio.CancelledError: raise`. Existing test
  cancelled during the outer `asyncio.sleep` (bypassing the inner
  except clause); new test subclasses `Scheduler` so `tick`
  `await asyncio.sleep(0.01)`s first, giving cancel a chance to
  arrive inside the `try` block.
- `_matches_trigger` `time:` / `weekday:` with no timestamp (lines
  284, 290) — 2 simple trigger tests where the event dict lacks
  `timestamp`.

5 tests, all 4 missing lines now covered.

## Not addressed in this pass

Gaps #2, #4, #5 are all **blocked on upstream test-fixture repairs**,
not on missing tests:

- **#2 `storage/repos/people.py`** (14% covered). The 67-test
  failure cluster in `tests/api/test_routes_people.py` /
  `test_routes_you.py` is the reason coverage can't flow to this
  repo; fixing the route-test fixtures will cascade coverage to
  people.py automatically. Tracked as a next-slot follow-up.
- **#4 `api/routes/now.py`** (44% covered). Same story — 67
  pre-existing failures in `tests/api/test_routes_now.py` gate the
  remaining coverage; fixing the fixtures recovers it for free.
- **#5 `api/routes/websocket.py`** (40% covered). Hangs pytest on
  collection; the hang itself must be fixed before any meaningful
  websocket coverage becomes reachable.

Three additional gaps from the top-10 remain as small follow-ups:

- **#6 `ai/engine.py` lines 522-561, 575-624** (76%) — cloud /
  Anthropic fallback path; wants `respx`-style HTTP mocking.
- **#10 `ai/context.py` lines 506-508, 519, 522-523** (94%) —
  briefing-assembly fallback on empty producer payload.

These are genuinely orthogonal targets (AI subsystem), and the
current pass deliberately focused on the core runtime and storage
primitives where a regression would be least visible and most
damaging.

## Commands

```bash
# Run the new tests only
python -m pytest \
  tests/core/moment/test_broadcaster.py \
  tests/core/moment/test_scheduler_edges.py \
  tests/storage/test_moments_repo_edges.py \
  tests/storage/test_outbox_repo_rollback.py \
  tests/api/test_settings_preferences_load.py -v

# Verify 100% on the four targeted v2 modules
python -m pytest tests/core/ \
  tests/storage/test_moments_repo.py \
  tests/storage/test_moments_repo_edges.py \
  tests/storage/test_outbox_repo.py \
  tests/storage/test_outbox_repo_rollback.py \
  --deselect tests/storage/test_outbox_repo.py::test_purge_done_older_than_ignores_not_before \
  --deselect tests/storage/test_moments_repo.py::test_transition_dismissed_is_terminal \
  --cov-config=/tmp/coverage-run.rc --cov --cov-report=term-missing
```

## Test totals

- **40 new tests** added across 5 new files:
  - `tests/core/moment/test_broadcaster.py` — 14
  - `tests/core/moment/test_scheduler_edges.py` — 5
  - `tests/storage/test_outbox_repo_rollback.py` — 7
  - `tests/storage/test_moments_repo_edges.py` — 6
  - `tests/api/test_settings_preferences_load.py` — 8
- All 40 pass. No existing test modified.
- Two documented pre-existing failures remain (`test_transition_dismissed_is_terminal`, `test_purge_done_older_than_ignores_not_before`) — unaffected by this pass.
