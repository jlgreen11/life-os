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

- [x] **Ollama latency baseline** — SHA `19a325d` · 2026-04-21 · added `scripts/measure_ollama_budget.py` (5 ops × N=10, p50/p95/p99 + token counts, embedding-based semantic search, graceful skip when Ollama unreachable) + stub `docs/plans/2026-04-22-ollama-baseline.md` (marked NOT RUN; real measurement deferred to Mac Mini where mistral + nomic-embed-text are installed). 15 smoke tests in `tests/scripts/test_measure_ollama_budget.py` cover pure helpers. `ruff check` clean.

- [x] **Promote engineering plan into the repo** — SHA `2caca68` · 2026-04-21 · wrote `docs/plans/2026-04-21-v2-rewrite-plan.md` (week-by-week sequence, 14-endpoint API contract + iOS compat shim, 13-table inventory w/ indexes + invariants, outbox pattern spec, Moment state-transition table). Links back to CEO plan.

- [x] **Profile the 5 existing SQLite DBs** — SHA `1f52f12` · 2026-04-21 · added `scripts/profile_v1_dbs.py` (read-only, `mode=ro`, FK graph, top-5, events by source) + stub `MIGRATION_PROFILE.md` (no local v1 DBs on this machine; operator runs on Mac Mini).
