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

- [x] **Profile the 5 existing SQLite DBs** — SHA `1f52f12` · 2026-04-21 · added `scripts/profile_v1_dbs.py` (read-only, `mode=ro`, FK graph, top-5, events by source) + stub `MIGRATION_PROFILE.md` (no local v1 DBs on this machine; operator runs on Mac Mini).
