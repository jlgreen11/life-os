# CLAUDE.md

Guidance for Claude Code (claude.ai/code) sessions working in this repo.

## Current state (2026-04-22)

This repo contains two systems in parallel:

- **v1** (on `master`): the live production system. Local-first AI running on a Mac Mini, NATS JetStream + 5 SQLite DBs + `services/*` orchestrator in `main.py`. Still running; untouched since the continuous-improvement agent was paused on 2026-04-21.
- **v2** (on `v2-rewrite`): a ground-up rewrite organized around the **Moment** primitive (time + context + evidence + proposed action + state). Built by an autonomous agent consuming `NEXT_TASKS.md`. Not yet cut over.

When working on this repo, check `git branch --show-current` first. Everything below describes **v2**; for v1 see `docs/archive/v1/`.

## v2 — Product thesis

> Life OS observes your digital life and gives you the right action at the right moment. Nothing fake.

Evidence-backed Moments only. Killed from v1: mood inference, decision profile, expertise map, values inference — any LLM output without falsifiable grounding.

## v2 — Architecture

```
  UI (web — HTMX+Tailwind+Jinja; iOS in Phase 2)
      │ REST + WebSocket
  API (FastAPI)  — 14 endpoints
      │
  Moment Engine
  ├── Action Queue ◀──▶ Scheduler (wall-clock + context triggers)
  └── Insight Producers (cadence, relationship, temporal, spatial,
      comm-template, routine) — all evidence-backed
      │
  In-process asyncio bus + transactional outbox (no NATS)
      │
  Connectors (Proton Mail, iMessage, CalDAV, iOS context — 4 active)
      │
  One SQLite file (WAL, synchronous=NORMAL) + LanceDB index
```

## v2 — Module layout

```
core/               Moment primitive, state machine, scheduler, engine
  moment/
    types.py        Enums + dataclasses (MomentState, InsightType, ActionKind)
    state.py        Legal-transition enforcement
    engine.py       MomentEngine — producer dispatch + persistence
    scheduler.py    Wall-clock + context-trigger firing loop
    producer.py     Producer ABC + registry
    broadcaster.py  WebSocket push
producers/          Insight producers (one file per InsightType)
ai/                 engine.py, pii.py (redact/restore), context.py (briefing assembly)
api/
  app.py            FastAPI factory
  routes/           now, you, people, settings, health, context (iOS compat), websocket
  schemas.py        Pydantic request/response
storage/
  schema.py         13-table DDL + SCHEMA_VERSION
  repos/            moments, outbox, signal_profiles, feedback_weights, …
web/
  templates/        base.html, now.html, you.html, people.html, settings.html + partials
  static/tokens.css Design tokens from DESIGN.md
scripts/
  migrate_v1_to_v2.py
  run-v2-autonomous.sh    ← orchestrator for the autonomous agent
  v2-agent-prompt.md      ← system prompt appended when the agent invokes Claude
  daily_integrity_check.py  cutover_{rehearsal,monitor,rollback}.py  v1_v2_diff.py
tests/              94+ test files across storage/, producers/, web/, regression/, scripts/
```

## v2 — Key docs (in order of precedence)

1. `NEXT_TASKS.md` — live task queue
2. `DONE_TASKS.md` — append-only log
3. `DESIGN.md` — design tokens + IA + Moment-card states
4. `docs/plans/2026-04-21-v2-rewrite-plan.md` — engineering plan
5. `docs/adr/` — Architecture Decision Records
6. `AUTONOMOUS.md` — agent runbook
7. CEO plan (off-repo): `~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md`

## Running the system

### v2 (Phase 1, local dev)

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v                        # full test suite
python scripts/migrate_v1_to_v2.py --dry-run      # dry-run migration
python -m life_os                                  # serves on :8080 (when entry point ships)
```

### v1 (still runs on master)

See `docs/archive/v1/how-it-works.md` for the v1 setup (Docker Compose + NATS + Ollama + `main.py`).

## Autonomous agent

A background Claude Code process consumes `NEXT_TASKS.md` and commits to `v2-rewrite`. Orchestrator: `scripts/run-v2-autonomous.sh`. System prompt: `scripts/v2-agent-prompt.md`. Full details: `AUTONOMOUS.md`.

**Hard rails (enforced by `.claude/settings.json`):**
- Never `git checkout master`, never `git push` (orchestrator pushes on COMPLETE iterations only)
- No `gh pr create/merge`, no `git rebase`, no `git reset --hard`
- No writes to `data/` (production DBs) or `config/settings.yaml`
- No `rm`, `chmod`, `sudo`, `launchctl`, `docker`, `brew`, `pip install`

## Linting and formatting

```bash
ruff check . --fix
ruff format .
pre-commit run --all-files
```

Ruff rules (from `pyproject.toml`): E, W, F, I, UP, B, SIM, RUF. Line length 120. Python 3.12+.

## Conventions

- **Fail-loud in v2.** (v1's fail-open created 100+ silent-failure PRs; v2 raises and uses the outbox for recovery.)
- **Append-only events**: the `events` table is immutable. Never UPDATE or DELETE.
- **Encrypted credentials**: Fernet via `ConfigEncryptor`. Never returned raw.
- **Idempotent Moments**: `UNIQUE(source_insight_type, evidence_hash)` prevents duplicates.
- **State-machine transitions are enforced at the repo layer** — illegal → `IllegalTransition` raised, transaction rolled back, history not appended.

## Configuration

`config/settings.yaml` (gitignored). Template: `config/settings.example.yaml`.

Key sections:
- `data_dir` — path to SQLite + LanceDB
- `ai.ollama_url / ollama_model` — local LLM
- `ai.use_cloud / cloud_api_key` — optional PII-shielded Anthropic Claude
- `connectors.*` — per-connector config
- `defaults.*` — user preferences (verbosity, tone, proactivity, autonomy, quiet hours)
