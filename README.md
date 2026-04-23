# Life OS

![Status](https://img.shields.io/badge/status-v2%20rewrite-orange) ![Python](https://img.shields.io/badge/python-3.12+-blue) ![iOS](https://img.shields.io/badge/iOS-SwiftUI-black) ![License](https://img.shields.io/badge/license-Proprietary-red) ![Local-first](https://img.shields.io/badge/privacy-local--first-green)

**A local-first AI that observes your digital life and gives you the right action at the right moment. Nothing fake.**

Life OS watches email, messages, calendar, and device context; produces **Moments** — time-anchored, evidence-backed suggestions with one-tap actions. Every insight is cited. Every action is reversible. No mood bars, no horoscope. Runs on a Mac Mini over Tailscale.

> **Status (2026-04-22)**: v2 rewrite in progress on [`v2-rewrite`](https://github.com/jlgreen11/life-os/tree/v2-rewrite). Built by an autonomous Claude Code agent consuming [`NEXT_TASKS.md`](NEXT_TASKS.md). The v1 implementation still runs in production on `master` until cutover. See the [v2 rewrite plan](docs/plans/2026-04-21-v2-rewrite-plan.md) for scope and schedule.

---

## What it does

Observes email, messages, calendar, and device context; produces **Moments** — time-anchored, evidence-backed suggestions with one-tap actions. Every insight has a citation. Every action is reversible. No mood bars, no horoscope. Runs on a Mac Mini over Tailscale.

Core primitive:

```
Moment = (time + context + evidence + proposed action + state)
```

## Current architecture (v2, on `v2-rewrite`)

```
  ┌────────────────────────────────────────────────────┐
  │              Web (HTMX + Tailwind + Jinja)          │
  │          iOS (SwiftUI, Phase 2 — in progress)       │
  └────────────────────────┬───────────────────────────┘
                           │ REST + WebSocket
  ┌────────────────────────▼───────────────────────────┐
  │                   API (FastAPI)                     │
  │   /api/now  /api/you  /api/people  /api/settings    │
  │                     /api/context/*                  │
  └────────────────────────┬───────────────────────────┘
                           │
  ┌────────────────────────▼───────────────────────────┐
  │                 MOMENT ENGINE                       │
  │   Action Queue ◀──▶ Scheduler (wall-clock + ctx)    │
  │                           ▲                         │
  │                   Insight Producers                 │
  │   cadence · relationship · temporal · spatial       │
  │       comm-template · routine                       │
  └────────────────────────┬───────────────────────────┘
                           │
  ┌────────────────────────▼───────────────────────────┐
  │          IN-PROCESS ASYNCIO EVENT BUS               │
  │           + transactional outbox                    │
  └────────────────────────┬───────────────────────────┘
                           │
  ┌────────────────────────▼───────────────────────────┐
  │                  Connectors (v1: 4 active)          │
  │   Proton Mail · iMessage · CalDAV · iOS context     │
  │      (dormant: Signal, Gmail, Plaid, HA, browser)   │
  └────────────────────────┬───────────────────────────┘
                           │
  ┌────────────────────────▼───────────────────────────┐
  │         Storage — 1 SQLite file + LanceDB index     │
  └────────────────────────────────────────────────────┘
```

## Key docs

| Doc | Purpose |
|---|---|
| [DESIGN.md](DESIGN.md) | Design tokens (type, color, spacing, elevation), 4-tab IA, Moment-card states |
| [docs/plans/2026-04-21-v2-rewrite-plan.md](docs/plans/2026-04-21-v2-rewrite-plan.md) | Engineering plan (14 API endpoints, 13-table schema, outbox spec, state machine) |
| [NEXT_TASKS.md](NEXT_TASKS.md) | Live task queue the autonomous agent consumes |
| [DONE_TASKS.md](DONE_TASKS.md) | Append-only log of completed tasks |
| [AUTONOMOUS.md](AUTONOMOUS.md) | Runbook for the autonomous rewrite agent |
| [docs/adr/](docs/adr/) | Architecture Decision Records (in progress) |
| [docs/archive/](docs/archive/) | v1-era docs (preserved for history) |

## Build & run (v2, Phase 1)

> v2 is on `v2-rewrite` and not yet cut over. These commands reflect where it's going; the actual cutover is documented in the forthcoming `docs/cutover-runbook.md`.

```bash
# 1. Environment
cp config/settings.example.yaml config/settings.yaml   # edit with creds
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Migrate v1 → v2 (dry-run first — NEVER touches prod data)
python scripts/migrate_v1_to_v2.py --dry-run

# 3. Run
python -m life_os         # serves on :8080
```

## Test

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Tech choices (v2)

- **Backend:** Python 3.12, asyncio, FastAPI
- **Storage:** one SQLite file (WAL, synchronous=NORMAL) + LanceDB vector index
- **Event bus:** in-process asyncio + transactional outbox (no NATS)
- **AI:** Ollama (mistral, quantized for foreground) + optional PII-shielded Anthropic Claude for complex reasoning
- **Web UI:** HTMX + Tailwind + modular Jinja (no SPA framework)
- **iOS:** SwiftUI (Phase 2, after Phase 1 acceptance KPIs converge)
- **Deployment:** Mac Mini + Tailscale (local-first)

## Killed from v1 (intentional)

Services that produced low-signal / un-falsifiable output are removed:

- Mood inference (progress bars with no grounding)
- Decision profile (can't measure from inbound email)
- Expertise map (no evidence path)
- Values inference (LLM horoscope)

Retained producers are all evidence-backed: cadence, relationship, temporal, spatial, communication-template, routine.

## Privacy model

- **Local-first:** all data in local SQLite + LanceDB; no cloud by default.
- **Encrypted credentials:** Fernet encryption for connector passwords/API keys at rest.
- **PII shield:** when the optional cloud AI path runs, names/emails/phones/addresses are tokenized before leaving the device; real values restored on response. Cloud model never sees PII.
- **Mood privacy:** n/a in v2 (mood tracking was removed entirely).

## Contributing

Personal project by [@jlgreen11](https://github.com/jlgreen11). The v2 rewrite is driven by an autonomous Claude Code agent consuming [`NEXT_TASKS.md`](NEXT_TASKS.md); see [AUTONOMOUS.md](AUTONOMOUS.md) for how it works. External contributions are not being accepted at this time — the license is Proprietary (see [LICENSE](LICENSE)). If you have feedback or want to discuss commercial use, open a GitHub issue or contact [@jlgreen11](https://github.com/jlgreen11) directly.

## Security

Security reports go through GitHub Security Advisories or directly to [@jlgreen11](https://github.com/jlgreen11). Full policy in [SECURITY.md](SECURITY.md). **Please do not open public issues for security problems.**

## License

Proprietary. All rights reserved. See [LICENSE](LICENSE). The codebase is publicly visible for transparency and portfolio purposes; it is not open source. The author reserves the right to relicense future releases under AGPL-3.0 or another permissive license.

---

*README last updated 2026-04-22.*
