# Architecture Decision Records

This directory captures the architectural decisions that shaped Life OS v2.
Each ADR is a short markdown file with the format:

- **Context** — the forces at play when the decision was made
- **Decision** — what was chosen, in one sentence, then supporting detail
- **Consequences** — positive + negative fallout we signed up for
- **Status** — `Proposed` · `Accepted` · `Superseded by ADR-NNN` · `Deprecated`

ADRs are append-only. Superseding a decision creates a new ADR that links
back to the one it replaces; we do not edit the old ADR except to update
its `Status` field.

## Filing conventions

- Filename: `YYYY-MM-DD-<slug>.md` using the date the ADR was written.
- Header: `ADR NNN: <one-line title>` where `NNN` is a zero-padded
  running counter (001, 002, …) assigned in this index at merge time.
- One decision per ADR. If a decision spawns follow-ups, file them as
  their own ADRs and cross-link.
- Backfilled ADRs (decisions made before the ADR process was formalized)
  carry both an `Authored` date (when the ADR was written) and a
  `Decision date` (when the choice was actually made).

## Index

| # | Title | Status | Decision date | Author |
|---|-------|--------|---------------|--------|
| [001](2026-04-22-feedback-events-disposition.md) | v1 `feedback_log` disposition in v2 | Accepted | 2026-04-22 | autonomous v2-rewrite agent (iter 15) |
| [002](2026-04-22-moment-primitive.md) | Moment as first-class primitive | Accepted (backfilled) | 2026-04-21 | autonomous v2-rewrite agent (iter 16) |
| [003](2026-04-22-kill-soft-insights.md) | Kill soft-insight services (mood / decision / expertise / values) | Accepted (backfilled) | 2026-04-21 | autonomous v2-rewrite agent (iter 16) |
| [004](2026-04-22-asyncio-outbox-over-nats.md) | Drop NATS; use in-process asyncio bus + transactional outbox | Accepted (backfilled) | 2026-04-21 | autonomous v2-rewrite agent (iter 16) |
| [005](2026-04-22-single-sqlite-db.md) | Consolidate 5 SQLite DBs into one `lifeos.db` | Accepted (backfilled) | 2026-04-21 | autonomous v2-rewrite agent (iter 16) |
| [006](2026-04-22-htmx-tailwind-jinja.md) | HTMX + Tailwind + Jinja over a JS SPA | Accepted (backfilled) | 2026-04-21 | autonomous v2-rewrite agent (iter 16) |
| [007](2026-04-22-web-first-ios-phase-2.md) | Web-first Phase 1; iOS native deferred to Phase 2 | Accepted (backfilled) | 2026-04-21 | autonomous v2-rewrite agent (iter 16) |

## How to write a new ADR

1. Copy the skeleton below into `docs/adr/YYYY-MM-DD-<slug>.md`.
2. Fill in Context, Decision, Consequences. Status starts `Proposed`.
3. Pick the next free number and append a row to the index above.
4. Commit both the new ADR and the updated index in the same commit.

```markdown
# ADR NNN: <one-line title>

- **Status:** Proposed
- **Authored:** YYYY-MM-DD
- **Decision date:** YYYY-MM-DD
- **Author:** <name / role>

## Context
<forces at play>

## Decision
<what was chosen>

## Consequences
### Positive
### Negative

## Alternatives considered
1. …
```
