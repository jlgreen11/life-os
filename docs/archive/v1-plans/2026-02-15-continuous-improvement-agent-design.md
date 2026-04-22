# Continuous Improvement Agent — Design

**Date:** 2026-02-15

## Problem

The existing weekly improvement agent (`scripts/run-improvement.sh`) runs once
per week via `launchd` on Wednesdays at 10 PM. It uses `claude --print` in a
single-shot mode with a basic system prompt. This is too infrequent and too
shallow to drive meaningful, sustained improvement of the Life OS codebase.

## Solution: Continuous Loop Agent

A bash script that runs in an infinite loop, invoking Claude Code once per
iteration. Each iteration autonomously discovers the highest-impact improvement,
implements it with full documentation, tests it, and ships it as a merged PR.

### Architecture

```
launchd (KeepAlive: true)
  └── run-continuous-improvement.sh (infinite loop)
        └── per iteration:
              1. git pull master
              2. python analyze-data-quality.py → JSON
              3. claude --print --dangerously-skip-permissions \
                   --append-system-prompt improvement-agent.md \
                   --model sonnet --max-budget-usd 5 \
                   "<analysis + state + git log>"
              4. Claude: discover → implement → test → commit → PR → merge
              5. Log result, update state.json
              6. Sleep 10s, repeat
```

### Components

| File | Purpose |
|------|---------|
| `scripts/improvement-agent.md` | System prompt defining the agent's workflow, priorities, constraints, and output format |
| `scripts/run-continuous-improvement.sh` | Bash loop that orchestrates each iteration |
| `scripts/com.lifeos.continuous-improve.plist` | launchd plist with KeepAlive for auto-restart |
| `data/improvement-runs/state.json` | Persistent state tracking all improvements |
| `data/improvement-runs/iter-*.log` | Per-iteration logs (last 100 retained) |

### Key Decisions

- **Model: Sonnet** — fast and cost-effective for iterative work. Configurable
  via `IMPROVEMENT_MODEL` env var if Opus quality is needed.
- **Budget: $5/iteration** — prevents runaway cost if Claude enters long tool
  loops. Configurable via `IMPROVEMENT_MAX_BUDGET`.
- **Auto-merge: yes, if tests pass** — the agent runs `pytest`, only merges on
  green. Failed iterations are logged but not merged.
- **Discovery: autonomous** — no fixed backlog. Each cycle analyzes data quality,
  searches for stubs/TODOs, and picks the highest-impact fix.
- **One improvement per cycle** — keeps PRs atomic and reviewable.
- **10s cooldown** — small pause between iterations to let git settle and avoid
  API rate limits. Configurable via `IMPROVEMENT_COOLDOWN`.
- **Failure backoff** — after 3 consecutive failures, sleep 5 minutes to avoid
  burning credits on a persistent issue.

### Safety

The agent system prompt explicitly forbids:
- Modifying user data, config files, or credentials
- Force pushes or destructive git operations
- Merging with failing tests
- Modifying the agent itself (no self-modification loops)
- Adding dependencies without justification
