# Life OS v2 — Autonomous Agent Runbook

> Everything here is OFF BY DEFAULT. Nothing runs until you explicitly
> `launchctl load` the plist. Everything is reversible.

## What it does

Runs Claude Code in a loop every ~30 minutes. Each iteration picks the top
unchecked item from `NEXT_TASKS.md` on the `v2-rewrite` branch, implements it
with tests, commits locally with a `WIP:` prefix, and moves the item to
`DONE_TASKS.md`. Never pushes. Never opens PRs. Never touches master.

## Pre-flight checklist (once, before starting)

1. **v2-rewrite branch exists.** If not: `git checkout -b v2-rewrite`.
2. **NEXT_TASKS.md exists on v2-rewrite** (seeded automatically when the
   branch is created from this master commit).
3. **v1 continuous-improvement agent is paused.** You cannot run both.
   Stop it: `launchctl bootout gui/$(id -u)/com.lifeos.continuous-improve`
4. **Working tree is clean.** `git status` should show nothing uncommitted.
5. **Editing the plist paths.** Open `scripts/com.lifeos.v2-autonomous.plist`
   and replace every `REPLACE_WITH_ABSOLUTE_PATH` with the real repo path
   (e.g., `/Users/jeremygreenwood/life-os` on the Mac Mini, or
   `/Users/jlg/Documents/GitHub/life-os` on the workstation). Replace
   `REPLACE_WITH_HOME_PATH` with `$HOME`.
6. **Copy plist into launchd's scan path:**
   `cp scripts/com.lifeos.v2-autonomous.plist ~/Library/LaunchAgents/`

## Start it (when you're ready to walk away)

```bash
launchctl load ~/Library/LaunchAgents/com.lifeos.v2-autonomous.plist
```

That's it. The agent starts. First iteration fires within 60s. Every
subsequent iteration fires ~30 min after the previous one finishes.

Watch it live:
```bash
tail -F data/v2-runs/launchd-stdout.log
```

See what it's committing:
```bash
git -C /path/to/repo log --oneline v2-rewrite -20
```

## Stop it (KILL SWITCH — memorize this)

```bash
launchctl bootout gui/$(id -u)/com.lifeos.v2-autonomous
```

Alternative (older syntax, if bootout doesn't work on your macOS version):
```bash
launchctl unload ~/Library/LaunchAgents/com.lifeos.v2-autonomous.plist
```

Agent dies immediately. Any in-flight Claude invocation finishes to
completion (up to 30 min) then exits cleanly.

## Inspect on return

```bash
# How many iterations ran?
cat data/v2-runs/state.json

# What did it accomplish?
git log --oneline v2-rewrite | head -50
git diff master..v2-rewrite --stat

# Were any iterations partial/failed?
grep -E "PARTIAL|FAILED" data/v2-runs/launchd-stdout.log | tail -20

# What's still in the queue?
head -50 NEXT_TASKS.md
head -30 DONE_TASKS.md
```

## If something went wrong

**Rollback entire v2 branch (nuclear):**
```bash
launchctl bootout gui/$(id -u)/com.lifeos.v2-autonomous
git checkout master
git branch -D v2-rewrite         # you have to do this manually; agent can't
```

**Rollback just the last N commits (surgical):**
```bash
launchctl bootout gui/$(id -u)/com.lifeos.v2-autonomous
git checkout v2-rewrite
# Agent commits are all `WIP:` prefix. Inspect, then:
git reset --soft HEAD~N          # keep changes staged for review
# OR: git reset --hard HEAD~N    # (you must run this — agent cannot)
```

**Pause without stopping (cooldown extend):**
Edit `scripts/run-v2-autonomous.sh`, change `V2_COOLDOWN` default to a large
number, reload:
```bash
launchctl bootout gui/$(id -u)/com.lifeos.v2-autonomous
launchctl load ~/Library/LaunchAgents/com.lifeos.v2-autonomous.plist
```

## Budget accounting

Default: **$10 per iteration × 48 iterations/day = $480/day max**. With
average iteration cost more like $2–4, realistic is $100–200/day.

To lower spend: set `V2_MAX_BUDGET=5` in the plist's `EnvironmentVariables`.

Hard per-run cap via `--max-budget-usd` flag on every Claude invocation means
there's no runaway scenario. If Claude hits budget mid-task, it stops; agent
sees partial outcome; backs off longer.

## Defaults summary

| Setting | Default | Env var | Why |
|---|---|---|---|
| Model | opus | `V2_MODEL` | Quality > cost for architectural work |
| Budget/iteration | $10 | `V2_MAX_BUDGET` | Most iterations run $2–4 |
| Cooldown between | 30 min | `V2_COOLDOWN` | ~48 iter/day; room to interject |
| Partial backoff | 1 hour | `V2_PARTIAL_COOLDOWN` | Slow down after blockers |
| Fail backoff | 15 min | `V2_FAIL_BACKOFF` | After 3 failed exits |
| Hard timeout | 30 min | `V2_CLAUDE_TIMEOUT` | Per Claude invocation |

## Safety rails (what the agent cannot do)

Enforced by `.claude/settings.json`:

- No `git push` anything
- No `git checkout master`
- No `git reset --hard`, `rebase`, `revert`, `cherry-pick`, `merge`, `branch -D`
- No `gh pr create/merge`, no GitHub API
- No `rm`, `chmod`, `sudo`, `launchctl`, `docker`, `brew`, `pip install`
- No writes to `data/` or `config/settings.yaml`

If any of these are attempted, the tool call is denied and the agent must
either find another path or mark the task as blocked.

## Why it won't run away

- **Cooldown:** 30 min between iterations means ≤48 iter/day
- **Budget cap:** hard `--max-budget-usd 10` per iteration
- **Backoff:** partial → 1h, 3 failures → 15 min
- **Settings deny list:** destructive ops blocked
- **Target branch:** locked to `v2-rewrite`, master untouched
- **Local only:** no pushes, no external side effects
- **KeepAlive w/ throttle:** launchd won't spin if script exits immediately
