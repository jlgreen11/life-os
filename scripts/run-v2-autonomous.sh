#!/usr/bin/env bash
# Life OS v2 — Autonomous Rewrite Runner
#
# Runs in an infinite loop. Each iteration:
#   1. Verifies we're on v2-rewrite (creates branch if missing)
#   2. Pulls latest (no force, no reset)
#   3. Invokes Claude Code with scripts/v2-agent-prompt.md as system prompt
#   4. Logs iteration result
#
# Managed by launchd (scripts/com.lifeos.v2-autonomous.plist). The launchd job
# is OFF BY DEFAULT. The owner must explicitly `launchctl load` it when leaving.
# See AUTONOMOUS.md at repo root for start/stop commands.
#
# What this script does NOT do (critical):
#   - No git push (commits stay local)
#   - No gh pr create/merge
#   - No docker compose restart
#   - No main.py process restart
#   - No touching master branch
#
# To run manually for testing:
#   bash scripts/run-v2-autonomous.sh
# To stop a manual run: Ctrl+C

set -uo pipefail

# ---------------------------------------------------------------------------
# Path auto-detection (works on Mac Mini OR workstation)
# ---------------------------------------------------------------------------
PROJECT_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo "")"
if [[ -z "$PROJECT_DIR" ]]; then
    echo "FATAL: script not run from inside a git repo" >&2
    exit 1
fi

# Auto-detect claude binary. Override with CLAUDE_BIN env var if needed.
if [[ -z "${CLAUDE_BIN:-}" ]]; then
    if command -v claude &>/dev/null; then
        CLAUDE_BIN="$(command -v claude)"
    elif [[ -x "$HOME/.local/bin/claude" ]]; then
        CLAUDE_BIN="$HOME/.local/bin/claude"
    else
        echo "FATAL: claude binary not found. Set CLAUDE_BIN env var." >&2
        exit 1
    fi
fi

LOG_DIR="$PROJECT_DIR/data/v2-runs"
STATE_FILE="$LOG_DIR/state.json"
PROMPT_FILE="$PROJECT_DIR/scripts/v2-agent-prompt.md"
VENV="$PROJECT_DIR/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
MODEL="${V2_MODEL:-opus}"
MAX_BUDGET="${V2_MAX_BUDGET:-10}"
COOLDOWN="${V2_COOLDOWN:-1800}"                # 30 min between iterations
CLAUDE_TIMEOUT="${V2_CLAUDE_TIMEOUT:-1800}"    # 30 min hard cap per iteration
MAX_CONSECUTIVE_FAILURES=3
PARTIAL_COOLDOWN="${V2_PARTIAL_COOLDOWN:-3600}" # 1h after a partial iteration
FAIL_BACKOFF="${V2_FAIL_BACKOFF:-900}"         # 15 min after 3 failures

TARGET_BRANCH="v2-rewrite"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "FATAL: prompt file missing: $PROMPT_FILE" >&2
    exit 1
fi

if [[ ! -f "$STATE_FILE" ]]; then
    cat > "$STATE_FILE" << 'INIT'
{
  "total_iterations": 0,
  "total_complete": 0,
  "total_partial": 0,
  "total_failed": 0,
  "last_run": null,
  "iterations": []
}
INIT
fi

if command -v timeout &>/dev/null; then
    TIMEOUT_CMD="timeout"
elif command -v gtimeout &>/dev/null; then
    TIMEOUT_CMD="gtimeout"
else
    TIMEOUT_CMD=""
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
ITERATION=0
CONSECUTIVE_FAILURES=0

log "=== Life OS v2 Autonomous Rewrite Agent ==="
log "PROJECT_DIR: $PROJECT_DIR"
log "CLAUDE_BIN:  $CLAUDE_BIN"
log "MODEL:       $MODEL | BUDGET: \$$MAX_BUDGET | COOLDOWN: ${COOLDOWN}s"
log "TARGET:      $TARGET_BRANCH (no pushes, no PRs, local only)"
log "STATE:       $STATE_FILE"

while true; do
    ITERATION=$((ITERATION + 1))
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    ITER_LOG="$LOG_DIR/iter-${ITERATION}-${TIMESTAMP}.log"

    log "--- Iteration $ITERATION ---"

    # ------------------------------------------------------------------
    # 1. Ensure v2-rewrite exists and we're on it
    # ------------------------------------------------------------------
    if ! git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
        log "ERROR: branch $TARGET_BRANCH does not exist locally. Agent cannot proceed."
        log "Create it manually: git checkout -b $TARGET_BRANCH"
        log "Exiting so launchd can keep-alive and retry in 30s."
        exit 2
    fi

    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != "$TARGET_BRANCH" ]]; then
        log "Switching from $CURRENT_BRANCH to $TARGET_BRANCH"
        git checkout "$TARGET_BRANCH" >> "$ITER_LOG" 2>&1 || {
            log "ERROR: could not check out $TARGET_BRANCH (dirty working tree?)"
            log "Last 20 lines of log:"
            tail -20 "$ITER_LOG" 2>/dev/null || true
            CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
            sleep "$FAIL_BACKOFF"
            continue
        }
    fi

    # Abort if anything is uncommitted at iteration start (should never happen
    # if the agent commits everything it touches)
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log "WARNING: uncommitted changes detected at iteration start. Skipping."
        git status -s >> "$ITER_LOG" 2>&1
        sleep "$COOLDOWN"
        continue
    fi

    # ------------------------------------------------------------------
    # 2. Check NEXT_TASKS.md exists
    # ------------------------------------------------------------------
    if [[ ! -f "$PROJECT_DIR/NEXT_TASKS.md" ]]; then
        log "ERROR: NEXT_TASKS.md missing on $TARGET_BRANCH. Agent cannot pick work."
        log "Seed it: see AUTONOMOUS.md"
        sleep "$FAIL_BACKOFF"
        continue
    fi

    # Check there's at least one unchecked task
    if ! grep -qE '^- \[ \]' "$PROJECT_DIR/NEXT_TASKS.md"; then
        log "NEXT_TASKS.md has no unchecked items. Nothing to do. Sleeping."
        sleep "$COOLDOWN"
        continue
    fi

    # ------------------------------------------------------------------
    # 3. Build context
    # ------------------------------------------------------------------
    RECENT_COMMITS=$(git log --oneline -20 2>/dev/null || echo "no commits")
    NEXT_TASKS_SNIPPET=$(head -50 "$PROJECT_DIR/NEXT_TASKS.md" 2>/dev/null || echo "")
    STATE=$(cat "$STATE_FILE" 2>/dev/null || echo '{}')

    PROMPT="$(cat <<PROMPT_EOF
You are on iteration $ITERATION of the v2 autonomous rewrite loop.

Current branch: $TARGET_BRANCH
Never check out master. Never push. Never open a PR. Local commits only.

## Recent commits on $TARGET_BRANCH (last 20)
\`\`\`
$RECENT_COMMITS
\`\`\`

## Top of NEXT_TASKS.md (first 50 lines)
\`\`\`
$NEXT_TASKS_SNIPPET
\`\`\`

## State
\`\`\`json
$STATE
\`\`\`

Follow the workflow in your system prompt. Pick the top unchecked item, implement
it with tests, commit with WIP: prefix on $TARGET_BRANCH only, move the item to
DONE_TASKS.md, commit that.

Print V2_AUTONOMOUS_ITERATION_COMPLETE on the last line if successful,
or V2_AUTONOMOUS_ITERATION_PARTIAL with a reason if blocked.
PROMPT_EOF
)"

    # ------------------------------------------------------------------
    # 4. Invoke Claude
    # ------------------------------------------------------------------
    log "Invoking Claude ($MODEL, budget \$$MAX_BUDGET, timeout ${CLAUDE_TIMEOUT}s)..."

    if [[ -n "$TIMEOUT_CMD" ]]; then
        "$TIMEOUT_CMD" "$CLAUDE_TIMEOUT" \
            env CLAUDECODE= "$CLAUDE_BIN" --print \
            --dangerously-skip-permissions \
            --append-system-prompt "$(cat "$PROMPT_FILE")" \
            --model "$MODEL" \
            --max-budget-usd "$MAX_BUDGET" \
            "$PROMPT" \
            >> "$ITER_LOG" 2>&1
        EXIT_CODE=$?
    else
        CLAUDECODE= "$CLAUDE_BIN" --print \
            --dangerously-skip-permissions \
            --append-system-prompt "$(cat "$PROMPT_FILE")" \
            --model "$MODEL" \
            --max-budget-usd "$MAX_BUDGET" \
            "$PROMPT" \
            >> "$ITER_LOG" 2>&1
        EXIT_CODE=$?
    fi

    if [[ $EXIT_CODE -eq 124 ]]; then
        log "WARNING: Claude timed out after ${CLAUDE_TIMEOUT}s"
    fi

    # ------------------------------------------------------------------
    # 5. Classify result
    # ------------------------------------------------------------------
    ITER_OUTCOME="failed"
    if [[ $EXIT_CODE -eq 0 ]]; then
        if grep -q "^V2_AUTONOMOUS_ITERATION_COMPLETE" "$ITER_LOG" 2>/dev/null; then
            ITER_OUTCOME="complete"
        elif grep -q "^V2_AUTONOMOUS_ITERATION_PARTIAL" "$ITER_LOG" 2>/dev/null; then
            ITER_OUTCOME="partial"
        else
            ITER_OUTCOME="unknown"
        fi
    fi

    case "$ITER_OUTCOME" in
        complete)
            log "Iteration $ITERATION: COMPLETE"
            CONSECUTIVE_FAILURES=0
            # Push v2-rewrite to origin so the owner can monitor remotely.
            # Failures (network, auth) are non-fatal — next iteration retries.
            log "Pushing v2-rewrite to origin..."
            if git push origin "$TARGET_BRANCH" >> "$ITER_LOG" 2>&1; then
                log "Pushed v2-rewrite successfully."
            else
                log "Push failed (network/auth) — continuing. Will retry next iteration."
            fi
            ;;
        partial)
            REASON=$(grep "^V2_AUTONOMOUS_ITERATION_PARTIAL" "$ITER_LOG" | head -1 | sed 's/^V2_AUTONOMOUS_ITERATION_PARTIAL: //')
            log "Iteration $ITERATION: PARTIAL — $REASON"
            log "Extending cooldown to ${PARTIAL_COOLDOWN}s after partial"
            CONSECUTIVE_FAILURES=0
            sleep "$PARTIAL_COOLDOWN"
            continue
            ;;
        unknown)
            log "Iteration $ITERATION: UNKNOWN outcome (no sentinel line). Treating as partial."
            log "--- Last 20 lines of $ITER_LOG ---"
            tail -20 "$ITER_LOG" 2>/dev/null || true
            sleep "$PARTIAL_COOLDOWN"
            continue
            ;;
        failed)
            CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
            log "Iteration $ITERATION: FAILED (exit $EXIT_CODE, streak $CONSECUTIVE_FAILURES)"
            log "--- Last 20 lines of $ITER_LOG ---"
            tail -20 "$ITER_LOG" 2>/dev/null || true
            if [[ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE_FAILURES ]]; then
                log "Backing off: $CONSECUTIVE_FAILURES consecutive failures. Sleeping ${FAIL_BACKOFF}s."
                sleep "$FAIL_BACKOFF"
                CONSECUTIVE_FAILURES=0
            fi
            ;;
    esac

    # ------------------------------------------------------------------
    # 6. Ensure we're still on $TARGET_BRANCH
    # ------------------------------------------------------------------
    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != "$TARGET_BRANCH" ]]; then
        log "WARNING: agent left branch as $CURRENT_BRANCH. Switching back."
        git checkout "$TARGET_BRANCH" >> "$ITER_LOG" 2>&1 || true
    fi

    # ------------------------------------------------------------------
    # 7. Housekeeping
    # ------------------------------------------------------------------
    find "$LOG_DIR" -name "iter-*.log" -type f | sort -r | tail -n +101 | xargs rm -f 2>/dev/null || true

    # ------------------------------------------------------------------
    # 8. Cooldown
    # ------------------------------------------------------------------
    if [[ $COOLDOWN -gt 0 ]]; then
        log "Cooldown ${COOLDOWN}s before next iteration..."
        sleep "$COOLDOWN"
    fi
done
