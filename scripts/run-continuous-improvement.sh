#!/usr/bin/env bash
# Life OS — Continuous Improvement Agent Runner
#
# Runs in an infinite loop. Each iteration:
#   1. Pulls latest master
#   2. Runs data quality analysis
#   3. Invokes Claude Code to find and ship one improvement
#   4. Logs the result
#
# Managed by launchd (com.lifeos.continuous-improve.plist) with KeepAlive
# so it auto-restarts on crash. Can also be run manually:
#   bash scripts/run-continuous-improvement.sh
#
# To stop: launchctl bootout gui/$(id -u)/com.lifeos.continuous-improve
set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLAUDE_BIN="${CLAUDE_BIN:-/Users/jeremygreenwood/.local/bin/claude}"
PROJECT_DIR="/Users/jeremygreenwood/life-os"
LOG_DIR="$PROJECT_DIR/data/improvement-runs"
STATE_FILE="$LOG_DIR/state.json"
VENV="$PROJECT_DIR/.venv/bin/activate"

# Model: sonnet is fast and cost-effective for iterative improvements.
# Change to "opus" for higher quality at higher cost.
MODEL="${IMPROVEMENT_MODEL:-sonnet}"

# Max spend per iteration (USD). Prevents runaway costs if Claude gets
# stuck in a long tool-use loop.
MAX_BUDGET="${IMPROVEMENT_MAX_BUDGET:-5}"

# Cooldown between iterations (seconds). 0 = no delay.
COOLDOWN="${IMPROVEMENT_COOLDOWN:-10}"

# Max consecutive failures before backing off (sleep 5 min).
MAX_CONSECUTIVE_FAILURES=3

# Timeout per Claude invocation (seconds). Kills a hung process that's
# waiting for input or stuck in a loop. Default: 30 minutes.
CLAUDE_TIMEOUT="${IMPROVEMENT_TIMEOUT:-1800}"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# Initialize state file if it doesn't exist
if [[ ! -f "$STATE_FILE" ]]; then
    cat > "$STATE_FILE" << 'INIT'
{
  "total_iterations": 0,
  "last_run": null,
  "improvements": []
}
INIT
fi

# Verify claude is installed
if [[ ! -x "$CLAUDE_BIN" ]]; then
    echo "FATAL: claude not found at $CLAUDE_BIN" >&2
    exit 1
fi

# Resolve timeout command: GNU coreutils `timeout` on Linux,
# `gtimeout` on macOS via Homebrew (brew install coreutils).
# Falls back to no timeout if neither is available.
if command -v timeout &>/dev/null; then
    TIMEOUT_CMD="timeout"
elif command -v gtimeout &>/dev/null; then
    TIMEOUT_CMD="gtimeout"
else
    TIMEOUT_CMD=""
fi

# ---------------------------------------------------------------------------
# Helper: log with timestamp
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
ITERATION=0
CONSECUTIVE_FAILURES=0

log "=== Life OS Continuous Improvement Agent ==="
log "Model: $MODEL | Budget: \$$MAX_BUDGET/iteration | Cooldown: ${COOLDOWN}s"
log "State: $STATE_FILE"
log "Starting loop..."

while true; do
    ITERATION=$((ITERATION + 1))
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    ITER_LOG="$LOG_DIR/iter-${ITERATION}-${TIMESTAMP}.log"

    log "--- Iteration $ITERATION ---"

    # ------------------------------------------------------------------
    # 1. Ensure we're on master and up to date
    # ------------------------------------------------------------------
    log "Pulling latest master..."
    git checkout master >> "$ITER_LOG" 2>&1 || true
    git pull --ff-only >> "$ITER_LOG" 2>&1 || {
        log "WARNING: git pull failed (maybe merge conflict). Resetting to origin/master."
        git fetch origin >> "$ITER_LOG" 2>&1
        git reset --hard origin/master >> "$ITER_LOG" 2>&1
    }

    # Clean up any leftover improvement branches from prior failed runs
    git branch --list 'improve/*' | while read -r branch; do
        git branch -D "$branch" >> "$ITER_LOG" 2>&1 || true
    done

    # Clean up merged remote improvement branches (these accumulate over time).
    # Only delete branches whose PRs have already been merged into master.
    git fetch origin --prune >> "$ITER_LOG" 2>&1 || true
    git branch -r --list 'origin/improve/*' --merged origin/master 2>/dev/null | while read -r rbranch; do
        local_name="${rbranch#origin/}"
        git push origin --delete "$local_name" >> "$ITER_LOG" 2>&1 || true
    done

    # ------------------------------------------------------------------
    # 2. Run data quality analysis
    # ------------------------------------------------------------------
    log "Running data quality analysis..."
    ANALYSIS=""
    if [[ -f "$VENV" ]]; then
        ANALYSIS=$(source "$VENV" && python scripts/analyze-data-quality.py 2>/dev/null) || ANALYSIS='{"error": "analysis script failed"}'
    else
        ANALYSIS='{"error": "venv not found"}'
    fi

    # ------------------------------------------------------------------
    # 3. Read current state and recent git log
    # ------------------------------------------------------------------
    STATE=$(cat "$STATE_FILE" 2>/dev/null || echo '{"total_iterations":0,"improvements":[]}')
    RECENT_COMMITS=$(git log --oneline -30 2>/dev/null || echo "no commits")

    # ------------------------------------------------------------------
    # 4. Build the prompt with all context
    # ------------------------------------------------------------------
    PROMPT="$(cat <<PROMPT_EOF
You are on iteration $ITERATION of the continuous improvement loop.

## Data Quality Analysis
\`\`\`json
$ANALYSIS
\`\`\`

## Recent Commits (last 30)
\`\`\`
$RECENT_COMMITS
\`\`\`

## Current State
\`\`\`json
$STATE
\`\`\`

Analyze the codebase, identify the single highest-impact improvement, implement
it with full comments and documentation, write tests, and ship it as a merged PR.
Follow the workflow in your system prompt exactly. Update the state file when done.
PROMPT_EOF
)"

    # ------------------------------------------------------------------
    # 5. Run Claude Code
    # ------------------------------------------------------------------
    log "Invoking Claude ($MODEL, budget \$$MAX_BUDGET, timeout ${CLAUDE_TIMEOUT}s)..."
    # Note: set -e is NOT active (script uses set -uo pipefail only).
    # Do NOT add set -e here — it would persist into subsequent iterations
    # and cause the script to crash on any unprotected command failure.
    #
    # timeout(1)/gtimeout(1) kills the process if it exceeds CLAUDE_TIMEOUT
    # seconds. This prevents a hung Claude process (e.g. waiting for stdin
    # after a CLI update changes flag names) from blocking the loop forever.
    # Exit code 124 = timed out.
    if [[ -n "$TIMEOUT_CMD" ]]; then
        "$TIMEOUT_CMD" "$CLAUDE_TIMEOUT" \
            env CLAUDECODE= "$CLAUDE_BIN" --print \
            --dangerously-skip-permissions \
            --append-system-prompt "$(cat scripts/improvement-agent.md)" \
            --model "$MODEL" \
            --max-budget-usd "$MAX_BUDGET" \
            "$PROMPT" \
            >> "$ITER_LOG" 2>&1
        EXIT_CODE=$?
    else
        CLAUDECODE= "$CLAUDE_BIN" --print \
            --dangerously-skip-permissions \
            --append-system-prompt "$(cat scripts/improvement-agent.md)" \
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
    # 6. Evaluate result
    # ------------------------------------------------------------------
    if [[ $EXIT_CODE -eq 0 ]]; then
        log "Iteration $ITERATION completed successfully."
        CONSECUTIVE_FAILURES=0
    else
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        log "WARNING: Iteration $ITERATION failed (exit code $EXIT_CODE, streak: $CONSECUTIVE_FAILURES)"

        # Log the last 20 lines of the iteration log so the launchd
        # stdout/stderr captures contain actionable failure context
        # without needing to SSH in and read individual iter logs.
        log "--- Last 20 lines of $ITER_LOG ---"
        tail -20 "$ITER_LOG" 2>/dev/null || true
        log "--- end ---"

        # Back off after repeated failures to avoid burning credits
        if [[ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE_FAILURES ]]; then
            log "Backing off: $CONSECUTIVE_FAILURES consecutive failures. Sleeping 5 minutes."
            sleep 300
            CONSECUTIVE_FAILURES=0
        fi
    fi

    # Ensure we're back on master for the next iteration regardless of
    # what state Claude left git in
    git checkout master >> "$ITER_LOG" 2>&1 || true

    # Capture git HEAD before pull to detect if code changed
    OLD_HEAD=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    git pull --ff-only >> "$ITER_LOG" 2>&1 || true

    # ------------------------------------------------------------------
    # 6.5. Ensure Life OS is running (restart if code updated)
    # ------------------------------------------------------------------
    # Check if Life OS is running on EVERY iteration (not just when code changes).
    # This ensures the service stays up even if it crashes, is stopped manually,
    # or fails to start after a previous PR merge.
    #
    # Critical for the improvement loop: without this check, all merged PRs would
    # update the codebase but the running process would never pick up the changes
    # if Life OS isn't running, rendering all improvements inert.
    #
    # Supports two deployment modes:
    #   1. Docker Compose (recommended) — restarts via `docker compose restart`
    #   2. Local Python process — kills and restarts main.py directly
    NEW_HEAD=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    CODE_UPDATED=false

    if [[ "$OLD_HEAD" != "$NEW_HEAD" && "$OLD_HEAD" != "unknown" ]]; then
        CODE_UPDATED=true
        log "Code updated ($OLD_HEAD -> $NEW_HEAD). Will restart Life OS with new code..."
    fi

    # Detect deployment mode and ensure Life OS is running.
    # Priority order:
    #   1. Docker Compose (if docker-compose.yml exists)
    #   2. Local Python process

    # Check if docker-compose.yml exists (Docker deployment configured)
    if [[ -f "$PROJECT_DIR/docker-compose.yml" ]]; then
        # Check if container is currently running
        if docker compose ps lifeos 2>/dev/null | grep -q "Up"; then
            if [[ "$CODE_UPDATED" == "true" ]]; then
                log "Life OS container is running. Restarting with new code..."
                docker compose restart lifeos >> "$ITER_LOG" 2>&1

                # Wait for container to initialize
                sleep 5

                # Verify container is running
                if docker compose ps lifeos 2>/dev/null | grep -q "Up"; then
                    log "Life OS container restarted successfully."
                else
                    log "ERROR: Life OS container failed to restart. Check logs:"
                    log "  docker compose logs lifeos"
                fi
            else
                log "Life OS container is running (no code changes, skipping restart)."
            fi
        else
            log "Life OS container is offline. Starting it..."
            docker compose up -d lifeos >> "$ITER_LOG" 2>&1

            # Wait for container to initialize
            sleep 5

            # Verify container is running
            if docker compose ps lifeos 2>/dev/null | grep -q "Up"; then
                log "Life OS container started successfully."
            else
                log "ERROR: Life OS container failed to start. Check logs:"
                log "  docker compose logs lifeos"
            fi
        fi

    # No Docker Compose config — use local Python deployment
    else
        # Check if Life OS is currently running
        if LIFEOS_PID=$(pgrep -f "python.*main.py" | head -1); then
            if [[ "$CODE_UPDATED" == "true" ]]; then
                log "Life OS process found (PID $LIFEOS_PID). Restarting with new code..."
                kill "$LIFEOS_PID" >> "$ITER_LOG" 2>&1 || true

                # Wait up to 10 seconds for graceful shutdown
                for i in {1..10}; do
                    if ! kill -0 "$LIFEOS_PID" 2>/dev/null; then
                        log "Life OS stopped gracefully."
                        break
                    fi
                    sleep 1
                done

                # Force kill if still running
                if kill -0 "$LIFEOS_PID" 2>/dev/null; then
                    log "WARNING: Life OS did not stop gracefully. Sending SIGKILL..."
                    kill -9 "$LIFEOS_PID" >> "$ITER_LOG" 2>&1 || true
                    sleep 2
                fi

                # Start Life OS in the background
                log "Starting Life OS with updated code..."
                cd "$PROJECT_DIR"
                if [[ -f "$VENV" ]]; then
                    source "$VENV"
                    nohup python main.py >> "$LOG_DIR/lifeos.log" 2>&1 &
                    NEW_PID=$!
                    log "Life OS restarted (PID $NEW_PID)."

                    # Give it 3 seconds to initialize
                    sleep 3

                    # Verify it's still running
                    if kill -0 "$NEW_PID" 2>/dev/null; then
                        log "Life OS is running successfully."
                    else
                        log "ERROR: Life OS failed to start. Check logs:"
                        log "  tail -100 $LOG_DIR/lifeos.log"
                    fi
                else
                    log "ERROR: Cannot start Life OS — venv not found at $VENV"
                fi
            else
                log "Life OS process is running (PID $LIFEOS_PID, no code changes)."
            fi
        else
            log "Life OS process not found. Starting it..."
            cd "$PROJECT_DIR"
            if [[ -f "$VENV" ]]; then
                source "$VENV"
                nohup python main.py >> "$LOG_DIR/lifeos.log" 2>&1 &
                NEW_PID=$!
                log "Life OS started (PID $NEW_PID)."

                # Give it 3 seconds to initialize
                sleep 3

                # Verify it's still running
                if kill -0 "$NEW_PID" 2>/dev/null; then
                    log "Life OS is running successfully."
                else
                    log "ERROR: Life OS failed to start. Check logs:"
                    log "  tail -100 $LOG_DIR/lifeos.log"
                fi
            else
                log "ERROR: Cannot start Life OS — venv not found at $VENV"
            fi
        fi
    fi

    # ------------------------------------------------------------------
    # 7. Housekeeping: keep only last 100 iteration logs
    # ------------------------------------------------------------------
    find "$LOG_DIR" -name "iter-*.log" -type f | sort -r | tail -n +101 | xargs rm -f 2>/dev/null || true

    # ------------------------------------------------------------------
    # 8. Cooldown
    # ------------------------------------------------------------------
    if [[ $COOLDOWN -gt 0 ]]; then
        sleep "$COOLDOWN"
    fi
done
