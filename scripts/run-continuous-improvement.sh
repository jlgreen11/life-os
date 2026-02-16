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
    log "Invoking Claude ($MODEL, budget \$$MAX_BUDGET)..."
    set +e
    CLAUDECODE= "$CLAUDE_BIN" --print \
        --dangerously-skip-permissions \
        --append-system-prompt "$(cat scripts/improvement-agent.md)" \
        --model "$MODEL" \
        --max-budget-usd "$MAX_BUDGET" \
        "$PROMPT" \
        >> "$ITER_LOG" 2>&1
    EXIT_CODE=$?
    set -e

    # ------------------------------------------------------------------
    # 6. Evaluate result
    # ------------------------------------------------------------------
    if [[ $EXIT_CODE -eq 0 ]]; then
        log "Iteration $ITERATION completed successfully."
        CONSECUTIVE_FAILURES=0
    else
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        log "WARNING: Iteration $ITERATION failed (exit code $EXIT_CODE, streak: $CONSECUTIVE_FAILURES)"

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
    # 6.5. Restart Life OS if code was updated
    # ------------------------------------------------------------------
    # After pulling latest code, check if the codebase changed. If so,
    # restart the Life OS service to ensure fixes actually deploy.
    # This is critical for the improvement loop — without restart, all
    # merged PRs update the codebase but the running process never picks
    # up the changes, rendering all improvements inert.
    #
    # Supports two deployment modes:
    #   1. Docker Compose (recommended) — restarts via `docker compose restart`
    #   2. Local Python process — kills and restarts main.py directly
    NEW_HEAD=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    if [[ "$OLD_HEAD" != "$NEW_HEAD" && "$OLD_HEAD" != "unknown" ]]; then
        log "Code updated ($OLD_HEAD -> $NEW_HEAD). Restarting Life OS..."

        # Check if running via Docker Compose
        if docker compose ps lifeos 2>/dev/null | grep -q "lifeos"; then
            log "Detected Docker Compose deployment. Restarting container..."
            docker compose restart lifeos >> "$ITER_LOG" 2>&1

            # Wait for container to be healthy
            sleep 5

            # Verify container is running
            if docker compose ps lifeos 2>/dev/null | grep -q "Up"; then
                log "Life OS container restarted successfully."
            else
                log "ERROR: Life OS container failed to restart. Check: docker compose logs lifeos"
            fi

        # Check if running as local Python process
        elif LIFEOS_PID=$(pgrep -f "python.*main.py" | head -1); then
            log "Detected local Python deployment. Restarting process (PID $LIFEOS_PID)..."
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
                log "Life OS started (PID $NEW_PID)."

                # Give it 3 seconds to initialize
                sleep 3

                # Verify it's still running
                if kill -0 "$NEW_PID" 2>/dev/null; then
                    log "Life OS is running successfully."
                else
                    log "ERROR: Life OS failed to start. Check $LOG_DIR/lifeos.log"
                fi
            else
                log "ERROR: Cannot start Life OS — venv not found at $VENV"
            fi

        else
            log "WARNING: No running Life OS instance found (expected Docker container or main.py process)."
            log "To start Life OS manually:"
            log "  Docker: docker compose up -d lifeos"
            log "  Local:  source $VENV && python main.py"
        fi
    else
        log "No code changes detected. Skipping restart."
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
