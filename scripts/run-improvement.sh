#!/usr/bin/env bash
# Life OS — Weekly Improvement Agent Runner
set -euo pipefail

CLAUDE_BIN="/Users/jeremygreenwood/.local/bin/claude"
PROJECT_DIR="/Users/jeremygreenwood/life-os"
LOG_DIR="$PROJECT_DIR/data/improvement-runs"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d_%H%M%S).log"

echo "=== Life OS Improvement Run ===" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"

# Verify claude is installed
if [[ ! -x "$CLAUDE_BIN" ]]; then
    echo "ERROR: claude not found at $CLAUDE_BIN" >> "$LOG_FILE"
    exit 1
fi

cd "$PROJECT_DIR"

# Run Claude Code with the improvement skill as system prompt
set +e
"$CLAUDE_BIN" --print \
    --append-system-prompt "$(cat scripts/improve-lifeos.md)" \
    "Analyze Life OS data quality and make improvements" \
    >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
set -e

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "WARNING: claude exited with code $EXIT_CODE" >> "$LOG_FILE"
fi

echo "Completed: $(date)" >> "$LOG_FILE"

# Keep only the last 12 weekly run logs (3 months)
find "$LOG_DIR" -name "*.log" -not -name "launchd-*" -type f | sort -r | tail -n +13 | xargs rm -f 2>/dev/null || true
