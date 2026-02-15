#!/usr/bin/env bash
# Life OS — Weekly Improvement Agent Runner
set -euo pipefail

LOG_DIR="/Users/jeremygreenwood/life-os/data/improvement-runs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d_%H%M%S).log"

echo "=== Life OS Improvement Run ===" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"

cd /Users/jeremygreenwood/life-os

# Run Claude Code with the improvement skill
claude --print --skill scripts/improve-lifeos.md \
    "Analyze Life OS data quality and make improvements" \
    >> "$LOG_FILE" 2>&1 || true

echo "Completed: $(date)" >> "$LOG_FILE"
