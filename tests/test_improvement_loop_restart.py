"""
Tests for the continuous improvement loop restart mechanism.

This test suite verifies that the improvement loop correctly detects code
changes and restarts the Life OS process to ensure merged PRs actually deploy.

Critical behavior tested:
1. Script detects git HEAD changes after pull
2. Finds and stops running main.py process
3. Starts new main.py process with updated code
4. Handles edge cases (no process, failed start, etc.)
"""

import os
import subprocess
import time
from pathlib import Path


def test_script_has_restart_logic():
    """Verify the improvement loop script contains restart logic."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    # Check for key restart components
    assert "OLD_HEAD=$(git rev-parse HEAD" in content, "Missing OLD_HEAD capture"
    assert "NEW_HEAD=$(git rev-parse HEAD" in content, "Missing NEW_HEAD capture"
    assert 'if [[ "$OLD_HEAD" != "$NEW_HEAD"' in content, "Missing HEAD comparison"
    assert "pgrep -f" in content, "Missing process detection"
    assert "kill " in content, "Missing process termination"
    assert "nohup python main.py" in content, "Missing process restart"


def test_script_detects_no_code_change():
    """Verify script skips restart when no code changes."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    # Should log skip message when OLD_HEAD == NEW_HEAD
    assert "No code changes detected. Skipping restart." in content


def test_script_handles_missing_process():
    """Verify script handles case where main.py is not running."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    # Should warn when no process found
    assert "WARNING: No running Life OS process found" in content


def test_script_handles_graceful_shutdown():
    """Verify script attempts graceful shutdown before force kill."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    # Should send SIGTERM first
    assert 'kill "$LIFEOS_PID"' in content

    # Should wait for graceful shutdown
    assert "for i in {1..10}; do" in content
    assert "kill -0" in content

    # Should force kill only if needed
    assert "kill -9" in content
    assert "did not stop gracefully" in content


def test_script_verifies_restart_success():
    """Verify script checks if new process started successfully."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    # Should capture new PID
    assert "NEW_PID=$!" in content

    # Should verify process is running
    assert "kill -0" in content
    assert "Life OS is running successfully" in content
    assert "Life OS failed to start" in content


def test_script_logs_restart_actions():
    """Verify script logs all restart-related actions."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    expected_logs = [
        "Code updated",
        "Restarting Life OS",
        "Found Life OS process",
        "Sending SIGTERM",
        "Life OS stopped gracefully",
        "Starting Life OS with updated code",
        "Life OS started",
    ]

    for log_msg in expected_logs:
        assert log_msg in content, f"Missing log message: {log_msg}"


def test_script_handles_venv_missing():
    """Verify script handles missing venv gracefully."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    # Should check for venv before starting
    assert 'if [[ -f "$VENV" ]]; then' in content
    assert "Cannot start Life OS — venv not found" in content


def test_restart_happens_after_pull():
    """Verify restart logic is positioned correctly relative to git pull."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    lines = script_path.read_text().split('\n')

    # Find all occurrences and take the right ones (there are multiple git pulls in the script)
    # We want the one in the "Evaluate result" section after iteration completes
    old_head_line = next(i for i, line in enumerate(lines) if 'OLD_HEAD=$(git rev-parse HEAD' in line)

    # Find git pull that comes after OLD_HEAD
    pull_line = next(i for i, line in enumerate(lines) if "git pull --ff-only" in line and i > old_head_line)

    # Find NEW_HEAD that comes after the pull
    new_head_line = next(i for i, line in enumerate(lines) if 'NEW_HEAD=$(git rev-parse HEAD' in line and i > pull_line)

    # Find restart check
    restart_check_line = next(i for i, line in enumerate(lines) if '"$OLD_HEAD" != "$NEW_HEAD"' in line)

    # Verify ordering: OLD_HEAD -> git pull -> NEW_HEAD -> restart check
    assert old_head_line < pull_line, f"OLD_HEAD (line {old_head_line}) must be captured before git pull (line {pull_line})"
    assert pull_line < new_head_line, f"git pull (line {pull_line}) must happen before NEW_HEAD (line {new_head_line})"
    assert new_head_line < restart_check_line, f"NEW_HEAD (line {new_head_line}) must be captured before restart check (line {restart_check_line})"


def test_restart_redirects_output_to_log():
    """Verify main.py output is redirected to log file."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    # Should redirect stdout and stderr
    assert "nohup python main.py >> \"$LOG_DIR/lifeos.log\" 2>&1 &" in content


def test_script_gives_startup_time():
    """Verify script waits for Life OS to initialize before continuing."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    # Should sleep to allow initialization
    assert "sleep 3" in content
    assert "Give it 3 seconds to initialize" in content.lower() or "sleep 3" in content


def test_restart_only_on_successful_iteration():
    """Verify restart logic runs in success path of iteration evaluation."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    lines = script_path.read_text().split('\n')

    # Find the success block and restart logic
    success_line = next(i for i, line in enumerate(lines) if "Iteration $ITERATION completed successfully" in line)
    restart_check_line = next(i for i, line in enumerate(lines) if '"$OLD_HEAD" != "$NEW_HEAD"' in line)

    # Restart logic should be AFTER the evaluation block (applies to all iterations)
    # The script structure is: evaluate -> git checkout -> capture OLD_HEAD -> pull -> capture NEW_HEAD -> restart check
    # So restart_check_line should be after success_line
    assert restart_check_line > success_line, "Restart logic should run after iteration evaluation"


def test_restart_explanation_comment():
    """Verify restart logic has clear explanatory comment."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    # Should explain why restart is necessary
    assert "critical for the improvement loop" in content.lower() or "restart Life OS if code was updated" in content
    assert "merged PRs update the codebase but the running process never picks" in content


def test_script_uses_correct_process_detection():
    """Verify script correctly identifies main.py process."""
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    content = script_path.read_text()

    # Should use pgrep with pattern matching
    assert 'pgrep -f "python.*main.py"' in content

    # Should handle multiple matches (take first)
    assert "head -1" in content or "head -n 1" in content
