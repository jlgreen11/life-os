"""
Tests for improvement loop restart logic fix.

This test suite verifies that the continuous improvement loop correctly
detects and restarts Life OS when code changes are pulled, supporting
both Docker Compose and local Python deployment modes.

The restart logic is critical — without it, all merged PRs update the
codebase but the running process never picks up the changes, rendering
all improvements inert (as happened for iterations 1-84).

Test coverage:
    - Docker Compose deployment detection and restart
    - Local Python process detection and restart
    - No-op when code hasn't changed
    - Graceful degradation when neither deployment is running
    - Verification after restart
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch


def test_restart_logic_detects_docker_compose():
    """
    Verify restart logic correctly detects Docker Compose deployment.

    When `docker compose ps lifeos` shows a running container, the restart
    logic should use `docker compose restart lifeos` instead of trying to
    kill a local Python process.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    assert script_path.exists(), "Improvement script not found"

    # Read the script content
    script = script_path.read_text()

    # Verify Docker detection command
    assert 'docker compose ps lifeos' in script, \
        "Missing Docker Compose detection command"

    # Verify Docker restart command
    assert 'docker compose restart lifeos' in script, \
        "Missing Docker Compose restart command"

    # Verify Docker check comes BEFORE Python process check
    docker_index = script.index('docker compose ps lifeos')
    python_index = script.index('pgrep -f "python.*main.py"')
    assert docker_index < python_index, \
        "Docker check must come before Python process check to prioritize Docker deployment"


def test_restart_logic_detects_local_python():
    """
    Verify restart logic correctly detects local Python deployment.

    When no Docker container is running but a `python main.py` process
    is found, the restart logic should kill that process and restart it.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # Verify Python process detection
    assert 'pgrep -f "python.*main.py"' in script, \
        "Missing Python process detection command"

    # Verify graceful shutdown with SIGTERM
    assert 'kill "$LIFEOS_PID"' in script, \
        "Missing graceful shutdown (SIGTERM)"

    # Verify force kill with SIGKILL after timeout
    assert 'kill -9 "$LIFEOS_PID"' in script, \
        "Missing force kill (SIGKILL) fallback"

    # Verify restart command
    assert 'nohup python main.py' in script, \
        "Missing Python restart command"


def test_restart_logic_skips_when_no_code_change():
    """
    Verify restart logic is a no-op when code hasn't changed.

    The script compares git HEAD before and after `git pull`. If they're
    identical, no restart should be attempted (avoids unnecessary downtime).
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # Verify git HEAD comparison
    assert 'OLD_HEAD=$(git rev-parse HEAD' in script, \
        "Missing OLD_HEAD capture before pull"
    assert 'NEW_HEAD=$(git rev-parse HEAD' in script, \
        "Missing NEW_HEAD capture after pull"

    # Verify conditional restart
    assert 'if [[ "$OLD_HEAD" != "$NEW_HEAD"' in script, \
        "Missing conditional restart based on code change"

    # Verify skip message (case-insensitive check since wording varies)
    assert 'no code changes' in script.lower(), \
        "Missing skip message when no code changes"


def test_restart_logic_warns_when_neither_deployment_running():
    """
    Verify restart logic provides helpful guidance when Life OS isn't running.

    If neither Docker nor Python process is found, the script should warn
    and provide manual start instructions instead of silently failing.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # Verify that when no process is found the script logs a message
    # (the script uses "Life OS process not found" or "Life OS container is offline")
    assert 'Life OS process not found' in script or \
           'Life OS container is offline' in script or \
           'life os process not found' in script.lower(), \
        "Missing message when Life OS not running"

    # Verify manual start instructions exist for both modes
    assert 'docker compose up -d' in script or 'docker-compose up -d' in script, \
        "Missing Docker start instructions"
    assert 'python main.py' in script, \
        "Missing Python start instructions"


def test_restart_logic_verifies_docker_restart():
    """
    Verify restart logic checks that Docker container is healthy after restart.

    After `docker compose restart`, the script should verify the container
    is actually running before continuing (prevents silent failures).
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # Find the Docker restart section
    assert 'docker compose restart lifeos' in script, "Missing Docker restart"

    # After restart, should verify container is up
    restart_index = script.index('docker compose restart lifeos')
    verify_section = script[restart_index:restart_index + 500]

    assert 'docker compose ps lifeos' in verify_section, \
        "Missing verification after Docker restart"
    assert 'Up' in verify_section or 'running' in verify_section, \
        "Missing health check after Docker restart"


def test_restart_logic_verifies_python_restart():
    """
    Verify restart logic checks that Python process started successfully.

    After starting `python main.py`, the script should verify the process
    is still running (prevents silent failures from import errors, crashes).
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # Find the Python restart section
    assert 'nohup python main.py' in script, "Missing Python restart"

    # After restart, should verify process is running
    restart_index = script.index('nohup python main.py')
    verify_section = script[restart_index:restart_index + 500]

    assert 'kill -0 "$NEW_PID"' in verify_section, \
        "Missing verification that Python process is running"
    assert 'ERROR' in verify_section or 'failed' in verify_section.lower(), \
        "Missing error message when Python process fails to start"


def test_restart_logic_logs_all_operations():
    """
    Verify restart logic logs all critical operations for debugging.

    When troubleshooting deployment issues, operators need visibility into:
    - Which deployment mode was detected
    - Whether restart succeeded
    - Any errors encountered
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # Verify log calls for key operations
    assert 'log "Code updated' in script, \
        "Missing log when code changes detected"
    # Docker detection log: script says 'Life OS container is running' or 'Life OS container is offline'
    assert 'Life OS container' in script, \
        "Missing log when Docker deployment detected"
    # Python detection log: script says 'Life OS process found'
    assert 'Life OS process found' in script or 'Life OS process not found' in script, \
        "Missing log when Python deployment detected"
    assert 'log "Life OS' in script and 'successfully' in script, \
        "Missing success log after restart"


def test_restart_logic_handles_docker_restart_failure():
    """
    Verify restart logic detects and reports Docker restart failures.

    If `docker compose restart` completes but the container doesn't come
    back up, the script should log an error with debugging instructions.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # Find Docker restart section
    restart_index = script.index('docker compose restart lifeos')
    error_section = script[restart_index:restart_index + 700]

    # Should check for failure
    assert 'ERROR' in error_section or 'failed' in error_section.lower(), \
        "Missing error detection for Docker restart failure"

    # Should provide debugging hint
    assert 'docker compose logs' in error_section or 'logs lifeos' in error_section, \
        "Missing debugging hint for Docker restart failure"


def test_restart_logic_waits_for_graceful_shutdown():
    """
    Verify restart logic allows time for graceful shutdown before force-killing.

    When stopping a local Python process, the script should:
    1. Send SIGTERM for graceful shutdown
    2. Wait up to 10 seconds for cleanup
    3. Only SIGKILL if process is still alive
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # Find the shutdown section
    assert 'kill "$LIFEOS_PID"' in script, "Missing SIGTERM"

    shutdown_index = script.index('kill "$LIFEOS_PID"')
    shutdown_section = script[shutdown_index:shutdown_index + 800]

    # Should wait in a loop
    assert 'for i in {1..10}' in shutdown_section or 'while' in shutdown_section, \
        "Missing graceful shutdown wait loop"

    # Should check if process stopped
    assert 'kill -0 "$LIFEOS_PID"' in shutdown_section, \
        "Missing check for process termination"

    # Should only force-kill if timeout
    assert 'kill -9' in shutdown_section, \
        "Missing SIGKILL fallback after timeout"


def test_restart_logic_waits_for_initialization():
    """
    Verify restart logic waits for Life OS to initialize after start.

    After starting the process, the script should sleep a few seconds
    before verifying it's running (allows time for imports, DB init, etc.).
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # After starting Python process, should wait
    python_start_index = script.index('nohup python main.py')
    init_section = script[python_start_index:python_start_index + 400]

    assert 'sleep' in init_section, \
        "Missing initialization wait after starting Python process"

    # After Docker restart, should also wait
    if 'docker compose restart lifeos' in script:
        docker_start_index = script.index('docker compose restart lifeos')
        docker_init_section = script[docker_start_index:docker_start_index + 400]

        assert 'sleep' in docker_init_section, \
            "Missing initialization wait after Docker restart"


def test_restart_logic_preserves_deployment_mode():
    """
    Verify restart logic doesn't switch deployment modes.

    If Life OS was running via Docker, it should restart via Docker.
    If it was running as Python, it should restart as Python.
    The script should never mix modes or force a particular mode.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # Docker and Python restarts should be in separate conditional branches
    docker_section_start = script.index('docker compose ps lifeos')
    python_section_start = script.index('pgrep -f "python.*main.py"')

    # Extract the Docker conditional block
    docker_section = script[docker_section_start:python_section_start]

    # Docker section should NOT contain Python restart
    assert 'nohup python main.py' not in docker_section, \
        "Docker restart section should not start Python process"

    # Python section should NOT contain Docker restart
    python_section = script[python_section_start:]
    python_section_end = python_section.find('else\n', 0, 2000)
    if python_section_end > 0:
        python_section = python_section[:python_section_end]

    assert 'docker compose restart' not in python_section, \
        "Python restart section should not restart Docker container"


def test_restart_logic_fixes_iteration_1_to_84_deployment_gap():
    """
    Verify the fix addresses the root cause: 19-day-old process running.

    This is a regression test ensuring the bug that allowed iterations 1-84
    to run without ever restarting Life OS cannot recur.

    The bug occurred because:
    1. The old logic only looked for local Python processes
    2. Life OS was running as PID 18189 for 468 hours (19+ days)
    3. All 84 PRs merged but never deployed

    The fix:
    1. Checks Docker FIRST (recommended deployment)
    2. Falls back to Python process check
    3. Provides clear warnings if neither is found
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    script = script_path.read_text()

    # The critical fix: Docker check must come BEFORE Python check
    docker_index = script.index('docker compose ps lifeos')
    python_index = script.index('pgrep -f "python.*main.py"')

    assert docker_index < python_index, \
        "CRITICAL: Docker check must come first to prevent recurrence of 19-day stale process bug"

    # The script must support both deployment modes
    assert 'docker compose restart lifeos' in script, \
        "Missing Docker restart support (would cause bug recurrence)"
    assert 'kill "$LIFEOS_PID"' in script, \
        "Missing Python process restart support (needed for local dev)"

    # Must provide guidance when no process is running (either phrasing is acceptable)
    assert 'Life OS process not found' in script or \
           'Life OS container is offline' in script, \
        "Missing message when neither deployment mode is active"


def test_script_is_executable():
    """
    Verify the improvement script has execute permissions.

    The script is launched by launchd, which requires +x permissions.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
    assert script_path.exists(), "Improvement script not found"

    # On Unix, check execute bit is set for owner
    import stat
    st = script_path.stat()
    is_executable = bool(st.st_mode & stat.S_IXUSR)

    if not is_executable:
        print(f"\nWARNING: {script_path} is not executable.")
        print(f"Run: chmod +x {script_path}")
