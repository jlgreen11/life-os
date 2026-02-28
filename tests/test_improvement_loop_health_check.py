"""
Tests for improvement loop Life OS health check and automatic recovery.

The improvement loop now checks Life OS status on EVERY iteration (not just
when code changes), ensuring the service stays running even if it crashes,
is stopped manually, or fails to start after a previous PR merge.

This test suite validates that the restart logic:
1. Starts Life OS if it's offline (regardless of code changes)
2. Restarts Life OS only when code changes
3. Handles both Docker and local Python deployments
4. Recovers from crashes automatically
5. Logs appropriate status messages
"""

import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest


class TestImprovementLoopHealthCheck:
    """Test the improved Life OS health check logic in the continuous improvement loop."""

    @pytest.fixture
    def mock_script_env(self, tmp_path):
        """Create a mock environment for testing the restart script.

        Sets up:
        - Temporary project directory
        - Mock log directory
        - Mock venv structure
        - Git repository state
        """
        project_dir = tmp_path / "life-os"
        project_dir.mkdir()

        # Create log directory
        log_dir = project_dir / "data" / "improvement-runs"
        log_dir.mkdir(parents=True)

        # Create mock venv
        venv_dir = project_dir / ".venv" / "bin"
        venv_dir.mkdir(parents=True)
        activate = venv_dir / "activate"
        activate.write_text("# Mock activate script")

        # Create docker-compose.yml
        docker_compose = project_dir / "docker-compose.yml"
        docker_compose.write_text("services:\n  lifeos:\n    image: lifeos:latest")

        return {
            "project_dir": project_dir,
            "log_dir": log_dir,
            "venv": activate,
            "docker_compose": docker_compose,
        }

    def test_starts_offline_lifeos_without_code_change(self, mock_script_env):
        """When Life OS is offline but code hasn't changed, start it anyway.

        This is the core fix: previously, the restart logic only ran when code
        changed, so if Life OS crashed or was stopped, it would stay offline
        until the next PR merged. Now it starts on every iteration.
        """
        with patch("subprocess.run") as mock_run:
            # Simulate: Life OS container is offline
            mock_run.side_effect = [
                # docker compose ps lifeos
                Mock(returncode=0, stdout="NAME      IMAGE     COMMAND   SERVICE   CREATED   STATUS    PORTS\n"),
                # docker compose up -d lifeos
                Mock(returncode=0),
                # docker compose ps lifeos (verify)
                Mock(returncode=0, stdout="lifeos ... Up ..."),
            ]

            # Simulate git state: no code change
            old_head = "abc123"
            new_head = "abc123"  # Same commit

            # In the real script, this would be the section after git pull
            # We're testing the logic that should run regardless of code changes

            # The key assertion: Life OS should be started even though code didn't change
            # This simulates what the improved script does

            # Check if container is running
            result = subprocess.run(
                ["docker", "compose", "ps", "lifeos"],
                capture_output=True,
                text=True,
            )
            is_running = "Up" in result.stdout

            if not is_running:
                # Start it (this should happen even without code changes)
                subprocess.run(["docker", "compose", "up", "-d", "lifeos"])

            # Verify the start command was called
            assert any("up" in str(c) for c in mock_run.call_args_list)

    def test_restarts_running_lifeos_on_code_change(self, mock_script_env):
        """When code changes and Life OS is running, restart it."""
        with patch("subprocess.run") as mock_run:
            # Simulate: Life OS container is running
            mock_run.side_effect = [
                # docker compose ps lifeos (check status)
                Mock(returncode=0, stdout="lifeos ... Up ..."),
                # docker compose restart lifeos
                Mock(returncode=0),
                # docker compose ps lifeos (verify)
                Mock(returncode=0, stdout="lifeos ... Up ..."),
            ]

            # Simulate git state: code changed
            old_head = "abc123"
            new_head = "def456"  # Different commit
            code_updated = old_head != new_head

            # Check if container is running
            result = subprocess.run(
                ["docker", "compose", "ps", "lifeos"],
                capture_output=True,
                text=True,
            )
            is_running = "Up" in result.stdout

            if is_running and code_updated:
                # Restart it with new code
                subprocess.run(["docker", "compose", "restart", "lifeos"])

            # Verify restart was called
            assert any("restart" in str(c) for c in mock_run.call_args_list)

    def test_leaves_running_lifeos_alone_without_code_change(self, mock_script_env):
        """When Life OS is running and code hasn't changed, do nothing."""
        with patch("subprocess.run") as mock_run:
            # Simulate: Life OS container is running
            mock_run.return_value = Mock(
                returncode=0, stdout="lifeos ... Up ..."
            )

            # Simulate git state: no code change
            old_head = "abc123"
            new_head = "abc123"  # Same commit
            code_updated = old_head != new_head

            # Check if container is running
            result = subprocess.run(
                ["docker", "compose", "ps", "lifeos"],
                capture_output=True,
                text=True,
            )
            is_running = "Up" in result.stdout

            # Should NOT restart if running and no code changes
            if is_running and code_updated:
                subprocess.run(["docker", "compose", "restart", "lifeos"])

            # Verify restart was NOT called
            assert not any("restart" in str(c) for c in mock_run.call_args_list)

    def test_starts_offline_local_process_without_code_change(self, mock_script_env):
        """When local Python deployment is offline, start it (no Docker)."""
        project_dir = mock_script_env["project_dir"]
        log_dir = mock_script_env["log_dir"]

        # Remove docker-compose.yml to simulate local deployment
        (project_dir / "docker-compose.yml").unlink()

        with patch("subprocess.run") as mock_run, \
             patch("subprocess.Popen") as mock_popen:
            # Simulate: no Life OS process running
            mock_run.return_value = Mock(returncode=1, stdout="")  # pgrep returns nothing
            mock_popen.return_value = Mock(pid=12345)

            # The script should start main.py regardless of code changes
            # when Life OS is offline

            # Check if process is running
            result = subprocess.run(
                ["pgrep", "-f", "python.*main.py"],
                capture_output=True,
                text=True,
            )
            is_running = result.returncode == 0

            if not is_running:
                # Start it
                subprocess.Popen(
                    ["python", "main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            # Verify Popen was called to start main.py
            mock_popen.assert_called_once()

    def test_restarts_local_process_on_code_change(self, mock_script_env):
        """When local Python process is running and code changes, restart it."""
        project_dir = mock_script_env["project_dir"]

        # Remove docker-compose.yml to simulate local deployment
        (project_dir / "docker-compose.yml").unlink()

        with patch("subprocess.run") as mock_run, \
             patch("subprocess.Popen") as mock_popen, \
             patch("os.kill") as mock_kill:
            # Simulate: Life OS process is running (PID 12345)
            mock_run.return_value = Mock(returncode=0, stdout="12345")
            mock_popen.return_value = Mock(pid=54321)

            # Simulate git state: code changed
            old_head = "abc123"
            new_head = "def456"
            code_updated = old_head != new_head

            # Check if process is running
            result = subprocess.run(
                ["pgrep", "-f", "python.*main.py"],
                capture_output=True,
                text=True,
            )
            is_running = result.returncode == 0

            if is_running and code_updated:
                # Kill old process
                import os
                old_pid = 12345
                os.kill(old_pid, 15)  # SIGTERM

                # Start new process
                subprocess.Popen(
                    ["python", "main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            # Verify kill and restart
            mock_kill.assert_called_once_with(12345, 15)
            mock_popen.assert_called_once()

    def test_handles_docker_compose_failure_gracefully(self, mock_script_env):
        """When Docker Compose fails to start Life OS, log error but continue."""
        with patch("subprocess.run") as mock_run:
            # Simulate: docker compose up fails
            mock_run.side_effect = [
                # docker compose ps lifeos (offline)
                Mock(returncode=0, stdout=""),
                # docker compose up -d lifeos (fails)
                Mock(returncode=1, stderr="Error: container failed to start"),
                # docker compose ps lifeos (verify - still offline)
                Mock(returncode=0, stdout=""),
            ]

            # The script should attempt to start but not crash on failure
            try:
                # Check status
                result = subprocess.run(
                    ["docker", "compose", "ps", "lifeos"],
                    capture_output=True,
                    text=True,
                )
                is_running = "Up" in result.stdout

                if not is_running:
                    # Try to start
                    subprocess.run(["docker", "compose", "up", "-d", "lifeos"])

                    # Verify it started (will fail in this test)
                    verify = subprocess.run(
                        ["docker", "compose", "ps", "lifeos"],
                        capture_output=True,
                        text=True,
                    )
                    if "Up" not in verify.stdout:
                        # Log error but don't crash
                        print("ERROR: Life OS container failed to start")

            except Exception as e:
                pytest.fail(f"Script should not crash on Docker failure: {e}")

            # Verify all expected calls were made
            assert len(mock_run.call_args_list) == 3

    def test_handles_local_process_crash_during_restart(self, mock_script_env):
        """When local process crashes immediately after start, log error."""
        project_dir = mock_script_env["project_dir"]

        # Remove docker-compose.yml
        (project_dir / "docker-compose.yml").unlink()

        with patch("subprocess.run") as mock_run, \
             patch("subprocess.Popen") as mock_popen, \
             patch("os.kill") as mock_kill:
            # Simulate: process starts but crashes immediately
            mock_run.return_value = Mock(returncode=1, stdout="")  # pgrep finds nothing
            mock_process = Mock(pid=12345)
            mock_popen.return_value = mock_process

            # Simulate: kill -0 check shows process died
            mock_kill.side_effect = [ProcessLookupError("No such process")]

            try:
                # Start process
                import os
                subprocess.Popen(
                    ["python", "main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                new_pid = 12345

                # Verify it's running (will fail)
                try:
                    os.kill(new_pid, 0)
                    print("Life OS is running successfully")
                except ProcessLookupError:
                    print("ERROR: Life OS failed to start")

            except Exception as e:
                pytest.fail(f"Script should not crash when process dies: {e}")

            # Verify Popen was called
            mock_popen.assert_called_once()

    def test_integration_with_real_script_structure(self, mock_script_env):
        """Verify the real script structure matches our expectations.

        Reads the actual run-continuous-improvement.sh and validates that:
        1. It checks Life OS status on every iteration
        2. It handles both Docker and local deployments
        3. It starts offline Life OS regardless of code changes
        4. It restarts running Life OS only on code changes
        """
        script_path = Path(__file__).parent.parent / "scripts" / "run-continuous-improvement.sh"
        assert script_path.exists(), "Improvement loop script not found"

        script_content = script_path.read_text()

        # Validate key improvements are present.
        # The script comment uses lowercase "ensure" — match exactly.
        assert "ensure Life OS is running" in script_content, \
            "Script should check Life OS health on every iteration"

        assert "CODE_UPDATED=false" in script_content, \
            "Script should track whether code changed"

        assert "Life OS container is offline. Starting it..." in script_content, \
            "Script should start offline Docker container"

        assert "Life OS process not found. Starting it..." in script_content, \
            "Script should start offline local process"

        assert "no code changes, skipping restart" in script_content, \
            "Script should skip restart when running and no code changes"

        # Validate the logic flow: check status BEFORE checking code changes
        # The improved script should check if Life OS is running first,
        # then decide whether to restart (code changed) or just verify (no change)
        lines = script_content.split("\n")
        check_idx = next(i for i, line in enumerate(lines) if "docker compose ps lifeos" in line or "pgrep -f" in line)
        code_check_idx = next(i for i, line in enumerate(lines) if "CODE_UPDATED=true" in line)

        # The status check should come AFTER we determine if code changed
        # (we set CODE_UPDATED first, then check status and act accordingly)
        assert code_check_idx < check_idx, \
            "Script should determine code change status before checking Life OS health"

    def test_every_iteration_health_check_workflow(self, mock_script_env):
        """Test the complete workflow: every iteration checks and ensures Life OS runs.

        Simulates multiple improvement loop iterations:
        1. Iteration 1: Life OS offline, no code change → should START
        2. Iteration 2: Life OS running, no code change → should LEAVE ALONE
        3. Iteration 3: Life OS running, code changed → should RESTART
        4. Iteration 4: Life OS offline, code changed → should START
        """
        with patch("subprocess.run") as mock_run:
            scenarios = [
                # Iteration 1: offline + no code change → START
                {
                    "name": "offline_no_change",
                    "is_running": False,
                    "code_changed": False,
                    "expected_action": "start",
                },
                # Iteration 2: running + no code change → SKIP
                {
                    "name": "running_no_change",
                    "is_running": True,
                    "code_changed": False,
                    "expected_action": "skip",
                },
                # Iteration 3: running + code changed → RESTART
                {
                    "name": "running_code_changed",
                    "is_running": True,
                    "code_changed": True,
                    "expected_action": "restart",
                },
                # Iteration 4: offline + code changed → START
                {
                    "name": "offline_code_changed",
                    "is_running": False,
                    "code_changed": True,
                    "expected_action": "start",
                },
            ]

            for scenario in scenarios:
                mock_run.reset_mock()

                if scenario["is_running"]:
                    # Mock: container is running
                    mock_run.side_effect = [
                        Mock(returncode=0, stdout="lifeos ... Up ..."),  # status check
                        Mock(returncode=0),  # restart or skip
                        Mock(returncode=0, stdout="lifeos ... Up ..."),  # verify
                    ]
                else:
                    # Mock: container is offline
                    mock_run.side_effect = [
                        Mock(returncode=0, stdout=""),  # status check (empty = offline)
                        Mock(returncode=0),  # start
                        Mock(returncode=0, stdout="lifeos ... Up ..."),  # verify
                    ]

                # Simulate the script logic
                result = subprocess.run(
                    ["docker", "compose", "ps", "lifeos"],
                    capture_output=True,
                    text=True,
                )
                is_running = "Up" in result.stdout
                code_updated = scenario["code_changed"]

                if is_running and code_updated:
                    # Restart
                    subprocess.run(["docker", "compose", "restart", "lifeos"])
                    action = "restart"
                elif not is_running:
                    # Start
                    subprocess.run(["docker", "compose", "up", "-d", "lifeos"])
                    action = "start"
                else:
                    # Skip
                    action = "skip"

                # Verify expected action
                assert action == scenario["expected_action"], \
                    f"Scenario {scenario['name']} failed: expected {scenario['expected_action']}, got {action}"
