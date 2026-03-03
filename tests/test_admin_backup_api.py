"""
Tests for admin backup listing and restore endpoints:

- GET  /api/admin/backups          — List available database backups
- POST /api/admin/backups/restore  — Restore a database from a backup file
"""

import os
import sqlite3
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from storage.manager import DatabaseManager
from web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers — build a mock life_os with real DB
# ---------------------------------------------------------------------------


def _make_life_os_with_real_db(db: DatabaseManager):
    """Build a mock life_os that uses a real DatabaseManager for DB operations.

    All non-DB attributes are stubbed so that ``create_web_app`` /
    ``register_routes`` doesn't blow up accessing unrelated services.
    """
    life_os = Mock()
    life_os.config = {}
    life_os.db = db

    # Stubs required by register_routes
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0})
    life_os.vector_store.search = Mock(return_value=[])
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={})
    life_os.signal_extractor.get_current_mood = Mock(
        return_value=Mock(
            energy_level=0.5,
            stress_level=0.3,
            social_battery=0.5,
            cognitive_load=0.3,
            emotional_valence=0.5,
            confidence=0.5,
            trend="stable",
        )
    )
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={})
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={})
    life_os.ai_engine = Mock()
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})

    return life_os


def _create_fake_backup(backup_dir: str, db_name: str, timestamp: str) -> str:
    """Create a valid SQLite backup file in the given directory.

    Args:
        backup_dir: Directory to create the backup in.
        db_name: Logical database name (e.g. ``"user_model"``).
        timestamp: Timestamp string in ``YYYYMMDDTHHMMSS`` format.

    Returns:
        Absolute path to the created backup file.
    """
    os.makedirs(backup_dir, exist_ok=True)
    filename = f"{db_name}_{timestamp}.db"
    path = os.path.join(backup_dir, filename)
    # Create a valid SQLite database so integrity checks pass
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE _backup_marker (id INTEGER PRIMARY KEY)")
    conn.close()
    return path


# ---------------------------------------------------------------------------
# GET /api/admin/backups
# ---------------------------------------------------------------------------


class TestListBackups:
    """Tests for the backup listing endpoint."""

    def test_returns_empty_when_no_backups(self, db):
        """Response lists zero backups when no backup files exist."""
        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/backups?db_name=user_model")
        assert resp.status_code == 200

        data = resp.json()
        assert data["db_name"] == "user_model"
        assert data["backups"] == []
        assert data["count"] == 0

    def test_returns_backups_sorted_newest_first(self, tmp_data_dir):
        """Backups are returned newest-first when multiple exist."""
        db = DatabaseManager(data_dir=tmp_data_dir)
        db.initialize_all()

        backup_dir = os.path.join(tmp_data_dir, "backups")
        _create_fake_backup(backup_dir, "user_model", "20260301T100000")
        _create_fake_backup(backup_dir, "user_model", "20260303T120000")
        _create_fake_backup(backup_dir, "user_model", "20260302T080000")

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/backups?db_name=user_model")
        assert resp.status_code == 200

        data = resp.json()
        assert data["count"] == 3
        # Verify newest first
        timestamps = [b["created_at"] for b in data["backups"]]
        assert timestamps == sorted(timestamps, reverse=True)
        assert "20260303" in data["backups"][0]["filename"]

    def test_rejects_invalid_db_name(self, db):
        """Invalid db_name returns HTTP 400."""
        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/backups?db_name=not_a_database")
        assert resp.status_code == 400
        assert "Invalid db_name" in resp.json()["detail"]

    def test_defaults_to_user_model(self, db):
        """When no db_name is specified, defaults to user_model."""
        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/backups")
        assert resp.status_code == 200
        assert resp.json()["db_name"] == "user_model"

    def test_filters_by_db_name(self, tmp_data_dir):
        """Backups for different databases are not mixed together."""
        db = DatabaseManager(data_dir=tmp_data_dir)
        db.initialize_all()

        backup_dir = os.path.join(tmp_data_dir, "backups")
        _create_fake_backup(backup_dir, "user_model", "20260301T100000")
        _create_fake_backup(backup_dir, "events", "20260302T120000")

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/backups?db_name=events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["backups"][0]["db_name"] == "events"


# ---------------------------------------------------------------------------
# POST /api/admin/backups/restore
# ---------------------------------------------------------------------------


class TestRestoreBackup:
    """Tests for the backup restore endpoint."""

    def test_restores_valid_backup(self, tmp_data_dir):
        """A valid backup is successfully restored."""
        db = DatabaseManager(data_dir=tmp_data_dir)
        db.initialize_all()

        backup_dir = os.path.join(tmp_data_dir, "backups")
        backup_path = _create_fake_backup(backup_dir, "user_model", "20260303T120000")

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/backups/restore",
            json={"backup_path": backup_path, "db_name": "user_model"},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "restored"
        assert data["db_name"] == "user_model"

    def test_rejects_missing_backup_path(self, db):
        """Request without backup_path returns HTTP 422 (Pydantic validation)."""
        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post("/api/admin/backups/restore", json={"db_name": "user_model"})
        assert resp.status_code == 422

    def test_rejects_path_traversal(self, tmp_data_dir):
        """Paths outside data/backups/ are rejected (path traversal protection)."""
        db = DatabaseManager(data_dir=tmp_data_dir)
        db.initialize_all()

        # Create a file outside the backups directory
        evil_path = os.path.join(tmp_data_dir, "evil.db")
        conn = sqlite3.connect(evil_path)
        conn.execute("CREATE TABLE x (id INTEGER)")
        conn.close()

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/backups/restore",
            json={"backup_path": evil_path, "db_name": "user_model"},
        )
        assert resp.status_code == 400
        assert "backups directory" in resp.json()["detail"]

    def test_rejects_nonexistent_backup(self, tmp_data_dir):
        """A path that doesn't exist returns HTTP 404."""
        db = DatabaseManager(data_dir=tmp_data_dir)
        db.initialize_all()

        # Ensure the backups directory exists so path traversal check passes
        backup_dir = os.path.join(tmp_data_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        fake_path = os.path.join(backup_dir, "user_model_20260101T000000.db")

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/backups/restore",
            json={"backup_path": fake_path, "db_name": "user_model"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_rejects_invalid_db_name(self, db):
        """Invalid db_name in restore request returns HTTP 400."""
        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/backups/restore",
            json={"backup_path": "/some/path.db", "db_name": "invalid_db"},
        )
        assert resp.status_code == 400
        assert "Invalid db_name" in resp.json()["detail"]

    def test_restore_archives_current_db(self, tmp_data_dir):
        """Restoring creates a .pre_restore.db archive of the current database."""
        db = DatabaseManager(data_dir=tmp_data_dir)
        db.initialize_all()

        backup_dir = os.path.join(tmp_data_dir, "backups")
        backup_path = _create_fake_backup(backup_dir, "user_model", "20260303T120000")

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/backups/restore",
            json={"backup_path": backup_path, "db_name": "user_model"},
        )
        assert resp.status_code == 200

        # The restore_from_backup method archives the old DB
        archive_path = os.path.join(tmp_data_dir, "user_model.pre_restore.db")
        assert os.path.exists(archive_path), (
            f"Expected archive at {archive_path}, found: {os.listdir(tmp_data_dir)}"
        )

    def test_rejects_dotdot_traversal(self, tmp_data_dir):
        """Paths with .. that escape the backups directory are rejected."""
        db = DatabaseManager(data_dir=tmp_data_dir)
        db.initialize_all()

        # Create the backups directory so it exists
        backup_dir = os.path.join(tmp_data_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)

        # Construct a path that uses .. to escape
        traversal_path = os.path.join(backup_dir, "..", "user_model.db")

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/backups/restore",
            json={"backup_path": traversal_path, "db_name": "user_model"},
        )
        # Should be rejected — the resolved path is outside data/backups/
        assert resp.status_code == 400
        assert "backups directory" in resp.json()["detail"]
