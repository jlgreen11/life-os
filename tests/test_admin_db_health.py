"""
Tests for admin database health endpoints:

- GET  /api/admin/db-integrity      — integrity_check + blob probes on all 5 databases
- POST /api/admin/rebuild-user-model — Runtime recovery of corrupted user_model.db
"""

import os
import sqlite3
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from storage.manager import DatabaseManager
from web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers — build a mock life_os with real or fake DB connections
# ---------------------------------------------------------------------------

def _make_life_os_with_real_db(db: DatabaseManager):
    """Build a mock life_os that uses a real DatabaseManager for DB operations.

    All non-DB attributes are stubbed so that ``create_web_app`` / ``register_routes``
    doesn't blow up accessing services unrelated to these endpoints.
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
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.5, stress_level=0.3, social_battery=0.5,
        cognitive_load=0.3, emotional_valence=0.5, confidence=0.5, trend="stable",
    ))
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


# ---------------------------------------------------------------------------
# GET /api/admin/db-integrity
# ---------------------------------------------------------------------------

class TestDbIntegrity:
    """Tests for the database integrity check endpoint."""

    def test_returns_all_five_databases(self, db):
        """Response contains status for all 5 databases with 'ok' status."""
        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/db-integrity")
        assert resp.status_code == 200

        data = resp.json()
        assert "databases" in data
        assert "checked_at" in data

        expected_dbs = ["events", "entities", "state", "user_model", "preferences"]
        for name in expected_dbs:
            assert name in data["databases"], f"Missing database: {name}"
            assert data["databases"][name]["status"] == "ok"
            assert data["databases"][name]["detail"] == "ok"

    def test_reports_corrupted_db(self, tmp_data_dir):
        """A corrupted database file is reported with 'corrupted' or 'error' status."""
        # Create a valid DB set first
        db = DatabaseManager(data_dir=tmp_data_dir)
        db.initialize_all()

        # Corrupt user_model.db by writing garbage bytes
        db_path = os.path.join(tmp_data_dir, "user_model.db")
        with open(db_path, "wb") as f:
            f.write(b"CORRUPTED GARBAGE DATA " * 100)

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/db-integrity")
        assert resp.status_code == 200

        data = resp.json()
        um_status = data["databases"]["user_model"]["status"]
        assert um_status in ("corrupted", "error"), f"Expected corrupted/error, got {um_status}"

        # Other databases should still be ok
        assert data["databases"]["events"]["status"] == "ok"
        assert data["databases"]["entities"]["status"] == "ok"

    def test_checked_at_is_iso_format(self, db):
        """The checked_at field is a valid ISO 8601 timestamp."""
        from datetime import datetime as dt

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/db-integrity")
        data = resp.json()

        # Should parse without error
        dt.fromisoformat(data["checked_at"])

    def test_detects_blob_probe_corruption(self, db):
        """Endpoint detects blob overflow page corruption via get_database_health().

        Simulates a scenario where PRAGMA quick_check passes but blob probes
        fail — the kind of corruption only caught by the thorough health check.
        """
        # Mock get_database_health to return blob probe failure for user_model
        fake_health = {
            "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
            "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
            "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
            "user_model": {
                "status": "corrupted",
                "errors": ["blob probe failed: database disk image is malformed"],
                "path": "/tmp/user_model.db",
                "size_bytes": 4096,
            },
            "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
        }
        life_os = _make_life_os_with_real_db(db)
        life_os.db.get_database_health = Mock(return_value=fake_health)

        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/db-integrity")
        assert resp.status_code == 200

        data = resp.json()
        assert data["databases"]["user_model"]["status"] == "corrupted"
        assert "blob probe failed" in data["databases"]["user_model"]["detail"]
        # Other databases should be ok
        assert data["databases"]["events"]["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /api/admin/rebuild-user-model
# ---------------------------------------------------------------------------

class TestRebuildUserModel:
    """Tests for the runtime user_model.db rebuild endpoint."""

    def test_skips_healthy_db(self, db):
        """When user_model.db passes full health check, status is 'skipped'."""
        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post("/api/admin/rebuild-user-model")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "skipped"
        assert "healthy" in data["reason"].lower()

    def test_rebuilds_corrupted_db(self, tmp_data_dir):
        """When user_model.db is corrupted, it gets rebuilt with fresh schema."""
        db = DatabaseManager(data_dir=tmp_data_dir)
        db.initialize_all()

        # Corrupt user_model.db
        db_path = os.path.join(tmp_data_dir, "user_model.db")
        with open(db_path, "wb") as f:
            f.write(b"CORRUPTED GARBAGE DATA " * 100)
        # Also remove WAL/SHM if they exist (they reference the old file)
        for ext in ("-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post("/api/admin/rebuild-user-model")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "rebuilt"
        assert "message" in data

        # Verify the database is now accessible and has the episodes table
        with db.get_connection("user_model") as conn:
            check = conn.execute("PRAGMA quick_check").fetchone()
            assert check[0] == "ok"
            # Verify key tables exist
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "episodes" in tables
            assert "semantic_facts" in tables
            assert "signal_profiles" in tables

    def test_rebuild_creates_backup(self, tmp_data_dir):
        """The corrupted file is backed up before rebuild."""
        db = DatabaseManager(data_dir=tmp_data_dir)
        db.initialize_all()

        # Corrupt user_model.db
        db_path = os.path.join(tmp_data_dir, "user_model.db")
        with open(db_path, "wb") as f:
            f.write(b"CORRUPTED GARBAGE DATA " * 100)
        for ext in ("-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)

        life_os = _make_life_os_with_real_db(db)
        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post("/api/admin/rebuild-user-model")
        assert resp.status_code == 200
        assert resp.json()["status"] == "rebuilt"

        # Check that a backup file exists (.corrupt. or .corrupted- prefix)
        backup_files = [
            f for f in os.listdir(tmp_data_dir)
            if "user_model.db" in f and ("corrupt" in f.lower())
        ]
        assert len(backup_files) >= 1, f"Expected backup file, found: {os.listdir(tmp_data_dir)}"

    def test_rebuild_proceeds_on_blob_probe_corruption(self, db):
        """Rebuild proceeds when blob probes detect corruption (not just B-tree).

        Simulates a scenario where PRAGMA quick_check would pass but blob
        overflow probes detect corruption — the rebuild should proceed.
        """
        # Mock get_database_health to report user_model as corrupted via blob probes
        fake_health = {
            "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
            "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
            "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
            "user_model": {
                "status": "corrupted",
                "errors": ["blob probe failed: database disk image is malformed"],
                "path": "/tmp/user_model.db",
                "size_bytes": 4096,
            },
            "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
        }
        life_os = _make_life_os_with_real_db(db)
        life_os.db.get_database_health = Mock(return_value=fake_health)

        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.post("/api/admin/rebuild-user-model")
        assert resp.status_code == 200

        data = resp.json()
        # Should NOT be "skipped" — it should attempt the rebuild
        assert data["status"] != "skipped", (
            "Rebuild was skipped despite blob probe corruption — "
            "the endpoint is still using quick_check instead of get_database_health()"
        )
