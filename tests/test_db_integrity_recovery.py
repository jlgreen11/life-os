"""
Tests for database integrity checking and auto-recovery.

Verifies that ``_check_and_recover_db`` uses ``PRAGMA integrity_check``
for user_model databases (catching corruption that ``quick_check`` misses)
and that ``get_database_health`` mirrors the same behaviour.
"""

import re
import sqlite3

import pytest

from storage.manager import DatabaseManager


@pytest.fixture()
def db_manager(tmp_path):
    """A DatabaseManager using a temporary data directory."""
    return DatabaseManager(data_dir=str(tmp_path))


@pytest.fixture()
def healthy_user_model_db(db_manager):
    """Initialise a healthy user_model.db and return the manager."""
    db_manager._init_user_model_db()
    return db_manager


def _corrupt_db(db_path):
    """Corrupt a SQLite database by truncating it to a partial page.

    Truncating to 512 bytes leaves the file header intact but removes
    the page data, which both quick_check and integrity_check will detect.
    """
    with open(db_path, "r+b") as f:
        f.truncate(512)


class _ConnectionWrapper:
    """Wraps a sqlite3.Connection to track executed SQL statements.

    Used for verifying which PRAGMA is invoked, since
    ``sqlite3.Connection.execute`` is a read-only C extension attribute
    that cannot be monkeypatched.
    """

    def __init__(self, real_conn, tracker: list):
        self._conn = real_conn
        self._tracker = tracker

    def execute(self, sql, *args, **kwargs):
        """Delegate to the real connection and record the SQL."""
        self._tracker.append(sql.lower().strip())
        return self._conn.execute(sql, *args, **kwargs)

    def close(self):
        return self._conn.close()

    def fetchone(self):
        return self._conn.fetchone()

    def fetchall(self):
        return self._conn.fetchall()

    def __getattr__(self, name):
        return getattr(self._conn, name)


# -----------------------------------------------------------------------
# _check_and_recover_db
# -----------------------------------------------------------------------


class TestCheckAndRecoverDb:
    """Tests for _check_and_recover_db corruption detection."""

    def test_healthy_db_returns_false(self, healthy_user_model_db):
        """A clean user_model.db should not trigger recovery."""
        result = healthy_user_model_db._check_and_recover_db("user_model")
        assert result is False

    def test_nonexistent_db_returns_false(self, db_manager):
        """A missing database file should not trigger recovery."""
        result = db_manager._check_and_recover_db("user_model")
        assert result is False

    def test_corrupt_user_model_detected_and_backed_up(self, healthy_user_model_db, tmp_path):
        """Corruption in user_model.db should be detected and the file backed up."""
        db_path = tmp_path / "user_model.db"
        assert db_path.exists()

        _corrupt_db(db_path)

        result = healthy_user_model_db._check_and_recover_db("user_model")
        assert result is True

        # The original file should have been renamed (backed up).
        assert not db_path.exists()

        # A backup file with .corrupt.<timestamp> suffix should exist.
        backups = list(tmp_path.glob("user_model.db.corrupt.*"))
        assert len(backups) >= 1
        # Verify the timestamp pattern in the backup name.
        assert re.search(r"\.corrupt\.\d{8}T\d{6}Z$", backups[0].name)

    def test_backup_naming_pattern(self, healthy_user_model_db, tmp_path):
        """Backup files follow the .corrupt.<YYYYMMDDTHHMMSSZ> naming convention."""
        db_path = tmp_path / "user_model.db"

        _corrupt_db(db_path)

        healthy_user_model_db._check_and_recover_db("user_model")

        backups = list(tmp_path.glob("user_model.db.corrupt.*"))
        assert len(backups) >= 1
        pattern = re.compile(r"^user_model\.db\.corrupt\.\d{8}T\d{6}Z$")
        assert pattern.match(backups[0].name), f"Unexpected backup name: {backups[0].name}"

    def test_wal_shm_sidecars_also_backed_up(self, healthy_user_model_db, tmp_path):
        """WAL and SHM sidecar files should also be backed up on corruption."""
        db_path = tmp_path / "user_model.db"

        # Create sidecar files that would exist during WAL mode.
        wal_path = tmp_path / "user_model.db-wal"
        shm_path = tmp_path / "user_model.db-shm"
        wal_path.write_bytes(b"wal data")
        shm_path.write_bytes(b"shm data")

        _corrupt_db(db_path)

        healthy_user_model_db._check_and_recover_db("user_model")

        # All three original files should be gone.
        assert not db_path.exists()
        assert not wal_path.exists()
        assert not shm_path.exists()

        # All three should have backup copies.
        assert len(list(tmp_path.glob("user_model.db.corrupt.*"))) >= 1
        assert len(list(tmp_path.glob("user_model.db-wal.corrupt.*"))) >= 1
        assert len(list(tmp_path.glob("user_model.db-shm.corrupt.*"))) >= 1


# -----------------------------------------------------------------------
# get_database_health
# -----------------------------------------------------------------------


class TestGetDatabaseHealth:
    """Tests for get_database_health integrity checking."""

    def test_healthy_db_reports_ok(self, db_manager):
        """All databases should report 'ok' after clean initialisation."""
        db_manager.initialize_all()
        health = db_manager.get_database_health()

        for db_name in ("events", "entities", "state", "user_model", "preferences"):
            assert health[db_name]["status"] == "ok", f"{db_name} should be ok"
            assert health[db_name]["errors"] == []

    def test_corrupt_user_model_reports_corrupted(self, db_manager, tmp_path):
        """A corrupted user_model.db should be reported as 'corrupted'."""
        db_manager.initialize_all()

        db_path = tmp_path / "user_model.db"
        _corrupt_db(db_path)

        health = db_manager.get_database_health()
        assert health["user_model"]["status"] == "corrupted"
        assert len(health["user_model"]["errors"]) > 0


# -----------------------------------------------------------------------
# Conditional check type verification
# -----------------------------------------------------------------------


class TestConditionalCheckType:
    """Verify that the correct PRAGMA is used per database name."""

    def test_user_model_uses_integrity_check(self, db_manager, monkeypatch):
        """_check_and_recover_db should use integrity_check for user_model."""
        db_manager._init_user_model_db()

        tracked_sql: list[str] = []
        original_connect = sqlite3.connect

        def tracking_connect(path, *args, **kwargs):
            """Return a wrapper that records SQL statements."""
            conn = original_connect(path, *args, **kwargs)
            return _ConnectionWrapper(conn, tracked_sql)

        monkeypatch.setattr(sqlite3, "connect", tracking_connect)

        db_manager._check_and_recover_db("user_model")

        pragma_calls = [s for s in tracked_sql if "integrity_check" in s or "quick_check" in s]
        assert any("integrity_check" in p for p in pragma_calls), (
            f"Expected integrity_check for user_model, got: {pragma_calls}"
        )

    def test_state_db_uses_quick_check(self, db_manager, monkeypatch):
        """_check_and_recover_db should use quick_check for non-user_model databases."""
        db_manager._init_state_db()

        tracked_sql: list[str] = []
        original_connect = sqlite3.connect

        def tracking_connect(path, *args, **kwargs):
            conn = original_connect(path, *args, **kwargs)
            return _ConnectionWrapper(conn, tracked_sql)

        monkeypatch.setattr(sqlite3, "connect", tracking_connect)

        db_manager._check_and_recover_db("state")

        pragma_calls = [s for s in tracked_sql if "integrity_check" in s or "quick_check" in s]
        assert any("quick_check" in p for p in pragma_calls), (
            f"Expected quick_check for state, got: {pragma_calls}"
        )
        assert not any("integrity_check" in p for p in pragma_calls), (
            "state should NOT use integrity_check"
        )

    def test_health_check_uses_integrity_for_user_model(self, db_manager, monkeypatch):
        """get_database_health should use integrity_check for user_model."""
        db_manager.initialize_all()

        tracked_sql: list[str] = []
        original_connect = sqlite3.connect

        def tracking_connect(path, *args, **kwargs):
            conn = original_connect(path, *args, **kwargs)
            return _ConnectionWrapper(conn, tracked_sql)

        monkeypatch.setattr(sqlite3, "connect", tracking_connect)

        db_manager.get_database_health()

        # Find all pragma calls that include a db_name context.
        integrity_calls = [s for s in tracked_sql if "integrity_check" in s]
        quick_calls = [s for s in tracked_sql if "quick_check" in s]

        # We expect at least one integrity_check (for user_model) and
        # at least one quick_check (for the other four databases).
        assert len(integrity_calls) >= 1, (
            f"Expected at least 1 integrity_check call, got: {integrity_calls}"
        )
        assert len(quick_calls) >= 1, (
            f"Expected at least 1 quick_check call, got: {quick_calls}"
        )
