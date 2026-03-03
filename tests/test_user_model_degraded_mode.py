"""Tests for the user_model.db degraded-mode flag on DatabaseManager.

Validates that:
- The flag defaults to False on a fresh instance.
- Post-init verification failure sets the flag to True.
- get_database_health() exposes the flag for the /health endpoint.
- is_user_model_healthy() returns the inverse of the flag.
- A successful initialization leaves the flag at False.
"""

from unittest.mock import patch

from storage.manager import DatabaseManager


def test_default_flag_is_false(tmp_path):
    """A freshly constructed DatabaseManager has user_model_degraded=False."""
    mgr = DatabaseManager(data_dir=str(tmp_path))
    assert mgr.user_model_degraded is False


def test_is_user_model_healthy_default(tmp_path):
    """is_user_model_healthy() returns True on a fresh instance."""
    mgr = DatabaseManager(data_dir=str(tmp_path))
    assert mgr.is_user_model_healthy() is True


def test_degraded_after_double_verify_failure(tmp_path):
    """When _verify_db_functional returns False twice, the flag is set to True.

    Simulates the scenario where user_model.db is persistently corrupt:
    the first failure triggers a fresh-start attempt, the second failure
    causes _init_user_model_db to give up and set the degraded flag.
    """
    mgr = DatabaseManager(data_dir=str(tmp_path))

    # Initialize the non-user_model databases normally so that the
    # DatabaseManager is otherwise healthy.
    mgr._init_events_db()
    mgr._init_entities_db()
    mgr._init_state_db()
    mgr._init_preferences_db()

    # Patch _verify_db_functional to always return False (persistent corruption).
    with patch.object(mgr, "_verify_db_functional", return_value=False):
        mgr._init_user_model_db()

    assert mgr.user_model_degraded is True
    assert mgr.is_user_model_healthy() is False


def test_health_includes_degraded_mode_true(tmp_path):
    """get_database_health() includes degraded_mode=True when the flag is set."""
    mgr = DatabaseManager(data_dir=str(tmp_path))
    mgr.initialize_all()

    # Manually set the flag to simulate a prior init failure.
    mgr.user_model_degraded = True

    health = mgr.get_database_health()
    assert "user_model" in health
    assert health["user_model"]["degraded_mode"] is True


def test_health_includes_degraded_mode_false(db):
    """get_database_health() includes degraded_mode=False on a healthy instance."""
    health = db.get_database_health()
    assert "user_model" in health
    assert health["user_model"]["degraded_mode"] is False


def test_health_no_degraded_mode_on_other_dbs(db):
    """Only the user_model entry has the degraded_mode key."""
    health = db.get_database_health()
    for name in ("events", "entities", "state", "preferences"):
        assert "degraded_mode" not in health[name]


def test_successful_init_leaves_flag_false(db):
    """A fully successful initialization leaves user_model_degraded=False."""
    assert db.user_model_degraded is False
    assert db.is_user_model_healthy() is True


def test_flag_cleared_after_recovery(tmp_path):
    """When _check_and_recover_db resets a corrupt DB, the flag is cleared.

    Simulates: flag was True from a prior failure, then _check_and_recover_db
    detects corruption and resets the DB file, clearing the flag so that the
    subsequent schema recreation starts from a clean state.
    """
    mgr = DatabaseManager(data_dir=str(tmp_path))
    mgr.initialize_all()

    # Simulate a prior degraded state.
    mgr.user_model_degraded = True

    # Corrupt the user_model.db file so _check_and_recover_db detects it.
    db_path = mgr.data_dir / "user_model.db"
    db_path.write_bytes(b"corrupt data that is not a valid sqlite file")

    # Re-initialize — _check_and_recover_db should detect corruption,
    # reset the file, and clear the degraded flag.
    mgr._user_model_verify_retries = 0
    mgr._init_user_model_db()

    assert mgr.user_model_degraded is False
    assert mgr.is_user_model_healthy() is True
