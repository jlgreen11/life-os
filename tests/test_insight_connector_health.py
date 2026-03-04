"""
Tests for the connector_health correlator in the InsightEngine.

Verifies that persistent connector failures (error state) and silently
stalled connectors (last_sync > 48h) are surfaced as actionable_alert
insights, while healthy connectors produce no insights.
"""

from datetime import datetime, timedelta, timezone

from services.insight_engine.engine import InsightEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_connector_state(db, connector_id: str, *, status: str = "ok",
                            enabled: int = 1, last_sync: str | None = None,
                            error_count: int = 0, last_error: str | None = None):
    """Insert a row into the connector_state table in state.db."""
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO connector_state
               (connector_id, status, enabled, last_sync, error_count, last_error, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                connector_id,
                status,
                enabled,
                last_sync,
                error_count,
                last_error,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


# ===========================================================================
# Test 1: Error-state connector produces a high-severity actionable_alert
# ===========================================================================

def test_error_connector_produces_alert(db, user_model_store):
    """A connector with status='error' should generate an actionable_alert insight."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    _insert_connector_state(
        db, "google",
        status="error",
        last_sync="2026-02-20T10:00:00+00:00",
        error_count=5,
        last_error="Authentication failed: token expired",
    )

    insights = engine._connector_health_insights()

    assert len(insights) == 1
    insight = insights[0]
    assert insight.type == "actionable_alert"
    assert insight.category == "connector_error"
    assert insight.entity == "google"
    assert "google" in insight.summary
    assert "Authentication failed" in insight.summary
    assert insight.confidence == 0.95
    assert insight.staleness_ttl_hours == 24
    # Evidence should contain structured fields
    assert any("connector_id=google" in e for e in insight.evidence)
    assert any("status=error" in e for e in insight.evidence)


# ===========================================================================
# Test 2: Stalled connector (last_sync > 48h) produces medium-severity alert
# ===========================================================================

def test_stalled_connector_produces_alert(db, user_model_store):
    """A connector with last_sync > 48h but status != 'error' should produce a medium insight."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    stale_sync = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
    _insert_connector_state(
        db, "proton_mail",
        status="ok",
        last_sync=stale_sync,
        error_count=0,
    )

    insights = engine._connector_health_insights()

    assert len(insights) == 1
    insight = insights[0]
    assert insight.type == "actionable_alert"
    assert insight.category == "connector_stalled"
    assert insight.entity == "proton_mail"
    assert "proton_mail" in insight.summary
    assert "not synced" in insight.summary
    assert insight.confidence == 0.8
    assert any("hours_since_sync=72" in e for e in insight.evidence)


# ===========================================================================
# Test 3: Healthy connector produces no insight
# ===========================================================================

def test_healthy_connector_no_insight(db, user_model_store):
    """A connector with status='ok' and recent last_sync should produce no insights."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    recent_sync = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _insert_connector_state(
        db, "signal",
        status="ok",
        last_sync=recent_sync,
        error_count=0,
    )

    insights = engine._connector_health_insights()
    assert insights == []


# ===========================================================================
# Test 4: Multiple failing connectors produce separate insights
# ===========================================================================

def test_multiple_failing_connectors(db, user_model_store):
    """Each failing connector should produce its own insight."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    # One in error state
    _insert_connector_state(
        db, "google",
        status="error",
        last_sync="2026-02-20T10:00:00+00:00",
        error_count=10,
        last_error="Auth expired",
    )

    # One stalled
    stale_sync = (datetime.now(timezone.utc) - timedelta(hours=96)).isoformat()
    _insert_connector_state(
        db, "caldav",
        status="ok",
        last_sync=stale_sync,
    )

    # One healthy (should not produce insight)
    recent_sync = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    _insert_connector_state(
        db, "imessage",
        status="ok",
        last_sync=recent_sync,
    )

    insights = engine._connector_health_insights()

    assert len(insights) == 2
    entities = {i.entity for i in insights}
    assert entities == {"google", "caldav"}

    # Verify each has a unique dedup_key
    dedup_keys = {i.dedup_key for i in insights}
    assert len(dedup_keys) == 2


# ===========================================================================
# Test 5: DB error returns empty list (fail-open)
# ===========================================================================

def test_db_error_returns_empty(db, user_model_store):
    """If the connector_state query fails, the correlator should return [] (fail-open)."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    # Patch get_connection to raise an exception for state.db
    original_get_connection = db.get_connection

    class _FakeCtx:
        """Context manager that raises on __enter__."""
        def __enter__(self):
            raise RuntimeError("Simulated DB failure")
        def __exit__(self, *args):
            pass

    def patched_get_connection(db_name):
        if db_name == "state":
            return _FakeCtx()
        return original_get_connection(db_name)

    db.get_connection = patched_get_connection

    insights = engine._connector_health_insights()
    assert insights == []

    # Restore
    db.get_connection = original_get_connection


# ===========================================================================
# Test 6: Disabled connectors are excluded
# ===========================================================================

def test_disabled_connector_excluded(db, user_model_store):
    """Connectors with enabled=0 should not produce insights even if in error state."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    _insert_connector_state(
        db, "finance",
        status="error",
        enabled=0,
        error_count=5,
        last_error="API key invalid",
    )

    insights = engine._connector_health_insights()
    assert insights == []


# ===========================================================================
# Test 7: Connector with null last_sync in error state
# ===========================================================================

def test_error_connector_null_last_sync(db, user_model_store):
    """Error connector with no last_sync should say 'never' in the summary."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")

    _insert_connector_state(
        db, "google",
        status="error",
        last_sync=None,
        error_count=3,
        last_error="Setup incomplete",
    )

    insights = engine._connector_health_insights()

    assert len(insights) == 1
    assert "never" in insights[0].summary
