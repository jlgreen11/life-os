"""
Life OS — Tests for cache_age_seconds / oldest_insight_at in /api/insights/summary.

When insights are served from the persistent store, clients need to know how
old the data actually is.  The endpoint now computes cache_age_seconds (float)
and oldest_insight_at (ISO string) from the oldest created_at timestamp among
the returned insights so callers can surface a staleness indicator in the UI.

Tests:
  - cache_age_seconds and oldest_insight_at are null when no insights exist.
  - cache_age_seconds is approximately correct for a single known-age insight.
  - cache_age_seconds reflects the OLDEST insight when multiple are present.
  - oldest_insight_at is a parseable ISO timestamp pointing to the oldest insight.
  - cache_age_seconds is positive (insights are always in the past).
  - New fields do not break the existing insights / generated_at contract.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from services.insight_engine.engine import InsightEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def life_os_mock(db, event_store, user_model_store):
    """Minimal LifeOS mock with a real InsightEngine and real DB components."""
    mock = MagicMock()
    mock.db = db
    mock.event_store = event_store
    mock.user_model_store = user_model_store
    mock.signal_extractor = MagicMock()
    mock.signal_extractor.get_user_summary.return_value = {"profiles": {}}
    mock.vector_store = MagicMock()
    mock.event_bus = MagicMock()
    mock.event_bus.is_connected = False
    mock.connectors = []
    mock.notification_manager = MagicMock()
    mock.notification_manager.get_stats.return_value = {}
    mock.feedback_collector = MagicMock()
    mock.feedback_collector.get_feedback_summary.return_value = {}
    mock.rules_engine = MagicMock()
    mock.task_manager = MagicMock()
    mock.ai_engine = MagicMock()
    mock.browser_orchestrator = MagicMock()
    mock.onboarding = MagicMock()
    mock.insight_engine = InsightEngine(db=db, ums=user_model_store)
    return mock


@pytest.fixture
def client(life_os_mock):
    """TestClient bound to the full FastAPI app with the mock LifeOS."""
    from web.app import create_web_app

    app = create_web_app(life_os_mock)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_insight(
    db,
    *,
    type_: str = "behavioral_pattern",
    summary: str = "Test insight",
    confidence: float = 0.7,
    category: str = "place",
    entity: str | None = None,
    feedback: str | None = None,
    staleness_ttl_hours: int = 168,
    created_at: datetime | None = None,
    created_offset_hours: float = 0.0,
) -> str:
    """Insert a synthetic insight row with a controllable created_at timestamp.

    Args:
        db: DatabaseManager instance.
        type_: Insight type string.
        summary: Human-readable summary.
        confidence: Confidence score (0–1).
        category: Broad category string.
        entity: Optional entity string.
        feedback: Optional feedback string.
        staleness_ttl_hours: Hours until the insight is stale.
        created_at: Explicit datetime for created_at; overrides created_offset_hours.
        created_offset_hours: Hours to subtract from now (positive = older).

    Returns:
        The generated insight ID (UUID string).
    """
    insight_id = str(uuid.uuid4())
    dedup_key = str(uuid.uuid4())[:16]
    if created_at is not None:
        ts = created_at.isoformat()
    else:
        ts = (
            datetime.now(timezone.utc) - timedelta(hours=created_offset_hours)
        ).isoformat()
    evidence = json.dumps(["test_evidence"])
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO insights
               (id, type, summary, confidence, evidence, category, entity,
                feedback, staleness_ttl_hours, dedup_key, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                insight_id,
                type_,
                summary,
                confidence,
                evidence,
                category,
                entity,
                feedback,
                staleness_ttl_hours,
                dedup_key,
                ts,
            ),
        )
    return insight_id


# ---------------------------------------------------------------------------
# Null-case: no insights in DB
# ---------------------------------------------------------------------------


def test_cache_age_null_when_no_insights(client):
    """When the DB contains no insights, cache_age_seconds must be null."""
    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["cache_age_seconds"] is None, (
        "cache_age_seconds must be null when there are no insights"
    )


def test_oldest_insight_at_null_when_no_insights(client):
    """When the DB contains no insights, oldest_insight_at must be null."""
    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["oldest_insight_at"] is None, (
        "oldest_insight_at must be null when there are no insights"
    )


# ---------------------------------------------------------------------------
# Single insight: cache age reflects actual age
# ---------------------------------------------------------------------------


def test_cache_age_approximately_correct_for_single_insight(client, life_os_mock):
    """cache_age_seconds is within 2 seconds of the known age for a single insight."""
    offset_hours = 2.0
    _insert_insight(
        life_os_mock.db,
        summary="Two-hour-old insight",
        created_offset_hours=offset_hours,
    )

    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()

    cache_age = data["cache_age_seconds"]
    assert cache_age is not None, "cache_age_seconds must not be null when insights exist"

    expected_seconds = offset_hours * 3600
    tolerance = 2.0  # Allow up to 2 seconds of test-execution skew
    assert abs(cache_age - expected_seconds) <= tolerance, (
        f"cache_age_seconds={cache_age} should be ~{expected_seconds}s "
        f"(offset {offset_hours}h); tolerance ±{tolerance}s"
    )


def test_cache_age_positive_for_past_insight(client, life_os_mock):
    """cache_age_seconds is positive for any insight created in the past."""
    _insert_insight(
        life_os_mock.db,
        summary="Five-minute-old insight",
        created_offset_hours=0.1,  # 6 minutes ago
    )

    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    cache_age = response.json()["cache_age_seconds"]
    assert cache_age is not None
    assert cache_age > 0, "cache_age_seconds must be positive for a past insight"


# ---------------------------------------------------------------------------
# Multiple insights: oldest one drives the cache age
# ---------------------------------------------------------------------------


def test_cache_age_reflects_oldest_insight(client, life_os_mock):
    """cache_age_seconds must reflect the OLDEST insight, not the newest."""
    # Newest insight: 1 hour old
    _insert_insight(
        life_os_mock.db,
        summary="Recent insight",
        created_offset_hours=1.0,
    )
    # Oldest insight: 5 hours old — this should drive cache_age_seconds
    _insert_insight(
        life_os_mock.db,
        summary="Old insight",
        created_offset_hours=5.0,
    )

    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()

    cache_age = data["cache_age_seconds"]
    assert cache_age is not None

    # Should be ~5 hours (18000 s), not ~1 hour (3600 s)
    expected_oldest = 5.0 * 3600
    tolerance = 2.0
    assert abs(cache_age - expected_oldest) <= tolerance, (
        f"cache_age_seconds={cache_age} should track the oldest insight "
        f"(~{expected_oldest}s); got closer to the newer one instead"
    )


def test_oldest_insight_at_points_to_oldest_timestamp(client, life_os_mock):
    """oldest_insight_at must be the ISO timestamp of the oldest insight."""
    # Newest: 30 minutes ago
    _insert_insight(
        life_os_mock.db,
        summary="Newer insight",
        created_offset_hours=0.5,
    )
    # Oldest: 4 hours ago
    oldest_dt = datetime.now(timezone.utc) - timedelta(hours=4)
    _insert_insight(
        life_os_mock.db,
        summary="Older insight",
        created_at=oldest_dt,
    )

    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()

    oldest_at_str = data["oldest_insight_at"]
    assert oldest_at_str is not None, "oldest_insight_at must not be null"

    # Parse it and verify it is within 1 second of the inserted timestamp
    oldest_at_returned = datetime.fromisoformat(oldest_at_str)
    if oldest_at_returned.tzinfo is None:
        oldest_at_returned = oldest_at_returned.replace(tzinfo=timezone.utc)

    # Normalise both to UTC for comparison
    if oldest_dt.tzinfo is None:
        oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)

    diff = abs((oldest_at_returned - oldest_dt).total_seconds())
    assert diff <= 1.0, (
        f"oldest_insight_at={oldest_at_str!r} differs from inserted "
        f"timestamp by {diff}s (expected ≤ 1s)"
    )


# ---------------------------------------------------------------------------
# Backward-compatibility: existing keys still present
# ---------------------------------------------------------------------------


def test_existing_response_keys_still_present(client, life_os_mock):
    """Adding cache fields must not remove 'insights' or 'generated_at'."""
    _insert_insight(life_os_mock.db, summary="Compat check insight")

    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()

    assert "insights" in data, "Legacy 'insights' key must still be present"
    assert "generated_at" in data, "Legacy 'generated_at' key must still be present"
    assert "cache_age_seconds" in data, "'cache_age_seconds' key must be present"
    assert "oldest_insight_at" in data, "'oldest_insight_at' key must be present"


def test_oldest_insight_at_is_parseable_iso_string(client, life_os_mock):
    """oldest_insight_at must be a parseable ISO 8601 string when insights exist."""
    _insert_insight(
        life_os_mock.db,
        summary="ISO parse check",
        created_offset_hours=1.0,
    )

    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    oldest_at = response.json()["oldest_insight_at"]
    assert oldest_at is not None

    # Must parse without raising ValueError
    dt = datetime.fromisoformat(oldest_at)
    assert isinstance(dt, datetime), "oldest_insight_at must parse to a datetime object"
