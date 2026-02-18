"""
Life OS — Tests for /api/insights/summary delegating to InsightEngine.

Iteration 235 replaced the 220-line hand-rolled insights_summary() endpoint
body with a delegation to InsightEngine.generate_insights().  The old
implementation covered only 3 of 14 insight categories and lacked deduplication
and source-weight gating.

The new endpoint:
  1. Calls life_os.insight_engine.generate_insights() to refresh insights.
  2. Reads all non-expired, non-negative insights from the ``insights`` DB table.
  3. Returns them sorted by confidence (from the DB's ORDER BY confidence DESC).
  4. Returns ``{"insights": [...], "generated_at": "..."}`` — same contract
     as before, compatible with existing callers.

Tests:
  - Endpoint returns 200 with required keys.
  - Insights stored in DB are returned in the response.
  - Negative-feedback insights are excluded from the response.
  - Expired insights (past staleness_ttl_hours) are excluded.
  - Endpoint returns 200 even when generate_insights() raises an exception.
  - generate_insights() failure still returns stored insights from DB.
  - generated_at is a valid ISO timestamp.
  - evidence field is deserialized from JSON string to list.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.insight_engine.engine import InsightEngine
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def life_os_mock(db, event_store, user_model_store):
    """Minimal LifeOS mock with a real InsightEngine wired to the temp DB.

    The real InsightEngine ensures the generate_insights() → DB write → DB read
    roundtrip is exercised end-to-end.  No source_weight_manager is provided so
    all insights pass the weight gate (confidence multiplied by 1.0 by default).
    """
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
    # Wire a real InsightEngine so generate_insights() actually stores to DB
    mock.insight_engine = InsightEngine(db=db, ums=user_model_store)
    return mock


@pytest.fixture
def client(life_os_mock):
    """TestClient bound to the full FastAPI app with the mock LifeOS."""
    from web.app import create_web_app
    app = create_web_app(life_os_mock)
    return TestClient(app)


def _insert_insight(db, *, type_: str, summary: str, confidence: float,
                    category: str, entity: str | None = None,
                    feedback: str | None = None, staleness_ttl_hours: int = 168,
                    created_offset_hours: int = 0) -> str:
    """Insert a synthetic insight row into the user_model insights table.

    Args:
        db: DatabaseManager instance.
        type_: Insight type (e.g. "relationship_intelligence").
        summary: Human-readable summary string.
        confidence: Confidence score (0–1).
        category: Broad category (e.g. "contact_gap").
        entity: Optional entity string (contact address, place name).
        feedback: Optional feedback string ("useful" / "dismissed" / None).
        staleness_ttl_hours: Hours until the insight is considered stale.
        created_offset_hours: Hours to subtract from now for created_at
            (positive = older insight).

    Returns:
        The generated insight ID (UUID string).
    """
    insight_id = str(uuid.uuid4())
    dedup_key = str(uuid.uuid4())[:16]
    created_at = (
        datetime.now(timezone.utc) - timedelta(hours=created_offset_hours)
    ).isoformat()
    evidence = json.dumps(["test_evidence"])
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO insights
               (id, type, summary, confidence, evidence, category, entity,
                feedback, staleness_ttl_hours, dedup_key, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (insight_id, type_, summary, confidence, evidence, category,
             entity, feedback, staleness_ttl_hours, dedup_key, created_at),
        )
    return insight_id


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------

def test_endpoint_returns_200_with_required_keys(client):
    """GET /api/insights/summary returns 200 with 'insights' and 'generated_at' keys."""
    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data, "Response must contain 'insights' key"
    assert "generated_at" in data, "Response must contain 'generated_at' key"
    assert isinstance(data["insights"], list), "'insights' must be a list"


def test_generated_at_is_valid_iso_timestamp(client):
    """generated_at must be a parseable ISO 8601 timestamp."""
    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    generated_at = response.json()["generated_at"]
    # Should parse without raising
    dt = datetime.fromisoformat(generated_at)
    assert dt.tzinfo is not None, "generated_at must be timezone-aware"


def test_empty_db_returns_empty_insights_list(client):
    """When no insights are stored, endpoint returns an empty list."""
    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    assert response.json()["insights"] == []


# ---------------------------------------------------------------------------
# DB read tests — verify stored insights are returned
# ---------------------------------------------------------------------------

def test_stored_insight_is_returned(client, life_os_mock):
    """Insights stored in the DB (by generate_insights or directly) are returned."""
    _insert_insight(
        life_os_mock.db,
        type_="behavioral_pattern",
        summary="You visit Cafe X frequently.",
        confidence=0.75,
        category="place",
        entity="Cafe X",
    )

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    assert len(insights) >= 1
    entities = [i.get("entity") for i in insights]
    assert "Cafe X" in entities, "Stored place insight should appear in response"


def test_multiple_stored_insights_all_returned(client, life_os_mock):
    """All non-expired, non-negative stored insights are returned."""
    _insert_insight(life_os_mock.db, type_="behavioral_pattern",
                    summary="Place A", confidence=0.8, category="place")
    _insert_insight(life_os_mock.db, type_="relationship_intelligence",
                    summary="Contact B overdue", confidence=0.6, category="contact_gap",
                    entity="contact@example.com")

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    assert len(insights) >= 2, "Both stored insights should be returned"


# ---------------------------------------------------------------------------
# Filtering tests — negative feedback and expired insights
# ---------------------------------------------------------------------------

def test_negative_feedback_insight_excluded(client, life_os_mock):
    """Insights marked with feedback='negative' must not appear in the response."""
    negative_id = _insert_insight(
        life_os_mock.db,
        type_="behavioral_pattern",
        summary="Dismissed insight",
        confidence=0.7,
        category="place",
        feedback="dismissed",  # stored as 'negative' equivalent but 'dismissed' is not 'negative'
    )
    # Insert one marked explicitly as 'negative'
    with life_os_mock.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE insights SET feedback = 'negative' WHERE id = ?",
            (negative_id,),
        )

    _insert_insight(
        life_os_mock.db,
        type_="relationship_intelligence",
        summary="Active insight",
        confidence=0.75,
        category="contact_gap",
        entity="keep@example.com",
    )

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    summaries = [i["summary"] for i in insights]
    assert "Dismissed insight" not in summaries, (
        "Insight with negative feedback should be excluded"
    )
    assert "Active insight" in summaries, (
        "Non-negative insight should still appear"
    )


def test_expired_insight_excluded(client, life_os_mock):
    """Insights older than their staleness_ttl_hours must not appear."""
    # Insert an insight with 1-hour TTL that is already 2 hours old
    _insert_insight(
        life_os_mock.db,
        type_="behavioral_pattern",
        summary="Expired insight",
        confidence=0.8,
        category="place",
        staleness_ttl_hours=1,
        created_offset_hours=2,  # 2 hours old > 1 hour TTL → expired
    )

    _insert_insight(
        life_os_mock.db,
        type_="relationship_intelligence",
        summary="Fresh insight",
        confidence=0.75,
        category="contact_gap",
        entity="fresh@example.com",
        staleness_ttl_hours=168,   # 7 days TTL, just created → fresh
        created_offset_hours=0,
    )

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    summaries = [i["summary"] for i in insights]
    assert "Expired insight" not in summaries, (
        "Insight past its staleness TTL should be excluded"
    )
    assert "Fresh insight" in summaries, (
        "Fresh insight within TTL should appear"
    )


# ---------------------------------------------------------------------------
# Failure-tolerance tests
# ---------------------------------------------------------------------------

def test_endpoint_returns_200_when_generate_insights_raises(client, life_os_mock):
    """If generate_insights() raises, the endpoint still returns 200.

    Fail-open behavior: callers must always receive a valid response even when
    the InsightEngine encounters an internal error.
    """
    with patch.object(
        life_os_mock.insight_engine,
        "generate_insights",
        side_effect=RuntimeError("Simulated InsightEngine failure"),
    ):
        response = client.get("/api/insights/summary")

    assert response.status_code == 200
    data = response.json()
    assert "insights" in data


def test_stored_insights_returned_even_when_generate_insights_fails(client, life_os_mock):
    """When generate_insights() fails, previously stored insights are still served.

    The endpoint reads from the DB after generate_insights() regardless of
    whether it succeeded, ensuring cached insights are never lost.
    """
    _insert_insight(
        life_os_mock.db,
        type_="behavioral_pattern",
        summary="Cached insight from last run",
        confidence=0.7,
        category="place",
    )

    with patch.object(
        life_os_mock.insight_engine,
        "generate_insights",
        side_effect=RuntimeError("Simulated InsightEngine failure"),
    ):
        response = client.get("/api/insights/summary")

    assert response.status_code == 200
    insights = response.json()["insights"]
    summaries = [i["summary"] for i in insights]
    assert "Cached insight from last run" in summaries, (
        "Cached stored insights must be returned even when generate_insights fails"
    )


# ---------------------------------------------------------------------------
# Response structure tests
# ---------------------------------------------------------------------------

def test_evidence_field_deserialized_from_json_string(client, life_os_mock):
    """The evidence field stored as a JSON string must be returned as a list."""
    _insert_insight(
        life_os_mock.db,
        type_="behavioral_pattern",
        summary="Evidence deserialization test",
        confidence=0.6,
        category="place",
        entity="TestPlace",
    )

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    target = next(
        (i for i in insights if i.get("entity") == "TestPlace"), None
    )
    assert target is not None
    # The raw DB value is '["test_evidence"]' — must be parsed to a list
    assert isinstance(target["evidence"], list), (
        "evidence field should be deserialized from JSON string to a Python list"
    )


def test_insights_include_required_fields(client, life_os_mock):
    """Each returned insight must include id, type, summary, confidence, category."""
    _insert_insight(
        life_os_mock.db,
        type_="spending_pattern",
        summary="Field presence check",
        confidence=0.65,
        category="spending",
        entity="FieldCheckEntity",
    )

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    insights = response.json()["insights"]
    target = next(
        (i for i in insights if i.get("entity") == "FieldCheckEntity"), None
    )
    assert target is not None
    for field in ("id", "type", "summary", "confidence", "category"):
        assert field in target, f"Required field '{field}' missing from insight dict"
