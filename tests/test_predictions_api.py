"""
Tests for GET /api/predictions and POST /api/predictions/{id}/feedback.

The predictions API exposes the system's forward-looking predictions — follow-up
opportunities, reminders, calendar conflicts, routine deviations, etc. — as a
first-class REST resource so that clients can display them directly rather than
parsing the embedded morning briefing text.

Before this endpoint existed the only way to see active predictions was:
  1. Read the /api/briefing text (opaque, LLM-formatted)
  2. Query the database directly (not exposed externally)

This test suite verifies:
  - GET /api/predictions returns surfaced, unresolved predictions
  - Filtering by prediction_type, min_confidence, include_resolved works
  - The supporting_signals JSON field is deserialized in the response
  - POST /api/predictions/{id}/feedback resolves the prediction
  - POST returns 404 for unknown prediction IDs
  - Resolved predictions are excluded from the default active list
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_life_os(db, user_model_store):
    """Mock LifeOS with real db and user_model_store for SQL round-trips."""
    life_os = Mock()
    life_os.db = db
    life_os.user_model_store = user_model_store

    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = False
    life_os.event_store = Mock()
    life_os.vector_store = Mock()
    life_os.signal_extractor = Mock()
    life_os.task_manager = Mock()
    life_os.notification_manager = Mock()
    life_os.prediction_engine = Mock()
    life_os.rules_engine = Mock()
    life_os.feedback_collector = Mock()
    life_os.ai_engine = Mock()
    life_os.browser_orchestrator = Mock()
    life_os.onboarding = Mock()
    life_os.source_weight_manager = Mock()
    life_os.config = {}

    return life_os


@pytest.fixture
def app(mock_life_os):
    return create_web_app(mock_life_os)


@pytest.fixture
def client(app):
    return TestClient(app)


def _store_prediction(db, **kwargs):
    """Helper: insert a prediction row directly into the test database.

    Accepts keyword overrides so individual tests can set any field.
    Returns the prediction dict (including its generated id).
    """
    pred = {
        "id": str(uuid.uuid4()),
        "prediction_type": kwargs.get("prediction_type", "opportunity"),
        "description": kwargs.get("description", "Test prediction"),
        "confidence": kwargs.get("confidence", 0.75),
        "confidence_gate": kwargs.get("confidence_gate", "DEFAULT"),
        "time_horizon": kwargs.get("time_horizon"),
        "suggested_action": kwargs.get("suggested_action"),
        "supporting_signals": kwargs.get("supporting_signals", {"contact": "test@example.com"}),
        "was_surfaced": kwargs.get("was_surfaced", 1),
        "user_response": kwargs.get("user_response"),
        "was_accurate": kwargs.get("was_accurate"),
        "filter_reason": kwargs.get("filter_reason"),
        "resolution_reason": kwargs.get("resolution_reason"),
        "resolved_at": kwargs.get("resolved_at"),
    }
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced,
                user_response, was_accurate, filter_reason, resolution_reason, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred["id"],
                pred["prediction_type"],
                pred["description"],
                pred["confidence"],
                pred["confidence_gate"],
                pred["time_horizon"],
                pred["suggested_action"],
                json.dumps(pred["supporting_signals"]),
                pred["was_surfaced"],
                pred["user_response"],
                pred["was_accurate"],
                pred["filter_reason"],
                pred["resolution_reason"],
                pred["resolved_at"],
            ),
        )
    return pred


# ---------------------------------------------------------------------------
# GET /api/predictions — basic response shape
# ---------------------------------------------------------------------------


def test_list_predictions_empty(client):
    """Empty database returns an empty list with correct envelope."""
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["predictions"] == []
    assert data["count"] == 0
    assert "generated_at" in data
    assert "filters" in data


def test_list_predictions_returns_active(client, mock_life_os):
    """Surfaced, unresolved predictions appear in the response."""
    pred = _store_prediction(mock_life_os.db, description="Follow up with Alice")
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["predictions"][0]["id"] == pred["id"]
    assert data["predictions"][0]["description"] == "Follow up with Alice"


def test_list_predictions_response_fields(client, mock_life_os):
    """Every prediction in the response has the expected fields."""
    _store_prediction(
        mock_life_os.db,
        prediction_type="reminder",
        description="Review Q1 plan",
        confidence=0.8,
        confidence_gate="DEFAULT",
        suggested_action="Open the Q1 planning doc",
        supporting_signals={"event": "Q1 Planning"},
    )
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    p = resp.json()["predictions"][0]

    required_fields = [
        "id", "prediction_type", "description", "confidence",
        "confidence_gate", "time_horizon", "suggested_action",
        "supporting_signals", "was_surfaced", "user_response",
        "was_accurate", "filter_reason", "resolution_reason",
        "created_at", "resolved_at",
    ]
    for field in required_fields:
        assert field in p, f"Missing field: {field}"


def test_list_predictions_supporting_signals_deserialized(client, mock_life_os):
    """supporting_signals is returned as a parsed dict, not a JSON string."""
    _store_prediction(
        mock_life_os.db,
        supporting_signals={"contact_email": "alice@example.com", "gap_days": 14},
    )
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    signals = resp.json()["predictions"][0]["supporting_signals"]
    assert isinstance(signals, dict)
    assert signals["contact_email"] == "alice@example.com"
    assert signals["gap_days"] == 14


# ---------------------------------------------------------------------------
# GET /api/predictions — filtering
# ---------------------------------------------------------------------------


def test_filter_by_prediction_type(client, mock_life_os):
    """?prediction_type= returns only matching predictions."""
    _store_prediction(mock_life_os.db, prediction_type="opportunity", description="Opportunity A")
    _store_prediction(mock_life_os.db, prediction_type="reminder", description="Reminder B")
    _store_prediction(mock_life_os.db, prediction_type="conflict", description="Conflict C")

    resp = client.get("/api/predictions?prediction_type=reminder")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["predictions"][0]["prediction_type"] == "reminder"
    assert data["filters"]["prediction_type"] == "reminder"


def test_filter_by_min_confidence(client, mock_life_os):
    """?min_confidence= excludes predictions below the threshold."""
    _store_prediction(mock_life_os.db, confidence=0.9, description="High confidence")
    _store_prediction(mock_life_os.db, confidence=0.5, description="Medium confidence")
    _store_prediction(mock_life_os.db, confidence=0.2, description="Low confidence")

    resp = client.get("/api/predictions?min_confidence=0.6")
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 1
    assert preds[0]["description"] == "High confidence"
    for p in preds:
        assert p["confidence"] >= 0.6


def test_filtered_predictions_excluded(client, mock_life_os):
    """Predictions with filter_reason set are never returned (internal telemetry)."""
    _store_prediction(mock_life_os.db, description="Active prediction", filter_reason=None)
    _store_prediction(
        mock_life_os.db,
        description="Filtered prediction",
        filter_reason="confidence_too_low",
        was_surfaced=0,
    )
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 1
    assert preds[0]["description"] == "Active prediction"


def test_resolved_predictions_excluded_by_default(client, mock_life_os):
    """Resolved predictions are excluded unless include_resolved=true."""
    _store_prediction(mock_life_os.db, description="Active")
    _store_prediction(
        mock_life_os.db,
        description="Resolved",
        resolved_at="2026-02-18T10:00:00Z",
        was_accurate=1,
        user_response="acted_on",
    )
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    # Only the active prediction should appear
    assert len(preds) == 1
    assert preds[0]["description"] == "Active"


def test_include_resolved_returns_both(client, mock_life_os):
    """include_resolved=true returns both active and recently resolved predictions."""
    _store_prediction(mock_life_os.db, description="Active")
    _store_prediction(
        mock_life_os.db,
        description="Resolved",
        resolved_at="2026-02-18T10:00:00Z",
        was_accurate=1,
        user_response="acted_on",
    )
    resp = client.get("/api/predictions?include_resolved=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    descriptions = {p["description"] for p in data["predictions"]}
    assert "Active" in descriptions
    assert "Resolved" in descriptions


def test_unsurfaced_predictions_excluded(client, mock_life_os):
    """Predictions that never passed the confidence gate (was_surfaced=0) are excluded."""
    _store_prediction(mock_life_os.db, description="Surfaced", was_surfaced=1)
    _store_prediction(
        mock_life_os.db,
        description="Not surfaced",
        was_surfaced=0,
        filter_reason=None,  # No filter reason but was never surfaced
    )
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 1
    assert preds[0]["description"] == "Surfaced"


def test_predictions_ordered_by_confidence_descending(client, mock_life_os):
    """Predictions are returned highest confidence first."""
    _store_prediction(mock_life_os.db, confidence=0.5, description="Medium")
    _store_prediction(mock_life_os.db, confidence=0.9, description="High")
    _store_prediction(mock_life_os.db, confidence=0.3, description="Low")

    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    confidences = [p["confidence"] for p in preds]
    assert confidences == sorted(confidences, reverse=True)
    assert preds[0]["description"] == "High"


def test_limit_parameter(client, mock_life_os):
    """?limit= caps the number of returned predictions."""
    for i in range(10):
        _store_prediction(mock_life_os.db, description=f"Prediction {i}")

    resp = client.get("/api/predictions?limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3


def test_limit_clamped_at_200(client, mock_life_os):
    """limit parameter is silently clamped at 200 to prevent huge queries."""
    # Store 5 predictions — requesting 999 should still only return 5
    for i in range(5):
        _store_prediction(mock_life_os.db, description=f"P{i}")

    resp = client.get("/api/predictions?limit=999")
    assert resp.status_code == 200
    # We can't check that limit=200 was applied (fewer rows exist), but the
    # endpoint should not raise an error and should return all 5
    assert resp.json()["count"] == 5


def test_filters_reflected_in_response(client, mock_life_os):
    """The response 'filters' key echoes back the applied filter values."""
    resp = client.get(
        "/api/predictions?prediction_type=opportunity&min_confidence=0.4&include_resolved=true"
    )
    assert resp.status_code == 200
    filters = resp.json()["filters"]
    assert filters["prediction_type"] == "opportunity"
    assert filters["min_confidence"] == pytest.approx(0.4)
    assert filters["include_resolved"] is True


def test_generated_at_is_valid_timestamp(client):
    """generated_at in the response is a parseable ISO-8601 UTC timestamp."""
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    ts = resp.json()["generated_at"]
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    assert abs((now - dt).total_seconds()) < 60


# ---------------------------------------------------------------------------
# POST /api/predictions/{id}/feedback
# ---------------------------------------------------------------------------


def test_feedback_marks_prediction_resolved(client, mock_life_os):
    """POST feedback marks the prediction as resolved in the database."""
    pred = _store_prediction(mock_life_os.db)

    resp = client.post(f"/api/predictions/{pred['id']}/feedback?was_accurate=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "recorded"
    assert data["prediction_id"] == pred["id"]
    assert data["was_accurate"] is True

    # Verify the database was actually updated
    with mock_life_os.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (pred["id"],),
        ).fetchone()
    assert row["was_accurate"] == 1
    assert row["resolved_at"] is not None


def test_feedback_inaccurate(client, mock_life_os):
    """POST feedback with was_accurate=false sets was_accurate=0."""
    pred = _store_prediction(mock_life_os.db)

    resp = client.post(f"/api/predictions/{pred['id']}/feedback?was_accurate=false")
    assert resp.status_code == 200
    assert resp.json()["was_accurate"] is False

    with mock_life_os.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred["id"],),
        ).fetchone()
    assert row["was_accurate"] == 0


def test_feedback_with_user_response(client, mock_life_os):
    """user_response label is persisted when provided."""
    pred = _store_prediction(mock_life_os.db)

    resp = client.post(
        f"/api/predictions/{pred['id']}/feedback?was_accurate=true&user_response=acted_on"
    )
    assert resp.status_code == 200

    with mock_life_os.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT user_response FROM predictions WHERE id = ?",
            (pred["id"],),
        ).fetchone()
    assert row["user_response"] == "acted_on"


def test_feedback_404_for_unknown_id(client, mock_life_os):
    """POST feedback for a non-existent prediction returns 404."""
    resp = client.post("/api/predictions/nonexistent-uuid/feedback?was_accurate=true")
    assert resp.status_code == 404
    assert "nonexistent-uuid" in resp.json()["detail"]


def test_feedback_resolves_prediction_excluded_from_active_list(client, mock_life_os):
    """After feedback, the prediction no longer appears in the default active list."""
    pred = _store_prediction(mock_life_os.db, description="To be resolved")

    # Confirm it appears before feedback
    resp = client.get("/api/predictions")
    assert resp.json()["count"] == 1

    # Submit feedback
    client.post(f"/api/predictions/{pred['id']}/feedback?was_accurate=true")

    # Confirm it disappears from the default (active-only) list
    resp = client.get("/api/predictions")
    assert resp.json()["count"] == 0
