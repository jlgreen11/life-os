"""
Tests for the /api/user-model/signal-profiles endpoint.

The signal-profiles API exposes all 9 behavioral signal profiles collected by
the SignalPipeline — linguistic, linguistic_inbound, cadence, mood_signals,
relationships, topics, temporal, spatial, and decision — as a first-class REST
resource.

Previously these profiles were only accessible internally (InsightEngine,
SemanticFactInferrer, PredictionEngine) with no way to query them directly.
This endpoint closes that gap by surfacing 730K+ accumulated behavioral samples
for inspection and debugging.
"""

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
    """Create a mock LifeOS instance backed by a real temporary SQLite database.

    Uses the real user_model_store fixture so that update_signal_profile /
    get_signal_profile round-trips exercise actual SQL rather than mocks.
    """
    life_os = Mock()
    life_os.db = db
    life_os.user_model_store = user_model_store

    # Stub out services not touched by the signal-profiles endpoint.
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
    life_os.config = {}

    return life_os


@pytest.fixture
def app(mock_life_os):
    """Create a FastAPI test application."""
    return create_web_app(mock_life_os)


@pytest.fixture
def client(app):
    """Create a synchronous TestClient."""
    return TestClient(app)


@pytest.fixture
def populated_profiles(user_model_store):
    """Seed the database with representative signal profiles.

    Covers a cross-section of real profile types with realistic data so tests
    can assert on both structure and content without hitting the live database.
    """
    profiles = {
        "linguistic": {
            "avg_sentence_length": 18.4,
            "formality_score": 0.72,
            "punctuation_density": 0.12,
            "unique_word_ratio": 0.58,
            "question_rate": 0.15,
            "hedge_rate": 0.09,
        },
        "temporal": {
            "chronotype": "early_bird",
            "peak_productive_hour": 9,
            "work_boundary_adherence": 0.83,
            "hourly_activity": {"9": 42, "10": 38, "14": 31},
        },
        "decision": {
            "avg_decision_speed_hours": 2.3,
            "delegation_comfort": 0.64,
            "delegation_by_domain": {"email": 0.7, "calendar": 0.5},
        },
        "relationships": {
            "contacts": {
                "alice@example.com": {
                    "outbound_count": 47,
                    "inbound_count": 52,
                    "gap_days": 1.5,
                }
            }
        },
    }

    for ptype, data in profiles.items():
        user_model_store.update_signal_profile(ptype, data)

    return profiles


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_get_all_profiles_returns_200(client, populated_profiles):
    """GET /api/user-model/signal-profiles returns HTTP 200."""
    response = client.get("/api/user-model/signal-profiles")
    assert response.status_code == 200


def test_get_all_profiles_response_structure(client, populated_profiles):
    """Response envelope contains profiles, types_with_data, and generated_at."""
    response = client.get("/api/user-model/signal-profiles")
    data = response.json()

    assert "profiles" in data
    assert "types_with_data" in data
    assert "generated_at" in data

    # types_with_data must be a subset of profiles keys.
    assert set(data["types_with_data"]) == set(data["profiles"].keys())


def test_get_all_profiles_returns_only_populated_types(client, populated_profiles):
    """Empty profile types (no data yet) are omitted from the response."""
    response = client.get("/api/user-model/signal-profiles")
    data = response.json()

    # Only the 4 seeded types should appear — not the 5 un-seeded ones.
    assert set(data["types_with_data"]) == {"linguistic", "temporal", "decision", "relationships"}
    assert len(data["profiles"]) == 4


def test_profile_entry_structure(client, populated_profiles):
    """Each profile entry contains data, samples_count, and updated_at."""
    response = client.get("/api/user-model/signal-profiles")
    profiles = response.json()["profiles"]

    for ptype, entry in profiles.items():
        assert "data" in entry, f"Profile {ptype!r} missing 'data' key"
        assert "samples_count" in entry, f"Profile {ptype!r} missing 'samples_count' key"
        assert "updated_at" in entry, f"Profile {ptype!r} missing 'updated_at' key"


def test_profile_data_is_dict(client, populated_profiles):
    """The 'data' field of each profile is a deserialized dict, not a JSON string."""
    response = client.get("/api/user-model/signal-profiles")
    profiles = response.json()["profiles"]

    for ptype, entry in profiles.items():
        assert isinstance(entry["data"], dict), (
            f"Profile {ptype!r} data should be a dict, got {type(entry['data'])}"
        )


def test_profile_data_content_matches_stored_values(client, populated_profiles):
    """Data values retrieved via the API match what was written by update_signal_profile."""
    response = client.get("/api/user-model/signal-profiles")
    profiles = response.json()["profiles"]

    # Spot-check the linguistic profile.
    ling = profiles["linguistic"]["data"]
    assert ling["formality_score"] == pytest.approx(0.72, abs=0.01)
    assert ling["avg_sentence_length"] == pytest.approx(18.4, abs=0.1)

    # Spot-check the temporal profile.
    temp = profiles["temporal"]["data"]
    assert temp["chronotype"] == "early_bird"
    assert temp["peak_productive_hour"] == 9


def test_samples_count_increments_per_update(client, user_model_store):
    """samples_count reflects the number of update_signal_profile calls."""
    # Two separate calls to update the same profile type.
    user_model_store.update_signal_profile("spatial", {"top_location": "home"})
    user_model_store.update_signal_profile("spatial", {"top_location": "office"})

    response = client.get("/api/user-model/signal-profiles?profile_type=spatial")
    entry = response.json()["profiles"]["spatial"]

    # Each update increments samples_count by 1 (COALESCE + 1 logic).
    assert entry["samples_count"] == 2


def test_generated_at_is_recent_iso_timestamp(client):
    """generated_at is a valid ISO-8601 timestamp generated within the last minute."""
    response = client.get("/api/user-model/signal-profiles")
    assert response.status_code == 200

    generated_at = response.json()["generated_at"]
    dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    assert abs((now - dt).total_seconds()) < 60


# ---------------------------------------------------------------------------
# Filter by profile_type
# ---------------------------------------------------------------------------


def test_filter_by_valid_profile_type(client, populated_profiles):
    """?profile_type=temporal returns only the temporal profile."""
    response = client.get("/api/user-model/signal-profiles?profile_type=temporal")
    assert response.status_code == 200

    data = response.json()
    assert list(data["profiles"].keys()) == ["temporal"]
    assert data["types_with_data"] == ["temporal"]


def test_filter_by_profile_type_with_nested_data(client, populated_profiles):
    """Single-profile response includes the full nested data dict."""
    response = client.get("/api/user-model/signal-profiles?profile_type=decision")
    assert response.status_code == 200

    entry = response.json()["profiles"]["decision"]
    assert entry["data"]["avg_decision_speed_hours"] == pytest.approx(2.3, abs=0.01)
    assert "delegation_by_domain" in entry["data"]


def test_filter_empty_profile_type_returns_empty_profiles(client):
    """A valid but un-seeded profile type returns an empty profiles dict (no data yet)."""
    # "cadence" is valid but not seeded in this test.
    response = client.get("/api/user-model/signal-profiles?profile_type=cadence")
    assert response.status_code == 200

    data = response.json()
    assert data["profiles"] == {}
    assert data["types_with_data"] == []


def test_filter_unknown_profile_type_returns_404(client):
    """An unrecognised profile_type query param raises HTTP 404."""
    response = client.get("/api/user-model/signal-profiles?profile_type=nonexistent_type")
    assert response.status_code == 404

    detail = response.json()["detail"]
    assert "nonexistent_type" in detail
    # The error message should list the valid types so the caller can self-correct.
    assert "linguistic" in detail


def test_filter_all_known_valid_types_accepted(client):
    """Every documented profile type is accepted without a 404."""
    known_types = [
        "linguistic",
        "linguistic_inbound",
        "cadence",
        "mood_signals",
        "relationships",
        "topics",
        "temporal",
        "spatial",
        "decision",
    ]
    for ptype in known_types:
        response = client.get(f"/api/user-model/signal-profiles?profile_type={ptype}")
        assert response.status_code == 200, (
            f"Expected 200 for profile_type={ptype!r}, got {response.status_code}"
        )


# ---------------------------------------------------------------------------
# Empty database
# ---------------------------------------------------------------------------


def test_empty_database_returns_empty_profiles(client):
    """When no profiles have been stored, returns an empty profiles dict."""
    # Do not inject populated_profiles — database is empty.
    response = client.get("/api/user-model/signal-profiles")
    assert response.status_code == 200

    data = response.json()
    assert data["profiles"] == {}
    assert data["types_with_data"] == []
    assert "generated_at" in data


# ---------------------------------------------------------------------------
# Round-trip integrity
# ---------------------------------------------------------------------------


def test_all_seeded_types_round_trip(client, user_model_store):
    """All 9 profile types can be written and read back via the API."""
    test_data = {"key": "value", "count": 42}

    for ptype in [
        "linguistic",
        "linguistic_inbound",
        "cadence",
        "mood_signals",
        "relationships",
        "topics",
        "temporal",
        "spatial",
        "decision",
    ]:
        user_model_store.update_signal_profile(ptype, test_data)

    response = client.get("/api/user-model/signal-profiles")
    assert response.status_code == 200

    profiles = response.json()["profiles"]
    assert len(profiles) == 9

    for ptype, entry in profiles.items():
        assert entry["data"]["key"] == "value"
        assert entry["data"]["count"] == 42
        assert entry["samples_count"] >= 1
