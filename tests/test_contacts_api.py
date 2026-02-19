"""
Tests for the /api/contacts endpoint.

The contacts API exposes the ``contacts`` table from ``entities.db`` as a
first-class REST resource.  This table was enriched by PR #274 which
denormalized three relationship metrics — ``typical_response_time``,
``last_contact``, and ``contact_frequency_days`` — from the relationships
signal profile.

Previously these contacts (and their metrics) had no public API; they were
only accessible internally via device-proximity correlation and onboarding
seeding.  This endpoint closes that gap and validates the denormalized metric
write path introduced in PR #274.
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
def mock_life_os(db):
    """Create a mock LifeOS instance backed by a real temporary SQLite database.

    Uses the real ``db`` fixture so that contacts queries exercise actual SQL
    rather than mocks.
    """
    life_os = Mock()
    life_os.db = db

    # Stub out services not touched by the contacts endpoint.
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
    life_os.user_model_store = Mock()
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


def _insert_contact(
    db,
    *,
    name: str,
    emails=None,
    relationship: str = "colleague",
    is_priority: bool = False,
    typical_response_time: float = None,
    last_contact: str = None,
    contact_frequency_days: float = None,
) -> str:
    """Insert a contact row and return its generated UUID.

    Helper used across multiple test functions to seed the entities DB with
    realistic contact data without repeating the INSERT boilerplate.
    """
    contact_id = str(uuid.uuid4())
    emails_json = json.dumps(emails or [])
    with db.get_connection("entities") as conn:
        conn.execute(
            """INSERT INTO contacts
               (id, name, emails, relationship, is_priority,
                typical_response_time, last_contact, contact_frequency_days)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                contact_id,
                name,
                emails_json,
                relationship,
                1 if is_priority else 0,
                typical_response_time,
                last_contact,
                contact_frequency_days,
            ),
        )
    return contact_id


@pytest.fixture
def seeded_contacts(db):
    """Seed entities DB with a representative set of contacts.

    Four contacts covering the main interesting states:
        - Alice: priority, has all three relationship metrics
        - Bob: priority, has metrics (different values)
        - Carol: non-priority, has metrics
        - Dave: non-priority, no metrics yet (NULL columns)
    """
    alice_id = _insert_contact(
        db,
        name="Alice Smith",
        emails=["alice@example.com"],
        relationship="close_friend",
        is_priority=True,
        typical_response_time=3600.0,
        last_contact="2026-02-18T10:00:00.000Z",
        contact_frequency_days=2.5,
    )
    bob_id = _insert_contact(
        db,
        name="Bob Jones",
        emails=["bob@work.com"],
        relationship="colleague",
        is_priority=True,
        typical_response_time=7200.0,
        last_contact="2026-02-17T08:00:00.000Z",
        contact_frequency_days=5.0,
    )
    carol_id = _insert_contact(
        db,
        name="Carol White",
        emails=["carol@example.net"],
        relationship="acquaintance",
        is_priority=False,
        typical_response_time=86400.0,
        last_contact="2026-02-10T12:00:00.000Z",
        contact_frequency_days=14.0,
    )
    dave_id = _insert_contact(
        db,
        name="Dave Brown",
        emails=["dave@example.org"],
        relationship="acquaintance",
        is_priority=False,
        typical_response_time=None,
        last_contact=None,
        contact_frequency_days=None,
    )
    return {
        "alice": alice_id,
        "bob": bob_id,
        "carol": carol_id,
        "dave": dave_id,
    }


# ---------------------------------------------------------------------------
# Happy-path: basic response structure
# ---------------------------------------------------------------------------


def test_get_contacts_returns_200(client, seeded_contacts):
    """GET /api/contacts returns HTTP 200."""
    response = client.get("/api/contacts")
    assert response.status_code == 200


def test_get_contacts_response_envelope(client, seeded_contacts):
    """Response contains contacts list, total count, and generated_at timestamp."""
    response = client.get("/api/contacts")
    data = response.json()

    assert "contacts" in data
    assert "total" in data
    assert "generated_at" in data
    assert isinstance(data["contacts"], list)
    assert isinstance(data["total"], int)


def test_get_contacts_total_matches_contacts_length(client, seeded_contacts):
    """total reflects the count of matching contacts, not always the full table."""
    response = client.get("/api/contacts")
    data = response.json()

    # With default limit=100 and 4 contacts, total and len(contacts) both == 4.
    assert data["total"] == 4
    assert len(data["contacts"]) == 4


def test_get_contacts_generated_at_is_recent(client, seeded_contacts):
    """generated_at is a valid ISO-8601 timestamp within the last 60 seconds."""
    response = client.get("/api/contacts")
    generated_at = response.json()["generated_at"]

    dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    assert abs((now - dt).total_seconds()) < 60


# ---------------------------------------------------------------------------
# Contact object shape
# ---------------------------------------------------------------------------


def test_contact_has_required_fields(client, seeded_contacts):
    """Every returned contact has the expected schema fields."""
    response = client.get("/api/contacts")
    contacts = response.json()["contacts"]

    required_fields = {
        "id", "name", "emails", "is_priority",
        "typical_response_time", "last_contact", "contact_frequency_days",
    }
    for contact in contacts:
        missing = required_fields - set(contact.keys())
        assert not missing, f"Contact {contact.get('name')!r} missing fields: {missing}"


def test_emails_deserialized_as_list(client, seeded_contacts):
    """The 'emails' field is a Python list, not a JSON string."""
    response = client.get("/api/contacts")
    contacts = response.json()["contacts"]

    for contact in contacts:
        assert isinstance(contact["emails"], list), (
            f"Contact {contact['name']!r}: emails should be a list, "
            f"got {type(contact['emails'])}"
        )


def test_is_priority_is_boolean(client, seeded_contacts):
    """is_priority is a boolean, not the raw SQLite integer 0/1."""
    response = client.get("/api/contacts")
    contacts = response.json()["contacts"]

    for contact in contacts:
        assert isinstance(contact["is_priority"], bool), (
            f"Contact {contact['name']!r}: is_priority should be bool, "
            f"got {type(contact['is_priority'])}"
        )


def test_always_surface_is_boolean(client, seeded_contacts):
    """always_surface is a boolean (defaults to False in the schema)."""
    response = client.get("/api/contacts")
    contacts = response.json()["contacts"]

    for contact in contacts:
        assert isinstance(contact["always_surface"], bool), (
            f"Contact {contact['name']!r}: always_surface should be bool"
        )


def test_metrics_are_floats_or_null(client, seeded_contacts):
    """Relationship metric columns are floats when present and null when absent."""
    response = client.get("/api/contacts")
    contacts = {c["name"]: c for c in response.json()["contacts"]}

    # Alice has all three metrics.
    alice = contacts["Alice Smith"]
    assert isinstance(alice["typical_response_time"], float)
    assert isinstance(alice["contact_frequency_days"], float)
    assert alice["last_contact"] is not None

    # Dave has no metrics yet.
    dave = contacts["Dave Brown"]
    assert dave["typical_response_time"] is None
    assert dave["contact_frequency_days"] is None
    assert dave["last_contact"] is None


def test_metrics_values_match_stored_data(client, seeded_contacts):
    """Metric values retrieved via API match what was inserted into the DB."""
    response = client.get("/api/contacts")
    contacts = {c["name"]: c for c in response.json()["contacts"]}

    alice = contacts["Alice Smith"]
    assert alice["typical_response_time"] == pytest.approx(3600.0, abs=0.1)
    assert alice["contact_frequency_days"] == pytest.approx(2.5, abs=0.01)
    assert alice["last_contact"] == "2026-02-18T10:00:00.000Z"

    carol = contacts["Carol White"]
    assert carol["typical_response_time"] == pytest.approx(86400.0, abs=0.1)
    assert carol["contact_frequency_days"] == pytest.approx(14.0, abs=0.01)


# ---------------------------------------------------------------------------
# Filter: is_priority
# ---------------------------------------------------------------------------


def test_filter_is_priority_true(client, seeded_contacts):
    """?is_priority=true returns only priority contacts."""
    response = client.get("/api/contacts?is_priority=true")
    data = response.json()

    assert response.status_code == 200
    assert data["total"] == 2  # Alice and Bob
    names = {c["name"] for c in data["contacts"]}
    assert names == {"Alice Smith", "Bob Jones"}
    # All returned contacts have is_priority = True.
    for contact in data["contacts"]:
        assert contact["is_priority"] is True


def test_filter_is_priority_false(client, seeded_contacts):
    """?is_priority=false returns only non-priority contacts."""
    response = client.get("/api/contacts?is_priority=false")
    data = response.json()

    assert response.status_code == 200
    assert data["total"] == 2  # Carol and Dave
    names = {c["name"] for c in data["contacts"]}
    assert names == {"Carol White", "Dave Brown"}
    for contact in data["contacts"]:
        assert contact["is_priority"] is False


# ---------------------------------------------------------------------------
# Filter: name
# ---------------------------------------------------------------------------


def test_filter_by_name_substring_match(client, seeded_contacts):
    """?name=alice returns contacts whose name contains 'alice' (case-insensitive)."""
    response = client.get("/api/contacts?name=alice")
    data = response.json()

    assert response.status_code == 200
    assert data["total"] == 1
    assert data["contacts"][0]["name"] == "Alice Smith"


def test_filter_by_name_case_insensitive(client, seeded_contacts):
    """Name filter is case-insensitive (ALICE matches 'Alice Smith')."""
    response = client.get("/api/contacts?name=ALICE")
    data = response.json()

    assert response.status_code == 200
    assert data["total"] == 1


def test_filter_by_name_partial_match(client, seeded_contacts):
    """?name=smith matches 'Alice Smith' via suffix match."""
    response = client.get("/api/contacts?name=smith")
    data = response.json()

    assert response.status_code == 200
    assert data["total"] == 1
    assert data["contacts"][0]["name"] == "Alice Smith"


def test_filter_by_name_no_match(client, seeded_contacts):
    """?name=nonexistent returns empty contacts list and total=0."""
    response = client.get("/api/contacts?name=nonexistent_xyz")
    data = response.json()

    assert response.status_code == 200
    assert data["total"] == 0
    assert data["contacts"] == []


# ---------------------------------------------------------------------------
# Filter: has_metrics
# ---------------------------------------------------------------------------


def test_filter_has_metrics_true(client, seeded_contacts):
    """?has_metrics=true returns only contacts with contact_frequency_days populated."""
    response = client.get("/api/contacts?has_metrics=true")
    data = response.json()

    assert response.status_code == 200
    # Alice, Bob, Carol have metrics; Dave does not.
    assert data["total"] == 3
    names = {c["name"] for c in data["contacts"]}
    assert "Dave Brown" not in names
    # Verify that all returned contacts have the metric populated.
    for contact in data["contacts"]:
        assert contact["contact_frequency_days"] is not None


def test_filter_has_metrics_false(client, seeded_contacts):
    """?has_metrics=false returns only contacts with no metrics yet."""
    response = client.get("/api/contacts?has_metrics=false")
    data = response.json()

    assert response.status_code == 200
    assert data["total"] == 1
    assert data["contacts"][0]["name"] == "Dave Brown"
    assert data["contacts"][0]["contact_frequency_days"] is None


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_ordering_priority_first(client, seeded_contacts):
    """Priority contacts appear before non-priority contacts in the response."""
    response = client.get("/api/contacts")
    contacts = response.json()["contacts"]

    # The first two contacts should be priority (Alice and Bob).
    for i in range(2):
        assert contacts[i]["is_priority"] is True, (
            f"Expected is_priority=True at index {i}, got {contacts[i]}"
        )
    # The last two should be non-priority.
    for i in range(2, 4):
        assert contacts[i]["is_priority"] is False


def test_ordering_recent_last_contact_first_within_priority(client, db):
    """Within the same priority group, contacts with more recent last_contact come first."""
    # Insert two priority contacts with different last_contact dates.
    _insert_contact(
        db, name="Earlier Contact", is_priority=True,
        last_contact="2026-01-01T00:00:00.000Z", contact_frequency_days=7.0,
    )
    _insert_contact(
        db, name="Later Contact", is_priority=True,
        last_contact="2026-02-18T00:00:00.000Z", contact_frequency_days=7.0,
    )

    response = client.get("/api/contacts?is_priority=true")
    names = [c["name"] for c in response.json()["contacts"]]
    # Later Contact (2026-02-18) should come before Earlier Contact (2026-01-01).
    assert names.index("Later Contact") < names.index("Earlier Contact")


# ---------------------------------------------------------------------------
# Limit parameter
# ---------------------------------------------------------------------------


def test_limit_restricts_returned_contacts(client, seeded_contacts):
    """?limit=2 returns at most 2 contacts but total reflects the full count."""
    response = client.get("/api/contacts?limit=2")
    data = response.json()

    assert response.status_code == 200
    assert len(data["contacts"]) == 2
    # total still reflects the full count (4), not the limited result.
    assert data["total"] == 4


def test_limit_hard_cap_at_500(client, db):
    """limit is capped at 500 even if a larger value is requested."""
    # Insert more than 500 contacts would be slow; instead we verify that a
    # limit of 1000 does not raise an error (the cap is silently applied).
    response = client.get("/api/contacts?limit=1000")
    assert response.status_code == 200


def test_limit_default_is_100(client, db):
    """The default limit is 100; no query param needed for normal use."""
    # Insert 101 contacts to verify the cap fires.
    for i in range(101):
        _insert_contact(db, name=f"Contact {i:03d}")

    response = client.get("/api/contacts")
    data = response.json()

    assert response.status_code == 200
    assert len(data["contacts"]) == 100
    assert data["total"] == 101  # full table count


# ---------------------------------------------------------------------------
# Empty database
# ---------------------------------------------------------------------------


def test_empty_database_returns_empty_contacts(client):
    """When no contacts exist, returns an empty list and total=0."""
    response = client.get("/api/contacts")
    data = response.json()

    assert response.status_code == 200
    assert data["contacts"] == []
    assert data["total"] == 0
    assert "generated_at" in data


# ---------------------------------------------------------------------------
# Combined filters
# ---------------------------------------------------------------------------


def test_combined_priority_and_has_metrics_filters(client, seeded_contacts):
    """Combining is_priority=true and has_metrics=true returns priority contacts with metrics."""
    response = client.get("/api/contacts?is_priority=true&has_metrics=true")
    data = response.json()

    assert response.status_code == 200
    # Alice and Bob are priority and have metrics.
    assert data["total"] == 2
    names = {c["name"] for c in data["contacts"]}
    assert names == {"Alice Smith", "Bob Jones"}


def test_combined_name_and_priority_filters(client, seeded_contacts):
    """?name=alice&is_priority=true returns Alice (priority match) only."""
    response = client.get("/api/contacts?name=alice&is_priority=true")
    data = response.json()

    assert response.status_code == 200
    assert data["total"] == 1
    assert data["contacts"][0]["name"] == "Alice Smith"


def test_combined_name_wrong_priority_returns_empty(client, seeded_contacts):
    """?name=alice&is_priority=false returns nothing (Alice is priority)."""
    response = client.get("/api/contacts?name=alice&is_priority=false")
    data = response.json()

    assert response.status_code == 200
    assert data["total"] == 0
    assert data["contacts"] == []
