"""Tests for :mod:`api.routes.you` — ``GET /api/you`` self-portrait.

Every test spins up a fresh in-memory SQLite with the full v2 schema,
seeds the signal-profile rows that the repository reads, wires a real
:class:`~storage.repos.people.PeopleRepository` onto a dummy LifeOS
double, and hits the app via :class:`fastapi.testclient.TestClient`.
No mocks of the storage layer — same pattern :mod:`tests.api.test_routes_now`
established.

Coverage (per NEXT_TASKS.md Week 8 acceptance):

- Empty DB returns every list section as ``[]`` (never ``None``).
- Populated temporal / comm_template / routine / cadence profiles each
  flow into the matching YouOut section.
- 503 when ``people_repo`` is not wired onto life_os.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from storage import schema
from storage.repos.people import PeopleRepository

REF_NOW = 1_777_204_800  # 2026-04-22T12:00:00Z


class Clock:
    def __init__(self, t: float = REF_NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class DummyLifeOS:
    def __init__(self, people_repo=None) -> None:
        self.config: dict = {}
        self.people_repo = people_repo


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        c.execute(stmt)
    c.commit()
    yield c
    c.close()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def repo(conn, clock):
    return PeopleRepository(conn, now_fn=clock)


@pytest.fixture
def client(repo):
    life_os = DummyLifeOS(people_repo=repo)
    return TestClient(create_app(life_os))


def _insert_profile(conn, producer: str, key: str, profile: dict, updated_at: int = REF_NOW) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile, updated_at) VALUES (?, ?, ?, ?)",
        (producer, key, json.dumps(profile), updated_at),
    )
    conn.commit()


def _insert_contact(conn, contact_id: str, name: str) -> None:
    conn.execute(
        "INSERT INTO entities (id, kind, name) VALUES (?, 'contact', ?)",
        (contact_id, name),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# empty state
# ---------------------------------------------------------------------------


def test_get_you_empty_state_returns_empty_lists(client: TestClient) -> None:
    resp = client.get("/api/you")
    assert resp.status_code == 200
    data = resp.json()
    assert data["observed_months"] == 0
    assert data["interactions_count"] == 0
    assert data["confidence_pct"] == 0
    assert data["when_at_best"] == []
    assert data["how_you_write"] == []
    assert data["your_routines"] == []
    assert data["drifting"] == []


def test_get_you_returns_503_when_repo_missing() -> None:
    app = create_app(DummyLifeOS(people_repo=None))
    resp = TestClient(app).get("/api/you")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# populated sections
# ---------------------------------------------------------------------------


def test_get_you_renders_temporal_focus_windows(client: TestClient, conn) -> None:
    _insert_profile(
        conn,
        "temporal",
        "self",
        {
            "data_days": 30,
            "tz_offset_hours": 0,
            "focus_windows": [
                {"start_hour": 9, "end_hour": 11, "description": "deep work"},
                {"start_hour": 14, "end_hour": 16, "description": "afternoon focus"},
            ],
        },
    )
    data = client.get("/api/you").json()
    assert data["when_at_best"] == [
        "09:00-11:00 - deep work",
        "14:00-16:00 - afternoon focus",
    ]


def test_get_you_renders_routines(client: TestClient, conn) -> None:
    _insert_profile(
        conn,
        "routine",
        "plan_week",
        {
            "description": "plan your week",
            "weekday": 6,
            "hour": 17,
            "last_occurrences": ["evt-1", "evt-2", "evt-3", "evt-4"],
            "consistency": 0.85,
        },
    )
    data = client.get("/api/you").json()
    assert len(data["your_routines"]) == 1
    routine = data["your_routines"][0]
    assert routine["name"] == "plan_week"
    assert routine["description"] == "plan your week"
    assert routine["detected"] is True
    assert routine["sample_size"] == 4
    assert routine["confidence"] == pytest.approx(0.85)


def test_get_you_skips_routines_with_empty_description(client: TestClient, conn) -> None:
    _insert_profile(conn, "routine", "no-desc", {"description": "", "consistency": 0.5})
    data = client.get("/api/you").json()
    assert data["your_routines"] == []


def test_get_you_renders_personas_capped_at_six(client: TestClient, conn) -> None:
    for i in range(8):
        _insert_profile(
            conn,
            "comm_template",
            f"c{i}",
            {
                "contact_name": f"Contact {i}",
                "template_style": "casual",
                "formality": 0.3,
                "last_event_ids": ["a", "b"],
            },
            updated_at=REF_NOW + i,
        )
    data = client.get("/api/you").json()
    assert len(data["how_you_write"]) == 6
    first = data["how_you_write"][0]
    assert first["tone"] == "casual"
    assert first["formality"] == pytest.approx(0.3)
    assert first["sample_size"] == 2


def test_get_you_surfaces_drifting_contacts(client: TestClient, conn) -> None:
    # Past drift threshold (30 days since, usual cadence 7d).
    _insert_profile(
        conn,
        "cadence",
        "alice",
        {
            "expected_cadence_days": 7.0,
            "count": 12,
            "last_inbound_ts": REF_NOW - 30 * 86400,
            "contact_name": "Alice",
            "last_inbound_event_ids": ["evt-a"],
        },
    )
    # Not drifted — 5 days since, usual cadence 7d.
    _insert_profile(
        conn,
        "cadence",
        "bob",
        {
            "expected_cadence_days": 7.0,
            "count": 20,
            "last_inbound_ts": REF_NOW - 5 * 86400,
            "contact_name": "Bob",
            "last_inbound_event_ids": ["evt-b"],
        },
    )
    # Low history — below MIN_CADENCE_COUNT even though overdue.
    _insert_profile(
        conn,
        "cadence",
        "carol",
        {
            "expected_cadence_days": 7.0,
            "count": 2,
            "last_inbound_ts": REF_NOW - 30 * 86400,
            "contact_name": "Carol",
        },
    )
    data = client.get("/api/you").json()
    assert [d["contact_id"] for d in data["drifting"]] == ["alice"]
    assert data["drifting"][0]["days_since_last"] == 30
    assert data["drifting"][0]["usual_cadence_days"] == 7


def test_get_you_confidence_and_interactions_from_events(client: TestClient, conn) -> None:
    now = REF_NOW
    # 3 events spread across ~65 days (~2 months).
    for i, age_days in enumerate([0, 30, 65]):
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, 'email.received', 'gmail', ?, 'normal', '{}')",
            (f"evt-{i}", now - age_days * 86400),
        )
    # Two producers with ≥1 decision (0.8 and 0.6) → avg 0.7 → 70%.
    conn.execute(
        "INSERT INTO feedback_weights (insight_type, weight, decision_count) VALUES "
        "('cadence', 0.8, 3), ('relationship', 0.6, 5), ('temporal', 1.0, 0)"
    )
    conn.commit()
    data = client.get("/api/you").json()
    assert data["interactions_count"] == 3
    assert data["observed_months"] == 2  # 65 days // 30
    assert data["confidence_pct"] == 70


def test_get_you_uses_entities_name_when_profile_missing_contact_name(client: TestClient, conn) -> None:
    _insert_contact(conn, "dave-id", "Dave from Entities")
    _insert_profile(
        conn,
        "cadence",
        "dave-id",
        {
            "expected_cadence_days": 5.0,
            "count": 10,
            "last_inbound_ts": REF_NOW - 30 * 86400,
            "last_inbound_event_ids": ["e"],
        },
    )
    data = client.get("/api/you").json()
    assert data["drifting"][0]["name"] == "Dave from Entities"
