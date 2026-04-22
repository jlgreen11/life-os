"""Tests for :mod:`api.routes.people` — roster and per-contact dossier.

Coverage (per NEXT_TASKS.md Week 8 acceptance):

- ``GET /api/people`` empty state returns the YouOut envelope + empty
  lists (never ``None``).
- Drifting contacts land in ``needs_attention``; recent contacts land
  in ``active_this_week``.
- Search (``q=...``) filters by name substring.
- Pagination: ``page_size`` caps the page; ``total`` reflects full
  post-filter count.
- ``GET /api/people/{id}`` returns the dossier for a known contact,
  404 for unknown.
- 503 when ``people_repo`` is not wired.
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
    return TestClient(create_app(DummyLifeOS(people_repo=repo)))


def _insert_profile(conn, producer, key, profile, updated_at=REF_NOW) -> None:
    conn.execute(
        "INSERT INTO signal_profiles (producer, key, profile, updated_at) VALUES (?, ?, ?, ?)",
        (producer, key, json.dumps(profile), updated_at),
    )
    conn.commit()


def _insert_contact(conn, contact_id, name) -> None:
    conn.execute(
        "INSERT INTO entities (id, kind, name) VALUES (?, 'contact', ?)",
        (contact_id, name),
    )
    conn.commit()


def _insert_event(conn, event_id, ts, contact_id) -> None:
    payload = json.dumps({"contact_id": contact_id, "subject": "hello"})
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp, priority, payload) "
        "VALUES (?, 'email.received', 'gmail', ?, 'normal', ?)",
        (event_id, ts, payload),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# GET /api/people
# ---------------------------------------------------------------------------


def test_list_people_empty_state_returns_empty_lists(client: TestClient) -> None:
    resp = client.get("/api/people")
    assert resp.status_code == 200
    data = resp.json()
    # YOU envelope is always present, even empty.
    assert "you" in data
    assert isinstance(data["you"], dict)
    assert data["needs_attention"] == []
    assert data["active_this_week"] == []
    assert data["total"] == 0
    assert data["query"] is None


def test_list_people_returns_503_when_repo_missing() -> None:
    app = create_app(DummyLifeOS(people_repo=None))
    resp = TestClient(app).get("/api/people")
    assert resp.status_code == 503


def test_list_people_sorts_drifting_into_needs_attention(client: TestClient, conn) -> None:
    _insert_contact(conn, "alice", "Alice")
    _insert_contact(conn, "bob", "Bob")
    # Alice is drifting.
    _insert_profile(
        conn,
        "cadence",
        "alice",
        {
            "expected_cadence_days": 7.0,
            "count": 12,
            "last_inbound_ts": REF_NOW - 30 * 86400,
            "contact_name": "Alice",
            "last_inbound_event_ids": ["e"],
        },
    )
    # Bob was contacted yesterday — active this week, not drifting.
    _insert_profile(
        conn,
        "cadence",
        "bob",
        {
            "expected_cadence_days": 7.0,
            "count": 30,
            "last_inbound_ts": REF_NOW - 1 * 86400,
            "contact_name": "Bob",
            "last_inbound_event_ids": ["e"],
        },
    )
    data = client.get("/api/people").json()
    assert [c["name"] for c in data["needs_attention"]] == ["Alice"]
    assert [c["name"] for c in data["active_this_week"]] == ["Bob"]
    # needs_attention entries carry a deviation (positive = overdue).
    assert data["needs_attention"][0]["cadence_deviation_days"] == 23
    assert data["total"] == 2


def test_list_people_search_filters_by_name(client: TestClient, conn) -> None:
    _insert_contact(conn, "alice", "Alice Smith")
    _insert_contact(conn, "bob", "Bob Jones")
    _insert_profile(
        conn,
        "cadence",
        "alice",
        {
            "expected_cadence_days": 7.0,
            "count": 12,
            "last_inbound_ts": REF_NOW - 1 * 86400,
            "contact_name": "Alice Smith",
            "last_inbound_event_ids": ["e"],
        },
    )
    _insert_profile(
        conn,
        "cadence",
        "bob",
        {
            "expected_cadence_days": 7.0,
            "count": 12,
            "last_inbound_ts": REF_NOW - 1 * 86400,
            "contact_name": "Bob Jones",
            "last_inbound_event_ids": ["e"],
        },
    )
    data = client.get("/api/people", params={"q": "alice"}).json()
    assert data["total"] == 1
    assert data["query"] == "alice"
    assert [c["name"] for c in data["active_this_week"]] == ["Alice Smith"]


def test_list_people_pagination_clips_page(client: TestClient, conn) -> None:
    # 30 contacts, all active this week.
    for i in range(30):
        cid = f"c-{i:03d}"
        _insert_contact(conn, cid, f"Contact {i:03d}")
        _insert_profile(
            conn,
            "cadence",
            cid,
            {
                "expected_cadence_days": 7.0,
                "count": 12,
                "last_inbound_ts": REF_NOW - 1 * 86400,
                "contact_name": f"Contact {i:03d}",
                "last_inbound_event_ids": ["e"],
            },
        )
    page1 = client.get("/api/people", params={"page": 1, "page_size": 10}).json()
    page2 = client.get("/api/people", params={"page": 2, "page_size": 10}).json()
    assert page1["total"] == 30
    assert page2["total"] == 30
    assert len(page1["active_this_week"]) == 10
    assert len(page2["active_this_week"]) == 10
    page1_ids = {c["contact_id"] for c in page1["active_this_week"]}
    page2_ids = {c["contact_id"] for c in page2["active_this_week"]}
    assert page1_ids.isdisjoint(page2_ids)


def test_list_people_rejects_invalid_pagination(client: TestClient) -> None:
    assert client.get("/api/people", params={"page": 0}).status_code == 422
    assert client.get("/api/people", params={"page_size": 0}).status_code == 422
    assert client.get("/api/people", params={"page_size": 99999}).status_code == 422


# ---------------------------------------------------------------------------
# GET /api/people/{contact_id}
# ---------------------------------------------------------------------------


def test_get_dossier_returns_full_shape(client: TestClient, conn) -> None:
    _insert_contact(conn, "alice", "Alice")
    _insert_profile(
        conn,
        "cadence",
        "alice",
        {
            "expected_cadence_days": 7.0,
            "count": 12,
            "last_inbound_ts": REF_NOW - 3 * 86400,
            "contact_name": "Alice",
            "last_inbound_event_ids": ["e"],
        },
    )
    _insert_profile(
        conn,
        "comm_template",
        "alice",
        {
            "contact_name": "Alice",
            "template_style": "casual",
            "template_body": "Hey {name},",
            "last_event_ids": ["a", "b"],
        },
    )
    # Three events over last 3 days — drop into sparkline.
    for i in range(3):
        _insert_event(conn, f"evt-{i}", REF_NOW - i * 86400, "alice")

    resp = client.get("/api/people/alice")
    assert resp.status_code == 200
    data = resp.json()
    assert data["contact_id"] == "alice"
    assert data["name"] == "Alice"
    assert data["last_contact_ts"] == REF_NOW - 3 * 86400
    assert data["usual_cadence_days"] == 7
    assert data["comm_template"] == "Hey {name},"
    # 14-day sparkline; last 3 days each have 1 event.
    assert len(data["cadence_sparkline"]) == 14
    assert data["cadence_sparkline"][-1] == 1
    assert data["cadence_sparkline"][-2] == 1
    assert data["cadence_sparkline"][-3] == 1
    # predicted_next is a human string (next in 4 days: 3 days since + 7d cadence).
    assert data["predicted_next"] == "Next contact expected in ~4 days"


def test_get_dossier_unknown_contact_returns_404(client: TestClient) -> None:
    resp = client.get("/api/people/does-not-exist")
    assert resp.status_code == 404


def test_get_dossier_overdue_predicted_next(client: TestClient, conn) -> None:
    _insert_contact(conn, "frank", "Frank")
    _insert_profile(
        conn,
        "cadence",
        "frank",
        {
            "expected_cadence_days": 7.0,
            "count": 20,
            "last_inbound_ts": REF_NOW - 30 * 86400,
            "contact_name": "Frank",
            "last_inbound_event_ids": ["e"],
        },
    )
    data = client.get("/api/people/frank").json()
    assert data["predicted_next"] == "Overdue by 23 days"


def test_get_dossier_no_cadence_profile_has_null_predicted_next(client: TestClient, conn) -> None:
    _insert_contact(conn, "greta", "Greta")
    data = client.get("/api/people/greta").json()
    assert data["predicted_next"] is None
    assert data["last_contact_ts"] is None
    assert data["usual_cadence_days"] is None
    assert data["cadence_sparkline"] == [0] * 14
    assert data["recent_topics"] == []
    assert data["comm_template"] is None


def test_get_dossier_returns_503_when_repo_missing() -> None:
    app = create_app(DummyLifeOS(people_repo=None))
    resp = TestClient(app).get("/api/people/alice")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# HTML pages: GET /people, GET /people/{id}
# ---------------------------------------------------------------------------


def test_people_page_renders_html_with_search_input(client: TestClient, conn) -> None:
    _insert_contact(conn, "alice", "Alice")
    _insert_profile(
        conn,
        "cadence",
        "alice",
        {
            "expected_cadence_days": 7.0,
            "count": 12,
            "last_inbound_ts": REF_NOW - 1 * 86400,
            "contact_name": "Alice",
            "last_inbound_event_ids": ["e"],
        },
    )
    resp = client.get("/people")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert 'id="people-results"' in body
    assert 'hx-get="/api/people"' in body
    assert 'data-active-tab="people"' in body
    # Roster row is server-rendered for the no-JS first paint.
    assert "Alice" in body


def test_people_page_returns_503_when_repo_missing() -> None:
    app = create_app(DummyLifeOS(people_repo=None))
    resp = TestClient(app).get("/people")
    assert resp.status_code == 503


def test_list_people_htmx_returns_partial(client: TestClient, conn) -> None:
    _insert_contact(conn, "alice", "Alice")
    _insert_profile(
        conn,
        "cadence",
        "alice",
        {
            "expected_cadence_days": 7.0,
            "count": 12,
            "last_inbound_ts": REF_NOW - 1 * 86400,
            "contact_name": "Alice",
            "last_inbound_event_ids": ["e"],
        },
    )
    resp = client.get("/api/people", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    # Partial returns the YOU pinned row + the two sub-list sections,
    # but NOT the page chrome (no <html> / no nav).
    assert 'data-section="you-pinned"' in body
    assert 'data-section="active-this-week"' in body
    assert "<html" not in body
    assert 'data-section="people-header"' not in body


def test_list_people_htmx_search_filters(client: TestClient, conn) -> None:
    _insert_contact(conn, "alice", "Alice Smith")
    _insert_contact(conn, "bob", "Bob Jones")
    for cid, name in [("alice", "Alice Smith"), ("bob", "Bob Jones")]:
        _insert_profile(
            conn,
            "cadence",
            cid,
            {
                "expected_cadence_days": 7.0,
                "count": 12,
                "last_inbound_ts": REF_NOW - 1 * 86400,
                "contact_name": name,
                "last_inbound_event_ids": ["e"],
            },
        )
    resp = client.get(
        "/api/people",
        headers={"HX-Request": "true"},
        params={"q": "alice"},
    )
    assert resp.status_code == 200
    body = resp.text
    assert "Alice Smith" in body
    assert "Bob Jones" not in body


def test_contact_page_renders_dossier_html(client: TestClient, conn) -> None:
    _insert_contact(conn, "alice", "Alice")
    _insert_profile(
        conn,
        "cadence",
        "alice",
        {
            "expected_cadence_days": 7.0,
            "count": 12,
            "last_inbound_ts": REF_NOW - 3 * 86400,
            "contact_name": "Alice",
            "last_inbound_event_ids": ["e"],
        },
    )
    resp = client.get("/people/alice")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert 'data-active-tab="people"' in body
    assert 'data-slot="dossier-name">Alice<' in body
    assert 'data-section="dossier-cadence"' in body


def test_contact_page_unknown_contact_returns_404(client: TestClient) -> None:
    resp = client.get("/people/does-not-exist")
    assert resp.status_code == 404


def test_contact_page_returns_503_when_repo_missing() -> None:
    app = create_app(DummyLifeOS(people_repo=None))
    resp = TestClient(app).get("/people/alice")
    assert resp.status_code == 503


def test_list_people_non_htmx_still_returns_json(client: TestClient) -> None:
    """Plain GET /api/people without HX-Request must keep JSON shape."""
    resp = client.get("/api/people")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    data = resp.json()
    assert set(data.keys()) >= {"you", "needs_attention", "active_this_week", "total", "query"}
