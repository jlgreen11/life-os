"""Tests for :mod:`api.routes.context` — iOS compat shim + legacy proxies.

Three categories mirror the module layout (per NEXT_TASKS.md Week 8):

1. **Context pipeline** — realistic iOS payloads round-trip to the v2
   ``events`` table and summary reads aggregate them back out.
2. **Proxies** — ``/api/status``, ``/api/briefing``, ``/api/feedback``,
   ``/api/preferences``, and the ``/ws`` WebSocket.
3. **501 stubs** — every v1 endpoint that has no v2 replacement yet
   returns a structured 501 body.

Every fixture builds a fresh in-memory SQLite with the full v2 schema;
no mocking of the storage layer (same pattern as the other Week 8
route tests).
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from storage import schema

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class DummyLifeOS:
    """Minimal life_os double the context shim dereferences."""

    def __init__(
        self,
        *,
        db: sqlite3.Connection | None = None,
        ai_engine: Any | None = None,
        context_assembler: Any | None = None,
        ws_manager: Any | None = None,
    ) -> None:
        self.config: dict = {}
        self.db = db
        self.ai_engine = ai_engine
        self.context_assembler = context_assembler
        self.ws_manager = ws_manager


@pytest.fixture
def db() -> sqlite3.Connection:
    """A fresh in-memory SQLite with the v2 schema, FKs on."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        conn.execute(stmt)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def client(db: sqlite3.Connection) -> TestClient:
    return TestClient(create_app(DummyLifeOS(db=db)))


# ---------------------------------------------------------------------------
# /api/context/event
# ---------------------------------------------------------------------------


IOS_LOCATION_EVENT = {
    "type": "context.location",
    "source": "ios_app",
    "timestamp": "2026-04-22T12:00:00Z",
    "payload": {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "place_name": "Home",
        "place_type": "home",
        "horizontal_accuracy": 12.5,
    },
    "metadata": {
        "device_model": "iPhone15,2",
        "os_version": "17.2",
        "battery_level": 0.82,
        "network_type": "wifi",
        "app_state": "active",
    },
}

IOS_DEVICE_EVENT = {
    "type": "context.device_nearby",
    "source": "ios_app",
    "timestamp": "2026-04-22T12:05:00Z",
    "payload": {
        "device_name": "Mac-mini.local",
        "device_type": "computer",
        "signal_strength": -42,
        "is_connected": True,
    },
}


def test_context_event_stores_row_and_returns_id(client: TestClient, db: sqlite3.Connection) -> None:
    resp = client.post("/api/context/event", json=IOS_LOCATION_EVENT)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "received"
    assert isinstance(body["event_id"], str) and len(body["event_id"]) > 0

    row = db.execute("SELECT type, source, timestamp, priority, metadata FROM events").fetchone()
    assert row["type"] == "location.changed"
    assert row["source"] == "ios_app"
    assert row["priority"] == "silent"
    assert "context.location" in row["metadata"]
    # Timestamp is stored as Unix seconds — sanity-check it's in 2026.
    assert row["timestamp"] > 1_700_000_000


def test_context_event_unknown_type_falls_back_to_command(client: TestClient, db: sqlite3.Connection) -> None:
    evt = dict(IOS_LOCATION_EVENT)
    evt["type"] = "context.made.up"
    resp = client.post("/api/context/event", json=evt)
    assert resp.status_code == 200
    row = db.execute("SELECT type FROM events").fetchone()
    assert row["type"] == "system.user.command"


def test_context_event_accepts_missing_timestamp(client: TestClient, db: sqlite3.Connection) -> None:
    evt = dict(IOS_LOCATION_EVENT)
    evt.pop("timestamp")
    resp = client.post("/api/context/event", json=evt)
    assert resp.status_code == 200
    row = db.execute("SELECT timestamp FROM events").fetchone()
    assert row["timestamp"] > 0


def test_context_event_rejects_extra_fields(client: TestClient) -> None:
    evt = dict(IOS_LOCATION_EVENT)
    evt["unknown"] = "drop-me"
    resp = client.post("/api/context/event", json=evt)
    assert resp.status_code == 422


def test_context_event_503_when_db_missing() -> None:
    c = TestClient(create_app(DummyLifeOS(db=None)))
    resp = c.post("/api/context/event", json=IOS_LOCATION_EVENT)
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /api/context/batch
# ---------------------------------------------------------------------------


def test_context_batch_inserts_every_event(client: TestClient, db: sqlite3.Connection) -> None:
    payload = {"events": [IOS_LOCATION_EVENT, IOS_DEVICE_EVENT]}
    resp = client.post("/api/context/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "received"
    assert body["count"] == 2
    assert len(body["event_ids"]) == 2

    count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 2


def test_context_batch_empty_list_is_200(client: TestClient) -> None:
    resp = client.post("/api/context/batch", json={"events": []})
    assert resp.status_code == 200
    assert resp.json() == {"status": "received", "count": 0, "event_ids": []}


# ---------------------------------------------------------------------------
# /api/context/summary
# ---------------------------------------------------------------------------


def test_context_summary_aggregates_locations_and_devices(
    client: TestClient,
    db: sqlite3.Connection,
) -> None:
    client.post("/api/context/event", json=IOS_LOCATION_EVENT)
    client.post("/api/context/event", json=IOS_DEVICE_EVENT)

    resp = client.get("/api/context/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "context_summary"
    assert "Home" in body["unique_places"]
    assert "Mac-mini.local" in body["unique_devices"]
    assert len(body["locations"]) == 1
    assert len(body["devices"]) == 1
    assert "1 location updates" in body["content"]


def test_context_summary_empty_install(client: TestClient) -> None:
    body = client.get("/api/context/summary").json()
    assert body["unique_places"] == []
    assert body["unique_devices"] == []
    assert body["locations"] == []
    assert body["devices"] == []


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------


def test_status_reports_live_counts(client: TestClient) -> None:
    client.post("/api/context/event", json=IOS_LOCATION_EVENT)
    body = client.get("/api/status").json()
    assert body["ok"] is True
    assert body["event_count"] == 1
    assert body["moment_count"] == 0


def test_status_fails_open_without_db() -> None:
    c = TestClient(create_app(DummyLifeOS(db=None)))
    resp = c.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["event_count"] == 0


# ---------------------------------------------------------------------------
# /api/briefing
# ---------------------------------------------------------------------------


class FakeAssembler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def assemble_briefing_context(self, user_id: str, date: Any) -> dict[str, Any]:
        self.calls.append((user_id, date))
        return {"calendar": [], "moments": [], "preferences": {}}


class FakeEngine:
    def __init__(self, text: str = "Your day looks clear.", raises: Exception | None = None) -> None:
        self.text = text
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    async def briefing_synthesis(self, context: dict[str, Any]) -> str:
        self.calls.append(context)
        if self.raises is not None:
            raise self.raises
        return self.text


def test_briefing_returns_synthesised_text(db: sqlite3.Connection) -> None:
    engine = FakeEngine("Morning. One meeting at 10.")
    assembler = FakeAssembler()
    life_os = DummyLifeOS(db=db, ai_engine=engine, context_assembler=assembler)
    c = TestClient(create_app(life_os))

    body = c.get("/api/briefing").json()
    assert body["briefing"] == "Morning. One meeting at 10."
    assert "generated_at" in body
    assert len(engine.calls) == 1
    assert len(assembler.calls) == 1


def test_briefing_missing_wiring_returns_null_with_error(client: TestClient) -> None:
    body = client.get("/api/briefing").json()
    assert body["briefing"] is None
    assert "not wired" in body["error"]


def test_briefing_engine_failure_returns_null_not_500(db: sqlite3.Connection) -> None:
    engine = FakeEngine(raises=RuntimeError("ollama down"))
    assembler = FakeAssembler()
    life_os = DummyLifeOS(db=db, ai_engine=engine, context_assembler=assembler)
    c = TestClient(create_app(life_os))

    resp = c.get("/api/briefing")
    assert resp.status_code == 200
    body = resp.json()
    assert body["briefing"] is None
    assert "temporarily unavailable" in body["error"]


# ---------------------------------------------------------------------------
# /api/feedback
# ---------------------------------------------------------------------------


def test_feedback_persists_as_system_event(client: TestClient, db: sqlite3.Connection) -> None:
    resp = client.post("/api/feedback", json={"message": "love the new flow"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "received"

    row = db.execute("SELECT type, payload, metadata FROM events").fetchone()
    assert row["type"] == "system.user.feedback"
    assert "love the new flow" in row["payload"]
    assert "feedback" in row["metadata"]


def test_feedback_requires_message(client: TestClient) -> None:
    resp = client.post("/api/feedback", json={})
    assert resp.status_code == 422


def test_feedback_503_without_db() -> None:
    c = TestClient(create_app(DummyLifeOS(db=None)))
    resp = c.post("/api/feedback", json={"message": "hi"})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /api/preferences
# ---------------------------------------------------------------------------


def test_preferences_post_upserts_value(client: TestClient, db: sqlite3.Connection) -> None:
    resp = client.post("/api/preferences", json={"key": "autonomy", "value": "suggest"})
    assert resp.status_code == 200

    row = db.execute("SELECT key, value, encrypted FROM preferences WHERE key='autonomy'").fetchone()
    assert row["value"] == "suggest"
    assert row["encrypted"] == 0


def test_preferences_put_is_v1_alias(client: TestClient, db: sqlite3.Connection) -> None:
    resp = client.put("/api/preferences", json={"key": "proactivity", "value": 0.7})
    assert resp.status_code == 200
    row = db.execute("SELECT value FROM preferences WHERE key='proactivity'").fetchone()
    # Non-string values round-trip through JSON encoding.
    assert row["value"] == "0.7"


def test_preferences_round_trip_nested_object(client: TestClient, db: sqlite3.Connection) -> None:
    payload = {"key": "quiet_hours", "value": {"start": "22:00", "end": "07:00"}}
    client.post("/api/preferences", json=payload)
    row = db.execute("SELECT value FROM preferences WHERE key='quiet_hours'").fetchone()
    assert row["value"].startswith("{") and "22:00" in row["value"]


def test_preferences_upsert_replaces_existing(client: TestClient, db: sqlite3.Connection) -> None:
    client.post("/api/preferences", json={"key": "tone", "value": "warm"})
    client.post("/api/preferences", json={"key": "tone", "value": "direct"})
    rows = db.execute("SELECT value FROM preferences WHERE key='tone'").fetchall()
    assert len(rows) == 1
    assert rows[0]["value"] == "direct"


# ---------------------------------------------------------------------------
# WebSocket /ws
# ---------------------------------------------------------------------------


class RecordingManager:
    def __init__(self) -> None:
        self.connected: list[Any] = []
        self.disconnected: list[Any] = []

    async def connect(self, websocket: Any) -> None:
        self.connected.append(websocket)

    async def disconnect(self, websocket: Any) -> None:
        self.disconnected.append(websocket)


def test_websocket_accepts_connection_without_manager(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_text("hello")
        ws.close()


def test_websocket_registers_with_manager_when_wired(db: sqlite3.Connection) -> None:
    manager = RecordingManager()
    life_os = DummyLifeOS(db=db, ws_manager=manager)
    c = TestClient(create_app(life_os))
    with c.websocket_connect("/ws") as ws:
        ws.send_text("ping")
        ws.close()
    assert len(manager.connected) == 1
    assert len(manager.disconnected) == 1


# ---------------------------------------------------------------------------
# 501 stubs
# ---------------------------------------------------------------------------


STUB_ENDPOINTS = [
    ("POST", "/api/command"),
    ("GET", "/api/notifications"),
    ("GET", "/api/tasks"),
    ("POST", "/api/tasks"),
    ("POST", "/api/search"),
]


@pytest.mark.parametrize(("method", "path"), STUB_ENDPOINTS)
def test_stub_returns_501_with_helpful_body(
    client: TestClient,
    method: str,
    path: str,
) -> None:
    resp = client.request(method, path)
    assert resp.status_code == 501
    body = resp.json()
    assert body["error"] == "not_implemented_in_v2"
    assert body["endpoint"].startswith(method)
    assert body["follow_up"]  # non-empty guidance string
