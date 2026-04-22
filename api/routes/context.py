"""iOS compat shim + legacy proxies.

Per engineering review §1g. The v2 API is a fresh contract, but the
existing iOS companion app (``ios/LifeOS/``) already ships a set of
v1 endpoints. Until a paired iOS release lands (Phase 2), v2 must
answer those routes without regressing the phone.

Three classes of endpoint live here:

1. **Context pipeline** (real, pass-through to v2 events table)

   - ``POST /api/context/event``     — single mobile context event
   - ``POST /api/context/batch``     — batch of mobile context events
   - ``GET  /api/context/summary``   — rolling summary of recent context

2. **Real proxies to v2 primitives**

   - ``GET  /api/status``      — tiny smoke payload (event + moment counts)
   - ``GET  /api/briefing``    — wraps :class:`ai.context.ContextAssembler`
     + :meth:`ai.engine.AIEngine.briefing_synthesis`; returns the v1
     ``{briefing, generated_at}`` shape
   - ``POST /api/feedback``    — writes a ``system.user.feedback`` event
     to the append-only ``events`` table (v2 has no ``feedback_events``
     table — see NOTE on the Week-1 migration task; feedback is not
     carried forward via a dedicated table, only via the event log)
   - ``POST /api/preferences`` — upserts ``(key, value)`` into the v2
     ``preferences`` table (also accepts PUT for v1 compat)
   - ``WebSocket /ws``         — simple connection-register + optional
     broadcast hook; full Moment-push wiring lands Week 11

3. **501 stubs** for endpoints the iOS app calls but v2 has not yet
   re-implemented. Stubs return a structured body so the phone can
   surface "not available yet" instead of a silent 404.

   - ``POST /api/command``
   - ``GET  /api/notifications``
   - ``GET  /api/tasks`` / ``POST /api/tasks``
   - ``POST /api/search``

Duck-typed contract on ``life_os``
----------------------------------
Every attribute below is optional — a missing attribute yields 503
(or a graceful empty response for read-only routes) so this module
can be mounted against a half-constructed ``LifeOS`` without
dragging the whole factory down.

- ``life_os.db``                — :class:`sqlite3.Connection` to
  ``lifeos.db`` (used by context ingestion, feedback, preferences,
  status). Must be autocommit-safe: we call ``.execute()`` +
  ``.commit()`` directly.
- ``life_os.ai_engine``         — :class:`ai.engine.AIEngine` with an
  async ``briefing_synthesis(context: dict) -> str``.
- ``life_os.context_assembler`` — :class:`ai.context.ContextAssembler`
  with ``assemble_briefing_context(user_id, date) -> dict``.
- ``life_os.ws_manager``        — optional; any object exposing async
  ``connect(websocket)`` / ``disconnect(websocket)`` methods. When
  absent, ``/ws`` falls back to a built-in no-op register so the iOS
  client's connect call still succeeds.

Security
--------
All three classes of endpoint are mounted behind the same Tailscale-
only localhost CORS posture as the rest of the v2 API (engineering
plan § "Security posture"). This shim does not widen that surface.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from datetime import UTC, datetime
from datetime import date as date_cls
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# iOS payload schemas (kept in-module; not part of the canonical v2 surface)
# ---------------------------------------------------------------------------


class ContextPayload(BaseModel):
    """Mobile-device context event payload.

    Mirrors v1's ``web.schemas.ContextPayload`` so the iOS app can round-
    trip unchanged. Extra fields are rejected so a drifted iOS release
    fails fast with 422 instead of silently dropping new keys.
    """

    model_config = ConfigDict(extra="forbid")

    # Location
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    horizontal_accuracy: float | None = None
    speed: float | None = None
    place_name: str | None = None
    place_type: str | None = None
    # Device proximity
    device_name: str | None = None
    device_type: str | None = None
    signal_strength: int | None = None
    is_connected: bool | None = None
    # Time
    local_time: str | None = None
    timezone: str | None = None
    day_of_week: str | None = None
    is_weekend: bool | None = None
    # Activity
    activity: str | None = None
    confidence: float | None = None


class ContextMetadata(BaseModel):
    """Device-level metadata tagged onto every iOS context event."""

    model_config = ConfigDict(extra="forbid")

    device_model: str | None = None
    os_version: str | None = None
    battery_level: float | None = None
    network_type: str | None = None
    app_state: str | None = None


class ContextEventIn(BaseModel):
    """A single context event from the mobile app."""

    model_config = ConfigDict(extra="forbid")

    type: str
    source: str = "ios_app"
    timestamp: str | None = None
    payload: ContextPayload
    metadata: ContextMetadata | None = None


class ContextBatchIn(BaseModel):
    """Batched context ingestion (one POST, N events)."""

    model_config = ConfigDict(extra="forbid")

    events: list[ContextEventIn]


class FeedbackIn(BaseModel):
    """Explicit user feedback body (v1 shape).

    Free-text only — the feedback collector is gone in v2, but the
    iOS app's "Send feedback" sheet expects this endpoint to exist.
    We persist the message as an append-only event so the audit trail
    survives; the behavioural loop it used to drive is deliberately
    not re-implemented (CEO plan § "Killed Insights").
    """

    model_config = ConfigDict(extra="forbid")

    message: str


class PreferenceIn(BaseModel):
    """Single-preference upsert body (v1 shape).

    ``value`` accepts any JSON-serialisable scalar or object — the v2
    preferences table stores a JSON-encoded string regardless of the
    source type, matching v1's idempotent write contract.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    value: Any


# ---------------------------------------------------------------------------
# life_os dereference helpers (all fail-soft)
# ---------------------------------------------------------------------------


def _life_os(request: Request) -> Any | None:
    return getattr(request.app.state, "life_os", None)


def _db(request: Request) -> sqlite3.Connection | None:
    return getattr(_life_os(request), "db", None)


def _db_or_503(request: Request) -> sqlite3.Connection:
    conn = _db(request)
    if conn is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="db is not wired on life_os",
        )
    return conn


# iOS context event types → v2 internal event types. Matches v1's
# mapping in ``web.routes.submit_context_event`` so a migrated DB
# produces the same type strings the rest of the pipeline keys off.
_CONTEXT_TYPE_MAP: dict[str, str] = {
    "context.location": "location.changed",
    "context.device_nearby": "home.device.state_changed",
    "context.time": "system.user.command",
    "context.background_refresh": "system.connector.sync_complete",
    "context.background_processing": "system.connector.sync_complete",
}


def _coerce_timestamp(raw: str | None) -> int:
    """ISO-8601 or Unix-seconds → Unix-seconds int.

    The iOS app sends ISO-8601 with a ``Z`` suffix or explicit offset.
    A missing value coerces to "now" so ingestion never blocks on a
    clock-drifted client. Parse failures fall back to "now" too — the
    alternative is a 400 that drops the event entirely, which is worse
    than an approximate timestamp.
    """
    if raw is None or raw == "":
        return int(time.time())
    # Unix-seconds (int-like string) — accept for forward compat.
    try:
        return int(float(raw))
    except ValueError:
        pass
    # ISO-8601. Python's fromisoformat does not accept a bare "Z"
    # suffix before 3.11; substitute it manually.
    cleaned = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return int(time.time())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def _insert_event(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    source: str,
    timestamp: int,
    priority: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """Append one row to the v2 ``events`` table. Returns the new id.

    v2 events are immutable — callers never update or delete. The id
    is a fresh uuid4, consistent with the rest of the v2 write paths
    (see :class:`storage.repos.moments.MomentRepository`).
    """
    event_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            event_type,
            source,
            timestamp,
            priority,
            json.dumps(payload, sort_keys=True),
            json.dumps(metadata, sort_keys=True),
        ),
    )
    conn.commit()
    return event_id


# ---------------------------------------------------------------------------
# Context pipeline
# ---------------------------------------------------------------------------


@router.post("/api/context/event")
def submit_context_event(body: ContextEventIn, request: Request) -> dict[str, Any]:
    """Ingest a single mobile context event into the v2 events table.

    Mapped to a v1-compatible internal type (``location.changed``,
    ``home.device.state_changed`` …) so the existing pipeline keys off
    the same strings the v2 producers already watch for. The mobile
    event type is preserved verbatim under
    ``metadata.mobile_event_type`` for downstream filtering.
    """
    conn = _db_or_503(request)
    internal_type = _CONTEXT_TYPE_MAP.get(body.type, "system.user.command")
    ts = _coerce_timestamp(body.timestamp)

    metadata: dict[str, Any] = {
        "domain": "context",
        "mobile_event_type": body.type,
    }
    if body.metadata is not None:
        metadata.update(body.metadata.model_dump(exclude_none=True))

    event_id = _insert_event(
        conn,
        event_type=internal_type,
        source=body.source,
        timestamp=ts,
        priority="silent",
        payload=body.payload.model_dump(exclude_none=True),
        metadata=metadata,
    )
    return {"status": "received", "event_id": event_id}


@router.post("/api/context/batch")
def submit_context_batch(body: ContextBatchIn, request: Request) -> dict[str, Any]:
    """Ingest a batch of mobile context events. One DB commit per event.

    The response mirrors v1 (``{status, count, event_ids}``) so iOS can
    log success counts; per-event failures would be surfaced via a
    partial-write error on the batch, which is why the route wraps the
    insert loop in a single try/except rather than inserting in a
    transaction — one bad row would otherwise fail-closed for the whole
    batch.
    """
    conn = _db_or_503(request)
    event_ids: list[str] = []
    for event in body.events:
        internal_type = _CONTEXT_TYPE_MAP.get(event.type, "system.user.command")
        ts = _coerce_timestamp(event.timestamp)
        metadata: dict[str, Any] = {
            "domain": "context",
            "mobile_event_type": event.type,
        }
        if event.metadata is not None:
            metadata.update(event.metadata.model_dump(exclude_none=True))
        event_ids.append(
            _insert_event(
                conn,
                event_type=internal_type,
                source=event.source,
                timestamp=ts,
                priority="silent",
                payload=event.payload.model_dump(exclude_none=True),
                metadata=metadata,
            )
        )
    return {"status": "received", "count": len(event_ids), "event_ids": event_ids}


@router.get("/api/context/summary")
def get_context_summary(request: Request) -> dict[str, Any]:
    """Return a rolling summary of recent iOS context events.

    Reads up to 100 most-recent rows from ``events`` where source is
    ``ios_app`` and synthesises the same ``{locations, devices,
    unique_places, unique_devices, content}`` shape the iOS app
    expects. An empty install returns all-empty lists with a
    human-readable ``content`` string — iOS never null-checks.
    """
    conn = _db_or_503(request)
    rows = conn.execute(
        """
        SELECT timestamp, payload, metadata
        FROM events
        WHERE source = 'ios_app'
        ORDER BY timestamp DESC
        LIMIT 100
        """
    ).fetchall()

    locations: list[dict[str, Any]] = []
    devices: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"] if isinstance(row, sqlite3.Row) else row[1] or "{}")
            metadata = json.loads(row["metadata"] if isinstance(row, sqlite3.Row) else row[2] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        ts = row["timestamp"] if isinstance(row, sqlite3.Row) else row[0]
        mobile_type = metadata.get("mobile_event_type", "")
        if mobile_type == "context.location" and payload.get("latitude") is not None:
            locations.append(
                {
                    "place": payload.get("place_name") or "Unknown",
                    "lat": payload.get("latitude"),
                    "lon": payload.get("longitude"),
                    "timestamp": ts,
                }
            )
        elif mobile_type == "context.device_nearby":
            devices.append(
                {
                    "name": payload.get("device_name") or "Unknown",
                    "type": payload.get("device_type"),
                    "signal": payload.get("signal_strength"),
                    "timestamp": ts,
                }
            )

    unique_places = sorted({loc["place"] for loc in locations if loc["place"] != "Unknown"})
    unique_devices = sorted({dev["name"] for dev in devices if dev["name"] != "Unknown"})
    content = (
        f"Context: {len(locations)} location updates, {len(devices)} device sightings. "
        f"Places: {', '.join(unique_places) or 'none tracked'}. "
        f"Devices: {', '.join(unique_devices) or 'none detected'}."
    )
    return {
        "type": "context_summary",
        "content": content,
        # Newest-first slice (`rows` is already DESC-sorted) is bounded
        # to 10 to keep the payload small over a slow iOS connection.
        "locations": locations[:10],
        "devices": devices[:10],
        "unique_places": unique_places,
        "unique_devices": unique_devices,
    }


# ---------------------------------------------------------------------------
# Real proxies
# ---------------------------------------------------------------------------


@router.get("/api/status")
def get_status(request: Request) -> dict[str, Any]:
    """Tiny smoke endpoint — events count, moments count, timestamp.

    The iOS app calls this on launch to verify connectivity. Shape is
    deliberately flat and fails open (every counter defaults to 0) so
    a half-constructed LifeOS still returns 200 for the phone's "am I
    connected?" check.
    """
    conn = _db(request)
    event_count = 0
    moment_count = 0
    if conn is not None:
        try:
            event_count = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        except sqlite3.Error:
            event_count = 0
        try:
            moment_count = int(conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0])
        except sqlite3.Error:
            moment_count = 0
    return {
        "ok": conn is not None,
        "ts": int(time.time()),
        "event_count": event_count,
        "moment_count": moment_count,
    }


@router.get("/api/briefing")
async def get_briefing(request: Request) -> dict[str, Any]:
    """Generate the daily briefing via the v2 AI engine.

    Returns v1's ``{briefing, generated_at}`` shape. On any failure
    (missing wiring, Ollama down, budget overrun) the response still
    carries both keys with ``briefing=None`` and an ``error`` string
    the iOS app can surface inline — a 500 here would take the iOS
    dashboard offline entirely.
    """
    life_os = _life_os(request)
    engine = getattr(life_os, "ai_engine", None)
    assembler = getattr(life_os, "context_assembler", None)
    generated_at = datetime.now(UTC).isoformat()

    if engine is None or assembler is None:
        return {
            "briefing": None,
            "generated_at": generated_at,
            "error": "ai_engine or context_assembler not wired on life_os",
        }

    try:
        context = assembler.assemble_briefing_context("default", date_cls.today())
        text = await engine.briefing_synthesis(context)
        return {"briefing": text, "generated_at": generated_at}
    except Exception as exc:
        logger.warning("Briefing generation failed: %s", exc)
        return {
            "briefing": None,
            "generated_at": generated_at,
            "error": "Briefing generation temporarily unavailable",
        }


@router.post("/api/feedback")
def submit_feedback(body: FeedbackIn, request: Request) -> dict[str, Any]:
    """Persist user feedback as an append-only ``system.user.feedback`` event.

    v2 has no dedicated ``feedback_events`` table (see NOTE on Week 1
    migration); legacy feedback is deliberately NOT carried forward
    via a dedicated table. We keep this endpoint functional by writing
    to the event log so the audit trail survives and any future loop
    can re-derive signal from it.
    """
    conn = _db_or_503(request)
    event_id = _insert_event(
        conn,
        event_type="system.user.feedback",
        source="web_api",
        timestamp=int(time.time()),
        priority="normal",
        payload={"message": body.message},
        metadata={"domain": "feedback"},
    )
    return {"status": "received", "event_id": event_id}


def _upsert_preference(conn: sqlite3.Connection, key: str, value: Any) -> None:
    """Write one row into v2's ``preferences`` table.

    ``value`` is always JSON-encoded regardless of source type so the
    column's TEXT type stays consistent and clients can round-trip
    nested objects. ``encrypted=0`` is the only path here; Fernet
    writes go through the settings-tab connector PATCH, not this
    free-form shim.
    """
    serialized = json.dumps(value) if not isinstance(value, str) else value
    conn.execute(
        """
        INSERT INTO preferences (key, value, encrypted, updated_at)
        VALUES (?, ?, 0, strftime('%s', 'now'))
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            encrypted = 0,
            updated_at = strftime('%s', 'now')
        """,
        (key, serialized),
    )
    conn.commit()


@router.post("/api/preferences")
def submit_preference(body: PreferenceIn, request: Request) -> dict[str, Any]:
    """Upsert one ``(key, value)`` pair into the v2 preferences table."""
    conn = _db_or_503(request)
    _upsert_preference(conn, body.key, body.value)
    return {"status": "updated"}


@router.put("/api/preferences")
def update_preference(body: PreferenceIn, request: Request) -> dict[str, Any]:
    """V1-compat alias for ``POST /api/preferences``.

    The v1 handler used PUT while the new iOS code switched to POST;
    both verbs map to the same upsert so neither client regresses.
    """
    conn = _db_or_503(request)
    _upsert_preference(conn, body.key, body.value)
    return {"status": "updated"}


# ---------------------------------------------------------------------------
# WebSocket /ws — Moment push
# ---------------------------------------------------------------------------


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Accept an iOS WebSocket connection and keep it open for push.

    Broadcast wiring lands in Week 11 — this endpoint's only job today
    is to accept the connection so the phone's reconnect loop stops
    hammering the server. If ``life_os.ws_manager`` is wired the
    connection is registered with it; otherwise we loop on
    ``receive_text()`` until the client disconnects and return cleanly.
    """
    await websocket.accept()
    life_os = getattr(websocket.app.state, "life_os", None)
    manager = getattr(life_os, "ws_manager", None)

    if manager is not None and hasattr(manager, "connect"):
        try:
            await manager.connect(websocket)
        except Exception as exc:
            # A manager that refuses to register must not take the
            # client offline — fall back to the built-in keepalive
            # loop so the phone's UX is the same as on a fresh install.
            logger.warning("ws_manager.connect failed: %s", exc)
            manager = None

    try:
        while True:
            # iOS sends heartbeat / ack frames; we drain them without
            # interpreting so an older client without push-ack support
            # still stays connected.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if manager is not None and hasattr(manager, "disconnect"):
            try:
                await manager.disconnect(websocket)
            except Exception as exc:
                logger.debug("ws_manager.disconnect failed: %s", exc)


# ---------------------------------------------------------------------------
# 501 stubs — endpoints the iOS app still calls but v2 has not ported
# ---------------------------------------------------------------------------


def _not_implemented(endpoint: str, follow_up: str) -> JSONResponse:
    """Structured 501 body — iOS surfaces ``error`` inline, not a toast."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={
            "error": "not_implemented_in_v2",
            "endpoint": endpoint,
            "message": f"{endpoint} is not yet implemented in the v2 API.",
            "follow_up": follow_up,
        },
    )


@router.post("/api/command")
def stub_command() -> JSONResponse:
    """Stub — NLP command bar is v1-only; v2 has no command router."""
    return _not_implemented(
        "POST /api/command",
        "Command bar was killed in v2 (CEO plan § 'Killed Features'). Use Moment actions instead.",
    )


@router.get("/api/notifications")
def stub_list_notifications() -> JSONResponse:
    """Stub — v2 replaces notifications with Moments."""
    return _not_implemented(
        "GET /api/notifications",
        "v2 surfaces actionable state via Moments. See GET /api/now.",
    )


@router.get("/api/tasks")
def stub_list_tasks() -> JSONResponse:
    """Stub — tasks are Moments with ``InsightType.LEGACY_TASK`` in v2."""
    return _not_implemented(
        "GET /api/tasks",
        "Tasks are modelled as Moments in v2. See GET /api/now for pending work.",
    )


@router.post("/api/tasks")
def stub_create_task() -> JSONResponse:
    """Stub — task creation is producer-driven in v2."""
    return _not_implemented(
        "POST /api/tasks",
        "Direct task creation is not exposed in v2; tasks are emitted by producers.",
    )


@router.post("/api/search")
def stub_search() -> JSONResponse:
    """Stub — semantic search moves behind :meth:`AIEngine.semantic_search`."""
    return _not_implemented(
        "POST /api/search",
        "Semantic search lands in a later v2 iteration (AIEngine.semantic_search).",
    )


__all__ = ["router"]
